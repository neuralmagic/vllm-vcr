//! An AWS SDK `HttpClient` whose TLS is the system OpenSSL: hyper 1 over
//! hyper-tls/native-tls, with proxies from the `HTTPS_PROXY`/`HTTP_PROXY`/
//! `NO_PROXY` environment (CONNECT tunnels for https targets).

use std::collections::HashMap;
use std::error::Error;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::task::{Context, Poll};
use std::time::Duration;

use aws_smithy_runtime_api::client::http::{
    HttpClient, HttpConnector, HttpConnectorFuture, HttpConnectorSettings, SharedHttpConnector,
};
use aws_smithy_runtime_api::client::orchestrator::HttpRequest;
use aws_smithy_runtime_api::client::result::ConnectorError;
use aws_smithy_runtime_api::client::runtime_components::RuntimeComponents;
use aws_smithy_types::body::SdkBody;
use aws_smithy_types::retry::ErrorKind;
use http::Uri;
use hyper::rt::{Read, ReadBufCursor, Write};
use hyper_tls::HttpsConnector;
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::proxy::Tunnel;
use hyper_util::client::legacy::connect::{Connected, Connection, HttpConnector as TcpConnector};
use hyper_util::client::proxy::matcher::Matcher;
use hyper_util::rt::{TokioExecutor, TokioIo};
use tokio::net::TcpStream;
use tower_service::Service;

type BoxError = Box<dyn Error + Send + Sync>;
type HyperClient = Client<HttpsConnector<ProxyConnector>, SdkBody>;

#[derive(Debug)]
pub struct OpensslHttpClient {
    tls: native_tls::TlsConnector,
    proxy: Arc<Matcher>,
    cache: RwLock<HashMap<CacheKey, SharedHttpConnector>>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CacheKey {
    connect_timeout: Option<Duration>,
    read_timeout: Option<Duration>,
}

impl OpensslHttpClient {
    pub fn new() -> Result<Self, native_tls::Error> {
        Ok(Self {
            tls: native_tls::TlsConnector::builder()
                .min_protocol_version(Some(native_tls::Protocol::Tlsv12))
                .build()?,
            proxy: Arc::new(Matcher::from_env()),
            cache: RwLock::new(HashMap::new()),
        })
    }

    fn build_connector(&self, settings: &HttpConnectorSettings) -> OpensslConnector {
        let mut tcp = TcpConnector::new();
        tcp.enforce_http(false);
        tcp.set_nodelay(true);
        tcp.set_connect_timeout(settings.connect_timeout());
        let proxied = ProxyConnector {
            tcp,
            proxy: Arc::clone(&self.proxy),
        };
        let https = HttpsConnector::from((proxied, self.tls.clone().into()));
        OpensslConnector {
            client: Arc::new(Client::builder(TokioExecutor::new()).build(https)),
            proxy: Arc::clone(&self.proxy),
            read_timeout: settings.read_timeout(),
        }
    }
}

impl HttpClient for OpensslHttpClient {
    // The orchestrator calls this once per request, so the connector (and its
    // hyper connection pool) is cached per timeout settings.
    fn http_connector(
        &self,
        settings: &HttpConnectorSettings,
        _components: &RuntimeComponents,
    ) -> SharedHttpConnector {
        let key = CacheKey {
            connect_timeout: settings.connect_timeout(),
            read_timeout: settings.read_timeout(),
        };
        if let Ok(cache) = self.cache.read()
            && let Some(connector) = cache.get(&key)
        {
            return connector.clone();
        }
        let connector = SharedHttpConnector::new(self.build_connector(settings));
        match self.cache.write() {
            Ok(mut cache) => cache.entry(key).or_insert(connector).clone(),
            Err(_) => connector,
        }
    }
}

#[derive(Clone, Debug)]
struct OpensslConnector {
    client: Arc<HyperClient>,
    proxy: Arc<Matcher>,
    read_timeout: Option<Duration>,
}

impl HttpConnector for OpensslConnector {
    fn call(&self, request: HttpRequest) -> HttpConnectorFuture {
        let client = Arc::clone(&self.client);
        let proxy = Arc::clone(&self.proxy);
        let read_timeout = self.read_timeout;
        HttpConnectorFuture::new(async move {
            let mut request = request
                .try_into_http1x()
                .map_err(|e| ConnectorError::user(e.into()))?;
            add_proxy_auth(&mut request, &proxy);
            let send = client.request(request);
            let response = match read_timeout {
                Some(timeout) => tokio::time::timeout(timeout, send)
                    .await
                    .map_err(|e| ConnectorError::timeout(e.into()))?,
                None => send.await,
            }
            .map_err(|e| classify(e.into()))?;
            let response = response.map(SdkBody::from_body_1_x);
            response
                .try_into()
                .map_err(|e: aws_smithy_runtime_api::http::HttpError| {
                    ConnectorError::other(e.into(), None)
                })
        })
    }
}

/// Plain-http requests through a proxy go in absolute form to the proxy itself,
/// so the proxy credentials ride on the request. For https the credentials go
/// on the CONNECT (see [`ProxyConnector`]).
fn add_proxy_auth(request: &mut http::Request<SdkBody>, proxy: &Matcher) {
    if request.uri().scheme() != Some(&http::uri::Scheme::HTTP)
        || request
            .headers()
            .contains_key(http::header::PROXY_AUTHORIZATION)
    {
        return;
    }
    if let Some(intercept) = proxy.intercept(request.uri())
        && let Some(auth) = intercept.basic_auth()
    {
        request
            .headers_mut()
            .insert(http::header::PROXY_AUTHORIZATION, auth.clone());
    }
}

/// Map hyper's errors onto the SDK's retry classes the way
/// aws-smithy-http-client does, so a refused or reset connection is retried
/// and a certificate or protocol error is not.
fn classify(err: BoxError) -> ConnectorError {
    if let Some(hyper_err) = find_source::<hyper::Error>(err.as_ref()) {
        if hyper_err.is_timeout() {
            return ConnectorError::timeout(err);
        }
        if hyper_err.is_user() {
            return ConnectorError::user(err);
        }
        if hyper_err.is_closed()
            || hyper_err.is_canceled()
            || find_source::<std::io::Error>(hyper_err).is_some()
        {
            return ConnectorError::io(err);
        }
        if hyper_err.is_incomplete_message() {
            return ConnectorError::other(err, Some(ErrorKind::TransientError));
        }
        return ConnectorError::other(err, None);
    }
    if let Some(util_err) = find_source::<hyper_util::client::legacy::Error>(err.as_ref()) {
        // A TLS failure surfaces as a connect error wrapping native_tls, not io.
        if find_source::<native_tls::Error>(util_err).is_some() {
            return ConnectorError::other(err, None);
        }
        if util_err.is_connect() || find_source::<std::io::Error>(util_err).is_some() {
            return ConnectorError::io(err);
        }
    }
    ConnectorError::other(err, None)
}

fn find_source<'a, E: Error + 'static>(err: &'a (dyn Error + 'static)) -> Option<&'a E> {
    let mut next = Some(err);
    while let Some(err) = next {
        if let Some(matching) = err.downcast_ref::<E>() {
            return Some(matching);
        }
        next = err.source();
    }
    None
}

/// TCP connector that honors the proxy environment: a CONNECT tunnel for https
/// targets, a connection to the proxy (absolute-form requests) for http
/// targets, a direct connection otherwise. hyper-tls layers TLS on top for
/// https targets in both the tunnelled and the direct case.
#[derive(Clone, Debug)]
struct ProxyConnector {
    tcp: TcpConnector,
    proxy: Arc<Matcher>,
}

type ConnectFuture = Pin<Box<dyn Future<Output = Result<ProxyStream, BoxError>> + Send>>;

impl Service<Uri> for ProxyConnector {
    type Response = ProxyStream;
    type Error = BoxError;
    type Future = ConnectFuture;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.tcp.poll_ready(cx).map_err(Into::into)
    }

    fn call(&mut self, dst: Uri) -> Self::Future {
        let Some(intercept) = self.proxy.intercept(&dst) else {
            let fut = self.tcp.call(dst);
            return Box::pin(async move {
                Ok(ProxyStream {
                    inner: fut.await?,
                    via_proxy: false,
                })
            });
        };
        let proxy_uri = intercept.uri().clone();
        if dst.scheme() == Some(&http::uri::Scheme::HTTPS) {
            let mut tunnel = Tunnel::new(proxy_uri, self.tcp.clone());
            if let Some(auth) = intercept.basic_auth() {
                tunnel = tunnel.with_auth(auth.clone());
            }
            let fut = tunnel.call(dst);
            return Box::pin(async move {
                Ok(ProxyStream {
                    inner: fut.await?,
                    via_proxy: false,
                })
            });
        }
        let fut = self.tcp.call(proxy_uri);
        Box::pin(async move {
            Ok(ProxyStream {
                inner: fut.await?,
                via_proxy: true,
            })
        })
    }
}

/// A TCP stream that reports whether requests on it must be in absolute form
/// (plain http through a proxy).
#[derive(Debug)]
struct ProxyStream {
    inner: TokioIo<TcpStream>,
    via_proxy: bool,
}

impl Connection for ProxyStream {
    fn connected(&self) -> Connected {
        self.inner.connected().proxy(self.via_proxy)
    }
}

impl Read for ProxyStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: ReadBufCursor<'_>,
    ) -> Poll<Result<(), std::io::Error>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl Write for ProxyStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, std::io::Error>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_flush(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(), std::io::Error>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(), std::io::Error>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }

    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[std::io::IoSlice<'_>],
    ) -> Poll<Result<usize, std::io::Error>> {
        Pin::new(&mut self.inner).poll_write_vectored(cx, bufs)
    }
}
