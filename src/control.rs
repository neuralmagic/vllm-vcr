//! Runtime control plane for the mock engine: an HTTP/JSON server on its own port
//! (`--control-address`) that reads and patches the engine's mutable knobs and
//! reports request counters. Every call is forwarded to each engine loop as an
//! [`EngineInput::Control`] message and answered over a oneshot, so the engine
//! stays single-owner and a patch lands between two steps, never in the middle
//! of one.

use std::net::SocketAddr;

use anyhow::{Context as _, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::engine_core::EngineInput;
use crate::{FailureType, LogFilter};

/// A control call handed to one engine loop. The engine answers on `reply`.
pub(crate) struct ControlCall {
    pub request: ControlRequest,
    pub reply: oneshot::Sender<Result<ControlReply, ControlError>>,
}

#[derive(Debug, Clone)]
pub(crate) enum ControlRequest {
    GetConfig,
    PatchConfig(Box<ConfigPatch>),
    GetStats,
    ResetStats,
}

#[derive(Debug)]
pub(crate) enum ControlReply {
    Config(Config),
    Stats(Stats),
}

/// A rejected control call. Carries the HTTP status the server answers with.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ControlError {
    #[serde(skip)]
    pub status: u16,
    pub error: String,
}

impl ControlError {
    pub fn conflict(msg: impl Into<String>) -> Self {
        ControlError {
            status: StatusCode::CONFLICT.as_u16(),
            error: msg.into(),
        }
    }

    pub fn bad_request(msg: impl Into<String>) -> Self {
        ControlError {
            status: StatusCode::BAD_REQUEST.as_u16(),
            error: msg.into(),
        }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        ControlError {
            status: StatusCode::INTERNAL_SERVER_ERROR.as_u16(),
            error: msg.into(),
        }
    }
}

/// The engine knobs that can change at runtime, as reported by `GET /config`.
/// Milliseconds for every latency field, matching the CLI flags of the same name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub time_to_first_token: u64,
    pub time_to_first_token_std_dev: u64,
    pub inter_token_latency: u64,
    pub inter_token_latency_std_dev: u64,
    pub prefill_overhead: u64,
    pub prefill_time_per_token: u64,
    pub prefill_time_std_dev: u64,
    pub time_factor_under_load: f64,
    pub max_num_seqs: u64,
    pub max_num_batched_tokens: u64,
    pub max_model_len: u64,
    pub failure_injection_rate: f64,
    pub failure_types: Vec<FailureType>,
    pub log_requests: bool,
}

/// Partial update for `PATCH /config`. Absent fields keep their current value.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ConfigPatch {
    pub time_to_first_token: Option<u64>,
    pub time_to_first_token_std_dev: Option<u64>,
    pub inter_token_latency: Option<u64>,
    pub inter_token_latency_std_dev: Option<u64>,
    pub prefill_overhead: Option<u64>,
    pub prefill_time_per_token: Option<u64>,
    pub prefill_time_std_dev: Option<u64>,
    pub time_factor_under_load: Option<f64>,
    pub max_num_seqs: Option<u64>,
    pub max_num_batched_tokens: Option<u64>,
    pub max_model_len: Option<u64>,
    pub failure_injection_rate: Option<f64>,
    pub failure_types: Option<Vec<FailureType>>,
    pub log_requests: Option<bool>,
}

impl ConfigPatch {
    /// Whether the patch touches a field the latency model is built from.
    pub fn changes_latency(&self) -> bool {
        self.time_to_first_token.is_some()
            || self.time_to_first_token_std_dev.is_some()
            || self.inter_token_latency.is_some()
            || self.inter_token_latency_std_dev.is_some()
            || self.prefill_overhead.is_some()
            || self.prefill_time_per_token.is_some()
            || self.prefill_time_std_dev.is_some()
            || self.time_factor_under_load.is_some()
            || self.max_num_seqs.is_some()
    }

    pub fn validate(&self) -> Result<(), ControlError> {
        if let Some(rate) = self.failure_injection_rate
            && !(0.0..=1.0).contains(&rate)
        {
            return Err(ControlError::bad_request(format!(
                "failure_injection_rate must be in [0, 1], got {rate}"
            )));
        }
        if let Some(factor) = self.time_factor_under_load
            && (factor.is_nan() || factor < 1.0)
        {
            return Err(ControlError::bad_request(format!(
                "time_factor_under_load must be >= 1.0, got {factor}"
            )));
        }
        if let Some(types) = &self.failure_types
            && types.is_empty()
        {
            return Err(ControlError::bad_request(
                "failure_types must name at least one type",
            ));
        }
        Ok(())
    }
}

/// Request counters for one engine. `GET /stats` sums them across engines.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stats {
    /// Requests received from the frontend, whatever became of them.
    pub requests_received: u64,
    /// Requests that ran to their stop condition.
    pub requests_completed: u64,
    /// Requests finished with an error finish reason: injected failures,
    /// context-length rejects, failed KV pulls, duplicate ids.
    pub requests_failed: u64,
    /// Requests aborted by the frontend or by shutdown.
    pub requests_aborted: u64,
    /// Requests in the running batch right now.
    pub running: u64,
    /// Requests waiting for a batch slot right now.
    pub waiting: u64,
}

impl Stats {
    fn add(&mut self, other: Stats) {
        self.requests_received += other.requests_received;
        self.requests_completed += other.requests_completed;
        self.requests_failed += other.requests_failed;
        self.requests_aborted += other.requests_aborted;
        self.running += other.running;
        self.waiting += other.waiting;
    }
}

/// Senders into every engine loop. Cloneable so axum can share it as state.
#[derive(Clone)]
pub(crate) struct Engines(pub Vec<mpsc::UnboundedSender<EngineInput>>);

/// Shared handler state: the engines plus, when the process installed a
/// reloadable subscriber, the handle that swaps the log filter.
#[derive(Clone)]
struct AppState {
    engines: Engines,
    log: Option<LogFilter>,
}

/// Body of `GET`/`PUT /log`: a `tracing` `EnvFilter` directive string such as
/// `info`, `debug`, or `vllm_vcr::engine=trace,info`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogLevel {
    pub filter: String,
}

impl Engines {
    async fn call(
        &self,
        engine: &mpsc::UnboundedSender<EngineInput>,
        request: ControlRequest,
    ) -> Result<ControlReply, ControlError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        engine
            .send(EngineInput::Control(ControlCall {
                request,
                reply: reply_tx,
            }))
            .map_err(|_| ControlError::internal("engine loop is gone"))?;
        reply_rx
            .await
            .map_err(|_| ControlError::internal("engine dropped the control reply"))?
    }

    /// Send the same request to every engine, in order, collecting the replies.
    async fn broadcast(&self, request: ControlRequest) -> Result<Vec<ControlReply>, ControlError> {
        let mut replies = Vec::with_capacity(self.0.len());
        for engine in &self.0 {
            replies.push(self.call(engine, request.clone()).await?);
        }
        Ok(replies)
    }
}

impl IntoResponse for ControlError {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (status, Json(self)).into_response()
    }
}

fn first_config(replies: Vec<ControlReply>) -> Result<Json<Config>, ControlError> {
    match replies.into_iter().next() {
        Some(ControlReply::Config(config)) => Ok(Json(config)),
        _ => Err(ControlError::internal("engine returned no config")),
    }
}

async fn get_config(State(state): State<AppState>) -> Result<Json<Config>, ControlError> {
    first_config(state.engines.broadcast(ControlRequest::GetConfig).await?)
}

async fn patch_config(
    State(state): State<AppState>,
    Json(patch): Json<ConfigPatch>,
) -> Result<Json<Config>, ControlError> {
    patch.validate()?;
    first_config(
        state
            .engines
            .broadcast(ControlRequest::PatchConfig(Box::new(patch)))
            .await?,
    )
}

async fn get_stats(State(state): State<AppState>) -> Result<Json<Stats>, ControlError> {
    let mut total = Stats::default();
    for reply in state.engines.broadcast(ControlRequest::GetStats).await? {
        if let ControlReply::Stats(stats) = reply {
            total.add(stats);
        }
    }
    Ok(Json(total))
}

async fn reset_stats(State(state): State<AppState>) -> Result<StatusCode, ControlError> {
    state.engines.broadcast(ControlRequest::ResetStats).await?;
    Ok(StatusCode::NO_CONTENT)
}

fn log_filter(state: &AppState) -> Result<&LogFilter, ControlError> {
    state.log.as_ref().ok_or_else(|| ControlError {
        status: StatusCode::NOT_IMPLEMENTED.as_u16(),
        error: "this process has no reloadable log filter".to_string(),
    })
}

async fn get_log(State(state): State<AppState>) -> Result<Json<LogLevel>, ControlError> {
    let filter = log_filter(&state)?
        .current()
        .map_err(ControlError::internal)?;
    Ok(Json(LogLevel { filter }))
}

async fn put_log(
    State(state): State<AppState>,
    Json(level): Json<LogLevel>,
) -> Result<Json<LogLevel>, ControlError> {
    let handle = log_filter(&state)?;
    let parsed = tracing_subscriber::EnvFilter::try_new(&level.filter)
        .map_err(|error| ControlError::bad_request(format!("invalid log filter: {error}")))?;
    handle.set(parsed).map_err(ControlError::internal)?;
    info!(filter = %level.filter, "log filter changed via control API");
    let filter = handle.current().map_err(ControlError::internal)?;
    Ok(Json(LogLevel { filter }))
}

pub(crate) fn router(engines: Engines, log: Option<LogFilter>) -> Router {
    Router::new()
        .route("/config", get(get_config).patch(patch_config))
        .route("/stats", get(get_stats))
        .route("/stats/reset", post(reset_stats))
        .route("/log", get(get_log).put(put_log))
        .with_state(AppState { engines, log })
}

/// Bind `addr` and serve the control API until `shutdown` fires.
pub(crate) async fn serve(
    addr: String,
    engines: Engines,
    log: Option<LogFilter>,
    shutdown: CancellationToken,
) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("binding control address {addr}"))?;
    let local: SocketAddr = listener.local_addr()?;
    info!(%local, "control API listening");
    axum::serve(listener, router(engines, log))
        .with_graceful_shutdown(shutdown.cancelled_owned())
        .await
        .context("control API server")
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt as _;

    /// A stand-in engine loop that answers control calls from a fixed config and
    /// counts patches, so the HTTP layer is tested without a real engine.
    fn fake_engine(
        config: Config,
    ) -> (
        mpsc::UnboundedSender<EngineInput>,
        tokio::task::JoinHandle<u64>,
    ) {
        let (tx, mut rx) = mpsc::unbounded_channel::<EngineInput>();
        let handle = tokio::spawn(async move {
            let mut config = config;
            let mut patches = 0;
            while let Some(input) = rx.recv().await {
                let EngineInput::Control(call) = input else {
                    continue;
                };
                let reply = match call.request {
                    ControlRequest::GetConfig => Ok(ControlReply::Config(config.clone())),
                    ControlRequest::PatchConfig(patch) => {
                        patches += 1;
                        if let Some(v) = patch.inter_token_latency {
                            config.inter_token_latency = v;
                        }
                        Ok(ControlReply::Config(config.clone()))
                    }
                    ControlRequest::GetStats => Ok(ControlReply::Stats(Stats {
                        requests_received: 3,
                        requests_completed: 2,
                        ..Stats::default()
                    })),
                    ControlRequest::ResetStats => Ok(ControlReply::Stats(Stats::default())),
                };
                call.reply.send(reply).ok();
            }
            patches
        });
        (tx, handle)
    }

    fn base_config() -> Config {
        Config {
            time_to_first_token: 0,
            time_to_first_token_std_dev: 0,
            inter_token_latency: 5,
            inter_token_latency_std_dev: 0,
            prefill_overhead: 0,
            prefill_time_per_token: 0,
            prefill_time_std_dev: 0,
            time_factor_under_load: 1.0,
            max_num_seqs: 128,
            max_num_batched_tokens: 2048,
            max_model_len: 0,
            failure_injection_rate: 0.0,
            failure_types: vec![FailureType::Error],
            log_requests: false,
        }
    }

    async fn send(
        router: Router,
        method: &str,
        uri: &str,
        body: &str,
    ) -> (StatusCode, serde_json::Value) {
        let request = Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        // axum's own rejections (422 on a bad body) are plain text, not JSON.
        let json = serde_json::from_slice(&bytes).unwrap_or_else(|_| {
            serde_json::Value::String(String::from_utf8_lossy(&bytes).into_owned())
        });
        (status, json)
    }

    #[tokio::test]
    async fn get_config_returns_engine_config() {
        let (tx, _) = fake_engine(base_config());
        let (status, json) = send(router(Engines(vec![tx]), None), "GET", "/config", "").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["inter_token_latency"], 5);
        assert_eq!(json["failure_types"], serde_json::json!(["error"]));
    }

    #[tokio::test]
    async fn patch_fans_out_to_every_engine() {
        let (tx_a, handle_a) = fake_engine(base_config());
        let (tx_b, handle_b) = fake_engine(base_config());
        let router = router(Engines(vec![tx_a.clone(), tx_b.clone()]), None);
        let (status, json) = send(
            router,
            "PATCH",
            "/config",
            r#"{"inter_token_latency": 30000}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["inter_token_latency"], 30000);
        drop((tx_a, tx_b));
        assert_eq!(handle_a.await.unwrap(), 1);
        assert_eq!(handle_b.await.unwrap(), 1);
    }

    #[tokio::test]
    async fn patch_rejects_invalid_values_before_reaching_engines() {
        let (tx, handle) = fake_engine(base_config());
        let router = router(Engines(vec![tx.clone()]), None);
        for body in [
            r#"{"failure_injection_rate": 1.5}"#,
            r#"{"time_factor_under_load": 0.5}"#,
            r#"{"failure_types": []}"#,
        ] {
            let (status, json) = send(router.clone(), "PATCH", "/config", body).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert!(json["error"].is_string(), "{body}");
        }
        let (status, _) = send(
            router,
            "PATCH",
            "/config",
            r#"{"failure_types": ["bogus"]}"#,
        )
        .await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        drop(tx);
        assert_eq!(handle.await.unwrap(), 0);
    }

    #[tokio::test]
    async fn stats_sum_across_engines_and_reset_returns_no_content() {
        let (tx_a, _) = fake_engine(base_config());
        let (tx_b, _) = fake_engine(base_config());
        let router = router(Engines(vec![tx_a, tx_b]), None);
        let (status, json) = send(router.clone(), "GET", "/stats", "").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["requests_received"], 6);
        assert_eq!(json["requests_completed"], 4);
        let (status, _) = send(router, "POST", "/stats/reset", "").await;
        assert_eq!(status, StatusCode::NO_CONTENT);
    }

    #[tokio::test]
    async fn dead_engine_is_a_500() {
        let (tx, rx) = mpsc::unbounded_channel::<EngineInput>();
        drop(rx);
        let (status, json) = send(router(Engines(vec![tx]), None), "GET", "/config", "").await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(json["error"], "engine loop is gone");
    }

    /// A reload handle plus the subscriber it points at. The handle only resolves
    /// while the subscriber is alive, so callers hold both; it does not need to be
    /// installed globally.
    fn reloadable_filter(initial: &str) -> (LogFilter, impl tracing::Subscriber) {
        use tracing_subscriber::layer::SubscriberExt as _;
        let (layer, handle) =
            tracing_subscriber::reload::Layer::new(tracing_subscriber::EnvFilter::new(initial));
        let subscriber = tracing_subscriber::registry().with(layer);
        (LogFilter::new(handle), subscriber)
    }

    #[tokio::test]
    async fn log_filter_round_trips_and_rejects_garbage() {
        let (tx, _) = fake_engine(base_config());
        let (filter, _subscriber) = reloadable_filter("info");
        let router = router(Engines(vec![tx]), Some(filter));

        let (status, json) = send(router.clone(), "GET", "/log", "").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["filter"], "info");

        let (status, json) = send(
            router.clone(),
            "PUT",
            "/log",
            r#"{"filter": "vllm_vcr=debug,warn"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["filter"], "vllm_vcr=debug,warn");
        let (_, json) = send(router.clone(), "GET", "/log", "").await;
        assert_eq!(json["filter"], "vllm_vcr=debug,warn");

        let (status, json) = send(router, "PUT", "/log", r#"{"filter": "vllm_vcr=loud"}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            json["error"]
                .as_str()
                .unwrap()
                .starts_with("invalid log filter")
        );
    }

    #[tokio::test]
    async fn log_routes_are_not_implemented_without_a_filter_handle() {
        let (tx, _) = fake_engine(base_config());
        let router = router(Engines(vec![tx]), None);
        let (status, _) = send(router.clone(), "GET", "/log", "").await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        let (status, _) = send(router, "PUT", "/log", r#"{"filter": "debug"}"#).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    }
}
