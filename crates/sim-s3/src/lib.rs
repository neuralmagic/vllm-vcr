//! S3 and HuggingFace object I/O for vllm-vcr trace files: a [`TraceUri`] is a local path,
//! an `s3://` object, or an `hf://` HuggingFace Hub file, fetched/uploaded via the AWS
//! default credential chain or HuggingFace Hub API.

use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

use anyhow::{Context as _, Result, bail};
use hf_hub::{HFClient, HFResult};
use tracing::info;

#[cfg(feature = "s3")]
mod openssl_http_client;

/// Whether the process's OpenSSL is running with its FIPS provider as the
/// default (`EVP_default_properties_is_fips_enabled`). `None` where the
/// binary does not link OpenSSL 3 (macOS uses Security.framework).
pub fn openssl_fips_enabled() -> Option<bool> {
    #[cfg(all(target_os = "linux", ossl300))]
    {
        // SAFETY: a NULL libctx selects the default library context; the call
        // reads a flag and has no preconditions.
        Some(
            unsafe { openssl_sys::EVP_default_properties_is_fips_enabled(std::ptr::null_mut()) }
                == 1,
        )
    }
    #[cfg(not(all(target_os = "linux", ossl300)))]
    {
        None
    }
}

/// A HuggingFace Hub client whose HTTP stack is reqwest over native-tls, so TLS
/// goes through the system OpenSSL. Endpoint, token, and cache dir come from the
/// usual `HF_*` environment variables.
pub fn hf_client() -> HFResult<HFClient> {
    let http = reqwest::Client::builder().use_native_tls().build()?;
    HFClient::builder().client(http).build()
}

/// Split a Hub repo id into `(owner, name)`, rejecting ids without an owner.
pub fn hf_repo_parts(repo_id: &str) -> Result<(&str, &str)> {
    match hf_hub::split_id(repo_id) {
        ("", _) | (_, "") => bail!("HuggingFace repo id must be owner/name, got {repo_id:?}"),
        parts => Ok(parts),
    }
}
use url::Url;

/// Whether a raw path string is an `s3://` or `hf://` URI rather than a local path.
pub fn is_remote(uri: &str) -> bool {
    uri.len() >= 5
        && (uri[..5].eq_ignore_ascii_case("s3://") || uri[..5].eq_ignore_ascii_case("hf://"))
}

/// A trace location, parsed (and validated) at the CLI boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceUri {
    Local(PathBuf),
    S3 { bucket: String, key: String },
    HuggingFace { repo: String, file: String },
}

impl FromStr for TraceUri {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        if is_remote(s) {
            if s.starts_with("s3://") || s.starts_with("S3://") {
                if !cfg!(feature = "s3") {
                    return Err("s3:// URIs need a vllm-vcr built with the s3 feature".to_string());
                }
                let (bucket, key) = parse_s3_uri(s).map_err(|e| format!("{e:#}"))?;
                Ok(TraceUri::S3 { bucket, key })
            } else if s.starts_with("hf://") || s.starts_with("HF://") {
                let (repo, file) = parse_hf_uri(s).map_err(|e| format!("{e:#}"))?;
                Ok(TraceUri::HuggingFace { repo, file })
            } else {
                Err("unknown remote URI scheme (expected s3:// or hf://)".to_string())
            }
        } else {
            Ok(TraceUri::Local(PathBuf::from(s)))
        }
    }
}

impl fmt::Display for TraceUri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TraceUri::Local(path) => write!(f, "{}", path.display()),
            TraceUri::S3 { bucket, key } => write!(f, "s3://{bucket}/{key}"),
            TraceUri::HuggingFace { repo, file } => write!(f, "hf://{repo}/{file}"),
        }
    }
}

impl TraceUri {
    pub fn is_remote(&self) -> bool {
        matches!(self, TraceUri::S3 { .. } | TraceUri::HuggingFace { .. })
    }

    /// The local path, when this is a local target (`None` for remote).
    pub fn local_path(&self) -> Option<&Path> {
        match self {
            TraceUri::Local(path) => Some(path),
            TraceUri::S3 { .. } | TraceUri::HuggingFace { .. } => None,
        }
    }

    /// A local path holding this trace's bytes: the path itself when local, or a
    /// scratch file fetched from S3 or HuggingFace.
    pub async fn materialize(&self, scratch_dir: &Path) -> Result<PathBuf> {
        match self {
            TraceUri::Local(path) => Ok(path.clone()),
            TraceUri::S3 { bucket, key } => self.fetch(bucket, key, scratch_dir).await,
            TraceUri::HuggingFace { repo, file } => self.fetch_hf(repo, file).await,
        }
    }

    /// Where to write this trace locally before upload: its own path when local,
    /// else a scratch path under `scratch_dir`.
    pub fn write_path(&self, scratch_dir: &Path) -> PathBuf {
        match self {
            TraceUri::Local(path) => path.clone(),
            TraceUri::S3 { key, .. } => scratch_path(&self.to_string(), key, scratch_dir),
            TraceUri::HuggingFace { file, .. } => {
                scratch_path(&self.to_string(), file, scratch_dir)
            }
        }
    }

    /// Upload a finalized local file to this target; a no-op when local.
    pub async fn upload(&self, local: &Path) -> Result<()> {
        let TraceUri::S3 { bucket, key } = self else {
            return Ok(());
        };
        self.put(bucket, key, local).await
    }

    #[cfg(not(feature = "s3"))]
    async fn put(&self, _bucket: &str, _key: &str, _local: &Path) -> Result<()> {
        bail!("{self}: built without the s3 feature")
    }

    #[cfg(feature = "s3")]
    async fn put(&self, bucket: &str, key: &str, local: &Path) -> Result<()> {
        use aws_sdk_s3::primitives::ByteStream;

        let size = std::fs::metadata(local).map(|m| m.len()).ok();
        info!(local = %local.display(), uri = %self, bucket, key, bytes = size, "S3 PUT: uploading trace");
        let started = Instant::now();
        let body = ByteStream::from_path(local)
            .await
            .with_context(|| format!("opening {} for upload", local.display()))?;
        s3_client()
            .await?
            .put_object()
            .bucket(bucket)
            .key(key)
            .body(body)
            .send()
            .await
            .with_context(|| format!("S3 PUT {self}"))?;
        info!(uri = %self, bytes = size, elapsed_ms = started.elapsed().as_millis(), "S3 PUT: trace uploaded");
        Ok(())
    }

    #[cfg(not(feature = "s3"))]
    async fn fetch(&self, _bucket: &str, _key: &str, _scratch_dir: &Path) -> Result<PathBuf> {
        bail!("{self}: built without the s3 feature")
    }

    #[cfg(feature = "s3")]
    async fn fetch(&self, bucket: &str, key: &str, scratch_dir: &Path) -> Result<PathBuf> {
        let dest = scratch_path(&self.to_string(), key, scratch_dir);
        info!(uri = %self, bucket, key, dest = %dest.display(), "S3 GET: fetching trace to scratch");
        let started = Instant::now();
        let response = s3_client()
            .await?
            .get_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .with_context(|| format!("S3 GET {self}"))?;
        let content_length = response.content_length();
        let bytes = response
            .body
            .collect()
            .await
            .with_context(|| format!("reading S3 object body: {self}"))?
            .into_bytes();
        std::fs::write(&dest, &bytes)
            .with_context(|| format!("writing scratch {} for {self}", dest.display()))?;
        info!(uri = %self, bytes = bytes.len(), content_length, dest = %dest.display(), elapsed_ms = started.elapsed().as_millis(), "S3 GET: trace materialized");
        Ok(dest)
    }

    async fn fetch_hf(&self, repo: &str, filename: &str) -> Result<PathBuf> {
        info!(
            uri = %self,
            repo = repo,
            file = filename,
            "HF GET: downloading from HuggingFace Hub"
        );

        let started = Instant::now();

        let (owner, name) = hf_repo_parts(repo)?;
        let client = hf_client().context("initializing HuggingFace client")?;
        let local_path = client
            .dataset(owner, name)
            .download_file()
            .filename(filename)
            .send()
            .await
            .with_context(|| format!("downloading {filename} from {repo}"))?;

        let elapsed = started.elapsed();
        let size = std::fs::metadata(&local_path).map(|m| m.len()).ok();

        info!(
            uri = %self,
            path = %local_path.display(),
            bytes = size,
            elapsed_ms = elapsed.as_millis(),
            "HF GET: file cached"
        );

        Ok(local_path)
    }
}

fn parse_s3_uri(uri: &str) -> Result<(String, String)> {
    let url = Url::parse(uri).with_context(|| format!("parsing S3 URI: {uri}"))?;
    if url.scheme() != "s3" {
        bail!(
            "expected an s3:// URI, got scheme {:?}: {uri}",
            url.scheme()
        );
    }
    let bucket = url
        .host_str()
        .filter(|host| !host.is_empty())
        .with_context(|| format!("S3 URI has no bucket: {uri}"))?
        .to_string();
    let key = url.path().trim_start_matches('/').to_string();
    if key.is_empty() {
        bail!("S3 URI has no object key: {uri}");
    }
    Ok((bucket, key))
}

fn parse_hf_uri(uri: &str) -> Result<(String, String)> {
    let url = Url::parse(uri).with_context(|| format!("parsing HuggingFace URI: {uri}"))?;
    if url.scheme() != "hf" {
        bail!(
            "expected an hf:// URI, got scheme {:?}: {uri}",
            url.scheme()
        );
    }

    // URL parsing puts the first path segment in the host (`hf://org/repo/file` →
    // host=org, path=/repo/file). Reassemble org/repo/.../file before splitting.
    let mut segments: Vec<&str> = Vec::new();
    if let Some(host) = url.host_str() {
        if !host.is_empty() {
            segments.push(host);
        }
    }
    let path = url.path().trim_start_matches('/');
    if !path.is_empty() {
        segments.extend(path.split('/').filter(|segment| !segment.is_empty()));
    }

    if segments.len() < 3 {
        bail!("HuggingFace URI must be hf://org/repo/file (got {})", uri);
    }

    let repo = format!("{}/{}", segments[0], segments[1]);
    let file = segments[2..].join("/");

    if file.is_empty() {
        bail!("HuggingFace URI has no file path: {uri}");
    }

    Ok((repo, file))
}

fn key_basename(key: &str) -> &str {
    key.rsplit('/')
        .find(|segment| !segment.is_empty())
        .unwrap_or("trace.jsonl")
}

/// Scratch path for a remote object: basename (keeping its suffix for gzip
/// detection) tagged with a hash of the URI so distinct objects don't collide.
fn scratch_path(uri: &str, key: &str, scratch_dir: &Path) -> PathBuf {
    use std::hash::{Hash as _, Hasher as _};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    uri.hash(&mut hasher);
    scratch_dir.join(format!(
        "sim-s3-{:016x}-{}",
        hasher.finish(),
        key_basename(key)
    ))
}

#[cfg(feature = "s3")]
async fn s3_client() -> Result<aws_sdk_s3::Client> {
    use aws_config::BehaviorVersion;
    use tracing::debug;

    let http_client = openssl_http_client::OpensslHttpClient::new()
        .context("building the OpenSSL HTTP client")?;
    let config = aws_config::defaults(BehaviorVersion::latest())
        .http_client(http_client)
        .load()
        .await;
    // S3-compatible endpoints (MinIO/LocalStack) only serve path-style; real AWS
    // (no endpoint override) uses virtual-host style.
    let force_path_style = config.endpoint_url().is_some();
    debug!(
        region = config.region().map(|r| r.as_ref()),
        endpoint = config.endpoint_url(),
        force_path_style,
        fips = ?openssl_fips_enabled(),
        "built S3 client from default credential chain"
    );
    Ok(aws_sdk_s3::Client::from_conf(
        aws_sdk_s3::config::Builder::from(&config)
            .force_path_style(force_path_style)
            .build(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_remote_matches_s3_and_hf_schemes() {
        assert!(is_remote("s3://bucket/key"));
        assert!(is_remote("S3://Bucket/Key"));
        assert!(is_remote("hf://org/repo/file.json"));
        assert!(!is_remote("/tmp/trace.jsonl.gz"));
        assert!(!is_remote("trace.jsonl"));
        assert!(!is_remote("file:///tmp/trace.jsonl"));
        assert!(!is_remote(""));
        assert!(!is_remote("s3:"));
    }

    #[cfg(feature = "s3")]
    #[test]
    fn parses_s3_uri_into_typed_variant() {
        let uri: TraceUri = "s3://my-bucket/traces/abc/tap-trace.jsonl.gz"
            .parse()
            .unwrap();
        assert_eq!(
            uri,
            TraceUri::S3 {
                bucket: "my-bucket".to_string(),
                key: "traces/abc/tap-trace.jsonl.gz".to_string(),
            }
        );
        assert!(uri.is_remote());
        assert!(uri.local_path().is_none());
        assert_eq!(
            uri.to_string(),
            "s3://my-bucket/traces/abc/tap-trace.jsonl.gz"
        );
    }

    #[test]
    fn parses_bare_path_as_local() {
        let uri: TraceUri = "/tmp/trace.jsonl".parse().unwrap();
        assert_eq!(uri, TraceUri::Local(PathBuf::from("/tmp/trace.jsonl")));
        assert!(!uri.is_remote());
        assert_eq!(uri.local_path(), Some(Path::new("/tmp/trace.jsonl")));
    }

    #[cfg(not(feature = "s3"))]
    #[test]
    fn rejects_s3_uri_without_feature() {
        let err = "s3://bucket/key".parse::<TraceUri>().unwrap_err();
        assert!(err.contains("s3 feature"), "{err}");
    }

    #[cfg(feature = "s3")]
    #[test]
    fn rejects_malformed_s3_uri() {
        assert!("s3://bucket".parse::<TraceUri>().is_err()); // no key
        assert!("s3://bucket/".parse::<TraceUri>().is_err()); // empty key
        assert!("s3:///key".parse::<TraceUri>().is_err()); // no bucket
    }

    #[test]
    fn key_basename_keeps_gz_suffix() {
        assert_eq!(
            key_basename("traces/abc/tap-trace.jsonl.gz"),
            "tap-trace.jsonl.gz"
        );
        assert_eq!(key_basename("flat.jsonl"), "flat.jsonl");
        assert_eq!(key_basename("trailing/"), "trailing");
    }

    #[cfg(feature = "s3")]
    #[test]
    fn write_path_is_stable_per_uri_and_collision_free() {
        let dir = Path::new("/tmp/scratch");
        let a1: TraceUri = "s3://b/traces/a/tap-trace.jsonl.gz".parse().unwrap();
        let a2: TraceUri = "s3://b/traces/a/tap-trace.jsonl.gz".parse().unwrap();
        let b: TraceUri = "s3://b/traces/b/tap-trace.jsonl.gz".parse().unwrap();

        assert_eq!(a1.write_path(dir), a2.write_path(dir));
        assert_ne!(a1.write_path(dir), b.write_path(dir));
        assert!(
            a1.write_path(dir)
                .to_string_lossy()
                .ends_with("-tap-trace.jsonl.gz")
        );

        let local: TraceUri = "/tmp/x.jsonl".parse().unwrap();
        assert_eq!(local.write_path(dir), PathBuf::from("/tmp/x.jsonl"));
    }

    #[test]
    fn hf_repo_parts_requires_owner_and_name() {
        assert_eq!(
            hf_repo_parts("Qwen/Qwen3-0.6B").unwrap(),
            ("Qwen", "Qwen3-0.6B")
        );
        assert!(hf_repo_parts("Qwen3-0.6B").is_err());
        assert!(hf_repo_parts("Qwen/").is_err());
        assert!(hf_repo_parts("/Qwen3").is_err());
    }

    #[test]
    fn parses_hf_uri_basic() {
        let uri: TraceUri = "hf://neuralmagic/vllm-traces/trace.jsonl.gz"
            .parse()
            .unwrap();
        assert_eq!(
            uri,
            TraceUri::HuggingFace {
                repo: "neuralmagic/vllm-traces".to_string(),
                file: "trace.jsonl.gz".to_string(),
            }
        );
        assert!(uri.is_remote());
        assert_eq!(
            uri.to_string(),
            "hf://neuralmagic/vllm-traces/trace.jsonl.gz"
        );
    }

    #[test]
    fn parses_hf_uri_with_nested_file_path() {
        let uri: TraceUri = "hf://org/repo/data/file.json".parse().unwrap();
        assert_eq!(
            uri,
            TraceUri::HuggingFace {
                repo: "org/repo".to_string(),
                file: "data/file.json".to_string(),
            }
        );
        assert_eq!(uri.to_string(), "hf://org/repo/data/file.json");
    }

    #[test]
    fn rejects_malformed_hf_uri() {
        assert!("hf://repo".parse::<TraceUri>().is_err()); // no org/repo/file
        assert!("hf://org/repo".parse::<TraceUri>().is_err()); // no file
    }
}
