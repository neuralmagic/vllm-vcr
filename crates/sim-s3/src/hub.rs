//! HuggingFace Hub file resolution driven by the `HF_*` env vars the vLLM frontend reads:
//! `HF_HUB_CACHE` (else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`), `HF_HUB_OFFLINE`,
//! `HF_ENDPOINT`, and `HF_TOKEN` (else the cached login token).

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use hf_hub::{Cache, Repo, RepoType};
use tracing::info;

/// HuggingFace Hub cache location and offline mode.
#[derive(Debug, Clone)]
pub struct HubSettings {
    cache: Cache,
    offline: bool,
}

impl HubSettings {
    /// Read the cache location and offline mode from `HF_HUB_CACHE`, `HF_HOME`, and
    /// `HF_HUB_OFFLINE` (`1`, `ON`, `YES`, or `TRUE`, case-insensitive).
    pub fn from_env() -> Self {
        let cache = std::env::var_os("HF_HUB_CACHE")
            .map(|path| Cache::new(path.into()))
            .unwrap_or_else(Cache::from_env);
        let offline = std::env::var("HF_HUB_OFFLINE").is_ok_and(|value| {
            matches!(
                value.to_ascii_uppercase().as_str(),
                "1" | "ON" | "YES" | "TRUE"
            )
        });
        Self { cache, offline }
    }

    pub fn new(cache_dir: PathBuf, offline: bool) -> Self {
        Self {
            cache: Cache::new(cache_dir),
            offline,
        }
    }

    pub fn cache_dir(&self) -> &Path {
        self.cache.path()
    }

    /// `filename` from the `main` snapshot of `repo_id`, downloading it into the cache on a
    /// miss. Fails without touching the network when offline and the file is not cached.
    pub fn get_blocking(
        &self,
        repo_type: RepoType,
        repo_id: &str,
        filename: &str,
    ) -> Result<PathBuf> {
        let repo = Repo::new(repo_id.to_string(), repo_type);
        if let Some(path) = self.cached(&repo, repo_id, filename)? {
            return Ok(path);
        }
        let mut builder =
            hf_hub::api::sync::ApiBuilder::from_env().with_cache_dir(self.cache.path().clone());
        if let Some(token) = env_token() {
            builder = builder.with_token(Some(token));
        }
        let api = builder
            .build()
            .map_err(|e| anyhow::anyhow!("initializing HuggingFace API: {e}"))?;
        api.repo(repo)
            .get(filename)
            .map_err(|e| anyhow::anyhow!("downloading {filename} from {repo_id}: {e}"))
    }

    /// Async form of [`Self::get_blocking`].
    pub async fn get(&self, repo_type: RepoType, repo_id: &str, filename: &str) -> Result<PathBuf> {
        let repo = Repo::new(repo_id.to_string(), repo_type);
        if let Some(path) = self.cached(&repo, repo_id, filename)? {
            return Ok(path);
        }
        let mut builder =
            hf_hub::api::tokio::ApiBuilder::from_env().with_cache_dir(self.cache.path().clone());
        if let Some(token) = env_token() {
            builder = builder.with_token(Some(token));
        }
        let api = builder
            .build()
            .map_err(|e| anyhow::anyhow!("initializing HuggingFace API: {e}"))?;
        api.repo(repo)
            .get(filename)
            .await
            .map_err(|e| anyhow::anyhow!("downloading {filename} from {repo_id}: {e}"))
    }

    /// The cached file, `None` when it must be downloaded, or an error when offline.
    fn cached(&self, repo: &Repo, repo_id: &str, filename: &str) -> Result<Option<PathBuf>> {
        if let Some(path) = self.cache.repo(repo.clone()).get(filename) {
            info!(repo = repo_id, file = filename, path = %path.display(), "HF: using cached file");
            return Ok(Some(path));
        }
        if self.offline {
            bail!(
                "{filename} from {repo_id} is not in HF cache {} and HF_HUB_OFFLINE is enabled",
                self.cache.path().display()
            );
        }
        info!(
            repo = repo_id,
            file = filename,
            cache = %self.cache.path().display(),
            "HF: downloading from HuggingFace Hub"
        );
        Ok(None)
    }
}

fn env_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .filter(|token| !token.is_empty())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use hf_hub::RepoType;

    use crate::hub::HubSettings;

    const SNAPSHOT_SHA: &str = "0123456789abcdef0123456789abcdef01234567";

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sim-s3-hub-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Lay out a Hub cache entry: `refs/main` naming a snapshot holding `files`.
    fn seed(cache: &Path, folder: &str, files: &[&str]) -> PathBuf {
        let repo = cache.join(folder);
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs").join("main"), SNAPSHOT_SHA).unwrap();
        let snapshot = repo.join("snapshots").join(SNAPSHOT_SHA);
        std::fs::create_dir_all(&snapshot).unwrap();
        for file in files {
            let path = snapshot.join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, file).unwrap();
        }
        snapshot
    }

    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }

    #[test]
    fn cached_model_file_resolves_offline() {
        let cache = scratch_dir("model");
        let snapshot = seed(&cache, "models--org--model", &["tokenizer.json"]);
        let hub = HubSettings::new(cache.clone(), true);
        let path = hub
            .get_blocking(RepoType::Model, "org/model", "tokenizer.json")
            .unwrap();
        assert_eq!(path, snapshot.join("tokenizer.json"));
        std::fs::remove_dir_all(&cache).unwrap();
    }

    #[test]
    fn cached_dataset_file_resolves_offline_async() {
        let cache = scratch_dir("dataset");
        let snapshot = seed(&cache, "datasets--org--traces", &["runs/trace.jsonl.gz"]);
        let hub = HubSettings::new(cache.clone(), true);
        let path =
            block_on(hub.get(RepoType::Dataset, "org/traces", "runs/trace.jsonl.gz")).unwrap();
        assert_eq!(path, snapshot.join("runs/trace.jsonl.gz"));
        std::fs::remove_dir_all(&cache).unwrap();
    }

    #[test]
    fn offline_miss_errors_with_repo_file_and_cache() {
        let cache = scratch_dir("miss");
        let hub = HubSettings::new(cache.clone(), true);
        for err in [
            hub.get_blocking(RepoType::Model, "org/absent", "tokenizer.json")
                .unwrap_err(),
            block_on(hub.get(RepoType::Dataset, "org/absent", "tokenizer.json")).unwrap_err(),
        ] {
            let err = err.to_string();
            assert!(err.contains("org/absent"), "{err}");
            assert!(err.contains("tokenizer.json"), "{err}");
            assert!(err.contains("HF_HUB_OFFLINE"), "{err}");
            assert!(err.contains(&cache.display().to_string()), "{err}");
        }
        std::fs::remove_dir_all(&cache).unwrap();
    }

    #[test]
    fn offline_snapshot_missing_file_errors() {
        let cache = scratch_dir("partial");
        seed(&cache, "models--org--model", &["config.json"]);
        let hub = HubSettings::new(cache.clone(), true);
        let err = hub
            .get_blocking(RepoType::Model, "org/model", "tokenizer.json")
            .unwrap_err()
            .to_string();
        assert!(err.contains("HF_HUB_OFFLINE"), "{err}");
        std::fs::remove_dir_all(&cache).unwrap();
    }

    #[test]
    fn cache_entries_are_scoped_by_repo_id_and_type() {
        let cache = scratch_dir("scoped");
        seed(&cache, "models--org--model", &["tokenizer.json"]);
        let hub = HubSettings::new(cache.clone(), true);
        assert!(
            hub.get_blocking(RepoType::Model, "org/other", "tokenizer.json")
                .is_err()
        );
        assert!(
            hub.get_blocking(RepoType::Dataset, "org/model", "tokenizer.json")
                .is_err()
        );
        std::fs::remove_dir_all(&cache).unwrap();
    }
}
