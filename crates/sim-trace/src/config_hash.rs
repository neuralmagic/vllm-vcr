//! Canonical deployment-config fingerprint.
//!
//! The CI profile-once/replay-many cache and the conformance gate both need a
//! stable answer to "was this trace captured under a config I can replay
//! against?". That answer is a hash of the deployment inputs that actually
//! change replay behavior. Computing it from a typed struct (rather than
//! passing an opaque `--config-hash` string by hand) means capture and replay
//! derive the same value from the same inputs, so a mismatch is always a real
//! config difference, never a typo.
//!
//! The canonical form is versioned (`SCHEME`). If the input set ever changes,
//! bump the scheme: old hashes then deliberately stop matching, because a trace
//! captured under the old input set is no longer interchangeable.

use sha2::{Digest, Sha256};

/// Canonical-form scheme tag. Bump when the fingerprint inputs change.
///
/// v2 adds the scheduler/decode-mode inputs (`enable_prefix_caching`,
/// `speculative`): these change replay behavior, so a prefix-cache-on trace and a
/// prefix-cache-off trace (or a spec-decode trace) of the same model/hardware must
/// not be interchangeable. v1 goldens keep their v1 hashes and stay valid (the sim
/// compares the stamped hash, it never recomputes), so the bump only affects new
/// captures.
const SCHEME: &str = "config-fingerprint-v2";

/// The deployment-config inputs that determine whether a captured trace is
/// valid to replay. Two deployments with the same fingerprint produce
/// interchangeable traces; any difference means the trace must not be replayed
/// against the other config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigFingerprint {
    /// Model identifier (HF repo or local path) the trace was captured against.
    pub model: String,
    /// GPU type, e.g. `"H200"`. Affects absolute latencies.
    pub gpu: String,
    /// Tensor-parallel degree.
    pub tp: u32,
    /// Tokens per KV block; changes prefix-cache structure.
    pub block_size: u32,
    /// Scheduler concurrency ceiling (`max_num_seqs`).
    pub max_num_seqs: u64,
    /// vLLM line/tag the engine spoke, e.g. `"v0.23.0"`. A different vLLM
    /// version can change scheduling/step behavior even at the same knobs.
    pub vllm_tag: String,
    /// Whether the engine ran with prefix caching on. Cache hits skip prefill, so
    /// a cache-on and a cache-off trace of the same workload are not interchangeable.
    pub enable_prefix_caching: bool,
    /// Speculative-decoding config the engine ran, as a canonical descriptor (e.g.
    /// `"ngram-k3"`), or `None` for standard decoding. Spec decode emits multiple
    /// tokens per step, changing the per-chunk ITL structure the replay reproduces.
    pub speculative: Option<String>,
}

impl ConfigFingerprint {
    /// The exact, order-fixed string that gets hashed. Never reorder fields:
    /// the order is part of the contract.
    fn canonical(&self) -> String {
        format!(
            "{SCHEME}\n\
             model={}\n\
             gpu={}\n\
             tp={}\n\
             block_size={}\n\
             max_num_seqs={}\n\
             vllm_tag={}\n\
             enable_prefix_caching={}\n\
             speculative={}\n",
            self.model,
            self.gpu,
            self.tp,
            self.block_size,
            self.max_num_seqs,
            self.vllm_tag,
            self.enable_prefix_caching,
            self.speculative.as_deref().unwrap_or("none"),
        )
    }

    /// Lowercase-hex SHA-256 of the canonical form. This is the value stamped
    /// into `TraceMeta.config_hash` at capture and checked via
    /// `--expect-config-hash` at replay.
    pub fn hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.canonical().as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

#[cfg(test)]
mod tests {
    use crate::config_hash::ConfigFingerprint;

    fn sample() -> ConfigFingerprint {
        ConfigFingerprint {
            model: "Qwen/Qwen3-8B".to_string(),
            gpu: "H200".to_string(),
            tp: 1,
            block_size: 16,
            max_num_seqs: 256,
            vllm_tag: "v0.23.0".to_string(),
            enable_prefix_caching: true,
            speculative: None,
        }
    }

    #[test]
    fn hash_is_deterministic() {
        assert_eq!(sample().hash(), sample().hash());
        // 64 hex chars = 32-byte sha256.
        assert_eq!(sample().hash().len(), 64);
        assert!(sample().hash().chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn every_field_changes_the_hash() {
        let base = sample().hash();
        let mutated = [
            ConfigFingerprint {
                model: "other/model".to_string(),
                ..sample()
            },
            ConfigFingerprint {
                gpu: "A100".to_string(),
                ..sample()
            },
            ConfigFingerprint { tp: 2, ..sample() },
            ConfigFingerprint {
                block_size: 32,
                ..sample()
            },
            ConfigFingerprint {
                max_num_seqs: 128,
                ..sample()
            },
            ConfigFingerprint {
                vllm_tag: "v0.22.1".to_string(),
                ..sample()
            },
            ConfigFingerprint {
                enable_prefix_caching: false,
                ..sample()
            },
            ConfigFingerprint {
                speculative: Some("ngram-k3".to_string()),
                ..sample()
            },
        ];
        for m in mutated {
            assert_ne!(base, m.hash(), "changing a field must change the hash");
        }
    }

    #[test]
    fn distinct_field_values_do_not_collide_across_boundaries() {
        // tp=1, block_size=16 must not hash the same as tp=16, block_size=1:
        // the field labels in the canonical form prevent value bleed.
        let a = ConfigFingerprint {
            tp: 1,
            block_size: 16,
            ..sample()
        };
        let b = ConfigFingerprint {
            tp: 16,
            block_size: 1,
            ..sample()
        };
        assert_ne!(a.hash(), b.hash());
    }
}
