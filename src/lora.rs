//! LoRA adapter bookkeeping for the mock engine.
//!
//! The frontend owns the `/v1/load_lora_adapter` HTTP surface; it relays each load/unload to
//! us as an `add_lora`/`remove_lora` engine-core utility call, and stamps the chosen adapter
//! onto every `EngineCoreRequest.lora_request` it forwards. Our job on this side is to:
//!
//!   1. answer those utility calls with a truthful `bool` (so the frontend's load/unload
//!      endpoints succeed),
//!   2. cap how many *distinct* adapters share the running batch (vLLM's `--max-loras`), and
//!   3. let the engine count per-adapter running/waiting requests for the
//!      `running_lora_adapters` / `waiting_lora_adapters` scheduler stats, which the frontend
//!      turns into the `vllm:lora_requests_info` gauge.
//!
//! The counting itself lives on the engine (it reads the running batch and waiting queue);
//! this module owns only the loaded-adapter set and the slot cap.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use vllm_engine_core_client::protocol::EngineCoreRequest;

/// The lora identity (int id + name) the engine needs, off an `add_lora`
/// utility call or a request's `lora_request`. Our own subset that deserializes
/// the same msgpack at every supported line: the crate's typed
/// `protocol::lora::LoraRequest` only exists on 0.23+ (lora is opaque rmpv on
/// 0.22), and serde ignores the fields we don't name. `Serialize` is for tests
/// that round-trip an `add_lora` call through the wire.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct LoraSpec {
    pub lora_int_id: u64,
    pub lora_name: String,
}

/// The adapter name a request runs against, read from
/// `EngineCoreRequest.lora_request`. That field is a typed `LoraRequest` on
/// 0.23+ and opaque `rmpv::Value` on 0.22, so this is the one access that gates
/// on the `vllm_lora_typed` capability (emitted by build.rs).
pub(crate) fn request_lora_name(request: &EngineCoreRequest) -> Option<&str> {
    #[cfg(vllm_lora_typed)]
    {
        request.lora_request.as_ref().map(|l| l.lora_name.as_str())
    }
    #[cfg(not(vllm_lora_typed))]
    {
        let value = request.lora_request.as_ref()?;
        value
            .as_map()?
            .iter()
            .find(|(k, _)| k.as_str() == Some("lora_name"))
            .and_then(|(_, v)| v.as_str())
    }
}

/// The set of adapters the frontend has loaded into this engine, plus the running-batch slot
/// cap. A real vLLM engine would hold adapter weights here; we only need their identities.
#[derive(Debug)]
pub(crate) struct LoraRegistry {
    /// Loaded adapters, keyed by `lora_int_id` -> `lora_name`. Mutated by `add`/`remove`.
    loaded: BTreeMap<u64, String>,
    /// Maximum distinct adapters allowed in the running batch at once (vLLM `--max-loras`).
    /// `0` disables the cap, so adapter diversity never gates admission.
    max_loras: usize,
}

impl LoraRegistry {
    pub(crate) fn new(max_loras: usize) -> Self {
        Self {
            loaded: BTreeMap::new(),
            max_loras,
        }
    }

    /// Register (or refresh) a loaded adapter. Always succeeds: a real engine could OOM on
    /// weights, but we hold nothing, so we accept every adapter the frontend hands us. Returns
    /// `true` to match what the frontend's `add_lora` utility expects.
    pub(crate) fn add(&mut self, lora: LoraSpec) -> bool {
        self.loaded.insert(lora.lora_int_id, lora.lora_name);
        true
    }

    /// Drop a loaded adapter by its int id. Returns whether it was present, mirroring vLLM's
    /// `remove_lora` (the frontend treats `false` as "adapter was not loaded").
    pub(crate) fn remove(&mut self, lora_int_id: u64) -> bool {
        self.loaded.remove(&lora_int_id).is_some()
    }

    /// Whether a request needing adapter `lora_name` can join a running batch that currently
    /// holds `running_adapters` distinct adapters. A `None` adapter (base model) never needs a
    /// slot. An adapter already resident is free. Otherwise it needs a fresh slot, which exists
    /// only while the batch holds fewer than `max_loras` distinct adapters. `max_loras == 0`
    /// disables the cap entirely.
    pub(crate) fn admits<'a>(
        &self,
        lora_name: Option<&str>,
        mut running_adapters: impl Iterator<Item = &'a str>,
    ) -> bool {
        if self.max_loras == 0 {
            return true;
        }
        let Some(name) = lora_name else {
            return true;
        };
        let mut distinct = std::collections::BTreeSet::new();
        let mut resident = false;
        for adapter in running_adapters.by_ref() {
            if adapter == name {
                resident = true;
            }
            distinct.insert(adapter);
        }
        resident || distinct.len() < self.max_loras
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lora(name: &str, id: u64) -> LoraSpec {
        LoraSpec {
            lora_int_id: id,
            lora_name: name.to_string(),
        }
    }

    #[test]
    fn add_then_remove_reports_presence() {
        let mut reg = LoraRegistry::new(0);
        assert!(reg.add(lora("a", 1)));
        assert!(reg.remove(1), "loaded adapter removes as present");
        assert!(!reg.remove(1), "second remove reports absent");
        assert!(!reg.remove(99), "unknown id reports absent");
    }

    #[test]
    fn cap_disabled_admits_anything() {
        let reg = LoraRegistry::new(0);
        // Even with two adapters already running, a third is fine when the cap is off.
        assert!(reg.admits(Some("c"), ["a", "b"].into_iter()));
    }

    #[test]
    fn base_model_request_never_needs_a_slot() {
        let reg = LoraRegistry::new(1);
        // Batch already holds its one allowed adapter; a base-model (None) request still fits.
        assert!(reg.admits(None, ["a"].into_iter()));
    }

    #[test]
    fn resident_adapter_admits_even_when_full() {
        let reg = LoraRegistry::new(1);
        assert!(
            reg.admits(Some("a"), ["a"].into_iter()),
            "same adapter is free"
        );
    }

    #[test]
    fn new_adapter_blocked_when_slots_full() {
        let reg = LoraRegistry::new(1);
        assert!(
            !reg.admits(Some("b"), ["a"].into_iter()),
            "no slot for a new adapter"
        );
    }

    #[test]
    fn new_adapter_admits_while_slots_remain() {
        let reg = LoraRegistry::new(2);
        assert!(
            reg.admits(Some("b"), ["a"].into_iter()),
            "second distinct adapter fits"
        );
        assert!(
            !reg.admits(Some("c"), ["a", "b"].into_iter()),
            "third does not"
        );
    }
}
