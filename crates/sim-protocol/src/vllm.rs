//! The engine-core protocol surface, normalized across the support window.
//!
//! Every crate in this workspace imports the vLLM wire types from here rather
//! than from `vllm_engine_core_client::protocol` directly, so API churn across
//! the window is absorbed in one place. The 0.25 restructure (types moved into
//! `protocol::{request, output, sampling}`, the flat output struct went
//! private) left the window with 0.24; since 0.25/0.26/0.27 share that layout,
//! the re-exports below are unconditional again.
//!
//! What remains shimmed:
//!
//! - The output envelope ([`Envelope`]) serializes through a private struct, so
//!   the constructor/accessor functions below are the only way to build or read
//!   one. They also keep call sites stable if the envelope shape drifts again.
//! - [`request_batch`] fills `RequestBatchOutputs.weight_version`, added in 0.32
//!   (`vllm_weight_version`).
//! - [`lora_request`] builds a `LoraRequest`, which lost the positional
//!   `tensorizer_config_dict` slot in 0.33 (`vllm_lora_tensorizer`).
//! - [`output_stream`] flattens the per-delivery stream 0.32 introduced back to
//!   one item per output (`vllm_stream_delivery`).

use std::collections::BTreeSet;

use anyhow::{Result, anyhow};
use futures::Stream;

pub use vllm_engine_core_client::protocol::dtype::ModelDtype;
pub use vllm_engine_core_client::protocol::lora::LoraRequest;
pub use vllm_engine_core_client::protocol::multimodal::MmModality;
pub use vllm_engine_core_client::protocol::output::{
    EngineCoreEvent, EngineCoreEventType, EngineCoreFinishReason, EngineCoreOutput,
    decode_engine_core_outputs,
};
pub use vllm_engine_core_client::protocol::request::{EngineCoreRequest, EngineCoreRequestType};
pub use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;
pub use vllm_engine_core_client::protocol::stats::SchedulerStats;
pub use vllm_engine_core_client::protocol::utility::UtilityOutput;
pub use vllm_engine_core_client::protocol::{decode_msgpack, encode_msgpack};
pub use vllm_engine_core_client::{EngineCoreOutputStream, EngineCoreStreamOutput, EngineId};

/// The engine-core output envelope: what the engine sends back to a frontend
/// client, and what [`decode_engine_core_outputs`] returns.
///
/// The classified enum since 0.25. It serializes through a private flat struct,
/// so it is not safe to construct inline; use the constructors below.
pub type Envelope = vllm_engine_core_client::protocol::output::EngineCoreOutputs;

/// Engine identity from the sim's engine index.
///
/// The wire encoding is two-byte little-endian. Callers keep the CLI's u32
/// engine-count type and the range check lives here.
pub fn engine_id_from_index(engine_index: u32) -> Result<EngineId> {
    let narrow = u16::try_from(engine_index)
        .map_err(|_| anyhow!("engine index {engine_index} exceeds the wire's u16 range"))?;
    Ok(EngineId::from_engine_index(narrow))
}

/// A plain LoRA adapter request: no base-model override, tensorizer config, or
/// in-place reload.
pub fn lora_request(lora_name: &str, lora_int_id: u64, lora_path: &str) -> LoraRequest {
    LoraRequest {
        lora_name: lora_name.to_string(),
        lora_int_id,
        lora_path: lora_path.to_string(),
        base_model_name: None,
        #[cfg(vllm_lora_tensorizer)]
        tensorizer_config_dict: None,
        load_inplace: false,
        is_3d_lora_weight: false,
    }
}

/// One item per engine output from a client request stream.
///
/// 0.32 groups outputs held back by the stream interval into a single
/// delivery; earlier lines already yield one output per item.
pub fn output_stream(
    stream: EngineCoreOutputStream,
) -> impl Stream<Item = vllm_engine_core_client::Result<EngineCoreStreamOutput>> + Send + Unpin {
    #[cfg(vllm_stream_delivery)]
    return stream.into_outputs();
    #[cfg(not(vllm_stream_delivery))]
    return stream;
}

/// Build a request-batch envelope: per-request outputs for one engine tick,
/// optionally carrying scheduler stats and the set of requests that finished.
///
/// Note: an envelope with no outputs, no stats, and no finished requests has an
/// ambiguous wire shape and will not decode back (the crate's `TryFrom` rejects
/// it). Callers always set at least one of the three.
pub fn request_batch(
    engine_index: u32,
    outputs: Vec<EngineCoreOutput>,
    scheduler_stats: Option<Box<SchedulerStats>>,
    timestamp: f64,
    finished_requests: Option<BTreeSet<String>>,
) -> Envelope {
    vllm_engine_core_client::protocol::output::RequestBatchOutputs {
        engine_index,
        outputs,
        scheduler_stats,
        timestamp,
        finished_requests,
        #[cfg(vllm_weight_version)]
        weight_version: None,
    }
    .into()
}

/// Build a utility-call envelope: the response to an `add_lora` /
/// `remove_lora` / `reset_prefix_cache` call the frontend is awaiting.
pub fn utility(engine_index: u32, timestamp: f64, output: UtilityOutput) -> Envelope {
    vllm_engine_core_client::protocol::output::UtilityCallOutput {
        engine_index,
        timestamp,
        output,
    }
    .into()
}

/// The per-request outputs in an envelope, empty for a utility or DP-control one.
pub fn request_outputs(envelope: &Envelope) -> &[EngineCoreOutput] {
    match envelope {
        Envelope::RequestBatch(batch) => &batch.outputs,
        _ => &[],
    }
}

/// The utility-call result an envelope carries, if it is a utility response.
pub fn utility_output(envelope: &Envelope) -> Option<&UtilityOutput> {
    match envelope {
        Envelope::Utility(call) => Some(&call.output),
        _ => None,
    }
}

/// The scheduler stats an envelope carries, if any.
pub fn scheduler_stats(envelope: &Envelope) -> Option<&SchedulerStats> {
    match envelope {
        Envelope::RequestBatch(batch) => batch.scheduler_stats.as_deref(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::vllm::{
        EngineCoreFinishReason, EngineCoreOutput, UtilityOutput, decode_msgpack, encode_msgpack,
        engine_id_from_index, request_batch, request_outputs, scheduler_stats, utility,
    };

    fn output(request_id: &str) -> EngineCoreOutput {
        EngineCoreOutput {
            request_id: request_id.to_string(),
            new_token_ids: vec![7],
            finish_reason: Some(EngineCoreFinishReason::Length),
            ..Default::default()
        }
    }

    /// The envelope must survive a round trip through the wire on every line:
    /// this is the shim's whole contract, since 0.25+ serializes through a
    /// private struct we can no longer name.
    #[test]
    fn request_batch_round_trips_through_msgpack() {
        let envelope = request_batch(
            3,
            vec![output("req-1")],
            None,
            1234.5,
            Some(BTreeSet::from(["req-1".to_string()])),
        );

        let encoded = encode_msgpack(&envelope).expect("encode");
        let decoded: super::Envelope = decode_msgpack(&encoded).expect("decode");

        assert_eq!(decoded, envelope);
        assert_eq!(request_outputs(&decoded).len(), 1);
        assert_eq!(request_outputs(&decoded)[0].request_id, "req-1");
        assert!(scheduler_stats(&decoded).is_none());
    }

    #[test]
    fn utility_round_trips_and_has_no_request_outputs() {
        let envelope = utility(
            0,
            9.0,
            UtilityOutput {
                call_id: 42_u64.into(),
                failure_message: None,
                result: None,
            },
        );

        let encoded = encode_msgpack(&envelope).expect("encode");
        let decoded: super::Envelope = decode_msgpack(&encoded).expect("decode");

        assert_eq!(decoded, envelope);
        assert!(request_outputs(&decoded).is_empty());
    }

    /// The identity is two-byte little-endian on every line; indexes that
    /// cannot encode must error instead of aliasing another engine.
    #[test]
    fn engine_id_round_trips_and_rejects_wide_indexes() {
        let id = engine_id_from_index(3).expect("engine index 3");
        assert_eq!(id.engine_index(), Some(3));
        assert!(engine_id_from_index(u32::from(u16::MAX) + 1).is_err());
    }
}
