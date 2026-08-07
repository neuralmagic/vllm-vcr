//! The engine-core protocol surface, normalized across the support window.
//!
//! Every crate in this workspace imports the vLLM wire types from here rather
//! than from `vllm_engine_core_client::protocol` directly, because where those
//! types live (and, for the output envelope, what they *are*) differs by line:
//!
//! - Up to 0.24, `protocol` re-exported the whole surface, and
//!   `EngineCoreOutputs` was the flat wire struct the engine fills in.
//! - From 0.25 the types moved into `protocol::{request, output, sampling}` and
//!   `mod.rs` stopped re-exporting them. `EngineCoreOutputs` was repurposed to
//!   the classified enum (0.24's `ClassifiedEngineCoreOutputs`) and the flat
//!   struct became private, so there is no longer a public struct to fill in.
//!
//! The wire itself did not move: both lines encode the same 8-field tuple. Only
//! the Rust API churned. So the re-exports below carry the identical types
//! across lines, and [`Envelope`] plus its constructors absorb the one place
//! where the shape of our own code has to differ.
//!
//! Gated on `vllm_outputs_enum` (see `sim_compat::capabilities`), which this
//! crate's `build.rs` emits for 0.25+.

use std::collections::BTreeSet;

#[cfg(vllm_outputs_enum)]
pub use vllm_engine_core_client::protocol::output::{
    EngineCoreFinishReason, EngineCoreOutput, decode_engine_core_outputs,
};
#[cfg(vllm_outputs_enum)]
pub use vllm_engine_core_client::protocol::request::{EngineCoreRequest, EngineCoreRequestType};
#[cfg(vllm_outputs_enum)]
pub use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;
#[cfg(not(vllm_outputs_enum))]
pub use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreRequest, EngineCoreRequestType,
    EngineCoreSamplingParams, decode_engine_core_outputs,
};

// Same path on every line in the window.
pub use vllm_engine_core_client::protocol::dtype::ModelDtype;
pub use vllm_engine_core_client::protocol::stats::SchedulerStats;
pub use vllm_engine_core_client::protocol::utility::UtilityOutput;
pub use vllm_engine_core_client::protocol::{decode_msgpack, encode_msgpack};

/// The engine-core output envelope: what the engine sends back to a frontend
/// client, and what [`decode_engine_core_outputs`] returns.
///
/// The classified enum on 0.25+, the flat wire struct before that. Both encode
/// to the same msgpack tuple, so this alias is safe to put on the wire; what it
/// is *not* safe to do is construct it inline, which is why the constructors
/// below exist.
#[cfg(vllm_outputs_enum)]
pub type Envelope = vllm_engine_core_client::protocol::output::EngineCoreOutputs;
#[cfg(not(vllm_outputs_enum))]
pub type Envelope = vllm_engine_core_client::protocol::EngineCoreOutputs;

/// Build a request-batch envelope: per-request outputs for one engine tick,
/// optionally carrying scheduler stats and the set of requests that finished.
///
/// Note for 0.25+: an envelope with no outputs, no stats, and no finished
/// requests has an ambiguous wire shape and will not decode back (the crate's
/// `TryFrom` rejects it). Callers always set at least one of the three.
pub fn request_batch(
    engine_index: u32,
    outputs: Vec<EngineCoreOutput>,
    scheduler_stats: Option<Box<SchedulerStats>>,
    timestamp: f64,
    finished_requests: Option<BTreeSet<String>>,
) -> Envelope {
    #[cfg(vllm_outputs_enum)]
    {
        vllm_engine_core_client::protocol::output::RequestBatchOutputs {
            engine_index,
            outputs,
            scheduler_stats,
            timestamp,
            finished_requests,
        }
        .into()
    }
    #[cfg(not(vllm_outputs_enum))]
    {
        Envelope {
            engine_index,
            outputs,
            scheduler_stats,
            timestamp,
            finished_requests,
            ..Default::default()
        }
    }
}

/// Build a utility-call envelope: the response to an `add_lora` /
/// `remove_lora` / `reset_prefix_cache` call the frontend is awaiting.
pub fn utility(engine_index: u32, timestamp: f64, output: UtilityOutput) -> Envelope {
    #[cfg(vllm_outputs_enum)]
    {
        vllm_engine_core_client::protocol::output::UtilityCallOutput {
            engine_index,
            timestamp,
            output,
        }
        .into()
    }
    #[cfg(not(vllm_outputs_enum))]
    {
        Envelope {
            engine_index,
            timestamp,
            utility_output: Some(output),
            ..Default::default()
        }
    }
}

/// The per-request outputs in an envelope, empty for a utility or DP-control one.
pub fn request_outputs(envelope: &Envelope) -> &[EngineCoreOutput] {
    #[cfg(vllm_outputs_enum)]
    {
        match envelope {
            Envelope::RequestBatch(batch) => &batch.outputs,
            _ => &[],
        }
    }
    #[cfg(not(vllm_outputs_enum))]
    {
        &envelope.outputs
    }
}

/// The utility-call result an envelope carries, if it is a utility response.
pub fn utility_output(envelope: &Envelope) -> Option<&UtilityOutput> {
    #[cfg(vllm_outputs_enum)]
    {
        match envelope {
            Envelope::Utility(call) => Some(&call.output),
            _ => None,
        }
    }
    #[cfg(not(vllm_outputs_enum))]
    {
        envelope.utility_output.as_ref()
    }
}

/// The scheduler stats an envelope carries, if any.
pub fn scheduler_stats(envelope: &Envelope) -> Option<&SchedulerStats> {
    #[cfg(vllm_outputs_enum)]
    {
        match envelope {
            Envelope::RequestBatch(batch) => batch.scheduler_stats.as_deref(),
            _ => None,
        }
    }
    #[cfg(not(vllm_outputs_enum))]
    {
        envelope.scheduler_stats.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::vllm::{
        EngineCoreFinishReason, EngineCoreOutput, UtilityOutput, decode_msgpack, encode_msgpack,
        request_batch, request_outputs, scheduler_stats, utility,
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
}
