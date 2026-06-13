//! Conversions between the engine-core wire protocol and the trace schema.
//!
//! These live here, as free functions, rather than as `From` impls on
//! `TraceFinishReason`: the trace schema is a vLLM-protocol-free crate, and the
//! orphan rule forbids implementing `From<EngineCoreFinishReason>` for a type
//! it owns from outside it. Free functions sidestep that entirely.

use sim_trace::trace::TraceFinishReason;
use vllm_engine_core_client::protocol::EngineCoreFinishReason;

/// Map a wire finish reason to its trace-schema form.
pub fn trace_finish_reason(reason: EngineCoreFinishReason) -> TraceFinishReason {
    match reason {
        EngineCoreFinishReason::Stop => TraceFinishReason::Stop,
        EngineCoreFinishReason::Length => TraceFinishReason::Length,
        EngineCoreFinishReason::Abort => TraceFinishReason::Abort,
        EngineCoreFinishReason::Error => TraceFinishReason::Error,
        EngineCoreFinishReason::Repetition => TraceFinishReason::Repetition,
    }
}

/// Map a trace-schema finish reason back to the wire form (used when replaying
/// a recorded trace through the engine).
pub fn engine_finish_reason(reason: TraceFinishReason) -> EngineCoreFinishReason {
    match reason {
        TraceFinishReason::Stop => EngineCoreFinishReason::Stop,
        TraceFinishReason::Length => EngineCoreFinishReason::Length,
        TraceFinishReason::Abort => EngineCoreFinishReason::Abort,
        TraceFinishReason::Error => EngineCoreFinishReason::Error,
        TraceFinishReason::Repetition => EngineCoreFinishReason::Repetition,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finish_reason_protocol_conversions_are_inverse() {
        for reason in [
            TraceFinishReason::Stop,
            TraceFinishReason::Length,
            TraceFinishReason::Abort,
            TraceFinishReason::Error,
            TraceFinishReason::Repetition,
        ] {
            let wire = engine_finish_reason(reason);
            assert_eq!(trace_finish_reason(wire), reason);
        }
    }
}
