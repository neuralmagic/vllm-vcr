//! Helpers for the `kv_transfer_params` the frontend ferries down from the
//! OpenAI request.
//!
//! The server merges them into `sampling_params.extra_args["kv_transfer_params"]`
//! (mirroring Python vLLM), so that is where the P/D intent (`do_remote_prefill`
//! / `do_remote_decode` / `remote_*`) arrives. In real vLLM the produce/consume
//! logic lives in the NixlConnector inside the engine; here our data plane plays
//! that role. Both the engine loop and the recording tap need to read these, so
//! the extraction lives in its own module.

use serde_json::Value as JsonValue;
use vllm_engine_core_client::protocol::EngineCoreRequest;

/// Pull the `kv_transfer_params` object out of a request, if present.
pub fn extract_kv_params(request: &EngineCoreRequest) -> Option<JsonValue> {
    request
        .sampling_params
        .as_ref()?
        .extra_args
        .as_ref()?
        .get("kv_transfer_params")
        .cloned()
}

/// Read a boolean flag out of a `kv_transfer_params` object.
pub fn kv_flag(kv: &JsonValue, key: &str) -> bool {
    kv.get(key).and_then(JsonValue::as_bool).unwrap_or(false)
}
