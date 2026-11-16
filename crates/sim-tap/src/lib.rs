//! Engine-core recording tap: a transparent ZMQ proxy between a real vLLM
//! frontend and a real engine-core that records per-request timing into a JSONL
//! trace (plus an optional step-stats sidecar stream). Behind a vLLM-Omni head
//! it taps one stage, LLM ([`tap`]) or diffusion ([`diffusion`]).

pub mod diffusion;
pub mod omni;
pub mod tap;
