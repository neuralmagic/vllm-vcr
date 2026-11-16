//! vLLM-Omni stage protocol.
//!
//! An omni head (API server and orchestrator) drives a pipeline of stages.
//! Every stage attaches to the head the same way ([`attach`]). On the data
//! path an LLM stage is vLLM's engine-core protocol with trailing fields
//! ([`llm`]), and a diffusion stage is a separate map-based protocol
//! ([`diffusion`]).

pub mod attach;
pub mod diffusion;
pub mod llm;
mod msgpack;
