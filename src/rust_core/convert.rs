//! Conversions between the engine-core wire protocol of the line being built and
//! the scheduler crate's own types.

use std::sync::Arc;

use sim_protocol::mock_engine::DEFAULT_MOCK_MAX_MODEL_LEN;
use sim_protocol::vllm::{
    EngineCoreEvent, EngineCoreEventType, EngineCoreFinishReason, EngineCoreOutput,
    EngineCoreRequest, EngineCoreSamplingParams, SchedulerStats,
};
use vllm_engine_core::hash::{BlockHash, HashAlgo, RequestBlockHasher, RequestExtraKeys, Value};
use vllm_engine_core::kv_cache::PrefixCacheStats as CorePrefixCacheStats;
use vllm_engine_core::kv_cache::{
    ExternalBlockHash, KvCacheConfig, KvCacheEvent, KvCacheGroupSpec, KvCacheSpec, KvCacheSpecType,
};
use vllm_engine_core::sched::request::{
    BlockHasher, EngineCoreEventType as CoreEventType, FinishReason, NewRequest, Request,
    SamplingParams, StopReason,
};
use vllm_engine_core::sched::request_queue::SchedulingPolicy as CorePolicy;
use vllm_engine_core::sched::scheduler::{
    EngineCoreOutput as CoreOutput, KvTransferConfig, KvTransferParams, SchedulerConfig,
};
use vllm_engine_core::sched::stats::SchedulerStats as CoreStats;
use vllm_engine_core_client::protocol::output::StopReason as WireStopReason;
use vllm_engine_core_client::protocol::stats::{BaseCacheStats, PrefillStats, PrefixCacheStats};
use vllm_engine_core_connectors::nixl::{NixlFinishedParams, TransferMode};

use crate::dataplane::PdRole;
use crate::{Opt, SchedulingPolicy, blockpool};

const HASH_ALGO: HashAlgo = HashAlgo::Sha256Cbor;

pub(crate) fn scheduler_config(opt: &Opt) -> SchedulerConfig {
    let max_model_len = if opt.max_model_len > 0 {
        opt.max_model_len
    } else {
        DEFAULT_MOCK_MAX_MODEL_LEN
    };
    SchedulerConfig {
        max_num_seqs: opt.max_num_seqs.max(1) as usize,
        max_num_batched_tokens: opt.max_num_batched_tokens.max(1) as usize,
        long_prefill_token_threshold: opt.long_prefill_token_threshold as usize,
        policy: match opt.scheduling_policy {
            SchedulingPolicy::Fcfs => CorePolicy::Fcfs,
            SchedulingPolicy::Priority => CorePolicy::Priority,
        },
        num_gpu_blocks: opt.kv_cache_size as usize,
        enable_prefix_caching: true,
        max_loras: (opt.max_loras > 0).then_some(opt.max_loras as usize),
        kv_transfer: Some(KvTransferConfig {
            is_kv_producer: opt.pd_role != PdRole::Decode,
            is_kv_consumer: opt.pd_role != PdRole::Prefill,
            recompute_kv_load_failures: true,
        }),
        log_stats: true,
        enable_kv_cache_events: opt.enable_kv_cache_events,
        kv_events_use_int_block_hashes: true,
        ..SchedulerConfig::for_test(opt.tokens_per_block, max_model_len as usize)
    }
}

pub(crate) fn kv_cache_config(opt: &Opt) -> KvCacheConfig {
    KvCacheConfig::new(
        opt.kv_cache_size as usize,
        vec![KvCacheGroupSpec::new(KvCacheSpec::new(
            KvCacheSpecType::FullAttention,
            opt.tokens_per_block,
        ))],
    )
}

pub(crate) fn request_block_hasher(opt: &Opt) -> Arc<RequestBlockHasher> {
    let none_hash = HASH_ALGO.hash_value(&Value::str(opt.kv_cache_none_seed.to_string()));
    Arc::new(RequestBlockHasher::new(
        HASH_ALGO,
        none_hash,
        opt.tokens_per_block,
        0,
    ))
}

struct PromptHasher {
    hasher: Arc<RequestBlockHasher>,
    extra_keys: RequestExtraKeys,
}

impl BlockHasher for PromptHasher {
    fn hash_new_blocks(&self, request: &Request) -> Vec<BlockHash> {
        self.hasher.new_block_hashes(
            request.all_token_ids(),
            request.block_hashes.len(),
            request.block_hashes.last(),
            &self.extra_keys,
        )
    }
}

/// The scheduler's view of an incoming request, generating at most `max_tokens`.
pub(crate) fn new_request(
    request: &EngineCoreRequest,
    sampling: &EngineCoreSamplingParams,
    max_tokens: usize,
    hasher: &Arc<RequestBlockHasher>,
) -> NewRequest {
    let lora = request.lora_request.as_ref();
    let cache_salt = request.cache_salt.clone().filter(|salt| !salt.is_empty());
    NewRequest {
        request_id: request.request_id.clone(),
        prompt_token_ids: Some(
            request
                .prompt_token_ids
                .iter()
                .flatten()
                .map(|&token| token as i32)
                .collect(),
        ),
        sampling_params: Some(SamplingParams {
            max_tokens: Some(max_tokens),
            min_tokens: sampling.min_tokens as usize,
            eos_token_id: sampling.eos_token_id.map(|token| token as i32),
            stop_token_ids: sampling
                .stop_token_ids
                .iter()
                .map(|&token| token as i32)
                .collect(),
            watermarking: sampling.watermarking,
            temperature: f64::from(sampling.temperature),
            ..Default::default()
        }),
        client_index: request.client_index as usize,
        arrival_time: Some(request.arrival_time),
        lora_int_id: lora.map(|lora| lora.lora_int_id as i64),
        lora_name: lora.map(|lora| lora.lora_name.clone()),
        priority: i64::from(request.priority),
        block_hasher: Some(Arc::new(PromptHasher {
            hasher: Arc::clone(hasher),
            extra_keys: RequestExtraKeys {
                lora: lora.map(|lora| {
                    (
                        Value::str(lora.lora_name.as_str()),
                        Value::str(lora.lora_path.as_str()),
                    )
                }),
                cache_salt: cache_salt.as_deref().map(Value::str),
            },
        })),
        cache_salt,
        ..Default::default()
    }
}

pub(crate) fn finish_reason(reason: FinishReason) -> EngineCoreFinishReason {
    match reason {
        FinishReason::Stop => EngineCoreFinishReason::Stop,
        FinishReason::Length => EngineCoreFinishReason::Length,
        FinishReason::Abort => EngineCoreFinishReason::Abort,
        FinishReason::Error => EngineCoreFinishReason::Error,
        FinishReason::Repetition => EngineCoreFinishReason::Repetition,
    }
}

/// The `kv_transfer_params` dict a finished prefill request hands to the decoder.
fn kv_transfer_params(params: &KvTransferParams) -> Option<serde_json::Value> {
    let finished = params.params.0.downcast_ref::<NixlFinishedParams>()?;
    let mut dict = serde_json::json!({
        "do_remote_prefill": finished.do_remote_prefill,
        "do_remote_decode": finished.do_remote_decode,
        "remote_block_ids": finished.remote_block_ids,
        "remote_engine_id": finished.remote_engine_id.as_ref(),
        "remote_request_id": finished.remote_request_id,
        "remote_host": finished.remote_host.as_ref(),
        "remote_port": finished.remote_port,
        "tp_size": finished.tp_size,
        "dcp_size": finished.dcp_size,
        "pp_size": finished.pp_size,
        "remote_num_tokens": finished.remote_num_tokens,
        "transfer_mode": finished.mode.as_str(),
    });
    if finished.mode == TransferMode::Pull {
        dict["remote_blocks_expiry_time"] = finished.remote_blocks_expiry_time.into();
        dict["remote_registration_generation"] = finished.remote_registration_generation.into();
    }
    if let Some(cached) = params.remote_prefill_cached_tokens {
        dict["remote_prefill_cached_tokens"] = cached.into();
    }
    Some(dict)
}

pub(crate) fn output(output: CoreOutput) -> EngineCoreOutput {
    EngineCoreOutput {
        kv_transfer_params: output
            .kv_transfer_params
            .as_ref()
            .and_then(kv_transfer_params)
            .map(Box::new),
        request_id: output.request_id,
        new_token_ids: output
            .new_token_ids
            .into_iter()
            .map(|token| token as u32)
            .collect(),
        finish_reason: output.finish_reason.map(finish_reason),
        stop_reason: output.stop_reason.map(|reason| match reason {
            StopReason::TokenId(token) => WireStopReason::TokenId(token),
            StopReason::Text(text) => WireStopReason::Text(text),
        }),
        events: output.events.map(|events| {
            events
                .into_iter()
                .map(|event| EngineCoreEvent {
                    r#type: match event.r#type {
                        CoreEventType::Queued => EngineCoreEventType::Queued,
                        CoreEventType::Scheduled => EngineCoreEventType::Scheduled,
                        CoreEventType::Preempted => EngineCoreEventType::Preempted,
                    },
                    timestamp: event.timestamp,
                })
                .collect()
        }),
        prefill_stats: output.prefill_stats.map(|stats| {
            Box::new(PrefillStats {
                num_prompt_tokens: stats.num_prompt_tokens,
                num_computed_tokens: stats.num_computed_tokens,
                num_cached_tokens: stats.num_cached_tokens,
                num_local_cached_tokens: stats.num_local_cached_tokens,
                num_external_cached_tokens: stats.num_external_cached_tokens,
                num_cache_creation_tokens: stats.num_cache_creation_tokens,
            })
        }),
        num_nans_in_logits: output.num_nans_in_logits as u32,
        ..Default::default()
    }
}

fn prefix_cache_stats(stats: &CorePrefixCacheStats) -> PrefixCacheStats {
    PrefixCacheStats {
        base: BaseCacheStats {
            reset: stats.reset,
            requests: stats.requests as u64,
            queries: stats.queries as u64,
            hits: stats.hits as u64,
        },
        preempted_requests: stats.preempted_requests as u64,
        preempted_queries: stats.preempted_queries as u64,
        preempted_hits: stats.preempted_hits as u64,
        ..Default::default()
    }
}

pub(crate) fn scheduler_stats(stats: CoreStats) -> SchedulerStats {
    SchedulerStats {
        num_running_reqs: stats.num_running_reqs as u64,
        num_waiting_reqs: stats.num_waiting_reqs as u64,
        num_skipped_waiting_reqs: stats.num_skipped_waiting_reqs as u64,
        kv_cache_usage: stats.kv_cache_usage,
        prefix_cache_stats: prefix_cache_stats(&stats.prefix_cache_stats),
        connector_prefix_cache_stats: stats
            .connector_prefix_cache_stats
            .as_ref()
            .map(|connector| prefix_cache_stats(&connector.stats)),
        ..Default::default()
    }
}

fn add_prefix_cache_stats(total: &mut PrefixCacheStats, older: &PrefixCacheStats) {
    total.base.reset |= older.base.reset;
    total.base.requests += older.base.requests;
    total.base.queries += older.base.queries;
    total.base.hits += older.base.hits;
    total.preempted_requests += older.preempted_requests;
    total.preempted_queries += older.preempted_queries;
    total.preempted_hits += older.preempted_hits;
}

/// Fold a newer stats snapshot into one not yet sent: gauges are replaced,
/// per-report cache counters add up.
pub(crate) fn merge_stats(pending: &mut Option<SchedulerStats>, newer: SchedulerStats) {
    let Some(older) = pending.take() else {
        *pending = Some(newer);
        return;
    };
    let mut merged = newer;
    add_prefix_cache_stats(&mut merged.prefix_cache_stats, &older.prefix_cache_stats);
    match (
        &mut merged.connector_prefix_cache_stats,
        older.connector_prefix_cache_stats,
    ) {
        (Some(total), Some(older)) => add_prefix_cache_stats(total, &older),
        (total @ None, older) => *total = older,
        (Some(_), None) => {}
    }
    *pending = Some(merged);
}

fn event_hash(hash: &ExternalBlockHash) -> blockpool::BlockHash {
    match hash {
        ExternalBlockHash::Int(hash) => *hash,
        ExternalBlockHash::Bytes(bytes) => {
            let bytes = bytes.as_bytes();
            bytes[bytes.len().saturating_sub(8)..]
                .iter()
                .fold(0, |acc, &byte| (acc << 8) | u64::from(byte))
        }
    }
}

pub(crate) fn kv_event(event: KvCacheEvent) -> blockpool::KvCacheEvent {
    match event {
        KvCacheEvent::BlockStored(stored) => blockpool::KvCacheEvent::Stored {
            block_hashes: stored.block_hashes.iter().map(event_hash).collect(),
            parent_hash: stored.parent_block_hash.as_ref().map(event_hash),
            token_ids: stored
                .token_ids
                .into_iter()
                .map(|token| token as u32)
                .collect(),
            block_size: stored.block_size,
            lora_name: stored.lora_name,
        },
        KvCacheEvent::BlockRemoved(removed) => blockpool::KvCacheEvent::Removed {
            block_hashes: removed.block_hashes.iter().map(event_hash).collect(),
        },
        KvCacheEvent::AllBlocksCleared(_) => blockpool::KvCacheEvent::AllCleared,
    }
}
