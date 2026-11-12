//! The worker half of the NIXL KV connector, on the simulated data plane.
//!
//! vLLM's NIXL scheduler half (`vllm-engine-core-connectors`) decides what moves;
//! each step it hands the worker a `NixlConnectorMetadata`, and the worker reports
//! what finished in the next `KvConnectorOutput`.
//!
//! ```text
//!   pull                prefill engine                     decode engine
//!   reqs_to_send ─▶ stage(): write blocks,         reqs_to_recv ─▶ start_pull(): READ the
//!                   hold them on a lease                           remote blocks
//!        ▲                                               │
//!        ├── "<request>:1"  the request was read ◀───────┤
//!        └── "HB:<requests>"  extend their leases ◀──────┘ heartbeat()
//!
//!   push                prefill engine                     decode engine
//!        ┌── "PUSH_REG:{...}"  where to write ◀────────── register_push()
//!        ▼                                         reqs_to_recv ─▶ expect_push()
//!   push_finished_blocks ─▶ finished_blocks()             ▲
//!        └── WRITE the blocks, then "<request>:1" ────────┘ check the bytes
//!
//!   poll() ─▶ finished_sending                     poll() ─▶ finished_recving / failed_recving
//! ```
//!
//! Reads, writes and outgoing messages run on one transfer thread per engine, so
//! the step clock never waits on the network.

use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::time::Instant;
use tracing::{debug, info, warn};
use vllm_engine_core::sched::scheduler::KvConnectorOutput;
use vllm_engine_core_connectors::nixl::{
    NixlConnectorScheduler, NixlKvConnector, NixlSchedulerConfig, PushRegistration, RemoteBlockIds,
    RemoteMeta, TransferMode,
};

use crate::dataplane::{
    BlockRead, BlockWrite, KvDataPlane, NixlConfig, PdRole, RemotePool, make_data_plane,
};
use crate::{KvTransferMode, Opt};

/// vLLM's `kv_lease_duration` default.
const KV_LEASE: Duration = Duration::from_secs(30);
/// How far a heartbeat extends a lease: vLLM's `kv_lease_duration * 2 // 3`.
const LEASE_EXTENSION: Duration = Duration::from_secs(20);
/// vLLM's `decoder_kv_blocks_ttl` default, which is also its push registration timeout.
const DECODER_KV_BLOCKS_TTL: Duration = Duration::from_secs(480);
/// vLLM's `kv_recompute_threshold` default, in tokens.
const KV_RECOMPUTE_THRESHOLD: i64 = 64;

const HEARTBEAT_PREFIX: &str = "HB:";
const PUSH_REGISTRATION_PREFIX: &str = "PUSH_REG:";

fn transfer_mode(opt: &Opt) -> TransferMode {
    match opt.kv_transfer_mode {
        KvTransferMode::Pull => TransferMode::Pull,
        KvTransferMode::Push => TransferMode::Push,
    }
}

/// The NIXL scheduler half for one engine, over a single full-attention KV cache group.
pub(crate) fn scheduler_half(opt: &Opt, engine_index: u32) -> anyhow::Result<NixlKvConnector> {
    let config = NixlSchedulerConfig {
        mode: transfer_mode(opt),
        engine_id: opt.engine_id.as_str().into(),
        side_channel_host: opt.side_channel_host.as_str().into(),
        side_channel_port: i64::from(opt.side_channel_port + engine_index),
        transfer_tp_size: 1,
        tensor_parallel_size: 1,
        decode_context_parallel_size: 1,
        pipeline_parallel_size: 1,
        kv_lease_duration: KV_LEASE.as_secs_f64(),
        heartbeat_interval: (KV_LEASE.as_secs() / 6) as f64,
        use_host_buffer: false,
        is_hma_required: false,
        num_kv_cache_groups: 1,
        transfer_group_ids: vec![0],
        blocks_per_sw: vec![0],
        ssm_spec_blocks: vec![None],
        has_mamba: false,
        bounded_replay: false,
        num_prefill_lookahead_tokens: 0,
        kv_recompute_threshold: KV_RECOMPUTE_THRESHOLD,
        bidirectional_kv_xfer: false,
        decoder_kv_blocks_ttl: DECODER_KV_BLOCKS_TTL.as_secs_f64(),
        is_kv_producer: opt.pd_role != PdRole::Decode,
        push_registration_timeout: DECODER_KV_BLOCKS_TTL.as_secs_f64(),
    };
    let scheduler = NixlConnectorScheduler::new(config)
        .map_err(|error| anyhow::anyhow!("configuring the NIXL connector: {error}"))?;
    Ok(NixlKvConnector::new(scheduler))
}

/// The remote block ids of the single transferred KV cache group.
pub(crate) fn remote_block_ids(remote: &RemoteMeta) -> &[i64] {
    match remote.block_ids.as_ref() {
        RemoteBlockIds::Flat(block_ids) => &block_ids.items,
        RemoteBlockIds::Grouped(groups) => groups
            .items
            .first()
            .map_or(&[], |block_ids| &block_ids.items),
    }
}

fn pool(engine_id: &str, host: &str, port: i64) -> RemotePool {
    RemotePool {
        engine_id: engine_id.to_string(),
        host: host.to_string(),
        port: u32::try_from(port).unwrap_or_default(),
    }
}

/// A request id without vLLM's per-request `-<8 hex>` suffix. The prefill and
/// decode halves of one client request share it.
fn base_request_id(request_id: &str) -> &str {
    match request_id.rsplit_once('-') {
        Some((base, suffix))
            if suffix.len() == 8
                && suffix
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) =>
        {
            base
        }
        _ => request_id,
    }
}

/// Where a decode engine wants a request's blocks written, as it tells the
/// prefill engine in a `PUSH_REG` message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Registration {
    request_id: String,
    decode_engine_id: String,
    decode_host: String,
    decode_port: i64,
    local_block_ids: Vec<u32>,
}

enum Job {
    Pull {
        request_id: String,
        remote: RemotePool,
        reads: Vec<BlockRead>,
        notify: String,
    },
    Push {
        request_id: String,
        remote: RemotePool,
        writes: Vec<BlockWrite>,
        notify: String,
    },
    Notify {
        remote: RemotePool,
        message: String,
    },
}

enum Done {
    Pull(String, anyhow::Result<u64>),
    Push(String, anyhow::Result<u64>),
}

/// A load the decode side is waiting on, by READ or by a prefill engine's WRITE.
struct Load {
    /// The earliest the load may be reported, however fast the bytes moved.
    ready_at: Instant,
    /// Whether the bytes arrived intact, once that is known.
    succeeded: Option<bool>,
    /// Push only: the blocks a prefill engine is to write, and when to stop waiting.
    awaited: Option<(Vec<(usize, u8)>, Instant)>,
}

type DataPlane = Arc<Mutex<Box<dyn KvDataPlane>>>;

fn lock(plane: &DataPlane) -> MutexGuard<'_, Box<dyn KvDataPlane>> {
    plane
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn run_jobs(plane: DataPlane, jobs: Receiver<Job>, done: Sender<Done>) {
    for job in jobs {
        let finished = match job {
            Job::Pull {
                request_id,
                remote,
                reads,
                notify,
            } => Done::Pull(
                request_id,
                lock(&plane).read_blocks(&remote, &reads, &notify),
            ),
            Job::Push {
                request_id,
                remote,
                writes,
                notify,
            } => Done::Push(
                request_id,
                lock(&plane).push_blocks(&remote, &writes, &notify),
            ),
            Job::Notify { remote, message } => {
                if let Err(error) = lock(&plane).notify(&remote, &message) {
                    debug!(%error, engine_id = remote.engine_id, "notifying a remote engine failed");
                }
                continue;
            }
        };
        if done.send(finished).is_err() {
            return;
        }
    }
}

pub(crate) struct NixlWorker {
    plane: DataPlane,
    mode: TransferMode,
    moves_bytes: bool,
    jobs: Sender<Job>,
    done: Receiver<Done>,
    /// Prefill side: finished requests whose blocks a remote engine may still need,
    /// with the time their lease runs out.
    leases: HashMap<String, Instant>,
    /// Decode side: loads in flight.
    loads: HashMap<String, Load>,
    /// Prefill side, push: blocks of finished requests no decode engine has asked for yet.
    unclaimed_blocks: HashMap<String, Vec<u32>>,
    /// Prefill side, push: decode engines waiting for a request that has not finished.
    registrations: HashMap<String, Registration>,
}

impl NixlWorker {
    pub(crate) fn new(opt: &Opt, engine_index: u32) -> Self {
        let plane = make_data_plane(
            opt.pd_role,
            NixlConfig {
                kv_block_bytes: opt.kv_block_bytes,
                kv_cache_blocks: opt.kv_cache_size as usize,
                engine_id: opt.engine_id.clone(),
                side_channel_host: opt.side_channel_host.clone(),
                side_channel_port: opt.side_channel_port + engine_index,
            },
        );
        let moves_bytes = plane.moves_bytes();
        let plane: DataPlane = Arc::new(Mutex::new(plane));
        let (jobs, jobs_rx) = channel();
        let (done_tx, done) = channel();
        let thread_plane = Arc::clone(&plane);
        if let Err(error) = std::thread::Builder::new()
            .name(format!("kv-transfer-{engine_index}"))
            .spawn(move || run_jobs(thread_plane, jobs_rx, done_tx))
        {
            warn!(%error, "starting the KV transfer thread failed; remote loads will fail");
        }
        NixlWorker {
            plane,
            mode: transfer_mode(opt),
            moves_bytes,
            jobs,
            done,
            leases: HashMap::new(),
            loads: HashMap::new(),
            unclaimed_blocks: HashMap::new(),
            registrations: HashMap::new(),
        }
    }

    pub(crate) fn mode(&self) -> TransferMode {
        self.mode
    }

    /// Prefill side: write a finished request's blocks and keep them for `lease`.
    pub(crate) fn stage(&mut self, request_id: &str, blocks: &[(usize, u8)], lease: Duration) {
        lock(&self.plane).write_blocks(blocks);
        self.leases
            .insert(request_id.to_string(), Instant::now() + lease);
    }

    /// Prefill side: the scheduler stopped tracking this request.
    pub(crate) fn drop_lease(&mut self, request_id: &str) {
        self.leases.remove(request_id);
        self.unclaimed_blocks.remove(request_id);
    }

    /// Decode side: READ a request's blocks from the engine that prefilled it. The
    /// load is reported no earlier than `min_duration` from now.
    pub(crate) fn start_pull(
        &mut self,
        request_id: &str,
        remote: &RemoteMeta,
        reads: Vec<BlockRead>,
        min_duration: Duration,
    ) {
        self.loads.insert(
            request_id.to_string(),
            Load {
                ready_at: Instant::now() + min_duration,
                succeeded: None,
                awaited: None,
            },
        );
        let job = Job::Pull {
            request_id: request_id.to_string(),
            remote: pool(&remote.engine_id, &remote.host, remote.port),
            reads,
            notify: format!("{}:1", remote.request_id),
        };
        if self.jobs.send(job).is_err() {
            self.fail_load(request_id, "the KV transfer thread is not running");
        }
    }

    /// Decode side, push: wait for a prefill engine to WRITE `blocks`. The load is
    /// reported no earlier than `min_duration` from now.
    pub(crate) fn expect_push(
        &mut self,
        request_id: &str,
        blocks: Vec<(usize, u8)>,
        min_duration: Duration,
    ) {
        let now = Instant::now();
        self.loads.insert(
            request_id.to_string(),
            Load {
                ready_at: now + min_duration,
                succeeded: (!self.moves_bytes).then_some(true),
                awaited: Some((blocks, now + DECODER_KV_BLOCKS_TTL)),
            },
        );
    }

    /// Decode side, push: tell the prefill engine where to write a request's blocks.
    pub(crate) fn register_push(&mut self, registration: &PushRegistration) {
        let message = Registration {
            request_id: registration.request_id.clone(),
            decode_engine_id: registration.decode_engine_id.to_string(),
            decode_host: registration.decode_host.to_string(),
            decode_port: registration.decode_port,
            local_block_ids: registration
                .local_block_ids
                .items
                .first()
                .cloned()
                .unwrap_or_default(),
        };
        let Ok(encoded) = serde_json::to_string(&message) else {
            return;
        };
        self.send(
            pool(
                &registration.remote_engine_id,
                &registration.remote_host,
                registration.remote_port,
            ),
            format!("{PUSH_REGISTRATION_PREFIX}{encoded}"),
        );
    }

    /// Prefill side, push: a finished request's blocks are ready to write.
    pub(crate) fn finished_blocks(&mut self, request_id: &str, block_ids: Vec<u32>) {
        let base = base_request_id(request_id);
        let claimed = self
            .registrations
            .keys()
            .find(|registered| base_request_id(registered) == base)
            .cloned();
        match claimed.and_then(|registered| self.registrations.remove(&registered)) {
            Some(registration) => self.start_push(request_id, &block_ids, &registration),
            None => {
                self.unclaimed_blocks
                    .insert(request_id.to_string(), block_ids);
            }
        }
    }

    /// Decode side: a load that failed before it started.
    pub(crate) fn fail_load(&mut self, request_id: &str, reason: &str) {
        warn!(request_id, reason, "remote KV load failed");
        self.loads.insert(
            request_id.to_string(),
            Load {
                ready_at: Instant::now(),
                succeeded: Some(false),
                awaited: None,
            },
        );
    }

    /// Decode side: a load with nothing left to move.
    pub(crate) fn finish_load(&mut self, request_id: &str) {
        self.loads.insert(
            request_id.to_string(),
            Load {
                ready_at: Instant::now(),
                succeeded: Some(true),
                awaited: None,
            },
        );
    }

    /// Decode side: tell the prefill engine its blocks will not be read.
    pub(crate) fn release_remote(&mut self, remote: &RemoteMeta) {
        self.send(
            pool(&remote.engine_id, &remote.host, remote.port),
            format!("{}:1", remote.request_id),
        );
    }

    /// Decode side: keep a prefill engine's leases alive for requests still waiting here.
    pub(crate) fn heartbeat<'a>(
        &mut self,
        engine_id: &str,
        host: &str,
        port: i64,
        request_ids: impl IntoIterator<Item = &'a String>,
    ) {
        let request_ids: Vec<&str> = request_ids.into_iter().map(String::as_str).collect();
        if request_ids.is_empty() {
            return;
        }
        self.send(
            pool(engine_id, host, port),
            format!("{HEARTBEAT_PREFIX}{}", request_ids.join(",")),
        );
    }

    fn send(&mut self, remote: RemotePool, message: String) {
        if self.moves_bytes && self.jobs.send(Job::Notify { remote, message }).is_err() {
            debug!("the KV transfer thread is not running; dropping a message");
        }
    }

    /// Prefill side, push: WRITE the last `registration.local_block_ids.len()` of a
    /// request's blocks into the decode engine's pool.
    fn start_push(&mut self, request_id: &str, block_ids: &[u32], registration: &Registration) {
        let wanted = &registration.local_block_ids;
        let Some(skipped) = block_ids.len().checked_sub(wanted.len()) else {
            warn!(
                request_id,
                "decode allocated more KV blocks than the prefill engine holds"
            );
            return;
        };
        let job = Job::Push {
            request_id: request_id.to_string(),
            remote: pool(
                &registration.decode_engine_id,
                &registration.decode_host,
                registration.decode_port,
            ),
            writes: block_ids[skipped..]
                .iter()
                .zip(wanted)
                .map(|(&local, &remote)| BlockWrite {
                    local_block_id: local as usize,
                    remote_block_id: i64::from(remote),
                })
                .collect(),
            notify: format!("{}:1", registration.request_id),
        };
        if self.jobs.send(job).is_err() {
            warn!(request_id, "the KV transfer thread is not running");
        }
    }

    /// Act on messages from remote engines. Returns the leased requests a decode
    /// engine reports as read.
    fn receive(&mut self, messages: Vec<String>, now: Instant) -> Vec<String> {
        let mut read = Vec::new();
        for message in messages {
            if let Some(request_ids) = message.strip_prefix(HEARTBEAT_PREFIX) {
                for request_id in request_ids.split(',') {
                    if let Some(expires_at) = self.leases.get_mut(request_id) {
                        *expires_at = (*expires_at).max(now + LEASE_EXTENSION);
                    }
                }
            } else if let Some(encoded) = message.strip_prefix(PUSH_REGISTRATION_PREFIX) {
                match serde_json::from_str::<Registration>(encoded) {
                    Ok(registration) => self.registered(registration),
                    Err(error) => warn!(%error, "dropping an unreadable push registration"),
                }
            } else if let Some((request_id, _consumers)) = message.rsplit_once(':') {
                if self.leases.remove(request_id).is_some() {
                    read.push(request_id.to_string());
                }
                if let Some(load) = self.loads.get_mut(request_id)
                    && let Some((blocks, _)) = &load.awaited
                {
                    let checked = lock(&self.plane).check_blocks(blocks);
                    if let Err(error) = &checked {
                        warn!(request_id, %error, "pushed KV failed verification");
                    }
                    load.succeeded = Some(checked.is_ok());
                }
            }
        }
        read
    }

    /// Prefill side, push: a decode engine asked for a request's blocks.
    fn registered(&mut self, registration: Registration) {
        let base = base_request_id(&registration.request_id);
        let finished = self
            .unclaimed_blocks
            .keys()
            .find(|request_id| base_request_id(request_id) == base)
            .cloned();
        match finished.and_then(|request_id| {
            let block_ids = self.unclaimed_blocks.remove(&request_id)?;
            Some((request_id, block_ids))
        }) {
            Some((request_id, block_ids)) => {
                self.start_push(&request_id, &block_ids, &registration);
            }
            None => {
                self.registrations
                    .insert(registration.request_id.clone(), registration);
            }
        }
    }

    /// What finished since the last call, or `None` when nothing did.
    pub(crate) fn poll(&mut self) -> Option<KvConnectorOutput> {
        let now = Instant::now();
        let mut finished_sending = Vec::new();

        while let Ok(done) = self.done.try_recv() {
            match done {
                Done::Pull(request_id, result) => {
                    match &result {
                        Ok(bytes) => info!(request_id, bytes, "loaded remote KV"),
                        Err(error) => warn!(request_id, %error, "remote KV load failed"),
                    }
                    if let Some(load) = self.loads.get_mut(&request_id) {
                        load.succeeded = Some(result.is_ok());
                    }
                }
                Done::Push(request_id, Ok(bytes)) => {
                    info!(request_id, bytes, "pushed KV to the decode engine");
                    if self.leases.remove(&request_id).is_some() {
                        finished_sending.push(request_id);
                    }
                }
                Done::Push(request_id, Err(error)) => {
                    warn!(request_id, %error, "pushing KV to the decode engine failed");
                }
            }
        }

        if self.moves_bytes && (self.mode == TransferMode::Push || !self.leases.is_empty()) {
            let messages = match self.plane.try_lock() {
                Ok(mut plane) => plane.take_notifications(),
                Err(_) => Vec::new(),
            };
            finished_sending.extend(self.receive(messages, now));
        }
        if !self.moves_bytes {
            finished_sending.extend(self.leases.drain().map(|(request_id, _)| request_id));
            self.unclaimed_blocks.clear();
        }
        self.leases.retain(|request_id, &mut expires_at| {
            if expires_at > now {
                return true;
            }
            warn!(
                request_id,
                "KV lease expired before a decode engine took the blocks"
            );
            finished_sending.push(request_id.clone());
            false
        });
        for request_id in &finished_sending {
            self.unclaimed_blocks.remove(request_id);
        }

        let mut finished_recving = Vec::new();
        let mut failed_recving = Vec::new();
        self.loads.retain(|request_id, load| {
            if load.succeeded.is_none()
                && let Some((_, give_up_at)) = &load.awaited
                && *give_up_at <= now
            {
                warn!(request_id, "no prefill engine pushed this request's KV");
                load.succeeded = Some(false);
            }
            match load.succeeded {
                Some(succeeded) if load.ready_at <= now => {
                    finished_recving.push(request_id.clone());
                    if !succeeded {
                        failed_recving.push(request_id.clone());
                    }
                    false
                }
                _ => true,
            }
        });

        if finished_recving.is_empty() && finished_sending.is_empty() {
            return None;
        }
        Some(KvConnectorOutput {
            finished_sending: (!finished_sending.is_empty()).then_some(finished_sending),
            finished_recving: (!finished_recving.is_empty()).then_some(finished_recving),
            failed_recving,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::time::Instant;
    use vllm_engine_core::sched::scheduler::KvConnectorOutput;
    use vllm_engine_core_connectors::nixl::{RemoteBlockIds, RemoteMeta, Seq, SeqKind};

    use crate::Opt;
    use crate::rust_core::nixl::{
        LEASE_EXTENSION, NixlWorker, Registration, base_request_id, remote_block_ids,
    };

    fn worker() -> NixlWorker {
        NixlWorker::new(&Opt::parse_from(["play"]), 0)
    }

    /// A worker whose transfers count as moving bytes, so leases wait for a remote
    /// engine instead of completing at once.
    fn waiting_worker() -> NixlWorker {
        let mut worker = worker();
        worker.moves_bytes = true;
        worker
    }

    fn remote(block_ids: RemoteBlockIds) -> RemoteMeta {
        RemoteMeta {
            block_ids: block_ids.into(),
            host: "127.0.0.1".into(),
            port: 5600,
            engine_id: "prefill-0".into(),
            request_id: "p-1".to_string(),
            blocks_expiry_time: None,
            num_tokens: None,
            registration_generation: 0,
        }
    }

    fn registration(request_id: &str, local_block_ids: Vec<u32>) -> Registration {
        Registration {
            request_id: request_id.to_string(),
            decode_engine_id: "decode-0".to_string(),
            decode_host: "127.0.0.1".to_string(),
            decode_port: 5601,
            local_block_ids,
        }
    }

    /// Poll until the worker reports something, as the engine does every idle step.
    fn poll_until_reported(worker: &mut NixlWorker) -> KvConnectorOutput {
        let give_up = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(output) = worker.poll() {
                return output;
            }
            assert!(std::time::Instant::now() < give_up, "worker never reported");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn remote_block_ids_reads_flat_and_grouped_lists() {
        let flat = RemoteBlockIds::Flat(Seq::new(SeqKind::List, vec![3, 4]));
        assert_eq!(remote_block_ids(&remote(flat)), [3, 4]);
        let grouped = RemoteBlockIds::Grouped(Seq::new(
            SeqKind::List,
            vec![Seq::new(SeqKind::List, vec![5, 6, 7])],
        ));
        assert_eq!(remote_block_ids(&remote(grouped)), [5, 6, 7]);
        assert!(remote_block_ids(&remote(RemoteBlockIds::empty_tuple())).is_empty());
    }

    #[test]
    fn base_request_id_strips_only_an_eight_hex_suffix() {
        assert_eq!(base_request_id("cmpl-abc-0123abcd"), "cmpl-abc");
        assert_eq!(base_request_id("cmpl-abc-0123ABCD"), "cmpl-abc-0123ABCD");
        assert_eq!(base_request_id("cmpl-abc-0123abc"), "cmpl-abc-0123abc");
        assert_eq!(base_request_id("plain"), "plain");
    }

    #[test]
    fn nothing_is_reported_while_idle() {
        assert!(worker().poll().is_none());
    }

    #[test]
    fn a_staged_request_is_sent_at_once_when_no_bytes_move() {
        let mut worker = worker();
        worker.stage("p-1", &[(0, 0xab)], Duration::from_secs(30));
        let output = worker.poll().expect("a report");
        assert_eq!(output.finished_sending, Some(vec!["p-1".to_string()]));
        assert_eq!(output.finished_recving, None);
        assert!(worker.poll().is_none());
    }

    #[test]
    fn a_lease_is_held_until_the_decoder_reports_the_read() {
        let mut worker = waiting_worker();
        worker.stage("p-1", &[], Duration::from_secs(30));
        assert!(worker.poll().is_none());

        let read = worker.receive(vec!["p-1:1".to_string()], Instant::now());
        assert_eq!(read, ["p-1"]);
        assert!(worker.leases.is_empty());
        assert!(
            worker
                .receive(vec!["p-1:1".to_string()], Instant::now())
                .is_empty()
        );
    }

    #[test]
    fn an_unread_lease_is_reported_when_it_expires() {
        let mut worker = waiting_worker();
        worker.stage("p-1", &[], Duration::from_millis(30));
        assert!(worker.poll().is_none());

        let output = poll_until_reported(&mut worker);
        assert_eq!(output.finished_sending, Some(vec!["p-1".to_string()]));
    }

    #[test]
    fn a_heartbeat_extends_only_the_leases_it_names() {
        let mut worker = waiting_worker();
        worker.stage("p-1", &[], Duration::from_secs(1));
        worker.stage("p-2", &[], Duration::from_secs(1));
        let now = Instant::now();

        let read = worker.receive(vec!["HB:p-1,unknown".to_string()], now);
        assert!(read.is_empty());
        assert_eq!(worker.leases["p-1"], now + LEASE_EXTENSION);
        assert!(worker.leases["p-2"] < now + Duration::from_secs(2));
    }

    #[test]
    fn a_heartbeat_never_shortens_a_lease() {
        let mut worker = waiting_worker();
        worker.stage("p-1", &[], Duration::from_secs(60));
        let before = worker.leases["p-1"];
        worker.receive(vec!["HB:p-1".to_string()], Instant::now());
        assert_eq!(worker.leases["p-1"], before);
    }

    #[test]
    fn a_dropped_lease_is_never_reported() {
        let mut worker = waiting_worker();
        worker.stage("p-1", &[], Duration::from_millis(1));
        worker.drop_lease("p-1");
        std::thread::sleep(Duration::from_millis(5));
        assert!(worker.poll().is_none());
    }

    #[test]
    fn a_pull_is_reported_once_and_not_before_its_minimum_duration() {
        let mut worker = worker();
        let started = std::time::Instant::now();
        let source = remote(RemoteBlockIds::empty_tuple());
        worker.start_pull("d-1", &source, Vec::new(), Duration::from_millis(60));

        let output = poll_until_reported(&mut worker);
        assert!(started.elapsed() >= Duration::from_millis(60));
        assert_eq!(output.finished_recving, Some(vec!["d-1".to_string()]));
        assert!(output.failed_recving.is_empty());
        assert!(worker.poll().is_none());
    }

    #[test]
    fn a_failed_load_is_reported_as_finished_and_failed() {
        let mut worker = worker();
        worker.fail_load("d-1", "test");
        let output = worker.poll().expect("a report");
        assert_eq!(output.finished_recving, Some(vec!["d-1".to_string()]));
        assert_eq!(output.failed_recving, ["d-1"]);
    }

    #[test]
    fn an_awaited_push_completes_when_the_prefill_engine_reports_the_write() {
        let mut worker = waiting_worker();
        worker.expect_push("d-1", vec![(3, 0xab)], Duration::ZERO);
        assert!(worker.poll().is_none());

        worker.receive(vec!["d-1:1".to_string()], Instant::now());
        let output = worker.poll().expect("a report");
        assert_eq!(output.finished_recving, Some(vec!["d-1".to_string()]));
        assert!(output.failed_recving.is_empty());
    }

    #[test]
    fn an_awaited_push_completes_at_once_when_no_bytes_move() {
        let mut worker = worker();
        worker.expect_push("d-1", vec![(3, 0xab)], Duration::ZERO);
        let output = worker.poll().expect("a report");
        assert_eq!(output.finished_recving, Some(vec!["d-1".to_string()]));
    }

    #[test]
    fn finished_blocks_are_pushed_to_a_decoder_that_registered_first() {
        let mut worker = waiting_worker();
        worker.stage("cmpl-7-0123abcd", &[], Duration::from_secs(30));
        let encoded = serde_json::to_string(&registration("cmpl-7-89abcdef", vec![4, 5])).unwrap();
        worker.receive(vec![format!("PUSH_REG:{encoded}")], Instant::now());
        assert_eq!(worker.registrations.len(), 1);

        worker.finished_blocks("cmpl-7-0123abcd", vec![10, 11, 12]);
        assert!(worker.registrations.is_empty());
        assert!(worker.unclaimed_blocks.is_empty());
        let output = poll_until_reported(&mut worker);
        assert_eq!(
            output.finished_sending,
            Some(vec!["cmpl-7-0123abcd".to_string()])
        );
    }

    #[test]
    fn a_late_registration_claims_blocks_that_already_finished() {
        let mut worker = waiting_worker();
        worker.stage("cmpl-7-0123abcd", &[], Duration::from_secs(30));
        worker.finished_blocks("cmpl-7-0123abcd", vec![10, 11]);
        assert_eq!(worker.unclaimed_blocks.len(), 1);

        let encoded = serde_json::to_string(&registration("cmpl-7-89abcdef", vec![4, 5])).unwrap();
        worker.receive(vec![format!("PUSH_REG:{encoded}")], Instant::now());
        assert!(worker.unclaimed_blocks.is_empty());
        let output = poll_until_reported(&mut worker);
        assert_eq!(
            output.finished_sending,
            Some(vec!["cmpl-7-0123abcd".to_string()])
        );
    }

    #[test]
    fn a_registration_for_another_request_claims_nothing() {
        let mut worker = waiting_worker();
        worker.finished_blocks("cmpl-7-0123abcd", vec![10, 11]);
        let encoded = serde_json::to_string(&registration("cmpl-8-89abcdef", vec![4])).unwrap();
        worker.receive(vec![format!("PUSH_REG:{encoded}")], Instant::now());
        assert_eq!(worker.unclaimed_blocks.len(), 1);
        assert_eq!(worker.registrations.len(), 1);
    }
}
