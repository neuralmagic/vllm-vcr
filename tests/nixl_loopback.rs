//! The first real NIXL transfer: a prefill agent fills + advertises patterned KV slots in
//! its registered pool and serves its metadata over a listener; a separate decode agent
//! fetches that metadata by host:port, posts a multi-descriptor NIXL READ (one descriptor
//! per paged block), and verifies the bytes. Two distinct agents in one process (same shape
//! as the cross-pod path, just over loopback).
//!
//! Needs real libnixl + UCX, so it runs on Linux and skips under the stub bindings.
//! Run it: `cargo test --features nixl` on a box with NIXL installed.

#![cfg(feature = "nixl")]

use vllm_vcr::dataplane::{
    BlockRead, BlockWrite, NixlConfig, PdRole, RemotePool, RequestKv, block_pattern,
    make_data_plane, nixl_is_stub,
};

fn cfg(engine_id: &str, port: u32) -> NixlConfig {
    NixlConfig {
        kv_block_bytes: 4096,
        kv_cache_blocks: 16,
        engine_id: engine_id.to_string(),
        side_channel_host: "127.0.0.1".to_string(),
        side_channel_port: port,
    }
}

#[test]
fn loopback_paged_dram_transfer() {
    if nixl_is_stub() {
        eprintln!("skipping NIXL loopback: built against stub bindings (no real libnixl)");
        return;
    }

    // Prefill: listens on 5600 and serves its pool. Decode: a distinct agent that pulls.
    let mut prefill = make_data_plane(PdRole::Prefill, cfg("mock-prefill", 5600));
    let mut decode = make_data_plane(PdRole::Decode, cfg("mock-decode", 5601));

    // Three non-contiguous slots, to prove the paging maps block_id -> addr (not a single
    // contiguous span): blocks 2, 5, 9 in the prefill's pool.
    let prefill_slots = [2usize, 5, 9];
    let prefill_kv = RequestKv {
        request_id: "req-loopback-1",
        num_tokens: 40, // 40 tokens / 16 -> 3 blocks
        block_ids: &prefill_slots,
    };

    let remote = prefill
        .advertise_prefilled(prefill_kv)
        .expect("prefill should fill + advertise paged KV");
    assert_eq!(remote.engine_id, "mock-prefill");
    assert_eq!(remote.block_ids, vec![2, 5, 9]);
    assert_eq!(remote.request_id, "req-loopback-1");
    // No mock-extension addressing in the advertised params: the decode learns the pool base
    // from the metadata side channel (PoolDescriptor) during the pull below.

    // The decode lands the blocks into its own (different) pool slots.
    let decode_slots = [0usize, 1, 2];
    let decode_kv = RequestKv {
        request_id: "req-loopback-1",
        num_tokens: 40,
        block_ids: &decode_slots,
    };

    let bytes = decode
        .pull_prefilled(decode_kv, &remote)
        .expect("decode should fetch md, multi-desc READ over NIXL, and verify the pattern");
    assert_eq!(
        bytes,
        3 * 4096,
        "expected all three paged blocks to transfer"
    );
}

/// The path the scheduler-backed engine drives: the prefill writes content-derived
/// bytes per block, the decode reads and verifies them block by block, and the
/// prefill hears which request was read.
#[test]
fn loopback_read_verifies_each_block_and_notifies_the_prefill() {
    if nixl_is_stub() {
        eprintln!("skipping NIXL loopback: built against stub bindings (no real libnixl)");
        return;
    }

    let mut prefill = make_data_plane(PdRole::Prefill, cfg("mock-prefill-b", 5610));
    let mut decode = make_data_plane(PdRole::Decode, cfg("mock-decode-b", 5611));
    assert!(prefill.moves_bytes());

    let patterns = [block_pattern(&[1; 16]), block_pattern(&[2; 16])];
    prefill.write_blocks(&[(4, patterns[0]), (11, patterns[1])]);

    let pool = RemotePool {
        engine_id: "mock-prefill-b".to_string(),
        host: "127.0.0.1".to_string(),
        port: 5610,
    };
    let reads = [
        BlockRead {
            remote_block_id: 4,
            local_block_id: 0,
            expected: patterns[0],
        },
        BlockRead {
            remote_block_id: 11,
            local_block_id: 1,
            expected: patterns[1],
        },
    ];
    let bytes = decode
        .read_blocks(&pool, &reads, "req-loopback-2")
        .expect("decode should read and verify both blocks");
    assert_eq!(bytes, 2 * 4096);

    let swapped = [
        BlockRead {
            expected: patterns[1],
            ..reads[0]
        },
        reads[1],
    ];
    assert!(
        decode
            .read_blocks(&pool, &swapped, "req-loopback-3")
            .is_err(),
        "a block carrying the wrong bytes must fail verification"
    );

    let give_up = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut notified = Vec::new();
    while !notified.contains(&"req-loopback-2".to_string()) {
        assert!(
            std::time::Instant::now() < give_up,
            "prefill was never notified"
        );
        notified.extend(prefill.take_notifications());
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

/// Push mode: the prefill writes its blocks straight into the decode's pool, the
/// decode hears which request landed, and the bytes check out block by block.
#[test]
fn loopback_push_writes_into_the_decode_pool_and_notifies_it() {
    if nixl_is_stub() {
        eprintln!("skipping NIXL loopback: built against stub bindings (no real libnixl)");
        return;
    }

    let mut prefill = make_data_plane(PdRole::Prefill, cfg("mock-prefill-c", 5620));
    let mut decode = make_data_plane(PdRole::Decode, cfg("mock-decode-c", 5621));

    let patterns = [block_pattern(&[3; 16]), block_pattern(&[4; 16])];
    prefill.write_blocks(&[(6, patterns[0]), (7, patterns[1])]);
    assert!(
        decode
            .check_blocks(&[(1, patterns[0]), (2, patterns[1])])
            .is_err(),
        "the decode pool must not hold the bytes before the push"
    );

    let pool = RemotePool {
        engine_id: "mock-decode-c".to_string(),
        host: "127.0.0.1".to_string(),
        port: 5621,
    };
    let writes = [
        BlockWrite {
            local_block_id: 6,
            remote_block_id: 1,
        },
        BlockWrite {
            local_block_id: 7,
            remote_block_id: 2,
        },
    ];
    let bytes = prefill
        .push_blocks(&pool, &writes, "req-loopback-4:1")
        .expect("prefill should write both blocks into the decode pool");
    assert_eq!(bytes, 2 * 4096);

    let give_up = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut notified = Vec::new();
    while !notified.contains(&"req-loopback-4:1".to_string()) {
        assert!(
            std::time::Instant::now() < give_up,
            "decode was never notified"
        );
        notified.extend(decode.take_notifications());
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    decode
        .check_blocks(&[(1, patterns[0]), (2, patterns[1])])
        .expect("the pushed blocks carry the prefill's bytes");
}
