//! A diffusion trace recorded from a real stage: FLUX.2-klein-4B behind a
//! vLLM-Omni v0.32.0 head on one H200, 72 image generation and edit requests
//! at concurrency 1 and 4 (deploy/omni/loadgen_images.py).

use std::path::Path;
use std::time::Duration;

use sim_trace::diffusion::{
    DiffusionOutcome, DiffusionReplay, DiffusionShape, ImageSize, read_trace_file, step_units,
    summarize,
};
use sim_trace::trace::{TraceKind, read_trace_file as read_token_trace_file};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/omni_v0.32.0_flux2_klein_h200.jsonl"
);

fn square(side: u32, steps: u32) -> DiffusionShape {
    DiffusionShape {
        size: Some(ImageSize {
            width: side,
            height: side,
        }),
        steps: Some(steps),
        images: 1,
        input_images: 0,
    }
}

#[test]
fn real_stage_trace_parses_with_every_request_finished() {
    let (meta, records) = read_trace_file(Path::new(FIXTURE)).expect("read fixture");

    assert_eq!(meta.kind, TraceKind::Diffusion);
    assert_eq!(
        meta.model.as_deref(),
        Some("black-forest-labs/FLUX.2-klein-4B")
    );
    assert_eq!(meta.extra.get("omni_stage_id"), Some(&serde_json::json!(0)));

    assert_eq!(records.len(), 72);
    assert!(
        records
            .iter()
            .all(|record| record.outcome == DiffusionOutcome::Finished)
    );
    assert!(
        records
            .iter()
            .all(|record| record.output_size == record.shape.size),
        "the stage returned the size each request asked for"
    );
    assert_eq!(
        records.iter().map(|record| record.concurrency).max(),
        Some(4)
    );

    let stats = summarize(&records);
    assert_eq!(stats.len(), 8, "eight request shapes, nine requests each");
    assert!(stats.iter().all(|shape| shape.finished == 9));
    assert_eq!(
        stats
            .iter()
            .filter(|shape| shape.shape.input_images == 1)
            .count(),
        2,
        "edits are their own shapes"
    );
}

#[test]
fn real_stage_trace_splits_a_queued_request_into_wait_slot_and_tail() {
    let (_, records) = read_trace_file(Path::new(FIXTURE)).expect("read fixture");

    let queued: Vec<_> = records
        .iter()
        .filter(|record| record.stage_metrics["scheduler_queue_wait_ms"] > 50.0)
        .collect();
    assert!(!queued.is_empty(), "concurrency 4 queues inside the stage");
    for record in &queued {
        let service = record.service();
        let waited = record.stage_metrics["scheduler_queue_wait_ms"];
        assert!(
            (service.total().as_secs_f64() * 1000.0 + waited - record.latency_ms).abs() < 1.0,
            "wait, slot, and tail add up to the latency at the tap"
        );
        assert!(
            service.tail >= Duration::from_millis(5),
            "post-processing happens after the slot is released"
        );
        assert!(service.slot > service.tail);
    }
}

#[test]
fn real_stage_trace_fits_a_step_model_and_replays_recorded_shapes() {
    let (_, records) = read_trace_file(Path::new(FIXTURE)).expect("read fixture");
    let mut replay = DiffusionReplay::from_records(&records).expect("replay");
    assert_eq!(replay.shapes(), 8);

    let fitted = replay.fitted();
    let overhead_ms = fitted.overhead.as_secs_f64() * 1000.0;
    let step_ms = fitted.step_time.as_secs_f64() * 1000.0;
    assert!(
        (50.0..200.0).contains(&overhead_ms) && (50.0..100.0).contains(&step_ms),
        "the fit is not skewed by the 16 s cold request: {overhead_ms} ms + {step_ms} ms/step"
    );

    let large = fitted.exec_time(step_units(
        ImageSize {
            width: 1024,
            height: 1024,
        },
        8,
        1,
    ));
    let recorded_large = replay
        .next_recorded(&square(1024, 8))
        .expect("recorded shape")
        .service
        .total();
    let error = large.as_secs_f64() / recorded_large.as_secs_f64() - 1.0;
    assert!(
        error.abs() < 0.15,
        "the fit is within 15% of a recorded shape, off by {error}"
    );

    assert_eq!(replay.next_recorded(&square(2048, 4)), None);
}

#[test]
fn token_reader_refuses_the_real_diffusion_trace() {
    let error = read_token_trace_file(Path::new(FIXTURE)).expect_err("not a token trace");
    assert!(format!("{error:#}").contains("diffusion trace"));
}
