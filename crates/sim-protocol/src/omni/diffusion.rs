//! Wire format between a vLLM-Omni head and a diffusion stage.
//!
//! Unlike an LLM stage this is not the engine-core protocol. After the same
//! registration and HELLO/INIT/READY handshake, the head binds a PUSH for
//! requests and a PULL for responses, and both sides exchange single-frame
//! msgpack maps keyed by `type`. Images travel inline as raw `uint8` pixels.

use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use rmpv::Value;
use sim_trace::diffusion::{DiffusionShape, ImageSize};
use zeromq::prelude::Socket as _;
use zeromq::{PullSocket, PushSocket};

use crate::frontend_connect::OpenHandshake;
use crate::omni::msgpack::{encode, field, map};
use crate::vllm::engine_id_from_index;

/// Metric a stage reports the time a request waited for an execution slot in.
const QUEUE_WAIT_METRIC: &str = sim_trace::diffusion::QUEUE_WAIT_METRIC;

/// Bytes per pixel of the `RGB` images a stage returns.
const RGB_CHANNELS: usize = 3;
const GRAY: u8 = 128;

/// A diffusion stage's data sockets and the addresses the head knows them by.
pub struct DiffusionSockets {
    pub requests: PullSocket,
    pub responses: PushSocket,
    pub input_address: String,
    pub output_address: String,
}

/// Run the handshake and connect the data sockets.
pub async fn connect(
    handshake_address: &str,
    connect_timeout: Duration,
) -> Result<DiffusionSockets> {
    let handshake = OpenHandshake::open(
        handshake_address,
        engine_id_from_index(0)?,
        false,
        true,
        connect_timeout,
    )
    .await?;
    attach(handshake).await
}

/// Report READY on an open handshake and connect the data sockets. The head
/// binds them once it has READY, so READY goes first.
pub async fn attach(mut handshake: OpenHandshake) -> Result<DiffusionSockets> {
    handshake.ready().await?;

    let (Some(input_address), Some(output_address)) = (
        handshake.init.addresses.inputs.first().cloned(),
        handshake.init.addresses.outputs.first().cloned(),
    ) else {
        bail!("handshake INIT carried no data addresses");
    };
    let mut requests = PullSocket::new();
    requests
        .connect(&input_address)
        .await
        .with_context(|| format!("connecting request pull to {input_address}"))?;
    let mut responses = PushSocket::new();
    responses
        .connect(&output_address)
        .await
        .with_context(|| format!("connecting response push to {output_address}"))?;
    Ok(DiffusionSockets {
        requests,
        responses,
        input_address,
        output_address,
    })
}

/// What a stage reads from an `add_request`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateRequest {
    pub request_id: String,
    /// `None` when the client sent no size.
    pub size: Option<ImageSize>,
    pub num_outputs: u32,
    pub num_inference_steps: Option<u32>,
    /// Images under `prompt.multi_modal_data.image`: zero for text-to-image.
    pub input_images: usize,
}

impl GenerateRequest {
    /// The request as a trace describes it.
    pub fn shape(&self) -> DiffusionShape {
        DiffusionShape {
            size: self.size,
            steps: self.num_inference_steps,
            images: self.num_outputs,
            input_images: u32::try_from(self.input_images).unwrap_or(u32::MAX),
        }
    }
}

/// A message from the head.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageRequest {
    Generate(GenerateRequest),
    Abort(Vec<String>),
    Rpc { rpc_id: String, method: String },
    Shutdown,
    Unknown(String),
}

fn string_field(map: &Value, key: &str) -> Result<String> {
    field(map, key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .with_context(|| format!("diffusion request has no string `{key}`"))
}

fn u32_field(map: &Value, key: &str) -> Option<u32> {
    field(map, key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

fn input_images(message: &Value) -> usize {
    let images = field(message, "prompt")
        .and_then(|prompt| field(prompt, "multi_modal_data"))
        .and_then(|data| field(data, "image"));
    match images {
        Some(Value::Array(images)) => images.len(),
        Some(Value::Nil) | None => 0,
        Some(_) => 1,
    }
}

pub fn decode_request(frame: &[u8]) -> Result<StageRequest> {
    let message =
        rmpv::decode::read_value(&mut &frame[..]).context("decoding diffusion request")?;
    let kind = string_field(&message, "type")?;
    Ok(match kind.as_str() {
        "add_request" => {
            let params = field(&message, "sampling_params").unwrap_or(&Value::Nil);
            let size = match (u32_field(params, "width"), u32_field(params, "height")) {
                (Some(width), Some(height)) => Some(ImageSize { width, height }),
                _ => None,
            };
            StageRequest::Generate(GenerateRequest {
                request_id: string_field(&message, "request_id")?,
                size,
                num_outputs: u32_field(params, "num_outputs_per_prompt").unwrap_or(1),
                num_inference_steps: u32_field(params, "num_inference_steps"),
                input_images: input_images(&message),
            })
        }
        "abort" => StageRequest::Abort(
            field(&message, "request_ids")
                .and_then(Value::as_array)
                .map(|ids| {
                    ids.iter()
                        .filter_map(|id| id.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
        ),
        "collective_rpc" => StageRequest::Rpc {
            rpc_id: string_field(&message, "rpc_id")?,
            method: string_field(&message, "method")?,
        },
        "shutdown" => StageRequest::Shutdown,
        _ => StageRequest::Unknown(kind),
    })
}

/// A finished request: `count` images of `size`, each noise in its top half
/// and flat gray below.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedImages<'a> {
    pub request_id: &'a str,
    pub size: ImageSize,
    pub count: u32,
    pub num_inference_steps: u32,
    /// Time the request waited for an execution slot, reported as
    /// `scheduler_queue_wait_ms`.
    pub queue_wait: Duration,
    /// Time the request executed. Reported, added to the wait, as
    /// `diffusion_engine_exec_time_ms`.
    pub exec_time: Duration,
}

/// `len` bytes of SplitMix64 output, the same for the same `seed`.
fn noise(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut bytes = vec![0; len];
    for chunk in bytes.chunks_mut(8) {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut word = state;
        word = (word ^ (word >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        word = (word ^ (word >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        word ^= word >> 31;
        chunk.copy_from_slice(&word.to_le_bytes()[..chunk.len()]);
    }
    bytes
}

/// The pixels of a mock image: noise in the top half of its rows, flat gray
/// in the rest.
fn mock_pixels(size: ImageSize, seed: u64) -> Result<Vec<u8>> {
    let row_bytes = usize::try_from(size.width)
        .ok()
        .and_then(|width| width.checked_mul(RGB_CHANNELS));
    let rows = usize::try_from(size.height).ok();
    let (Some(row_bytes), Some(rows)) = (row_bytes, rows) else {
        bail!("image byte count overflows usize");
    };
    let bytes = row_bytes
        .checked_mul(rows)
        .context("image byte count overflows usize")?;
    let mut pixels = noise(row_bytes * rows.div_ceil(2), seed);
    pixels.resize(bytes, GRAY);
    Ok(pixels)
}

fn mock_image(size: ImageSize, seed: u64) -> Result<Value> {
    let pixels = mock_pixels(size, seed)?;
    Ok(map(vec![
        ("__pil_image__", Value::from(true)),
        ("mode", Value::from("RGB")),
        (
            "shape",
            Value::Array(vec![
                Value::from(size.height),
                Value::from(size.width),
                Value::from(RGB_CHANNELS as u64),
            ]),
        ),
        ("data", Value::Binary(pixels)),
    ]))
}

pub fn encode_result(result: &GeneratedImages<'_>) -> Result<Vec<u8>> {
    let images = (0..result.count)
        .map(|index| mock_image(result.size, u64::from(index)))
        .collect::<Result<Vec<_>>>()?;
    let metrics = map(vec![
        ("image_num", Value::from(result.count)),
        ("width", Value::from(result.size.width)),
        ("height", Value::from(result.size.height)),
        (
            "num_inference_steps",
            Value::from(result.num_inference_steps),
        ),
        (
            "diffusion_engine_exec_time_ms",
            Value::from((result.queue_wait + result.exec_time).as_secs_f64() * 1000.0),
        ),
        (
            QUEUE_WAIT_METRIC,
            Value::from(result.queue_wait.as_secs_f64() * 1000.0),
        ),
    ]);
    encode(&map(vec![
        ("type", Value::from("result")),
        (
            "output",
            map(vec![
                ("request_id", Value::from(result.request_id)),
                ("finished", Value::from(true)),
                ("final_output_type", Value::from("image")),
                ("images", Value::Array(images)),
                ("metrics", metrics),
            ]),
        ),
    ]))
}

/// A request error. A `status_code` of 500 or above counts against the head's
/// per-stage circuit breaker, which opens for good after two in a row.
pub fn encode_error(request_id: &str, error: &str, status_code: u16) -> Result<Vec<u8>> {
    encode(&map(vec![
        ("type", Value::from("error")),
        ("request_id", Value::from(request_id)),
        ("error", Value::from(error)),
        ("status_code", Value::from(status_code)),
        ("error_type", Value::Nil),
        ("metrics", Value::Map(Vec::new())),
    ]))
}

pub fn encode_rpc_result(rpc_id: &str) -> Result<Vec<u8>> {
    encode(&map(vec![
        ("type", Value::from("rpc_result")),
        ("rpc_id", Value::from(rpc_id)),
        ("result", Value::Nil),
    ]))
}

/// The frame a stage sends, raw and not msgpack, when it dies with requests
/// outstanding.
pub const PROC_DEAD: &[u8] = b"DIFFUSION_PROC_DEAD";

/// A request the stage answered with images.
#[derive(Debug, Clone, PartialEq)]
pub struct FinishedRequest {
    pub request_id: String,
    /// Size of the first image returned. `None` when the output has no image.
    pub output_size: Option<ImageSize>,
    /// The numeric entries of the output's `metrics`.
    pub metrics: BTreeMap<String, f64>,
}

/// A request the stage answered with an error.
#[derive(Debug, Clone, PartialEq)]
pub struct FailedRequest {
    pub request_id: String,
    pub status_code: Option<u16>,
    /// The numeric entries of the error's `metrics`.
    pub metrics: BTreeMap<String, f64>,
}

/// A message from a stage, as far as a recorder reads it.
#[derive(Debug, Clone, PartialEq)]
pub enum StageResponse {
    /// A `result` whose output is finished.
    Finished(FinishedRequest),
    /// A `result` for a request that is still running: a lifecycle marker or
    /// a streamed chunk.
    Progress { request_id: String },
    /// An `error` that names a request.
    Failed(FailedRequest),
    /// [`PROC_DEAD`].
    Dead,
    /// A message that ends no request: `metrics`, `rpc_result`, an `error`
    /// for an rpc, or a type this build does not know.
    Other(String),
}

fn numeric_metrics(metrics: Option<&Value>) -> BTreeMap<String, f64> {
    metrics
        .and_then(Value::as_map)
        .map(|entries| {
            entries
                .iter()
                .filter_map(|(key, value)| Some((key.as_str()?.to_string(), value.as_f64()?)))
                .collect()
        })
        .unwrap_or_default()
}

fn first_image_size(output: &Value) -> Option<ImageSize> {
    let shape = field(output, "images")
        .and_then(Value::as_array)?
        .first()
        .and_then(|image| field(image, "shape"))
        .and_then(Value::as_array)?;
    let dimension = |index: usize| {
        shape
            .get(index)
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
    };
    Some(ImageSize {
        height: dimension(0)?,
        width: dimension(1)?,
    })
}

pub fn decode_response(frame: &[u8]) -> Result<StageResponse> {
    if frame == PROC_DEAD {
        return Ok(StageResponse::Dead);
    }
    let message =
        rmpv::decode::read_value(&mut &frame[..]).context("decoding diffusion response")?;
    let kind = string_field(&message, "type")?;
    Ok(match kind.as_str() {
        "result" => {
            let output = field(&message, "output").context("diffusion result has no `output`")?;
            let request_id = string_field(output, "request_id")?;
            if field(output, "finished").and_then(Value::as_bool) == Some(false) {
                StageResponse::Progress { request_id }
            } else {
                StageResponse::Finished(FinishedRequest {
                    request_id,
                    output_size: first_image_size(output),
                    metrics: numeric_metrics(field(output, "metrics")),
                })
            }
        }
        "error" => match field(&message, "request_id").and_then(Value::as_str) {
            Some(request_id) => StageResponse::Failed(FailedRequest {
                request_id: request_id.to_string(),
                status_code: field(&message, "status_code")
                    .and_then(Value::as_u64)
                    .and_then(|status| u16::try_from(status).ok()),
                metrics: numeric_metrics(field(&message, "metrics")),
            }),
            None => StageResponse::Other(kind),
        },
        _ => StageResponse::Other(kind),
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use std::collections::BTreeMap;

    use rmpv::Value;
    use sim_trace::diffusion::{DiffusionShape, ImageSize};

    use crate::omni::diffusion::{
        FailedRequest, FinishedRequest, GenerateRequest, GeneratedImages, PROC_DEAD, StageRequest,
        StageResponse, decode_request, decode_response, encode_error, encode_result,
        encode_rpc_result,
    };
    use crate::omni::msgpack::{encode, field, map};

    fn frame(message: &Value) -> Vec<u8> {
        encode(message).expect("encode")
    }

    fn pil_image(height: u32, width: u32) -> Value {
        map(vec![
            ("__pil_image__", Value::from(true)),
            ("mode", Value::from("RGB")),
            (
                "shape",
                Value::Array(vec![
                    Value::from(height),
                    Value::from(width),
                    Value::from(3),
                ]),
            ),
            (
                "data",
                Value::Binary(vec![0; (height * width * 3) as usize]),
            ),
        ])
    }

    fn add_request(prompt: Value, sampling_params: Value) -> Vec<u8> {
        frame(&map(vec![
            ("type", Value::from("add_request")),
            ("request_id", Value::from("img_gen-abc-12345678")),
            ("prompt", prompt),
            ("sampling_params", sampling_params),
            ("kv_sender_info", Value::Nil),
            ("kv_transfer_params", Value::Nil),
            ("payload_sender_info", Value::Nil),
        ]))
    }

    fn decoded(frame: &[u8]) -> GenerateRequest {
        match decode_request(frame).expect("decode") {
            StageRequest::Generate(request) => request,
            other => panic!("expected a generate request, got {other:?}"),
        }
    }

    #[test]
    fn text_to_image_request_decodes() {
        let request = decoded(&add_request(
            map(vec![
                ("prompt", Value::from("a red fox")),
                ("modalities", Value::Array(vec![Value::from("image")])),
            ]),
            map(vec![
                ("num_outputs_per_prompt", Value::from(2)),
                ("width", Value::from(768)),
                ("height", Value::from(512)),
                ("num_inference_steps", Value::from(30)),
                ("guidance_scale", Value::Nil),
            ]),
        ));
        assert_eq!(
            request,
            GenerateRequest {
                request_id: "img_gen-abc-12345678".to_string(),
                size: Some(ImageSize {
                    width: 768,
                    height: 512
                }),
                num_outputs: 2,
                num_inference_steps: Some(30),
                input_images: 0,
            }
        );
    }

    #[test]
    fn request_without_size_or_steps_leaves_them_unset() {
        let request = decoded(&add_request(
            map(vec![("prompt", Value::from("a red fox"))]),
            map(vec![
                ("num_outputs_per_prompt", Value::from(1)),
                ("width", Value::Nil),
                ("height", Value::Nil),
                ("num_inference_steps", Value::Nil),
            ]),
        ));
        assert_eq!(request.size, None);
        assert_eq!(request.num_inference_steps, None);
        assert_eq!(request.num_outputs, 1);
    }

    #[test]
    fn a_lone_dimension_is_not_a_size() {
        let request = decoded(&add_request(
            Value::Nil,
            map(vec![("width", Value::from(512)), ("height", Value::Nil)]),
        ));
        assert_eq!(request.size, None);
    }

    #[test]
    fn edit_request_counts_its_input_images() {
        let prompt = map(vec![
            ("prompt", Value::from("make it night")),
            (
                "multi_modal_data",
                map(vec![
                    (
                        "image",
                        Value::Array(vec![pil_image(4, 6), pil_image(4, 6)]),
                    ),
                    ("mask_image", pil_image(4, 6)),
                ]),
            ),
        ]);
        let request = decoded(&add_request(
            prompt,
            map(vec![("width", Value::from(6)), ("height", Value::from(4))]),
        ));
        assert_eq!(request.input_images, 2);
        assert_eq!(
            request.size,
            Some(ImageSize {
                width: 6,
                height: 4
            })
        );
    }

    #[test]
    fn a_single_input_image_map_counts_as_one() {
        let prompt = map(vec![(
            "multi_modal_data",
            map(vec![("image", pil_image(2, 2))]),
        )]);
        assert_eq!(decoded(&add_request(prompt, Value::Nil)).input_images, 1);
    }

    #[test]
    fn control_messages_decode() {
        let abort = frame(&map(vec![
            ("type", Value::from("abort")),
            (
                "request_ids",
                Value::Array(vec![Value::from("a"), Value::from("b")]),
            ),
        ]));
        assert_eq!(
            decode_request(&abort).expect("abort"),
            StageRequest::Abort(vec!["a".to_string(), "b".to_string()])
        );

        let rpc = frame(&map(vec![
            ("type", Value::from("collective_rpc")),
            ("rpc_id", Value::from("0123")),
            ("method", Value::from("profile")),
            ("timeout", Value::Nil),
            ("args", Value::Array(Vec::new())),
            ("kwargs", Value::Map(Vec::new())),
        ]));
        assert_eq!(
            decode_request(&rpc).expect("rpc"),
            StageRequest::Rpc {
                rpc_id: "0123".to_string(),
                method: "profile".to_string()
            }
        );

        let shutdown = frame(&map(vec![("type", Value::from("shutdown"))]));
        assert_eq!(
            decode_request(&shutdown).expect("shutdown"),
            StageRequest::Shutdown
        );

        let unknown = frame(&map(vec![("type", Value::from("future_thing"))]));
        assert_eq!(
            decode_request(&unknown).expect("unknown"),
            StageRequest::Unknown("future_thing".to_string())
        );
    }

    /// A frame vLLM-Omni v0.32.0's own encoder produced
    /// (deploy/omni/encode_groundtruth.py).
    fn groundtruth(hex: &str) -> Vec<u8> {
        let hex = hex.trim();
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex"))
            .collect()
    }

    #[test]
    fn groundtruth_text_to_image_request_decodes() {
        let frame = groundtruth(include_str!(
            "../../tests/fixtures/omni_v0.32.0_diffusion_t2i.hex"
        ));
        assert_eq!(
            decoded(&frame),
            GenerateRequest {
                request_id: "img_gen-0123456789abcdef-01234567".to_string(),
                size: Some(ImageSize {
                    width: 768,
                    height: 512
                }),
                num_outputs: 2,
                num_inference_steps: Some(30),
                input_images: 0,
            }
        );
    }

    #[test]
    fn groundtruth_prompt_only_request_decodes() {
        let frame = groundtruth(include_str!(
            "../../tests/fixtures/omni_v0.32.0_diffusion_bare.hex"
        ));
        assert_eq!(
            decoded(&frame),
            GenerateRequest {
                request_id: "img_gen-fedcba9876543210-76543210".to_string(),
                size: None,
                num_outputs: 1,
                num_inference_steps: None,
                input_images: 0,
            }
        );
    }

    #[test]
    fn groundtruth_edit_request_decodes() {
        let frame = groundtruth(include_str!(
            "../../tests/fixtures/omni_v0.32.0_diffusion_edit.hex"
        ));
        assert_eq!(
            decoded(&frame),
            GenerateRequest {
                request_id: "img_edit-0123456789abcdef-89abcdef".to_string(),
                size: Some(ImageSize {
                    width: 6,
                    height: 4
                }),
                num_outputs: 1,
                num_inference_steps: None,
                input_images: 2,
            }
        );
    }

    #[test]
    fn groundtruth_abort_decodes() {
        let frame = groundtruth(include_str!(
            "../../tests/fixtures/omni_v0.32.0_diffusion_abort.hex"
        ));
        assert_eq!(
            decode_request(&frame).expect("abort"),
            StageRequest::Abort(vec!["img_gen-a".to_string(), "img_gen-b".to_string()])
        );
    }

    #[test]
    fn malformed_requests_are_errors() {
        assert!(decode_request(&[0xc1]).is_err());
        assert!(decode_request(&frame(&Value::from("add_request"))).is_err());
        assert!(decode_request(&frame(&map(vec![("type", Value::from("add_request"))]))).is_err());
    }

    fn result_output(count: u32, size: ImageSize) -> Value {
        let encoded = encode_result(&GeneratedImages {
            request_id: "img_gen-abc-12345678",
            size,
            count,
            num_inference_steps: 28,
            queue_wait: Duration::from_millis(250),
            exec_time: Duration::from_millis(1500),
        })
        .expect("encode");
        let message = rmpv::decode::read_value(&mut &encoded[..]).expect("decode");
        assert_eq!(
            field(&message, "type").and_then(Value::as_str),
            Some("result")
        );
        assert!(
            field(&message, "finished").is_none(),
            "`finished` at the top level makes the head decode the whole message as an output"
        );
        field(&message, "output").expect("output").clone()
    }

    #[test]
    fn result_carries_what_the_head_rebuilds_an_output_from() {
        let size = ImageSize {
            width: 6,
            height: 4,
        };
        let output = result_output(2, size);

        assert_eq!(
            field(&output, "request_id").and_then(Value::as_str),
            Some("img_gen-abc-12345678")
        );
        assert_eq!(field(&output, "finished"), Some(&Value::from(true)));
        assert_eq!(
            field(&output, "final_output_type").and_then(Value::as_str),
            Some("image")
        );

        let images = field(&output, "images")
            .and_then(Value::as_array)
            .expect("images");
        assert_eq!(images.len(), 2);
        for image in images {
            assert_eq!(field(image, "__pil_image__"), Some(&Value::from(true)));
            assert_eq!(field(image, "mode").and_then(Value::as_str), Some("RGB"));
            assert_eq!(
                field(image, "shape"),
                Some(&Value::Array(vec![
                    Value::from(4),
                    Value::from(6),
                    Value::from(3)
                ])),
                "shape is [height, width, channels]"
            );
            let Some(Value::Binary(data)) = field(image, "data") else {
                panic!("image data must be msgpack bin");
            };
            assert_eq!(data.len(), 4 * 6 * 3);
        }
    }

    fn image_data(output: &Value) -> Vec<&Vec<u8>> {
        field(output, "images")
            .and_then(Value::as_array)
            .expect("images")
            .iter()
            .map(|image| match field(image, "data") {
                Some(Value::Binary(data)) => data,
                other => panic!("image data must be msgpack bin, got {other:?}"),
            })
            .collect()
    }

    #[test]
    fn result_images_are_noise_in_the_top_half_and_flat_below() {
        let size = ImageSize {
            width: 64,
            height: 96,
        };
        let output = result_output(2, size);
        let images = image_data(&output);

        for data in &images {
            let (top, bottom) = data.split_at(data.len() / 2);
            let mut seen = [false; 256];
            for byte in top {
                seen[usize::from(*byte)] = true;
            }
            assert!(
                seen.iter().all(|seen| *seen),
                "9216 noise bytes take every byte value"
            );
            let mean = top.iter().map(|byte| f64::from(*byte)).sum::<f64>() / top.len() as f64;
            assert!((mean - 127.5).abs() < 5.0, "mean noise byte {mean}");
            assert!(bottom.iter().all(|byte| *byte == 128));
        }
        assert_ne!(images[0], images[1], "each image of a result differs");

        let again = result_output(2, size);
        assert_eq!(
            image_data(&again),
            images,
            "the same request returns the same bytes"
        );
    }

    #[test]
    fn an_odd_row_count_rounds_the_noise_up() {
        let output = result_output(
            1,
            ImageSize {
                width: 3,
                height: 3,
            },
        );
        let images = image_data(&output);
        assert_eq!(images[0].len(), 27);
        assert!(
            images[0][..18].iter().any(|byte| *byte != 128),
            "two of three rows are noise"
        );
        assert_ne!(images[0][17], 0, "the last partial noise word is filled");
        assert!(images[0][18..].iter().all(|byte| *byte == 128));
    }

    #[test]
    fn result_metrics_are_all_numeric() {
        let output = result_output(
            1,
            ImageSize {
                width: 2,
                height: 2,
            },
        );
        let metrics = field(&output, "metrics")
            .and_then(Value::as_map)
            .expect("metrics");
        assert!(!metrics.is_empty());
        for (key, value) in metrics {
            assert!(
                value.is_number(),
                "metric {key} must be numeric, the head sums them"
            );
        }
        let metric = |name: &str| {
            field(&output, "metrics")
                .and_then(|metrics| field(metrics, name))
                .and_then(Value::as_f64)
        };
        assert_eq!(metric("scheduler_queue_wait_ms"), Some(250.0));
        assert_eq!(
            metric("diffusion_engine_exec_time_ms"),
            Some(1750.0),
            "a stage's exec time runs from admission, so it includes the wait"
        );
    }

    #[test]
    fn request_shape_is_the_request_as_sent() {
        let request = GenerateRequest {
            request_id: "img_edit-1".to_string(),
            size: Some(ImageSize {
                width: 768,
                height: 512,
            }),
            num_outputs: 2,
            num_inference_steps: None,
            input_images: 3,
        };
        assert_eq!(
            request.shape(),
            DiffusionShape {
                size: Some(ImageSize {
                    width: 768,
                    height: 512
                }),
                steps: None,
                images: 2,
                input_images: 3,
            }
        );
    }

    #[test]
    fn own_result_decodes_as_finished_with_its_size_and_metrics() {
        let frame = encode_result(&GeneratedImages {
            request_id: "img_gen-1",
            size: ImageSize {
                width: 6,
                height: 4,
            },
            count: 2,
            num_inference_steps: 28,
            queue_wait: Duration::ZERO,
            exec_time: Duration::from_millis(40),
        })
        .expect("encode");
        let StageResponse::Finished(finished) = decode_response(&frame).expect("decode") else {
            panic!("expected a finished request");
        };
        assert_eq!(finished.request_id, "img_gen-1");
        assert_eq!(
            finished.output_size,
            Some(ImageSize {
                width: 6,
                height: 4
            })
        );
        assert_eq!(
            finished.metrics.get("diffusion_engine_exec_time_ms"),
            Some(&40.0)
        );
        assert_eq!(finished.metrics.get("num_inference_steps"), Some(&28.0));
    }

    #[test]
    fn own_error_decodes_as_failed() {
        let frame = encode_error("img_gen-1", "too large", 400).expect("encode");
        assert_eq!(
            decode_response(&frame).expect("decode"),
            StageResponse::Failed(FailedRequest {
                request_id: "img_gen-1".to_string(),
                status_code: Some(400),
                metrics: BTreeMap::new(),
            })
        );
    }

    #[test]
    fn groundtruth_result_decodes_as_finished() {
        let frame = groundtruth(include_str!(
            "../../tests/fixtures/omni_v0.32.0_diffusion_result.hex"
        ));
        assert_eq!(
            decode_response(&frame).expect("decode"),
            StageResponse::Finished(FinishedRequest {
                request_id: "img_gen-0123456789abcdef-01234567".to_string(),
                output_size: Some(ImageSize {
                    width: 6,
                    height: 4
                }),
                metrics: BTreeMap::from([
                    ("preprocess_time_ms".to_string(), 0.25),
                    ("diffusion_engine_exec_time_ms".to_string(), 1500.5),
                    ("output_ready_wait_time_ms".to_string(), 0.0),
                    ("postprocess_time_ms".to_string(), 12.75),
                    ("scheduler_num_waiting_reqs".to_string(), 3.0),
                    ("scheduler_queue_wait_ms".to_string(), 420.0),
                    ("vae_decode_time_ms".to_string(), 30.5),
                    ("forward_time_ms".to_string(), 1040.0),
                ]),
            })
        );
    }

    #[test]
    fn groundtruth_lifecycle_marker_does_not_finish_the_request() {
        let frame = groundtruth(include_str!(
            "../../tests/fixtures/omni_v0.32.0_diffusion_result_started.hex"
        ));
        assert_eq!(
            decode_response(&frame).expect("decode"),
            StageResponse::Progress {
                request_id: "img_gen-0123456789abcdef-01234567".to_string()
            }
        );
    }

    #[test]
    fn groundtruth_error_decodes_as_failed() {
        let frame = groundtruth(include_str!(
            "../../tests/fixtures/omni_v0.32.0_diffusion_error.hex"
        ));
        assert_eq!(
            decode_response(&frame).expect("decode"),
            StageResponse::Failed(FailedRequest {
                request_id: "img_gen-fedcba9876543210-76543210".to_string(),
                status_code: Some(400),
                metrics: BTreeMap::from([("scheduler_num_waiting_reqs".to_string(), 1.0)]),
            })
        );
    }

    #[test]
    fn responses_that_end_no_request_are_other() {
        let abort_metrics = groundtruth(include_str!(
            "../../tests/fixtures/omni_v0.32.0_diffusion_abort_metrics.hex"
        ));
        assert_eq!(
            decode_response(&abort_metrics).expect("decode"),
            StageResponse::Other("metrics".to_string())
        );

        let rpc_result = encode_rpc_result("0123").expect("encode");
        assert_eq!(
            decode_response(&rpc_result).expect("decode"),
            StageResponse::Other("rpc_result".to_string())
        );

        let rpc_error = frame(&map(vec![
            ("type", Value::from("error")),
            ("rpc_id", Value::from("0123")),
            ("error", Value::from("Too many pending collective RPCs")),
        ]));
        assert_eq!(
            decode_response(&rpc_error).expect("decode"),
            StageResponse::Other("error".to_string())
        );
    }

    #[test]
    fn dead_sentinel_and_malformed_responses() {
        assert_eq!(
            decode_response(PROC_DEAD).expect("sentinel"),
            StageResponse::Dead
        );
        assert!(decode_response(&[0xc1]).is_err());
        assert!(decode_response(&frame(&map(vec![("type", Value::from("result"))]))).is_err());
    }

    #[test]
    fn error_and_rpc_result_shapes() {
        let error = encode_error("img_gen-1", "too large", 400).expect("encode");
        let error = rmpv::decode::read_value(&mut &error[..]).expect("decode");
        assert_eq!(field(&error, "type").and_then(Value::as_str), Some("error"));
        assert_eq!(
            field(&error, "request_id").and_then(Value::as_str),
            Some("img_gen-1")
        );
        assert_eq!(
            field(&error, "error").and_then(Value::as_str),
            Some("too large")
        );
        assert_eq!(field(&error, "status_code"), Some(&Value::from(400)));
        assert!(field(&error, "metrics").is_some_and(Value::is_map));

        let rpc = encode_rpc_result("0123").expect("encode");
        let rpc = rmpv::decode::read_value(&mut &rpc[..]).expect("decode");
        assert_eq!(
            field(&rpc, "type").and_then(Value::as_str),
            Some("rpc_result")
        );
        assert_eq!(field(&rpc, "rpc_id").and_then(Value::as_str), Some("0123"));
        assert_eq!(field(&rpc, "result"), Some(&Value::Nil));
    }
}
