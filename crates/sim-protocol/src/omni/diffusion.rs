//! Wire format between a vLLM-Omni head and a diffusion stage.
//!
//! Unlike an LLM stage this is not the engine-core protocol. After the same
//! registration and HELLO/INIT/READY handshake, the head binds a PUSH for
//! requests and a PULL for responses, and both sides exchange single-frame
//! msgpack maps keyed by `type`. Images travel inline as raw `uint8` pixels.

use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use rmpv::Value;
use zeromq::prelude::Socket as _;
use zeromq::{PullSocket, PushSocket};

use crate::frontend_connect::OpenHandshake;
use crate::omni::msgpack::{encode, field, map};
use crate::vllm::engine_id_from_index;

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

/// Run the handshake and connect the data sockets. The head binds them once it
/// has READY, so READY goes first.
pub async fn connect(
    handshake_address: &str,
    connect_timeout: Duration,
) -> Result<DiffusionSockets> {
    let mut handshake = OpenHandshake::open(
        handshake_address,
        engine_id_from_index(0)?,
        false,
        true,
        connect_timeout,
    )
    .await?;
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

/// Pixel dimensions of an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

impl ImageSize {
    pub fn pixels(self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
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

/// A finished request: `count` flat gray images of `size`.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedImages<'a> {
    pub request_id: &'a str,
    pub size: ImageSize,
    pub count: u32,
    pub num_inference_steps: u32,
    /// Reported as `diffusion_engine_exec_time_ms`.
    pub exec_time: Duration,
}

fn gray_image(size: ImageSize) -> Result<Value> {
    let bytes = usize::try_from(size.pixels())
        .ok()
        .and_then(|pixels| pixels.checked_mul(RGB_CHANNELS))
        .context("image byte count overflows usize")?;
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
        ("data", Value::Binary(vec![GRAY; bytes])),
    ]))
}

pub fn encode_result(result: &GeneratedImages<'_>) -> Result<Vec<u8>> {
    let images = (0..result.count)
        .map(|_| gray_image(result.size))
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
            Value::from(result.exec_time.as_secs_f64() * 1000.0),
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rmpv::Value;

    use crate::omni::diffusion::{
        GenerateRequest, GeneratedImages, ImageSize, StageRequest, decode_request, encode_error,
        encode_result, encode_rpc_result,
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
        assert_eq!(
            field(&output, "metrics")
                .and_then(|m| field(m, "diffusion_engine_exec_time_ms"))
                .and_then(Value::as_f64),
            Some(1500.0)
        );
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
