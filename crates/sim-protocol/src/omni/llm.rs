//! The fields a vLLM-Omni LLM stage carries past vLLM's engine-core tuples.
//!
//! An LLM stage speaks the engine-core protocol with two extensions.
//! `OmniEngineCoreRequest` appends fields to the `EngineCoreRequest` tuple,
//! which [`OmniRequestDecoder`] splits off. `OmniEngineCoreOutput` appends
//! fields to the `EngineCoreOutput` tuple, the first being `multimodal_output`,
//! which [`OmniOutputEncoder`] fills on a pipeline's final audio stage.

use anyhow::{Context as _, Result, anyhow, bail};

use crate::omni::msgpack::{encode, map};
use crate::vllm::{EngineCoreOutput, EngineCoreRequest, Envelope, decode_msgpack, encode_msgpack};

/// An `Add` request split into the vLLM request and the omni tail.
#[derive(Debug)]
pub struct OmniRequest {
    pub request: Box<EngineCoreRequest>,
    /// Fields past the end of the `EngineCoreRequest` tuple, in wire order:
    /// `additional_information`, `model_intermediate_buffer`,
    /// `payload_sender_info`.
    pub tail: Vec<rmpv::Value>,
}

/// Decodes `OmniEngineCoreRequest` tuples against this build's
/// `EngineCoreRequest` arity.
#[derive(Debug, Clone, Copy)]
pub struct OmniRequestDecoder {
    base_len: usize,
}

fn read_array_header(bytes: &[u8]) -> Result<(usize, usize)> {
    match bytes {
        [marker @ 0x90..=0x9f, ..] => Ok((usize::from(marker & 0x0f), 1)),
        [0xdc, a, b, ..] => Ok((usize::from(u16::from_be_bytes([*a, *b])), 3)),
        [0xdd, a, b, c, d, ..] => Ok((
            usize::try_from(u32::from_be_bytes([*a, *b, *c, *d]))
                .context("request array length exceeds usize")?,
            5,
        )),
        _ => bail!("engine request is not a msgpack array"),
    }
}

fn write_array_header(out: &mut Vec<u8>, len: usize) -> Result<()> {
    if let Ok(len) = u8::try_from(len)
        && len < 16
    {
        out.push(0x90 | len);
    } else if let Ok(len) = u16::try_from(len) {
        out.push(0xdc);
        out.extend_from_slice(&len.to_be_bytes());
    } else {
        let len = u32::try_from(len).context("request array length exceeds u32")?;
        out.push(0xdd);
        out.extend_from_slice(&len.to_be_bytes());
    }
    Ok(())
}

impl OmniRequestDecoder {
    pub fn new() -> Result<Self> {
        let encoded = encode_msgpack(&EngineCoreRequest::default())
            .map_err(|e| anyhow!("encoding default engine request: {e}"))?;
        let (base_len, _) = read_array_header(&encoded)?;
        Ok(Self { base_len })
    }

    pub fn decode(&self, bytes: &[u8]) -> Result<OmniRequest> {
        let (len, header_len) = read_array_header(bytes)?;
        if len <= self.base_len {
            let request = decode_msgpack(bytes).map_err(|e| anyhow!("{e}"))?;
            return Ok(OmniRequest {
                request,
                tail: Vec::new(),
            });
        }

        let body = &bytes[header_len..];
        let mut rest = body;
        for index in 0..self.base_len {
            rmpv::decode::read_value_ref(&mut rest)
                .with_context(|| format!("walking engine request field {index}"))?;
        }
        let base_body = &body[..body.len() - rest.len()];

        let mut base = Vec::with_capacity(header_len + base_body.len());
        write_array_header(&mut base, self.base_len)?;
        base.extend_from_slice(base_body);
        let request = decode_msgpack(&base).map_err(|e| anyhow!("{e}"))?;

        let mut tail = Vec::with_capacity(len - self.base_len);
        for index in self.base_len..len {
            tail.push(
                rmpv::decode::read_value(&mut rest)
                    .with_context(|| format!("decoding omni request field {index}"))?,
            );
        }
        Ok(OmniRequest { request, tail })
    }
}

/// msgpack ext type vLLM uses for an inline tensor buffer (`CUSTOM_TYPE_RAW_VIEW`).
const RAW_VIEW_EXT: i8 = 3;
const OUTPUTS_FIELD: usize = 1;
const NEW_TOKEN_IDS_FIELD: usize = 1;

/// The synthetic waveform a final audio stage attaches to its outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioOutput {
    /// Waveform samples produced per generated token.
    pub samples_per_token: u32,
    /// Reported as `sr` when set; the head assumes its own default otherwise.
    pub sample_rate: Option<u32>,
}

fn wire_tensor(dtype: &str, shape: Vec<u64>, data: Vec<u8>) -> rmpv::Value {
    rmpv::Value::Array(vec![
        rmpv::Value::from(dtype),
        rmpv::Value::Array(shape.into_iter().map(rmpv::Value::from).collect()),
        rmpv::Value::Ext(RAW_VIEW_EXT, data),
    ])
}

impl AudioOutput {
    /// `multimodal_output` for an output carrying `tokens` new tokens: silence
    /// of the matching duration.
    fn multimodal_output(&self, tokens: usize) -> Result<rmpv::Value> {
        let samples = tokens
            .checked_mul(self.samples_per_token as usize)
            .context("audio sample count overflows usize")?;
        let bytes = samples
            .checked_mul(size_of::<f32>())
            .context("audio byte count overflows usize")?;
        let mut entries = vec![(
            "audio",
            wire_tensor("float32", vec![samples as u64], vec![0; bytes]),
        )];
        if let Some(sample_rate) = self.sample_rate {
            entries.push((
                "sr",
                wire_tensor(
                    "int64",
                    Vec::new(),
                    i64::from(sample_rate).to_le_bytes().to_vec(),
                ),
            ));
        }
        Ok(map(entries))
    }
}

/// Encodes an output envelope as `OmniEngineCoreOutputs`: each output that
/// carries new tokens gets `multimodal_output` appended past the
/// `EngineCoreOutput` tuple.
#[derive(Debug, Clone, Copy)]
pub struct OmniOutputEncoder {
    base_len: usize,
    audio: AudioOutput,
}

impl OmniOutputEncoder {
    pub fn new(audio: AudioOutput) -> Result<Self> {
        let encoded = encode_msgpack(&EngineCoreOutput::default())
            .map_err(|e| anyhow!("encoding default engine output: {e}"))?;
        let (base_len, _) = read_array_header(&encoded)?;
        Ok(Self { base_len, audio })
    }

    pub fn encode(&self, envelope: &Envelope) -> Result<Vec<u8>> {
        let encoded =
            encode_msgpack(envelope).map_err(|e| anyhow!("encoding output envelope: {e}"))?;
        let mut value = rmpv::decode::read_value(&mut &encoded[..])
            .context("re-reading encoded output envelope")?;
        let outputs = match &mut value {
            rmpv::Value::Array(fields) => fields.get_mut(OUTPUTS_FIELD),
            _ => None,
        };
        let Some(rmpv::Value::Array(outputs)) = outputs else {
            bail!("output envelope has no outputs array");
        };
        for output in outputs {
            let rmpv::Value::Array(fields) = output else {
                bail!("engine output is not a msgpack array");
            };
            if fields.len() != self.base_len {
                bail!(
                    "engine output has {} fields, expected {}",
                    fields.len(),
                    self.base_len
                );
            }
            let tokens = match fields.get(NEW_TOKEN_IDS_FIELD) {
                Some(rmpv::Value::Array(tokens)) => tokens.len(),
                _ => bail!("engine output has no new_token_ids array"),
            };
            if tokens > 0 {
                fields.push(self.audio.multimodal_output(tokens)?);
            }
        }
        encode(&value)
    }
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use crate::omni::llm::{
        AudioOutput, OmniOutputEncoder, OmniRequestDecoder, read_array_header, write_array_header,
    };
    use crate::omni::msgpack::encode;
    use crate::vllm::{
        EngineCoreFinishReason, EngineCoreOutput, EngineCoreRequest, Envelope, UtilityOutput,
        decode_msgpack, encode_msgpack, request_batch, request_outputs, utility,
    };

    fn request_with_tail(request: &EngineCoreRequest, tail: &[Value]) -> Vec<u8> {
        let encoded = encode_msgpack(request).expect("encode request");
        let Value::Array(mut fields) =
            rmpv::decode::read_value(&mut &encoded[..]).expect("decode request")
        else {
            panic!("request is not an array");
        };
        fields.extend_from_slice(tail);
        encode(&Value::Array(fields)).expect("encode")
    }

    #[test]
    fn array_header_round_trips_at_every_width() {
        for len in [0usize, 15, 16, 23, 65_535, 65_536] {
            let mut out = Vec::new();
            write_array_header(&mut out, len).expect("write");
            let header_len = out.len();
            assert_eq!(read_array_header(&out).expect("read"), (len, header_len));
        }
        assert!(read_array_header(&[0xc0]).is_err());
        assert!(read_array_header(&[]).is_err());
    }

    #[test]
    fn plain_request_decodes_with_an_empty_tail() {
        let decoder = OmniRequestDecoder::new().expect("decoder");
        let request = EngineCoreRequest {
            request_id: "req-plain".to_string(),
            ..Default::default()
        };
        let decoded = decoder
            .decode(&encode_msgpack(&request).expect("encode"))
            .expect("decode");
        assert_eq!(*decoded.request, request);
        assert!(decoded.tail.is_empty());
    }

    #[test]
    fn omni_tail_is_split_off_and_the_request_survives() {
        let decoder = OmniRequestDecoder::new().expect("decoder");
        let request = EngineCoreRequest {
            request_id: "req-omni".to_string(),
            ..Default::default()
        };
        let tail = [
            Value::Map(vec![(Value::from("entries"), Value::Map(vec![]))]),
            Value::Nil,
            Value::Binary(vec![1, 2, 3]),
        ];

        let decoded = decoder
            .decode(&request_with_tail(&request, &tail))
            .expect("decode");

        assert_eq!(*decoded.request, request);
        assert_eq!(decoded.tail, tail);
    }

    #[test]
    fn omni_tail_breaks_the_plain_vllm_decoder() {
        let request = EngineCoreRequest::default();
        let bytes = request_with_tail(&request, &[Value::Nil, Value::Nil, Value::from(1)]);
        assert!(crate::vllm::decode_msgpack::<EngineCoreRequest>(&bytes).is_err());
    }

    #[test]
    fn truncated_omni_request_is_an_error() {
        let decoder = OmniRequestDecoder::new().expect("decoder");
        let bytes = request_with_tail(&EngineCoreRequest::default(), &[Value::from("x")]);
        assert!(decoder.decode(&bytes[..bytes.len() - 1]).is_err());
    }

    fn output(request_id: &str, tokens: Vec<u32>) -> EngineCoreOutput {
        EngineCoreOutput {
            request_id: request_id.to_string(),
            new_token_ids: tokens,
            ..Default::default()
        }
    }

    /// The outputs of an encoded envelope, each as its raw field array.
    fn wire_outputs(encoded: &[u8]) -> Vec<Vec<Value>> {
        let Value::Array(fields) = rmpv::decode::read_value(&mut &encoded[..]).expect("decode")
        else {
            panic!("envelope is not an array");
        };
        let Value::Array(outputs) = &fields[1] else {
            panic!("outputs is not an array");
        };
        outputs
            .iter()
            .map(|o| o.as_array().expect("output array").clone())
            .collect()
    }

    fn tensor_parts(tensor: &Value) -> (&str, Vec<u64>, &[u8]) {
        let parts = tensor.as_array().expect("tensor triple");
        assert_eq!(parts.len(), 3);
        let shape = parts[1]
            .as_array()
            .expect("shape")
            .iter()
            .map(|d| d.as_u64().expect("dim"))
            .collect();
        let Value::Ext(kind, data) = &parts[2] else {
            panic!("tensor data is not an ext");
        };
        assert_eq!(*kind, 3, "CUSTOM_TYPE_RAW_VIEW");
        (parts[0].as_str().expect("dtype"), shape, data)
    }

    #[test]
    fn audio_is_appended_past_the_vllm_output_fields() {
        let base_len = wire_outputs(
            &encode_msgpack(&request_batch(
                0,
                vec![output("r", vec![1])],
                None,
                0.0,
                None,
            ))
            .expect("encode"),
        )[0]
        .len();
        let encoder = OmniOutputEncoder::new(AudioOutput {
            samples_per_token: 480,
            sample_rate: None,
        })
        .expect("encoder");

        let envelope = request_batch(0, vec![output("r", vec![7, 8, 9])], None, 1.0, None);
        let outputs = wire_outputs(&encoder.encode(&envelope).expect("encode"));

        assert_eq!(outputs[0].len(), base_len + 1);
        let multimodal = outputs[0][base_len]
            .as_map()
            .expect("multimodal_output map");
        assert_eq!(multimodal.len(), 1);
        assert_eq!(multimodal[0].0.as_str(), Some("audio"));
        let (dtype, shape, data) = tensor_parts(&multimodal[0].1);
        assert_eq!(dtype, "float32");
        assert_eq!(shape, vec![3 * 480]);
        assert_eq!(data.len(), 3 * 480 * 4);
        assert!(data.iter().all(|b| *b == 0));
    }

    #[test]
    fn sample_rate_is_a_scalar_int64_tensor() {
        let encoder = OmniOutputEncoder::new(AudioOutput {
            samples_per_token: 1,
            sample_rate: Some(44_100),
        })
        .expect("encoder");
        let envelope = request_batch(0, vec![output("r", vec![1])], None, 1.0, None);
        let outputs = wire_outputs(&encoder.encode(&envelope).expect("encode"));

        let multimodal = outputs[0].last().expect("field").as_map().expect("map");
        let sr = multimodal
            .iter()
            .find(|(k, _)| k.as_str() == Some("sr"))
            .map(|(_, v)| v)
            .expect("sr entry");
        let (dtype, shape, data) = tensor_parts(sr);
        assert_eq!(dtype, "int64");
        assert!(shape.is_empty());
        assert_eq!(data, 44_100i64.to_le_bytes());
    }

    #[test]
    fn outputs_without_tokens_keep_the_vllm_shape() {
        let encoder = OmniOutputEncoder::new(AudioOutput {
            samples_per_token: 480,
            sample_rate: None,
        })
        .expect("encoder");
        let finished = EngineCoreOutput {
            finish_reason: Some(EngineCoreFinishReason::Abort),
            ..output("r-abort", Vec::new())
        };
        let envelope = request_batch(0, vec![finished, output("r", vec![1])], None, 1.0, None);

        let plain = wire_outputs(&encode_msgpack(&envelope).expect("encode"));
        let omni = wire_outputs(&encoder.encode(&envelope).expect("encode"));

        assert_eq!(omni[0], plain[0]);
        assert_eq!(omni[1].len(), plain[1].len() + 1);
        assert_eq!(omni[1][..plain[1].len()], plain[1][..]);
    }

    /// The vLLM client's envelope decoder skips fields past the ones it knows,
    /// so an omni envelope must still read back as the envelope it came from.
    #[test]
    fn omni_envelope_decodes_back_to_the_vllm_envelope() {
        let encoder = OmniOutputEncoder::new(AudioOutput {
            samples_per_token: 16,
            sample_rate: Some(24_000),
        })
        .expect("encoder");
        let envelope = request_batch(2, vec![output("r", vec![4, 5])], None, 9.5, None);

        let decoded: Envelope =
            decode_msgpack(&encoder.encode(&envelope).expect("encode")).expect("decode");

        assert_eq!(decoded, envelope);
        assert_eq!(request_outputs(&decoded)[0].new_token_ids, vec![4, 5]);
    }

    #[test]
    fn utility_envelope_is_unchanged() {
        let encoder = OmniOutputEncoder::new(AudioOutput {
            samples_per_token: 480,
            sample_rate: None,
        })
        .expect("encoder");
        let envelope = utility(
            0,
            1.0,
            UtilityOutput {
                call_id: 7_u64.into(),
                failure_message: None,
                result: None,
            },
        );
        assert_eq!(
            encoder.encode(&envelope).expect("encode"),
            encode_msgpack(&envelope).expect("encode")
        );
    }
}
