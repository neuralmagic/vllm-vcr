//! ZMQ IO loop: decode frontend frames into `EngineInput`, push `EngineOutput` back.
//! Same shape as vLLM's in-tree `vllm-mock-engine` IO loop, built entirely on the
//! public `vllm-engine-core-client` protocol API.

use anyhow::{Context as _, Result, anyhow, bail};
use futures::{Stream, StreamExt as _, stream};
use sim_protocol::mock_engine::MockEngineDataSockets;
use sim_protocol::omni::llm::{AudioOutput, OmniOutputEncoder, OmniRequest, OmniRequestDecoder};
use sim_protocol::vllm::{
    EngineCoreRequest, EngineCoreRequestType, Envelope, decode_msgpack, encode_msgpack,
};
use sim_protocol::wire::request_type_from_frame;
use tokio::sync::mpsc;
use tracing::{debug, warn};
use zeromq::{DealerSocket, PushSocket, SocketRecv as _, SocketSend as _, ZmqMessage};

use crate::engine_core::{EngineInput, EngineOutput, UtilityRequestSpec};

/// Send one engine output batch to the client over the appropriate push socket.
async fn send_engine_outputs_to_client(
    push_sockets: &mut [PushSocket],
    codec: Codec,
    EngineOutput {
        client_index,
        outputs,
    }: EngineOutput,
) -> Result<()> {
    let socket = match push_sockets.get_mut(client_index as usize) {
        Some(s) => s,
        None => {
            warn!(
                client_index,
                socket_count = push_sockets.len(),
                "client_index out of range, dropping output batch"
            );
            return Ok(());
        }
    };
    let message = ZmqMessage::from(codec.encode_outputs(&outputs)?);
    socket.send(message).await?;
    Ok(())
}

/// How this engine's data frames are encoded: plain vLLM engine-core, or a
/// vLLM-Omni LLM stage, whose requests and outputs carry fields past the vLLM
/// tuples.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Codec {
    Vllm,
    Omni {
        requests: OmniRequestDecoder,
        /// Set on a pipeline's final audio stage.
        outputs: Option<OmniOutputEncoder>,
    },
}

impl Codec {
    pub(crate) fn omni(audio: Option<AudioOutput>) -> Result<Self> {
        Ok(Self::Omni {
            requests: OmniRequestDecoder::new()?,
            outputs: audio.map(OmniOutputEncoder::new).transpose()?,
        })
    }

    fn decode_add(self, frame: &[u8]) -> Result<Box<EngineCoreRequest>> {
        match self {
            Self::Vllm => Ok(decode_msgpack(frame)?),
            Self::Omni { requests, .. } => {
                let OmniRequest { request, tail } = requests.decode(frame)?;
                if tail.iter().any(|field| !field.is_nil()) {
                    debug!(
                        request_id = %request.request_id,
                        tail_fields = tail.len(),
                        "dropping omni request tail"
                    );
                }
                Ok(request)
            }
        }
    }

    fn encode_outputs(self, outputs: &Envelope) -> Result<Vec<u8>> {
        match self {
            Self::Omni {
                outputs: Some(encoder),
                ..
            } => encoder.encode(outputs),
            Self::Vllm | Self::Omni { outputs: None, .. } => Ok(encode_msgpack(outputs)?),
        }
    }
}

/// Continuously receive and decode messages from one dealer socket into `EngineInput`.
fn dealer_input_stream(
    dealer: DealerSocket,
    codec: Codec,
) -> impl Stream<Item = Result<EngineInput>> {
    stream::unfold(dealer, move |mut dealer| async move {
        let input = loop {
            let message = match dealer.recv().await.context("dealer recv failed") {
                Ok(message) => message,
                Err(err) => break Err(err),
            };
            match decode_request(message, codec) {
                Ok(input) => break Ok(input),
                Err(err) => warn!(%err, "failed to decode engine request; ignoring"),
            }
        };
        Some((input, dealer))
    })
}

/// Decode a `ZmqMessage` into an `EngineInput`.
fn decode_request(message: ZmqMessage, codec: Codec) -> Result<EngineInput> {
    let frames = message.into_vec();
    if frames.len() < 2 {
        bail!(
            "engine request needs at least 2 frames, got {}",
            frames.len()
        );
    }

    let request_type_frame = frames[0].as_ref();
    let Some(request_type) = request_type_from_frame(request_type_frame) else {
        bail!("unknown engine request type: {:?}", request_type_frame);
    };

    let input = match request_type {
        EngineCoreRequestType::Add => {
            // The Python frontend's MsgpackEncoder splits large tensor payloads
            // (multimodal pixels, prompt embeds) out of the primary msgpack frame
            // into trailing aux frames, so a multimodal Add arrives as >2 frames.
            // The request decoder records those payloads as aux-frame indices, and
            // replay serves trace tokens regardless of prompt content, so decode
            // frame[1] and drop the aux frames (frames[2..]).
            EngineInput::Request(codec.decode_add(frames[1].as_ref())?)
        }
        EngineCoreRequestType::Abort => {
            let request_ids: Vec<String> = decode_msgpack(frames[1].as_ref())?;
            EngineInput::Abort(request_ids)
        }
        EngineCoreRequestType::Utility => {
            let request: UtilityRequestSpec = decode_msgpack(frames[1].as_ref())?;
            EngineInput::Utility(request)
        }
        EngineCoreRequestType::StartDpWave => EngineInput::StartDpWave,
    };

    Ok(input)
}

/// Run the IO loop: dealer frames -> `input_tx`, `output_rx` -> push sockets.
///
/// The loop has no shutdown signal of its own: it must keep delivering outputs while
/// the engine drains in-flight requests, so it exits only when the engine loop is done
/// (the output channel closes, or an input send finds the engine gone), after flushing
/// every queued output - the final tokens and abort notices of a graceful shutdown.
pub(crate) async fn run_io_loop(
    data_sockets: Vec<MockEngineDataSockets>,
    codec: Codec,
    input_tx: mpsc::UnboundedSender<EngineInput>,
    mut output_rx: mpsc::UnboundedReceiver<EngineOutput>,
) -> Result<()> {
    let (dealers, mut push_sockets): (Vec<_>, Vec<_>) = data_sockets
        .into_iter()
        .map(|sockets| (sockets.dealer, sockets.push))
        .unzip();
    let mut input_streams = stream::select_all(
        dealers
            .into_iter()
            .map(|dealer| dealer_input_stream(dealer, codec))
            .map(Box::pin),
    );

    loop {
        tokio::select! {
            biased;
            output = output_rx.recv() => {
                let Some(output) = output else {
                    // Engine loop exited and every queued output is flushed.
                    return Ok(());
                };
                send_engine_outputs_to_client(&mut push_sockets, codec, output).await?;
            }

            input = input_streams.next() => {
                let input = input.ok_or_else(|| anyhow!("engine input streams closed"))??;
                if input_tx.send(input).is_err() {
                    // Engine loop exited between our select arms; flush and leave.
                    while let Some(output) = output_rx.recv().await {
                        send_engine_outputs_to_client(&mut push_sockets, codec, output).await?;
                    }
                    return Ok(());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an `Add` engine request message: type frame, the msgpack request,
    /// then `aux_frames` trailing aux frames (what the Python frontend appends
    /// for split tensor payloads).
    fn add_message(req: &EngineCoreRequest, aux_frames: usize) -> ZmqMessage {
        let mut msg = ZmqMessage::from(vec![0u8]); // 0 == Add
        msg.push_back(encode_msgpack(req).expect("encode request").into());
        for i in 0..aux_frames {
            msg.push_back(vec![0xABu8; i + 1].into());
        }
        msg
    }

    #[test]
    fn decodes_text_add_with_exactly_two_frames() {
        let req = EngineCoreRequest {
            request_id: "req-text".to_string(),
            ..Default::default()
        };
        let EngineInput::Request(decoded) =
            decode_request(add_message(&req, 0), Codec::Vllm).expect("decode 2-frame add")
        else {
            panic!("expected EngineInput::Request");
        };
        assert_eq!(decoded.request_id, "req-text");
    }

    #[test]
    fn decodes_multimodal_add_ignoring_trailing_aux_frames() {
        // A multimodal Add arrives as >2 frames (frame[1] msgpack + aux tensor
        // frames). Decode must read frame[1] and ignore the aux frames.
        let req = EngineCoreRequest {
            request_id: "req-mm".to_string(),
            ..Default::default()
        };
        let EngineInput::Request(decoded) = decode_request(add_message(&req, 3), Codec::Vllm)
            .expect("decode 5-frame multimodal add")
        else {
            panic!("expected EngineInput::Request");
        };
        assert_eq!(decoded.request_id, "req-mm");
    }

    /// An `Add` whose request tuple carries `tail` past the vLLM fields, the
    /// shape `OmniEngineCoreRequest` puts on the wire.
    fn omni_add_message(req: &EngineCoreRequest, tail: &[rmpv::Value]) -> ZmqMessage {
        let encoded = encode_msgpack(req).expect("encode request");
        let rmpv::Value::Array(mut fields) =
            rmpv::decode::read_value(&mut &encoded[..]).expect("decode request")
        else {
            panic!("request is not an array");
        };
        fields.extend_from_slice(tail);
        let mut frame = Vec::new();
        rmpv::encode::write_value(&mut frame, &rmpv::Value::Array(fields)).expect("encode");
        let mut msg = ZmqMessage::from(vec![0u8]);
        msg.push_back(frame.into());
        msg
    }

    fn omni_tail() -> [rmpv::Value; 3] {
        [
            rmpv::Value::Nil,
            rmpv::Value::Nil,
            rmpv::Value::Map(vec![(
                rmpv::Value::from("zmq_port"),
                rmpv::Value::from(50051),
            )]),
        ]
    }

    #[test]
    fn omni_decoder_accepts_an_add_with_an_omni_tail() {
        let req = EngineCoreRequest {
            request_id: "req-omni".to_string(),
            ..Default::default()
        };
        let decoder = Codec::omni(None).expect("codec");
        let EngineInput::Request(decoded) =
            decode_request(omni_add_message(&req, &omni_tail()), decoder).expect("decode omni add")
        else {
            panic!("expected EngineInput::Request");
        };
        assert_eq!(*decoded, req);
    }

    #[test]
    fn omni_decoder_accepts_a_plain_add() {
        let req = EngineCoreRequest {
            request_id: "req-plain".to_string(),
            ..Default::default()
        };
        let decoder = Codec::omni(None).expect("codec");
        let EngineInput::Request(decoded) =
            decode_request(add_message(&req, 0), decoder).expect("decode plain add")
        else {
            panic!("expected EngineInput::Request");
        };
        assert_eq!(*decoded, req);
    }

    #[test]
    fn vllm_decoder_rejects_an_add_with_an_omni_tail() {
        let req = EngineCoreRequest::default();
        assert!(decode_request(omni_add_message(&req, &omni_tail()), Codec::Vllm).is_err());
    }

    #[test]
    fn rejects_messages_with_fewer_than_two_frames() {
        assert!(decode_request(ZmqMessage::from(vec![0u8]), Codec::Vllm).is_err());
    }
}
