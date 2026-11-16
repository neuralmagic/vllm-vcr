//! A scripted vLLM-Omni head for end-to-end tests: real sockets, with the
//! head's side of the wire written out by hand. It mirrors what a v0.32.0 head
//! does, including the part plain vLLM does not: the data sockets are bound
//! only after the stage reports READY.

#![allow(dead_code)]

use std::future::Future;
use std::time::Duration;

use rmpv::Value;
use sim_protocol::vllm::{EngineCoreRequest, encode_msgpack};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use vllm_vcr::{Opt, run};
use zeromq::prelude::{Socket as _, SocketRecv as _, SocketSend as _};
use zeromq::{RouterSocket, ZmqMessage};

pub const TIMEOUT: Duration = Duration::from_secs(10);

pub async fn within<T>(what: &str, future: impl Future<Output = T>) -> T {
    tokio::time::timeout(TIMEOUT, future)
        .await
        .unwrap_or_else(|_| panic!("timed out: {what}"))
}

pub fn map(entries: Vec<(&str, Value)>) -> Value {
    Value::Map(
        entries
            .into_iter()
            .map(|(key, value)| (Value::from(key), value))
            .collect(),
    )
}

pub fn field<'a>(map: &'a Value, key: &str) -> &'a Value {
    map.as_map()
        .unwrap_or_else(|| panic!("not a map: {map}"))
        .iter()
        .find(|(k, _)| k.as_str() == Some(key))
        .map(|(_, v)| v)
        .unwrap_or_else(|| panic!("missing key {key} in {map}"))
}

pub fn encode(value: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, value).expect("encode");
    out
}

pub fn decode(bytes: &[u8]) -> Value {
    rmpv::decode::read_value(&mut &bytes[..]).expect("decode")
}

/// A local TCP endpoint nothing is bound to yet.
pub fn unbound_endpoint() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("reserve port");
    let port = listener.local_addr().expect("local addr").port();
    format!("tcp://127.0.0.1:{port}")
}

pub async fn bound_router() -> (RouterSocket, String) {
    let mut socket = RouterSocket::new();
    let endpoint = socket.bind("tcp://127.0.0.1:0").await.expect("bind");
    (socket, endpoint.to_string())
}

/// The head's control plane: master, handshake, and coordinator sockets, plus
/// the data addresses it will bind once the stage is READY.
pub struct Head {
    pub master: RouterSocket,
    pub master_address: String,
    pub handshake: RouterSocket,
    pub handshake_address: String,
    pub coordinator: RouterSocket,
    pub coordinator_address: String,
    pub input_address: String,
    pub output_address: String,
}

impl Head {
    pub async fn start() -> Self {
        let (master, master_address) = bound_router().await;
        let (handshake, handshake_address) = bound_router().await;
        let (coordinator, coordinator_address) = bound_router().await;
        Self {
            master,
            master_address,
            handshake,
            handshake_address,
            coordinator,
            coordinator_address,
            input_address: unbound_endpoint(),
            output_address: unbound_endpoint(),
        }
    }

    /// Answer one stage registration and return what the stage sent.
    pub async fn accept_registration(&mut self) -> Value {
        let frames = within("registration", self.master.recv())
            .await
            .expect("registration")
            .into_vec();
        let mut reply = ZmqMessage::from(frames[0].to_vec());
        reply.push_back(
            encode(&map(vec![
                (
                    "handshake_address",
                    Value::from(self.handshake_address.as_str()),
                ),
                ("input_address", Value::from(self.input_address.as_str())),
                ("output_address", Value::from(self.output_address.as_str())),
                ("replica_id", Value::from(0)),
                (
                    "coordinator_router_address",
                    Value::from(self.coordinator_address.as_str()),
                ),
            ]))
            .into(),
        );
        self.master.send(reply).await.expect("registration reply");
        decode(frames.last().expect("registration payload"))
    }

    /// HELLO, INIT, READY. Returns the stage's handshake identity.
    pub async fn handshake(&mut self) -> Vec<u8> {
        let hello = within("HELLO", self.handshake.recv())
            .await
            .expect("HELLO")
            .into_vec();
        let identity = hello[0].to_vec();
        let message = decode(&hello[1]);
        assert_eq!(field(&message, "status").as_str(), Some("HELLO"));
        assert_eq!(field(&message, "local"), &Value::from(false));
        assert_eq!(field(&message, "headless"), &Value::from(true));

        let mut init = ZmqMessage::from(identity.clone());
        init.push_back(
            encode(&map(vec![
                (
                    "addresses",
                    map(vec![
                        (
                            "inputs",
                            Value::Array(vec![Value::from(self.input_address.as_str())]),
                        ),
                        (
                            "outputs",
                            Value::Array(vec![Value::from(self.output_address.as_str())]),
                        ),
                        ("coordinator_input", Value::Nil),
                        ("coordinator_output", Value::Nil),
                        ("frontend_stats_publish_address", Value::Nil),
                    ]),
                ),
                ("parallel_config", Value::Map(Vec::new())),
            ]))
            .into(),
        );
        self.handshake.send(init).await.expect("INIT");

        let ready = within("READY", self.handshake.recv())
            .await
            .expect("READY")
            .into_vec();
        assert_eq!(
            field(&decode(&ready[1]), "status").as_str(),
            Some("READY"),
            "the stage must report READY before its data sockets can connect"
        );
        identity
    }

    pub async fn replica_event(&mut self) -> serde_json::Value {
        let frames = within("replica event", self.coordinator.recv())
            .await
            .expect("replica event")
            .into_vec();
        serde_json::from_slice(frames.last().expect("event payload")).expect("event json")
    }

    /// The stage's first report, sent once it is attached.
    pub async fn expect_replica_up(&mut self, stage_id: u32) {
        assert_eq!(
            self.replica_event().await,
            serde_json::json!({
                "input_addr": self.input_address,
                "output_addr": self.output_address,
                "stage_id": stage_id,
                "event_type": "update",
                "status": "up",
                "queue_length": 0,
            })
        );
    }

    pub async fn expect_replica_down(&mut self) {
        loop {
            let event = self.replica_event().await;
            if event["event_type"] == "heartbeat" {
                continue;
            }
            assert_eq!(event["event_type"], "update");
            assert_eq!(event["status"], "down");
            return;
        }
    }
}

pub struct Stage {
    pub task: Option<JoinHandle<anyhow::Result<()>>>,
    pub shutdown: CancellationToken,
}

impl Stage {
    /// Run `play` as an omni stage registering with `master_address`.
    pub fn spawn(master_address: &str, stage_id: &str, flags: &[&str]) -> Self {
        let mut args = vec![
            "play",
            "--omni-master-address",
            master_address,
            "--omni-stage-id",
            stage_id,
        ];
        args.extend_from_slice(flags);
        let opt = Opt::parse_from(&args);
        let shutdown = CancellationToken::new();
        let task = Some(tokio::spawn(run(opt, shutdown.clone())));
        Self { task, shutdown }
    }

    pub async fn finished(mut self) {
        let task = self.task.take().expect("stage task");
        within("stage exit", task)
            .await
            .expect("stage task join")
            .expect("stage result");
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        self.shutdown.cancel();
    }
}

pub fn request_with_tail(request: &EngineCoreRequest, tail: &[Value]) -> Vec<u8> {
    let Value::Array(mut fields) = decode(&encode_msgpack(request).expect("encode request")) else {
        panic!("request is not an array");
    };
    fields.extend_from_slice(tail);
    encode(&Value::Array(fields))
}

/// The samples in an output's `multimodal_output.audio`, when it has one.
pub fn audio_samples(output: &[Value], vllm_fields: usize) -> Option<u64> {
    let multimodal = output.get(vllm_fields)?;
    let tensor = field(multimodal, "audio")
        .as_array()
        .expect("tensor triple");
    assert_eq!(tensor[0].as_str(), Some("float32"));
    let shape = tensor[1].as_array().expect("shape");
    assert_eq!(shape.len(), 1);
    let samples = shape[0].as_u64().expect("samples");
    let Value::Ext(_, data) = &tensor[2] else {
        panic!("tensor data is not an ext");
    };
    assert_eq!(data.len() as u64, samples * 4);
    Some(samples)
}
