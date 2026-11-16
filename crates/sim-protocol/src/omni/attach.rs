//! Attaching a stage replica to a vLLM-Omni head.
//!
//! A headless replica registers with the head's `OmniMasterServer`, which
//! hands back the handshake and data addresses of a replica slot, and once
//! attached reports its status to the head's `OmniCoordinator`. Both are the
//! same for every stage kind.

use std::time::Duration;

use anyhow::{Context as _, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use tokio::time::timeout;
use zeromq::prelude::{Socket as _, SocketRecv as _, SocketSend as _};
use zeromq::{DealerSocket, ZmqMessage};

use crate::vllm::{decode_msgpack, encode_msgpack};

/// Replica id that asks the master to assign the next free slot.
const AUTO_ASSIGN_REPLICA_ID: i64 = -1;

#[derive(Debug, Serialize)]
struct StageConfigStub {
    engine: &'static str,
}

#[derive(Debug, Serialize)]
struct StageRegistrationRequest<'a> {
    stage_id: u32,
    replica_id: i64,
    stage_config: StageConfigStub,
    replica_bind_address: Option<&'a str>,
}

/// The master's reply to a stage registration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageRegistration {
    pub handshake_address: String,
    pub input_address: String,
    pub output_address: String,
    pub replica_id: u32,
    #[serde(default)]
    pub coordinator_router_address: Option<String>,
}

fn encode_registration(stage_id: u32, replica_bind_address: Option<&str>) -> Result<Vec<u8>> {
    encode_msgpack(&StageRegistrationRequest {
        stage_id,
        replica_id: AUTO_ASSIGN_REPLICA_ID,
        stage_config: StageConfigStub { engine: "vllm-vcr" },
        replica_bind_address,
    })
    .map_err(|e| anyhow!("encoding stage registration: {e}"))
}

/// Register one replica of `stage_id` with the head's `OmniMasterServer`.
pub async fn register_stage(
    master_address: &str,
    stage_id: u32,
    replica_bind_address: Option<&str>,
    reply_timeout: Duration,
) -> Result<StageRegistration> {
    register_raw(
        master_address,
        encode_registration(stage_id, replica_bind_address)?,
        reply_timeout,
    )
    .await
}

/// Send an already-encoded registration to the head's `OmniMasterServer`.
pub async fn register_raw(
    master_address: &str,
    registration: Vec<u8>,
    reply_timeout: Duration,
) -> Result<StageRegistration> {
    let mut socket = DealerSocket::new();
    socket
        .connect(master_address)
        .await
        .with_context(|| format!("connecting to omni master {master_address}"))?;
    socket
        .send(ZmqMessage::from(registration))
        .await
        .context("sending stage registration")?;
    let frames = timeout(reply_timeout, socket.recv())
        .await
        .with_context(|| format!("no registration reply from omni master {master_address}"))?
        .context("receiving stage registration reply")?
        .into_vec();
    let Some(payload) = frames.last() else {
        bail!("omni master sent an empty registration reply");
    };
    decode_msgpack(payload.as_ref()).map_err(|e| anyhow!("decoding stage registration reply: {e}"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReplicaStatus {
    Up,
    Down,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReplicaEventType {
    Update,
    Heartbeat,
}

#[derive(Debug, Serialize)]
struct ReplicaEvent<'a> {
    input_addr: &'a str,
    output_addr: &'a str,
    stage_id: u32,
    event_type: ReplicaEventType,
    status: ReplicaStatus,
    queue_length: usize,
}

/// A stage replica's connection to the head's `OmniCoordinator`.
pub struct CoordinatorClient {
    socket: DealerSocket,
    input_addr: String,
    output_addr: String,
    stage_id: u32,
}

impl CoordinatorClient {
    /// `input_addr` / `output_addr` are the replica's data addresses from the
    /// handshake INIT, which the coordinator keys the replica by.
    pub async fn connect(
        coordinator_address: &str,
        stage_id: u32,
        input_addr: String,
        output_addr: String,
    ) -> Result<Self> {
        let mut socket = DealerSocket::new();
        socket
            .connect(coordinator_address)
            .await
            .with_context(|| format!("connecting to omni coordinator {coordinator_address}"))?;
        Ok(Self {
            socket,
            input_addr,
            output_addr,
            stage_id,
        })
    }

    fn encode(
        &self,
        event_type: ReplicaEventType,
        status: ReplicaStatus,
        queue_length: usize,
    ) -> Result<Vec<u8>> {
        serde_json::to_vec(&ReplicaEvent {
            input_addr: &self.input_addr,
            output_addr: &self.output_addr,
            stage_id: self.stage_id,
            event_type,
            status,
            queue_length,
        })
        .context("encoding replica event")
    }

    pub async fn send(
        &mut self,
        event_type: ReplicaEventType,
        status: ReplicaStatus,
        queue_length: usize,
    ) -> Result<()> {
        let payload = self.encode(event_type, status, queue_length)?;
        self.socket
            .send(ZmqMessage::from(payload))
            .await
            .context("sending replica event to omni coordinator")
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rmpv::Value;
    use zeromq::prelude::{Socket as _, SocketRecv as _, SocketSend as _};
    use zeromq::{DealerSocket, RouterSocket, ZmqMessage};

    use crate::omni::attach::{
        CoordinatorClient, ReplicaEventType, ReplicaStatus, StageRegistration, encode_registration,
        register_stage,
    };
    use crate::omni::msgpack::{encode, field, map};

    fn get<'a>(map: &'a Value, key: &str) -> &'a Value {
        field(map, key).unwrap_or_else(|| panic!("missing key {key}"))
    }

    #[test]
    fn registration_is_a_map_with_auto_assigned_replica_and_a_stage_config() {
        let encoded = encode_registration(2, Some("10.0.0.7")).expect("encode");
        let value = rmpv::decode::read_value(&mut &encoded[..]).expect("decode");

        assert_eq!(get(&value, "stage_id"), &Value::from(2));
        assert_eq!(get(&value, "replica_id"), &Value::from(-1));
        assert_eq!(
            get(&value, "replica_bind_address"),
            &Value::from("10.0.0.7")
        );
        assert!(
            !get(&value, "stage_config").is_nil(),
            "the head rejects a registration whose stage_config is None"
        );
    }

    #[test]
    fn registration_without_bind_address_sends_nil() {
        let encoded = encode_registration(0, None).expect("encode");
        let value = rmpv::decode::read_value(&mut &encoded[..]).expect("decode");
        assert!(get(&value, "replica_bind_address").is_nil());
    }

    #[tokio::test]
    async fn register_stage_round_trips_with_a_router_peer() {
        let mut router = RouterSocket::new();
        let endpoint = router
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("bind")
            .to_string();

        let head = tokio::spawn(async move {
            let frames = router.recv().await.expect("recv").into_vec();
            assert_eq!(frames.len(), 2, "identity + payload");
            let request = rmpv::decode::read_value(&mut frames[1].as_ref()).expect("decode");
            assert_eq!(get(&request, "stage_id"), &Value::from(1));

            let mut reply = ZmqMessage::from(frames[0].to_vec());
            reply.push_back(
                encode(&map(vec![
                    ("handshake_address", Value::from("tcp://10.0.0.1:5000")),
                    ("input_address", Value::from("tcp://10.0.0.1:5001")),
                    ("output_address", Value::from("tcp://10.0.0.1:5002")),
                    ("replica_id", Value::from(3)),
                    (
                        "coordinator_router_address",
                        Value::from("tcp://10.0.0.1:5003"),
                    ),
                ]))
                .expect("encode")
                .into(),
            );
            router.send(reply).await.expect("send");
        });

        let registration = register_stage(&endpoint, 1, None, Duration::from_secs(5))
            .await
            .expect("register");
        head.await.expect("head task");

        assert_eq!(
            registration,
            StageRegistration {
                handshake_address: "tcp://10.0.0.1:5000".to_string(),
                input_address: "tcp://10.0.0.1:5001".to_string(),
                output_address: "tcp://10.0.0.1:5002".to_string(),
                replica_id: 3,
                coordinator_router_address: Some("tcp://10.0.0.1:5003".to_string()),
            }
        );
    }

    #[tokio::test]
    async fn register_stage_times_out_when_the_master_never_replies() {
        let mut router = RouterSocket::new();
        let endpoint = router
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("bind")
            .to_string();

        let error = register_stage(&endpoint, 0, None, Duration::from_millis(100))
            .await
            .expect_err("no reply");
        assert!(format!("{error:#}").contains("no registration reply"));
    }

    #[test]
    fn registration_reply_without_coordinator_decodes() {
        let payload = encode(&map(vec![
            ("handshake_address", Value::from("tcp://h:1")),
            ("input_address", Value::from("tcp://h:2")),
            ("output_address", Value::from("tcp://h:3")),
            ("replica_id", Value::from(0)),
            ("coordinator_router_address", Value::Nil),
        ]))
        .expect("encode");
        let registration: StageRegistration =
            crate::vllm::decode_msgpack(&payload).expect("decode");
        assert_eq!(registration.coordinator_router_address, None);
    }

    #[tokio::test]
    async fn replica_event_matches_the_coordinator_json_schema() {
        let mut router = RouterSocket::new();
        let endpoint = router
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("bind")
            .to_string();
        let mut client = CoordinatorClient::connect(
            &endpoint,
            4,
            "tcp://h:10".to_string(),
            "tcp://h:11".to_string(),
        )
        .await
        .expect("connect");

        client
            .send(ReplicaEventType::Heartbeat, ReplicaStatus::Up, 7)
            .await
            .expect("send");

        let frames = router.recv().await.expect("recv").into_vec();
        let event: serde_json::Value =
            serde_json::from_slice(frames.last().expect("payload").as_ref()).expect("json");
        assert_eq!(
            event,
            serde_json::json!({
                "input_addr": "tcp://h:10",
                "output_addr": "tcp://h:11",
                "stage_id": 4,
                "event_type": "heartbeat",
                "status": "up",
                "queue_length": 7,
            })
        );
    }

    #[tokio::test]
    async fn replica_down_update_uses_lowercase_wire_names() {
        let client = CoordinatorClient {
            socket: DealerSocket::new(),
            input_addr: "i".to_string(),
            output_addr: "o".to_string(),
            stage_id: 0,
        };
        let encoded = client
            .encode(ReplicaEventType::Update, ReplicaStatus::Down, 0)
            .expect("encode");
        let event: serde_json::Value = serde_json::from_slice(&encoded).expect("json");
        assert_eq!(event["event_type"], "update");
        assert_eq!(event["status"], "down");
    }
}
