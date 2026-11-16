//! Tapping a vLLM-Omni LLM stage.
//!
//! An omni stage does two things a plain vLLM engine does not: it registers
//! with the head's `OmniMasterServer` before the handshake, and it reports to
//! the head's `OmniCoordinator` after it. The tap stands in the middle of
//! both. The stage registers with the tap, which registers with the head on
//! its behalf and answers with the tap's own addresses. The stage's status
//! events are forwarded to the head with the tap-side data addresses swapped
//! for the head-side ones, which are what the head routes by.

use std::time::Duration;

use anyhow::{Context as _, Result, anyhow};
use serde::Deserialize;
use sim_protocol::omni::attach::{StageRegistration, register_raw};
use sim_protocol::vllm::{decode_msgpack, encode_msgpack};
use tokio::task::JoinHandle;
use tracing::{info, warn};
use zeromq::prelude::{Socket as _, SocketRecv as _, SocketSend as _};
use zeromq::{DealerSocket, RouterSocket, ZmqMessage};

const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(30);

/// Where the tap meets an omni head and the stage it records.
#[derive(Debug, Clone)]
pub struct OmniTap {
    /// The head's `OmniMasterServer` endpoint.
    pub head_master: String,
    /// Where the tap binds its own master, for the real stage to register with.
    pub master_listen: String,
    /// Where the tap binds its own coordinator, for the real stage to report to.
    pub coordinator_listen: String,
}

/// The data addresses the stage hands out on its side of the tap.
pub(crate) struct TapAddresses<'a> {
    pub handshake: &'a str,
    pub input: &'a str,
    pub output: &'a str,
}

#[derive(Deserialize)]
struct RegistrationInfo {
    stage_id: u32,
}

/// A stage registered through the tap.
pub(crate) struct StageAttachment {
    pub stage_id: u32,
    /// The replica slot the head allocated.
    pub head: StageRegistration,
    coordinator: RouterSocket,
    _master: RouterSocket,
}

/// The registration the stage is answered with: the head's slot, with every
/// address replaced by the tap's.
fn stage_reply(
    head: &StageRegistration,
    tap: &TapAddresses<'_>,
    coordinator_listen: &str,
) -> StageRegistration {
    StageRegistration {
        handshake_address: tap.handshake.to_string(),
        input_address: tap.input.to_string(),
        output_address: tap.output.to_string(),
        replica_id: head.replica_id,
        coordinator_router_address: head
            .coordinator_router_address
            .as_ref()
            .map(|_| coordinator_listen.to_string()),
    }
}

/// Wait for the real stage to register, pass its registration to the head
/// unchanged, and answer the stage with the tap's addresses.
pub(crate) async fn accept_stage(
    omni: &OmniTap,
    tap: &TapAddresses<'_>,
) -> Result<StageAttachment> {
    let mut coordinator = RouterSocket::new();
    coordinator
        .bind(&omni.coordinator_listen)
        .await
        .with_context(|| format!("binding tap coordinator at {}", omni.coordinator_listen))?;
    let mut master = RouterSocket::new();
    master
        .bind(&omni.master_listen)
        .await
        .with_context(|| format!("binding tap master at {}", omni.master_listen))?;
    info!(master_listen = %omni.master_listen, "waiting for the real stage to register");

    let frames = master
        .recv()
        .await
        .context("receiving stage registration")?
        .into_vec();
    let (Some(identity), Some(registration)) = (frames.first(), frames.last()) else {
        return Err(anyhow!("stage sent an empty registration"));
    };
    let RegistrationInfo { stage_id } = decode_msgpack(registration.as_ref())
        .map_err(|e| anyhow!("decoding stage registration: {e}"))?;

    let head = register_raw(
        &omni.head_master,
        registration.to_vec(),
        REGISTRATION_TIMEOUT,
    )
    .await?;
    info!(
        stage_id,
        replica_id = head.replica_id,
        head_handshake = %head.handshake_address,
        "registered the stage with the omni head"
    );

    let reply = encode_msgpack(&stage_reply(&head, tap, &omni.coordinator_listen))
        .map_err(|e| anyhow!("encoding stage registration reply: {e}"))?;
    let reply = ZmqMessage::try_from(vec![identity.clone(), reply.into()])
        .map_err(|e| anyhow!("building stage registration reply: {e}"))?;
    master
        .send(reply)
        .await
        .context("answering stage registration")?;

    Ok(StageAttachment {
        stage_id,
        head,
        coordinator,
        _master: master,
    })
}

/// A status event with the tap-side data addresses replaced by the head-side
/// ones. An event that is not a JSON object is passed through unchanged.
fn rewrite_event(event: &[u8], head_input: &str, head_output: &str) -> Vec<u8> {
    let Ok(serde_json::Value::Object(mut fields)) = serde_json::from_slice(event) else {
        warn!("stage status event is not a JSON object; forwarding it unchanged");
        return event.to_vec();
    };
    fields.insert("input_addr".to_string(), head_input.into());
    fields.insert("output_addr".to_string(), head_output.into());
    serde_json::to_vec(&fields).unwrap_or_else(|_| event.to_vec())
}

impl StageAttachment {
    /// Forward the stage's status events to the head's coordinator.
    /// `head_input` / `head_output` are the data addresses from the head's
    /// handshake INIT. `None` when the head runs no coordinator.
    pub(crate) fn relay_status(
        self,
        head_input: String,
        head_output: String,
    ) -> Option<JoinHandle<()>> {
        let head_coordinator = self.head.coordinator_router_address?;
        let mut from_stage = self.coordinator;
        let master = self._master;
        Some(tokio::spawn(async move {
            let _master = master;
            let mut to_head = DealerSocket::new();
            if let Err(error) = to_head.connect(&head_coordinator).await {
                warn!(%error, head_coordinator, "tap could not reach the omni coordinator");
                return;
            }
            loop {
                let frames = match from_stage.recv().await {
                    Ok(message) => message.into_vec(),
                    Err(error) => {
                        warn!(%error, "tap coordinator recv error; stopping the status relay");
                        return;
                    }
                };
                let Some(event) = frames.last() else {
                    continue;
                };
                let event = rewrite_event(event, &head_input, &head_output);
                if let Err(error) = to_head.send(ZmqMessage::from(event)).await {
                    warn!(%error, "tap could not forward a stage status event");
                }
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use sim_protocol::omni::attach::StageRegistration;

    use crate::omni::{TapAddresses, rewrite_event, stage_reply};

    const TAP: TapAddresses<'static> = TapAddresses {
        handshake: "tcp://127.0.0.1:5580",
        input: "tcp://127.0.0.1:29560",
        output: "tcp://127.0.0.1:29561",
    };

    fn head(coordinator: Option<&str>) -> StageRegistration {
        StageRegistration {
            handshake_address: "tcp://10.0.0.1:40000".to_string(),
            input_address: "tcp://10.0.0.1:40001".to_string(),
            output_address: "tcp://10.0.0.1:40002".to_string(),
            replica_id: 3,
            coordinator_router_address: coordinator.map(str::to_string),
        }
    }

    #[test]
    fn stage_is_answered_with_the_taps_addresses_and_the_heads_replica_id() {
        let reply = stage_reply(
            &head(Some("tcp://10.0.0.1:40003")),
            &TAP,
            "tcp://127.0.0.1:21214",
        );
        assert_eq!(
            reply,
            StageRegistration {
                handshake_address: "tcp://127.0.0.1:5580".to_string(),
                input_address: "tcp://127.0.0.1:29560".to_string(),
                output_address: "tcp://127.0.0.1:29561".to_string(),
                replica_id: 3,
                coordinator_router_address: Some("tcp://127.0.0.1:21214".to_string()),
            }
        );
    }

    #[test]
    fn stage_gets_no_coordinator_when_the_head_has_none() {
        let reply = stage_reply(&head(None), &TAP, "tcp://127.0.0.1:21214");
        assert_eq!(reply.coordinator_router_address, None);
    }

    #[test]
    fn status_event_addresses_are_swapped_and_the_rest_is_kept() {
        let event = br#"{"input_addr":"tcp://127.0.0.1:29560","output_addr":"tcp://127.0.0.1:29561","stage_id":1,"event_type":"heartbeat","status":"up","queue_length":4}"#;
        let rewritten: serde_json::Value = serde_json::from_slice(&rewrite_event(
            event,
            "tcp://10.0.0.1:40001",
            "tcp://10.0.0.1:40002",
        ))
        .expect("json");
        assert_eq!(
            rewritten,
            serde_json::json!({
                "input_addr": "tcp://10.0.0.1:40001",
                "output_addr": "tcp://10.0.0.1:40002",
                "stage_id": 1,
                "event_type": "heartbeat",
                "status": "up",
                "queue_length": 4,
            })
        );
    }

    #[test]
    fn unparsable_status_event_passes_through() {
        assert_eq!(rewrite_event(b"not json", "a", "b"), b"not json");
        assert_eq!(rewrite_event(b"[1,2]", "a", "b"), b"[1,2]");
    }
}
