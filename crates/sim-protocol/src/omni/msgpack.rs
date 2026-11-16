//! Helpers for the untyped msgpack maps vLLM-Omni exchanges.

use anyhow::{Context as _, Result};
use rmpv::Value;

/// The value under `key` in a msgpack map.
pub(crate) fn field<'a>(map: &'a Value, key: &str) -> Option<&'a Value> {
    map.as_map()?
        .iter()
        .find(|(k, _)| k.as_str() == Some(key))
        .map(|(_, v)| v)
}

/// A msgpack map with string keys, in the given order.
pub(crate) fn map(entries: Vec<(&str, Value)>) -> Value {
    Value::Map(
        entries
            .into_iter()
            .map(|(key, value)| (Value::from(key), value))
            .collect(),
    )
}

pub(crate) fn encode(value: &Value) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, value).context("encoding msgpack value")?;
    Ok(out)
}
