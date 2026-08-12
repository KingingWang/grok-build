use serde::{Deserialize, Deserializer};
use serde_json::Value;

pub fn empty_string_as_none<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let opt = Option::<String>::deserialize(deserializer)?;
    Ok(opt.filter(|s| !s.is_empty()))
}

/// Deserialize `Option<T>` where an empty string (`""`) is treated as `None`.
///
/// OpenAI-compatible providers sometimes emit `"finish_reason": ""` instead of
/// `null` on intermediate streaming chunks. Requires
/// `#[serde(default, deserialize_with = "…")]`.
pub fn empty_string_as_none_value<'de, T, D>(deserializer: D) -> Result<Option<T>, D::Error>
where
    T: Deserialize<'de>,
    D: Deserializer<'de>,
{
    match Option::<Value>::deserialize(deserializer)? {
        None => Ok(None),
        Some(value) if value.as_str() == Some("") => Ok(None),
        Some(value) => T::deserialize(value)
            .map(Some)
            .map_err(serde::de::Error::custom),
    }
}

/// Deserialize `Option<Option<T>>`: absent (`None`) leaves, `null` (`Some(None)`)
/// clears, a value sets. Requires `#[serde(default, deserialize_with = "…")]`.
pub fn double_option<'de, T, D>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    T: Deserialize<'de>,
    D: Deserializer<'de>,
{
    Ok(Some(Option::deserialize(deserializer)?))
}
