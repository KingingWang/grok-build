//! [`SamplerConfig`] is the per-request configuration handed to the sampler.
//! It deliberately does **not** alias `xai_grok_sampling_types::SamplingConfig`.
//! Aliasing would pull transitive dependencies on shell-specific types (`xai-grok-tools`, etc.) into the sampler crate.

use std::path::PathBuf;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use xai_grok_sampling_types::{
    ApiBackend, CompactionAtTokens, CompactionsRemaining, ConversationGroupId,
    DoomLoopRecoveryPolicy, ReasoningEffort, ReasoningSummary,
};

use crate::attribution::SharedAttributionCallback;
use crate::retry::{DEFAULT_MAX_RETRIES, RATE_LIMIT_RETRY_THRESHOLD};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum AuthScheme {
    #[default]
    Bearer,
    XApiKey,
}

/// Set by the shell: `Zstd` only toward the cli-chat-proxy that advertised it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RequestCompression {
    #[default]
    None,
    Zstd,
}

/// All knobs that control a single sampling request.
/// Auth is selected separately via `auth_scheme`, while `api_backend` controls only the request/response protocol shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerConfig {
    pub api_key: Option<String>,
    pub base_url: String,
    /// Resolved local directory for this model's mTLS client identity.
    #[serde(default)]
    pub mtls_cert_dir: Option<PathBuf>,
    pub model: String,
    pub max_completion_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub api_backend: ApiBackend,
    #[serde(default)]
    pub auth_scheme: AuthScheme,
    #[serde(default)]
    pub request_compression: RequestCompression,
    /// Extra request headers applied verbatim. The sampler never inspects the URL to derive headers.
    /// Callers (the session) inject proxy auth and other access headers here before constructing the config.
    pub extra_headers: IndexMap<String, String>,
    /// Additional Responses API `include` values not represented by the typed client.
    #[serde(default)]
    pub extra_response_includes: Vec<String>,
    /// Query parameters folded into every request URL (percent-encoded).
    #[serde(default)]
    pub query_params: IndexMap<String, String>,
    /// Header name to environment variable, resolved into request headers at client build and never persisted.
    #[serde(default)]
    pub env_http_headers: IndexMap<String, String>,
    /// Total context window size in tokens.
    /// The sampler does not enforce it; the session uses it for compaction decisions.
    pub context_window: u64,
    pub force_http1: bool,
    pub max_retries: Option<u32>,
    /// Total-attempt ceiling for rate-limited requests.
    /// `None` keeps the actor's [`RetryPolicy::rate_limit_retry_threshold`].
    #[serde(default)]
    pub rate_limit_retry_threshold: Option<u32>,
    pub stream_tool_calls: bool,
    /// Whether to use a streaming HTTP request for inference. When `false`,
    /// the sampler issues a single non-streaming request and surfaces the
    /// full response at once (no incremental token events). Default `true`
    /// (streaming); set to `false` via the model config to disable streaming.
    #[serde(default = "default_stream")]
    pub stream: bool,
    /// Whether Responses API system messages should be sent through the
    /// top-level `instructions` field instead of `input` system messages.
    #[serde(default)]
    pub responses_system_prompt_as_instructions: bool,
    pub idle_timeout_secs: Option<u64>,
    /// Overall request timeout in seconds. Covers the full request for
    /// non-streaming (send + body) and the response-header wait for
    /// streaming; the L2 idle timeout covers inter-chunk gaps separately.
    /// `None` falls back to [`DEFAULT_REQUEST_TIMEOUT_SECS`] (300s).
    pub request_timeout_secs: Option<u64>,

    // Reasoning effort
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Overrides the Responses API `reasoning.summary` the request builder sets; `None` leaves it as built.
    #[serde(default)]
    pub reasoning_summary: Option<ReasoningSummary>,

    // Client identity
    pub origin_client: Option<OriginClientInfo>,
    pub client_identifier: Option<String>,
    pub deployment_id: Option<String>,
    pub user_id: Option<String>,
    /// Stable root conversation identifier emitted as `x-grok-conv-group-id`.
    #[serde(default)]
    pub conversation_group_id: Option<ConversationGroupId>,
    pub client_version: Option<String>,

    /// Hook invoked on every 401 response with the bearer that was actually sent on the wire.
    /// Implementations typically compare it against a live credential source to tell a stale token from a server-rejected live one.
    /// `None` (default) is a no-op; the 401 arm still returns `SamplingError::Auth`.
    #[serde(skip)]
    pub attribution_callback: Option<SharedAttributionCallback>,

    /// Resolves a fresh bearer for each request. `None` uses the construction-time `api_key`.
    #[serde(skip)]
    pub bearer_resolver: Option<SharedBearerResolver>,

    /// Whether the session can reactively refresh credentials on a 401
    /// (OAuth session-token refresh, `auth_provider` re-mint, or devbox
    /// re-mint). Set by the session when it builds the per-request config.
    ///
    /// When `true`, the sampler emits the first 401 of a request to the
    /// session so it can refresh once. When `false` (static BYOK / api-key
    /// auth with no refresh mechanism), a 401 is retried in-loop with
    /// exponential backoff like every other HTTP error code, up to the
    /// time budget. `SamplingError::Auth` (client-side construction
    /// failure) is always emitted to the session regardless of this flag.
    #[serde(skip)]
    pub auth_refresh_available: bool,

    #[serde(default)]
    pub supports_backend_search: bool,

    /// Per-model config for the `x-compactions-remaining` header; `None` disables it.
    #[serde(default)]
    pub compactions_remaining: Option<CompactionsRemaining>,

    /// Per-model config for the `x-compaction-at` header; `None` disables it.
    #[serde(default)]
    pub compaction_at_tokens: Option<CompactionAtTokens>,

    /// Server-side doom-loop check policy; `None` disables it.
    /// It also absorbs the reported trigger events (unlike environment headers in [`Self::extra_headers`], this gates the client's decode behavior).
    #[serde(default)]
    pub doom_loop_recovery: Option<DoomLoopRecoveryPolicy>,

    /// Per-request header injector (e.g. OTel traceparent). Called in `post()`.
    #[serde(skip)]
    pub header_injector: Option<SharedHeaderInjector>,
}

impl Default for SamplerConfig {
    /// Empty defaults so callers can use `..Default::default()` and new fields don't ripple through every literal site.
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: String::new(),
            mtls_cert_dir: None,
            model: String::new(),
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            api_backend: ApiBackend::default(),
            auth_scheme: AuthScheme::default(),
            request_compression: RequestCompression::default(),
            extra_headers: IndexMap::new(),
            extra_response_includes: Vec::new(),
            query_params: IndexMap::new(),
            env_http_headers: IndexMap::new(),
            context_window: 0,
            force_http1: false,
            max_retries: None,
            rate_limit_retry_threshold: None,
            stream_tool_calls: false,
            stream: true,
            responses_system_prompt_as_instructions: false,
            idle_timeout_secs: None,
            request_timeout_secs: None,
            reasoning_effort: None,
            reasoning_summary: None,
            origin_client: None,
            client_identifier: None,
            deployment_id: None,
            user_id: None,
            conversation_group_id: None,
            client_version: None,
            attribution_callback: None,
            bearer_resolver: None,
            auth_refresh_available: false,
            supports_backend_search: false,
            compactions_remaining: None,
            compaction_at_tokens: None,
            doom_loop_recovery: None,
            header_injector: None,
        }
    }
}

/// Cheap sync read of the current bearer for [`SamplerConfig::bearer_resolver`].
pub trait BearerResolver: Send + Sync + std::fmt::Debug {
    fn current_bearer(&self) -> Option<String>;

    /// Awaited by the client right before it stamps a request; [`Self::current_bearer`] is read afterwards.
    /// A resolver that can renew its bearer does so here when the cached one would not survive the send, so the request never leaves with no credential.
    /// Default: no-op.
    fn prepare_for_send(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async {})
    }
}

pub type SharedBearerResolver = std::sync::Arc<dyn BearerResolver>;

/// Host trace hooks for the per-attempt HTTP span; the sampler has no OpenTelemetry dependency.
pub trait HeaderInjector: Send + Sync + std::fmt::Debug {
    fn inject(&self, headers: &mut reqwest::header::HeaderMap);

    /// Runs right after each streaming HTTP span is created and before it has children.
    /// Default: no-op.
    fn set_span_parent(&self, _span: &tracing::Span, _traceparent: &str) {}
}

pub type SharedHeaderInjector = std::sync::Arc<dyn HeaderInjector>;

/// Retry knobs for the sampler's internal transport-error retry loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of retries before giving up. Acts as a safety net;
    /// the primary cap is [`Self::max_retry_duration_secs`].
    pub max_retries: u32,
    /// Total-attempt ceiling for rate-limited requests before escalating to the caller.
    /// Lower than `max_retries` because rate-limit waits can be long.
    ///
    /// Deprecated: 429 now retries within the same time budget as every
    /// other HTTP code; this field is retained for serde/config
    /// compatibility and no longer caps 429 retries.
    pub rate_limit_retry_threshold: u32,
    #[serde(default)]
    pub retry_only_before_output: bool,
    /// Total time budget for the retry loop, in seconds. Once the elapsed
    /// time since the first failure exceeds this, no further retries are
    /// attempted and the last error becomes Fatal. Default 600 (10 min).
    #[serde(default = "default_max_retry_duration_secs")]
    pub max_retry_duration_secs: u64,
}

/// Default total retry time budget: 10 minutes.
pub const DEFAULT_MAX_RETRY_DURATION_SECS: u64 = 600;

/// Default overall request timeout in seconds. Covers the full request
/// lifecycle for non-streaming (send + body read) and the response-header
/// wait for streaming. The L2 per-chunk idle timeout
/// ([`SamplerConfig::idle_timeout_secs`]) covers inter-chunk gaps
/// separately. Default 300s (5 min), matching the default idle timeout.
pub const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 300;

fn default_max_retry_duration_secs() -> u64 {
    DEFAULT_MAX_RETRY_DURATION_SECS
}

/// Default for [`SamplerConfig::stream`]: streaming enabled.
fn default_stream() -> bool {
    true
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: DEFAULT_MAX_RETRIES,
            rate_limit_retry_threshold: RATE_LIMIT_RETRY_THRESHOLD,
            retry_only_before_output: false,
            max_retry_duration_secs: DEFAULT_MAX_RETRY_DURATION_SECS,
        }
    }
}

/// Identity of the client that originated the request, used for User-Agent rendering.
/// The shell layer composes this with platform info into a final UA string.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OriginClientInfo {
    pub product: String,
    pub version: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Configs serialized before the field existed must keep deserializing.
    #[test]
    fn config_without_doom_loop_recovery_deserializes_to_none() {
        let mut stripped = serde_json::to_value(SamplerConfig::default()).unwrap();
        let object = stripped.as_object_mut().unwrap();
        object.remove("doom_loop_recovery");
        object.remove("extra_response_includes");
        object.remove("mtls_cert_dir");
        object.remove("rate_limit_retry_threshold");
        let config: SamplerConfig = serde_json::from_value(stripped).unwrap();
        assert!(config.doom_loop_recovery.is_none());
        assert!(config.extra_response_includes.is_empty());
        assert!(config.mtls_cert_dir.is_none());
        assert!(config.rate_limit_retry_threshold.is_none());

        let with_policy = SamplerConfig {
            doom_loop_recovery: Some(DoomLoopRecoveryPolicy {
                max_threshold: 8,
                max_retries: 2,
                ..Default::default()
            }),
            ..Default::default()
        };
        let round_tripped: SamplerConfig =
            serde_json::from_value(serde_json::to_value(&with_policy).unwrap()).unwrap();
        assert_eq!(
            round_tripped.doom_loop_recovery,
            with_policy.doom_loop_recovery
        );
    }
}
