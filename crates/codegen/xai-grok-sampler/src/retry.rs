//! Retry classification, backoff, and decision-making.
//!
//! Pure logic only: no I/O, no notifications, no logging side-effects.
//! The actor (M4) wraps this with the actual retry loop.
//!
//! # Retry behavior summary
//!
//! **Retried** — every HTTP status code is retried with exponential
//! backoff (2s, 4s, 8s, ..., capped 30s) up to a total time budget
//! (default 10 min, see [`crate::config::RetryPolicy::max_retry_duration`]):
//! - 4xx (400, 403, 404, 408, 422, ...), 429, 5xx (500, 502, 503, 504, 520)
//! - Connection errors (timeout, refused, reset)
//! - `EventStreamError` / `StreamError` (mid-stream failures)
//! - `EmptyResponse` (model returned no content/tool calls)
//! - `IdleTimeout` (model stuck — a fresh sample may complete)
//! - Context-window overflow and `x-should-retry: false` responses are
//!   retried too — the time budget bounds the cost, and a transient cause
//!   behind what looks like a deterministic failure can still clear.
//!
//! **Auth path** (not counted against the in-loop retry budget):
//! - 401 Unauthorized / `Auth` → emitted to the session, which refreshes
//!   credentials and resubmits (itself a retry). The session-level refresh
//!   is preferable to burning the time budget re-sending an expired token.
//! - Encrypted-content mismatch (`encrypted_content` in a 400) → emitted to
//!   the session for immediate user feedback (retrying cannot decrypt it).
//!
//! **Special handling** (not counted against retry budget):
//! - 413 / image processing errors → strip images and retry
//!
//! **Not retried** (Fatal immediately, non-HTTP-code deterministic errors):
//! - `InvalidConfiguration` (config issue)
//! - `Serialization` (response parsing failure)
//! - `MaxTokensTruncation` (by design)
//!
//! `RATE_LIMIT_RETRY_THRESHOLD` is retained for config compatibility but no
//! longer caps 429 retries — 429 now retries within the same time budget as
//! every other HTTP code (honoring `Retry-After` when present).

use std::time::Duration;

use xai_grok_sampling_types::SamplingError;

/// Legacy rate-limit (429) retry cap. Kept for config/serde compatibility;
/// no longer used to cap 429 retries — 429 now retries within the same
/// time budget as every other HTTP status code.
pub const RATE_LIMIT_RETRY_THRESHOLD: u32 = 2;

/// Default max retries when no env or model override is set.
///
/// This count is a safety net only — the real cap is the time budget in
/// [`crate::config::RetryPolicy::max_retry_duration`] (default 10 min). 30
/// retries with the 30s backoff cap allow ~14 min of count budget, so the
/// time budget (10 min) trips first under default configuration. Users who
/// want a smaller count-based cap can set `GROK_MAX_RETRIES`.
pub const DEFAULT_MAX_RETRIES: u32 = 30;

/// Resolve max API retries from an optional env override, model config,
/// or default ([`DEFAULT_MAX_RETRIES`]).
pub(crate) fn resolve_max_retries_with_env(
    env_override: Option<&str>,
    model_max_retries: Option<u32>,
) -> u32 {
    env_override
        .and_then(|value| value.parse::<u32>().ok())
        .or(model_max_retries)
        .unwrap_or(DEFAULT_MAX_RETRIES)
}

/// Resolve max API retries: `GROK_MAX_RETRIES` env > model config > default ([`DEFAULT_MAX_RETRIES`]).
pub fn resolve_max_retries(model_max_retries: Option<u32>) -> u32 {
    let env_override = std::env::var("GROK_MAX_RETRIES").ok();
    resolve_max_retries_with_env(env_override.as_deref(), model_max_retries)
}

/// Backoff for doom-loop resamples: near-immediate with a small jitter.
/// Loops are stochastic at sampling temperature, so a fresh sample is the
/// remedy — waiting buys nothing beyond de-syncing concurrent resamples.
pub fn doom_loop_backoff(retry_count: u32) -> Duration {
    use std::hash::{Hash, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};

    static JITTER_SEQ: AtomicU64 = AtomicU64::new(0);

    let mut hasher = std::hash::DefaultHasher::new();
    JITTER_SEQ.fetch_add(1, Ordering::Relaxed).hash(&mut hasher);
    retry_count.hash(&mut hasher);
    Duration::from_millis(hasher.finish() % 251)
}

/// Exponential backoff (2s, 4s, 8s, ..., capped 30s) with +/-20% jitter
/// to prevent thundering-herd retry storms.
pub fn retry_backoff_with_jitter(retry_count: u32) -> Duration {
    use std::hash::{Hash, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};

    static JITTER_SEQ: AtomicU64 = AtomicU64::new(0);

    let shift = retry_count.saturating_sub(1);
    let base_ms = 2000u64.checked_shl(shift).unwrap_or(u64::MAX).min(30_000);
    let jitter_range = base_ms / 5;
    let mut hasher = std::hash::DefaultHasher::new();
    JITTER_SEQ.fetch_add(1, Ordering::Relaxed).hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    let jitter = hasher.finish() % (jitter_range * 2 + 1);
    Duration::from_millis(base_ms - jitter_range + jitter)
}

/// What the actor should do next given a sampling error and retry context.
///
/// Pure data: callers (the actor's per-request task) are responsible for
/// performing the actual sleep, image strip, client rebuild, or emit.
#[derive(Debug)]
pub enum RetryDecision {
    /// Retry with exponential backoff (transport errors, 5xx,
    /// empty responses).
    Retry { backoff: Duration },

    /// Retry honoring the server's `Retry-After` header (429 rate
    /// limits). `is_rate_limited` distinguishes 429s from generic
    /// retry-with-backoff cases for telemetry.
    RetryWithBackoff {
        backoff: Duration,
        is_rate_limited: bool,
    },

    /// Retry after stripping inline images from the request (413
    /// Payload Too Large or image processing rejection).
    RetryWithImageStrip,

    /// Retry after rebuilding the HTTP client with HTTP/1.1 (transport
    /// error, first retry only).
    RetryWithClientRebuild { backoff: Duration },

    /// Emit the error to the session and let it decide what to do
    /// (auth refresh, encrypted-content mismatch).
    EmitToSession(SamplingError),

    /// Fatal: no further retries possible. Surface to the caller as the
    /// final outcome of the sampling request.
    Fatal(SamplingError),
}

/// Classify a sampling error into a [`RetryDecision`].
///
/// `retry_count` is the number of retries already performed (0 on first
/// failure). `max_retries` is the count-based safety-net budget; the
/// actor's retry loop additionally enforces a total time budget (see
/// [`crate::config::RetryPolicy::max_retry_duration_secs`]).
///
/// The function is pure: it does not sleep, log, or perform I/O.
///
/// Note: a *server* 401 (`Api { status: 401 }`) is intentionally NOT
/// short-circuited here. It falls through to the generic retry-with-backoff
/// arm, retrying within the time budget like every other HTTP error code.
/// The first-401 "give the session one refresh chance" interception for
/// refresh-capable models lives in [`crate::actor::request_task::apply_retry_decision`],
/// which has access to [`SamplerConfig::auth_refresh_available`]; static-BYOK
/// models (no refresh mechanism) retry in-loop instead.
pub fn classify_error(err: &SamplingError, retry_count: u32, max_retries: u32) -> RetryDecision {
    if err.is_encrypted_content_error() {
        return RetryDecision::EmitToSession(clone_error(err));
    }
    if max_retries == 0 {
        return RetryDecision::Fatal(clone_error(err));
    }

    // 413 Payload Too Large: strip inline images and try once. The
    // caller checks if there are images left after the strip; if not,
    // upgrade to Fatal.
    if err.is_payload_too_large() {
        return RetryDecision::RetryWithImageStrip;
    }

    // Image processing errors (direct 400 or proxy-wrapped 500): strip
    // images and retry, same recovery as 413.
    if err.is_image_processing_error() {
        return RetryDecision::RetryWithImageStrip;
    }

    // Note: `x-should-retry: false` and context-window overflow errors
    // are intentionally NOT short-circuited to Fatal here. Per the fork's
    // retry policy, every HTTP error code is retried up to the time budget
    // (default 10 min); only the count budget (max_retries) or the time
    // budget in the actor's retry loop can make them Fatal. A transient
    // cause behind what looks like a deterministic failure can still clear,
    // and the time budget bounds the cost.

    // Doom-loop failures: always Retry with near-immediate backoff. The
    // recovery loop intercepts these BEFORE classification and runs its own
    // budget (`policy.max_retries`, enforced by disarming the abort); this
    // arm only keeps classification total so a stray doom failure through
    // any other path can never be Fatal.
    if matches!(err, SamplingError::DoomLoopDetected { .. }) {
        return RetryDecision::Retry {
            backoff: doom_loop_backoff(retry_count + 1),
        };
    }

    // Rate-limited (429): retry within the same budget as every other
    // HTTP code. No special cap — the actor's time budget bounds total
    // wait. `RetryWithBackoff` preserves the `is_rate_limited` flag for
    // telemetry and honors the server's `Retry-After` when present.
    if err.is_rate_limited() {
        let next_attempt = retry_count + 1;
        if max_retries == 0 || next_attempt >= max_retries {
            return RetryDecision::Fatal(clone_error(err));
        }
        let backoff = err
            .retry_after()
            .map(Duration::from_secs)
            .unwrap_or_else(|| retry_backoff_with_jitter(next_attempt));
        return RetryDecision::RetryWithBackoff {
            backoff,
            is_rate_limited: true,
        };
    }

    // Generic HTTP and transport errors. Under the time-budget policy,
    // EVERY HTTP status code is retryable. We explicitly handle the
    // error types where `is_retryable()` returns false but should still
    // be retried:
    // - Auth errors (client-side bearer or server 401 lifted by HTTP client)
    // - IdleTimeout (model stuck - a fresh sample may complete)
    // - Api errors with non-retryable status codes (400, 403, 404, etc.)
    // First retry rebuilds the HTTP client with HTTP/1.1 to escape
    // poisoned HTTP/2 pools; later retries just back off.
    let is_http_or_transport_error = matches!(
        err,
        SamplingError::Auth { .. }
            | SamplingError::Http(_)
            | SamplingError::Api { .. }
            | SamplingError::IdleTimeout { .. }
    );

    if err.is_retryable() || is_http_or_transport_error {
        let next_attempt = retry_count + 1;
        if max_retries == 0 || next_attempt >= max_retries {
            return RetryDecision::Fatal(clone_error(err));
        }
        let backoff = err
            .retry_after()
            .map(Duration::from_secs)
            .unwrap_or_else(|| retry_backoff_with_jitter(next_attempt));
        if next_attempt == 1 {
            return RetryDecision::RetryWithClientRebuild { backoff };
        }
        return RetryDecision::Retry { backoff };
    }

    // Everything else is fatal.
    RetryDecision::Fatal(clone_error(err))
}

/// Build a human-readable, telemetry-friendly description of a sampling
/// error.
///
/// `retry_count`, when present, is rendered as a "Request failed after
/// N retries." prefix. The function is pure string formatting: no
/// logging, no I/O, no allocation beyond the produced `String`.
pub fn format_sampling_error(err: &SamplingError, retry_count: Option<u32>) -> String {
    let retry_prefix = match retry_count {
        Some(count) => format!("Request failed after {} retries. ", count),
        None => String::new(),
    };

    match err {
        SamplingError::Auth { message, .. } => {
            format!(
                "{}Authentication failed: {}. Please check your API key configuration.",
                retry_prefix, message
            )
        }
        SamplingError::InvalidConfiguration(msg) => {
            format!(
                "{}Invalid configuration: {}. Please check your model settings.",
                retry_prefix, msg
            )
        }

        SamplingError::Http(e) => {
            let mut details = Vec::new();
            if e.is_timeout() {
                details.push("timeout".to_string());
            }
            if e.is_connect() {
                details.push("connection failed".to_string());
            }
            if let Some(status) = e.status() {
                details.push(format!("status {}", status));
            }
            if let Some(url) = e.url() {
                details.push(format!("url: {}", url));
            }
            let detail_str = if details.is_empty() {
                e.to_string()
            } else {
                format!("{} ({})", e, details.join(", "))
            };
            format!(
                "{}HTTP request failed: {}. This may be a network issue or the API endpoint may be unavailable.",
                retry_prefix, detail_str
            )
        }
        SamplingError::Serialization(e) => {
            format!(
                "{}Failed to parse API response at line {} column {}: {}. This indicates an unexpected response format from the server.",
                retry_prefix,
                e.line(),
                e.column(),
                e
            )
        }
        SamplingError::Api {
            status, message, ..
        } => {
            let status_hint = match status.as_u16() {
                400 => " (bad request - check your input)",
                401 | 403 => " (authentication issue - check your API key)",
                404 => " (endpoint not found - check model configuration)",
                413 => " (request too large - try /compact or start new session)",
                429 => " (rate limited - please wait and retry)",
                500 => " (server internal error)",
                #[allow(clippy::manual_range_patterns)]
                502 | 503 | 504 => " (server unavailable - please retry)",
                _ => "",
            };
            format!(
                "{}API error (HTTP {}{}): {}",
                retry_prefix,
                status.as_u16(),
                status_hint,
                message
            )
        }
        SamplingError::EventStreamError(msg) => {
            format!(
                "{}Event stream error: {}. The connection to the server was interrupted.",
                retry_prefix, msg
            )
        }
        SamplingError::StreamError {
            error_type,
            message,
            ..
        } => {
            format!(
                "{}Server stream error ({}): {}. The server encountered an error while streaming the response.",
                retry_prefix, error_type, message
            )
        }
        SamplingError::IdleTimeout { elapsed_secs } => {
            format!(
                "{}Model stopped responding after {}s. The model may be overloaded or stuck. Try again or use a different model.",
                retry_prefix, elapsed_secs
            )
        }
        SamplingError::EmptyResponse { context } => {
            format!(
                "{}Empty response from model ({}): model={}, had_reasoning={}, finish_reason={}, completion_tokens={}",
                retry_prefix,
                context.reason,
                context.model,
                context.had_reasoning,
                context.finish_reason_str(),
                context.completion_tokens.unwrap_or(0),
            )
        }
        SamplingError::MaxTokensTruncation => {
            format!("{}Response truncated by max_tokens.", retry_prefix)
        }
        SamplingError::DoomLoopDetected { triggers, .. } => {
            format!(
                "{}Server detected a reasoning loop ({}); resampling the response.",
                retry_prefix,
                triggers.join(", ")
            )
        }
    }
}

/// Reconstruct an owned [`SamplingError`] from a borrowed one.
///
/// `SamplingError` does not implement `Clone` because its `Http` and
/// `Serialization` variants wrap non-`Clone` types. The retry loop
/// only borrows the error during classification, then needs to surface
/// it; this helper produces a faithful copy where possible. `Http`
/// falls back to a structured `EventStreamError` (still retryable, like
/// the original transport error). `Serialization` must stay
/// `Serialization`: laundering it into `EventStreamError` would flip a
/// fatal response-parse failure into a retryable one and burn the full
/// retry budget re-generating a response that fails the same way.
pub(crate) fn clone_error(err: &SamplingError) -> SamplingError {
    match err {
        SamplingError::Auth {
            message,
            credential,
        } => SamplingError::Auth {
            message: message.clone(),
            credential: *credential,
        },
        SamplingError::InvalidConfiguration(msg) => SamplingError::InvalidConfiguration(msg),
        SamplingError::Http(e) => {
            // reqwest::Error is not Clone; preserve the rendered message
            // as an EventStreamError (the closest retryable transport
            // variant) so callers see an equivalent description.
            SamplingError::EventStreamError(e.to_string())
        }
        SamplingError::Serialization(e) => {
            // serde_json::Error is not Clone; its Display already carries the
            // original line/column exactly once.
            SamplingError::serialization_message(e)
        }
        SamplingError::Api {
            status,
            message,
            model_metadata,
            retry_after_secs,
            should_retry,
            error_code,
        } => SamplingError::Api {
            status: *status,
            message: message.clone(),
            model_metadata: model_metadata.clone(),
            retry_after_secs: *retry_after_secs,
            should_retry: *should_retry,
            error_code: error_code.clone(),
        },
        SamplingError::EventStreamError(msg) => SamplingError::EventStreamError(msg.clone()),
        SamplingError::StreamError {
            error_type,
            message,
            code,
        } => SamplingError::StreamError {
            error_type: error_type.clone(),
            message: message.clone(),
            code: code.clone(),
        },
        SamplingError::IdleTimeout { elapsed_secs } => SamplingError::IdleTimeout {
            elapsed_secs: *elapsed_secs,
        },
        SamplingError::EmptyResponse { context } => SamplingError::EmptyResponse {
            context: context.clone(),
        },
        SamplingError::MaxTokensTruncation => SamplingError::MaxTokensTruncation,
        SamplingError::DoomLoopDetected {
            triggers,
            aborted_at_chunk,
        } => SamplingError::DoomLoopDetected {
            triggers: triggers.clone(),
            aborted_at_chunk: *aborted_at_chunk,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::StatusCode;
    use xai_grok_sampling_types::ApiErrorCode;

    fn api_err(status: StatusCode, message: &str) -> SamplingError {
        SamplingError::Api {
            status,
            message: message.to_string(),
            model_metadata: None,
            retry_after_secs: None,
            should_retry: None,
            error_code: None,
        }
    }

    fn api_err_with_retry_after(status: StatusCode, retry_after: u64) -> SamplingError {
        SamplingError::Api {
            status,
            message: "x".to_string(),
            model_metadata: None,
            retry_after_secs: Some(retry_after),
            should_retry: None,
            error_code: None,
        }
    }

    #[test]
    fn resolve_max_retries_env_override_takes_precedence() {
        assert_eq!(resolve_max_retries_with_env(Some("9"), Some(3)), 9);
    }

    #[test]
    fn resolve_max_retries_falls_back_to_model() {
        assert_eq!(resolve_max_retries_with_env(None, Some(7)), 7);
    }

    #[test]
    fn resolve_max_retries_default() {
        assert_eq!(
            resolve_max_retries_with_env(None, None),
            DEFAULT_MAX_RETRIES
        );
    }

    #[test]
    fn resolve_max_retries_invalid_env_falls_through() {
        assert_eq!(resolve_max_retries_with_env(Some("abc"), Some(4)), 4);
    }

    #[test]
    fn backoff_first_retry_is_around_two_seconds() {
        let backoff = retry_backoff_with_jitter(1);
        // Base 2000ms +/- 20% jitter (400ms range).
        assert!(
            backoff >= Duration::from_millis(1600) && backoff <= Duration::from_millis(2400),
            "first retry backoff out of range: {:?}",
            backoff
        );
    }

    #[test]
    fn backoff_doubles_then_caps_at_thirty_seconds() {
        // retry_count=2: base 4s
        let r2 = retry_backoff_with_jitter(2);
        assert!(r2 >= Duration::from_millis(3200) && r2 <= Duration::from_millis(4800));

        // retry_count=10: base would be 2^10 * 2000 = 2.048s but capped to 30s
        let r10 = retry_backoff_with_jitter(10);
        assert!(r10 >= Duration::from_millis(24_000) && r10 <= Duration::from_millis(36_000));
    }

    #[test]
    fn backoff_zero_retry_count_is_well_defined() {
        // retry_count = 0 corresponds to "before the first retry"; ensure
        // it does not panic and stays in the lowest backoff bucket.
        let backoff = retry_backoff_with_jitter(0);
        assert!(backoff >= Duration::from_millis(1600) && backoff <= Duration::from_millis(2400));
    }

    #[test]
    fn classify_auth_error_is_retryable_in_pure_classifier() {
        // `SamplingError::Auth` (client-side bearer failure OR server 401
        // lifted by the HTTP client) is retryable in the pure classifier.
        // Refresh-capable models get the first 401 intercepted in
        // `apply_retry_decision` (emitted to the session for a one-shot
        // refresh); static-BYOK models retry here in-loop with backoff.
        let err = SamplingError::auth_unknown("bad token");
        match classify_error(&err, 0, 5) {
            RetryDecision::RetryWithClientRebuild { .. } => {}
            other => panic!("expected RetryWithClientRebuild for Auth, got {other:?}"),
        }
    }

    #[test]
    fn classify_unauthorized_is_retryable_in_pure_classifier() {
        // A server 401 is no longer short-circuited to EmitToSession in the
        // pure classifier; it falls through to the generic retry-with-backoff
        // arm (the first-401 "refresh once" interception for refresh-capable
        // models lives in `apply_retry_decision`, which has access to
        // `SamplerConfig::auth_refresh_available`). Static-BYOK models thus
        // retry 401 in-loop; refresh-capable models get one refresh chance.
        let err = api_err(StatusCode::UNAUTHORIZED, "no");
        match classify_error(&err, 0, 5) {
            RetryDecision::RetryWithClientRebuild { .. } => {}
            other => panic!("expected RetryWithClientRebuild for 401, got {other:?}"),
        }
    }

    #[test]
    fn classify_encrypted_content_emits_to_session() {
        let err = api_err(
            StatusCode::BAD_REQUEST,
            "Could not decrypt the provided encrypted_content",
        );
        match classify_error(&err, 0, 5) {
            RetryDecision::EmitToSession(_) => {}
            other => panic!("expected EmitToSession, got {other:?}"),
        }
    }

    #[test]
    fn classify_payload_too_large_strips_images() {
        let err = api_err(StatusCode::PAYLOAD_TOO_LARGE, "too big");
        assert!(matches!(
            classify_error(&err, 0, 5),
            RetryDecision::RetryWithImageStrip
        ));
    }

    #[test]
    fn classify_image_processing_error_400_strips_images() {
        let err = api_err(StatusCode::BAD_REQUEST, "Could not process image");
        assert!(matches!(
            classify_error(&err, 0, 5),
            RetryDecision::RetryWithImageStrip
        ));
    }

    #[test]
    fn classify_image_processing_error_500_wrapped_strips_images() {
        let err = api_err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "upstream: 400 Bad Request: Could not process image",
        );
        assert!(matches!(
            classify_error(&err, 0, 5),
            RetryDecision::RetryWithImageStrip
        ));
    }

    #[test]
    fn classify_image_processing_error_takes_priority_over_5xx_retry() {
        // A 500 wrapping "Could not process image" is retryable by status
        // code alone — verify the image-processing guard intercepts first.
        let err = api_err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Could not process image: bad format",
        );
        assert!(
            err.is_retryable(),
            "500 is retryable without the image-processing guard"
        );
        assert!(matches!(
            classify_error(&err, 0, 5),
            RetryDecision::RetryWithImageStrip
        ));
    }

    #[test]
    fn classify_image_400_strips_even_with_should_retry_false() {
        // The server stamps invalid-image 400s with `x-should-retry: false`;
        // that verdict only covers resending the SAME payload, and the strip
        // retry sends a different one — the code must beat the header veto.
        let err = SamplingError::Api {
            status: StatusCode::BAD_REQUEST,
            message: "some future wording without the legacy phrase".into(),
            model_metadata: None,
            retry_after_secs: None,
            should_retry: Some(false),
            error_code: Some(ApiErrorCode::InvalidImage),
        };
        assert!(matches!(
            classify_error(&err, 0, 5),
            RetryDecision::RetryWithImageStrip
        ));
    }

    #[test]
    fn classify_image_stream_error_strips_instead_of_blind_retry() {
        // Coded mid-stream image rejections must strip immediately; a
        // code-less stream error keeps the ordinary retry path.
        let err = SamplingError::StreamError {
            error_type: "invalid_request_error".into(),
            message: "Base64 string of provided image cannot be decoded.".into(),
            code: Some(ApiErrorCode::InvalidImage),
        };
        assert!(matches!(
            classify_error(&err, 0, 5),
            RetryDecision::RetryWithImageStrip
        ));

        let unrelated = SamplingError::StreamError {
            error_type: "overloaded_error".into(),
            message: "The server is overloaded.".into(),
            code: None,
        };
        assert!(!matches!(
            classify_error(&unrelated, 0, 5),
            RetryDecision::RetryWithImageStrip
        ));
    }

    #[test]
    fn classify_rate_limited_uses_retry_after() {
        let err = api_err_with_retry_after(StatusCode::TOO_MANY_REQUESTS, 7);
        match classify_error(&err, 0, 5) {
            RetryDecision::RetryWithBackoff {
                backoff,
                is_rate_limited,
            } => {
                assert!(is_rate_limited);
                assert_eq!(backoff, Duration::from_secs(7));
            }
            other => panic!("expected RetryWithBackoff, got {other:?}"),
        }
    }

    #[test]
    fn classify_rate_limited_retries_within_budget_not_threshold() {
        let err = api_err(StatusCode::TOO_MANY_REQUESTS, "slow");
        // 429 is no longer capped at a low threshold; it retries like any
        // other HTTP code up to max_retries. retry_count=1, max_retries=5
        // -> next_attempt=2 < 5 -> RetryWithBackoff.
        match classify_error(&err, 1, 5) {
            RetryDecision::RetryWithBackoff {
                is_rate_limited, ..
            } => {
                assert!(is_rate_limited);
            }
            other => panic!("expected RetryWithBackoff, got {other:?}"),
        }
    }

    #[test]
    fn classify_rate_limited_exhausts_at_max_retries() {
        let err = api_err(StatusCode::TOO_MANY_REQUESTS, "slow");
        // retry_count=4, max_retries=5 -> next_attempt=5 >= 5 -> Fatal
        // (count budget exhausted, same as every other HTTP code).
        match classify_error(&err, 4, 5) {
            RetryDecision::Fatal(SamplingError::Api { status, .. }) => {
                assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
            }
            other => panic!("expected Fatal at budget, got {other:?}"),
        }
    }

    #[test]
    fn zero_retry_budget_never_reuses_a_model_output_cap() {
        for err in [
            api_err(StatusCode::INTERNAL_SERVER_ERROR, "boom"),
            api_err(StatusCode::PAYLOAD_TOO_LARGE, "too big"),
            api_err(StatusCode::BAD_REQUEST, "Could not process image"),
            SamplingError::EmptyResponse {
                context: xai_grok_sampling_types::EmptyResponseContext {
                    reason: xai_grok_sampling_types::EmptyReason::NoVisibleContent,
                    had_reasoning: false,
                    content_len: 0,
                    tool_call_count: 0,
                    finish_reason: Some("stop".into()),
                    completion_tokens: Some(1),
                    reasoning_tokens: Some(0),
                    prompt_tokens: Some(10),
                    model: "m".into(),
                    first_choice_seen: true,
                },
            },
        ] {
            assert!(matches!(
                classify_error(&err, 0, 0),
                RetryDecision::Fatal(_)
            ));
        }
    }

    #[test]
    fn classify_5xx_first_retry_rebuilds_client() {
        let err = api_err(StatusCode::INTERNAL_SERVER_ERROR, "boom");
        match classify_error(&err, 0, 5) {
            RetryDecision::RetryWithClientRebuild { backoff } => {
                assert!(backoff >= Duration::from_millis(1600));
            }
            other => panic!("expected RetryWithClientRebuild, got {other:?}"),
        }
    }

    #[test]
    fn classify_5xx_subsequent_retry_uses_plain_retry() {
        let err = api_err(StatusCode::BAD_GATEWAY, "boom");
        match classify_error(&err, 1, 5) {
            RetryDecision::Retry { backoff } => {
                assert!(backoff >= Duration::from_millis(3200));
            }
            other => panic!("expected Retry, got {other:?}"),
        }
    }

    #[test]
    fn classify_5xx_exhausted_retries_is_fatal() {
        let err = api_err(StatusCode::SERVICE_UNAVAILABLE, "boom");
        match classify_error(&err, 4, 5) {
            RetryDecision::Fatal(SamplingError::Api { .. }) => {}
            other => panic!("expected Fatal, got {other:?}"),
        }
    }

    #[test]
    fn classify_event_stream_error_is_retryable() {
        let err = SamplingError::EventStreamError("connection reset".into());
        match classify_error(&err, 0, 5) {
            RetryDecision::RetryWithClientRebuild { .. } => {}
            other => panic!("expected RetryWithClientRebuild, got {other:?}"),
        }
    }

    #[test]
    fn classify_stream_error_is_retryable() {
        let err = SamplingError::StreamError {
            error_type: "transient".into(),
            message: "x".into(),
            code: None,
        };
        match classify_error(&err, 0, 5) {
            RetryDecision::RetryWithClientRebuild { .. } => {}
            other => panic!("expected RetryWithClientRebuild for StreamError, got {other:?}"),
        }
    }

    #[test]
    fn classify_idle_timeout_is_retryable() {
        let err = SamplingError::IdleTimeout { elapsed_secs: 300 };
        // IdleTimeout is now retried (a fresh sample may complete) up to the
        // time budget; the first retry rebuilds the HTTP client like other
        // transport errors.
        match classify_error(&err, 0, 5) {
            RetryDecision::RetryWithClientRebuild { .. } => {}
            other => panic!("expected RetryWithClientRebuild for IdleTimeout, got {other:?}"),
        }
    }

    #[test]
    fn classify_invalid_config_is_fatal() {
        let err = SamplingError::InvalidConfiguration("missing model");
        assert!(matches!(
            classify_error(&err, 0, 5),
            RetryDecision::Fatal(SamplingError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn classify_api_400_non_encrypted_is_retryable() {
        let err = api_err(StatusCode::BAD_REQUEST, "Invalid model parameter");
        // 400 is an HTTP error code and is now retried up to the time budget.
        assert!(matches!(
            classify_error(&err, 0, 5),
            RetryDecision::RetryWithClientRebuild { .. }
        ));
    }

    fn serialization_err() -> SamplingError {
        SamplingError::Serialization(serde_json::from_str::<i32>("not a number").unwrap_err())
    }

    /// Regression: `clone_error` used to launder `Serialization` into the
    /// retryable `EventStreamError`, turning a deterministic parse failure
    /// into a full-budget retry storm.
    #[test]
    fn clone_error_preserves_serialization_and_non_retryability() {
        let cloned = clone_error(&serialization_err());
        assert!(
            matches!(cloned, SamplingError::Serialization(_)),
            "expected Serialization, got {cloned:?}"
        );
        assert!(!cloned.is_retryable());
        assert!(
            cloned.to_string().contains("line 1 column"),
            "original position text must survive the clone: {cloned}"
        );
    }

    /// The tee cell captures mid-stream errors via `clone_error`; dropping
    /// the code there would silently disable mid-stream strip recovery.
    #[test]
    fn clone_error_preserves_stream_error_code() {
        let cloned = clone_error(&SamplingError::StreamError {
            error_type: "invalid_request_error".into(),
            message: "bad image".into(),
            code: Some(ApiErrorCode::InvalidImage),
        });
        let SamplingError::StreamError { code, .. } = &cloned else {
            panic!("expected StreamError, got {cloned:?}");
        };
        assert_eq!(*code, Some(ApiErrorCode::InvalidImage));
        assert!(cloned.is_image_processing_error());
    }

    #[test]
    fn classify_serialization_is_fatal_on_first_attempt() {
        match classify_error(&serialization_err(), 0, 15) {
            RetryDecision::Fatal(SamplingError::Serialization(_)) => {}
            other => panic!("expected Fatal(Serialization) on attempt 1, got {other:?}"),
        }
    }

    #[test]
    fn format_includes_retry_prefix_when_count_present() {
        let err = SamplingError::auth_unknown("bad");
        let s = format_sampling_error(&err, Some(3));
        assert!(s.starts_with("Request failed after 3 retries."));
    }

    #[test]
    fn format_omits_retry_prefix_when_count_absent() {
        let err = SamplingError::auth_unknown("bad");
        let s = format_sampling_error(&err, None);
        assert!(!s.starts_with("Request failed after"));
        assert!(s.starts_with("Authentication failed:"));
    }

    #[test]
    fn format_includes_status_hint_for_known_codes() {
        let err = api_err(StatusCode::PAYLOAD_TOO_LARGE, "big");
        let s = format_sampling_error(&err, None);
        assert!(s.contains("HTTP 413"));
        assert!(s.contains("request too large"));
    }

    #[test]
    fn format_idle_timeout_includes_elapsed_secs() {
        let err = SamplingError::IdleTimeout { elapsed_secs: 240 };
        let s = format_sampling_error(&err, None);
        assert!(s.contains("240s"));
    }

    #[test]
    fn should_retry_false_still_retries_under_time_budget() {
        // The server hint `x-should-retry: false` no longer short-circuits
        // to Fatal; every HTTP code is retried up to the time budget.
        let err = SamplingError::Api {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: "boom".into(),
            model_metadata: None,
            retry_after_secs: None,
            should_retry: Some(false),
            error_code: None,
        };
        assert!(matches!(
            classify_error(&err, 0, 15),
            RetryDecision::RetryWithClientRebuild { .. }
        ));
    }

    #[test]
    fn context_length_overflow_is_retried_under_time_budget() {
        // Context-window overflow is an HTTP error code (500 here) and is
        // now retried up to the time budget like every other code; the
        // time budget in the actor loop bounds the cost.
        let err = SamplingError::Api {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: "none: The prompt is too long for this model's context window.".into(),
            model_metadata: None,
            retry_after_secs: None,
            should_retry: None,
            error_code: None,
        };
        assert!(matches!(
            classify_error(&err, 0, 15),
            RetryDecision::RetryWithClientRebuild { .. }
        ));
    }

    #[test]
    fn should_retry_true_falls_through_to_existing_logic() {
        let err = SamplingError::Api {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: "boom".into(),
            model_metadata: None,
            retry_after_secs: None,
            should_retry: Some(true),
            error_code: None,
        };
        assert!(matches!(
            classify_error(&err, 0, 15),
            RetryDecision::RetryWithClientRebuild { .. }
        ));
    }

    #[test]
    fn should_retry_absent_falls_through() {
        let err = SamplingError::Api {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: "boom".into(),
            model_metadata: None,
            retry_after_secs: None,
            should_retry: None,
            error_code: None,
        };
        assert!(matches!(
            classify_error(&err, 0, 15),
            RetryDecision::RetryWithClientRebuild { .. }
        ));
    }

    #[test]
    fn classify_doom_loop_detected_is_retry_with_immediate_backoff() {
        let err = SamplingError::DoomLoopDetected {
            triggers: vec!["tail_repetition:8@thinking".into()],
            aborted_at_chunk: None,
        };
        // Whatever the counters say, classification is Retry — the recovery
        // loop owns the budget by disarming the abort when it is spent.
        for retry_count in [0, 5, 99] {
            match classify_error(&err, retry_count, 2) {
                RetryDecision::Retry { backoff } => {
                    assert!(backoff <= Duration::from_millis(250), "near-immediate");
                }
                other => panic!("expected Retry, got {other:?}"),
            }
        }
    }

    #[test]
    fn should_retry_false_on_429_still_retries() {
        // 429 with `x-should-retry: false` is still retried up to the
        // time budget (every HTTP code is retried).
        let err = SamplingError::Api {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "rate limited".into(),
            model_metadata: None,
            retry_after_secs: Some(10),
            should_retry: Some(false),
            error_code: None,
        };
        assert!(matches!(
            classify_error(&err, 0, 15),
            RetryDecision::RetryWithBackoff {
                is_rate_limited: true,
                ..
            }
        ));
    }

    #[test]
    fn classify_cloudflare_525_is_retryable_under_time_budget() {
        // Cloudflare 525/526 are now retried like every other HTTP code
        // under the time-budget policy.
        let err = api_err(
            StatusCode::from_u16(525).unwrap(),
            "Secure connection to Grok failed. (HTTP 525).",
        );
        match classify_error(&err, 0, 15) {
            RetryDecision::RetryWithClientRebuild { .. } => {}
            other => panic!("expected RetryWithClientRebuild for 525, got {other:?}"),
        }
    }
}
