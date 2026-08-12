//! Per-request streaming task.
//!
//! Spawned by the actor's `Submit` handler. Owns the retry loop and
//! consumes a Layer 2 stream from the matching backend transform.
//! Cancellation is cooperative via `CancellationToken`.

use std::pin::pin;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use futures_util::stream::BoxStream;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use xai_grok_sampling_types::{
    ApiErrorCode, AssistantItem, ConversationItem, ConversationRequest, ConversationResponse,
    EmptyResponseContext, SamplingError, SentCredential, StopReason, TokenUsage, ToolCall,
    error::Result as SamplingResult, messages, reported_cost_ticks, rs, synthesized_reasoning_item,
};

use crate::client::{ApiBackend, SamplingClient};
use crate::config::{RetryPolicy, SamplerConfig};
use crate::doom_loop_recovery::{FailedResponseCapture, append_recovery_context};
use crate::events::{SamplingErrorInfo, SamplingErrorKind, SamplingEvent, StripReason};
use crate::metrics::InferenceLatencyStats;
use crate::retry::{
    self as retry_mod, RetryDecision, classify_error, clone_error, resolve_max_retries,
};
use crate::stream::responses::stream_responses_tracked;
use crate::stream::{stream_chat_completions, stream_messages};
use crate::types::RequestId;

/// Default per-chunk idle timeout when neither config nor caller
/// supplies one. Matches the shell's session-level default
/// (5 minutes -- long enough for cold-start reasoning, short enough
/// to detect dead streams before the user gives up).
const DEFAULT_IDLE_TIMEOUT_SECS: u64 = 300;

/// Result type for the `submit_and_collect` oneshot. Carries the rich
/// `SamplingError` so callers can inspect retryability, status code,
/// etc., without losing information through the
/// `SamplingErrorInfo` round trip.
pub(crate) type CompletionResult =
    Result<(ConversationResponse, InferenceLatencyStats), SamplingError>;

/// Outcome of a single attempt within the retry loop.
enum AttemptOutcome {
    /// Stream emitted [`SamplingEvent::Completed`] with a non-empty
    /// response.
    Completed {
        response: Box<ConversationResponse>,
        metrics: InferenceLatencyStats,
    },
    /// Stream emitted [`SamplingEvent::Completed`] but the response
    /// was empty (no text, no tool calls). The retry loop treats this
    /// as a transient failure (the model returned reasoning-only or
    /// the stream was truncated). Metrics from the empty attempt are
    /// discarded; a successful retry produces fresh ones.
    Empty { context: EmptyResponseContext },
    /// Stream emitted [`SamplingEvent::Failed`]. The captured raw
    /// error is what the retry loop classifies; if no rich error was
    /// captured (e.g. the failure was synthesised inside the L2
    /// transform), `error` was reconstructed from the
    /// [`SamplingErrorInfo`].
    Failed {
        error: SamplingError,
        recovery_items: Vec<xai_grok_sampling_types::ConversationItem>,
    },
    /// `cancel_token` fired mid-attempt. The retry loop bails out
    /// without further attempts.
    Cancelled,
    /// Failed to construct the underlying raw stream (e.g., HTTP
    /// connect error before any chunks arrive).
    InitFailed { error: SamplingError },
}

/// Run a single sampling request to completion (or final failure).
///
/// Returns the request id so the actor can clean it up from
/// `active_requests` via [`tokio::task::JoinSet::join_next`].
pub(crate) async fn run_request_task(
    request_id: RequestId,
    request: ConversationRequest,
    config: SamplerConfig,
    retry_policy: RetryPolicy,
    event_tx: mpsc::UnboundedSender<SamplingEvent>,
    cancel_token: CancellationToken,
    completion_tx: Option<oneshot::Sender<CompletionResult>>,
) -> RequestId {
    let mut completion_tx = completion_tx;
    let idle_timeout = Duration::from_secs(
        config
            .idle_timeout_secs
            .unwrap_or(DEFAULT_IDLE_TIMEOUT_SECS),
    );
    let configured_max_retries = config.max_retries.or(Some(retry_policy.max_retries));
    let max_retries = if configured_max_retries == Some(0) {
        0
    } else {
        resolve_max_retries(configured_max_retries)
    };

    // `0` means unset — fall back to the default. The threshold no longer
    // caps 429 retries; only `RATE_LIMIT_RETRY_DISABLED` is honored (see
    // `classify_error`).
    let rate_limit_threshold = if retry_policy.rate_limit_retry_threshold == 0 {
        retry_mod::RATE_LIMIT_RETRY_THRESHOLD
    } else {
        retry_policy.rate_limit_retry_threshold
    };

    // Build the initial client. Configuration errors here are fatal
    // (no point retrying with the same broken config).
    let mut client = match SamplingClient::new(config.clone()) {
        Ok(c) => c,
        Err(err) => {
            emit_failed(&event_tx, &request_id, &err);
            send_completion(&mut completion_tx, Err(err));
            return request_id;
        }
    };

    let sampling_span = crate::sampling_log::request_span(
        &request_id,
        &config.model,
        &format!("{:?}", client.api_backend()),
        &config.base_url,
        &client.auth_info(),
    );
    if let Some(eff) = config.reasoning_effort {
        sampling_span.record("reasoning_effort", eff.as_str());
    }

    let mut request = request;
    let mut retry_count: u32 = 0;
    // Doom-loop recovery keeps its own resample budget, independent of the
    // transport/empty budget above.
    let doom_policy = config.doom_loop_recovery;
    let doom_max_retries = doom_policy.map_or(0, |p| p.max_retries);
    let mut doom_retry_count: u32 = 0;
    let output_observed = Arc::new(AtomicBool::new(false));

    // Wall-clock start of the retry loop. The total time budget
    // (`retry_policy.max_retry_duration_secs`, default 10 min) caps how
    // long the loop keeps retrying before surfacing the last error as
    // Fatal — independent of the count-based `max_retries` safety net.
    let retry_started_at = Instant::now();
    let max_retry_duration = Duration::from_secs(retry_policy.max_retry_duration_secs);

    loop {
        if cancel_token.is_cancelled() {
            handle_cancellation(&event_tx, &request_id, &mut completion_tx);
            return request_id;
        }

        // Once the resample budget is spent, the attempt runs with the abort
        // disarmed so it can complete and be accepted as-is.
        let doom_check = doom_policy.filter(|_| doom_retry_count < doom_max_retries);
        let outcome = run_one_attempt(
            &client,
            request.clone(),
            request_id.clone(),
            idle_timeout,
            &event_tx,
            &cancel_token,
            doom_check,
            Arc::clone(&output_observed),
            config.stream,
        )
        .instrument(sampling_span.clone())
        .await;

        let effective_max_retries =
            if retry_policy.retry_only_before_output && output_observed.load(Ordering::Relaxed) {
                0
            } else {
                max_retries
            };

        match outcome {
            AttemptOutcome::Completed {
                response,
                mut metrics,
            } => {
                metrics.attempts = retry_count + doom_retry_count + 1;
                if let Some(policy) = doom_policy {
                    let confident = policy.confident_triggers(&response.doom_loop_signals);
                    if !confident.is_empty() {
                        tracing::warn!(
                            target: crate::sampling_log::TARGET,
                            triggers = ?confident,
                            attempt = doom_retry_count + 1,
                            outcome = "accepted_after_budget",
                            "doom-loop recovery: resample budget spent; accepting as-is"
                        );
                    }
                }
                // Surface token usage on the sampling span alongside effort.
                if let Some(usage) = response.usage.as_ref() {
                    sampling_span.record("output_tokens", usage.completion_tokens);
                    sampling_span.record("reasoning_tokens", usage.reasoning_tokens);
                }
                // Emit Completed only after the loop succeeds; the L2
                // stream's terminal event was suppressed by
                // `run_one_attempt`.
                let _ = event_tx.send(SamplingEvent::Completed {
                    request_id: request_id.clone(),
                    response: response.clone(),
                    metrics: metrics.clone(),
                });
                send_completion(&mut completion_tx, Ok((*response, metrics)));
                return request_id;
            }
            AttemptOutcome::Empty { context } => {
                tracing::warn!(
                    target: crate::sampling_log::TARGET,
                    empty_response = true,
                    empty_reason = context.reason.as_str(),
                    had_reasoning = context.had_reasoning,
                    content_len = context.content_len,
                    tool_call_count = context.tool_call_count,
                    completion_tokens = context.completion_tokens.unwrap_or(0),
                    reasoning_tokens = context.reasoning_tokens.unwrap_or(0),
                    finish_reason = context.finish_reason_str(),
                    first_choice_seen = context.first_choice_seen,
                    model = %context.model,
                    "empty response from model: {reason} (retrying)",
                    reason = context.reason,
                );
                let err = SamplingError::EmptyResponse { context };
                if time_budget_exhausted(
                    retry_started_at,
                    max_retry_duration,
                    &err,
                    &event_tx,
                    &request_id,
                    &mut completion_tx,
                ) {
                    return request_id;
                }
                if !apply_retry_decision(
                    &err,
                    &mut retry_count,
                    effective_max_retries,
                    rate_limit_threshold,
                    &event_tx,
                    &request_id,
                    &mut request,
                    &mut client,
                    &config,
                    &cancel_token,
                    &mut completion_tx,
                )
                .await
                {
                    return request_id;
                }
            }
            AttemptOutcome::Failed {
                error,
                recovery_items,
            } => {
                // Doom-loop resamples run on their own budget and never
                // consult the transport classifier, so no classifier change
                // can silently debit the transport budget for a doom failure.
                if let SamplingError::DoomLoopDetected { .. } = &error {
                    // Callers that opted into `retry_only_before_output`
                    // cannot retract text already handed to them, so a
                    // resample there would leave the poisoned prefix in the
                    // accepted output. Fail the request instead.
                    if retry_policy.retry_only_before_output
                        && output_observed.load(Ordering::Relaxed)
                    {
                        emit_failed(&event_tx, &request_id, &error);
                        send_completion(&mut completion_tx, Err(clone_error(&error)));
                        return request_id;
                    }
                    let backoff = retry_mod::doom_loop_backoff(doom_retry_count + 1);
                    doom_retry_count += 1;
                    append_recovery_context(&mut request, recovery_items);
                    tracing::warn!(
                        target: crate::sampling_log::TARGET,
                        reason = %error,
                        attempt = doom_retry_count,
                        max_retries = doom_max_retries,
                        outcome = "resampled_with_reminder",
                        "doom-loop recovery: retaining the failed response and retrying with guidance"
                    );
                    emit_retrying(
                        &event_tx,
                        &request_id,
                        doom_retry_count,
                        doom_max_retries,
                        &error,
                    );
                    if sleep_or_cancel(backoff, &cancel_token).await {
                        continue;
                    }
                    handle_cancellation(&event_tx, &request_id, &mut completion_tx);
                    return request_id;
                }
                if time_budget_exhausted(
                    retry_started_at,
                    max_retry_duration,
                    &error,
                    &event_tx,
                    &request_id,
                    &mut completion_tx,
                ) {
                    return request_id;
                }
                if !apply_retry_decision(
                    &error,
                    &mut retry_count,
                    effective_max_retries,
                    rate_limit_threshold,
                    &event_tx,
                    &request_id,
                    &mut request,
                    &mut client,
                    &config,
                    &cancel_token,
                    &mut completion_tx,
                )
                .await
                {
                    return request_id;
                }
            }
            AttemptOutcome::Cancelled => {
                handle_cancellation(&event_tx, &request_id, &mut completion_tx);
                return request_id;
            }
            AttemptOutcome::InitFailed { error } => {
                if time_budget_exhausted(
                    retry_started_at,
                    max_retry_duration,
                    &error,
                    &event_tx,
                    &request_id,
                    &mut completion_tx,
                ) {
                    return request_id;
                }
                if !apply_retry_decision(
                    &error,
                    &mut retry_count,
                    effective_max_retries,
                    rate_limit_threshold,
                    &event_tx,
                    &request_id,
                    &mut request,
                    &mut client,
                    &config,
                    &cancel_token,
                    &mut completion_tx,
                )
                .await
                {
                    return request_id;
                }
            }
        }
    }
}

/// Apply a [`RetryDecision`]. Returns `true` if the loop should
/// continue, `false` if the request is finished (either fatal or
/// emit-to-session). Performs the side-effects of the decision:
/// sleeping, rebuilding the client, stripping images, emitting the
/// `Retrying` event.
#[allow(clippy::too_many_arguments)]
async fn apply_retry_decision(
    err: &SamplingError,
    retry_count: &mut u32,
    max_retries: u32,
    rate_limit_threshold: u32,
    event_tx: &mpsc::UnboundedSender<SamplingEvent>,
    request_id: &RequestId,
    request: &mut ConversationRequest,
    client: &mut SamplingClient,
    config: &SamplerConfig,
    cancel_token: &CancellationToken,
    completion_tx: &mut Option<oneshot::Sender<CompletionResult>>,
) -> bool {
    // Server 401 interception: when the session can reactively refresh
    // credentials (`auth_refresh_available` — OAuth session refresh,
    // auth_provider re-mint, or devbox), emit the first 401 of this
    // request to the session so it can refresh once and resubmit. When
    // no refresh mechanism exists (static BYOK / api-key auth), fall
    // through to `classify_error`'s generic retry-with-backoff arm so
    // the 401 retries in-loop within the time budget like every other
    // HTTP error code. `SamplingError::Auth` (client-side bearer
    // construction failure) is handled by `classify_error` (EmitToSession)
    // regardless of this flag.
    //
    // Only the first 401 per request task is emitted for refresh; a 401
    // that recurs after the session's refresh+resubmit (a fresh task,
    // retry_count == 0) is emitted again, bounded by the session's
    // `AuthRetrySchedule`.
    if err.is_auth_error() && config.auth_refresh_available && *retry_count == 0 {
        emit_failed(event_tx, request_id, err);
        send_completion(completion_tx, Err(clone_error(err)));
        return false;
    }

    let decision = classify_error(err, *retry_count, max_retries, rate_limit_threshold);

    // Connection-reset / broken-pipe on body upload often means nginx
    // rejected an oversized payload before responding 413. Strip
    // images proactively before any retry of those errors so we don't
    // burn budget re-uploading the same large body. Gated on the decision
    // actually retrying: a Fatal (budget exhausted) must not mutate the
    // request or tell the user images were "left out of the retry".
    let will_retry = matches!(
        decision,
        RetryDecision::Retry { .. }
            | RetryDecision::RetryWithBackoff { .. }
            | RetryDecision::RetryWithClientRebuild { .. }
    );
    if will_retry && err.is_likely_body_rejected() {
        let stripped_urls = request.strip_images();
        if !stripped_urls.is_empty() {
            tracing::warn!(
                stripped = stripped_urls.len(),
                "stripped {} image(s) before retry (likely nginx 413 via connection reset)",
                stripped_urls.len()
            );
            emit_images_stripped(
                event_tx,
                request_id,
                stripped_urls,
                StripReason::PayloadHeuristic,
            );
        }
    }

    match decision {
        RetryDecision::Retry { backoff } => {
            *retry_count += 1;
            emit_retrying(event_tx, request_id, *retry_count, max_retries, err);
            if sleep_or_cancel(backoff, cancel_token).await {
                true
            } else {
                handle_cancellation(event_tx, request_id, completion_tx);
                false
            }
        }
        RetryDecision::RetryWithBackoff { backoff, .. } => {
            *retry_count += 1;
            emit_retrying(event_tx, request_id, *retry_count, max_retries, err);
            if sleep_or_cancel(backoff, cancel_token).await {
                true
            } else {
                handle_cancellation(event_tx, request_id, completion_tx);
                false
            }
        }
        RetryDecision::RetryWithImageStrip => {
            let stripped_urls = request.strip_images();
            if stripped_urls.is_empty() {
                // Nothing left to strip; upgrade to fatal.
                emit_failed(event_tx, request_id, err);
                send_completion(completion_tx, Err(clone_error(err)));
                return false;
            }
            // Only the deterministic signal (a 400 stamped with the
            // invalid-image code) is a server rejection. Everything else
            // that reaches this arm (413 body-size verdicts, proxy-wrapped
            // 500s, the legacy phrase match, coded mid-stream errors) is a
            // heuristic that must stay request-local.
            // Exhaustive: a new error variant must choose its strip label
            // here instead of silently landing on the heuristic branch.
            let reason = match err {
                SamplingError::Api {
                    status,
                    error_code: Some(ApiErrorCode::InvalidImage),
                    ..
                } if status.as_u16() == 400 => StripReason::ServerRejected,
                SamplingError::Api { .. }
                | SamplingError::StreamError { .. }
                | SamplingError::Auth { .. }
                | SamplingError::InvalidConfiguration(_)
                | SamplingError::Http(_)
                | SamplingError::Serialization(_)
                | SamplingError::EventStreamError(_)
                | SamplingError::IdleTimeout { .. }
                | SamplingError::EmptyResponse { .. }
                | SamplingError::MaxTokensTruncation
                | SamplingError::DoomLoopDetected { .. } => StripReason::PayloadHeuristic,
            };
            tracing::warn!(
                stripped = stripped_urls.len(),
                reason = reason.as_str(),
                error = %err,
                "stripped {} image(s) after an image-related error; retrying without them",
                stripped_urls.len()
            );
            emit_images_stripped(event_tx, request_id, stripped_urls, reason);
            *retry_count += 1;
            emit_retrying(event_tx, request_id, *retry_count, max_retries, err);
            true
        }
        RetryDecision::RetryWithClientRebuild { backoff } => {
            *retry_count += 1;
            emit_retrying(event_tx, request_id, *retry_count, max_retries, err);
            if !sleep_or_cancel(backoff, cancel_token).await {
                handle_cancellation(event_tx, request_id, completion_tx);
                return false;
            }

            // Rebuild client with HTTP/1.1 fallback to escape poisoned
            // HTTP/2 connection pools.
            let mut http1_config = config.clone();
            http1_config.force_http1 = true;
            match SamplingClient::new(http1_config) {
                Ok(fresh) => {
                    *client = fresh;
                    tracing::info!("rebuilt sampling client with HTTP/1.1 fallback for retry");
                }
                Err(rebuild_err) => {
                    tracing::warn!(
                        error = %rebuild_err,
                        "failed to rebuild HTTP/1.1 client for retry; reusing existing client"
                    );
                }
            }
            true
        }
        RetryDecision::EmitToSession(emitted_err) => {
            emit_failed(event_tx, request_id, &emitted_err);
            send_completion(completion_tx, Err(emitted_err));
            false
        }
        RetryDecision::Fatal(fatal_err) => {
            // Distinguish true budget exhaustion (count cap hit on a
            // retryable error) from an immediately-fatal error
            // (non-retryable variant, or a zero retry budget). The
            // time-budget cap is enforced earlier in the loop
            // (`time_budget_exhausted`); here only the count cap or a
            // non-retryable variant reaches Fatal.
            let next_attempt = *retry_count + 1;
            let budget_exhausted = err.is_retryable() && next_attempt >= max_retries;
            if budget_exhausted {
                let exhausted_span = tracing::info_span!(
                    "http.retries_exhausted",
                    total_attempts = next_attempt as i64,
                    model = %config.model,
                    error = %err,
                    status_code = tracing::field::Empty,
                );
                let status_code = match err {
                    SamplingError::Api { status, .. } => Some(status.as_u16()),
                    SamplingError::Http(e) => e.status().map(|s| s.as_u16()),
                    _ => None,
                };
                if let Some(status) = status_code {
                    exhausted_span.record("status_code", status as i64);
                }
                exhausted_span.in_scope(|| {});
            }
            emit_failed(event_tx, request_id, &fatal_err);
            send_completion(completion_tx, Err(fatal_err));
            false
        }
    }
}

async fn sleep_or_cancel(duration: Duration, cancel_token: &CancellationToken) -> bool {
    tokio::select! {
        biased;
        _ = cancel_token.cancelled() => false,
        _ = tokio::time::sleep(duration) => true,
    }
}

/// Check whether the retry time budget is exhausted.
///
/// When the elapsed time since `retry_started_at` has passed
/// `max_retry_duration`, surface `err` as a terminal failure (emit
/// `Failed` + send the completion result) and return `true` so the caller
/// stops the retry loop. Returns `false` (no side-effects) when there is
/// budget remaining.
fn time_budget_exhausted(
    retry_started_at: Instant,
    max_retry_duration: Duration,
    err: &SamplingError,
    event_tx: &mpsc::UnboundedSender<SamplingEvent>,
    request_id: &RequestId,
    completion_tx: &mut Option<oneshot::Sender<CompletionResult>>,
) -> bool {
    if retry_started_at.elapsed() < max_retry_duration {
        return false;
    }
    tracing::warn!(
        target: crate::sampling_log::TARGET,
        elapsed_secs = retry_started_at.elapsed().as_secs(),
        budget_secs = max_retry_duration.as_secs(),
        error = %err,
        "retry time budget exhausted; surfacing last error as fatal"
    );
    emit_failed(event_tx, request_id, err);
    send_completion(completion_tx, Err(clone_error(err)));
    true
}

/// Run a single attempt: build the raw stream, drive it through the
/// matching L2 transform, and forward all non-terminal events to
/// `event_tx`. Captures the rich `SamplingError` from the underlying
/// raw stream so the retry loop can classify it accurately.
///
/// `doom_check` is the doom-loop policy while the resample budget lasts;
/// `None` disarms the mid-stream abort and the terminal confidence check so
/// the attempt completes and its response can be accepted.
#[allow(clippy::too_many_arguments)]
async fn run_one_attempt(
    client: &SamplingClient,
    request: ConversationRequest,
    request_id: RequestId,
    idle_timeout: Duration,
    event_tx: &mpsc::UnboundedSender<SamplingEvent>,
    cancel_token: &CancellationToken,
    doom_check: Option<xai_grok_sampling_types::DoomLoopRecoveryPolicy>,
    output_observed: Arc<AtomicBool>,
    stream: bool,
) -> AttemptOutcome {
    if !stream {
        return run_one_attempt_non_stream(
            client,
            request,
            request_id,
            idle_timeout,
            event_tx,
            cancel_token,
            doom_check,
            Arc::clone(&output_observed),
        )
        .await;
    }
    match client.api_backend() {
        ApiBackend::ChatCompletions => {
            let (raw, metadata) = match client.conversation_stream(request).await {
                Ok(pair) => pair,
                Err(e) => return AttemptOutcome::InitFailed { error: e },
            };
            let (teed, captured) = tee_errors(raw);
            let l2 = stream_chat_completions(teed, metadata, request_id.clone(), idle_timeout);
            drive_l2(
                l2,
                request_id,
                event_tx,
                cancel_token,
                captured,
                None,
                FailedResponseCapture::default(),
                output_observed,
            )
            .await
        }
        ApiBackend::Responses => {
            let (raw, metadata, doom_loop) =
                match client.conversation_stream_responses(request).await {
                    Ok(parts) => parts,
                    Err(e) => return AttemptOutcome::InitFailed { error: e },
                };
            if doom_check.is_none()
                && let Some(collector) = &doom_loop
            {
                collector.disarm_abort();
            }
            let (teed, captured) = tee_errors(raw);
            // Only an armed attempt can replay its failed turn, so only an
            // armed attempt pays for buffering it.
            let failed_response = if doom_check.is_some() {
                FailedResponseCapture::armed()
            } else {
                FailedResponseCapture::default()
            };
            let l2 = stream_responses_tracked(
                teed,
                metadata,
                request_id.clone(),
                idle_timeout,
                doom_loop,
                Arc::clone(&output_observed),
                failed_response.clone(),
            );
            drive_l2(
                l2,
                request_id,
                event_tx,
                cancel_token,
                captured,
                doom_check,
                failed_response,
                output_observed,
            )
            .await
        }
        ApiBackend::Messages => {
            let (raw, metadata) = match client.conversation_stream_messages(request).await {
                Ok(pair) => pair,
                Err(e) => return AttemptOutcome::InitFailed { error: e },
            };
            let (teed, captured) = tee_errors(raw);
            let l2 = stream_messages(teed, metadata, request_id.clone(), idle_timeout);
            drive_l2(
                l2,
                request_id,
                event_tx,
                cancel_token,
                captured,
                None,
                FailedResponseCapture::default(),
                output_observed,
            )
            .await
        }
    }
}

/// Non-streaming variant of [`run_one_attempt`]: issues a single HTTP
/// request without `stream: true` and surfaces the full response at once.
///
/// The Responses backend reuses the streaming L2 transform by feeding it a
/// synthetic `ResponseCompleted` event built from the non-streaming
/// `rs::Response`, so `ConversationResponse` construction (usage, cost,
/// items, stop_reason) stays identical to the streaming path. The Chat
/// Completions and Messages backends build the [`ConversationResponse`]
/// directly and emit a minimal `FirstToken` + `Completed` event pair
/// through [`drive_l2`], so the retry loop's output-observed / truncation /
/// empty-detection logic still runs.
#[allow(clippy::too_many_arguments)]
async fn run_one_attempt_non_stream(
    client: &SamplingClient,
    request: ConversationRequest,
    request_id: RequestId,
    idle_timeout: Duration,
    event_tx: &mpsc::UnboundedSender<SamplingEvent>,
    cancel_token: &CancellationToken,
    doom_check: Option<xai_grok_sampling_types::DoomLoopRecoveryPolicy>,
    output_observed: Arc<AtomicBool>,
) -> AttemptOutcome {
    use futures_util::stream;

    let backend = client.api_backend();
    let stream_start = Instant::now();
    let empty_captured: ErrorCell = Arc::new(Mutex::new(None));

    match backend {
        ApiBackend::Responses => {
            let response = match client.conversation_responses(request).await {
                Ok(r) => r,
                Err(e) => return AttemptOutcome::InitFailed { error: e },
            };
            // Build a synthetic ResponseCompleted event and reuse the
            // streaming L2 transform so the resulting ConversationResponse
            // (usage, cost, items, stop_reason) is constructed identically.
            // The single-event synthetic stream is immediately ready, so the
            // idle timeout never fires.
            let completed =
                rs::ResponseStreamEvent::ResponseCompleted(rs::ResponseCompletedEvent {
                    response,
                    sequence_number: 0,
                });
            let raw: BoxStream<'static, SamplingResult<rs::ResponseStreamEvent>> =
                stream::iter(vec![Ok(completed)]).boxed();
            // Mirror the streaming Responses arm: an armed doom check gets
            // a capture buffer so a doomed attempt can replay its failed
            // turn; every other attempt holds the default (no buffering).
            let failed_response = if doom_check.is_some() {
                FailedResponseCapture::armed()
            } else {
                FailedResponseCapture::default()
            };
            let l2 = stream_responses_tracked(
                raw,
                None,
                request_id.clone(),
                idle_timeout,
                None,
                Arc::clone(&output_observed),
                failed_response.clone(),
            );
            drive_l2(
                l2,
                request_id,
                event_tx,
                cancel_token,
                Arc::clone(&empty_captured),
                doom_check,
                failed_response,
                output_observed,
            )
            .await
        }
        ApiBackend::ChatCompletions => {
            let resp = match client.conversation(request).await {
                Ok(r) => r,
                Err(e) => return AttemptOutcome::InitFailed { error: e },
            };
            let has_output = chat_response_has_output(&resp);
            let response = conversation_response_from_chat(&resp);
            let metrics = InferenceLatencyStats::from_timestamps(stream_start, &[], Instant::now());
            let events =
                build_terminal_event_stream(request_id.clone(), response, metrics, has_output);
            drive_l2(
                events,
                request_id,
                event_tx,
                cancel_token,
                Arc::clone(&empty_captured),
                doom_check,
                FailedResponseCapture::default(),
                output_observed,
            )
            .await
        }
        ApiBackend::Messages => {
            let resp = match client.conversation_messages(request).await {
                Ok(r) => r,
                Err(e) => return AttemptOutcome::InitFailed { error: e },
            };
            let has_output = messages_response_has_output(&resp);
            let response = conversation_response_from_messages(resp);
            let metrics = InferenceLatencyStats::from_timestamps(stream_start, &[], Instant::now());
            let events =
                build_terminal_event_stream(request_id.clone(), response, metrics, has_output);
            drive_l2(
                events,
                request_id,
                event_tx,
                cancel_token,
                Arc::clone(&empty_captured),
                doom_check,
                FailedResponseCapture::default(),
                output_observed,
            )
            .await
        }
    }
}

/// Build a minimal two-event L2 stream for non-streaming backends: a
/// `FirstToken` (only when real output was produced, mirroring the streaming
/// path which emits FirstToken on the first content/tool delta) followed by
/// the terminal `Completed`. [`drive_l2`] applies its normal truncation /
/// empty-detection logic to the `Completed` event.
fn build_terminal_event_stream(
    request_id: RequestId,
    response: ConversationResponse,
    metrics: InferenceLatencyStats,
    has_output: bool,
) -> impl futures_util::Stream<Item = SamplingEvent> {
    let mut events: Vec<SamplingEvent> = Vec::with_capacity(2);
    if has_output {
        events.push(SamplingEvent::FirstToken {
            request_id: request_id.clone(),
        });
    }
    events.push(SamplingEvent::Completed {
        request_id,
        response: Box::new(response),
        metrics,
    });
    futures_util::stream::iter(events)
}

/// Build a [`ConversationResponse`] from a non-streaming Chat Completions
/// response, mirroring the streaming L2 transform's final construction.
fn conversation_response_from_chat(
    resp: &xai_grok_sampling_types::ChatCompletionResponse,
) -> ConversationResponse {
    use std::sync::Arc;

    let model = resp.model.clone();
    let (content, reasoning, tool_calls, finish_reason) = resp
        .choices
        .first()
        .map(|c| {
            let m = &c.message;
            let content = m.content.clone().unwrap_or_default();
            let reasoning = m.reasoning_content.clone().filter(|s| !s.is_empty());
            let tool_calls: Vec<ToolCall> = m
                .tool_calls
                .iter()
                .map(|tc| ToolCall {
                    id: Arc::<str>::from(tc.id.as_str()),
                    name: tc.function.name.clone(),
                    arguments: Arc::<str>::from(tc.function.arguments.as_str()),
                })
                .collect();
            let finish = c.finish_reason.map(StopReason::from);
            (content, reasoning, tool_calls, finish)
        })
        .unwrap_or_default();

    // Honor tool calls by overriding the stop reason if the model forgot.
    let stop_reason = if !tool_calls.is_empty() {
        Some(StopReason::ToolCalls)
    } else {
        finish_reason
    };

    let usage = resp.usage.clone().map(TokenUsage::from);
    let cost_usd_ticks = resp
        .usage
        .as_ref()
        .and_then(|u| reported_cost_ticks(u.cost_in_usd_ticks));

    let mut items: Vec<ConversationItem> = Vec::new();
    if let Some(r) = reasoning {
        items.push(ConversationItem::Reasoning(synthesized_reasoning_item(r)));
    }
    items.push(ConversationItem::Assistant(AssistantItem {
        content: Arc::<str>::from(content),
        tool_calls,
        model_id: Some(model),
        model_fingerprint: None,
        reasoning_effort: None,
    }));

    ConversationResponse {
        items,
        stop_reason,
        usage,
        cost_usd_ticks,
        message_chunks_emitted: 0,
        doom_loop_signals: Vec::new(),
        stop_message: None,
        message_id: None,
        raw_stop_reason: None,
        stop_sequence: None,
    }
}

/// Whether a non-streaming Chat Completions response produced real output
/// (text or tool calls), mirroring the streaming FirstToken gate.
fn chat_response_has_output(resp: &xai_grok_sampling_types::ChatCompletionResponse) -> bool {
    resp.choices.first().is_some_and(|c| {
        let m = &c.message;
        m.content.as_deref().is_some_and(|s| !s.is_empty()) || !m.tool_calls.is_empty()
    })
}

/// Build a [`ConversationResponse`] from a non-streaming Anthropic Messages
/// response, mirroring the streaming L2 transform's final construction.
fn conversation_response_from_messages(resp: messages::MessagesResponse) -> ConversationResponse {
    use std::sync::Arc;
    use xai_grok_sampling_types::messages::ContentBlock;

    let model = resp.model.clone();
    let resp_id = resp.id.clone();
    let mut content = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut reasoning: Option<String> = None;

    for block in resp.content {
        match block {
            ContentBlock::Text { text, .. } => {
                if !content.is_empty() {
                    content.push('\n');
                }
                content.push_str(&text);
            }
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                tool_calls.push(ToolCall {
                    id: Arc::<str>::from(id),
                    name,
                    arguments: Arc::<str>::from(serde_json::to_string(&input).unwrap_or_default()),
                });
            }
            ContentBlock::Thinking { thinking, .. } => {
                if !thinking.is_empty() {
                    reasoning = Some(thinking);
                }
            }
            _ => {}
        }
    }

    let stop_reason = if !tool_calls.is_empty() {
        Some(StopReason::ToolCalls)
    } else {
        resp.stop_reason.map(messages_stop_reason_to_conversation)
    };

    let usage = {
        let u = resp.usage;
        let total_prompt = u
            .input_tokens
            .saturating_add(u.cache_read_input_tokens)
            .saturating_add(u.cache_creation_input_tokens);
        if total_prompt > 0 || u.output_tokens > 0 {
            Some(TokenUsage {
                prompt_tokens: total_prompt,
                completion_tokens: u.output_tokens,
                total_tokens: total_prompt.saturating_add(u.output_tokens),
                reasoning_tokens: 0,
                cached_prompt_tokens: u.cache_read_input_tokens,
                cache_creation_prompt_tokens: u.cache_creation_input_tokens,
            })
        } else {
            None
        }
    };

    let mut items: Vec<ConversationItem> = Vec::new();
    if let Some(r) = reasoning {
        items.push(ConversationItem::Reasoning(synthesized_reasoning_item(r)));
    }
    items.push(ConversationItem::Assistant(AssistantItem {
        content: Arc::<str>::from(content),
        tool_calls,
        model_id: Some(model),
        model_fingerprint: None,
        reasoning_effort: None,
    }));

    ConversationResponse {
        items,
        stop_reason,
        usage,
        // Anthropic Messages API carries no cost on the wire.
        cost_usd_ticks: None,
        message_chunks_emitted: 0,
        doom_loop_signals: Vec::new(),
        stop_message: None,
        message_id: Some(resp_id),
        raw_stop_reason: None,
        stop_sequence: None,
    }
}

/// Whether a non-streaming Messages response produced real output.
fn messages_response_has_output(resp: &messages::MessagesResponse) -> bool {
    resp.content.iter().any(|block| match block {
        messages::ContentBlock::Text { text, .. } => !text.is_empty(),
        messages::ContentBlock::ToolUse { .. } => true,
        _ => false,
    })
}

/// Map an Anthropic Messages `StopReason` to the backend-neutral
/// [`StopReason`], mirroring the streaming L2 transform.
fn messages_stop_reason_to_conversation(sr: messages::StopReason) -> StopReason {
    match sr {
        messages::StopReason::EndTurn => StopReason::Stop,
        messages::StopReason::MaxTokens => StopReason::Length,
        messages::StopReason::StopSequence => StopReason::Stop,
        messages::StopReason::ToolUse => StopReason::ToolCalls,
        messages::StopReason::Refusal => StopReason::ContentFilter,
        messages::StopReason::PauseTurn => StopReason::Stop,
        messages::StopReason::ModelContextWindowExceeded => StopReason::Length,
        messages::StopReason::Unknown(_) => StopReason::Stop,
    }
}

/// Captured-error cell shared between the tee adapter and the
/// per-request task.
type ErrorCell = Arc<Mutex<Option<SamplingError>>>;

/// Wrap a raw chunk stream so its first error is captured into a
/// shared cell. The wrapped stream still yields the original
/// `Result<T, SamplingError>` items unchanged so the L2 transform sees
/// them and converts them to `SamplingErrorInfo` for events.
fn tee_errors<'a, T: Send + 'a>(
    raw: BoxStream<'a, SamplingResult<T>>,
) -> (BoxStream<'a, SamplingResult<T>>, ErrorCell) {
    let cell: ErrorCell = Arc::new(Mutex::new(None));
    let cell_clone = Arc::clone(&cell);
    let teed = raw
        .map(move |item| {
            if let Err(ref e) = item
                && let Ok(mut guard) = cell_clone.lock()
                && guard.is_none()
            {
                // Capture only the first error -- subsequent errors
                // on a torn-down stream are usually secondary effects
                // of the same disconnect.
                *guard = Some(clone_error(e));
            }
            item
        })
        .boxed();
    (teed, cell)
}

/// Drive an L2 event stream: forward non-terminal events to
/// `event_tx`, watch `cancel_token`, return `AttemptOutcome` based on
/// the terminal event (or cancellation). `doom_check`, when set, turns a
/// completed response carrying confident doom-loop signals into a
/// retryable failure (belt-and-braces behind the mid-stream abort).
#[allow(clippy::too_many_arguments)]
async fn drive_l2(
    l2: impl futures_util::Stream<Item = SamplingEvent>,
    request_id: RequestId,
    event_tx: &mpsc::UnboundedSender<SamplingEvent>,
    cancel_token: &CancellationToken,
    captured: ErrorCell,
    doom_check: Option<xai_grok_sampling_types::DoomLoopRecoveryPolicy>,
    failed_response: FailedResponseCapture,
    output_observed: Arc<AtomicBool>,
) -> AttemptOutcome {
    let mut l2 = pin!(l2);
    loop {
        tokio::select! {
            biased;
            _ = cancel_token.cancelled() => {
                return AttemptOutcome::Cancelled;
            }
            next = l2.next() => match next {
                Some(SamplingEvent::Completed { response, metrics, .. }) => {
                    output_observed.store(true, Ordering::Relaxed);
                    // Doom outranks the truncation/empty classes: a confident
                    // loop poisons the attempt whatever else it looks like.
                    if let Some(policy) = doom_check {
                        let triggers = policy.confident_triggers(&response.doom_loop_signals);
                        if !triggers.is_empty() {
                            return AttemptOutcome::Failed {
                                error: SamplingError::DoomLoopDetected {
                                    triggers,
                                    aborted_at_chunk: None,
                                },
                                recovery_items: failed_response.take_items(),
                            };
                        }
                    }
                    if response.stop_reason == Some(xai_grok_sampling_types::StopReason::Length) {
                        return AttemptOutcome::Failed {
                            error: SamplingError::MaxTokensTruncation,
                            recovery_items: Vec::new(),
                        };
                    }
                    // A content-filtered turn (Anthropic refusal, OpenAI
                    // content_filter stop reason) is legitimately content-less and
                    // deterministic — resampling it would retry-storm.
                    let content_filtered = response.stop_reason
                        == Some(xai_grok_sampling_types::StopReason::ContentFilter);
                    if !content_filtered && let Some(reason) = response.empty_reason() {
                        let context = build_empty_context(reason, &response);
                        return AttemptOutcome::Empty { context };
                    }
                    return AttemptOutcome::Completed { response, metrics };
                }
                Some(SamplingEvent::Failed { error: info, .. }) => {
                    let raw = captured
                        .lock()
                        .ok()
                        .and_then(|mut g| g.take());
                    let error = raw.unwrap_or_else(|| synthesize_from_info(&info));
                    let recovery_items = if matches!(error, SamplingError::DoomLoopDetected { .. }) {
                        failed_response.take_items()
                    } else {
                        Vec::new()
                    };
                    return AttemptOutcome::Failed {
                        error,
                        recovery_items,
                    };
                }
                Some(other) => {
                    if matches!(
                        other,
                        SamplingEvent::FirstToken { .. }
                            | SamplingEvent::ChannelToken { .. }
                            | SamplingEvent::ToolCallDelta { .. }
                            | SamplingEvent::BackendToolCallStarted { .. }
                            | SamplingEvent::BackendToolCallCompleted { .. }
                    ) {
                        output_observed.store(true, Ordering::Relaxed);
                    }
                    let _ = event_tx.send(retag(other, &request_id));
                }
                None => {
                    // L2 streams always terminate with Completed or
                    // Failed; reaching None means the producer was
                    // dropped without termination -- treat as a
                    // synthetic transport error.
                    return AttemptOutcome::Failed {
                        error: SamplingError::EventStreamError(
                            "stream dropped without terminal event".to_string(),
                        ),
                        recovery_items: Vec::new(),
                    };
                }
            }
        }
    }
}

/// Re-tag a forwarded event with the canonical request_id. The L2
/// transform tags events with the id we passed in, so this is
/// usually a no-op; keeping the helper makes the data-flow explicit.
fn retag(event: SamplingEvent, _request_id: &RequestId) -> SamplingEvent {
    event
}

/// Reconstruct a [`SamplingError`] from a [`SamplingErrorInfo`] when
/// the L2 transform fired a synthesised Failed event (idle timeout,
/// `ResponseFailed`, server error event) and there is no captured raw
/// error in the cell.
fn synthesize_from_info(info: &SamplingErrorInfo) -> SamplingError {
    match info.kind {
        SamplingErrorKind::IdleTimeout => SamplingError::IdleTimeout {
            elapsed_secs: info
                .message
                .split_whitespace()
                .find_map(|tok| tok.strip_suffix('s').and_then(|n| n.parse::<u64>().ok()))
                .unwrap_or(0),
        },
        SamplingErrorKind::Auth => SamplingError::Auth {
            message: info.message.clone(),
            credential: info.credential,
        },
        // Must stay Serialization: EventStreamError is retryable, and a
        // response-parse failure is deterministic on retry. `info.message`
        // is the variant's rendered Display, so rebuild via the constructor
        // that owns the prefix-stripping.
        SamplingErrorKind::Serialization => {
            SamplingError::serialization_from_rendered(&info.message)
        }
        SamplingErrorKind::Http => SamplingError::EventStreamError(info.message.clone()),
        SamplingErrorKind::Api | SamplingErrorKind::RateLimited => {
            let status = info
                .status_code
                .and_then(|c| reqwest::StatusCode::from_u16(c).ok())
                .unwrap_or(reqwest::StatusCode::INTERNAL_SERVER_ERROR);
            SamplingError::Api {
                status,
                message: info.message.clone(),
                model_metadata: info.model_metadata.clone(),
                retry_after_secs: info.retry_after_secs,
                should_retry: info.should_retry,
                error_code: info.error_code.clone(),
            }
        }
        SamplingErrorKind::EmptyResponse => {
            if let Some(ctx) = &info.empty_response_context {
                SamplingError::EmptyResponse {
                    context: ctx.clone(),
                }
            } else {
                SamplingError::EventStreamError(info.message.clone())
            }
        }
        SamplingErrorKind::MaxTokensTruncation => SamplingError::MaxTokensTruncation,
        SamplingErrorKind::DoomLoopDetected => SamplingError::DoomLoopDetected {
            triggers: info.doom_loop_triggers.clone().unwrap_or_default(),
            aborted_at_chunk: info.doom_loop_aborted_at_chunk,
        },
    }
}

/// Build an [`EmptyResponseContext`] from a completed-but-empty response.
fn build_empty_context(
    reason: xai_grok_sampling_types::EmptyReason,
    response: &ConversationResponse,
) -> EmptyResponseContext {
    let had_reasoning = response
        .reasoning_items()
        .any(|r| !r.summary.is_empty() || r.content.is_some() || r.encrypted_content.is_some());
    let (content_len, tool_call_count, model, first_choice_seen) = match response.assistant() {
        Some(a) => (
            a.content.len(),
            a.tool_calls.len(),
            a.model_id.clone().unwrap_or_default(),
            // If model_id is set, the L2 saw at least one choice.
            a.model_id.is_some(),
        ),
        None => (0, 0, String::new(), false),
    };

    let finish_reason = response.stop_reason.map(|sr| sr.as_str().to_owned());
    let (completion_tokens, reasoning_tokens, prompt_tokens) = response
        .usage
        .as_ref()
        .map(|u| {
            (
                Some(u.completion_tokens),
                Some(u.reasoning_tokens),
                Some(u.prompt_tokens),
            )
        })
        .unwrap_or((None, None, None));

    EmptyResponseContext {
        reason,
        had_reasoning,
        content_len,
        tool_call_count,
        finish_reason,
        completion_tokens,
        reasoning_tokens,
        prompt_tokens,
        model,
        first_choice_seen,
    }
}

fn emit_failed(
    event_tx: &mpsc::UnboundedSender<SamplingEvent>,
    request_id: &RequestId,
    err: &SamplingError,
) {
    let info = SamplingErrorInfo::from(err);
    let _ = event_tx.send(SamplingEvent::Failed {
        request_id: request_id.clone(),
        error: info,
    });
}

fn emit_retrying(
    event_tx: &mpsc::UnboundedSender<SamplingEvent>,
    request_id: &RequestId,
    attempt: u32,
    max_retries: u32,
    err: &SamplingError,
) {
    let info = SamplingErrorInfo::from(err);
    let _ = event_tx.send(SamplingEvent::Retrying {
        request_id: request_id.clone(),
        attempt,
        max_retries,
        kind: info.kind,
        reason: err.to_string(),
        doom_loop_triggers: info.doom_loop_triggers,
        doom_loop_aborted_at_chunk: info.doom_loop_aborted_at_chunk,
    });
}

fn emit_images_stripped(
    event_tx: &mpsc::UnboundedSender<SamplingEvent>,
    request_id: &RequestId,
    stripped_urls: Vec<std::sync::Arc<str>>,
    reason: StripReason,
) {
    let _ = event_tx.send(SamplingEvent::ImagesStripped {
        request_id: request_id.clone(),
        stripped_urls,
        reason,
    });
}

fn handle_cancellation(
    event_tx: &mpsc::UnboundedSender<SamplingEvent>,
    request_id: &RequestId,
    completion_tx: &mut Option<oneshot::Sender<CompletionResult>>,
) {
    // No status code, no upstream API error -- this is a client-side
    // termination. Use kind=Api so consumers that switch on kind have
    // a sensible default; the message clearly identifies it.
    let info = SamplingErrorInfo {
        kind: SamplingErrorKind::Api,
        status_code: None,
        message: "request cancelled".to_string(),
        is_retryable: false,
        retry_after_secs: None,
        should_retry: None,
        error_code: None,
        model_metadata: None,
        empty_response_context: None,
        doom_loop_triggers: None,
        doom_loop_aborted_at_chunk: None,
        credential: SentCredential::Unknown,
    };
    let _ = event_tx.send(SamplingEvent::Failed {
        request_id: request_id.clone(),
        error: info,
    });
    send_completion(
        completion_tx,
        Err(SamplingError::auth_unknown("request cancelled")),
    );
}

fn send_completion(
    completion_tx: &mut Option<oneshot::Sender<CompletionResult>>,
    result: CompletionResult,
) {
    if let Some(tx) = completion_tx.take() {
        let _ = tx.send(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;
    use xai_grok_sampling_types::ApiErrorCode;

    #[test]
    fn synthesize_idle_timeout_extracts_elapsed_secs() {
        let info = SamplingErrorInfo {
            kind: SamplingErrorKind::IdleTimeout,
            status_code: None,
            message: "inference idle timeout after 240s with no chunks".to_string(),
            is_retryable: false,
            retry_after_secs: None,
            should_retry: None,
            error_code: None,
            model_metadata: None,
            empty_response_context: None,
            doom_loop_triggers: None,
            doom_loop_aborted_at_chunk: None,
            credential: SentCredential::Unknown,
        };
        let err = synthesize_from_info(&info);
        match err {
            SamplingError::IdleTimeout { elapsed_secs } => assert_eq!(elapsed_secs, 240),
            other => panic!("expected IdleTimeout, got {other:?}"),
        }
    }

    #[test]
    fn synthesize_api_500_round_trips() {
        let info = SamplingErrorInfo {
            kind: SamplingErrorKind::Api,
            status_code: Some(500),
            message: "boom".to_string(),
            is_retryable: true,
            retry_after_secs: None,
            should_retry: Some(false),
            error_code: None,
            model_metadata: None,
            empty_response_context: None,
            doom_loop_triggers: None,
            doom_loop_aborted_at_chunk: None,
            credential: SentCredential::Unknown,
        };
        let err = synthesize_from_info(&info);
        match err {
            SamplingError::Api {
                status,
                message,
                should_retry,
                ..
            } => {
                assert_eq!(status.as_u16(), 500);
                assert_eq!(message, "boom");
                assert_eq!(should_retry, Some(false), "server veto must survive");
            }
            other => panic!("expected Api, got {other:?}"),
        }
    }

    /// A coded invalid-image error must survive the info round trip: the
    /// message alone would not classify, so losing the code would silently
    /// disable strip recovery on synthesized failures.
    #[test]
    fn synthesize_preserves_error_code_and_image_classification() {
        let original = SamplingError::Api {
            status: reqwest::StatusCode::BAD_REQUEST,
            message: "some future wording without the legacy phrase".to_string(),
            model_metadata: None,
            retry_after_secs: None,
            should_retry: None,
            error_code: Some(ApiErrorCode::InvalidImage),
        };
        assert!(original.is_image_processing_error());

        let info = SamplingErrorInfo::from(&original);
        assert_eq!(info.error_code, Some(ApiErrorCode::InvalidImage));

        let round_tripped = synthesize_from_info(&info);
        assert!(
            round_tripped.is_image_processing_error(),
            "round-tripped error must still classify: {round_tripped:?}"
        );
    }

    /// A StreamError-sourced info has `status_code: None`; synthesis falls
    /// back to a 500 Api error. That fallback must stay inside the
    /// classifier's 400|500 gate, or coded mid-stream image rejections
    /// silently stop stripping after the round trip.
    #[test]
    fn synthesize_stream_sourced_info_still_classifies_for_strip() {
        let original = SamplingError::StreamError {
            error_type: "invalid_request_error".into(),
            message: "bad image".into(),
            code: Some(ApiErrorCode::InvalidImage),
        };
        assert!(original.is_image_processing_error());

        let info = SamplingErrorInfo::from(&original);
        assert_eq!(info.status_code, None, "stream errors carry no status");

        let round_tripped = synthesize_from_info(&info);
        assert!(
            round_tripped.is_image_processing_error(),
            "stream-sourced round trip must still classify: {round_tripped:?}"
        );
    }

    #[test]
    fn synthesize_rate_limited_preserves_retry_after() {
        let info = SamplingErrorInfo {
            kind: SamplingErrorKind::RateLimited,
            status_code: Some(429),
            message: "slow down".to_string(),
            is_retryable: true,
            retry_after_secs: Some(7),
            should_retry: None,
            error_code: None,
            model_metadata: None,
            empty_response_context: None,
            doom_loop_triggers: None,
            doom_loop_aborted_at_chunk: None,
            credential: SentCredential::Unknown,
        };
        let err = synthesize_from_info(&info);
        match err {
            SamplingError::Api {
                status,
                retry_after_secs,
                ..
            } => {
                assert_eq!(status.as_u16(), 429);
                assert_eq!(retry_after_secs, Some(7));
            }
            other => panic!("expected Api(429), got {other:?}"),
        }
    }

    #[test]
    fn synthesize_serialization_stays_serialization() {
        // Round-trip a REAL error's Display so a Display-template rewording
        // cannot silently reintroduce double-prefixing.
        let original = SamplingError::Serialization(
            serde_json::from_str::<i32>("missing field `delta`").unwrap_err(),
        );
        let info = SamplingErrorInfo::from(&original);
        let err = synthesize_from_info(&info);
        assert!(
            matches!(err, SamplingError::Serialization(_)),
            "expected Serialization, got {err:?}"
        );
        assert!(!err.is_retryable());
        assert_eq!(
            err.to_string(),
            info.message,
            "rebuilt Display must round-trip without double-prefixing"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn retry_sleep_returns_immediately_on_cancellation() {
        let cancel_token = CancellationToken::new();
        let sleeper = sleep_or_cancel(Duration::from_secs(120), &cancel_token);
        tokio::pin!(sleeper);

        cancel_token.cancel();
        assert!(!sleeper.await);
    }

    #[tokio::test(start_paused = true)]
    async fn retry_decision_cancellation_emits_terminal_cancel() {
        let cancel_token = CancellationToken::new();
        cancel_token.cancel();
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        let (completion_tx, completion_rx) = oneshot::channel::<CompletionResult>();
        let mut completion_tx = Some(completion_tx);
        let mut retry_count = 0;
        let mut request = ConversationRequest::default();
        let config = SamplerConfig {
            base_url: "http://localhost".into(),
            model: "test-model".into(),
            ..Default::default()
        };
        let mut client = SamplingClient::new(config.clone()).expect("test client");
        let error = SamplingError::EventStreamError("retry me".into());

        let should_continue = apply_retry_decision(
            &error,
            &mut retry_count,
            2,
            crate::retry::RATE_LIMIT_RETRY_THRESHOLD,
            &event_tx,
            &RequestId::from("cancel-backoff"),
            &mut request,
            &mut client,
            &config,
            &cancel_token,
            &mut completion_tx,
        )
        .await;

        assert!(!should_continue);
        assert!(matches!(
            event_rx.recv().await,
            Some(SamplingEvent::Retrying { .. })
        ));
        assert!(matches!(
            event_rx.recv().await,
            Some(SamplingEvent::Failed { .. })
        ));
        assert!(completion_rx.await.expect("completion sent").is_err());
    }

    #[tokio::test]
    async fn tee_captures_first_error_only() {
        let items: Vec<SamplingResult<u32>> = vec![
            Ok(1),
            Err(SamplingError::EventStreamError("first".into())),
            Err(SamplingError::EventStreamError("second".into())),
        ];
        let raw = stream::iter(items).boxed();
        let (mut teed, cell) = tee_errors(raw);
        while teed.next().await.is_some() {}
        let captured = cell.lock().unwrap().take().expect("error captured");
        match captured {
            SamplingError::EventStreamError(msg) => assert_eq!(msg, "first"),
            other => panic!("expected EventStreamError, got {other:?}"),
        }
    }
}
