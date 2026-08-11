//! Scripted [`Provider`] for tests. Replays a queue of per-round `StreamEvent` lists so an
//! integration test can drive a multi-round `Agent::run_turn` (tool-use round → tool-result round →
//! final text round) without touching the network.
//!
//! Activated by `meka acp` only when the `MEKA_ACP_MOCK_PROVIDER` environment variable is set to
//! `1`. The variable also names the file containing the JSON-encoded script (see
//! [`crate::provider::mock::load_script_from_env`]). Anything else (production, REPL, oneshot) is
//! unaffected; this module is only reachable via the env-gated path.
//!
//! The mock is intentionally minimal: text deltas, thinking deltas, tool-use lifecycle,
//! `MessageEnd`, plus a synthetic `Fail` event that returns an error from [`Provider::stream`] so
//! the agent's non-Interrupted error path can be exercised end-to-end. Image content and
//! token-usage events are not supported; tests that need them should extend the mock first.

use std::{collections::VecDeque, sync::Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    error::Result,
    provider::{Message, ModelInfo, Provider, StopReason, StreamEvent, TokenUsage, ToolDefinition},
};

/// Serialized event used by [`MockProvider`]. Mirrors the runtime [`StreamEvent`] enum but uses
/// owned struct-tagged variants so scripts can be loaded from JSON (`serde`'s internally-tagged
/// enums don't accept tuple/newtype variants). `Sleep` is the one non-stream-event variant. It
/// stalls the mock so a test can fire `session/cancel` mid-turn; the sleep races against the
/// cancellation token, so cancel cuts it short cleanly.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MockEvent {
    Text {
        text: String,
    },
    ThinkingDelta {
        text: String,
    },
    /// Caps an in-flight thinking block. `signature` mirrors the real Claude wire shape but is
    /// `None` in every test today; the agent treats it as opaque pass-through (see
    /// [`crate::frontend::FrontendEvent::ThinkingBlock`]).
    ThinkingComplete {
        signature: Option<String>,
    },
    ToolUseStart {
        id: String,
        name: String,
    },
    ToolInputDelta {
        delta: String,
    },
    ToolUseEnd {
        input: serde_json::Value,
    },
    MessageEnd {
        stop_reason: MockStopReason,
    },
    Sleep {
        ms: u64,
    },
    /// Synthetic provider failure. The stream returns `Err(MekaError::Provider(message))`
    /// immediately, exercising the non-Interrupted error arm of `Agent::run_turn` (which the ACP
    /// layer maps to a JSON-RPC `internal_error`).
    Fail {
        message: String,
    },
    /// Synthetic *transient* provider failure. The stream returns
    /// `Err(MekaError::RetryableProvider { .. })` immediately, exercising `Agent::run_streaming`'s
    /// retry-with-backoff path. Each retry consumes one more round from the script, so a
    /// `[FailRetryable, ..success events..]` script simulates "first attempt overloaded, retry
    /// succeeds" — the same shape a real transient 429/529 followed by success takes.
    FailRetryable {
        message: String,
        retry_after_secs: Option<u64>,
    },
    /// Synthetic *malformed-request* failure. The stream returns
    /// `Err(MekaError::InvalidRequest(message))` immediately, exercising `Agent::run_turn`'s
    /// degrade-and-retry path. Each attempt consumes one round, so
    /// `[FailInvalidRequest, ..success events..]` simulates "the provider refused the content meka
    /// just appended, the retry without it succeeds".
    FailInvalidRequest {
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MockStopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    /// Model-side refusal. Maps to `StopReason::Refusal("")`; the text content of the assistant
    /// message is what the user sees as the refusal explanation.
    Refusal,
}

impl From<MockStopReason> for StopReason {
    fn from(reason: MockStopReason) -> Self {
        match reason {
            MockStopReason::EndTurn => StopReason::EndTurn,
            MockStopReason::ToolUse => StopReason::ToolUse,
            MockStopReason::MaxTokens => StopReason::MaxTokens,
            MockStopReason::Refusal => StopReason::Refusal(String::new()),
        }
    }
}

/// A scripted multi-round response. Each call to [`Provider::stream`] drains one round
/// (`Vec<MockEvent>`); subsequent rounds satisfy subsequent agent loop iterations after tool
/// results return.
#[derive(Debug, Default)]
pub struct MockProvider {
    rounds: Mutex<VecDeque<Vec<MockEvent>>>,
    /// Canned response for [`Provider::fetch_model_info`]; `None` mimics a provider/model with no
    /// models API (returns `Ok(None)`).
    model_info: Option<ModelInfo>,
    /// Value [`Provider::needs_effort_catalog`] returns; `true` mimics Codex wanting its effort
    /// catalog probed so a resolver test can assert a table-known model still gets probed.
    reports_effort: bool,
}

impl MockProvider {
    pub fn from_rounds(rounds: Vec<Vec<MockEvent>>) -> Self {
        Self {
            rounds: Mutex::new(rounds.into()),
            ..Default::default()
        }
    }

    /// Build a mock whose `fetch_model_info` returns `info`, for model-metadata resolver tests.
    #[cfg(test)]
    pub fn with_model_info(info: ModelInfo) -> Self {
        Self {
            model_info: Some(info),
            ..Default::default()
        }
    }

    /// Like [`Self::with_model_info`] but also advertises an effort catalog
    /// ([`Provider::needs_effort_catalog`] → `true`), mimicking Codex.
    #[cfg(test)]
    pub fn with_effort_catalog(info: ModelInfo) -> Self {
        Self {
            model_info: Some(info),
            reports_effort: true,
            ..Default::default()
        }
    }
}

#[async_trait]
impl Provider for MockProvider {
    async fn complete(
        &self,
        _system_prompt: &str,
        _messages: &[Message],
        _tools: &[ToolDefinition],
    ) -> Result<(
        Message,
        StopReason,
        TokenUsage,
        Vec<crate::provider::Notice>,
    )> {
        // Tests only drive the streaming path; `complete` is reached only via auto-compaction,
        // which the ACP test suite doesn't exercise. If a future test needs it, populate the rounds
        // queue the same way and add a matching impl here.
        Err(crate::error::MekaError::Provider(
            "MockProvider::complete is not implemented".to_string(),
        ))
    }

    async fn fetch_model_info(&self) -> Result<Option<ModelInfo>> {
        Ok(self.model_info.clone())
    }

    fn needs_effort_catalog(&self) -> bool {
        self.reports_effort
    }

    async fn stream(
        &self,
        _system_prompt: &str,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        event_sender: mpsc::Sender<StreamEvent>,
        cancellation: CancellationToken,
    ) -> Result<()> {
        let events = {
            let mut rounds = self
                .rounds
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            rounds.pop_front().unwrap_or_default()
        };
        for event in events {
            if cancellation.is_cancelled() {
                return Ok(());
            }
            match event {
                MockEvent::Fail { message } => {
                    return Err(crate::error::MekaError::Provider(message));
                }
                MockEvent::FailRetryable {
                    message,
                    retry_after_secs,
                } => {
                    return Err(crate::error::MekaError::RetryableProvider {
                        message,
                        retry_after: retry_after_secs.map(std::time::Duration::from_secs),
                    });
                }
                MockEvent::FailInvalidRequest { message } => {
                    return Err(crate::error::MekaError::InvalidRequest(message));
                }
                MockEvent::Sleep { ms } => {
                    // Race the sleep against cancellation so a mid-turn `session/cancel` doesn't
                    // have to wait for the full delay to elapse.
                    tokio::select! {
                        _ = tokio::time::sleep(std::time::Duration::from_millis(ms)) => {}
                        _ = cancellation.cancelled() => return Ok(()),
                    }
                    continue;
                }
                event => {
                    let stream_event = match event {
                        MockEvent::Text { text } => StreamEvent::TextDelta(text),
                        MockEvent::ThinkingDelta { text } => StreamEvent::ThinkingDelta(text),
                        MockEvent::ThinkingComplete { signature } => {
                            StreamEvent::ThinkingComplete { signature }
                        }
                        MockEvent::ToolUseStart { id, name } => {
                            StreamEvent::ToolUseStart { id, name }
                        }
                        MockEvent::ToolInputDelta { delta } => StreamEvent::ToolInputDelta(delta),
                        MockEvent::ToolUseEnd { input } => StreamEvent::ToolUseEnd { input },
                        MockEvent::MessageEnd { stop_reason } => StreamEvent::MessageEnd {
                            stop_reason: stop_reason.into(),
                        },
                        MockEvent::Sleep { .. }
                        | MockEvent::Fail { .. }
                        | MockEvent::FailRetryable { .. }
                        | MockEvent::FailInvalidRequest { .. } => {
                            unreachable!("handled above")
                        }
                    };
                    if event_sender.send(stream_event).await.is_err() {
                        // Receiver dropped; the test ended early. Not an error.
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "mock"
    }
}

/// Read the JSON script from the path named in `MEKA_ACP_MOCK_PROVIDER_SCRIPT`. Returns `Ok(None)`
/// when the env var is unset; `Err` only on actual parse failure (so the meka startup path can
/// choose to log+abort vs proceed).
pub fn load_script_from_env() -> Result<Option<Vec<Vec<MockEvent>>>> {
    let Ok(path) = std::env::var("MEKA_ACP_MOCK_PROVIDER_SCRIPT") else {
        return Ok(None);
    };
    let body = std::fs::read_to_string(&path).map_err(|error| {
        crate::error::MekaError::Config(format!(
            "MEKA_ACP_MOCK_PROVIDER_SCRIPT='{}' could not be read: {}",
            path, error,
        ))
    })?;
    let rounds: Vec<Vec<MockEvent>> = serde_json::from_str(&body).map_err(|error| {
        crate::error::MekaError::Config(format!(
            "MEKA_ACP_MOCK_PROVIDER_SCRIPT='{}' is not valid JSON: {}",
            path, error,
        ))
    })?;
    Ok(Some(rounds))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_provider_drains_one_round_per_stream_call() {
        let provider = MockProvider::from_rounds(vec![
            vec![
                MockEvent::Text {
                    text: "hello".into(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
            vec![MockEvent::Text {
                text: "second".into(),
            }],
        ]);

        let (tx, mut rx) = mpsc::channel(8);
        provider
            .stream("", &[], &[], tx, CancellationToken::new())
            .await
            .expect("first round");
        // First round emits two events then the channel sender drops.
        assert!(matches!(
            rx.recv().await,
            Some(StreamEvent::TextDelta(ref t)) if t == "hello"
        ));
        assert!(matches!(
            rx.recv().await,
            Some(StreamEvent::MessageEnd { .. })
        ));

        // Second call drains the second round.
        let (tx2, mut rx2) = mpsc::channel(8);
        provider
            .stream("", &[], &[], tx2, CancellationToken::new())
            .await
            .expect("second round");
        assert!(matches!(
            rx2.recv().await,
            Some(StreamEvent::TextDelta(ref t)) if t == "second"
        ));
    }

    #[tokio::test]
    async fn test_mock_provider_completes_when_script_exhausted() {
        let provider = MockProvider::from_rounds(vec![]);
        let (tx, mut rx) = mpsc::channel(8);
        provider
            .stream("", &[], &[], tx, CancellationToken::new())
            .await
            .expect("empty script");
        assert!(rx.recv().await.is_none(), "exhausted script emits nothing");
    }

    /// `Fail` returns `Err(MekaError::Provider(_))` from [`Provider::stream`] without emitting any
    /// events. The agent loop turns that into a non-Interrupted `run_turn` error, which the ACP
    /// layer maps to a JSON-RPC `internal_error` response.
    #[tokio::test]
    async fn test_mock_provider_fail_event_returns_error() {
        let provider = MockProvider::from_rounds(vec![vec![MockEvent::Fail {
            message: "boom".into(),
        }]]);
        let (tx, mut rx) = mpsc::channel(8);
        let result = provider
            .stream("", &[], &[], tx, CancellationToken::new())
            .await;
        let error = result.expect_err("Fail must propagate as Err");
        assert!(
            matches!(&error, crate::error::MekaError::Provider(message) if message == "boom"),
            "unexpected error: {:?}",
            error
        );
        // No events were sent before the failure.
        assert!(rx.recv().await.is_none(), "Fail must not emit events");
    }

    /// `FailRetryable` returns `Err(MekaError::RetryableProvider { .. })` carrying the configured
    /// `retry_after`, without emitting any events — mirrors `Fail`'s shape but with the typed
    /// variant `Agent::run_streaming`'s retry loop pattern-matches on.
    #[tokio::test]
    async fn test_mock_provider_fail_retryable_event_returns_typed_error() {
        let provider = MockProvider::from_rounds(vec![vec![MockEvent::FailRetryable {
            message: "overloaded".into(),
            retry_after_secs: Some(3),
        }]]);
        let (tx, mut rx) = mpsc::channel(8);
        let result = provider
            .stream("", &[], &[], tx, CancellationToken::new())
            .await;
        match result.expect_err("FailRetryable must propagate as Err") {
            crate::error::MekaError::RetryableProvider {
                message,
                retry_after,
            } => {
                assert_eq!(message, "overloaded");
                assert_eq!(retry_after, Some(std::time::Duration::from_secs(3)));
            }
            other => panic!("expected RetryableProvider, got {other:?}"),
        }
        assert!(
            rx.recv().await.is_none(),
            "FailRetryable must not emit events"
        );
    }

    /// `ThinkingDelta` + `ThinkingComplete` map straight through to the same-named `StreamEvent`
    /// variants. The agent loop collapses the pair into a single `FrontendEvent::ThinkingBlock`,
    /// which the ACP frontend renders as a `SessionUpdate::AgentThoughtChunk` notification.
    #[tokio::test]
    async fn test_mock_provider_emits_thinking_delta_and_complete() {
        let provider = MockProvider::from_rounds(vec![vec![
            MockEvent::ThinkingDelta {
                text: "let me think...".into(),
            },
            MockEvent::ThinkingComplete { signature: None },
            MockEvent::Text {
                text: "done".into(),
            },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::EndTurn,
            },
        ]]);

        let (tx, mut rx) = mpsc::channel(8);
        provider
            .stream("", &[], &[], tx, CancellationToken::new())
            .await
            .expect("stream");
        assert!(matches!(
            rx.recv().await,
            Some(StreamEvent::ThinkingDelta(ref t)) if t == "let me think..."
        ));
        assert!(matches!(
            rx.recv().await,
            Some(StreamEvent::ThinkingComplete { signature: None })
        ));
        assert!(matches!(
            rx.recv().await,
            Some(StreamEvent::TextDelta(ref t)) if t == "done"
        ));
        assert!(matches!(
            rx.recv().await,
            Some(StreamEvent::MessageEnd { .. })
        ));
    }
}
