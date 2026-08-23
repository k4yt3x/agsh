//! Scripted [`Provider`] for tests. Replays a queue of per-round `StreamEvent` lists so an
//! integration test can drive a multi-round `Agent::run_turn` (tool-use round → tool-result round →
//! final text round) without touching the network.
//!
//! Activated when `MEKA_MOCK_PROVIDER` is set to `1`, which every surface honours: `meka acp`,
//! `meka serve`, and the CLI entry point behind the REPL and `--oneshot`. That last one is what
//! lets a test ask what two `meka` processes do to each other's sessions. A second variable,
//! `MEKA_MOCK_PROVIDER_SCRIPT`, names the file holding the JSON-encoded script (see
//! [`crate::provider::mock::load_script_from_env`]). Nothing here is reachable without them, and
//! the whole module is compiled out of a release build.
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
    provider::{
        ContentBlock, Message, Provider, Role, StopReason, StreamEvent, TokenUsage, ToolDefinition,
    },
};

/// Serialized event used by [`MockProvider`]. Mirrors the runtime [`StreamEvent`] enum but uses
/// owned struct-tagged variants so scripts can be loaded from JSON (`serde`'s internally-tagged
/// enums don't accept tuple/newtype variants). `Sleep` and `Stall` are the two non-stream-event
/// variants; both delay the mock so a test can act mid-turn, and they differ in whether
/// cancellation cuts them short.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MockEvent {
    Text {
        text: String,
    },
    ThinkingDelta {
        text: String,
    },
    /// Caps an in-flight thinking block. `opaque` carries whichever wire shape the script is
    /// standing in for -- a Claude signature, or a Responses reasoning item's sealed content. The
    /// agent treats it as pass-through (see [`crate::frontend::FrontendEvent::ThinkingBlock`]), so
    /// a script sets it to check that the turn it records can be replayed. Defaulted so existing
    /// scripts keep loading.
    ThinkingComplete {
        #[serde(default)]
        opaque: Option<crate::provider::OpaqueReasoning>,
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
    /// A delay cancellation does *not* cut short, standing in for the part of a turn a drain has
    /// to wait out rather than interrupt: a tool call already inside a syscall, an MCP round-trip
    /// under its own timeout, the conversation commit at the end. Cancellation is checked between
    /// events, so a real turn's response to it is never instant either.
    ///
    /// [`Self::Sleep`] is the opposite and stays the default choice: a test asserting that cancel
    /// *works* wants the sleep to end the moment the token fires. This one exists for the tests
    /// asserting what happens to a turn that is still unwinding.
    Stall {
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
    /// A provider-side advisory forwarded to the frontend mid-stream. Exists so a test can put a
    /// notice *before* a failure and assert the retry still fires: notices are the one event the
    /// agent forwards without marking the turn as having produced output.
    Notice {
        message: String,
    },
    /// Synthetic *transport* failure: the stream returns `Err(MekaError::StreamError(message))`,
    /// which the agent retries only while nothing user-visible has been emitted. Each attempt
    /// consumes one round.
    FailStream {
        message: String,
    },
    /// Synthetic *context-window overflow*. The stream returns `Err(MekaError::ContextOverflow)`,
    /// which is the one recovery path in `Agent::run_turn` no test could previously reach: the
    /// emergency compact-and-retry only fires on this error, and nothing could produce it. Each
    /// attempt consumes one round, so `[FailContextOverflow, ..success events..]` is "the request
    /// was too large, the compacted retry fit".
    FailContextOverflow {
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

/// A scripted multi-round response. Each call to [`Provider::stream`] *or* [`Provider::complete`]
/// drains one round (`Vec<MockEvent>`); subsequent rounds satisfy subsequent agent loop iterations
/// after tool results return. The two paths share the one queue, so a script that spawns a
/// sub-agent (which runs non-streaming) must budget a round for each of the sub-agent's turns.
#[derive(Debug, Default)]
pub struct MockProvider {
    rounds: Mutex<VecDeque<Vec<MockEvent>>>,
    /// Messages handed to each [`Provider::complete`] call, in order.
    ///
    /// Recorded because some behaviour is only observable in the *request*: whether the checkpoint
    /// turn respects `context_messages`, or emits two consecutive user turns. A test that rebuilds
    /// the expected list itself asserts on its own arithmetic and passes even when the production
    /// path is reverted, which is worse than no test at all.
    completions: Mutex<Vec<Vec<Message>>>,
    /// What each [`Provider::stream`] call was handed, in order.
    ///
    /// The streaming counterpart to [`Self::completions`], and added for the same reason plus one
    /// more: some behaviour is only observable in the request and only on the streaming path.
    /// Whether `[session].context_messages` is re-applied on every round of a turn, for instance,
    /// cannot be seen in the response at all, so a test that did not record this had nothing to
    /// assert against and passed with the production path reverted.
    streams: Mutex<Vec<StreamRequest>>,
}

/// One recorded [`Provider::stream`] call. Owned rather than borrowed because a test inspects it
/// after the turn has finished and the caller's slices are long gone.
///
/// Only ever read from `#[cfg(test)]` code, but recorded unconditionally: the recording lives in
/// `stream`, which is one function compiled for both builds, and splitting it would put a `cfg` in
/// the middle of the mock's hot path to save three fields in a binary that already carries the
/// whole mock.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone)]
pub struct StreamRequest {
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    /// The prompt this request was attributed to, read the way a real provider reads it while
    /// building its billing header.
    ///
    /// Recorded for the same reason the fields above are, with the sharpest version of it: the
    /// streaming path hands `stream` to `tokio::spawn`, task-locals do not cross a spawn, and
    /// nothing in the response reveals whether the attribution made it. Without this the wrapper
    /// that carries it over can be deleted and every test still passes.
    pub prompt_id: Option<uuid::Uuid>,
}

impl MockProvider {
    /// What each `stream` call was handed so far, in order.
    #[cfg(test)]
    pub fn streams(&self) -> Vec<StreamRequest> {
        self.streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// The messages behind each `complete` call so far, in order.
    #[cfg(test)]
    pub fn completions(&self) -> Vec<Vec<Message>> {
        self.completions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn from_rounds(rounds: Vec<Vec<MockEvent>>) -> Self {
        Self {
            rounds: Mutex::new(rounds.into()),
            ..Default::default()
        }
    }
}

#[async_trait]
impl Provider for MockProvider {
    /// Drains one round and folds it into a finished message, so a single script drives either
    /// path. Non-streaming is not an exotic corner: sub-agents run this way (`Agent::new_subagent`
    /// sets `streaming: false`), as does auto-compaction.
    async fn complete(
        &self,
        _system_prompt: &str,
        messages: &[Message],
        _tools: &[ToolDefinition],
    ) -> Result<(
        Message,
        StopReason,
        TokenUsage,
        Vec<crate::provider::Notice>,
    )> {
        self.completions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(messages.to_vec());

        let events = {
            let mut rounds = self
                .rounds
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            rounds.pop_front().unwrap_or_default()
        };

        let mut content: Vec<ContentBlock> = Vec::new();
        let mut text = String::new();
        let mut pending_tool: Option<(String, String)> = None;
        let mut stop_reason = StopReason::EndTurn;

        for event in events {
            match event {
                MockEvent::Fail { message } => {
                    return Err(crate::error::MekaError::Provider(message));
                }
                MockEvent::FailStream { message } => {
                    return Err(crate::error::MekaError::StreamError(message));
                }
                // No frontend on the non-streaming path, so there is nothing to forward it to.
                MockEvent::Notice { .. } => {}
                MockEvent::FailContextOverflow { message } => {
                    return Err(crate::error::MekaError::ContextOverflow(message));
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
                // `complete` takes no cancellation token, so the two delays are the same thing
                // here.
                MockEvent::Sleep { ms } | MockEvent::Stall { ms } => {
                    tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
                }
                MockEvent::Text { text: chunk } => text.push_str(&chunk),
                // Thinking has no place in a non-streaming reply here: the real providers return it
                // as a block, but no `complete` caller in meka reads one.
                MockEvent::ThinkingDelta { .. } | MockEvent::ThinkingComplete { .. } => {}
                MockEvent::ToolUseStart { id, name } => {
                    // Flush first, so text that preceded this call stays ahead of it and text
                    // between two calls stays between them. Accumulating everything and appending
                    // at the end would reorder blocks relative to a real provider.
                    if !text.is_empty() {
                        content.push(ContentBlock::Text {
                            text: std::mem::take(&mut text),
                        });
                    }
                    pending_tool = Some((id, name));
                }
                MockEvent::ToolInputDelta { .. } => {}
                MockEvent::ToolUseEnd { input } => {
                    if let Some((id, name)) = pending_tool.take() {
                        content.push(ContentBlock::ToolUse { id, name, input });
                    }
                }
                MockEvent::MessageEnd {
                    stop_reason: reason,
                } => stop_reason = reason.into(),
            }
        }

        if !text.is_empty() {
            content.push(ContentBlock::Text { text });
        }

        Ok((
            Message {
                role: Role::Assistant,
                content,
            },
            stop_reason,
            TokenUsage::default(),
            Vec::new(),
        ))
    }

    async fn stream(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        event_sender: mpsc::Sender<StreamEvent>,
        cancellation: CancellationToken,
    ) -> Result<()> {
        self.streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(StreamRequest {
                system_prompt: system_prompt.to_string(),
                messages: messages.to_vec(),
                tools: tools.to_vec(),
                prompt_id: crate::provider::current_prompt_id(),
            });

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
                // Each failure sends `StreamEvent::Error` before returning its typed error, the
                // way the real SSE drivers do. `Agent`'s handler documents that as an invariant of
                // every producer and acts on it, so a mock that skipped it would exercise a path
                // production never takes.
                MockEvent::Fail { message } => {
                    send_stream_error(&event_sender, &message).await;
                    return Err(crate::error::MekaError::Provider(message));
                }
                MockEvent::FailStream { message } => {
                    send_stream_error(&event_sender, &message).await;
                    return Err(crate::error::MekaError::StreamError(message));
                }
                MockEvent::Notice { message } => {
                    if event_sender
                        .send(StreamEvent::Notice(crate::provider::Notice::info(message)))
                        .await
                        .is_err()
                    {
                        return Ok(());
                    }
                }
                MockEvent::FailContextOverflow { message } => {
                    send_stream_error(&event_sender, &message).await;
                    return Err(crate::error::MekaError::ContextOverflow(message));
                }
                MockEvent::FailRetryable {
                    message,
                    retry_after_secs,
                } => {
                    send_stream_error(&event_sender, &message).await;
                    return Err(crate::error::MekaError::RetryableProvider {
                        message,
                        retry_after: retry_after_secs.map(std::time::Duration::from_secs),
                    });
                }
                MockEvent::FailInvalidRequest { message } => {
                    send_stream_error(&event_sender, &message).await;
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
                MockEvent::Stall { ms } => {
                    tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
                    continue;
                }
                event => {
                    let stream_event = match event {
                        MockEvent::Notice { message } => {
                            StreamEvent::Notice(crate::provider::Notice::info(message))
                        }
                        MockEvent::Text { text } => StreamEvent::TextDelta(text),
                        MockEvent::ThinkingDelta { text } => StreamEvent::ThinkingDelta(text),
                        MockEvent::ThinkingComplete { opaque } => {
                            StreamEvent::ThinkingComplete { opaque }
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
                        | MockEvent::Stall { .. }
                        | MockEvent::Fail { .. }
                        | MockEvent::FailStream { .. }
                        | MockEvent::FailRetryable { .. }
                        | MockEvent::FailInvalidRequest { .. }
                        | MockEvent::FailContextOverflow { .. } => {
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

/// Read the JSON script from the path named in `MEKA_MOCK_PROVIDER_SCRIPT`. Returns `Ok(None)`
/// when the env var is unset; `Err` only on actual parse failure (so the meka startup path can
/// choose to log+abort vs proceed).
pub fn load_script_from_env() -> Result<Option<Vec<Vec<MockEvent>>>> {
    let Ok(path) = std::env::var("MEKA_MOCK_PROVIDER_SCRIPT") else {
        return Ok(None);
    };
    let body = std::fs::read_to_string(&path).map_err(|error| {
        crate::error::MekaError::Config(format!(
            "MEKA_MOCK_PROVIDER_SCRIPT='{}' could not be read: {}",
            path, error,
        ))
    })?;
    let rounds: Vec<Vec<MockEvent>> = serde_json::from_str(&body).map_err(|error| {
        crate::error::MekaError::Config(format!(
            "MEKA_MOCK_PROVIDER_SCRIPT='{}' is not valid JSON: {}",
            path, error,
        ))
    })?;
    Ok(Some(rounds))
}

/// Send the `StreamEvent::Error` that every real driver emits immediately before returning its own
/// typed error. A dropped receiver is the normal shutdown race and carries no information here.
async fn send_stream_error(event_sender: &mpsc::Sender<StreamEvent>, message: &str) {
    if event_sender
        .send(StreamEvent::Error(message.to_string()))
        .await
        .is_err()
    {
        tracing::trace!("stream event receiver dropped");
    }
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

    /// `Fail` returns `Err(MekaError::Provider(_))` from [`Provider::stream`], preceded by the
    /// `StreamEvent::Error` every real driver sends before its typed return. The agent loop turns
    /// the typed error into a non-Interrupted `run_turn` error, which the ACP layer maps to a
    /// JSON-RPC `internal_error` response.
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
        // The error rides the channel first, then the channel closes. `Agent` depends on that
        // ordering: its `StreamEvent::Error` arm deliberately does not return, because the typed
        // error arriving from the join handle is the one carrying the retry classification.
        assert!(
            matches!(rx.recv().await, Some(StreamEvent::Error(message)) if message == "boom"),
            "Fail must announce itself on the stream before returning",
        );
        assert!(rx.recv().await.is_none(), "and then the stream is over");
    }

    /// `FailRetryable` returns `Err(MekaError::RetryableProvider { .. })` carrying the configured
    /// `retry_after` — mirrors `Fail`'s shape, including the preceding `StreamEvent::Error`, but
    /// with the typed variant `Agent::run_streaming`'s retry loop pattern-matches on.
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
            matches!(rx.recv().await, Some(StreamEvent::Error(message)) if message == "overloaded"),
            "FailRetryable must announce itself on the stream before returning",
        );
        assert!(rx.recv().await.is_none(), "and then the stream is over");
    }

    /// `complete` folds one round into a message, preserving block order: text that preceded a tool
    /// call stays ahead of it and text between two calls stays between them. Sub-agents take this
    /// path, so a reordering here would show up as a sub-agent's narration landing after its work.
    #[tokio::test]
    async fn test_mock_provider_complete_folds_a_round_preserving_block_order() {
        let provider = MockProvider::from_rounds(vec![
            vec![
                MockEvent::Text {
                    text: "before ".into(),
                },
                MockEvent::ToolUseStart {
                    id: "call-1".into(),
                    name: "read_file".into(),
                },
                MockEvent::ToolUseEnd {
                    input: serde_json::json!({"path": "a.txt"}),
                },
                MockEvent::Text {
                    text: "after".into(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::ToolUse,
                },
            ],
            vec![MockEvent::Text {
                text: "second round".into(),
            }],
        ]);

        let (message, stop_reason, ..) = provider.complete("", &[], &[]).await.expect("complete");
        assert!(matches!(message.role, Role::Assistant));
        assert_eq!(message.content.len(), 3);
        assert!(matches!(&message.content[0], ContentBlock::Text { text } if text == "before "));
        assert!(
            matches!(&message.content[1], ContentBlock::ToolUse { id, name, .. } if id == "call-1" && name == "read_file")
        );
        assert!(matches!(&message.content[2], ContentBlock::Text { text } if text == "after"));
        assert!(matches!(stop_reason, StopReason::ToolUse));

        // The queue is shared with `stream`, so the first round is gone for both paths.
        let (message, ..) = provider.complete("", &[], &[]).await.expect("second round");
        assert!(
            matches!(&message.content[0], ContentBlock::Text { text } if text == "second round")
        );
    }

    /// An exhausted script yields an empty assistant message rather than an error, matching
    /// `stream`'s "drains nothing, returns Ok" behaviour.
    #[tokio::test]
    async fn test_mock_provider_complete_on_exhausted_script_is_empty() {
        let provider = MockProvider::from_rounds(vec![]);
        let (message, stop_reason, ..) = provider.complete("", &[], &[]).await.expect("complete");
        assert!(message.content.is_empty());
        assert!(matches!(stop_reason, StopReason::EndTurn));
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
            MockEvent::ThinkingComplete { opaque: None },
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
            Some(StreamEvent::ThinkingComplete { opaque: None })
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
