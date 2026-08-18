//! Anthropic-flavoured providers.
//!
//! Both speak the same protocol -- Anthropic's Messages API, `POST {base}/v1/messages` -- so they
//! share it through [`shared`]. They differ in how they authenticate and in whose client they look
//! like on the wire:
//!
//! - [`messages`]: the Messages API against `api.anthropic.com` or any Anthropic-compatible
//!   endpoint (Ollama, LiteLLM, Databricks, a gateway), authenticated by an API key.
//! - [`subscription`]: the same protocol against Anthropic, authenticated by a Claude
//!   subscription's OAuth tokens and shaped to match the Claude Code CLI exactly -- beta headers,
//!   attestation, injected system block. Deviating from that shape gets the request rejected.

pub mod messages;
mod shared;
pub mod subscription;

pub use messages::AnthropicMessagesProvider;
pub use subscription::ClaudeSubscriptionProvider;
