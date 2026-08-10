# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- OpenAI-compatible streaming dropped a tool call or text that shared a chunk with `finish_reason`.
- REPL text streamed before an interrupt or error no longer leaks into the next turn's output.

## [0.37.0] - 2026-08-10

### Added

- `POST /v1/sessions/{id}/turn` accepts inline `images`, so a remote client can send a picture.
- `GET /v1/info` reports `vision` so clients can check before attaching an image.
- `GET /v1/sessions/{id}` reports `turn_in_flight` so a reconnecting client needn't guess.
- `POST /v1/sessions` accepts `supports_permission_prompts`; false denies gated tools at once.
- MCP `tools/call` now carries `meka/sessionId` in `_meta` so servers can scope session state.
- ACP multi-root workspaces: extra folders are searched and named, not silently dropped.
- Fork a session into an independent copy from the CLI, REPL, HTTP, or ACP.
- ACP clients now see `execute_command` output as it is produced, not only when the command exits.
- Editors that render agent-owned terminals now show shell output live, in a real terminal.

### Changed

- `termimad` render mode is now coloured from the same theme as `syntect` instead of greyscale.
- `termimad` no longer centres a top-level heading, which read as a formatting glitch.
- Tools, skills, and MCP instructions moved from the system prompt into the per-turn context.
- **Breaking:** `-c` takes no session id; use `-r`/`--resume <SESSION>` to resume a specific one.
- **Breaking:** `execute_command` no longer runs in the ACP client's terminal; meka always owns it.

### Fixed

- `--oneshot` now honours `-c`/`-r` instead of silently starting a new session.
- Loading a deferred tool no longer reorders the tools array, which invalidated the prompt cache.
- A new skill or a reconnecting MCP server no longer re-caches the whole conversation.
- ACP clients now see images a tool looked at, instead of an `[image]` placeholder.
- `meka session import` no longer leaves a restored session for retention GC to delete.
- Session export/import no longer drops a session's additional workspace roots.
- Re-attaching a session over HTTP now drops orphaned tool calls that broke its next turn.
- An MCP server that fails its initial connect is now retried in the background until it comes up.
- MCP `tools/call` now actually sends `meka/toolUseId`; it was never populated at the call site.
- A command printing a non-UTF-8 byte had its whole output dropped; it is now decoded lossily.
- `syntect` and `raw` dropped a reply's trailing table or unclosed code fence instead of showing it.
- `termimad` mode parses real CommonMark: `-`/`+` bullets, `__bold__`, `_italic_`, and links.
- `termimad` mode reflows multi-line paragraphs to the terminal width instead of per line.
- `termimad` mode no longer splits a fenced code block that contains a blank line.
- A `#` or `|` line inside a fenced code block no longer gets blank lines inserted into it.

### Security

- ACP `ask` mode delegated shell commands to the client's terminal, bypassing meka's sandbox.

## [0.36.0] - 2026-08-01

### Added

- ACP clients now see a sub-agent's tool calls stream into its `spawn_agent` tool call.
- MCP servers can now prompt for input in an ACP editor; these were previously always declined.
- `session/prompt` responses now carry session-cumulative token usage.

### Changed

- Upgrade `agent-client-protocol` to 2.0, matching the major version Zed uses.
- `search_contents` now stops searching once the inline match cap is exceeded.

### Fixed

- `find_files` and `search_contents` now stop after 60 seconds instead of running unbounded.
- Both search tools now honour Ctrl+C and ACP `session/cancel`.
- A search that was cut short now says so instead of reporting "no matches".
- Searching no longer logs one warning per unreadable path.
- meka can now read and edit files outside the project an ACP editor has open.
- An ACP write that bypassed the editor now says so in the tool result.

## [0.35.0] - 2026-07-24

### Added

- Claude Opus 5 and Sonnet 5 to the model catalog.

### Changed

- Default new Claude provider profiles to `claude-opus-5`.
- Never send reasoning effort to a Claude Haiku model; the tier has no effort knob.

### Fixed

- Don't send `temperature` to Opus 5, Sonnet 5, or any unrecognised model.
- Default Claude Mythos Preview to `high` effort; it supports `max` but not `xhigh`.

## [0.34.0] - 2026-07-22

### Added

- `claude-api` now supports the `output_config.effort` knob.
- Detect a model's context window from the provider API when unknown, cached in the DB.
- `/status` now shows the resolved model, provider, reasoning effort, and thinking state.

### Changed

- **Breaking:** Reject unknown keys in config files instead of silently ignoring them.
- Unify the effort knob into one `effort` config key for all providers.
- Reasoning effort now defaults per model: `xhigh` where supported, else `high`.
- An explicit effort override is absolute: sent verbatim, no validation or clamping.
- `openai-codex` reads its models catalog for the effort default, not a name guess.
- Match Claude Code 2.1.217 request fidelity.

### Removed

- **Breaking:** Remove the `reasoning_effort` config key; use `effort` instead.

### Fixed

- Recognise Claude Opus 4.5 as effort-capable; it was wrongly omitting effort.
- Don't send reasoning effort to OpenAI models that don't support it, including local models.
- Infer real context windows for gpt-5 models, not 128k.
- Infer 1M context for Claude Opus 4.6+/Sonnet 4.6/Fable 5; Haiku and older stay 200k.

## [0.33.1] - 2026-07-15

### Fixed

- Auto-compaction on openai-codex no longer fails; `complete` now aggregates its own SSE stream.

## [0.33.0] - 2026-07-15

### Added

- Codex login accepts a pasted callback URL when the loopback callback can't be reached.
- Recursive sub-agents, bounded by `session.subagent_max_depth` (default 3) and `max_depth`.
- `spawn_agent` gains a `permission` param to restrict a sub-agent below the parent's level.

### Changed

- Default OpenAI model for new provider profiles is now `gpt-5.6-sol`.

### Fixed

- Report Codex tool-call turns as `tool_use`, not `end_turn`, so they no longer warn.
- Retry a transient mid-stream transport error before any output, instead of failing the turn.
- Add a Codex SSE idle timeout so a hung stream fails fast and retries instead of blocking.

## [0.32.1] - 2026-07-06

### Fixed

- Parse Claude's `extra_usage` credit fields as floats.

## [0.32.0] - 2026-07-04

### Added

- Syntax-highlight fenced code blocks by their language in the terminal renderer (syntect mode).

### Changed

- **Breaking:** Rename the `bat` render mode to `syntect`, after the actual highlighter.
- Upgrade `rmcp` to 2.1 and `agent-client-protocol` to 1.0.

### Removed

- **Breaking:** Remove the MCP sampling, roots, and logging handlers and the `sampling` config/CLI.

## [0.31.0] - 2026-07-02

### Added

- `meka session export --format json` writes a structured, round-trippable export (with sub-agents).
- `meka session import <file>` recreates a session tree from a JSON export under fresh IDs.

### Fixed

- Persist user input images instead of dropping them at save time, so they survive resume/export.

## [0.30.0] - 2026-07-02

### Added

- `/usage` command shows account rate-limit windows for `claude-oauth` and `openai-codex`.
- `meka account usage` / `whoami` / `stats` CLI with `--format plain|json` for scripting.
- Retry transient provider errors (429/5xx incl. 529) with bounded backoff, honoring `Retry-After`.

### Fixed

- Handle Claude's mid-stream `event: error` instead of silently dropping it.

## [0.29.4] - 2026-06-30

### Changed

- Resolve relative links in fetched pages to absolute URLs against the page's final URL.

### Fixed

- Preserve `<nav>` / `<footer>` links when fetching a page: those subtrees were dropped as boilerplate.

## [0.29.3] - 2026-06-29

### Fixed

- Capture a stdio MCP server's stderr into tracing instead of letting it corrupt the REPL display.

## [0.29.2] - 2026-06-26

### Changed

- Compaction keeps a token-budgeted recent tail and uses a richer, security-aware summary prompt.
- Compact proactively when the next request would overflow the context window, not only reactively.

### Fixed

- Recover from a provider context-window overflow by compacting once and retrying instead of failing.

## [0.29.1] - 2026-06-25

### Added

- Report context usage to ACP clients via the standard `usage_update` notification.
- Expose `/status` and `/mcp` as ACP slash commands alongside skills.

## [0.29.0] - 2026-06-23

### Added

- Log a warning for unrecognized provider stop reasons (e.g. `pause_turn`) for diagnosability.

### Changed

- Match newest Claude Code request fidelity.

### Fixed

- Recover thinking-only model turns with a one-shot nudge instead of ending the turn with no output.

## [0.28.0] - 2026-06-17

### Added

- REPL now Tab-completes slash-command names and highlights the command token.
- REPL now Tab-completes slash-command arguments (permission levels, skills, MCP servers, /cd paths).
- `meka history list` / `meka history clear` view and clear REPL input history.
- `recall` / `recall_read` tools search and read the full conversation, including compacted turns.

### Changed

- Session subcommands `list`, `export`, and `delete` moved under `meka session` for consistency.
- `meka session export` now exports the full conversation, including turns hidden by compaction.

## [0.27.3] - 2026-06-11

### Changed

- Detect adaptive thinking / effort by excluding known pre-4.6 models, not gating on `>= 4.6`.

### Fixed

- Execute tool calls present in the assistant message regardless of the reported stop reason.
- Show a stand-in message when the model refuses with empty content, instead of a blank turn.

## [0.27.2] - 2026-06-01

### Added

- ACP editor integration gained plans, embedded resource/image input, titles, and tool-call detail.
- `[providers.<name>]` gains `context_window`, `vision`, and `max_output_tokens` overrides.

### Fixed

- Gate adaptive thinking / effort on the parsed Claude model version (`>= 4.6`), not an allowlist.
- REPL log warnings now appear during a turn instead of being buffered until the next prompt.
- Interrupting a turn now persists the partial assistant text so it survives resume.
- `meka acp` exits on client disconnect (stdin EOF) or SIGTERM/Ctrl-C, releasing its session lock.

## [0.27.1] - 2026-05-29

### Fixed

- `!` shell escape and `scratchpad_load_file`/`save_file` now honour `/cd` instead of the process cwd.

## [0.27.0] - 2026-05-29

### Added

- `meka acp` subcommand for editors that speak the Agent Client Protocol.
- `meka serve` subcommand exposes the agent over HTTP+JSON.
- `meka provider` suite (add/list/use/login/remove) to configure and switch named provider profiles.
- REPL input history persists across runs in the SQLite DB.
- `MEKA_SANDBOX_BACKEND` overrides `[shell].sandbox_backend`; mekabox uses it to pin Landlock.
- `--sandbox-backend` flag, so the backend is settable via config, env, and CLI consistently.
- `MEKA_RENDER_MODE` overrides `[display].render_mode` for CI / non-TTY runs.
- `/status` shows live context-window usage (tokens / window, percent used, tokens left).
- Cumulative `/status` stats now persist per session and continue across resume.
- `display.show_context_in_prompt` shows a live context gauge in the REPL prompt (opt-in).

### Changed

- Renamed the project `agsh` → `meka`: binary, `~/.config/meka` config dir, `MEKA_*` env vars.
- Renamed the database `sessions.db` → `meka.db`; it now holds more than sessions.
- Providers are now named `[providers.<name>]` profiles with secrets stored in the DB, not config.
- `serde_yaml` (unmaintained) replaced with the maintained `serde_norway` fork.
- `edit_file` now rejects an ambiguous `old_string` (multiple matches without `replace_all`).
- Replaced `todo_write`/`todo_read` with one `todo` tool: `title`, `set` patches, `cancelled`.

### Removed

- `meka setup` wizard and all provider env vars (`MEKA_PROVIDER`, `MEKA_MODEL`, API keys, tokens).
- `[agent] max_turn_requests` cap; it was cutting off legitimate long-running workflows.

### Fixed

- User message persists eagerly so a crash mid-turn no longer loses it.
- OpenAI streaming now requests token usage (`stream_options.include_usage`); it previously reported zero.
- Claude streaming usage is merged across `message_start`/`message_delta` instead of last-event-wins.
- Auto-compact now measures total context tokens (all tiers + output), correct with Claude caching.

## [0.26.2] - 2026-05-22

### Added

- `spawn_agent` accepts a `skill` parameter to run an installed skill in the sub-agent.

### Changed

- Loaded skill bodies now lead with the skill's base directory so bundled files resolve.

## [0.26.1] - 2026-05-21

### Added

- CI runs `cargo audit` to flag known security advisories in dependencies.

### Changed

- Stream-event channel is now bounded; in-memory event log is pruned after compaction to bound memory.
- `grep` traverses directories iteratively, so a deeply-nested tree can't overflow the stack.
- `grep` no longer descends into symlinked directories, removing any symlink-cycle traversal risk.

### Fixed

- Large-output shell commands no longer spuriously time out — stdout/stderr are drained before the wait.
- Malformed OpenAI tool-call arguments are rejected explicitly instead of run with empty input.
- `write_file` rejects symlinked targets on Windows, matching the `O_NOFOLLOW` behavior on Unix.
- Landlock sandbox (ABI v6+) now blocks abstract Unix sockets and cross-domain signals.
- Blocking skill-discovery, skill-load, and OS-detection calls no longer stall the async runtime.
- Session-lock guard drop order is now explicit, removing a field-reorder use-after-free hazard.
- Numeric casts on tool inputs (offsets, limits, sizes) are bounds-checked instead of overflowing.
- `todo_write` rejects an unrecognized task status instead of silently mis-rendering it.

## [0.26.0] - 2026-05-20

### Added

- `agsh skill update` re-fetches skills from their `source_url` and replaces them on disk.
- Skill frontmatter gains optional `author` and `source_url` fields.

### Changed

- CLI list tables (`skill`/`mcp list`, `mcp tools`, `list`) share one column formatter with dynamic widths.

### Removed

- Skill frontmatter fields `when_to_use`, `allowed_tools`, and `user_invocable`.

### Fixed

- Orphaned session lock files are pruned at startup and after deletions instead of accumulating.

## [0.25.2] - 2026-05-20

### Changed

- Setup wizard no longer prompts for a sandbox backend — it auto-detects at runtime.

### Fixed

- Ctrl+C now interrupts turns started by `/skill` and `/mcp prompt` (and their sub-agents).

## [0.25.1] - 2026-05-19

### Added

- `scratchpad_merge` combines multiple entries into one without routing bytes through context.

### Changed

- `find_files` default cap raised to 500, with a `limit` param; truncation reports the real total.
- `write_file` marks its target as read so `edit_file` no longer needs `force: true` after.
- `read_file`/`find_files`/`search_contents`/`execute_command` descriptions note parallel dispatch.

## [0.25.0] - 2026-05-19

### Added

- `spawn_agent` gains `inherit_scratchpad`: grant the sub-agent read-only access to parent entries.
- `scratchpad_load_file` streams a file into the scratchpad without routing bytes through context.
- `scratchpad_save_file` writes a scratchpad entry to disk without routing bytes through context.
- `scratchpad_rename` renames an entry in place without round-tripping content through the model.

### Changed

- Sub-agents now run unbounded; the prior 20-round cap is removed, no replacement knob.
- `scratchpad_list` renders own and inherited entries in one table with an `Origin` column.
- Scratchpad tool output reports sizes in bytes (was mislabeled "characters" — always byte counts).

### Fixed

- Sub-agent writes to inherited scratchpad names now error instead of silently shadowing the parent.

## [0.24.1] - 2026-05-19

### Fixed

- Schema upgrade from a 0.23.x DB no longer fails with `no such column: parent_session_id`.

## [0.24.0] - 2026-05-19

### Added

- Sub-agent sessions persist as DB children for auditing; `agsh list --include-children` to view.
- Sub-agents now get `load_tool`, `render_image`, and all scratchpad tools (scoped to their own session).
- `RenderMode::Silent` suppresses all agent output; used by sub-agents.

### Changed

- Sub-agents inherit the parent's MCP tools.
- Sub-agents now run on `Agent::run_turn`; bespoke loop removed.
- Sub-agent token usage now rolls into the parent's `/status` totals.
- Added `idx_sessions_updated_at` so `list`, `resume`, and prune skip the temp sort.
- Session deletion / pruning / storage-limit eviction now rely on `ON DELETE CASCADE`.

### Removed

- Unused `sessions.metadata` column.

### Fixed

- Enable `PRAGMA foreign_keys = ON` so `messages` / `tool_outputs` FK clauses are enforced.

## [0.23.1] - 2026-05-17

### Changed

- Skills cached at startup with mtime-based auto-reload; parse warnings now fire at startup.

### Fixed

- `/skill <name>` error paths no longer hang the REPL when the skill is missing or non-invocable.

## [0.23.0] - 2026-05-17

### Added

- `[shell].sandbox_backend` selects between `"landlock"` and `"bubblewrap"` for Linux read-mode sandboxing.
- Setup wizard prompts for the Linux sandbox backend when both options are available.
- `todo_read` tool lets the model fetch the current task list on demand.
- Tool calls within one assistant message now dispatch in parallel, including multiple `spawn_agent` calls.

### Changed

- Linux read-mode sandbox auto-uses Bubblewrap when installed; set `sandbox_backend = "landlock"` to opt out.
- `execute_command` in read mode now hard-errors when the configured sandbox backend is unavailable.
- Sub-agents inherit the parent's permission level instead of being capped at read.
- Each sub-agent has a private todo list; `todo_write` from a sub-agent no longer renders to the user.

### Fixed

- Windows command timeout now kills the full process tree via a Job Object.
- Session DB path no longer falls back to a Linux-only default on macOS/Windows; set `AGSH_DATA_DIR`.

### Security

- macOS read-mode sandbox profile hardened: IPC mutation now blocked alongside filesystem writes.
- `sandbox-exec` invoked via absolute path `/usr/bin/sandbox-exec` instead of `$PATH` lookup.
- Read-mode shell now scrubs the child environment on Linux and macOS (Windows already did).

## [0.22.1] - 2026-05-12

### Added

- `--eager-load-tool SERVER:TOOL` adds session-only entries to a server's `eager_load_tools` list.

### Changed

- CLI `-h` output tightened to fit 80 columns across every subcommand.

## [0.22.0] - 2026-05-11

### Added

- `edit_file` gained `insert_before` / `insert_after` for anchor-based inserts without rewriting context.
- `read_file` gained a `regex` parameter mirroring `scratchpad_read`'s line-grep mode.
- Per-server `eager_load_tools` lets named MCP tools skip `load_tool` and ship in the cacheable prefix.
- `/history [N]` and `[display].resume_show_recent` reprint past turns in REPL style.

### Changed

- `edit_file` success responses now include a ±3-line snippet around the first edited site.
- `scratchpad_read`, `_edit`, `_list`, `_delete` ship default-active (no `load_tool` round-trip needed).

## [0.21.1] - 2026-05-10

### Changed

- Reqwest error messages now expose the full source chain (timeout, reset, TLS, etc.).

## [0.21.0] - 2026-05-10

### Added

- `agsh -c <prefix>` resumes a session by UUID prefix; ambiguous prefixes list matches.
- `openai-codex` provider sends tool-result images as `input_image` blocks (Responses API).

### Fixed

- Images >2000 px on either axis are downscaled in the Claude request path (Anthropic multi-image cap).

## [0.20.0] - 2026-05-09

### Added

- `/status` slash command shows turns, tokens, cache hit ratio, redactions, and message count.
- `[display].show_token_usage` toggles a per-turn `[in / cache hit % / out]` line on stderr.
- `TokenUsage` now carries `cache_creation_input_tokens` and `cache_read_input_tokens` from Anthropic.

### Changed

- Image redaction now drops to a watermark (~24 MiB) instead of the minimum, amortizing cache invalidation.
- Image redaction now prints a stderr advisory when it fires; was previously invisible at default verbosity.

### Fixed

- Claude requests reactively redact oldest tool-result images when body exceeds 30 MiB.

## [0.19.0] - 2026-05-07

### Added

- `--skill <NAME>` invokes a user-invocable skill as the first turn; `[PROMPT]` is prepended.
- `--oneshot` flag exits after the first turn finishes; requires `[PROMPT]` or `--skill`.

### Changed

- Bare `[PROMPT]` / `--skill` now drop into the REPL after the first turn.
- Tool indicators, thinking, todos, spacing newlines, setup prompts, OAuth URLs output to stderr.

### Fixed

- OAuth refresh re-reads the latest token from the DB, fixing `invalid_grant` between concurrent instances.

## [0.18.4] - 2026-05-04

### Added

- `--instructions` / `AGSH_INSTRUCTIONS` overrides `[prompt].instructions` for one run.
- `agbox` sets `AGSH_INSTRUCTIONS` so the agent knows it can install packages freely.

## [0.18.3] - 2026-05-02

### Fixed

- Skill discovery skips dot-prefixed entries (`.git`, `.vscode`, `.DS_Store`, etc.) instead of warning.

## [0.18.2] - 2026-05-01

### Security

- `canonicalize_for_tool` now errors on resolution failure; `write_file` canonicalizes the parent.
- JWT signing-key permissions now checked on the open `File` to close the stat-then-read TOCTOU.
- `search_contents` rejects invalid glob patterns instead of silently scanning the whole tree.
- OAuth callback `code`/`state`/`error` parameters are decoded with strict UTF-8, not lossy.
- Session DB pre-touched at 0600 and data/lock/config dirs born at 0700 to close umask windows.
- `set_permissions` failures on the config directory now log a warning instead of being discarded.
- `.expect()` panics on tool registration and compaction-boundary lookup replaced with `?`.
- New `AgshError::Internal` variant for logic-invariant failures that previously panicked.
- MCP tool annotation/meta serialization failures now warn-log instead of being silently dropped.
- `libc::kill` failures during process-group teardown now logged at `debug!`.

## [0.18.1] - 2026-04-30

### Added

- `[permissions]` config: pick enabled modes and start mode; `ask` is now opt-in by default.

## [0.18.0] - 2026-04-29

### Added

- `agsh skill list | get | show | add | remove` CLI subcommands for managing user skills.
- `/skill` REPL command: bare form lists skills; `/skill <name> [extra...]` invokes one,
  prepending any free-form extra text as the user's directive above the skill body.
- `--edit` flag on `agsh skill add` opens the new `SKILL.md` in `$EDITOR` after scaffolding.
- `--from-file` on `agsh skill add` copies an existing template instead of scaffolding from flags.

### Changed

- `/skill <name>` rejects skills marked `user_invocable: false` (gate now consumed).
- `/help` now lists `/skill` and `/mcp` slash commands (previously omitted).
- `scratchpad_read` description and `<large-output>` preview advertise no hard cap on `limit`.

## [0.17.2] - 2026-04-28

### Fixed

- Pinned reedline to a fork containing the fix for upstream `nushell/reedline` issue #1005.
- Long log lines through `ExternalPrinter` no longer trigger an apparent screen clear on REPL start.

## [0.17.1] - 2026-04-28

### Fixed

- Startup log lines no longer get clobbered by reedline's prompt redraw.
- `tracing` output flows through reedline's `ExternalPrinter` and prints above the live prompt.

## [0.17.0] - 2026-04-28

### Added

- `load_tool` meta-tool: exposes a deferred tool's schema for use on the next turn.
- `## Tool Discovery` system-prompt section: deferred tools grouped by source.
- `Conversation` newtype wraps the message log; only `append` plus three named methods mutate it.
- Event-sourced conversation persistence: `Vec<Event>` (`Append` + `CompactBoundary`).
- `CompactBoundary::loaded_tools_snapshot` carries the active deferred-tool set across compaction.

### Changed

- Deferred tools are activated by `load_tool` calls in the conversation (no in-memory state).
- System prompt is byte-stable across deferred-tool activation (cache breakpoint 2 stays warm).
- Resumed sessions reconstruct the active tool set from the conversation — no out-of-band state.
- REPL `agsh mcp tools` STATUS column renamed to VISIBILITY.
- `compact_session` appends a `compact_boundary` row instead of DELETEing — log stays append-only.
- All conversation persistence flows through `save_event` / `load_events`.
- Terminology unified: `Conversation` (type), `Event` (storage atom), `Message` (API atom).

### Removed

- `ToolRegistry::activate()` and dispatch-side auto-promotion of deferred tools.
- `SessionManager::clear_messages_only` — no caller after the event-log refactor.
- `pub` visibility on `save_message` / `load_messages` / `StoredMessage` — internal helpers now.

## [0.16.1] - 2026-04-26

### Fixed

- `Continuing session: ...` notice now respects `[display].newline_after_prompt`.

## [0.16.0] - 2026-04-25

### Added

- `openai-codex` provider: ChatGPT subscription auth via OpenAI Responses API.
- `OPENAI_CODEX_TOKEN` env var and `CODEX_CLIENT_ID` override for the Codex login flow.
- `agsh setup` wizard now offers a "ChatGPT subscription login" option.
- `[provider].effort` (claude-oauth): `output_config.effort` low/medium/high. Default high.
- `[provider].redact_thinking` (claude-oauth): send `redact-thinking-2026-02-12`. Default false.
- `[provider].device_id` (claude-oauth): override the persistent `metadata.user_id` device ID.

### Changed

- MCP tool namespace is now `mcp__<server>__<tool>`; matches Claude Code.
- Renamed provider `openai` → `openai-api` (room for a future Codex provider).
- Split provider `claude` into `claude-api` (API key) and `claude-oauth` (Claude Code OAuth).
- OAuth refresh tokens are preserved across the `claude` → `claude-oauth` rename.
- `claude-api` reads `CLAUDE_API_KEY` (no longer reads `ANTHROPIC_API_KEY`).
- `claude-oauth` wire format matches recent Claude Code (betas, context, fingerprint, cache, effort).
- `device_id` is generated/persisted only when the active provider is `claude-oauth`.
- `device_id` seeds from `~/.claude.json`'s `userID` when unset before generating a random one.
- `AuthCredential::OAuthToken` gains optional `account_id` for `openai-codex`'s account header.
- `oauth_tokens` table gains an `account_id` column; existing rows migrate with `NULL`.
- `openai-codex` reqwest client enables cookie jar so chatgpt.com bot-clearance cookies persist.
- `src/mcp.rs` (4754 lines) split into `mcp::{auth, transport, connector, handler}` submodules.
- `src/provider/claude/oauth.rs` (3286 lines) split into `oauth::attestation` + `claude::shared`.
- `src/config.rs` device_id / effort / credential helpers grouped into private inline submodules.
- `create_provider` replaced by `ProviderBuilder` (13 positional params → per-field setters).
- `claude-oauth` error-path body reads log at `warn!` on IO failure instead of silent fallback.
- MCP progress/elicitation sends log at `debug!` when the REPL receiver has been dropped.

### Fixed

- Missing `provider.name` errors with "no provider configured" before credential resolution.
- Unsupported `provider.name` errors with the list of valid providers.

## [0.15.1] - 2026-04-22

### Changed

- Tool-call indicators show the first required arg for MCP tools, not just built-ins.

## [0.15.0] - 2026-04-22

### Added

- `[display].input_style = "reverse"` uses ANSI reverse video (swaps terminal fg/bg).
- `[mcp].strict`, `grace_seconds`, `connect_timeout_seconds` tune the per-turn readiness gate.
- `[[mcp.servers]].disabled` skips a server at startup without removing it from config.
- `agsh mcp disable <name>` / `agsh mcp enable <name>` toggle the disabled flag in config.toml.
- `agsh mcp add --disabled` stages a server without connecting to it on the next start.
- `web_search` detects DuckDuckGo CAPTCHA pages and returns a clear error instead of silent empty.
- `[web]` gains reqwest knobs: request/connect/read timeouts, max redirects, proxy, CA bundle, TLS.

### Changed

- MCP servers connect in parallel in the background; REPL opens immediately, not after Σ(connect).
- Default strict gate: turns abort when any enabled MCP server isn't connected.
- `/mcp list` in the REPL shows live state (connected / pending / failed / disabled) per server.
- `web_search` output: normalized whitespace, source-domain line, bold markdown on matched terms.

### Removed

- `web_search` Google and Bing engines (both consistently bot-blocked).

## [0.14.0] - 2026-04-20

### Added

- `[tools]` config: `allowed_tools`, `disabled_tools`, and `tool_permissions` filters for built-in tools.
- `agsh tools list` prints every built-in tool with its effective permission and enabled state.
- `[display].input_style` styles REPL input so submitted prompts stand out in scrollback.

## [0.13.1] - 2026-04-20

### Changed

- `agsh mcp tools --help` description trimmed to a single line.
- Renamed `src/shell.rs` → `src/repl.rs` and `src/mcp/env.rs` → `src/mcp/expand.rs` for clearer module names.

## [0.13.0] - 2026-04-19

### Added

- `AGSH_CONFIG_DIR` env var overrides the default config directory on every platform.
- System prompt now lists every registered tool with its required permission level inline.
- Per-turn user message carries a `[Permission context]` block naming the current level.
- Per-tool MCP permission chain: `tool_permissions` > `permission` > `readOnlyHint` > `default_permission`.
- `[mcp] default_permission` config key: global fallback when no server/tool/hint applies.
- `[[mcp.servers]]` supports `allowed_tools` / `disabled_tools` / `tool_permissions` overrides.
- `agsh mcp add` flags: `--allow-tool`, `--disable-tool`, `--tool-permission NAME=LEVEL` (repeatable).
- `agsh mcp get <name>` now lists allow/block lists and per-tool permission overrides.
- Stale entries in `allowed_tools`/`disabled_tools`/`tool_permissions` emit a `warn!` at connect time.
- `agsh mcp tools <name>` lists every advertised tool with resolved permission and which chain step won.
- `agsh mcp` CLI: `list`, `get`, `add`, `remove`, `reconnect`, `login`, `logout` subcommands.
- `agsh mcp add <name> <url-or-command> [args]` auto-detects transport (URL → http, else stdio).
- `agsh mcp add` flags for env/headers, permission, auth (oauth, client-credentials, -jwt, token).
- `agsh mcp add` probes HTTP servers post-persist (RFC 6750 / RFC 9728): 3 s redirects-off GET.
- `agsh mcp add` auto-runs OAuth on auth-required / `--auth oauth`; `--no-login` skips.
- `agsh mcp add` auto-login failure or Ctrl-C rolls the entry back (config + creds + probe cache).
- `agsh mcp login <name>` assumes OAuth authorization_code on HTTP servers without an `[auth]` block.
- OAuth callback races the bound TCP listener against a stdin paste so logins work over SSH.
- `/mcp login <server>` and `/mcp logout <server>` REPL commands mirror the CLI subcommands.
- Server `InitializeResult.instructions` spliced into the system prompt each turn.
- Progress notifications forwarded to the REPL as a live status line under the running tool call.
- Form + URL elicitation — the shell prompts the user and returns typed values to the server.
- Tool annotations / `_meta` / `structuredContent` preserved through to the provider.
- Builtin MCP resource/prompt tools for list/read, subscribe/unsubscribe, and get-prompt flows.
- OAuth token revocation via `agsh mcp logout` (RFC 7009) + 15-min auth-probe cache for 401s.
- Tool-call timeout (`AGSH_MCP_TOOL_TIMEOUT`, default 600s) with best-effort cancellation.
- Exponential-backoff reconnect for HTTP MCP (5 attempts, 1s → 30s); stdio retries once.
- `${VAR}` / `${VAR:-default}` expansion across MCP command, args, env, url, headers, auth_token.
- `headers_helper` config field: per-server script emits dynamic HTTP headers at connect-time.
- Windows stdio: auto-wrap `npx`, `.cmd`, `.bat`, `.ps1` commands in `cmd /c`.
- Unicode + server-name sanitisation of MCP strings; `agsh`, `ide`, `mcp_*` names rejected.
- `sampling/createMessage` server-to-client flow, opt-in via `sampling = true` + `sampling_limit`.
- `roots/list` advertises the agsh current working directory.
- MCP image tool-result content reaches providers as image blocks instead of `[image content]`.
- OAuth callback listener binds to an ephemeral port when `redirect_port` is omitted.
- Ctrl-C now sends `notifications/cancelled` to the server with the in-flight request id.
- Dynamic tool list refresh on `tools/list_changed` — new tools picked up without restart.

### Changed

- `execute_command` description names the shell per platform and warns against double-PowerShell wrapping.
- Per-turn `[Permission context]` is a constant two-line block; no longer enumerates blocked tools.
- System prompt tool catalogue is leaner: name + permission for active tools, short summaries for deferred.
- System prompt and `body["tools"]` no longer depend on permission level; toggles keep the cache warm.
- **Breaking**: MCP tools with no `readOnlyHint` and no `[mcp].default_permission` now require `Write`.

### Fixed

- `${VAR}` expansion for MCP config preserves multi-byte UTF-8 (previously corrupted non-ASCII).
- MCP tools with an unserializable input schema are skipped with a warning.
- OAuth-authenticated MCP transports now reconnect cleanly mid-session.
- MCP `sampling/createMessage` has a 60 s provider timeout and refunds the sampling slot on error.
- `agsh mcp remove` now clears that server's entries from the resource-update ledger.
- `agsh mcp remove` now also best-effort revokes stored OAuth tokens at the provider (RFC 7009).
- MCP auth-probe cache with `ttl = 0` now correctly treats every entry as stale.
- rmcp's SSE-reconnect warning floored at `error` in default filter; CDN idle resets no longer spam.

### Security

- MCP progress + elicitation strings sanitised before reaching the terminal; no ANSI/RTL spoofing.
- MCP tool-result images capped at 10 MiB and restricted to PNG/JPEG/GIF/WebP; else a placeholder.
- MCP sampling `system_prompt` stripped of Cc/Cf codepoints before reaching the provider.
- `read_mcp_resource` + `get_mcp_prompt` + list tools sanitise server-supplied text and URIs.
- `read_mcp_resource` total output capped at 10 MiB; oversized chunks replaced with a marker.
- `headers_helper` stdout capped at 64 KiB, stderr at 4 KiB, to contain helper misbehaviour.
- OAuth revocation rejects redirects, caps metadata at 256 KiB, pins endpoint to issuer origin.
- OAuth callback `error=…` query parameter is stripped of Cc/Cf codepoints before display.
- JWT signing key files rejected on Unix when group/other perm bits are set (must be 0600).
- MCP cancellation notifications now time out after 2 s so a hung transport can't stall Ctrl-C.
- `agsh mcp add`/`remove` writes config.toml atomically and chmods it 0600 (dir 0700) on Unix.
- `agsh mcp add` propagates config-read errors instead of silently treating them as an empty file.

## [0.12.0] - 2026-04-18

### Added

- `tests/cli.rs` end-to-end smoke tests for `--version`, `--help`, unknown flags.
- `render::render_error` and `render::render_provider_setup_hint` helpers for CLI output.
- Module-level `//!` doc comments across the codebase; CI runs `cargo doc -D warnings`.
- CI test job runs on Linux, macOS, and Windows to cover platform-specific sandbox code.
- Windows `execute_command` sandbox via Low-integrity token with handle-list inheritance filter.
- Windows sandbox falls back to `CreateProcessWithTokenW` when `SE_INCREASE_QUOTA_NAME` is missing.

### Changed

- Session locking uses OS file locks via `fd-lock` so kernel-released locks survive hard kills.
- `SessionManager::lock_session` is now sync and returns a `SessionLock` RAII handle.
- Schema migration drops the legacy `sessions.locked_by` column to unstick old sessions.
- `execute_command` on Windows invokes PowerShell with `-NoProfile -NonInteractive` always.
- `execute_command` children no longer inherit the agent's stdin; they see immediate EOF.

### Fixed

- `default_database_path` falls back to `$HOME/.local/share` and errors cleanly when unset.
- Stuck sessions from PID-based locking surviving hard kills (resolved via OS file locks).
- Windows sandbox normal-exit drain now times out after 5s instead of hanging on a grandchild.

### Security

- File tools route I/O through the canonical path with `O_NOFOLLOW` on Unix, closing a symlink-swap TOCTOU.
- `fetch_url` caps response body at 10 MiB to defend against gzip/brotli decompression bombs.
- Session data dir, lock dir, and DB file are created 0700/0700/0600 on Unix regardless of umask.
- Tool calls with unparseable JSON arguments are now rejected instead of silently run with `{}`.
- Windows Low-integrity sandbox scrubs the child environment so provider API keys aren't inherited.
- `execute_command` on Unix kills the whole process group on timeout so grandchildren can't outlive it.
- LLM-supplied regex patterns are compiled with 1 MiB size/DFA limits to bound compile-time memory.
- Tool indicators strip ANSI CSI escapes and C0 controls so commands can't spoof the permission prompt.
- Permission enforcement now reads the shared permission atomically at the dispatch site.

## [0.11.0] - 2026-04-17

### Added

- `skill` tool for loading named skills.
- YAML frontmatter for skills (description, when_to_use, allowed_tools, version, user_invocable).
- `${AGSH_SKILL_DIR}` and `${AGSH_SESSION_ID}` substitution in skill bodies.
- `[prompt] instructions` config for system-wide instructions injected into every session's prompt.
- `fetch_url` returns a multimodal Image block for image URLs (sandboxed mode, no disk I/O).
- `fetch_url` and `read_file` convert TIFF, ICO, HDR, EXR, TGA, PNM, QOI, DDS, Farbfeld to PNG.
- `render_image` tool views in-memory base64 or scratchpad bytes as a multimodal Image block.

### Changed

- Skills are now directory-based (`~/.config/agsh/skills/<name>/SKILL.md`), not flat files.
- System prompt lists skills by description and when_to_use; agent invokes via `skill` tool.
- `find_files` and `search_contents` descriptions recommend narrow searches, broadening gradually.
- Tool output redirected to scratchpad is never truncated; internal caps are lifted.
- Highlight markdown with `syntect` directly instead of `bat`; reprints are roughly 50x faster.
- Embed Monokai Extended theme from bat for visual parity with the old renderer.
- Drop the `Last message:` banner on session resume; the resuming-session line is sufficient.

### Fixed

- macOS/Windows CI tests no longer read the host user's real `config.toml` — they now isolate via `AGSH_CONFIG_DIR`.
- `cargo doc -D warnings` cleared of broken intra-doc links and bare-URL lints.
- Rename `render_image` input `scratchpad` to `from_scratchpad` so it no longer clobbers the source.
- Remove redundant 30 KB caps on `execute_command` and `spawn_agent`; oversize handled upstream.
- Show primary param in the tool banner for `skill` and `render_image`.

### Security

- Omit environment info (PWD, date, shell, OS) from prompts in `none` permission mode.

## [0.10.3] - 2026-04-14

### Fixed

- Fix newlines in tool/ask banners breaking single-line display.

## [0.10.2] - 2026-04-14

### Added

- CI workflow for `cargo fmt --check`, `cargo clippy`, and `cargo test`.
- Tests for `validate_tool_use_chains` in session resume.
- `SessionLockGuard` for panic-safe session unlocking.

### Changed

- Replace `let _ =` silent error discards with explicit handling.
- Extract CSS selectors to `LazyLock` statics in web search parsing.
- Deduplicate tool registration via shared `register_core_tools` helper.
- Replace busy-wait polling with blocking `recv()` in REPL event loop.
- Flatten `execute_tool_calls` into smaller helper methods.
- Resolve all clippy warnings (collapsible ifs, ptr_arg, etc.).

### Fixed

- Add `// SAFETY:` comment to `libc::kill` in session locking.

## [0.10.1] - 2026-04-14

### Fixed

- Fix code blocks rendered without newlines in bat mode.
- Fix extra blank lines after trailing code blocks.
- Fix blank lines between code blocks and surrounding content.

## [0.10.0] - 2026-04-14

### Added

- `/export` slash command to export the current session as Markdown.
- Re-print last message when resuming a session with `-c`.
- Adaptive thinking for Claude 4.6+ models.
- `set_thinking_override` on Provider trait for compaction.
- Optional `reasoning_effort` config for OpenAI o-series models.

### Changed

- Combine `-s` and `-c` CLI flags into `-c [SESSION_ID]`.
- Wrap injected context in `<context>` XML tags for structured parsing.
- Thinking enabled by default (was disabled).
- Default thinking budget: 10K → 16K tokens.
- Default max_tokens: 8K → 32K (non-thinking), dynamic (thinking).
- Preserve thinking blocks in conversation history for Claude API.
- Disable thinking during session compaction.
- Updated context window defaults for GPT-4.1 (1M) and o-series (200K).

### Fixed

- Session list preview now shows actual user input instead of "[Environment context]".

## [0.9.4] - 2026-04-14

### Added

- Output spacing state machine replacing ad-hoc separator flags.
- Blank line after tables in buffer via `normalize_spacing`.
- Validation of tool_use/tool_result chains on session resume.
- Warnings for unparseable messages during session loading.

### Fixed

- Fix missing blank line between tool batches and following text.
- Fix double blank line after todo list before text responses.
- Fix table not followed by blank line in bat render mode.
- Fix `normalize_spacing` splitting tables on incomplete streaming rows.
- Orphaned tool_use blocks no longer cause API errors on resume.

## [0.9.3] - 2026-04-13

### Added

- Table pretty-printing (column alignment) in bat render mode.

### Fixed

- Fix table column misalignment with emoji/wide Unicode characters.

## [0.9.2] - 2026-04-13

### Fixed

- Restore blank line after todo list to separate it from following tool calls.

## [0.9.1] - 2026-04-13

### Fixed

- Remove blank lines between consecutive tool call batches.
- Fix double blank line after todo list display.
- Blank line before text only prints when transitioning from tools.

## [0.9.0] - 2026-04-13

### Added

- `bat` render mode as the new default with syntax-highlighted markdown.

### Changed

- Rename `rich` render mode to `termimad` (`rich` kept as alias).
- Ensure blank line after markdown headers in bat/raw modes.
- Ensure proper spacing around tool indicator batches.

## [0.8.1] - 2026-04-13

### Changed

- Compaction now uses a structured summary prompt with 6 sections.
- Compaction preserves scratchpad entries and recent messages.
- Compaction re-injects environment, todos, and scratchpad inventory.
- Images and large text blocks stripped before summarization.

## [0.8.0] - 2026-04-13

### Added

- `replace_all` parameter for `edit_file` tool to replace all occurrences in a file.
- `force` parameter for `edit_file` tool to bypass the read-before-edit requirement.
- Read-before-edit enforcement: `edit_file` requires `read_file` on the same path first.
- `todo_write` tool for structured task tracking within a session.
- Ask permission mode (`a`): prompts user to approve/deny each tool call individually.
- Extended thinking support for the Claude provider (`[thinking]` config section).
- Image multimodal support: `read_file` returns base64-encoded images for `.png`/`.jpg`/`.gif`/`.webp`/`.bmp`.
- `TokenUsage` tracking parsed from Claude and OpenAI API responses.
- Auto-compact: automatically compacts conversation when input tokens exceed 80% of context window.
- `spawn_agent` tool for delegating research tasks to read-only sub-agents.
- Deferred tool loading: MCP tools listed in system prompt but schemas sent on first use.
- `raw` parameter for `fetch_url` tool to return untreated HTML instead of markdown.
- Scratchpad provides session-scoped, name-keyed agent working memory.
- `scratchpad_write`, `scratchpad_read`, `scratchpad_edit`, `scratchpad_list`, `scratchpad_delete` tools.
- `scratchpad` parameter on all tools to save output directly.
- Auto-persist for oversized tool results (>30K chars) with `{tool}_{N}` naming.
- Per-tool output caps to prevent context overflow.
- `read_file` defaults to 2000 lines and rejects images over 3.75MB.
- Session export resolves persisted large outputs back to full content.

### Changed

- Permission levels expanded from 3 to 4: none, read, ask, write.
- `ToolResult.content` changed from `String` to `Vec<ToolResultContent>` for multimodal support.
- `Provider::complete()` now returns `TokenUsage` alongside the message and stop reason.
- `edit_file` success message now reports the number of replacements made.
- Tool outputs tied to session lifecycle: deleted with session/messages cleanup.

## [0.7.1] - 2026-04-04

### Changed

- Optimize prompt caching to avoid unnecessary KV cache invalidation across turns and tool-use loops.

## [0.7.0] - 2026-04-04

### Changed

- Adapted Claude provider to match current claude-code header and attestation requirements.

## [0.6.1] - 2026-03-28

### Fixed

- Fixed build failure with rmcp 1.3.0 by using `OAuthClientConfig` builder API.
- OpenAI provider not parsing top-level `name`/`arguments` in proxy tool call responses.

## [0.6.0] - 2026-03-25

### Added

- Shift+Enter as an alternative to Alt+Enter for inserting newlines in the REPL.
- `headers` parameter for `fetch_url` and `web_search` tools to set custom HTTP headers.
- `regex` parameter for `fetch_url` tool to filter page content by pattern.

### Changed

- Changed default web user agent to Chrome for better content fetching success rates.

## [0.5.3] - 2026-03-18

### Changed

- Reduced `fetch_url` default `max_length` from 50000 to 30000.

### Fixed

- User prompts are no longer recorded in history when the server returns an error.
- The blank line after the agent's response is now printed even when an error occurs.
- Partial assistant responses are now saved to history on Ctrl+C interrupt.

## [0.5.2] - 2026-03-17

### Changed

- `fetch_url` tool accepts optional `max_length` parameter (default: 50000, 0 for no limit).

## [0.5.1] - 2026-03-17

### Changed

- Generate dynamic billing header with content-based hashing for Claude OAuth requests.
- Replaced custom HTML search result parsers with `scraper` crate for CSS selectors.
- Replaced custom `urldecode` with `percent-encoding` crate (already a transitive dep).
- Replaced custom `ceil_char_boundary` utility with stdlib `str::ceil_char_boundary`.
- Reuse a single `reqwest::Client` for web tools instead of constructing one per request.
- Extracted duplicated timestamp calculation in Claude provider into a helper function.

### Fixed

- Claude OAuth requests failing with 400.
- `urldecode` incorrectly handling multi-byte UTF-8 percent-encoded sequences.

## [0.5.0] - 2026-03-16

### Added

- OAuth auth for MCP HTTP servers: client credentials, JWT signing, and PKCE.
- Persistent MCP OAuth credential storage in SQLite with automatic token refresh.

### Changed

- Default render mode changed from `raw` to `rich`.
- Raw render mode now prints output verbatim, only formatting tables with aligned columns.
- Upgraded `reqwest` from 0.12 to 0.13.

### Removed

- Custom raw mode ANSI markdown renderer (replaced with passthrough + table alignment).
- `unicode-width` direct dependency.

### Fixed

- Trailing newlines in agent responses causing duplicate blank lines before the next prompt.

## [0.4.1] - 2026-03-14

### Added

- `display.show_path_in_prompt` config to toggle working directory in the prompt.

## [0.4.0] - 2026-03-14

### Added

- Working directory displayed in the shell prompt with tilde shortening for home dir.
- `/cd` slash command for changing the working directory.
- MCP client support: external tool servers via `[[mcp.servers]]` with stdio and HTTP.
- MCP tools namespaced as `server__tool` with per-server permission configuration.
- `delete` subcommand to delete specific or all sessions.
- `list` subcommand to display past sessions with timestamps and preview text.
- `export` subcommand to export session history as Markdown.
- Raw markdown render mode with ANSI highlighting via `--render-mode raw` or config.
- Table column alignment in raw render mode with Unicode-width-aware CJK padding.
- `Database` error variant for SQLite errors (previously misclassified as `Config`).
- Unit tests for CLI parsing, slash commands, PKCE/OAuth, and rendering (31 tests).
- Unit tests for malformed API response handling (missing `id`, `name`, `message`).

### Changed

- Default render mode changed from `rich` to `raw`.
- Split `display.show_session_id` into `on_create` and `on_exit` variants.
- Replaced all `.expect()` calls in production code with error propagation via `?`.
- Replaced all `let _ =` on fallible operations with proper error logging.
- Removed organizational section divider comments to comply with coding guidelines.
- Deduplicated stop reason parsing into `parse_openai/claude_stop_reason` helpers.
- Deduplicated OpenAI streaming tool call finalization into a helper function.
- Config file writing now uses proper TOML serialization instead of string formatting.
- Replaced `unwrap_or_default()` in message serialization with error propagation.
- Added `tracing::warn!` for fallback JSON parsing of malformed tool arguments.
- Introduced `AgentOptions` struct to reduce `Agent::new` parameter count.
- Resolved all clippy warnings (collapsible if, wildcard patterns, C string literals).
- Renamed single-letter closure variables in provider parsing to descriptive names.
- Replaced `unwrap_or_default()` on tool call fields with proper error propagation.
- Replaced direct JSON indexing with `.get()` and error handling in provider parsing.
- Split `provider.rs` into module: shared types, `claude.rs`, and `openai.rs`.
- Split `tools.rs` into module: registry, `file.rs`, `search.rs`, `shell.rs`, `web.rs`.

### Fixed

- Streaming mode now shows full API error body instead of a generic error message.
- Multi-line paste now inserts all lines into the buffer instead of executing immediately.
- TOML injection in `write_config_file` when API keys contain special characters.
- Pre-existing test compilation errors in `ClaudeProvider::new` and `create_provider`.

## [0.3.1] - 2026-03-12

### Fixed

- OAuth token refresh failing with 400 due to missing `client_id` and form-encoded body.

## [0.3.0] - 2026-03-11

### Added

- First-launch setup wizard for provider, authentication, and model configuration.
- `agsh setup` subcommand to re-run the configuration wizard.
- OAuth Authorization Code flow with PKCE for Claude provider authentication.
- OAuth token auth for Claude via `CLAUDE_OAUTH_TOKEN` env var or config.
- Database-backed OAuth token storage with automatic refresh.
- Configurable OAuth token refresh endpoint via `provider.oauth_token_url`.

### Changed

- Renamed `anthropic` provider to `claude` (breaking: `--provider anthropic` removed).
- Renamed `ANTHROPIC_API_KEY` env var to `CLAUDE_API_KEY`.
- API key no longer required at startup when an OAuth token is stored in the database.

## [0.2.0] - 2026-03-06

### Added

- Slash commands: `/help`, `/exit`, `/clear`, `/session`, `/permission`, `/compact`.
- Skills are user-defined Markdown knowledge files the agent can discover and read.
- Configurable context window limiting via `[session] context_messages`.
- Automatic session cleanup via `[session] retention_days` and `max_storage_bytes`.

### Changed

- One-shot prompt is now a positional argument (`agsh "prompt"`) instead of a flag.
- Switched `reqwest` from `native-tls` to `rustls-tls` for pure-Rust TLS.
- Added release profile optimizations (`lto`, `codegen-units = 1`, `strip`).
- Added Rust dependency caching in CI workflow.
- Removed OpenSSL system dependency installation from CI.

## [0.1.2] - 2026-03-05

### Added

- Windows binary icon embedding via `winres`.

### Fixed

- Panic on multi-byte UTF-8 chars in web search HTML parsers (Google, Bing).

## [0.1.1] - 2026-03-05

### Added

- Read-only filesystem sandboxing for shell commands using Landlock and sandbox-exec.
- Configurable sandbox toggle via `[shell] sandbox` config option.
- Conditional system prompt for read mode based on sandbox availability.

### Fixed

- Panic on multi-byte UTF-8 chars in web search results and URL fetching truncation.

## [0.1.0] - 2026-03-05

### Added

- Interactive REPL shell with natural language input.
- One-shot mode via positional `[PROMPT]` argument.
- OpenAI and Claude LLM provider support with streaming.
- Three-level permission system (none/read/write) with Shift+Tab cycling.
- Built-in tools: `read_file`, `write_file`, `edit_file`, `find_files`, and more.
- Session persistence with SQLite (create, resume with `-s`, continue with `-c`).
- Session locking to prevent concurrent access.
- `!` prefix shell escape for direct command execution.
- `exit`/`quit` keywords and Ctrl+D to leave the shell.
- TOML configuration file with `[provider]`, `[display]`, and `[web]` sections.
- Configurable user agent for web requests via `[web] user_agent`.
- Cross-platform support for Windows (PowerShell) and macOS.
- Platform-specific OS detection in system prompt (Linux, macOS, Windows).
- Leading newline stripping from LLM streaming output.
- mdBook documentation site.
- GitHub Actions workflows for documentation deployment and release builds.
- MIT license.

[Unreleased]: https://github.com/k4yt3x/meka/compare/0.37.0...HEAD
[0.37.0]: https://github.com/k4yt3x/meka/compare/0.36.0...0.37.0
[0.36.0]: https://github.com/k4yt3x/meka/compare/0.35.0...0.36.0
[0.35.0]: https://github.com/k4yt3x/meka/compare/0.34.0...0.35.0
[0.34.0]: https://github.com/k4yt3x/meka/compare/0.33.1...0.34.0
[0.33.1]: https://github.com/k4yt3x/meka/compare/0.33.0...0.33.1
[0.33.0]: https://github.com/k4yt3x/meka/compare/0.32.1...0.33.0
[0.32.1]: https://github.com/k4yt3x/meka/compare/0.32.0...0.32.1
[0.32.0]: https://github.com/k4yt3x/meka/compare/0.31.0...0.32.0
[0.31.0]: https://github.com/k4yt3x/meka/compare/0.30.0...0.31.0
[0.30.0]: https://github.com/k4yt3x/meka/compare/0.29.4...0.30.0
[0.29.4]: https://github.com/k4yt3x/meka/compare/0.29.3...0.29.4
[0.29.3]: https://github.com/k4yt3x/meka/compare/0.29.2...0.29.3
[0.29.2]: https://github.com/k4yt3x/meka/compare/0.29.1...0.29.2
[0.29.1]: https://github.com/k4yt3x/meka/compare/0.29.0...0.29.1
[0.29.0]: https://github.com/k4yt3x/meka/compare/0.28.0...0.29.0
[0.28.0]: https://github.com/k4yt3x/meka/compare/0.27.3...0.28.0
[0.27.3]: https://github.com/k4yt3x/meka/compare/0.27.2...0.27.3
[0.27.2]: https://github.com/k4yt3x/meka/compare/0.27.1...0.27.2
[0.27.1]: https://github.com/k4yt3x/meka/compare/0.27.0...0.27.1
[0.27.0]: https://github.com/k4yt3x/meka/compare/0.26.2...0.27.0
[0.26.2]: https://github.com/k4yt3x/meka/compare/0.26.1...0.26.2
[0.26.1]: https://github.com/k4yt3x/meka/compare/0.26.0...0.26.1
[0.26.0]: https://github.com/k4yt3x/meka/compare/0.25.2...0.26.0
[0.25.2]: https://github.com/k4yt3x/meka/compare/0.25.1...0.25.2
[0.25.1]: https://github.com/k4yt3x/meka/compare/0.25.0...0.25.1
[0.25.0]: https://github.com/k4yt3x/meka/compare/0.24.1...0.25.0
[0.24.1]: https://github.com/k4yt3x/meka/compare/0.24.0...0.24.1
[0.24.0]: https://github.com/k4yt3x/meka/compare/0.23.1...0.24.0
[0.23.1]: https://github.com/k4yt3x/meka/compare/0.23.0...0.23.1
[0.23.0]: https://github.com/k4yt3x/meka/compare/0.22.1...0.23.0
[0.22.1]: https://github.com/k4yt3x/meka/compare/0.22.0...0.22.1
[0.22.0]: https://github.com/k4yt3x/meka/compare/0.21.1...0.22.0
[0.21.1]: https://github.com/k4yt3x/meka/compare/0.21.0...0.21.1
[0.21.0]: https://github.com/k4yt3x/meka/compare/0.20.0...0.21.0
[0.20.0]: https://github.com/k4yt3x/meka/compare/0.19.0...0.20.0
[0.19.0]: https://github.com/k4yt3x/meka/compare/0.18.4...0.19.0
[0.18.4]: https://github.com/k4yt3x/meka/compare/0.18.3...0.18.4
[0.18.3]: https://github.com/k4yt3x/meka/compare/0.18.2...0.18.3
[0.18.2]: https://github.com/k4yt3x/meka/compare/0.18.1...0.18.2
[0.18.1]: https://github.com/k4yt3x/meka/compare/0.18.0...0.18.1
[0.18.0]: https://github.com/k4yt3x/meka/compare/0.17.2...0.18.0
[0.17.2]: https://github.com/k4yt3x/meka/compare/0.17.1...0.17.2
[0.17.1]: https://github.com/k4yt3x/meka/compare/0.17.0...0.17.1
[0.17.0]: https://github.com/k4yt3x/meka/compare/0.16.1...0.17.0
[0.16.1]: https://github.com/k4yt3x/meka/compare/0.16.0...0.16.1
[0.16.0]: https://github.com/k4yt3x/meka/compare/0.15.1...0.16.0
[0.15.1]: https://github.com/k4yt3x/meka/compare/0.15.0...0.15.1
[0.15.0]: https://github.com/k4yt3x/meka/compare/0.14.0...0.15.0
[0.14.0]: https://github.com/k4yt3x/meka/compare/0.13.1...0.14.0
[0.13.1]: https://github.com/k4yt3x/meka/compare/0.13.0...0.13.1
[0.13.0]: https://github.com/k4yt3x/meka/compare/0.12.0...0.13.0
[0.12.0]: https://github.com/k4yt3x/meka/compare/0.11.0...0.12.0
[0.11.0]: https://github.com/k4yt3x/meka/compare/0.10.3...0.11.0
[0.10.3]: https://github.com/k4yt3x/meka/compare/0.10.2...0.10.3
[0.10.2]: https://github.com/k4yt3x/meka/compare/0.10.1...0.10.2
[0.10.1]: https://github.com/k4yt3x/meka/compare/0.10.0...0.10.1
[0.10.0]: https://github.com/k4yt3x/meka/compare/0.9.4...0.10.0
[0.9.4]: https://github.com/k4yt3x/meka/compare/0.9.3...0.9.4
[0.9.3]: https://github.com/k4yt3x/meka/compare/0.9.2...0.9.3
[0.9.2]: https://github.com/k4yt3x/meka/compare/0.9.1...0.9.2
[0.9.1]: https://github.com/k4yt3x/meka/compare/0.9.0...0.9.1
[0.9.0]: https://github.com/k4yt3x/meka/compare/0.8.1...0.9.0
[0.8.1]: https://github.com/k4yt3x/meka/compare/0.8.0...0.8.1
[0.8.0]: https://github.com/k4yt3x/meka/compare/0.7.1...0.8.0
[0.7.1]: https://github.com/k4yt3x/meka/compare/0.7.0...0.7.1
[0.7.0]: https://github.com/k4yt3x/meka/compare/0.6.1...0.7.0
[0.6.1]: https://github.com/k4yt3x/meka/compare/0.6.0...0.6.1
[0.6.0]: https://github.com/k4yt3x/meka/compare/0.5.3...0.6.0
[0.5.3]: https://github.com/k4yt3x/meka/compare/0.5.2...0.5.3
[0.5.2]: https://github.com/k4yt3x/meka/compare/0.5.1...0.5.2
[0.5.1]: https://github.com/k4yt3x/meka/compare/0.5.0...0.5.1
[0.5.0]: https://github.com/k4yt3x/meka/compare/0.4.1...0.5.0
[0.4.1]: https://github.com/k4yt3x/meka/compare/0.4.0...0.4.1
[0.4.0]: https://github.com/k4yt3x/meka/compare/0.3.1...0.4.0
[0.3.1]: https://github.com/k4yt3x/meka/compare/0.3.0...0.3.1
[0.3.0]: https://github.com/k4yt3x/meka/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/k4yt3x/meka/compare/0.1.2...0.2.0
[0.1.2]: https://github.com/k4yt3x/meka/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/k4yt3x/meka/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/k4yt3x/meka/releases/tag/0.1.0
