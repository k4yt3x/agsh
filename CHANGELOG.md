# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A session records the provider profile it runs on, and a resume restores it.
- A session's context window, vision flag and `max_output_tokens` come from its own profile.
- `/provider [name]` in the REPL shows or changes the profile the session runs on.
- `POST /v1/sessions` takes `provider`; `PATCH /v1/sessions/{id}` moves a session, loaded or not.
- ACP advertises `configOptions` (permission and provider) and handles `session/set_config_option`.
- `meka session list` gains a Provider column.
- Every HTTP session body reports the `provider` profile the session runs on.
- `meka session export` / `import` carry the provider profile.
- `meka provider login --api-key-stdin` rotates an API key without prompting or losing the profile.
- `meka mcp login --auth-token-stdin` / `--client-secret-stdin` set or rotate a server's secret.
- `meka mcp get` reports which kinds of credential a server has stored, without printing them.
- `meka provider set <name> <key> <value>` changes one profile setting, keeping the rest intact.
- `provider add` gains six flags for the remaining profile fields, so any profile is one command.

### Changed

- **Breaking:** `--model`, `--base-url`, `--thinking` and `--thinking-budget` are removed.
- **Breaking:** `meka session list --long` is removed along with the columns it showed.
- The thinking budget is per profile (`thinking_budget`), falling back to `[thinking].budget_tokens`.
- Profile keys have one canonical order, provider-scoped first, shared by the file, `-h` and docs.
- Every writer normalises a profile's key order; `provider set` no longer preserves key positions.
- `provider add` refuses a profile that would fail at startup, as `provider set` already did.
- `provider set` refuses a thinking setting on a backend that never sends one, as `add` drops it.
- A failing provider call is attempted three times rather than four, with 1s then 2s of backoff.
- A provider's `Retry-After` is honoured up to 60 seconds, where it was previously capped at 15.
- A 502 from `meka serve` relays the upstream's `Retry-After` when it gave one, capped at an hour.
- Both subscription backends share one OAuth refresh exchange, so their handling cannot drift.
- `claude-subscription` stops setting `Connection: keep-alive`, which HTTP/2 forbids and never sent.
- **Breaking:** a pre-migration backup supersedes the previous one; the copy 0.43 left is deleted.
- **Breaking:** `schedule_create` drops `isolated`; every job fires in the session that created it.
- **Breaking:** `POST /v1/sessions/{id}/schedule` rejects `isolated`; job views and webhooks drop it.
- **Breaking:** a scheduled job is refused on a sub-agent session, naming the one to use instead.
- **Breaking:** MCP `auth_token` and `client_secret` move out of `config.toml` into the database.
- **Breaking:** `meka mcp add` drops `--auth-token` and `--client-secret`; use the `-stdin` forms.
- **Breaking:** `meka mcp logout` clears every stored credential, not only the OAuth tokens.
- A stored bearer and an `[auth]` block are mutually exclusive; where both exist, the block wins.
- `meka mcp add` refuses a name that still holds a credential, rather than silently reusing it.
- **Breaking:** `GET /v1/info` drops `provider` and `model`; `GET /v1/providers` reports both.
- **Breaking:** `meka acp` / `meka serve` refuse `-c` and `-r`, which name one run's session.
- **Breaking:** the REPL and `--oneshot` resume at the recorded permission level, not the default.
- Existing sessions adopt `default_provider` when the store is brought forward.
- `--provider` and `--permission` on a resume repin the session.
- A session whose recorded profile is not configured is refused by name, never silently redirected.
- A profile's credential is checked when a session first needs it, not when a host starts up.
- A resume is no longer blocked by an ambiguous `default_provider` it does not use.
- `meka session import` refuses an archive with no profile when nothing can supply one.
- `meka provider remove` warns when it clears `default_provider`, and how many sessions it strands.
- `meka provider list` flags an unresolvable `default_provider` and a credential it cannot read.
- `meka provider add` writes the profile before the credential, so a failed write strands neither.
- `GET /v1/sessions/{id}/context` omits `window` for a profile it cannot resolve.
- `GET /v1/health/ready` reports a provider as configured when any profile is, not just the default.

### Fixed

- A store a 0.44 dev build renumbered past `mcp_credentials` has the table restored on next open.
- `provider add` wrote a negative integer for a setting above `i64::MAX`, leaving config unreadable.
- `provider set` deleted the comment and blank line above the key it changed.
- `provider add --thinking-budget` wrote the key onto backends that never send a thinking field.
- The interactive thinking-budget prompt offered the built-in default over `[thinking].budget_tokens`.
- `PATCH /v1/sessions/{id}` naming the recorded profile now re-syncs a diverged live agent.
- A sub-agent spawned during an ACP mid-turn repin recorded a profile it was not running on.
- The refusal for a profile with no model named the removed `--model` instead of `provider set`.
- A provider call that got no usable response is retried with backoff instead of ending the turn.
- A retry sequence starts no new attempt after five minutes, so a call that hung is not tried again.
- `meka serve` answers 502 rather than 500 for a retryable, mid-stream or context-overflow failure.
- A context overflow answers `/errors/context-overflow`, so a client stops retrying what cannot fit.
- A 400 from a usage or history probe no longer classifies as a malformed *turn*.
- A 429 or 5xx from an OAuth token endpoint is retried instead of killing the turn in progress.
- A rejected OAuth refresh names the profile to run `meka provider login` on.
- `provider add --client-id` minted the grant as the default OAuth client, so refresh always failed.
- The OpenAI API-key backends took an OAuth token they could never refresh, failing at its expiry.
- `meka serve` answers 503 `/errors/mcp-unavailable`, not 500, when a required MCP server is down.
- A `Retry-After` the provider gave is no longer discarded when reading its response body fails.
- A failed send reports the body size to one decimal; integer division rounded 2.9 MiB down to 2.
- A scheduled job can always cancel itself from the turn it wakes; an isolated fire could not.
- `meka provider add` destroyed every profile when `providers` was written as an inline table.
- `meka provider remove` on the same config reported the opposite of what it did.
- Spawning a sub-agent of a session that no longer exists returned an id with no row behind it.
- A rotated provider credential reaches a running `meka serve`, instead of being ignored until exit.
- `meka serve` and `meka acp` now defer MCP tool schemas; both shipped every one on every request.
- An MCP reconnect re-lists tools and re-reads instructions, instead of keeping the old set.
- A broken scheduled gate stops being reported once another host has evaluated it successfully.
- Compaction clears the tool-schema advisories it may have summarised away.
- The REPL's `/skill` completion follows skills the agent adds or deletes mid-session.
- `--api-key-stdin` no longer lets the model or base-URL prompt consume the piped key.
- `--api-key-stdin` is refused for the subscription backends instead of silently opening a browser.

## [0.43.0] - 2026-08-26

### Added

- A scheduled job's gate can call a read-only tool, MCP or built-in, instead of a shell command.
- Gate conditions `matches` (regex) and `at` (a JSON pointer, tested empty / not-empty / changed).
- A job that cannot fire is marked and explained on every listing, `GET /v1/schedule` included.
- The agent is told when a job stops being able to fire, or becomes able to again.
- `[schedule] claim_lease` sets how long a host's claim on a due occurrence is good for.
- meka migrates its own store on open, in one transaction; no upgrade from 0.42 on needs a script.
- An existing store is copied beside itself before a migration touches it, and the copy is kept.

### Changed

- **Breaking:** a gate is now `check` plus `when`; `command` and `fire` are gone (see Upgrading).
- **Breaking:** `GET /v1/schedule` renders a gate as `check`/`kind`/`when`, not `command`/`fire`.
- **Breaking:** a due job is leased rather than consumed, which adds three `scheduled_jobs` columns.
- A store from an older release than 0.42, or from a newer meka, is refused by name and left alone.
- A tool gate is authorised at the tool's own level, so gating no longer demands `unrestricted`.
- A gate's authority is re-resolved at fire time, tool level included, not trusted from creation.
- A gate refused for an unknown or non-read-only tool is a 422, not a 403; no token or level helps.
- `schedule_list` shows a tool gate's arguments and its kind; other listings show the kind only.
- A session at `none` accepts no new scheduled job over HTTP, since none of them could ever fire.
- A job whose delivery fails three times is parked and reported, instead of retried forever.
- A new meka icon replaces the old one for agsh.

### Fixed

- A session at `none` no longer fires scheduled jobs; the woken turn could not act or cancel.
- A one-shot job held back for permission is kept, not deleted; its gate was never evaluated.
- Cancelling a job a sweep removed first reports a miss; both doors used to report success.
- A host that crashes mid-delivery no longer loses the occurrence, or for a one-shot the whole job.
- A cancellation issued while a job is being delivered is no longer undone when the host hands back.
- `meka schedule list` no longer lets a job's prompt carry a terminal escape into the table.

## [0.42.2] - 2026-08-25

### Fixed

- Docs and HTTP auto-deny notices still named `write`, the permission mode retired in 0.42.0.

## [0.42.1] - 2026-08-24

### Changed

- Tool descriptions, per-turn context and CLI output are shorter; `-h` fits 80 columns again.

### Removed

- The dedicated message for the retired `write` permission mode; it reads as any unknown mode.
- The startup warning naming memory files left behind in the config directory.

### Fixed

- The agent no longer tells a chat or API user to press Shift+Tab; refusals name the level only.
- `/help` named a permission mode that does not exist and omitted two that do.
- Docs misstated permission modes, serve scopes, scratchpad tools, provider headers and defaults.
- Docs understated the read-mode environment allow-list, which passes TLS-trust variables.

## [0.42.0] - 2026-08-24

### Added

- `workspace` permission: writes confined to the workspace roots, reads still unrestricted.
- `--writable-root <PATH>` adds a directory to the workspace, repeatable; no config key.
- Windows confines the shell at `workspace` with a restricted token and ACEs released on exit.
- A workspace root at or above a masked directory (`/tmp`, `/run`) is refused with a warning.
- Memory search is ranked and tiered: several phrasings at once, word endings, typos, CJK.
- Memories take tags, a stamped `recorded` date, and render priority-0 bodies in full each turn.
- `meka memory export`, `edit`, `verify` and `add --tag` manage the store by hand.
- Memory files left in the config directory are reported at startup, naming the importer.
- `skill_search` greps every installed skill; `skill_write` / `skill_delete` let the agent author.
- `[skills].extra_paths` scans further directories for skills, read-only and never created.
- Skills read the spec's `license`, `compatibility` and `allowed-tools`, and keep every other key.
- Compaction runs a checkpoint turn first, so the agent saves what must outlast the summary.
- `/compact <instructions>`, `context_check` and `context_compact` steer and inspect compaction.
- `openai-responses`, a backend for the OpenAI Responses API, authenticated with an API key.
- `[providers.<name>].thinking` picks the wire encoding: `adaptive`, `budgeted`, or `off`.
- HTTP: compact, rewind, export and import sessions, and report context occupancy.
- HTTP: manage scheduled jobs, background tasks, skills and memory, behind four new scopes.
- HTTP: rejoin a turn's SSE stream with `Last-Event-ID`, and see compaction as it happens.
- `[[serve.webhooks]]` posts signed, content-free notifications for turns, tasks and jobs.
- `trust_read_only_hint` refuses an MCP server's `readOnlyHint`, keeping the tool out of read mode.
- `meka mcp add --auth-token-stdin` / `--client-secret-stdin` keep secrets out of `ps` and history.
- `[display].tool_params` and `[display].max_width` control tool-call and terminal rendering.
- `[schedule].max_consecutive_fires` interleaves one session's due backlog with other sessions'.

### Changed

- **Breaking:** permission `write` is retired; use `workspace` (confined) or `unrestricted`.
- **Breaking:** `workspace` is enabled by default, so Shift+Tab reaches it before `unrestricted`.
- **Breaking:** ACP `additionalDirectories` and `--writable-root` are writable at `workspace`.
- **Breaking:** `ask` runs the shell unsandboxed and unscrubbed: writes anywhere, full environment.
- **Breaking:** a gate needs `unrestricted`; `workspace` used to authorise one and no longer does.
- **Breaking:** `[shell].sandbox = false` now refuses the shell rather than unconfining it.
- **Breaking:** Landlock requires ABI v3; below it `truncate(2)` was unmediated. Use Bubblewrap.
- **Breaking:** `every = "..."` fires on its own grid; jobs that silently ran slow now run on time.
- **Breaking:** memories are rows in the database under `MEKA_DATA_DIR`, not files in the config.
- **Breaking:** a config-directory backup no longer captures memories; back up the data dir.
- **Breaking:** `memory_search` takes `queries` (a list) instead of `pattern`, and drops regex.
- **Breaking:** `GET /v1/memory` carries `recorded_at` and `tags`, drops `skipped`, 404s cleanly.
- **Breaking:** a written `SKILL.md` conforms to the Agent Skills spec: `name`, then `metadata`.
- **Breaking:** a skill name must follow the spec and match its directory, or it is skipped.
- **Breaking:** the `skill` tool is now `skill_read`; config entries naming `skill` go stale.
- **Breaking:** `meka skill add` swaps `--version` and `--author` for a repeatable `--metadata`.
- **Breaking:** `meka skill add --from-file` needs the `name` the spec makes mandatory.
- **Breaking:** `meka skill get` and `list` change columns; `get` prints every `metadata` key.
- **Breaking:** `claude-api` → `anthropic-messages`, `claude-oauth` → `claude-subscription`.
- **Breaking:** `openai-api` → `openai-chat-completions`, `openai-codex` → `chatgpt-subscription`.
- **Breaking:** HTTP `GET /v1/providers` reports the new backend names in each profile's `type`.
- **Breaking:** `[thinking].enabled` is retired; the per-profile `thinking` mode replaces it.
- **Breaking:** pre-4.6 Claude profiles need `thinking = "budgeted"`; the default is now adaptive.
- **Breaking:** `--thinking` takes `adaptive`, `budgeted` or `off` rather than a boolean.
- **Breaking:** unset `effort` sends no tier, so the provider's own default applies.
- **Breaking:** an unset context window is 1000000, not a guess from the model name.
- **Breaking:** `claude-subscription` matches Claude Code 2.1.241; unset `effort` now sends `high`.
- **Breaking:** a stdio MCP server gets a curated environment; declare its secrets in `env`.
- **Breaking:** deleting a session another meka process has open is refused, not silently done.
- **Breaking:** ACP's sticky options read "Always allow any `<tool>`", matching their real scope.
- **Breaking:** `/v1/docs` and `/v1/openapi.json` are off unless `[serve].docs = true`.
- **Breaking:** an empty list prints its "none found" line on stderr, so stdout stays pipeable.
- `effort` and the thinking encoding are no longer inferred from a model's name, on any backend.
- `agent_followup` never widens a recorded grant, even when the parent has moved sideways since.
- `memory_write` and `skill_write` keep a stored description when the call omits it.
- Memory has no discovery cap and no per-turn walk; the index is one indexed query over FTS5.
- The `[Skills]` index is ordered by priority and capped, then points at `skill_search`.
- An unknown or stale tool name now suggests the built-in it probably meant.
- `[display].input_style` applies on submit; text keeps the terminal's own colours while typing.
- Every built-in tool indicator is PascalCase; whole families used to render as raw snake_case.
- Every rendered line is budgeted whole, so none wraps by default and a cut keeps both ends.
- CLI output that is not requested data moved to stderr or the log, keeping stdout pipeable.
- Building meka needs Rust 1.95, declared in `Cargo.toml` so an older toolchain is refused by name.
- Upgrade `rmcp` to 3.1, reedline to 0.50, `base64` to 0.23, `infer` to 0.22, `termimad` to 0.35.

### Removed

- **Breaking:** in-binary store migration; 0.41 -> 0.42 is a script shipped with the release.
- **Breaking:** `source_url` and `meka skill update`; clone the source into `[skills].extra_paths`.
- **Breaking:** the read-time move of a skill's top-level `version` / `author` under `metadata`.
- **Breaking:** a top-level `priority` is no longer a skill's rank; the release script moves it.
- **Breaking:** reading a bare-string `tool_result` or bare `signature`; the script converts both.
- The model-metadata subsystem: the models-API probe, its cache table, and the window table.

### Fixed

- Every macOS `read` command failed: the Seatbelt profile no longer parses, so nothing ran at all.
- Under bubblewrap a `read` command whose cwd was masked silently ran in `$HOME` instead.
- Landlock allows writing `/dev/null`, so `cmd 2>/dev/null` no longer fails on that backend.
- The bubblewrap probe could strand a `bwrap` if meka died before the sandbox armed its own guard.
- A failed Windows replace could delete the target and the replacement; the content is now rescued.
- A Windows write replaced the target's ACL with the directory's; `ReplaceFileW` now keeps it.
- Windows writes past `MAX_PATH` keep working: the Win32 call re-adds the prefix meka strips.
- `~\path` is expanded on Windows; it was treated as a literal directory named `~`.
- `/cd` on Windows stored a `\\?\` path, which then reached the prompt, the model and the DB.
- Windows dropped an environment variable whose *value* was not valid UTF-8 from the sandbox.
- Two files whose names differ outside UTF-8 shared one temp file, splicing a write into the other.
- A non-UTF-8 path slipped past the hidden-file skip and read as missing to the search tools.
- `find_files` left a trailing `\` on a Windows root, so the glob's separator handling decided.
- File writes are atomic and serialised per path, so a crash or a concurrent edit cannot lose one.
- Store files and roots are created private, and a rewrite no longer re-modes a file or its target.
- `config.toml` writes take an exclusive lock and follow symlinks, so no edit is silently dropped.
- Store writes release their lock before `$EDITOR`; `$VISUAL` and editor arguments are honoured.
- `scratchpad_save_file` refuses to replace a file without `force`, and records its write.
- `read_file` discloses every cut, preserves CRLF, bounds itself at 16 MiB, and windows past it.
- Every door said a skill did not exist when its `SKILL.md` was there but would not parse.
- A listed skill can always be deleted. `con`, `two words` and `my:skill` had no way out but `rm`.
- `meka skill add --force` deleted the skill before writing, losing it if the write never happened.
- A skill rewrite alphabetised or re-nested frontmatter keys meka does not model.
- A skill's description and `compatibility` are stored verbatim, not sanitised into the only copy.
- A `SKILL.md` whose closing `---` ends the file was reported as having no frontmatter.
- A memory whose name meka would not write, including one over 64 characters, was unreachable.
- A turn that could not read the memory store told the model every memory had been deleted.
- `memory_search` was case-sensitive, stopped at 100 matches mid-walk, and mismeasured near misses.
- `memory_read` had no size bound, and said nothing for a memory that has only a description.
- Memory text reached the model unsanitised; only descriptions were filtered before.
- A memory's age was its edit time, and a future date read as "today" while sorting first.
- The per-turn memory index was read on every turn even with `[memory] enabled = false`.
- The per-turn diff listed every changed memory by name, unbounded; a bulk change is now counted.
- The `[Memory]` index told the model to call `memory_write` or `memory_search` when disabled.
- `chatgpt-subscription` discarded the encrypted reasoning it asked for; it is now replayed.
- Resuming a `chatgpt-subscription` session under Claude replayed OpenAI's blob as a signature.
- Reasoning was captured only when a summary came with it, losing it on every silent think.
- Reasoning summary sections ran together; each part now starts a new paragraph.
- `/status` shows the context window from turn zero, so a configured window is verifiable up front.
- `[session].context_messages` applies every round; a tool loop cannot carry the window past it.
- Unparseable tool arguments and truncated streams are rejected and retried, not committed as done.
- `--no-stream` shows the model's reply; the blocking path rendered nothing but tool indicators.
- `load_tool` promised a schema "on your next turn" even when every name failed to resolve.
- Shift+Tab stacked a new prompt line once a turn had scrolled the prompt to the screen bottom.
- An unreadable persisted permission was read as the process default, silently; it now warns.
- `[permissions].enabled` naming no usable mode fell back to a *wider* default set than written.
- ACP mode ids parse the way `--permission` does and echo back one of `availableModes`.
- ACP `session/close` and `session/cancel` no longer deadlock or strand an `fs/*` round trip.
- Two hosts on one database both fired every job; an occurrence is now claimed atomically.
- A deferred job's restore overwrote another host's claim, so an occurrence came due twice.
- A database hiccup during a sweep dropped an occurrence and skipped every other job due that tick.
- A one-shot job was destroyed when its gate timed out or failed to spawn, having answered nothing.
- A cancelled or retired job left its held-back state behind for the life of a `meka serve`.
- A gate is re-checked against the live permission, runs in its session's cwd, and `list` shows it.
- An `on-change` gate that exits non-zero is reported, not folded into the comparison or refused.
- `cron` parses the documented five fields, and a job whose next fire is years out is kept.
- A session is locked before its row exists, so nothing can sweep or interleave one mid-creation.
- `meka session fork` and `export` copied a conversation mid-turn, producing an unusable session.
- The retention sweep and `session delete --all` skip sessions in use, and say how many.
- Retention spares a session that owns a scheduled job, and its ancestors.
- The lock-file sweep unlinked locks live processes held, letting two attach to one session.
- A first launch racing another for a fresh database could die converting to WAL; it retries.
- `meka serve` evicted a session with a background task running, then swept its own live task.
- `agent_followup` is refused while another process is running a turn on the same sub-agent.
- A streaming turn survives its SSE consumer disconnecting for `[serve].stream_reattach_grace`.
- `meka mcp add` held the config lock across the browser login, hanging every other meka launch.
- MCP servers are closed on exit on every surface, and the close is bounded so it cannot hang exit.
- Every provider, client and MCP round trip is bounded and cancellable; a stall cannot park a turn.
- Shutdown drains in-flight turns, scheduled fires and background tasks instead of abandoning them.
- Blocking work moved off the async runtime, where it stalled other sessions under `meka serve`.
- Ctrl-C is one process-wide listener that escalates; the third press drains before it exits.
- A panic in a GC, scheduler, outcome or prune loop is logged and the next tick runs.
- OAuth callbacks decode `+` as a space, matching the form encoding the redirect actually uses.
- A base URL's trailing slash or version segment no longer reaches the wire doubled.

### Security

- A `bwrap` planted earlier on `$PATH` than the real one unconfined every sandboxed command.
- An MCP tool required at `ask` dispatched with neither a prompt nor a boundary.
- A memory name reached the model unsanitised, so a newline in one forged a `[Memory]` entry.
- A gate kept firing at a level `[permissions].enabled` no longer permits, after a restart.
- Secrets no longer reach a log, a `{:?}`, a 502 body, or the terminal echo at the API-key prompt.
- Streamed model output, MCP text and prompts drop escapes, bidi overrides and carriage returns.
- The `ask` prompt shows every argument, refuses on Ctrl+D, and ignores input typed before it drew.
- Skill writes refuse a symlinked path instead of writing outside the store.
- A token refresh cannot store a dead credential or overwrite a `provider login` mid-flight.
- `MEKA_DATA_DIR` must be absolute, and an empty or relative `MEKA_CONFIG_DIR` is ignored.
- A command-output capture is created at `0600` and swept after a day.
- `utoipa-swagger-ui` is vendored, so a build no longer downloads an unpinned, unverified zip.
- `event-listener` and `memmap2` move to patched releases, and the audit ignore list is gone.
- RUSTSEC-2026-0258: `h2` moves to 0.4.19, past an unbounded empty-DATA-frame denial of service.
- **Known limitation:** a Windows `workspace` command can read meka's memory, credentials included.

## [0.41.0] - 2026-08-13

### Changed

- **Breaking:** `web_search` is now `search_web`.
- **Breaking:** `recall` and `recall_read` are now `conversation_search` and `conversation_read`.
- **Breaking:** MCP meta-tools put the object first: `read_mcp_resource` is now `mcp_resource_read`.
- **Breaking:** a `disabled_tools` entry naming a renamed tool stops denying it; update on upgrade.

## [0.40.0] - 2026-08-12

### Added

- `agent_followup` asks a sub-agent another question; it keeps its own conversation.
- `agent_list` reports a session's sub-agents, with each one's turn count and last activity.
- `agent_delete` discards a sub-agent, its scratchpad, and any sub-agents it spawned.
- `agent_spawn` returns the sub-agent's id, so a worker can be reached again.
- `[subagents]` config: `disabled_servers` and `disabled_tools`, withheld from every sub-agent.
- `agent_spawn` takes `deny_servers` / `deny_tools`, unioned with what config already denies.
- A denied MCP server is unreachable through its resources and prompts, not just its tools.
- `agent_spawn` takes `memory: "read"` to grant a worker your memories; the default is none.
- `agent_spawn` takes `instructions: "inherit"` to hand a worker the instructions file.
- Neither grant can exceed what the spawning agent holds, so authority narrows down a chain.
- The `[Memory]` index names memory files that could not be read, with the reason for each.
- A resumed session tells the agent that state a tool held outside the conversation may be gone.

### Changed

- **Breaking:** the `spawn_agent` tool is now `agent_spawn`, matching every other tool family.
- **Breaking:** sub-agents start with no memory and no instructions unless `agent_spawn` asks.
- **Breaking:** Landlock blocks Unix sockets on kernel 7.1+; `docker` and `psql` fail in read mode.

### Fixed

- `[tools].disabled_tools` now applies to the MCP resource and prompt tools, which ignored it.
- `[tools]` no longer warns that `recall`, `schedule_*` or `task_*` match no built-in tool.
- `[tools].allowed_tools` naming an MCP meta-tool now says so instead of silently doing nothing.
- `meka acp` and `meka serve` warn about stale `[tools]` entries; only the REPL did.
- `agent_spawn` ignored a non-string `permission`, silently running the worker unrestricted.
- Sub-agents inherit auto-compaction, so a worker given a large task no longer just fails.
- A compacting sub-agent no longer disables extended thinking for a sibling running beside it.
- `memory_write` without a `body` erased the memory's body; it now keeps what is there.
- `memory_read` on a name whose file exists but will not parse reported it as never written.
- A Landlock setup failure could report the wrong `errno`, since `close(2)` may overwrite it.

### Security

- Read-mode commands under Landlock could write anywhere via `systemd-run`; now blocked on ABI v9+.
- Bubblewrap was never affected: its tmpfs masks already hid the systemd and D-Bus sockets.

## [0.39.0] - 2026-08-12

### Added

- `provider list` and `mcp list` now report stored credentials that no config entry claims.
- A live `Thinking... (150 tokens)` indicator, so redacted reasoning is no longer a silent pause.
- Background tool calls: `background: true` returns a task id and reports the result later.
- A background task always reports back, including as `interrupted` when a restart killed it.
- `[background]` config, off by default, with `enabled` and a per-session `max_tasks` ceiling.
- `task_list` / `task_cancel` tools, a `/tasks` command, and a `[Background]` context section.
- Ctrl+C cancels the turn and leaves background tasks running; a second press stops them.
- `load_tool` accepts an array of names, so several tools off one server cost one round trip.
- An unknown tool name suggests the closest match, allowing for a missing `mcp__` prefix.
- The agent is told when a call sends undeclared arguments or leaves documented parameters unset.
- `[Context budget]` tells the agent its context occupancy and when auto-compaction will fire.

### Changed

- **Breaking:** `[display].render_mode` defaults to `termimad`; set `syntect` for the old look.

### Fixed

- `mcp remove` clears the credential of a server deleted from config by hand instead of refusing.
- `provider remove` no longer reports success for a name with no profile and no credential.
- `edit_file` overwrote a file that changed after the read; it now reports the change instead.
- Under ACP, `read_file` with `regex` searched the file on disk rather than the editor's copy.
- A wrong-typed `scratchpad` argument is refused, rather than silently discarding the output.
- `/mcp <server>:<prompt>` with an unknown server hung the REPL instead of drawing a new prompt.
- `termimad` hard-wrapped redirected output to a 50-column fallback instead of not wrapping.
- `[display]` blank-line spacing now brackets slash-command output, not only agent responses.
- `/export` now reports the path it wrote to; it used to save a file and say nothing.
- `/history` on an empty conversation, and `/mcp reconnect|login|logout`, now say what happened.
- `meka tools list` used fixed column widths, so a long MCP tool name ran the columns together.
- Deferred tool summaries were cut at the first sentence, hiding parameters documented later.
- `[Tool discovery]` claimed deferred tools were "not yet callable" when calling one works.

## [0.38.0] - 2026-08-11

### Added

- Agent memory: durable Markdown notes in `~/.config/meka/memory/`, surviving compaction.
- `memory_write` / `memory_read` / `memory_search` / `memory_delete` tools, all at read permission.
- `meka memory` subcommands and `/memory [name]` to list, inspect, and curate saved memories.
- `[memory] enabled` config (default true) to drop the memory tools and index entirely.
- `[skills] enabled` config (default true) to drop the `skill` tool and skills index entirely.
- Scheduled wakeups: the agent can schedule its own future turns with `schedule_create`.
- `schedule_list` / `schedule_cancel` tools, `meka schedule list|cancel`, and `/schedule`.
- Job gates: a shell command decides whether a due job spends a model turn, so polling is cheap.
- `[schedule]` config for the poll interval, missed-job grace, gate timeout, and per-session cap.
- Scheduled jobs fire under `meka serve`, the REPL, and ACP; `serve` is the durable host.
- Instructions live in `instructions.md`, or split across `instructions/*.md`, in the config dir.
- `MEKA_INSTRUCTIONS_FILE` reads instructions from a path, for mounted ConfigMaps and secrets.
- `meka instructions show` / `path` report the resolved text and where it came from.
- `/rewind [N]` and `meka session rewind` drop recent turns, so a stuck session is recoverable.
- `[[mcp.servers]].required` and `meka mcp add --required` gate a turn on one server, not all.
- `meka session delete --older-than-days <DAYS>` prunes old sessions on demand.

### Changed

- **Breaking:** an unparseable `config.toml` is a startup error, not a silent fall back to defaults.
- **Breaking:** `[prompt].instructions` is gone; move the text to `instructions.md` to start.
- **Breaking:** `[mcp] strict` defaults to false; a server gates a turn only if `required = true`.
- **Breaking:** `[session] retention_days` has no default; unset now keeps every session forever.
- `--instructions` and `MEKA_INSTRUCTIONS` are unaffected; only the config-file key moved.
- `[session] retention_days = 0` is now rejected; it would have deleted everything each startup.
- A configured retention sweep now reports deletions at `warn` instead of `info`.
- `GET /v1/health/ready` ignores failed *optional* MCP servers; only `required` ones mean 503.
- reedline moved from a personal fork to upstream `main`.

### Removed

- **Breaking:** `[session] max_storage_bytes` is gone; delete the key or meka won't start.
- **Breaking:** `${MEKA_SKILL_DIR}` / `${MEKA_SESSION_ID}` in skills; use relative paths instead.

### Fixed

- One provider-rejected message killed a session for good; meka now drops the content and retries.
- The model is told what the provider refused, so it can adapt instead of losing the turn.
- Images were labelled from the filename, `Content-Type`, or MCP `mime_type` instead of the bytes.
- A session already holding a mislabelled image is repaired on resume, with no provider call.
- MCP images above the size providers accept were forwarded anyway, only to be rejected.
- OpenAI-compatible streaming dropped a tool call or text that shared a chunk with `finish_reason`.
- REPL text streamed before an interrupt or error no longer leaks into the next turn's output.
- Calling a tool from an unconnected MCP server said "Unknown tool" instead of naming the cause.
- An unreachable MCP server logged its failure on every background retry, forever; now once.
- `meka provider add <existing>` could overwrite a profile when `config.toml` failed to parse.
- `meka provider remove` truncated `config.toml` to nothing when the file couldn't be read.
- An absurdly large `retention_days` panicked the retention sweep instead of keeping everything.
- The skills and memory indexes no longer render when the tool that opens them is disabled.

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

[Unreleased]: https://github.com/k4yt3x/meka/compare/0.43.0...HEAD
[0.43.0]: https://github.com/k4yt3x/meka/compare/0.42.2...0.43.0
[0.42.2]: https://github.com/k4yt3x/meka/compare/0.42.1...0.42.2
[0.42.1]: https://github.com/k4yt3x/meka/compare/0.42.0...0.42.1
[0.42.0]: https://github.com/k4yt3x/meka/compare/0.41.0...0.42.0
[0.41.0]: https://github.com/k4yt3x/meka/compare/0.40.0...0.41.0
[0.40.0]: https://github.com/k4yt3x/meka/compare/0.39.0...0.40.0
[0.39.0]: https://github.com/k4yt3x/meka/compare/0.38.0...0.39.0
[0.38.0]: https://github.com/k4yt3x/meka/compare/0.37.0...0.38.0
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
