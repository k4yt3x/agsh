# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Rust Coding Guidelines

- Prioritize code correctness and clarity. Speed and efficiency are secondary priorities unless otherwise specified.
- Do not write organizational or comments that summarize the code. Comments should only be written in order to explain "why" the code is written in some way in the case there is a reason that is tricky / non-obvious.
- Never hand-wrap comments. Write each comment (and doc comment) as one line per paragraph and let `cargo +nightly fmt` wrap it (`.rustfmt.toml` has `wrap_comments = true`). If fmt's wrap lands awkwardly, reword the prose rather than inserting a manual break.
- Prefer implementing functionality in existing files unless it is a new logical component. Avoid creating many small files.
- Avoid using functions that panic like `unwrap()`, instead use mechanisms like `?` to propagate errors.
- Be careful with operations like indexing which may panic if the indexes are out of bounds.
- Never silently discard errors with `let _ =` on fallible operations. Always handle errors appropriately:
    - Propagate errors with `?` when the calling function should handle them
    - Use `.log_err()` or similar when you need to ignore errors but want visibility
    - Use explicit error handling with `match` or `if let Err(...)` when you need custom logic
    - Example: avoid `let _ = client.request(...).await?;` - use `client.request(...).await?;` instead
- When implementing async operations that may fail, ensure errors propagate to the UI layer so users get meaningful feedback.
- Never create files with `mod.rs` paths - prefer `src/some_module.rs` instead of `src/some_module/mod.rs`.
- When creating new crates, prefer specifying the library root path in `Cargo.toml` using `[lib] path = "...rs"` instead of the default `lib.rs`, to maintain consistent and descriptive naming (e.g., `gpui.rs` or `main.rs`).
- Avoid creative additions unless explicitly requested
- Use full words for variable names (no abbreviations like "q" for "queue")
- Use variable shadowing to scope clones in async contexts for clarity, minimizing the lifetime of borrowed references.
  Example:
    ```rust
    executor.spawn({
        let task_ran = task_ran.clone();
        async move {
            *task_ran.borrow_mut() = true;
        }
    });
    ```

## Logging and output

`meka` maintains a strict split between *CLI output* and *tracing logs*. The test is simple: **if the user doesn't have to see this to use the command, it belongs in `tracing`**. Default log level is `warn`, so `info!` / `debug!` are silent unless the user passes `-v`, `-vv`, or `RUST_LOG`. Aim for "quiet on success" (the Unix convention).

**Use `println!` / `eprintln!` only when the output is unavoidable:**

- **Requested data**: what the user literally ran the command to get: the `meka mcp list` table, `meka mcp get` details, `meka session list` session rows, `meka session export` markdown on stdout, `print_help`.
- **Actionable content the user must copy/type/visit**: OAuth authorisation URLs, callback paste prompts, elicitation form fields, setup-wizard prompts.
- **REPL command output**: `/permission`, `/session`, `/cd` errors, `!cmd` status, tool-use indicators, streaming assistant markdown, thinking blocks, `Unknown command` feedback.
- **Hard errors** propagated back to the user with context (`render::render_error`, clap-side validation errors).
- Use `stdout` (`println!`) for parseable command output a script might consume; `stderr` (`eprintln!`) for prompts, live UI, and contract errors.

### `stdout` vs `stderr`

When `println!` / `eprintln!` *is* the right call (the output is unavoidable per the list above), the choice of stream is not a style decision; it's a contract:

- **`stdout` (`println!`, `print!`)**: only the data the user invoked the command to obtain. Examples: the agent's streamed assistant response, an `meka session list` table, an `meka session export -` markdown body, an `meka skill show` body, `meka mcp list` / `mcp get` / `mcp tools` rows.
- **`stderr` (`eprintln!`, `eprint!`)**: everything else: tool-call indicators, thinking blocks, todo lists, spacing newlines, status confirmations, hints, errors, interrupt notices, setup-wizard prompts, OAuth URLs, REPL UI feedback (`/permission`, `/cd`, `Unknown command`, approval prompts, `!cmd` exit-code messages).

**Litmus test:** `meka ... 2>/dev/null | next-tool` should leave only the requested data on stdout. If a user can't usefully pipe the output, your `println!()` is probably an `eprintln!()`.

The streaming markdown renderer (`render::StreamingRenderer`) writes to stdout because the assistant response *is* the requested output for an agent turn. Every other helper in `render.rs` (`render_session_id`, `render_hint`, `render_error`, `render_thinking_block`, `render_todo_list`, `render_tool_indicator`) and every spacing-blank-line emitted around them goes to stderr.

**Use `tracing` for everything else:**

- `error!`: unrecoverable failure about to propagate up as an `MekaError`. Rare; the `?` operator usually already carries the info.
- `warn!`: recoverable fallback the user should know about by default: "failed to revoke token, continuing", "authorisation failed, rolling back", "probe: couldn't reach X". Also the right level for rollback and cleanup messages.
- `info!`: lifecycle signposts users *can* see with `-v`: "added X to config.toml", "authorized X", "connected to MCP server Y", "resuming session UUID", "auto-compacting", "exported session to path", `probe:` hints. This is the "quiet success" level (no output at default verbosity).
- `debug!`: diagnostics for module-level troubleshooting: "browser launch failed" (expected on headless), "reconnect attempt 2", raw callback parse details, `resource_metadata` URLs.

**Specifically, these informational CLI signposts are logs, not prints:**

- `ok:` confirmations (`added`, `removed`, `connected`, `authorized`, `cleared credentials`, `configuration saved`). Exit code carries success; don't reprint the command the user just ran.
- Probe results, running-OAuth banners, auto-compact hints, "resuming session: UUID", "exported to path".
- Rollback explanations ("interrupted, rolling back X", "authorisation failed, rolling back"): these are `warn!`, not print, because they are recoverable diagnostic information.

**Never mix the two:**

- Don't `eprintln!` "failed to open browser" on a fallback path when the URL is already printed. Users can copy it; the warning is noise. Use `tracing::debug!`.
- Don't `tracing::info!` a command's primary output; users would need `-v` to see what they asked for.
- Don't `tracing::warn!` something that isn't a warning. Lifecycle signposts are `info!`.

**Drop redundant preambles.** If you're about to print a progress line immediately followed by the actionable info, cut the preamble. "Opening browser..." then the URL is noise; just print the URL.

**Opt-in visibility.** When a config flag like `show_session_id_on_create` explicitly requests visible output, honour it via `println!` / `eprintln!`; don't silently demote it to `info!` and force `-v`.

## Configuration surfaces

meka has several configuration surfaces. Keep coverage principled rather than adding ad-hoc overrides:

- **`config.toml` is the complete source of truth** for non-secret settings — every persistent setting lives here.
- **Provider configuration is config-only, never env.** Providers are named profiles in `[providers.<name>]` (backend `type` + model/base_url/etc.); the active one is chosen by `--provider <name>` > `default_provider` > the sole profile. There is deliberately **no env tier** for provider selection, model, base_url, or credentials: an ambient `OPENAI_API_KEY` / `MEKA_PROVIDER` must never silently rebind which account a named profile bills. CLI `--provider`/`--model`/`--base-url` are the only per-run overrides. Profiles are managed via the `meka provider` suite (`add`/`list`/`use`/`login`/`remove`), mirroring `meka mcp`.
- **Secrets live in the database, never in config or env.** API keys and OAuth bundles are stored in the `provider_credentials` table keyed by profile name, acquired via `meka provider add` / `login` and deleted via `remove`. Per-profile keying lets two accounts of the same backend coexist.
- **Environment variables are operational-only**: config/data dirs (`MEKA_CONFIG_DIR`, `MEKA_DATA_DIR`), permission, instructions, sandbox backend, render mode, MCP timeout, and `RUST_LOG`. For these the precedence is **CLI flags > env > `config.toml`** (the idiom is `cli.x.or_else(env).or(file)` in `ResolvedConfig::from_cli`).
- **Session and display tuning stays config-only** (e.g. `context_messages`, `retention_days`, `auto_compact`, `newline_*`, `show_*`) — don't add env vars or flags for set-once preferences.

## Schema and versions

`src/session/migrations.rs` is a ledger of schema steps, applied on open inside the schema lock, in one transaction, behind an automatic backup. Three hard rules keep it worth having.

**1. Only the migration module may know that a previous version of meka wrote the store.** Banned everywhere else: fallback readers for a superseded row shape, deprecation notices about stored data, "this column used to be called X" branches, version sniffing, and anything whose condition is "was this row written by an older meka". If a reader seems to need one, a migration is missing. Write the migration.

The rule is about **the store**, which has a ledger. `config.toml` does not and cannot easily get one, because it is hand-edited and may be older or newer than the binary at any moment; config-format evolution is handled where it is parsed, with serde defaults and value aliases (`src/render.rs`'s `#[serde(alias = "rich")]` is a legitimate example, keeping a renamed *config value* working). Model-facing tolerance is also out of scope: `src/tools/todo.rs`'s aliases accept what a model might emit, which is nothing to do with meka's own history.

Rule 1 is **not** test-enforced, and that is worth knowing when you are tempted by a small exception: nothing will catch you. It decays gradually and invisibly, which is exactly why it is written down here rather than left to judgement.

The point is not tidiness. Backwards compatibility spread across readers costs one shim per reader per superseded shape, forever, and each shim is an invisible independent decision. A migration converts once and leaves exactly one shape in the world, which is what earns every other reader the right to assume the current schema unconditionally. That assumption is the whole return on this arrangement; the moment a second place starts tolerating an old shape, it is gone.

**2. A migration is frozen when it ships, and so are its dependencies.** Never edit an entry that has been released: users who ran it will not run it again, so an edit changes what *new* stores get and nothing else, and no test downstream can see the divergence. Append a new step instead. A migration also may not call meka's own code, only `rusqlite`, `serde_json` and the like. `gates_become_kind_and_spec` builds its JSON by hand rather than calling `Gate::spec`, because borrowing that function would have the migration quietly start doing something different the day those types are refactored, long after the stores it ran against stopped being able to notice.

**3. A migration must be safe to run twice.** `user_version` lives in the SQLite file header and the standard round trip drops it: `sqlite3 old.db .dump | sqlite3 new.db` yields the right schema at version 0 (plain `VACUUM` preserves it, `.dump` does not). Such a store is classified by shape, which can only answer "fresh" or "at the baseline", so every later step replays over data that already has it. Guard each `ALTER TABLE` on the current column set, prefer `CREATE … IF NOT EXISTS`, and have data conversions test for the shape they convert *from*. `gates_become_kind_and_spec` does all three; a bare `Step::Sql("ALTER TABLE … ADD COLUMN x")` would fail with `duplicate column name` and refuse such a store on every start, permanently.

Both halves of rule 2 *are* enforced, by `the_released_prefix_is_frozen` and `no_migration_calls_meka_s_own_code`. Rules 1 and 3 are not; they decay silently, which is why they are written here. Neither is a formality: the first is a prefix check so that appending stays legal and only edits fail, and if it ever fires, append rather than pasting current values over the expected ones. The second matches `super::` as well as `crate::`, because this module is a child of `session` and `super::super::schedule::Gate::spec` reaches the very function the rule forbids.

**Integrity guards are not compatibility, and stay where the data is read.** The distinction is whether the check names a release. "`gate_kind` and `gate_spec` must both be set or both be null" and "an unparseable `gate_permission` resolves to the level that authorises nothing" are equally true of a store meka created five minutes ago; they defend against hand-editing, corruption and bugs, and deleting them turns fail-closed paths into fail-open ones. Reconciliation is the same: `memory::store::reconcile_index` asks whether this database's FTS triggers are the ones this build requires and makes them so, which converges on the current shape rather than interpreting an old one.

Practical notes. The version is `PRAGMA user_version`, which is transactional and survives `VACUUM INTO`, so schema and version move together and a restored backup identifies itself; SQLite's own `PRAGMA schema_version` is an unrelated internal counter and must never be written. Numbers are sequential indices into the list, not encoded releases. Migration is forward-only: downgrading means restoring the backup. `classify_by_shape` is the single exception to rule 1 and the reason it costs nothing, because it runs once per store, stamps its answer, and is never consulted again.

## Whose fact is it

If another system is the authority for a fact, ask it or let the user state it - never encode it.

meka used to pick a reasoning-effort tier, a thinking encoding and a context window from the model's *name*, via version-parsing heuristics and family lists. Every one of those was a guess about someone else's system, and each went stale on its own schedule. Sorting a setting by who owns the fact says what to do with it:

- **The provider owns it** - a request parameter it will default sensibly if meka says nothing (`output_config.effort`, `reasoning.effort`). Omit it unless the user asked for a value. Omitting is not a degraded setting; it is how you request the provider's default. This also means meka can't be wrong about an endpoint it has never seen, which matters because `anthropic-messages`, `openai-chat-completions` and `openai-responses` reach any compatible server.
- **The user owns it** - something neither meka nor the provider can determine, or where a wrong guess is invisible (`context_window`, the `thinking` encoding). Give it a config key with one documented default; don't infer, probe, or cache. State the default in the docs and, where a setup flow exists, on screen.
- **meka owns it** - its own tool names, its own schema, its own defaults for its own behaviour. Encoding these is not the same thing, and needs no apology.

Two exceptions. A guess whose wrong answer is a **rejected request** may stay: `model_supports_temperature` is the clean case, an allowlist that fails toward omission. `claude-subscription` also keeps `model_supports_modern_features`, `model_is_haiku` and `model_supports_mid_conversation_system`, which gate beta headers and `context_management`; those are denylists that fail toward *sending*, and they survive only because that backend's endpoint is always Anthropic, so an unrecognised name is necessarily a real Claude. Don't copy that shape onto a backend reachable via `base_url`. And a value verified against a captured wire is a fact about the protocol, not a prediction (the `anthropic-beta` strings) - those are pinned deliberately and cite the capture.

A hardcoded fact about an external system will expire, and nothing in the build will notice: this repo shipped a "retiring 2026-08-05" line in two doc files that was still presented as upcoming a fortnight later.

## Built-in tool naming

Tool names are read by the model on every turn, and `tool_catalogue` is sorted, so a name is both a
label and a sort key. Two rules:

- **A family shares a noun prefix**: `<subsystem>_<verb>`. `memory_read`, `scratchpad_write`,
  `agent_spawn`, `schedule_cancel`. The prefix is what makes the family arrive as one block in the
  sorted catalogue instead of scattered through it, so it is functional rather than cosmetic. The
  prefix names what the tools act on, which is not always the module they live in: the `[background]`
  subsystem's tools are `task_list` and `task_cancel`, because a task is what the model manipulates.
  Where a subsystem *manages* more than one kind of object, each with its own set of operations,
  qualify before the verb and keep the object first so operations on one object stay adjacent:
  `mcp_resource_read`, `mcp_prompt_list`. A verb that merely mentions a noun does not make that noun
  a managed object. `scratchpad_save_file` and `scratchpad_load_file` both act on a scratchpad entry
  and take a path as the source or destination, so the scratchpad manages one kind of object, not
  two; `scratchpad_file_save` would invent a `file` object with no other operations.
- **A standalone tool reads as a natural verb phrase**: `<verb>_<object>`. `read_file`,
  `execute_command`, `fetch_url`, `search_web`. A subsystem with exactly one operation may use the
  bare noun: `todo`, `skill`.

Two deliberate exceptions, both worth keeping:

- **An industry-standard name beats internal consistency.** `read_file`, `write_file`, `edit_file`
  and `execute_command` are what every harness calls these, and models reach for them zero-shot.
  Renaming them to fit a pattern would trade real accuracy for tidiness. A family member that
  deliberately mirrors one keeps the echo for the same reason: `scratchpad_save_file` reads as the
  scratchpad's `write_file`, and its description says so.
- **`load_tool`** stays verb-first despite acting on meka's own registry: `tool_load` reads worse,
  and the name appears verbatim in the `[Tool discovery]` preamble the model reads every turn.

Renaming a tool is a breaking change: it appears in `[tools]` / `[subagents]` config lists, in
user-authored skills and instructions, and in the conversation history of every existing session
(where a resumed model may reach for the old name once). Prefer getting it right when the tool is
introduced. When a rename is right anyway, do it while the family is new rather than breaking users
twice, add a `**Breaking:**` changelog line, and update `BUILTIN_TOOL_NAMES` (kept sorted),
`MCP_META_TOOL_NAMES` if applicable, and `tool_display_name` in `src/render.rs`.

Two traps when carrying a rename through the tree. A blanket find-and-replace will rewrite tool
names meka does not own, because an MCP server's tool may contain a built-in's name as a substring
(`mcp__exa__web_search_exa` is Exa's, and renaming `web_search` must leave it alone); anchor every
substitution to a name boundary and read the hits. And a rename that reverses word order defeats
`did_you_mean_hint`, which is edit-distance based: `spawn_agent` is ~10 edits from `agent_spawn`,
far past the threshold, so neither a resumed model nor a stale `disabled_tools` entry gets pointed
at the new name. Both are silent, so neither shows up in the test suite.

## CLI help text

Clap `///` doc-comments must render within 80 columns when shown via `-h`. Verify by running the actual binary for every changed subcommand: source-line length doesn't account for clap's indent, value-name length, or auto-appended hints like `[possible values: ...]`. Put `Examples:` and other long-form prose after a blank `///` line so they only show in `--help`, not `-h`. When that long-form prose has multiple lines or indented blocks (e.g. an `Examples:` list), add `#[command(verbatim_doc_comment)]` to the struct/variant so clap preserves the line breaks instead of re-wrapping them into one paragraph.

Command and subcommand `///` summaries (the one-line description clap shows in the command list) don't end with a period; multi-sentence descriptions and `Examples:` prose use normal punctuation.

## Build & Formatting Commands

- Always run `cargo +nightly fmt` and `cargo sort -w` after editing code.
- Always run `cargo build` after completing all tasks.
- Always run `cargo doc --no-deps --document-private-items` after completing all tasks.

CI's `lint` job denies warnings on both clippy and rustdoc, so the bare commands above can pass locally yet fail CI. Reproduce the exact gate before declaring done:

- `cargo clippy --all-targets -- -D warnings` (note `--all-targets`: covers tests/benches, which plain `cargo clippy` skips).
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items`. Watch for `rustdoc::invalid_html_tags`: a bare `<word>` in a doc comment (e.g. `<name>`) is parsed as an unclosed HTML tag and fails the build. Wrap such tokens in backticks, but if the doc comment is also a clap `///` help string (backticks render literally in `-h`), rephrase to drop the angle brackets instead.
- `cargo +nightly fmt --check` is what CI runs; `cargo +nightly fmt` (no `--check`) fixes it.

## Changelog

- Update `CHANGELOG.md` after every meaningful change (new features, bug fixes, breaking changes, deprecations, removals)
- Follow the [Keep a Changelog 2.0.0](https://keepachangelog.com/en/2.0.0/) format
- Add entries under the `[Unreleased]` section
- Keep each changelog entry to around 100 characters
- Use only the six types (Added, Changed, Deprecated, Removed, Fixed, Security) and group entries by type
- `Fixed` means the behavior was wrong and is now correct; `Changed` means it worked as intended and now works differently
- Mark a breaking change with an inline `**Breaking:**` prefix inside its type, not in a separate section
- Lead a `Security` entry with its CVE id when one exists

## Documentation

- Update the mdBook docs under `docs/book/src/` when adding or changing user-facing features, configuration options, CLI behavior, etc.

## Prose style

- Avoid using em dashes (`—`) in writing.
