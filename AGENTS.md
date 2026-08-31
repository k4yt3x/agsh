# AGENTS.md

Guidance for AI agents working in this repository. Three sections: general principles, Rust
practice, then meka-specific rules.

---

# General principles

## Code

- Correctness and clarity first. Speed and efficiency are secondary unless stated otherwise.
- Comments explain *why*, never *what*: a constraint from outside this file, or why an
  obvious-looking alternative is wrong. Not history, not user-facing documentation, not argument for
  the choice; `git log` and `docs/book/src/` hold those. If it restates the code, delete it.
- Add functionality to existing files unless it is genuinely a new component. Avoid many small files.
- No creative additions beyond what was asked.
- Full words in names, no abbreviations.

## Enumerate the doors

Most defects that survive review are one rule enforced at one entry point and not at its siblings.

Before writing a guard, list every path that reaches the thing being guarded (create, copy, fork,
import, re-root, resume, re-attach, patch, delete) and decide for each. Place the check where those
paths converge. A rule placed at a door is a rule the next door forgets.

**Prefer one definition over many checks.** When an invariant is asked in more than one place, give it
a single named predicate and call that everywhere. Consolidation beats a test per site, because tests
only confirm the sites you thought of.

**A guard sits ahead of every side effect it protects**, not merely ahead of the failure. Refusing
after a write leaves the write behind.

## Verification

Verification is graded by what it catches, not by how much of it there is.

**Per change**: build, test suite, and the project's exact CI lint gate. Then:

- **Fake-guard every test written to protect a fix**: neuter the fix, confirm the test fails, restore.
  A test that cannot fail is worse than no test, because it reads as coverage.
- Enumerate the doors, as above.

**Per release**: one structural review, docs and changelog. Don't run a cross-platform suite by hand
per change; CI already runs the matrix on every push.

**Rarely**: mutation testing. Run it when a subsystem is new, not as a gate; its yield falls as the
suite densifies while its cost grows with the codebase.

Reproduce a defect before fixing it, and re-run the reproduction after. A fix verified only by a
passing suite was verified against the thing that already missed it.

Do not grow the suite reflexively. A test earns its place by being able to fail.

## Read the code, not the comment

A doc comment is a claim about the code, not evidence for it. Where a comment and its code disagree,
the comment is usually what was updated last and least. Verify a stated invariant against the
implementation before relying on it, and correct the comment when it is wrong.

## Whose fact is it

If another system is the authority for a fact, ask it or let the user state it. Never encode it.
A hardcoded fact about an external system expires, and nothing in the build notices.

- **The provider/service owns it**: a request parameter it defaults sensibly. Omit it unless the user
  asked for a value; omitting *is* how you request their default.
- **The user owns it**: anything neither side can determine, or where a wrong guess is invisible. One
  config key, one documented default. Don't infer, probe, or cache. State that default in the docs
  and, where a setup flow exists, on screen.
- **We own it**: our own names, schema, and defaults for our own behaviour. Encoding these is fine.

A guess is tolerable when its wrong answer is a *rejected request* and it fails toward omission. A
guess that fails toward *sending* survives only where the endpoint cannot vary, so never introduce one
on a backend reachable via a user-supplied `base_url`. A value verified against a captured wire is a
fact about the protocol rather than a prediction; pin it deliberately and cite the capture.

## Compatibility

Backwards compatibility spread across readers costs one shim per reader per superseded shape, forever.
Convert once, in one place, so every other reader may assume the current shape unconditionally. That
assumption is the entire return, and it is lost the moment a second place tolerates an old shape.

## Changelog

- Update `CHANGELOG.md` for every meaningful change, under `[Unreleased]`.
- [Keep a Changelog 2.0.0](https://keepachangelog.com/en/2.0.0/). Only Added, Changed, Deprecated,
  Removed, Fixed and Security, grouped by type.
- `Fixed` = the behaviour was wrong. `Changed` = it worked as intended and now works differently.
- Around 100 characters per entry.
- Breaking changes get an inline `**Breaking:**` prefix inside their type, not a separate section.
- Lead a `Security` entry with its CVE id when one exists.

## Prose style

Avoid em dashes (`—`).

---

# Rust practice

## Safety

- Avoid panicking calls (`unwrap()`, `expect()`, unchecked indexing). Propagate with `?`. The lints
  are `warn` in `Cargo.toml` and relaxed under `cfg(test)`, where panicking on failure is the point.
- Never discard errors with `let _ =`. Propagate with `?`, log explicitly when ignoring is correct, or
  handle with `match` / `if let Err(..)`.
- Errors from fallible async work must reach the UI layer so the user gets real feedback.

## Layout and style

- No `mod.rs`. Use `src/some_module.rs`.
- New crates set `[lib] path = "..."` in `Cargo.toml` for a descriptive root name.
- Never hand-wrap comments. One line per paragraph; `cargo +nightly fmt` wraps them
  (`.rustfmt.toml` sets `wrap_comments = true`). If a wrap lands awkwardly, reword rather than
  inserting a manual break.
- Shadow a binding to scope a clone in async contexts:

  ```rust
  executor.spawn({
      let task_ran = task_ran.clone();
      async move { *task_ran.borrow_mut() = true; }
  });
  ```

## Build gate

Run after editing: `cargo +nightly fmt` and `cargo sort -w`.

CI denies warnings on clippy and rustdoc, so the bare commands can pass locally and fail CI.
Reproduce the exact gate before declaring done:

```
cargo +nightly fmt --check
cargo sort -w --check
cargo clippy --locked --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --document-private-items
cargo test --locked
cargo check --locked --all-targets   # on the MSRV in Cargo.toml's rust-version
mdbook build docs/book
```

`--all-targets` matters: plain clippy skips tests and benches. In rustdoc, watch
`rustdoc::invalid_html_tags`: a bare `<word>` parses as an unclosed tag. Backtick it, or rephrase if
the comment is also a clap help string, where backticks render literally.

`fmt --check` does not enforce `comment_width`: `wrap_comments` silently declines some comments (in
a macro body, in a method chain) and still exits 0, so a paragraph left for rustfmt to wrap can ship
at 400 columns. After `fmt`, `awk 'length>100 && /^[[:space:]]*\/\//'` the changed files; reword
what it prints, or break it by hand.

## Clap help text

`///` doc comments must render within 80 columns under `-h`. Verify by running the binary for every
changed subcommand: source length ignores clap's indent, value-name width, and auto-appended hints.
Adding flags widens the whole column, so a new flag can push existing lines over.

Put `Examples:` and other long-form prose after a blank `///` line so it appears only under `--help`.
When that prose is multi-line or indented, add `#[command(verbatim_doc_comment)]`.

Command summaries take no trailing period; multi-sentence prose is punctuated normally.

---

# meka

## Output: prints vs. tracing

**If the user doesn't have to see it to use the command, it is a log.** Default level is `warn`, so
`info!` / `debug!` are silent unless the user passes `-v`, `-vv`, or `RUST_LOG`. Aim for quiet on
success.

`println!` / `eprintln!` only for: requested data; content the user must copy, type, or visit; REPL
command output; and hard errors. Everything else is `tracing`.

The stream is a contract:

- **stdout**: only the data the command was invoked to obtain.
- **stderr**: everything else, including prompts, live UI, indicators, hints, status and errors,
  and every spacing blank line emitted around them.

Litmus test: `meka ... 2>/dev/null | next-tool` must leave only the requested data on stdout.

Levels: `error!` for an unrecoverable failure about to propagate; `warn!` for a recoverable fallback
or rollback the user should see by default; `info!` for lifecycle signposts; `debug!` for
module-level diagnostics.

Don't invert it either: a command's primary output must not be a `tracing::info!`, or the user needs
`-v` to see what they asked for. `ok:` confirmations are logs, not prints; the exit code carries
success. Drop preambles before the actionable line. Honour a config flag that asks for visible
output; don't demote it to `info!`.

## Configuration surfaces

- **`config.toml` is the complete source of truth** for non-secret settings. Every persistent
  setting lives there.
- **Provider configuration is config-only, never env.** An ambient variable must never silently rebind
  which account a named profile bills. A session runs on the profile its own row names; the row moves
  only by an explicit act (`--provider` on a resume, `/provider`, `PATCH /v1/sessions/{id}`, ACP
  `session/set_config_option`). What a *new* session records follows `--provider` > `default_provider`
  > the sole profile. Profiles are managed by the `meka provider` suite (`add`/`list`/`set`/`use`/
  `login`/`remove`), mirroring `meka mcp`: `use` is the only writer of `default_provider`, and `login`
  rotates a credential without rebuilding the profile, which `remove` + `add` would discard.
- **Secrets live in the database**: `provider_credentials` keyed by profile name and
  `mcp_credentials` keyed by `(server_name, kind)`, so two accounts, or a client secret and its
  refreshable bundle, can coexist. Every secret is read from stdin, never taken
  as an argument, because arguments are visible in `ps` and shell history.
  - A field that may *contain* a secret is not itself one; it stays in `config.toml` with `${VAR}`
    expansion.
  - Retiring a config key that held a secret gets no compatibility shim. It stops being modelled and
    `deny_unknown_fields` names the key and line; the upgrade guide carries the remedy.
- **Environment variables are operational only**: `MEKA_CONFIG_DIR`, `MEKA_DATA_DIR`, permission,
  instructions, sandbox backend, render mode, MCP timeout, `RUST_LOG`. Precedence is CLI > env > file,
  written as `cli.x.or_else(env).or(file)` in `ResolvedConfig::from_cli`.
- **Session and display tuning is config-only.** No env vars or flags for set-once preferences.

## A profile is indivisible

A provider profile is a named bundle: backend, endpoint, credential, model, and every model-tied knob.
A session selects one by name and records that name. **Nothing overrides a field inside one**, or the
run gets a combination nobody configured and no field states the mismatch.

- **`--provider <name>` selects**, and is the only provider flag on a run.
- **`provider add` / `set` write profile fields.** `set` edits one key in place via `toml_edit`,
  preserving comments and order. It has no session scope.
- **A field belongs on the profile when it is user-owned and model-tracking**, per "Whose fact is
  it". Such a field gets a `provider add` flag and a `provider set` key, never a global CLI flag, env
  var, or session column. `type` and `device_id` are excluded: the first because the stored
  credential was acquired for the current backend, the second because meka resolves it itself.

## Schema and migrations

`src/session/migrations.rs` is an append-only ledger applied on open inside the schema lock, in one
transaction, behind an automatic backup. Four rules:

1. **Only the migration module may know an older meka wrote the store.** No fallback readers, version
   sniffing, or "this column used to be called X" branches anywhere else. If a reader seems to need
   one, a migration is missing.
2. **A migration is frozen once any store has run it**, including a development store. `user_version`
   is a positional index, so removing or reordering an entry makes some store skip a step and then
   stamp itself current. Append; never edit a released entry.
3. **A migration must be safe to run twice.** A `.dump`/restore round trip drops `user_version`, so
   steps replay over data that already has them. Guard `ALTER TABLE` on the current column set, prefer
   `IF NOT EXISTS`, and have conversions test for the shape they convert *from*.
4. **A migration may receive data it cannot work out, but may not call meka's own code** to get it.
   Only `rusqlite`, `serde_json` and the like. A function can change meaning years later; a `String`
   cannot. `Step::Contextual` takes plain data from the caller, and its `Context` is append-only for
   the same reason the ledger is.

Rules 2 and 4 are enforced by `the_ledger_is_append_only` and `no_migration_calls_meka_s_own_code`;
rules 1 and 3 are not, and decay silently. The first digests *every* entry, so a legitimate append
fires it too: add the new entry's line to the expected vector, never paste current values over the
existing ones.

Rule 1 has two sanctioned exceptions, both of which converge on the current shape rather than
interpreting an old one: `classify_by_shape`, which runs once per store and stamps its answer, and
`memory::store::reconcile_index`, which makes this database's FTS triggers the ones this build
requires. Rule 1 is also about *the store*, which has a ledger. Config-format evolution belongs where
config is parsed, in serde defaults and value aliases; tolerance for what a model might emit is out of
scope entirely.

**Integrity guards are not compatibility.** A check that is equally true of a store created five
minutes ago defends against corruption and hand-editing; deleting it turns a fail-closed path into a
fail-open one. Keep those where the data is read.

Practical notes. The version is `PRAGMA user_version` (transactional, survives `VACUUM INTO`); never
write SQLite's unrelated `PRAGMA schema_version`. Numbers are list indices, not releases. Migration is
forward-only: downgrading means restoring the backup, and each new copy supersedes the last, so only
the most recent schema-changing upgrade is undoable.

## Built-in tool naming

Names are read by the model every turn and `tool_catalogue` is sorted, so a name is both label and
sort key.

- **A family shares a noun prefix**: `<subsystem>_<verb>`, which is what makes the family arrive as
  one sorted block. It names what the tools act on, which is not always the module they live in.
  Where a subsystem manages more than one kind of object, qualify before the verb and keep the object
  first. A verb that merely mentions a noun does not make it a managed object.
- **A standalone tool reads as a verb phrase**: `<verb>_<object>`. A subsystem with one operation may
  use the bare noun.

Two exceptions. **An industry-standard name beats internal consistency**: models reach for
`read_file` and `execute_command` zero-shot, and renaming them trades accuracy for tidiness. And
**`load_tool` stays verb-first** despite acting on meka's own registry, because the name appears
verbatim in the `[Tool discovery]` preamble the model reads every turn.

Renaming a tool is breaking: names appear in config lists, user-authored skills, and the history of
every existing session. Prefer getting it right at introduction. When renaming anyway, add a
`**Breaking:**` changelog line and update `BUILTIN_TOOL_NAMES` (sorted), `MCP_META_TOOL_NAMES`, and
`tool_display_name` in `src/render.rs`. Two silent traps: a blanket find-and-replace rewrites MCP tool names containing a
built-in as a substring, so anchor every substitution to a name boundary; and reversing word order
defeats the edit-distance `did_you_mean_hint`, so nothing points a resumed model at the new name.

## Documentation

Update the mdBook docs under `docs/book/src/` for any user-facing change, and the upgrade guide for
anything marked `**Breaking:**`.
