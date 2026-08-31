//! `meka session` and `meka history`: the stored-conversation CLI.
//!
//! Split out of `main.rs`, which owned every subcommand's implementation alongside the process
//! wiring and the interactive loop. Nothing here is reachable from a turn: these are the commands a
//! human runs against conversations that already exist, so they take a [`SessionManager`] and
//! nothing else of the agent.
//!
//! Export and import are the substantial part. The JSON form is versioned
//! ([`SESSION_EXPORT_FORMAT_VERSION`]) and carries sub-agent descendants alongside their parent;
//! the Markdown form is a rendering for people, and is not re-importable.

use crate::{cli, conversation, provider, render, session::SessionManager};

/// On-wire format version for `meka session export --format json`. Bumped when the envelope shape
/// or the underlying [`crate::conversation::Event`] serialization changes incompatibly; `meka
/// session import` rejects versions it doesn't recognize.
pub(crate) const SESSION_EXPORT_FORMAT_VERSION: u32 = 1;

/// Sessions one `POST /v1/sessions/import` will accept.
///
/// Enforced by the HTTP handler, not by [`plan_import`], because the reason for it is
/// contention-specific: `import_sessions` runs the whole tree in one closure on the process's
/// single SQLite connection, so every other in-flight request queues behind it. A one-shot
/// `meka session import` restoring its own backup has nothing to contend with, and refusing it
/// would mean a tree that exported fine cannot be restored.
pub(crate) const MAX_IMPORT_SESSIONS: usize = 1_000;

/// Root envelope for a JSON session export. Carries the session plus any sub-agent descendants as a
/// flat, root-first list; parent links are by original id and get remapped on import. Deliberately
/// secret-free: credentials live in separate global tables and the `token_id` fingerprint is
/// omitted.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct SessionExport {
    format_version: u32,
    meka_version: String,
    exported_at: String,
    root_session_id: String,
    /// Reachable from outside because `POST /v1/sessions/import` enforces
    /// [`MAX_IMPORT_SESSIONS`] on the parsed body before handing it to [`plan_import`]; see that
    /// constant for why the cap is the handler's and not the planner's.
    pub(crate) sessions: Vec<ExportedSession>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct ExportedSession {
    id: String,
    parent_id: Option<String>,
    created_at: String,
    updated_at: String,
    cwd: Option<String>,
    permission: Option<String>,
    capabilities_json: Option<String>,
    /// Workspace roots beyond `cwd`. `#[serde(default)]` rather than a `format_version` bump:
    /// [`plan_import`] rejects any version it doesn't equal exactly, so bumping would make every
    /// export written before this field unimportable, while an absent field already means the
    /// single-root sessions those exports describe.
    #[serde(default)]
    additional_roots: Vec<std::path::PathBuf>,
    /// A sub-agent's spawn terms. `#[serde(default)]` for the same reason as `additional_roots`:
    /// an archive written before the field existed is still importable, and its sub-agents simply
    /// come back unfollowable rather than unimportable.
    #[serde(default)]
    subagent_spec_json: Option<String>,
    /// The provider profile the session ran on. `#[serde(default)]` for the same reason as the two
    /// fields above: an archive written before meka recorded this is still importable, and
    /// [`plan_import`] settles the empty case against the importing installation's default rather
    /// than storing a profile no configuration can name.
    #[serde(default)]
    provider: String,
    stats: crate::stats::SessionStatsSnapshot,
    events: Vec<ExportedEvent>,
    tool_outputs: std::collections::BTreeMap<String, String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct ExportedEvent {
    /// RFC 3339 timestamp the event row was persisted; preserved across import.
    at: String,
    event: crate::conversation::Event,
}

/// Returns the file the export landed in, or `None` when the body went to stdout.
///
/// The path is returned rather than only logged because `/export` in the REPL writes to a generated
/// name in the working directory: the CLI can leave "quiet on success" to the shell, but a REPL
/// user who is not told the name has no way to find the file.
pub(crate) async fn export_session(
    session_manager: &SessionManager,
    session_id: uuid::Uuid,
    output: Option<&str>,
    format: cli::SessionExportFormat,
) -> anyhow::Result<Option<std::path::PathBuf>> {
    if !session_manager.session_exists(session_id).await? {
        anyhow::bail!("session not found: {}", session_id);
    }

    let (body, default_ext) = match format {
        cli::SessionExportFormat::Markdown => {
            // Export the full event log so pre-compaction turns are included. Compaction only hides
            // older turns from the model (it appends a boundary, never deletes), so the export
            // walks the raw log and renders every turn plus a marker at each compaction point.
            let events = session_manager.load_events(session_id).await?;
            let tool_outputs: std::collections::HashMap<String, String> = session_manager
                .load_all_tool_outputs(session_id)
                .await?
                .into_iter()
                .collect();
            (
                format_session_as_markdown(session_id, &events, &tool_outputs),
                "md",
            )
        }
        cli::SessionExportFormat::Json => {
            let export = build_session_export(session_manager, session_id).await?;
            (serde_json::to_string_pretty(&export)?, "json")
        }
    };

    match output {
        Some("-") => {
            print!("{}", body);
            Ok(None)
        }
        Some(path) => {
            std::fs::write(path, &body)?;
            tracing::info!("exported session to {}", path);
            Ok(Some(std::path::PathBuf::from(path)))
        }
        None => {
            let path = std::path::PathBuf::from(format!("session-{}.{}", session_id, default_ext));
            std::fs::write(&path, &body)?;
            tracing::info!("exported session to {}", path.display());
            Ok(Some(path))
        }
    }
}

/// Assemble the structured JSON export envelope for a session and every sub-agent descendant.
/// Per-event timestamps and cumulative stats are preserved; `token_id` is intentionally excluded.
pub(crate) async fn build_session_export(
    session_manager: &SessionManager,
    root: uuid::Uuid,
) -> anyhow::Result<SessionExport> {
    let tree = session_manager.load_session_tree(root).await?;
    let mut sessions = Vec::with_capacity(tree.len());
    for meta in tree {
        let events = session_manager
            .load_events_with_timestamps(meta.id)
            .await?
            .into_iter()
            .map(|(at, event)| ExportedEvent { at, event })
            .collect();
        let tool_outputs = session_manager
            .load_all_tool_outputs(meta.id)
            .await?
            .into_iter()
            .collect();
        let stats = session_manager.load_session_stats(meta.id).await?;
        sessions.push(ExportedSession {
            id: meta.id.to_string(),
            parent_id: meta.parent_id.map(|id| id.to_string()),
            created_at: meta.created_at,
            updated_at: meta.updated_at,
            cwd: meta.cwd,
            permission: meta.permission,
            capabilities_json: meta.capabilities_json,
            additional_roots: meta.additional_roots,
            subagent_spec_json: meta.subagent_spec_json,
            provider: meta.provider,
            stats,
            events,
            tool_outputs,
        });
    }
    Ok(SessionExport {
        format_version: SESSION_EXPORT_FORMAT_VERSION,
        meka_version: env!("CARGO_PKG_VERSION").to_string(),
        exported_at: chrono::Utc::now().to_rfc3339(),
        root_session_id: root.to_string(),
        sessions,
    })
}

/// Import a session (and any sub-agent children) from a JSON export produced by
/// `meka session export --format json`. Reads `input` (a file path, or `-` for stdin), mints fresh
/// IDs for every session, rewires parent links, and persists the whole tree in one transaction.
/// Prints the new root session ID to stdout.
pub(crate) async fn import_session(
    session_manager: &SessionManager,
    input: &str,
    default_profile: Option<&str>,
) -> anyhow::Result<()> {
    let raw = if input == "-" {
        use std::io::Read as _;
        let mut buffer = String::new();
        std::io::stdin().read_to_string(&mut buffer)?;
        buffer
    } else {
        std::fs::read_to_string(input)
            .map_err(|error| anyhow::anyhow!("failed to read '{}': {}", input, error))?
    };

    let export: SessionExport = serde_json::from_str(&raw)
        .map_err(|error| anyhow::anyhow!("invalid session export JSON: {}", error))?;
    let (records, root_new_id) = plan_import(export, default_profile)?;

    let count = records.len();
    session_manager.import_sessions(records).await?;
    tracing::info!("imported {} session(s) from {}", count, input);
    // Human-facing confirmation and resume guidance go to stderr; the bare root ID stays on stdout
    // so `id=$(meka session import ...)` and piping keep working. Plain (unstyled) to match the
    // other one-shot CLI messages; `render_hint`'s dark-grey styling is for the REPL.
    //
    // The resume line is conditional for the reason `meka session fork`'s is: exporting a worker
    // on its own and importing it here produces a row that carries spawn terms with no parent, and
    // `meka -r` refuses exactly that. Printing it unconditionally handed the user a command this
    // release had just made illegal, one line after telling them the import worked.
    let spawned = match session_manager.spawn_terms(root_new_id).await {
        Ok(terms) => terms.is_some(),
        Err(error) => {
            tracing::debug!(
                "could not read the imported row to choose a hint, assuming top-level: {}",
                error
            );
            false
        }
    };
    if spawned {
        eprintln!(
            "Imported a sub-agent's conversation. It carries the terms the session that spawned \
             it set, and that session is not in this archive, so `meka -r` will refuse it. It is \
             readable with `meka session export` and `meka session list --include-children`."
        );
    } else if count > 1 {
        eprintln!(
            "Imported session with {} sub-agent(s). Resume with: meka -r {}",
            count - 1,
            root_new_id
        );
    } else {
        eprintln!("Imported session. Resume with: meka -r {}", root_new_id);
    }
    println!("{}", root_new_id);
    Ok(())
}

/// Result of the REPL's `/fork`, which has to hand the on-disk session lock from the session it is
/// leaving to the copy it is entering.
pub(crate) enum ForkHandoff {
    /// The copy exists and its lock is held. The caller assigns this over its current lock, which
    /// releases the original only once the new one is owned.
    Switched {
        id: uuid::Uuid,
        lock: crate::session::FileLock,
    },
    /// The copy exists but its lock could not be taken, so the caller stays where it is. The id is
    /// carried so the user can still be told where the copy went.
    LockFailed {
        id: uuid::Uuid,
        error: crate::error::MekaError,
    },
    /// The session being forked no longer exists.
    SourceGone,
}

/// Fork `source` and take the copy's lock, in that order and without touching the caller's own.
///
/// The ordering is the point. Releasing the current lock first and then failing to acquire the new
/// one would leave the REPL running against an unlocked session that a second `meka` process could
/// open and interleave events into. Acquiring first means the failure path is simply "stay put",
/// and the caller drops its old lock only by overwriting it with the new one.
pub(crate) async fn fork_and_lock(
    session_manager: &SessionManager,
    source: uuid::Uuid,
) -> anyhow::Result<ForkHandoff> {
    // Locked before the copy's row exists, not after: a row committed ahead of its lock is one a
    // concurrent `session delete --all` enumerates and deletes, after which this function would
    // lock the vanished id successfully and hand the REPL a session whose next turn dies on a
    // foreign-key violation. See `SessionManager::fork_session_locked`.
    let Some((forked, lock)) = session_manager
        .fork_session_locked(source, crate::session::ForkOverrides::default())
        .await?
    else {
        return Ok(ForkHandoff::SourceGone);
    };
    match lock {
        Ok(lock) => Ok(ForkHandoff::Switched {
            id: forked.id,
            lock,
        }),
        Err(error) => Ok(ForkHandoff::LockFailed {
            id: forked.id,
            error,
        }),
    }
}

/// Hold a session still while its conversation is copied out of it.
///
/// Both CLI doors that copy a conversation read a run of rows that a concurrent turn may be halfway
/// through writing. `Agent::run_turn` persists the user message *eagerly*, before the provider has
/// answered, so a copy taken mid-turn ends on an unanswered user message: the fork reads
/// `user, user, assistant` from its first resumed turn onward, and an exported snapshot reproduces
/// that shape through `meka session import`. Measured 10/10 and 30/30 across two runs -- this is
/// not a race that sometimes bites, it is what happens every time you fork a session that is
/// thinking.
///
/// `meka session rewind` already took this lock and was correspondingly never affected; fork and
/// export did not, which is the whole of the difference.
///
/// Deliberately *not* pushed down into `fork_session` or [`export_session`]: the REPL's `/fork` and
/// `/export` act on the session the REPL is already holding, and `flock` is per open file
/// description, so asking again inside the same process would refuse a session that is
/// legitimately ours.
fn hold_still_for_a_copy(
    session_manager: &SessionManager,
    session_id: uuid::Uuid,
) -> anyhow::Result<crate::session::FileLock> {
    session_manager.lock_session(session_id).map_err(|error| {
        anyhow::anyhow!(
            "{}. A conversation cannot be copied while it is being written; close the meka that \
             has it open and try again",
            error
        )
    })
}

/// `meka session fork <id>`: copy a session's conversation into a new one and print the new ID.
///
/// Output split mirrors [`import_session`]: the bare ID on stdout so `id=$(meka session fork …)`
/// works, the resume hint on stderr.
pub(crate) async fn fork_session_command(
    session_manager: &SessionManager,
    session_id: uuid::Uuid,
) -> anyhow::Result<()> {
    let _source = hold_still_for_a_copy(session_manager, session_id)?;
    let forked = session_manager
        .fork_session(session_id, crate::session::ForkOverrides::default())
        .await?
        .ok_or_else(|| anyhow::anyhow!("session not found: {}", session_id))?;

    tracing::info!("forked session {} into {}", session_id, forked.id);
    // A fork of a sub-agent is a sibling under the same parent, so the copy is a worker too and
    // `meka -r` refuses it. Printing the resume line anyway would hand the user a command that
    // answers with a refusal -- the copy is real and readable, but continuing it is the parent's
    // job, and that is what the hint has to say.
    // A read failure decides only which hint is printed, so it must not fail a fork that has
    // already landed -- but it must not be silent either: the fallback prints `meka -r`, which is
    // the wrong advice for a copy of a worker, so a reader of the logs needs to know the question
    // went unanswered.
    let spawned = match session_manager.spawn_terms(forked.id).await {
        Ok(terms) => terms,
        Err(error) => {
            tracing::debug!(
                "could not read the fork's row to choose a hint, assuming top-level: {}",
                error
            );
            None
        }
    };
    match spawned.map(|terms| terms.parent) {
        Some(Some(parent)) => eprintln!(
            "Forked session. It is a sub-agent of {parent}, like the session it copies, so \
             continue it with `agent_followup` from {parent} rather than `meka -r`."
        ),
        // A copy of a worker whose parent is not in this store. Still a worker, still not
        // resumable, and with no id to name; saying so beats printing `meka -r`.
        Some(None) => eprintln!(
            "Forked session. It carries the terms another session spawned the original under, and \
             that session is not in this store, so `meka -r` will refuse it."
        ),
        None => eprintln!("Forked session. Resume with: meka -r {}", forked.id),
    }
    println!("{}", forked.id);
    Ok(())
}

/// Turn a deserialized [`SessionExport`] into the parents-first
/// [`crate::session::ImportSessionRecord`] list to persist, plus the freshly-minted root session
/// ID. Validates the format version, mints a new ID per session, and remaps parent links (a parent
/// pointing outside the exported set collapses to `None`, importing that session as a new top-level
/// session). Pure and I/O-free so the ID-remap and ordering are unit-testable.
pub(crate) fn plan_import(
    export: SessionExport,
    // What an archive that names no profile adopts. Settled here rather than at the reader because
    // an import is where a session enters *this* installation, and this installation's default is
    // the only thing that can be known about an archive that names none. `None` refuses the import
    // rather than writing a session that cannot run.
    default_profile: Option<&str>,
) -> anyhow::Result<(Vec<crate::session::ImportSessionRecord>, uuid::Uuid)> {
    if export.format_version != SESSION_EXPORT_FORMAT_VERSION {
        anyhow::bail!(
            "unsupported session export format_version {} (this build supports {})",
            export.format_version,
            SESSION_EXPORT_FORMAT_VERSION
        );
    }
    if export.sessions.is_empty() {
        anyhow::bail!("session export contains no sessions");
    }
    // Caught here rather than at the `sessions.id` primary key, which would surface a caller's
    // malformed envelope as an internal error.
    let mut seen = std::collections::HashSet::with_capacity(export.sessions.len());
    for session in &export.sessions {
        if !seen.insert(session.id.clone()) {
            anyhow::bail!("session export contains duplicate id '{}'", session.id);
        }
    }

    let remap: std::collections::HashMap<String, uuid::Uuid> = export
        .sessions
        .iter()
        .map(|session| (session.id.clone(), uuid::Uuid::new_v4()))
        .collect();
    let root_new_id = remap
        .get(&export.root_session_id)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("root_session_id is not present in the sessions list"))?;

    let nodes: Vec<(String, Option<String>)> = export
        .sessions
        .iter()
        .map(|session| (session.id.clone(), session.parent_id.clone()))
        .collect();
    let order = parents_first_order(&nodes)?;

    let mut slots: Vec<Option<ExportedSession>> = export.sessions.into_iter().map(Some).collect();
    let mut records = Vec::with_capacity(order.len());
    for index in order {
        let session = slots[index]
            .take()
            .ok_or_else(|| anyhow::anyhow!("duplicate session index while ordering import"))?;
        let new_id = remap
            .get(&session.id)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("internal error: session id missing from ID remap"))?;
        let new_parent_id = session
            .parent_id
            .as_ref()
            .and_then(|parent| remap.get(parent).copied());
        records.push(crate::session::ImportSessionRecord {
            new_id,
            new_parent_id,
            created_at: session.created_at,
            cwd: session.cwd,
            permission: session.permission,
            capabilities_json: session.capabilities_json,
            additional_roots: session.additional_roots,
            subagent_spec_json: session.subagent_spec_json,
            provider: if session.provider.is_empty() {
                // Refused rather than left blank. A session with no profile cannot run, so
                // importing one is writing a row whose only future is a refusal the user has to
                // work backwards from -- and it would put a state into the store that no other
                // door can produce, which every reader would then have to know about.
                let Some(default_profile) = default_profile else {
                    anyhow::bail!(
                        "this archive names no provider profile, and no default is configured to \
                         give it one. Run `meka provider use <name>`, or import with \
                         `meka --provider <name> session import`"
                    );
                };
                default_profile.to_string()
            } else {
                session.provider
            },
            stats: session.stats,
            events: session
                .events
                .into_iter()
                .map(|event| (event.at, event.event))
                .collect(),
            tool_outputs: session.tool_outputs.into_iter().collect(),
        });
    }

    Ok((records, root_new_id))
}

/// Order sessions parents-first (a topological sort over `parent_id` edges, considering only
/// parents present in the set) so an importer can insert each session after its parent and satisfy
/// the `parent_session_id` foreign key. Returns indices into `nodes`. Errors on a cyclic
/// relationship. Sessions whose parent is absent from the set are treated as roots.
pub(crate) fn parents_first_order(
    nodes: &[(String, Option<String>)],
) -> anyhow::Result<Vec<usize>> {
    use std::collections::{HashMap, VecDeque};

    let index_of: HashMap<&str, usize> = nodes
        .iter()
        .enumerate()
        .map(|(index, (id, _))| (id.as_str(), index))
        .collect();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    let mut indegree = vec![0usize; nodes.len()];
    for (index, (_, parent)) in nodes.iter().enumerate() {
        if let Some(parent) = parent
            && let Some(&parent_index) = index_of.get(parent.as_str())
        {
            children[parent_index].push(index);
            indegree[index] += 1;
        }
    }
    let mut queue: VecDeque<usize> = (0..nodes.len()).filter(|&i| indegree[i] == 0).collect();
    let mut order = Vec::with_capacity(nodes.len());
    while let Some(node) = queue.pop_front() {
        order.push(node);
        for &child in &children[node] {
            indegree[child] -= 1;
            if indegree[child] == 0 {
                queue.push_back(child);
            }
        }
    }
    if order.len() != nodes.len() {
        anyhow::bail!("session export has a cyclic parent relationship");
    }
    Ok(order)
}

pub(crate) async fn run_session_subcommand(
    session_manager: &SessionManager,
    action: &cli::SessionAction,
    // The profile an archive that names none adopts on `import`. The caller reads two answers off
    // disk, and this is the flag-aware one, so `meka --provider work session import` chooses per
    // run; the migration context gets the other, which ignores `--provider` because it stamps rows
    // once and irreversibly. Unused by every other action here.
    default_profile: Option<&str>,
) -> anyhow::Result<()> {
    match action {
        cli::SessionAction::List {
            limit,
            include_children,
        } => list_sessions(session_manager, *limit, *include_children).await,
        cli::SessionAction::Export {
            session_id,
            output,
            format,
        } => {
            // Held for the same reason `fork` holds it: an export is a snapshot, and a snapshot of
            // a conversation mid-turn carries an unanswered user message that `meka session
            // import` then restores as an unusable session. See `hold_still_for_a_copy`.
            let _source = hold_still_for_a_copy(session_manager, *session_id)?;
            // The written path is only interesting to the REPL; out here the shell (and the `-o`
            // the user typed) already knows where it went.
            export_session(session_manager, *session_id, output.as_deref(), *format).await?;
            Ok(())
        }
        cli::SessionAction::Delete {
            session_ids,
            all,
            older_than_days,
        } => delete_sessions(session_manager, session_ids, *all, *older_than_days).await,
        cli::SessionAction::Import { input } => {
            import_session(session_manager, input, default_profile).await
        }
        cli::SessionAction::Fork { session_id } => {
            fork_session_command(session_manager, *session_id).await
        }
        cli::SessionAction::Rewind { session_id, turns } => {
            rewind_session_command(session_manager, *session_id, *turns).await
        }
    }
}

/// `meka session rewind`: drop the last `turns` turns from a session that isn't currently open.
///
/// The escape hatch for content `Agent::run_turn` can't repair itself, namely anything the provider
/// refuses that was committed before the current turn. Appends an `Event::Repair` with an empty
/// replacement, so nothing is deleted and `meka session export` still shows the dropped turns.
pub(crate) async fn rewind_session_command(
    session_manager: &SessionManager,
    session_id: uuid::Uuid,
    turns: usize,
) -> anyhow::Result<()> {
    // Rejected before anything else: `Conversation::rewind(0)` returns `None` unconditionally, so
    // the error below would otherwise say the session has "fewer than 0 turn(s)".
    if turns == 0 {
        anyhow::bail!("-n must be 1 or more");
    }
    // Held for the whole read-modify-write. A REPL, `meka serve`, or `meka acp` holding this
    // session has its own in-memory conversation that would overwrite the rewind on its next turn.
    if !session_manager.session_exists(session_id).await? {
        anyhow::bail!("session not found: {}", session_id);
    }
    let _lock = session_manager.lock_session(session_id)?;

    let events = session_manager.load_events(session_id).await?;
    let mut conversation = conversation::Conversation::from_events(events);

    let Some(event) = conversation.rewind(turns) else {
        anyhow::bail!(
            "nothing to rewind: session {} has fewer than {} turn(s)",
            session_id,
            turns
        );
    };
    session_manager.save_event(session_id, &event).await?;

    tracing::info!("rewound {} turn(s) from session {}", turns, session_id);
    eprintln!(
        "Rewound {} turn(s); {} message(s) remain. The full history is still in \
         `meka session export`.",
        turns,
        conversation.len(),
    );
    Ok(())
}

pub(crate) async fn list_sessions(
    session_manager: &SessionManager,
    limit: u32,
    include_children: bool,
) -> anyhow::Result<()> {
    let (sessions, _next_cursor) = session_manager
        .list_sessions(limit, include_children, None, None)
        .await?;

    if sessions.is_empty() {
        eprintln!("No sessions found.");
        return Ok(());
    }

    let rows: Vec<Vec<String>> = sessions
        .iter()
        .map(|session| {
            let mut row = vec![
                session.id.to_string(),
                format_timestamp(&session.updated_at),
                // Sanitised for the same reason every other authored cell in a meka table is: the
                // value is a config key the user chose, and a table is not a place to reproduce
                // control characters verbatim.
                render::sanitize_for_display(&session.provider),
            ];
            row.push(session.preview.clone());
            row
        })
        .collect();
    let headers: &[&str] = &["ID", "Updated", "Provider", "Preview"];
    print!("{}", render::format_columns(headers, &rows));

    Ok(())
}

pub(crate) async fn delete_sessions(
    session_manager: &SessionManager,
    session_ids: &[uuid::Uuid],
    all: bool,
    older_than_days: Option<u64>,
) -> anyhow::Result<()> {
    if all {
        let sweep = session_manager.delete_all_sessions().await?;
        tracing::info!("deleted {} session(s)", sweep.deleted);
        report_sessions_left_open(sweep);
        return Ok(());
    }

    // The manual counterpart to `[session].retention_days`, now that nothing prunes on its own.
    // Reports the count through `info!` like the `--all` and by-id branches below: the user ran
    // this to delete, not to obtain a number, and the exit code already carries success.
    if let Some(days) = older_than_days {
        // Zero would sweep everything, which is `--all` by another name and far too easy to type
        // by accident when you meant "today's".
        if days == 0 {
            anyhow::bail!(
                "--older-than-days 0 would delete every session; use --all if you mean that"
            );
        }
        let sweep = session_manager.delete_expired_sessions(days).await?;
        tracing::info!(
            "deleted {} session(s) not updated in {} days",
            sweep.deleted,
            days
        );
        report_sessions_left_open(sweep);
        return Ok(());
    }

    if session_ids.is_empty() {
        anyhow::bail!("specify one or more session IDs, --older-than-days <DAYS>, or --all");
    }

    let mut deleted = 0u64;
    // Reported at the end rather than returned at the first refusal. Every id the user named is a
    // separate request, and one of them being in use is no reason to leave the rest of the list
    // untried -- nor to swallow the count of what did go, which returning early also did.
    let mut refused = Vec::new();
    for session_id in session_ids {
        // The refusing door, not the plain one: this is a session this process has never had open,
        // and deleting one another meka is mid-conversation on cascades its messages away
        // underneath a live agent, which then fails every remaining turn on a foreign-key
        // violation.
        match session_manager
            .delete_session_unless_attached(*session_id)
            .await
        {
            Ok(true) => deleted += 1,
            // User-facing error: they asked to delete a specific ID and we couldn't find it, so
            // stderr (not silent) is right.
            Ok(false) => eprintln!("Session not found: {}", session_id),
            Err(error) => {
                eprintln!("Cannot delete {}: {}", session_id, error);
                refused.push(*session_id);
            }
        }
    }

    tracing::info!("deleted {} session(s)", deleted);
    // A non-zero exit, because the user named these and a silent skip is indistinguishable from
    // success. The per-id reasons are already on stderr; this is what a script reads.
    if !refused.is_empty() {
        // Does not name a cause. Every `Err` lands here, and `delete_session_unless_attached`
        // returns a database error as readily as a lock refusal -- so claiming "another meka has
        // them open" would report a full disk as a busy session. The per-id lines above carry the
        // real reason; this is the summary a script reads.
        anyhow::bail!(
            "{} of {} session(s) could not be deleted; see the errors above",
            refused.len(),
            session_ids.len()
        );
    }
    Ok(())
}

/// Say what a sweep spared, because a count of deletions alone reads as "everything matched went".
///
/// `warn!` rather than `info!`: the user asked for these to be gone and some of them are not, which
/// is a fallback they should see at the default level rather than a lifecycle signpost.
fn report_sessions_left_open(sweep: crate::session::SessionSweep) {
    if sweep.attached_elsewhere > 0 {
        tracing::warn!(
            "left {} session(s) alone: another meka process has them open. Close it and run this \
             again",
            sweep.attached_elsewhere
        );
    }
}

pub(crate) fn format_timestamp(rfc3339: &str) -> String {
    chrono::DateTime::parse_from_rfc3339(rfc3339)
        .map(|datetime| datetime.format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or_else(|_| rfc3339.to_string())
}

pub(crate) fn format_session_as_markdown(
    session_id: uuid::Uuid,
    events: &[conversation::Event],
    tool_outputs: &std::collections::HashMap<String, String>,
) -> String {
    use std::fmt::Write;

    let mut output = String::new();
    writeln!(output, "# Session {}\n", session_id).ok();

    // Walk the raw event log so the full conversation is exported, including turns a compaction
    // later hid from the model. Each `CompactBoundary` becomes a marker; the turns it summarized
    // stay above it (the kept tail is re-appended after it, so the recent turns appear on both
    // sides of the marker, as stored).
    for event in events {
        match event {
            conversation::Event::Append(message) => {
                write_message_markdown(&mut output, message, tool_outputs);
            }
            conversation::Event::CompactBoundary { summary, .. } => {
                writeln!(output, "---\n").ok();
                writeln!(output, "<details>").ok();
                writeln!(
                    output,
                    "<summary>Session compaction (summary the model saw in place of the turns above)</summary>\n"
                )
                .ok();
                writeln!(output, "{}\n", summary.text_content()).ok();
                writeln!(output, "</details>\n").ok();
            }
            // Same treatment as a boundary: mark what happened and render the replacement, leaving
            // the superseded messages above it. An export is the record of the session, and a
            // repair (or a rewind, which is a repair with nothing to put back) is the one place
            // where what the model saw and what actually happened diverge.
            conversation::Event::Repair {
                replaced_count,
                messages,
            } => {
                writeln!(output, "---\n").ok();
                writeln!(output, "<details>").ok();
                writeln!(
                    output,
                    "<summary>{} message(s) above replaced with {} (rejected by the provider, or rewound)</summary>\n",
                    replaced_count,
                    if messages.is_empty() {
                        "nothing".to_string()
                    } else {
                        format!("{} message(s)", messages.len())
                    },
                )
                .ok();
                for message in messages {
                    write_message_markdown(&mut output, message, tool_outputs);
                }
                writeln!(output, "</details>\n").ok();
            }
        }
    }

    output
}

pub(crate) fn write_message_markdown(
    output: &mut String,
    message: &provider::Message,
    tool_outputs: &std::collections::HashMap<String, String>,
) {
    use std::fmt::Write;

    match message.role {
        provider::Role::User => {
            // A "user" message can be either a plain user turn or a tool_results envelope.
            // Inspect content blocks rather than role to decide.
            let has_tool_results = message
                .content
                .iter()
                .any(|block| matches!(block, provider::ContentBlock::ToolResult { .. }));
            if has_tool_results {
                for block in &message.content {
                    if let provider::ContentBlock::ToolResult {
                        content, is_error, ..
                    } = block
                    {
                        let label = if *is_error {
                            "Tool result (error)"
                        } else {
                            "Tool result"
                        };
                        writeln!(output, "<details>").ok();
                        writeln!(output, "<summary>{}</summary>\n", label).ok();
                        let text = provider::ContentBlock::tool_result_text_content(content);
                        let text = resolve_large_output_tags(&text, tool_outputs);
                        writeln!(output, "```\n{}\n```\n", text).ok();
                        writeln!(output, "</details>\n").ok();
                    }
                }
            } else {
                writeln!(output, "## User\n").ok();
                writeln!(output, "{}\n", message.text_content()).ok();
            }
        }
        provider::Role::Assistant => {
            writeln!(output, "## Assistant\n").ok();
            for block in &message.content {
                match block {
                    provider::ContentBlock::Text { text } => {
                        writeln!(output, "{}\n", text).ok();
                    }
                    provider::ContentBlock::ToolUse { name, input, .. } => {
                        let input_pretty = serde_json::to_string_pretty(input)
                            .unwrap_or_else(|_| input.to_string());
                        writeln!(output, "<details>").ok();
                        writeln!(output, "<summary>Tool call: {}</summary>\n", name).ok();
                        writeln!(output, "```json\n{}\n```\n", input_pretty).ok();
                        writeln!(output, "</details>\n").ok();
                    }
                    provider::ContentBlock::ToolResult { .. }
                    | provider::ContentBlock::Thinking { .. }
                    | provider::ContentBlock::RedactedThinking { .. }
                    | provider::ContentBlock::Image { .. } => {}
                }
            }
        }
    }
}

pub(crate) fn resolve_large_output_tags(
    text: &str,
    tool_outputs: &std::collections::HashMap<String, String>,
) -> String {
    let re = match regex::Regex::new(r#"<large-output name="([^"]+)"[^>]*>[\s\S]*?</large-output>"#)
    {
        Ok(re) => re,
        Err(_) => return text.to_string(),
    };

    re.replace_all(text, |caps: &regex::Captures| {
        let name = &caps[1];
        match tool_outputs.get(name) {
            Some(content) => content.clone(),
            None => caps[0].to_string(),
        }
    })
    .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user_msg(text: &str) -> provider::Message {
        provider::Message::user(text)
    }

    fn assistant_text(text: &str) -> provider::Message {
        provider::Message::assistant_text(text)
    }

    #[test]
    fn test_parents_first_order_rejects_cycle() {
        let nodes = vec![
            ("a".to_string(), Some("b".to_string())),
            ("b".to_string(), Some("a".to_string())),
        ];
        assert!(parents_first_order(&nodes).is_err());
    }

    #[test]
    fn test_plan_import_rejects_unknown_format_version() {
        let export = SessionExport {
            format_version: SESSION_EXPORT_FORMAT_VERSION + 1,
            meka_version: "test".into(),
            exported_at: "now".into(),
            root_session_id: "r".into(),
            sessions: Vec::new(),
        };
        assert!(plan_import(export, None).is_err());
    }

    #[tokio::test]
    async fn test_session_export_import_round_trip() {
        use std::path::Path;

        use crate::{
            conversation::Event,
            provider::{ContentBlock, ImageSource, Message, Role, ToolResultContent},
        };

        let manager = SessionManager::open(Some(Path::new(":memory:")), &Default::default())
            .await
            .expect("open");

        // Root session with a representative mix of events: plain text, an input image, a
        // reasoning block, a tool_use/tool_result pair, and a compaction boundary.
        let root = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("root");
        let image = ImageSource {
            source_type: "base64".to_string(),
            media_type: "image/png".to_string(),
            data: "aGk=".to_string(),
        };
        let root_events = vec![
            Event::Append(Message::user("hello")),
            Event::Append(Message::user_with_images("look", vec![image])),
            Event::Append(Message {
                role: Role::Assistant,
                // Both opaque halves of a Responses reasoning block. Neither is readable and
                // neither is reconstructible, so an export that dropped them would leave the
                // imported session unable to replay its own reasoning, silently.
                content: vec![
                    ContentBlock::Thinking {
                        thinking: "weighing it up".to_string(),
                        opaque: Some(provider::OpaqueReasoning::Sealed {
                            encrypted_content: "OPAQUE".to_string(),
                            id: Some("rs_1".to_string()),
                        }),
                    },
                    ContentBlock::ToolUse {
                        id: "u1".to_string(),
                        name: "read".to_string(),
                        input: serde_json::json!({"path": "/x"}),
                    },
                ],
            }),
            Event::Append(Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "u1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "ok".to_string(),
                    }],
                    is_error: false,
                }],
            }),
            Event::CompactBoundary {
                summary: Message::user("[summary]"),
                replaced_count: 2,
                loaded_tools_snapshot: Default::default(),
            },
        ];
        for event in &root_events {
            manager
                .save_event(root, event)
                .await
                .expect("save root event");
        }
        manager
            .save_tool_output(root, "tool_1_output", "big output")
            .await
            .expect("tool output");
        let stats = crate::stats::SessionStatsSnapshot {
            turns: 3,
            input_tokens: 1000,
            ..Default::default()
        };
        manager
            .save_session_stats(root, &stats)
            .await
            .expect("stats");

        // A sub-agent child of the root, with the spawn terms `agent_followup` reconstructs from.
        // An archive that drops these imports a worker nobody can resume.
        let child_spec = r#"{"permission":"read","enabled_permissions":["read"],"denied_servers":["mekabridge"],"denied_tools":[],"memory":"none","inherited_scratchpad":[],"remaining_depth":0,"absolute_depth":1}"#;
        let child = manager
            .create_child_session(
                root,
                None,
                Some(child_spec.to_string()),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;
        for event in [
            Event::Append(Message::user("sub task")),
            Event::Append(Message::assistant_text("sub done")),
        ] {
            manager.save_event(child, &event).await.expect("save child");
        }

        // Export -> JSON -> back.
        let export = build_session_export(&manager, root).await.expect("export");
        assert_eq!(export.sessions.len(), 2, "root + child");
        assert_eq!(export.sessions[0].id, root.to_string(), "root first");
        let json = serde_json::to_string_pretty(&export).expect("serialize");
        assert!(
            !json.contains("token_id"),
            "the fingerprint must not be exported"
        );
        let reparsed: SessionExport = serde_json::from_str(&json).expect("deserialize");

        // Import under fresh IDs.
        let (records, root_new_id) = plan_import(reparsed, None).expect("plan");
        assert_ne!(root_new_id, root, "import mints a new id");
        manager.import_sessions(records).await.expect("import");

        // The tree came back: root + child, with the child's parent rewired to the new root.
        let tree = manager.load_session_tree(root_new_id).await.expect("tree");
        assert_eq!(tree.len(), 2);
        let child_new = tree
            .iter()
            .find(|meta| meta.id != root_new_id)
            .expect("child present");
        assert_eq!(child_new.parent_id, Some(root_new_id));
        // The spawn terms survived export -> JSON -> import. This is also the column-alignment
        // check on `import_sessions`' 18-parameter INSERT: reading the spec back verbatim off a
        // different column would surface here as a mismatch rather than silently.
        assert_eq!(
            manager
                .load_subagent_spec(child_new.id)
                .await
                .expect("load spec"),
            Some(child_spec.to_string()),
        );
        assert_eq!(
            manager
                .load_subagent_spec(root_new_id)
                .await
                .expect("load root spec"),
            None,
            "a top-level session has no spawn terms",
        );
        assert_eq!(
            child_new.cwd, None,
            "and neighbouring columns are undisturbed"
        );
        assert_eq!(child_new.permission, None);

        // The event log round-trips byte-for-byte against the untouched original.
        let imported = manager
            .load_events(root_new_id)
            .await
            .expect("load imported");
        let original = manager.load_events(root).await.expect("load original");
        assert_eq!(
            serde_json::to_string(&imported).unwrap(),
            serde_json::to_string(&original).unwrap(),
        );
        assert!(
            imported.iter().any(|event| match event {
                Event::Append(message) => message
                    .content
                    .iter()
                    .any(|block| matches!(block, ContentBlock::Image { .. })),
                _ => false,
            }),
            "the input image must survive the round trip",
        );
        assert!(
            imported.iter().any(|event| match event {
                Event::Append(message) => message.content.iter().any(|block| matches!(
                    block,
                    ContentBlock::Thinking {
                        opaque: Some(provider::OpaqueReasoning::Sealed { encrypted_content, id }),
                        ..
                    } if encrypted_content == "OPAQUE" && id.as_deref() == Some("rs_1")
                )),
                _ => false,
            }),
            "and so must the opaque reasoning, or the imported session cannot replay it",
        );

        // Child events, stats, and tool_outputs are preserved.
        assert_eq!(
            manager
                .load_events(child_new.id)
                .await
                .expect("load child events")
                .len(),
            2,
        );
        let imported_stats = manager
            .load_session_stats(root_new_id)
            .await
            .expect("load stats");
        assert_eq!(imported_stats.turns, 3);
        assert_eq!(imported_stats.input_tokens, 1000);
        assert_eq!(
            manager
                .load_all_tool_outputs(root_new_id)
                .await
                .expect("load outputs"),
            vec![("tool_1_output".to_string(), "big output".to_string())],
        );
    }

    /// One id being refused must not cost the user the rest of the list.
    ///
    /// Each id on the command line is a separate request, and a session another meka has open is a
    /// refusal about that one. Returning at the first refusal skipped every id after it -- and
    /// swallowed the count of what *had* been deleted on the way, so the run reported nothing at
    /// all about work it had actually done.
    #[tokio::test]
    async fn a_refused_session_does_not_abandon_the_rest_of_the_list() {
        use std::path::Path;

        let manager = SessionManager::open(Some(Path::new(":memory:")), &Default::default())
            .await
            .expect("open");
        let held = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let after = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let _lock = manager
            .lock_session(held)
            .expect("another process holds it");

        let outcome = delete_sessions(&manager, &[held, after], false, None).await;

        assert!(
            outcome.is_err(),
            "a refusal the user named has to reach the exit code"
        );
        assert!(
            manager.session_exists(held).await.expect("exists"),
            "the conversation somebody is having survives"
        );
        assert!(
            !manager.session_exists(after).await.expect("exists"),
            "and the id listed after it is still deleted rather than skipped"
        );
    }

    /// `/fork` must own the copy's lock before the REPL lets go of the one it is holding. That
    /// ordering is now structural rather than tested: [`fork_and_lock`] is
    /// handed no lock, so it has no way to release the caller's, and the caller can only give
    /// its up by assigning the returned one over it.
    ///
    /// What this pins is the pair of facts that make the structure sound: the returned lock is
    /// genuinely held on the copy (not a stale handle the REPL would rely on), and the source's
    /// lock is untouched, so the failure path really is "stay put".
    #[tokio::test]
    async fn test_fork_and_lock_holds_both_locks_at_the_handoff() {
        use std::path::Path;

        let manager = SessionManager::open(Some(Path::new(":memory:")), &Default::default())
            .await
            .expect("open");
        let source = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let source_lock = manager.lock_session(source).expect("lock source");

        let handoff = fork_and_lock(&manager, source).await.expect("fork");
        let ForkHandoff::Switched { id, lock } = handoff else {
            panic!("expected a switch");
        };

        assert!(
            manager.lock_session(id).is_err(),
            "the returned lock must actually be held on the copy"
        );
        assert!(
            manager.lock_session(source).is_err(),
            "and the source's lock must still be held: releasing it first is the bug"
        );

        // Only once the caller drops the old guard does the source become available again.
        drop(source_lock);
        manager.lock_session(source).expect("source is free again");
        drop(lock);
    }

    #[tokio::test]
    async fn test_fork_and_lock_reports_a_missing_source() {
        use std::path::Path;

        let manager = SessionManager::open(Some(Path::new(":memory:")), &Default::default())
            .await
            .expect("open");
        assert!(matches!(
            fork_and_lock(&manager, uuid::Uuid::new_v4())
                .await
                .expect("fork"),
            ForkHandoff::SourceGone,
        ));
    }

    /// Multi-root sessions used to come back from an export as single-root: the column existed but
    /// no export/import struct carried it.
    #[tokio::test]
    async fn test_session_export_preserves_additional_roots() {
        use std::path::{Path, PathBuf};

        let manager = SessionManager::open(Some(Path::new(":memory:")), &Default::default())
            .await
            .expect("open");
        let root = manager
            .create_session(
                Some(PathBuf::from("/work/main")),
                "test-profile".to_string(),
            )
            .await
            .expect("root");
        let roots = vec![PathBuf::from("/work/shared"), PathBuf::from("/work/docs")];
        manager
            .update_session_roots(root, &roots)
            .await
            .expect("roots");

        let export = build_session_export(&manager, root).await.expect("export");
        let json = serde_json::to_string(&export).expect("serialize");
        let reparsed: SessionExport = serde_json::from_str(&json).expect("deserialize");
        let (records, new_id) = plan_import(reparsed, None).expect("plan");
        manager.import_sessions(records).await.expect("import");

        assert_eq!(
            manager
                .session_info(new_id)
                .await
                .expect("info")
                .expect("row")
                .additional_roots,
            roots,
        );
    }

    /// An export written before `additional_roots` existed must still import. This is why the field
    /// is `#[serde(default)]` instead of a `format_version` bump, which
    /// `plan_import` would reject.
    #[test]
    fn test_plan_import_accepts_an_export_without_additional_roots() {
        let json = serde_json::json!({
            "format_version": SESSION_EXPORT_FORMAT_VERSION,
            "meka_version": "0.0.0",
            "exported_at": "2020-01-01T00:00:00Z",
            "root_session_id": "11111111-1111-4111-8111-111111111111",
            "sessions": [{
                "id": "11111111-1111-4111-8111-111111111111",
                "parent_id": null,
                "created_at": "2020-01-01T00:00:00Z",
                "updated_at": "2020-01-01T00:00:00Z",
                "cwd": null,
                "permission": null,
                "capabilities_json": null,
                "stats": crate::stats::SessionStatsSnapshot::default(),
                "events": [],
                "tool_outputs": {},
            }],
        });
        let export: SessionExport = serde_json::from_value(json).expect("deserialize");
        let (records, _) = plan_import(export, Some("work")).expect("plan");
        assert!(records[0].additional_roots.is_empty());
        assert_eq!(
            records[0].provider, "work",
            "an archive naming no profile adopts this installation's default"
        );
    }

    /// An archive naming no profile, imported where nothing can supply one, is refused.
    ///
    /// The alternative was writing the profile empty, which is the only way a session with no
    /// provider could enter the store other than through the ledger. That row cannot run, its
    /// refusal arrives whenever the user next resumes it, and its existence forced every reader to
    /// know about a state nothing else produces. Refusing keeps the invariant every other door
    /// already holds to: a session that exists names a profile that resolved when it was written.
    #[test]
    fn an_archive_with_no_profile_is_refused_when_nothing_can_supply_one() {
        let json = serde_json::json!({
            "format_version": SESSION_EXPORT_FORMAT_VERSION,
            "meka_version": "0.0.0",
            "exported_at": "2020-01-01T00:00:00Z",
            "root_session_id": "11111111-1111-4111-8111-111111111111",
            "sessions": [{
                "id": "11111111-1111-4111-8111-111111111111",
                "parent_id": null,
                "created_at": "2020-01-01T00:00:00Z",
                "updated_at": "2020-01-01T00:00:00Z",
                "cwd": null,
                "permission": null,
                "capabilities_json": null,
                "stats": crate::stats::SessionStatsSnapshot::default(),
                "events": [],
                "tool_outputs": {},
            }],
        });
        let export: SessionExport = serde_json::from_value(json).expect("deserialize");
        let Err(error) = plan_import(export, None) else {
            panic!("no default and no recorded profile must refuse the import");
        };
        assert!(
            error.to_string().contains("--provider"),
            "the refusal must name what supplies one: {error}"
        );
    }

    /// An archive cannot choose where a session's turns are sent.
    ///
    /// Until 0.44 a session could pin its own endpoint, and an archive carried that pin. The
    /// credential comes from whichever configured profile the row names, so honouring an
    /// archive-supplied endpoint posted that profile's stored key wherever the archive said, and
    /// `POST /v1/sessions/import` takes its archive from a request body behind nothing but a
    /// `sessions:w` token. It cost a trusted/untrusted split across the two import doors.
    ///
    /// A session now records a profile and nothing else, so the vector is closed by construction
    /// rather than by a refusal that has to be remembered. This pins that: an archive still
    /// *naming* the retired key imports cleanly and takes the endpoint from its profile, and no
    /// import door needs to know which caller it is serving.
    #[test]
    fn an_archive_cannot_pin_a_sessions_endpoint() {
        let archive = |base_url: serde_json::Value| {
            serde_json::json!({
                "format_version": SESSION_EXPORT_FORMAT_VERSION,
                "meka_version": "0.0.0",
                "exported_at": "2020-01-01T00:00:00Z",
                "root_session_id": "11111111-1111-4111-8111-111111111111",
                "sessions": [{
                    "id": "11111111-1111-4111-8111-111111111111",
                    "parent_id": null,
                    "created_at": "2020-01-01T00:00:00Z",
                    "updated_at": "2020-01-01T00:00:00Z",
                    "cwd": null,
                    "permission": null,
                    "capabilities_json": null,
                    "provider": "work",
                    "base_url_override": base_url,
                    "stats": crate::stats::SessionStatsSnapshot::default(),
                    "events": [],
                    "tool_outputs": {},
                }],
            })
        };

        // An archive written by a meka that still had the field. The key is not modelled, so serde
        // ignores it rather than refusing the archive: an import that failed here would strand a
        // backup the user took a fortnight ago.
        let hostile: SessionExport =
            serde_json::from_value(archive(serde_json::json!("https://elsewhere.invalid/v1")))
                .expect("an archive naming the retired key still deserializes");
        let (records, _) = plan_import(hostile, None).expect("plan");
        assert_eq!(
            records[0].provider, "work",
            "the session runs on its profile, and the endpoint comes from that profile alone"
        );

        let plain: SessionExport =
            serde_json::from_value(archive(serde_json::Value::Null)).expect("deserialize");
        let (records, _) = plan_import(plain, None).expect("plan");
        assert_eq!(records[0].provider, "work");
    }

    /// Regression: import restored the export's `updated_at`, and retention GC deletes by that
    /// column when `[session].retention_days` is set, so restoring an archive older than that was
    /// undone by the next launch before anyone could resume it.
    #[tokio::test]
    async fn test_import_survives_retention_gc() {
        use std::path::Path;

        let manager = SessionManager::open(Some(Path::new(":memory:")), &Default::default())
            .await
            .expect("open");
        let stale = (chrono::Utc::now() - chrono::TimeDelta::days(100)).to_rfc3339();
        let records = vec![crate::session::ImportSessionRecord {
            new_id: uuid::Uuid::new_v4(),
            new_parent_id: None,
            created_at: stale.clone(),
            cwd: None,
            permission: None,
            capabilities_json: None,
            additional_roots: Vec::new(),
            subagent_spec_json: None,
            provider: "test-profile".to_string(),
            stats: crate::stats::SessionStatsSnapshot::default(),
            events: Vec::new(),
            tool_outputs: Vec::new(),
        }];
        let imported_id = records[0].new_id;
        manager.import_sessions(records).await.expect("import");

        assert_eq!(
            manager
                .delete_expired_sessions(90)
                .await
                .expect("retention sweep")
                .deleted,
            0,
            "a freshly imported archive must not be swept on the next launch"
        );
        assert!(manager.session_exists(imported_id).await.expect("exists"));

        // `created_at` still carries the original for provenance.
        assert_eq!(
            manager
                .session_info(imported_id)
                .await
                .expect("info")
                .expect("row")
                .created_at,
            stale,
        );
    }

    #[test]
    fn full_export_includes_pre_compaction_turns() {
        // A compacted session: the early turns are hidden from the model behind a CompactBoundary,
        // but `meka session export` must still render them. Build the same event log compaction
        // produces and assert the export contains both the summarized turns and a boundary marker.
        let mut log = conversation::Conversation::new();
        log.append(user_msg("first question"));
        log.append(assistant_text("first answer"));
        log.append(user_msg("second question"));
        log.append(assistant_text("second answer"));
        log.replace_for_compaction(
            user_msg("[Conversation summary from session compaction]\n\nYou discussed things."),
            vec![assistant_text("kept tail answer")],
            std::collections::HashSet::new(),
        );

        let markdown = format_session_as_markdown(
            uuid::Uuid::nil(),
            log.events(),
            &std::collections::HashMap::new(),
        );

        // Pre-compaction turns survive in the export even though the model no longer sees them.
        assert!(
            markdown.contains("first question") && markdown.contains("second answer"),
            "full export must include pre-compaction turns:\n{markdown}"
        );
        // The boundary is marked, and its summary is available (collapsed).
        assert!(
            markdown.contains("Session compaction") && markdown.contains("You discussed things."),
            "full export must mark the compaction boundary:\n{markdown}"
        );
        // The retained tail (re-appended after the boundary) is present.
        assert!(
            markdown.contains("kept tail answer"),
            "full export must include the retained tail:\n{markdown}"
        );
    }
}
