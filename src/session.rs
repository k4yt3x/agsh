//! SQLite-backed session store. The tables this module owns are `sessions` and `messages` for the
//! conversation, `tool_outputs` for results too large to keep inline (referenced from the
//! conversation by handle), `provider_credentials` and `mcp_credentials` for secrets, and
//! `scheduled_jobs` and `background_tasks` for work the agent starts and does not wait for. They
//! are not the whole database, and this module does not define them: they and the memory store's
//! tables are created by [`migrations`], the single ledger that also brings an older store forward.
//! `initialize_schema` runs it, then hands the memory search index to
//! `crate::memory::store::reconcile_index`.
//!
//! One table is outside that: `prompt_history`, created by `crate::history` on the separate
//! connection the REPL opens for input history. It carries no schema the agent reads and is not
//! versioned with the rest.
//!
//! Per-session mutual exclusion is provided by an OS-level file lock ([`FileLock`]) so the
//! kernel reclaims it whenever the holder dies: no PID-aliveness check, no risk of stale locks.
//!
//! On Unix the data directory (`0700`), lock directory (`0700`), and the database file itself
//! (`0600`) are tightened after creation so the persisted OAuth tokens, MCP credentials, and
//! conversation content aren't readable by other local users regardless of the user's umask.

pub mod cli;
pub mod migrations;

use std::{
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
    sync::Arc,
};

use fd_lock::{RwLock as FdRwLock, RwLockWriteGuard as FdRwLockWriteGuard};
use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};
use tokio_rusqlite::Connection;
use uuid::Uuid;

use crate::{
    error::{MekaError, Result},
    provider::AuthCredential,
};

/// Raw row from the `messages` table, the on-disk shape of a single
/// [`crate::conversation::Event`]. Internal to the session module: only the encoder and decoder
/// helpers handle these directly. External consumers go through [`SessionManager::save_event`] /
/// [`SessionManager::load_events`].
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredMessage {
    role: String,
    content: String,
    created_at: String,
}

/// Result of [`SessionManager::create_session_with_metadata`]. Carries the canonical RFC 3339
/// `created_at` so the caller's in-memory state shares one timestamp with the DB row; without
/// this, the handler's `SessionEntry.created_at` and the DB `sessions.created_at` would each
/// capture `Utc::now()` independently and drift by a few ms. Re-attach reads the DB value,
/// so the in-memory value has to match for round-trip tests to be deterministic.
#[derive(Debug, Clone)]
pub struct CreatedSession {
    pub id: Uuid,
    /// RFC 3339 timestamp written to both `sessions.created_at` and `sessions.updated_at`.
    pub created_at: String,
}

/// Metadata for one session row, used by JSON session export to reconstruct a session and its
/// sub-agent tree. Omits the derived `preview` and the `token_id` fingerprint (which is tied to the
/// exporting deployment and must not travel).
#[derive(Debug, Clone)]
pub struct SessionMetaRow {
    pub id: Uuid,
    pub parent_id: Option<Uuid>,
    pub created_at: String,
    pub updated_at: String,
    pub cwd: Option<String>,
    pub permission: Option<String>,
    pub capabilities_json: Option<String>,
    /// Workspace roots beyond `cwd`. Empty for every non-ACP session, all of which are
    /// single-root.
    pub additional_roots: Vec<PathBuf>,
    /// The terms a sub-agent was spawned under, as stored JSON. `None` on every top-level session.
    pub subagent_spec_json: Option<String>,
    /// What the session runs on, so an export carries it and an import can restore it rather than
    /// landing every imported session on the empty profile no configuration can name.
    pub provider: String,
}

/// Per-surface overrides applied to the copy produced by [`SessionManager::fork_session`]. Each
/// `None` field inherits the source session's value.
///
/// `cwd` and `additional_roots` exist because ACP models `session/fork` as a session-*creation*
/// request: it carries its own workspace, which may legitimately differ from the source's.
/// `token_id` is never inherited (it fingerprints the bearer token that created a session, so the
/// forking caller's token is the only correct value); `None` simply leaves it NULL.
#[derive(Debug, Default, Clone)]
pub struct ForkOverrides {
    pub cwd: Option<std::path::PathBuf>,
    pub additional_roots: Option<Vec<PathBuf>>,
    pub token_id: Option<String>,
}

/// One session's worth of data for [`SessionManager::import_sessions`]. IDs are already freshly
/// minted and parent links remapped by the caller; the records must be ordered parents-first so
/// the `parent_session_id` foreign key is satisfied on insert.
pub struct ImportSessionRecord {
    pub new_id: Uuid,
    pub new_parent_id: Option<Uuid>,
    pub created_at: String,
    pub cwd: Option<String>,
    pub permission: Option<String>,
    pub capabilities_json: Option<String>,
    /// Workspace roots beyond `cwd`, carried across an export/import round trip. Defaults to empty
    /// for exports written before the field existed.
    pub additional_roots: Vec<PathBuf>,
    /// A sub-agent's spawn terms, carried so an imported worker is still followable. `None` for
    /// top-level sessions and for archives written before the field existed; an imported sub-agent
    /// without it can be read and deleted but not resumed.
    pub subagent_spec_json: Option<String>,
    /// What the imported session runs on. The caller settles this: an archive that carries a
    /// profile keeps it, and one written before the field existed adopts the importing
    /// installation's default, which is the only thing that can be known about it here.
    pub provider: String,
    pub stats: crate::stats::SessionStatsSnapshot,
    /// `(created_at, event)` pairs in chronological order; timestamps are preserved verbatim.
    pub events: Vec<(String, crate::conversation::Event)>,
    /// `(name, content)` scratchpad entries referenced by name from tool-call inputs.
    pub tool_outputs: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub id: Uuid,
    /// RFC 3339 timestamp the session row was first written. Surfaced alongside `updated_at`
    /// so re-attach can restore the original creation time rather than stamping a fresh
    /// `Utc::now()` on every reconstruction.
    pub created_at: String,
    pub updated_at: String,
    pub preview: String,
    /// Working directory captured at session creation. `None` when an archive omitted one and
    /// [`SessionManager::import_sessions`] stored that absence verbatim; ACP-facing code falls
    /// back to the process cwd for display.
    pub cwd: Option<std::path::PathBuf>,
    /// Permission level captured at session creation. NULL for REPL, ACP and sub-agent rows -- the
    /// first two derive permission from process config, and a sub-agent's level travels in
    /// `subagent_spec_json` instead -- and for an imported row whose archive omitted one. The HTTP
    /// API persists this so `POST /v1/sessions` with an explicit `permission` field survives
    /// GC-eviction + re-attach.
    pub permission: Option<String>,
    /// The name of the provider profile this session runs on, and nothing else: a profile is an
    /// indivisible bundle, so the name is the whole binding.
    pub provider: String,
    /// Per-session capability flags, as a serialized
    /// [`crate::server::http_frontend::SessionCapabilities`]. Deliberately not enumerated here:
    /// the flag set has grown twice, and each restatement went stale silently. NULL on the
    /// same sessions as `permission`, and for the same reason.
    pub capabilities_json: Option<String>,
    /// Workspace roots beyond `cwd`, from an ACP client's `additionalDirectories`. Empty whenever
    /// a session carries no extra roots, which is every non-ACP session.
    pub additional_roots: Vec<PathBuf>,
    /// SHA-256 fingerprint of the bearer token that created this session. `None` for every session
    /// not created via the HTTP API, including sub-agents, whose row omits the column entirely. A
    /// fork through the HTTP API does carry one: the token doing the forking, never the source's.
    pub token_id: Option<String>,
    /// The session this one was spawned from, for a sub-agent. `None` for a top-level session.
    /// Surfaced so a client listing with `include_children` can rebuild the spawn tree rather than
    /// receiving a flat list in which a worker is indistinguishable from the agent that dispatched
    /// it.
    pub parent_id: Option<Uuid>,
}

#[derive(Debug, Clone)]
pub struct ToolOutputSummary {
    pub name: String,
    pub size: usize,
    pub created_at: String,
}

/// Result of [`SessionManager::rename_tool_output`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenameOutcome {
    Renamed,
    NotFound,
    TargetExists,
}

#[derive(Clone)]
pub struct SessionManager {
    connection: Arc<Connection>,
    lock_dir: PathBuf,
    /// Resolved path to the on-disk database (or `:memory:`). Exposed via [`Self::database_path`]
    /// so the REPL can open a second connection for persistent input history.
    database_path: PathBuf,
    /// Set only for an in-memory database, whose lock dir is a fresh temp directory nothing else
    /// would ever clean up. Held behind an `Arc` so the removal happens when the *last* clone of
    /// this manager drops, not the first: `SessionManager` is cloned into sub-agents and tool
    /// builders, and any of those may still be locking sessions.
    _ephemeral_lock_dir: Option<Arc<EphemeralLockDir>>,
}

/// Removes an in-memory database's temp lock directory on drop.
///
/// On-disk databases keep a `locks/` directory beside the database file, which must outlive the
/// process; in-memory ones get a per-`open()` temp directory instead, so that concurrent tests
/// can't sweep each other's lock files through [`SessionManager::prune_orphan_lock_files`]. That
/// isolation is worth keeping, but without this guard each `open()` left an empty directory in
/// the system temp dir forever.
struct EphemeralLockDir(PathBuf);

impl Drop for EphemeralLockDir {
    fn drop(&mut self) {
        // Any `.lock` files still inside belong to this manager's own sessions; a `FileLock`
        // that outlives the manager keeps working, because unlinking an open file on Unix doesn't
        // invalidate the descriptor the `flock` is held on.
        if let Err(error) = std::fs::remove_dir_all(&self.0) {
            tracing::debug!(
                "failed to remove ephemeral lock dir '{}': {}",
                self.0.display(),
                error,
            );
        }
    }
}

/// RAII handle for an exclusive OS file lock in the store's lock directory. Holding this value
/// keeps the underlying descriptor open; dropping it (including when the process exits or panics)
/// closes the FD, which causes the kernel to release the `flock`/`LockFileEx` lock automatically.
/// There is no "stale lock" failure mode; even `SIGKILL` is safe.
///
/// Two things are locked this way: a session, so only one meka is ever attached to a conversation,
/// and a provider profile's credential, so only one meka at a time is rotating its refresh token.
///
/// Internally this is a self-referential struct: `guard` borrows from `*lock` (a `Box` for stable
/// heap address). The explicit [`Drop`] impl drops `guard` before `lock` regardless of field
/// declaration order, the safety invariant of the lifetime transmute used during construction.
pub struct FileLock {
    guard: std::mem::ManuallyDrop<FdRwLockWriteGuard<'static, File>>,
    lock: std::mem::ManuallyDrop<Box<FdRwLock<File>>>,
}

impl Drop for FileLock {
    fn drop(&mut self) {
        // SAFETY: `guard` borrows from `*lock`; drop it first so the borrow never outlives the
        // borrowee. This ordering is explicit here and does not depend on the field declaration
        // order above. Neither field is touched again after this.
        unsafe {
            std::mem::ManuallyDrop::drop(&mut self.guard);
            std::mem::ManuallyDrop::drop(&mut self.lock);
        }
    }
}

/// Where a session's lock lives while a host and its agent both need to reach it.
///
/// The lock has to be taken the instant the row exists, and for a session the agent creates that
/// instant is inside `Agent::run_turn_retaining` -- seconds or minutes before the host gets control
/// back. Claiming it afterwards, which is what the REPL used to do, left a fresh session unlocked
/// for the whole of its first turn, and a second `meka` invocation attached to it and interleaved
/// its own messages into the same conversation.
///
/// The host still owns the lifetime: the lock outlives any one turn, `/fork` replaces it, and it is
/// dropped last on the way out so the session stays held until the process really ends. Sharing a
/// slot rather than moving the lock either way is what lets both of those be true at once.
///
/// `std::sync::Mutex` rather than tokio's: every access is a move in or out with nothing awaited in
/// between, and a blocking lock keeps the drop order at process exit obvious.
pub type SessionLockSlot = std::sync::Arc<std::sync::Mutex<Option<FileLock>>>;

/// What became of a compare-and-swap on a provider's stored credential.
///
/// No equality derive: comparing credentials is not something callers should be doing, and
/// [`AuthCredential`]'s own `Debug` redacts, so this stays printable without leaking a token.
#[derive(Debug, Clone)]
pub enum CredentialWrite {
    /// The row still held what this write was derived from, and now holds the new value.
    Stored,
    /// Something else wrote first. This is what the row holds now.
    ///
    /// Newer in *write order*, which is the only thing this type knows and less than it sounds: it
    /// is neither necessarily unexpired nor necessarily the same kind of credential. Whether it is
    /// worth switching to is the caller's question -- see `provider::is_worth_adopting` -- and
    /// retrying is never the answer, because the value this write was derived from is gone.
    Superseded(Box<AuthCredential>),
    /// The profile has no stored credential at all -- removed while this write was in flight.
    /// Re-creating it would resurrect an account the user just disconnected.
    Gone,
}

/// What a sweep over many sessions did, so its caller can say what it left behind.
///
/// A bare count reads as "everything that matched was deleted", and that reading is what made the
/// retention sweep destructive: it announced `deleted 1 session(s)` in an unrelated terminal and
/// said nothing about the conversation an operator had open in another one. A sweep that spares
/// something has to be able to say so.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SessionSweep {
    pub deleted: u64,
    /// Sessions that matched but were left alone, because another meka process has them open or
    /// because their lock could not be established either way.
    pub attached_elsewhere: u64,
}

/// Open `path` (creating it) and take its exclusive `flock`.
///
/// `Ok(None)` means somebody else holds it; an `Err` means the question could not be asked at all
/// -- an unwritable lock directory, descriptors exhausted -- which is a different answer and
/// callers treat it differently.
///
/// A free function rather than a method because two stores need it: [`SessionManager`] locks
/// conversations, [`TokenStore`] locks provider profiles, and both live in the same directory.
fn try_lock_file(path: &std::path::Path) -> Result<Option<FileLock>> {
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path)
        .map_err(|error| {
            MekaError::Database(format!(
                "failed to open lock file '{}': {}",
                path.display(),
                error
            ))
        })?;

    let mut lock = Box::new(FdRwLock::new(file));
    let guard = match lock.try_write() {
        Ok(guard) => guard,
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => return Ok(None),
        Err(error) => {
            return Err(MekaError::Database(format!(
                "failed to acquire lock '{}': {}",
                path.display(),
                error
            )));
        }
    };

    // SAFETY: `guard` borrows from `*lock`. We move the box (not the RwLock inside it) into the
    // returned `FileLock`, so the RwLock's heap address is stable for as long as the box lives. The
    // explicit `Drop` impl on `FileLock` drops `guard` before `lock`, so the borrow never outlives
    // the borrowee.
    let guard: FdRwLockWriteGuard<'static, File> = unsafe { std::mem::transmute(guard) };

    Ok(Some(FileLock {
        guard: std::mem::ManuallyDrop::new(guard),
        lock: std::mem::ManuallyDrop::new(lock),
    }))
}

/// How long to keep trying to convert a rollback-journal database to WAL before giving up.
///
/// A deadline rather than an attempt count, because the two failure modes it spans cost wildly
/// different amounts of time. When SQLite skips the busy handler for this pragma an attempt returns
/// at once, and what is wanted is many of them across the contention window; when it consults the
/// handler an attempt blocks for the full `busy_timeout` first, and ten of those would turn a
/// five-second startup failure into a fifty-second one. Counting time bounds both.
///
/// Only a first run on a fresh install can need any of this: once the database is in WAL mode the
/// pragma takes no exclusive lock and cannot contend.
const WAL_CONVERSION_DEADLINE: std::time::Duration = std::time::Duration::from_secs(5);

/// Pause between those attempts. Blocking rather than async because the whole pragma batch runs on
/// the connection's own thread, where a sleep costs nothing else.
const WAL_CONVERSION_RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(50);

/// File stem of the process-wide schema lock, which lives beside the per-session ones.
///
/// Named rather than spelled inline at both sites, because the prune below has to exclude exactly
/// this file and a literal in two places is how that pairing gets broken.
const SCHEMA_LOCK_STEM: &str = "schema";

/// File-stem prefix of the per-provider-profile credential locks, which also live in the lock
/// directory.
///
/// Excluded from the prune for the same reason as [`SCHEMA_LOCK_STEM`]: a hashed profile name does
/// not parse as a UUID, and relying on that accident is how the next lock shaped like one inherits
/// the hazard.
const PROVIDER_LOCK_PREFIX: &str = "provider-";

/// Sessions a scheduled job still depends on, as a `WHERE` fragment over `sessions`.
///
/// A session with a job ahead of it is *not* expired, whatever `updated_at` says. Only turns bump
/// that column -- [`ScheduleStore::claim_occurrence`] and
/// [`ScheduleStore::complete_claim`] touch `scheduled_jobs` alone -- so a gated watcher
/// that evaluates every tick and rarely fires looks untouched for exactly as long as it is
/// working. The cascade then took the job with the session, and the sweep reported
/// `deleted 1 session(s)` without ever mentioning that a schedule went with it.
///
/// Sparing only the row that *owns* the job is not enough. `parent_session_id` carries
/// `ON DELETE CASCADE`, so deleting a stale parent silently takes its sub-agent children -- and a
/// job created against a child (reachable over HTTP, whose only gate is that the session exists)
/// goes with them. The recursive term walks parent links up from every job-owning session and
/// spares that whole chain.
///
/// A constant because it is applied twice, in two statements, and the pair is only sound while
/// they agree: see [`SessionManager::delete_the_unattached_among`].
///
/// [`ScheduleStore::claim_occurrence`]: crate::schedule::ScheduleStore::claim_occurrence
/// [`ScheduleStore::complete_claim`]: crate::schedule::ScheduleStore::complete_claim
const NOT_SPOKEN_FOR_BY_A_SCHEDULE: &str = "id NOT IN (SELECT session_id FROM scheduled_jobs) \
     AND id NOT IN ( \
         WITH RECURSIVE ancestors(id) AS ( \
             SELECT parent_session_id FROM sessions \
               WHERE parent_session_id IS NOT NULL \
                 AND id IN (SELECT session_id FROM scheduled_jobs) \
             UNION \
             SELECT s.parent_session_id FROM sessions s \
               JOIN ancestors a ON s.id = a.id \
              WHERE s.parent_session_id IS NOT NULL \
         ) \
         SELECT id FROM ancestors \
     )";

fn default_database_path() -> Result<PathBuf> {
    // `MEKA_DATA_DIR` is the cross-platform override, the only env var that works on every OS,
    // mirroring how `MEKA_CONFIG_DIR` overrides the config directory. The value points at the
    // `meka` data dir itself (the parent that contains `meka.db`). Useful for tests, portable
    // installs, and isolating per-project state from the global one.
    if let Ok(value) = std::env::var("MEKA_DATA_DIR")
        && !value.is_empty()
    {
        // Absolute only, matching `meka_config_dir`. A relative value means the database -- which
        // holds every provider credential -- lands under whatever directory meka happened to start
        // in, so `meka` in one project and `meka` in another silently get different credential
        // stores, and neither is the one the user set up. Ignored rather than fatal, for the same
        // reason the config directory ignores it: the platform default is always a usable answer.
        let path = PathBuf::from(value);
        if path.is_absolute() {
            return Ok(path.join("meka.db"));
        }
        tracing::warn!(
            "MEKA_DATA_DIR '{}' is not an absolute path; ignoring it and using the platform data \
             directory",
            path.display()
        );
    }

    // `dirs::data_dir()` honors XDG_DATA_HOME on Linux, returns `~/Library/Application Support` on
    // macOS, and `%APPDATA%` on Windows. No silent fallback: writing the session DB to a
    // wrong-for-the-platform path (e.g. the old Linux-only `~/.local/share` default) is worse than
    // asking the user to set `MEKA_DATA_DIR` explicitly.
    let base = dirs::data_dir().ok_or_else(|| {
        MekaError::Config(
            "could not determine a data directory for the database; \
             set MEKA_DATA_DIR to an absolute path"
                .into(),
        )
    })?;
    Ok(base.join("meka").join("meka.db"))
}

/// Create a directory (and any missing parents) born at mode 0700 on Unix. Avoids the umask window
/// that `create_dir_all` + later `set_permissions` would open: between `mkdir(2)` and `chmod(2)`,
/// the directory would be readable by other local users on a permissive umask.
/// `DirBuilderExt::mode` passes the mode straight to `mkdir`. Pre-existing directories keep their
/// mode; callers that need to tighten an already-existing dir should still follow up with
/// `restrict_permissions`.
#[cfg(unix)]
fn create_private_dir(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .mode(0o700)
        .recursive(true)
        .create(path)
}

#[cfg(not(unix))]
fn create_private_dir(path: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(path)
}

/// Restrict a path's permissions on Unix. Best-effort: if the call fails we log and continue,
/// because on some mounts (`/tmp` under specific overlay setups, NFS without proper support, etc.)
/// `chmod` returns `EPERM`/`EROFS` and refusing to open the session is a strictly worse failure
/// than leaving the file at the umask-derived mode.
#[cfg(unix)]
fn restrict_permissions(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    match std::fs::metadata(path) {
        Ok(metadata) => {
            let mut permissions = metadata.permissions();
            permissions.set_mode(mode);
            if let Err(error) = std::fs::set_permissions(path, permissions) {
                tracing::debug!(
                    "failed to restrict '{}' to mode {:o}: {}",
                    path.display(),
                    mode,
                    error
                );
            }
        }
        Err(error) => {
            tracing::debug!(
                "failed to stat '{}' while restricting permissions: {}",
                path.display(),
                error
            );
        }
    }
}

#[cfg(not(unix))]
fn restrict_permissions(_path: &Path, _mode: u32) {
    // Windows ACLs inherit from the parent directory; leave alone.
}

/// Copy the store aside before [`migrations::apply`] touches it, returning where it went.
///
/// `VACUUM INTO` rather than a file copy: it is atomic against concurrent writers, produces a
/// defragmented image rather than one that may span a live WAL, and carries `user_version` across,
/// so restoring the copy yields a store that identifies itself as the version it was and migrates
/// once when next opened. Lives here rather than in [`migrations`] because it is a question about
/// the store *file* -- a path, a mode -- and because that module is deliberately kept clear of
/// meka's own code.
///
/// Kept rather than cleaned up on success. It is the user's undo, and deleting it the moment the
/// migration works removes the safety net exactly when they might still want it.
///
/// Written to a staging name and renamed into place, so the name the docs tell people to restore
/// either does not exist or is a complete copy. `VACUUM INTO` does not unlink its output when a
/// write fails partway: measured under a write limit, it left a file with a zeroed page-1 header,
/// which SQLite then opens cleanly as an *empty database* that passes `integrity_check`. The retry
/// stepped past it to `.bak.1`, so the file wearing the documented name was the empty one. Staging
/// plus rename removes that whole class rather than special-casing it.
fn back_up_before_migrating(
    connection: &rusqlite::Connection,
    database_path: &Path,
    from: u32,
) -> Result<Option<PathBuf>> {
    // Nothing to preserve: an in-memory store is created empty by this very process.
    if database_path == Path::new(":memory:") {
        return Ok(None);
    }
    let target = free_backup_path(database_path, from)?;
    let staging = staging_path(&target);
    // `create_new` rather than `create`: it fails rather than following a symlink or truncating
    // something already there, and it is what makes the mode below a guarantee instead of a hope.
    // The mode matters because this file holds every credential and every conversation the store
    // does, and SQLite would otherwise create it at `0644 & ~umask` and leave it that way for as
    // long as the copy takes. `SessionManager::open` pre-touches the main database at `0600` for
    // exactly this reason; the backup gets the same treatment.
    let mut options = std::fs::OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(&staging).map_err(|error| {
        MekaError::Database(format!(
            "failed to create the pre-migration backup at '{}': {}. Nothing has been changed",
            staging.display(),
            error
        ))
    })?;

    // Bound rather than interpolated, so a data directory containing a quote is not a broken
    // statement. `VACUUM INTO` accepts a parameter here, and writes into the empty file just made.
    let Some(staging_text) = staging.to_str() else {
        remove_partial_backup(&staging);
        return Err(MekaError::Database(format!(
            "cannot write the pre-migration backup to '{}' because the path is not valid UTF-8. \
             Nothing has been changed. Move the store somewhere it is, or set MEKA_DATA_DIR",
            staging.display()
        )));
    };
    if let Err(error) = connection.execute("VACUUM INTO ?1", rusqlite::params![staging_text]) {
        remove_partial_backup(&staging);
        return Err(MekaError::Database(format!(
            "failed to back the store up to '{}' before migrating: {}. Nothing has been changed",
            target.display(),
            error
        )));
    }
    // Belt-and-braces on platforms where the mode above is a no-op, and against a umask that
    // somehow widened it.
    restrict_permissions(&staging, 0o600);
    std::fs::rename(&staging, &target).map_err(|error| {
        remove_partial_backup(&staging);
        MekaError::Database(format!(
            "failed to move the pre-migration backup into place at '{}': {}. Nothing has been \
             changed",
            target.display(),
            error
        ))
    })?;
    Ok(Some(target))
}

/// Remove the copies an earlier migration left, now that a fresher one is in place.
///
/// **Production reaches this only with a path [`back_up_before_migrating`] returned**, which is
/// what makes the ordering safe rather than merely intended. That function answers `Ok(Some(_))` on
/// one line, after the rename has put a complete copy at that name, so the caller's `?` is what
/// stops this running on a failure. Pruning before the copy landed would mean a failure between
/// choosing the name and renaming -- a path that is not valid UTF-8 is one, refused at
/// `staging.to_str()` -- leaves no backup at all.
///
/// One backup rather than one per release. The store holds every conversation and every memory, so
/// a copy per schema-changing upgrade is a full duplicate accumulating forever in a directory
/// nobody opens (`~/.local/share/meka`, `%APPDATA%\meka`), and conversations carry images as inline
/// base64, so "full" is not small.
///
/// **What that costs, stated rather than argued away.** The copy kept is of the store *after* the
/// migration before this one, so it cannot undo a conversion whose bug is found late: a step that
/// silently mis-converts, goes unnoticed, and is followed by another schema-changing upgrade takes
/// the only pre-conversion copy with it. That is a real loss and not the same thing as the copy
/// being stale, which is what an earlier draft of this comment claimed. It is accepted because the
/// alternative was an unbounded pile whose cost is certain, while this one is conditional on a bug
/// that outlives a release; `upgrading.md` says the same to users rather than implying the older
/// copy was never worth anything.
///
/// Only names this module could have produced are touched; see [`is_backup_name`], which matches
/// against what [`free_backup_path`] actually emits rather than against "some digits". A
/// `meka.db.mine.bak` someone parked beside the store is not meka's to delete.
///
/// Compared through [`std::ffi::OsStr::as_encoded_bytes`] rather than `to_string_lossy`, because
/// lossy conversion maps distinct names onto one string, which here would mean deleting a file that
/// merely resembles a backup. Every byte tested is ASCII, and ASCII cannot occur inside a
/// multi-byte sequence, so splitting on it is sound on both platforms.
///
/// Every failure is a `warn!` and nothing more, so that a filesystem that will not let a file go
/// cannot turn a migration that worked into a refusal to start. Reaching one takes something
/// narrower than a read-only directory -- that already stopped the copy itself, upstream of here --
/// such as an immutable file, a sticky-bit directory owned by someone else, or a Windows sharing
/// violation.
fn prune_older_backups(database_path: &Path, keep: &Path) {
    let (Some(directory), Some(store_name), Some(keep_name)) = (
        database_path.parent(),
        database_path.file_name(),
        keep.file_name(),
    ) else {
        return;
    };
    let entries = match std::fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) => {
            tracing::warn!(
                "could not list '{}' to remove older pre-migration backups: {}",
                directory.display(),
                error
            );
            return;
        }
    };
    for entry in entries {
        // Not `.flatten()`. An entry that cannot be read is a question this cannot answer, and
        // skipping it is right -- but silently is not, because the answer it stands in for is
        // "there may be a superseded copy still on disk".
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                tracing::warn!(
                    "could not read an entry of '{}' while removing older pre-migration backups: \
                     {}",
                    directory.display(),
                    error
                );
                continue;
            }
        };
        let name = entry.file_name();
        if name == keep_name || !is_backup_name(store_name, &name) {
            continue;
        }
        let path = entry.path();
        match std::fs::remove_file(&path) {
            Ok(()) => tracing::info!(
                "removed the superseded pre-migration backup '{}'",
                path.display()
            ),
            Err(error) => tracing::warn!(
                "could not remove the superseded pre-migration backup '{}': {}",
                path.display(),
                error
            ),
        }
    }
}

/// Whether `candidate` is a name [`free_backup_path`] could have produced for `store_name`.
///
/// `<store>.v<version>.bak`, optionally then `.<suffix>`. Both numbers are matched against what
/// that function can actually emit rather than against "some digits", because the two differ and
/// the difference is deletions: a `version` it renders through `Display` from a `u32` it is only
/// ever called with above zero, and a `suffix` from `1..1000`. So `.v0.`, `.v01.`, `.bak.0` and
/// `.bak.1000` are all shapes meka cannot write, and a `meka.db.v1.bak.0` is somebody's own
/// zero-indexed archive rather than a copy this module left.
///
/// **One shape, tracking the writer.** If [`free_backup_path`] ever names copies differently, this
/// follows it and matches only the new name. Keeping the old one alongside would be a reader
/// tolerating what an older meka wrote, which is the arrangement `migrations` exists to make
/// unnecessary, and there is no ledger for files sitting beside the store to convert them with. The
/// honest price of such a rename is that copies already on disk stop being recognised and are left
/// where they are, which errs toward keeping a file rather than deleting one.
///
/// A `.partial` is deliberately not a match. It is litter as well, but [`free_backup_path`] reads
/// its presence to step past a name it would otherwise reuse, and removing completed copies is the
/// whole of what this is for.
fn is_backup_name(store_name: &std::ffi::OsStr, candidate: &std::ffi::OsStr) -> bool {
    let store = store_name.as_encoded_bytes();
    let candidate = candidate.as_encoded_bytes();
    let Some(rest) = candidate.strip_prefix(store) else {
        return false;
    };
    let Some(rest) = rest.strip_prefix(b".v") else {
        return false;
    };
    let (version, rest) = split_digits(rest);
    // `from > 0` is guarded at the one call site, and `Display` never pads, so a leading zero and a
    // bare `0` are both names this module cannot have written.
    if !matches!(parse_number(version), Some(1..=u32::MAX)) {
        return false;
    }
    let Some(rest) = rest.strip_prefix(b".bak") else {
        return false;
    };
    if rest.is_empty() {
        return true;
    }
    let Some(rest) = rest.strip_prefix(b".") else {
        return false;
    };
    let (suffix, tail) = split_digits(rest);
    tail.is_empty() && matches!(parse_number(suffix), Some(1..MAX_BACKUP_SUFFIX))
}

/// The exclusive end of [`free_backup_path`]'s suffix range, named so the loop there and the match
/// in [`is_backup_name`] cannot drift apart.
const MAX_BACKUP_SUFFIX: u32 = 1000;

/// Parse a run of ASCII digits exactly as `free_backup_path` rendered one, or `None`.
///
/// Rejects what `Display` cannot produce (nothing at all, a leading zero) and what a `u32` cannot
/// hold, so an absurdly long run of digits is a non-match rather than a wrap or a panic.
fn parse_number(digits: &[u8]) -> Option<u32> {
    if digits.is_empty() || (digits.len() > 1 && digits[0] == b'0') {
        return None;
    }
    std::str::from_utf8(digits).ok()?.parse().ok()
}

/// Split a leading run of ASCII digits from the rest, for [`is_backup_name`]. Either half may be
/// empty; the caller decides which of those are acceptable.
fn split_digits(bytes: &[u8]) -> (&[u8], &[u8]) {
    let end = bytes
        .iter()
        .position(|byte| !byte.is_ascii_digit())
        .unwrap_or(bytes.len());
    bytes.split_at(end)
}

/// Clear away a backup that never finished. Best-effort by design: the caller is already returning
/// an error that stops the migration, and failing to tidy up must not replace that error with a
/// less useful one. Leaving it behind is safe either way, because only a completed copy is ever
/// renamed onto the name the docs name.
fn remove_partial_backup(staging: &Path) {
    if let Err(error) = std::fs::remove_file(staging) {
        tracing::debug!(
            "failed to remove the incomplete backup '{}': {}",
            staging.display(),
            error
        );
    }
}

/// `<store>.v<from>.bak`, or the first numbered variant whose name *and staging sibling* are both
/// unused.
///
/// Built through `OsString` rather than `format!` on a `String`, so a data directory whose name is
/// not valid UTF-8 still produces a path that names the file the user actually has.
///
/// Never reuses a name that exists. An earlier backup is the record of an earlier attempt, and a
/// migration that replaced it would destroy the copy taken before whatever failure made a second
/// attempt necessary.
///
/// **Both halves of the pair have to be free, and that is the whole of a bug worth remembering.**
/// The copy is staged at `<name>.partial` and renamed into place, so an abnormal exit between the
/// two leaves a `.partial` behind. Checking only the target then chose that same name again, and
/// `create_new` failed `EEXIST` on every subsequent start: measured, one `kill -9` during the
/// upgrade of a 90 MB store, then three consecutive runs all refusing with
/// `failed to create the pre-migration backup … File exists` and the store still at its old
/// version. A single interrupted upgrade wedged meka permanently, which is the opposite of what
/// staging was introduced to achieve.
///
/// Exhaustion is an error rather than a fallback to the unsuffixed name. It used to be safe to
/// return the occupied base, because `VACUUM INTO` refuses a non-empty target; staging moved the
/// write to `.partial` and the final step to `std::fs::rename`, which does not refuse anything. So
/// the old fallback silently overwrote the *oldest* backup, the one most likely to matter.
fn free_backup_path(database_path: &Path, from: u32) -> Result<PathBuf> {
    let base = {
        let mut name = database_path.as_os_str().to_os_string();
        name.push(format!(".v{}.bak", from));
        PathBuf::from(name)
    };
    if is_free_pair(&base) {
        return Ok(base);
    }
    for suffix in 1..MAX_BACKUP_SUFFIX {
        let mut name = base.as_os_str().to_os_string();
        name.push(format!(".{}", suffix));
        let candidate = PathBuf::from(name);
        if is_free_pair(&candidate) {
            return Ok(candidate);
        }
    }
    Err(MekaError::Database(format!(
        "cannot name a pre-migration backup: '{}' and its {} numbered variants are all taken. \
         Nothing has been changed. Move or delete the old copies beside the store",
        base.display(),
        MAX_BACKUP_SUFFIX - 1
    )))
}

/// Whether both the backup name and the staging name it implies are unused.
fn is_free_pair(target: &Path) -> bool {
    let staging = staging_path(target);
    is_free(target) && is_free(&staging)
}

/// Where a copy is written before it is renamed onto `target`.
///
/// One function so the caller and [`is_free_pair`] cannot disagree about the name, which is exactly
/// how the wedge above happened: the check looked at one path and the create at another.
fn staging_path(target: &Path) -> PathBuf {
    let mut name = target.as_os_str().to_os_string();
    name.push(".partial");
    PathBuf::from(name)
}

/// Whether nothing at all sits at this path, symlinks included.
///
/// `Path::exists` is the wrong question here: it follows symlinks, so it answers `false` for a
/// *dangling* one and this would hand back a name that is really a redirection.
/// `symlink_metadata` asks about the link rather than through it.
///
/// The hazard that prompted this is no longer reachable, and saying so is the honest version:
/// measured on the pre-staging code, `VACUUM INTO` followed such a link and wrote the whole
/// credential store outside the data directory at `0644`. Staging closed that by construction,
/// because `VACUUM INTO` now only ever touches `.partial` and `rename` replaces a symlink rather
/// than following it. This stays because a name occupied by a link is still occupied, and
/// handing it back would mean the log naming a file that is not the one written.
fn is_free(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_err()
}

impl SessionManager {
    /// Open the store, bringing its schema forward if it is behind.
    ///
    /// `context` carries the facts a migration cannot work out for itself; see
    /// [`migrations::Context`]. It is a parameter rather than something read here because this
    /// function must not know what a provider profile is: the ledger is the only place allowed to
    /// act on an older meka's store, and config is the only place that knows which profile is the
    /// default. A caller with nothing to carry forward passes the default, which every test does
    /// because a store it just created has no sessions to carry.
    pub async fn open(path: Option<&Path>, context: &migrations::Context) -> Result<Self> {
        let database_path = match path {
            Some(path) => path.to_path_buf(),
            None => default_database_path()?,
        };

        // In-memory SQLite databases (used by tests) have no on-disk parent; give each `open()`
        // call its own ephemeral lock dir under the system temp directory so concurrent tests don't
        // share lock files.
        let is_in_memory = database_path == Path::new(":memory:");
        let lock_dir = if is_in_memory {
            std::env::temp_dir().join(format!("meka-test-locks-{}", Uuid::new_v4()))
        } else {
            if let Some(parent) = database_path.parent() {
                create_private_dir(parent)?;
                // Pre-existing parents inherit their old mode; tighten if so.
                restrict_permissions(parent, 0o700);
            }
            database_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("locks")
        };
        create_private_dir(&lock_dir)?;
        restrict_permissions(&lock_dir, 0o700);

        // Pre-touch the DB file at 0600 so SQLite's `Connection::open` reuses an already-restricted
        // file rather than creating one at umask defaults that we then chmod down; the latter
        // leaves a window where another local user could open the file. `-wal`/`-shm` companions
        // still inherit the umask, but the parent directory's 0700 mode keeps them inaccessible to
        // other users.
        #[cfg(unix)]
        if !is_in_memory {
            use std::os::unix::fs::OpenOptionsExt;
            let _ = std::fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .write(true)
                .mode(0o600)
                .open(&database_path)
                .map_err(|error| {
                    MekaError::Database(format!(
                        "failed to pre-touch database '{}': {}",
                        database_path.display(),
                        error
                    ))
                })?;
        }

        let connection = Connection::open(&database_path)
            .await
            .map_err(|error| MekaError::Database(format!("failed to open database: {}", error)))?;

        // Belt-and-braces: if the file pre-existed at a more permissive mode (manual setup,
        // restored backup, etc.), tighten it now. The pre-touch above is the primary protection for
        // newly-created files.
        if !is_in_memory {
            restrict_permissions(&database_path, 0o600);
        }

        // SQLite defaults foreign-key enforcement to OFF per-connection; the `FOREIGN KEY` clauses
        // in `CREATE TABLE` are decorative without this. Set before `initialize_schema` so every
        // statement it runs, and every one after, sees enforcement active. Must run outside any
        // transaction to take effect.
        connection
            .call(|connection| -> rusqlite::Result<_> {
                // Restated rather than established: `rusqlite::Connection::open` already installs a
                // five-second busy timeout before any of this runs, so this pragma pins the value
                // meka wants against a future change in that default rather than supplying one.
                // Ordered before the WAL conversion below because that is where it would matter if
                // it were ever the only source.
                connection.execute_batch(
                    "PRAGMA busy_timeout = 5000;\n\
                     PRAGMA foreign_keys = ON;",
                )?;
                // The retry is the part that fixes something. Converting a rollback-journal
                // database to WAL takes an exclusive lock, and SQLite does not
                // always route *that* pragma's acquisition through the busy handler
                // -- so with a handler installed and waiting, the conversion still
                // returned `database is locked` outright. Measured at 2 to 9
                // failures per 200-1200 launches of several meka processes starting together, each
                // one a process exiting with `failed to set connection pragmas: database is
                // locked`. An already-WAL database takes no exclusive lock here and never contends,
                // so this only ever bit a first run on a fresh install -- a systemd unit and a
                // shell coming up together, which is the ordinary case.
                //
                // WAL is what lets the REPL's history connection read without blocking the agent's
                // writes, so a database left in rollback mode is a live contention problem rather
                // than a cosmetic one: worth several attempts before giving up. (On `:memory:` the
                // request is silently ignored and the first attempt always succeeds.)
                let giving_up_at = std::time::Instant::now() + WAL_CONVERSION_DEADLINE;
                loop {
                    match connection.execute_batch("PRAGMA journal_mode = WAL;") {
                        Ok(()) => break,
                        Err(error) if std::time::Instant::now() >= giving_up_at => {
                            return Err(error);
                        }
                        Err(_) => std::thread::sleep(WAL_CONVERSION_RETRY_DELAY),
                    }
                }
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to set connection pragmas: {}", error))
            })?;

        let manager = Self {
            connection: Arc::new(connection),
            _ephemeral_lock_dir: is_in_memory.then(|| Arc::new(EphemeralLockDir(lock_dir.clone()))),
            lock_dir,
            database_path,
        };
        manager.initialize_schema(context).await?;
        manager.prune_orphan_lock_files().await;
        Ok(manager)
    }

    /// Resolved path to the on-disk database (or `:memory:`). The REPL opens a second synchronous
    /// connection here for persistent input history (see [`crate::history::PromptHistory`]).
    pub fn database_path(&self) -> &Path {
        &self.database_path
    }

    async fn initialize_schema(&self, context: &migrations::Context) -> Result<()> {
        // Serialise schema work across processes.
        //
        // Two things below need it, and neither is safe on its own. The migration run decides what
        // to do by reading the store and then acts on that answer, so two processes that both read
        // "needs migrating" would both try. And `memory::store::reconcile_index` makes the
        // `sqlite_master` read that decides whether the FTS triggers have drifted *outside* the
        // transaction that replaces them; the replacement itself is one immediate transaction, so
        // no process ever sees a half-applied trigger set, but a second process can see a snapshot
        // the winner is about to invalidate and then act on it after it has stopped being true. A
        // systemd unit and a shell REPL starting together is exactly when that happens.
        //
        // An OS file lock rather than a SQLite transaction, the same primitive `FileLock` uses,
        // held for the whole of the schema work so the check and the write it authorises cannot be
        // split. The loser waits, then re-runs against the winner's finished schema and no-ops.
        let lock_path = self.lock_dir.join(format!("{}.lock", SCHEMA_LOCK_STEM));
        let mut lock_file = FdRwLock::new(
            std::fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .write(true)
                .open(&lock_path)
                .map_err(|error| {
                    MekaError::Database(format!(
                        "failed to open schema lock '{}': {}",
                        lock_path.display(),
                        error
                    ))
                })?,
        );
        let _schema_guard = lock_file.write().map_err(|error| {
            MekaError::Database(format!(
                "failed to acquire schema lock '{}': {}",
                lock_path.display(),
                error
            ))
        })?;

        // Migrations, not declaration. [`migrations::plan`] decides what this store needs and
        // [`migrations::apply`] performs it, both under the lock taken above, so two processes
        // starting together cannot each decide to migrate and then both try. Everything downstream
        // of this point may assume the current schema unconditionally, which is the whole benefit;
        // [`migrations`] states the rule that keeps it true.
        //
        // What the lock costs, and it is real: opening the store takes a write lock even when there
        // is nothing to do, so a long-running writer elsewhere fails commands that only read. An
        // external `BEGIN IMMEDIATE` held for eight seconds kills `meka --oneshot` at 5.1 seconds
        // with `failed to initialize schema: database is locked`, and `meka session list` -- a pure
        // read -- dies the same way. A rare, loud, retryable startup error is the accepted half of
        // that trade.
        let database_path = self.database_path.clone();
        let context = context.clone();
        let (plan, backup) = self
            .connection
            .call(move |connection| -> std::result::Result<_, MekaError> {
                let plan = migrations::plan(connection)?;
                // Before anything is written, and only when there is something to lose. `from > 0`
                // is what distinguishes carrying an existing store forward from building a new one:
                // a fresh store has no data to preserve, and copying the empty file it does not yet
                // have would leave a `.v0.bak` beside every first run. The copy carries its own
                // `user_version`, so restoring it yields a store that migrates once when next
                // opened rather than one mistaken for already-current.
                let backup = if plan.from > 0 && plan.has_work() {
                    back_up_before_migrating(connection, &database_path, plan.from)?
                } else {
                    None
                };
                // Here rather than inside the helper, so the `?` above is what guarantees a copy is
                // in place before an older one is removed. See `prune_older_backups`.
                if let Some(target) = &backup {
                    prune_older_backups(&database_path, target);
                }
                migrations::apply(connection, plan, &context)?;
                // Reconciliation rather than creation, and outside the ledger for that reason: it
                // asks whether this database's FTS triggers are the ones this build requires and
                // makes them so, which is as true of a store created a minute ago as of one carried
                // forward. `crate::memory::store` owns the reasoning.
                crate::memory::store::reconcile_index(connection).map_err(|error| {
                    MekaError::Database(format!("failed to reconcile the memory index: {}", error))
                })?;
                Ok((plan, backup))
            })
            .await
            .map_err(|error| match error {
                tokio_rusqlite::Error::Error(inner) => inner,
                other => MekaError::Database(format!("failed to initialize schema: {}", other)),
            })?;

        if plan.has_work() {
            match backup {
                Some(path) => tracing::info!(
                    "brought the store forward from schema version {} to {}; the pre-migration copy is at {}",
                    plan.from,
                    plan.head,
                    path.display()
                ),
                None => tracing::info!(
                    "brought the store forward from schema version {} to {}",
                    plan.from,
                    plan.head
                ),
            }
        }
        Ok(())
    }

    /// Create a new session, optionally recording its working directory. `cwd` is persisted as an
    /// absolute path string; pass `None` only for code paths that genuinely have no cwd context.
    ///
    /// Leaves the row unlocked, so a sweep can reach it before anyone claims it. Every host that
    /// creates a session a turn will run against wants [`Self::create_session_locked`] instead;
    /// this one is reached from tests.
    pub async fn create_session(
        &self,
        cwd: Option<std::path::PathBuf>,
        provider: impl Into<String>,
    ) -> Result<Uuid> {
        self.create_session_with_metadata(cwd, None, None, None, provider)
            .await
            .map(|created| created.id)
    }

    /// Like [`Self::create_session`] but also persists the HTTP API's per-session metadata
    /// (`permission` level, `capabilities_json` blob, and `token_id` fingerprint). The REPL
    /// and ACP paths derive permission from process config and don't have a bearer token.
    ///
    /// No production caller: every host that creates a session now goes through
    /// [`Self::create_session_locked`], which takes the lock before the row exists. This and
    /// [`Self::create_session`] remain as the unlocked doors, reached from tests and from callers
    /// that genuinely want a row nobody is holding.
    pub async fn create_session_with_metadata(
        &self,
        cwd: Option<std::path::PathBuf>,
        permission: Option<String>,
        capabilities_json: Option<String>,
        token_id: Option<String>,
        provider: impl Into<String>,
    ) -> Result<CreatedSession> {
        self.insert_session_row(
            Uuid::new_v4(),
            cwd,
            permission,
            capabilities_json,
            token_id,
            provider.into(),
        )
        .await
    }

    /// Create a session and take its lock, in that order: the lock **before** the row.
    ///
    /// The ordering is the entire point, and it is the reverse of what every caller used to do.
    /// Committing the row first leaves a window -- microseconds wide, but real -- in which the
    /// session is visible to `SELECT id FROM sessions` and held by nobody.
    /// [`Self::delete_all_sessions`] enumerates at delete time, so it lands inside that window,
    /// takes the lock legitimately, and cascades the conversation away underneath the process
    /// creating it. Measured at **42 lost turns in 11,948** with four creators against two
    /// `meka session delete --all` loops: each one ends `FOREIGN KEY constraint failed` with the
    /// user's prompt gone. Targeted `meka session delete <id>` never reproduced it, because its id
    /// list is gathered before the creator exists -- which is what identifies the window as
    /// belonging to creation rather than to deletion.
    ///
    /// Locking first closes it with nothing left over: a sweeper either cannot see the row yet, or
    /// sees it and finds the lock held. A lock file whose row never lands is swept by
    /// [`Self::prune_orphan_lock_files`] like any other orphan.
    ///
    /// An `Err` in the second half means the claim could not be *made* -- an unwritable lock
    /// directory, descriptors exhausted -- and never that somebody else holds it, because no other
    /// process can know this id yet. It is returned rather than logged-and-dropped so a caller that
    /// refuses can report the reason it actually hit. Callers differ on what it is worth: a host
    /// that must be alone refuses, and the agent's own path warns and runs the turn regardless
    /// rather than breaking installations that work today.
    pub async fn create_session_locked(
        &self,
        cwd: Option<std::path::PathBuf>,
        permission: Option<String>,
        capabilities_json: Option<String>,
        token_id: Option<String>,
        provider: impl Into<String>,
    ) -> Result<(CreatedSession, std::result::Result<FileLock, MekaError>)> {
        let session_id = Uuid::new_v4();
        let lock = self.claim_a_fresh_id(session_id);
        let created = self
            .insert_session_row(
                session_id,
                cwd,
                permission,
                capabilities_json,
                token_id,
                provider.into(),
            )
            .await?;
        Ok((created, lock))
    }

    /// How many sessions run on one provider profile.
    ///
    /// For `meka provider remove`, which otherwise strands them silently: the refusal only arrives
    /// when the user next resumes one, which can be long after the removal and somewhere else.
    pub async fn count_sessions_on_provider(&self, profile: &str) -> Result<u64> {
        let profile = profile.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                // Top-level only, matching what `meka session list` shows and what the warning's
                // own advice can act on. A sub-agent row copies its parent's binding, so counting
                // children reported a number many times what the user could see, about rows that
                // `meka -r <id> --provider <name>` is not for.
                connection.query_row(
                    "SELECT COUNT(*) FROM sessions
                     WHERE provider = ?1 AND parent_session_id IS NULL",
                    rusqlite::params![profile],
                    |row| row.get::<_, i64>(0),
                )
            })
            .await
            .map(|count| count.max(0) as u64)
            .map_err(|error| {
                MekaError::Database(format!("failed to count sessions on a provider: {}", error))
            })
    }

    /// What this session runs on, as recorded on its row.
    ///
    /// `None` means the session is gone. Every other answer is a profile name that resolved when it
    /// was written: every door that mints a session resolves one first and refuses without it, so
    /// there is no such thing as a session this meka created with none. The empty string is still
    /// reachable, on a carried-forward row the migration could resolve no profile for, and needs no
    /// branch here: like a name that has since left `config.toml`, it is refused by
    /// [`crate::provider::ProviderRegistry::settings`], by name.
    pub async fn recorded_provider(&self, session_id: Uuid) -> Result<Option<String>> {
        let id = session_id.to_string();
        self.connection
            .call(move |connection| {
                connection
                    .query_row(
                        "SELECT provider FROM sessions WHERE id = ?1",
                        [&id],
                        |row| row.get::<_, String>(0),
                    )
                    .optional()
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to read the session's provider: {}", error))
            })
    }

    /// Move a session onto another provider profile, permanently.
    ///
    /// The CLI's `/provider` and `--provider` repin and ACP's `session/set_config_option` land
    /// here; the HTTP surfaces move the same column through
    /// [`Self::update_session_metadata_atomic`], which writes it in one statement with permission
    /// and cwd. Either door leaves the row stating what the session actually runs on, rather than
    /// the drift the column exists to remove: overriding for one run and leaving the row saying
    /// something else.
    pub async fn set_recorded_provider(&self, session_id: Uuid, profile: &str) -> Result<bool> {
        let id = session_id.to_string();
        let profile = profile.to_string();
        let changed = self
            .connection
            .call(move |connection| {
                connection.execute(
                    "UPDATE sessions SET provider = ?2, updated_at = ?3 WHERE id = ?1",
                    rusqlite::params![id, profile, chrono::Utc::now().to_rfc3339()],
                )
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!(
                    "failed to record the session's provider: {}",
                    error
                ))
            })?;
        Ok(changed > 0)
    }

    /// Take the lock on an id that is about to become a row.
    ///
    /// The ordering rule in one place, because there are four doors that mint a session and each of
    /// them had it wrong in the same way: commit, then claim. A row visible to
    /// `SELECT id FROM sessions` and held by nobody is one a concurrent
    /// [`Self::delete_all_sessions`] enumerates, locks and cascades away underneath its creator.
    ///
    /// The `Err` is carried rather than logged and dropped so a caller that refuses can name the
    /// reason it hit. It never means "somebody else holds it": no other process can know this id.
    fn claim_a_fresh_id(&self, session_id: Uuid) -> std::result::Result<FileLock, MekaError> {
        self.lock_session(session_id).inspect_err(|error| {
            tracing::warn!(
                "could not lock session {} as it was created: {}; another meka process could \
                 attach to it or sweep it mid-turn",
                session_id,
                error
            );
        })
    }

    /// The row half of session creation, shared by the locked and unlocked doors so the columns
    /// are written in one place.
    async fn insert_session_row(
        &self,
        session_id: Uuid,
        cwd: Option<std::path::PathBuf>,
        permission: Option<String>,
        capabilities_json: Option<String>,
        token_id: Option<String>,
        provider: String,
    ) -> Result<CreatedSession> {
        let created_at = chrono::Utc::now().to_rfc3339();
        let cwd_string = cwd.map(|path| path.display().to_string());

        let created_at_for_db = created_at.clone();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO sessions (id, created_at, updated_at, cwd, permission, \
                     capabilities_json, token_id, provider)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    rusqlite::params![
                        session_id.to_string(),
                        created_at_for_db,
                        created_at_for_db,
                        cwd_string,
                        permission,
                        capabilities_json,
                        token_id,
                        provider,
                    ],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to create session: {}", error)))?;

        Ok(CreatedSession {
            id: session_id,
            created_at,
        })
    }

    /// Create a session whose `parent_session_id` references an existing session, used by
    /// `agent_spawn` so sub-agent conversations persist as children of the parent for auditing.
    /// Cascades on parent delete (see [`Self::delete_session`]). The optional `cwd` is the parent's
    /// cwd snapshot at spawn time.
    ///
    /// `subagent_spec_json` records the terms the worker was spawned under so `agent_followup` can
    /// rebuild it from them rather than from whatever the parent looks like at follow-up time. It
    /// is written with the row rather than updated afterwards: a spawn that fails between the
    /// two would otherwise leave a child that can be followed up on with no recorded terms.
    pub async fn create_child_session(
        &self,
        parent: Uuid,
        cwd: Option<std::path::PathBuf>,
        subagent_spec_json: Option<String>,
        // The profile the worker will actually be built on, which is the parent *agent's* live
        // binding and not necessarily the parent *row's*. The two differ for exactly as long as a
        // repin that could not take the runtime lock: ACP's `session/set_config_option` moves the
        // row mid-turn, `try_lock` fails, and the agent stays where it was until the next turn.
        // Selecting the column here recorded the profile the worker was not running on, so a later
        // `agent_followup` on that child resolved a different account from the one that did the
        // work. Passed in for the same reason the window is: everything about a worker's binding
        // comes off one cell.
        provider: String,
    ) -> Result<(Uuid, std::result::Result<FileLock, MekaError>)> {
        let session_id = Uuid::new_v4();
        // Locked before the row, like every other door that mints one -- and the exposure here is
        // not the microsecond window the others had. A sub-agent's row was never locked at any
        // point, so it sat claimable for the whole of the worker's run, which is seconds to
        // minutes. A concurrent `meka session delete --all` enumerates it, takes the lock nobody
        // holds, and cascades it away; the worker's next message insert then dies on
        // `FOREIGN KEY constraint failed` with its work gone. Demonstrated, not theorised.
        let lock = self.claim_a_fresh_id(session_id);
        let now = chrono::Utc::now().to_rfc3339();
        let cwd_string = cwd.map(|path| path.display().to_string());

        let inserted = self
            .connection
            .call(move |connection| -> rusqlite::Result<bool> {
                let rows = connection.execute(
                    // Still `SELECT … FROM sessions WHERE id = ?4` rather than a plain `VALUES`,
                    // because a parent that is gone must select no row and insert nothing. Only
                    // the provider stopped being read off that row; see the parameter's note.
                    "INSERT INTO sessions
                         (id, created_at, updated_at, parent_session_id, cwd, subagent_spec_json,
                          provider)
                     SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7
                     FROM sessions WHERE id = ?4",
                    rusqlite::params![
                        session_id.to_string(),
                        now,
                        now,
                        parent.to_string(),
                        cwd_string,
                        subagent_spec_json,
                        provider,
                    ],
                )?;
                Ok(rows > 0)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to create child session: {}", error))
            })?;

        // A parent that is gone selects no row, so this statement inserts nothing and succeeds.
        // Reading the count back is what keeps that an error: the `VALUES` form this replaced was
        // refused by `parent_session_id`'s foreign key, and without the check a spawn would hand
        // back an id with no row behind it -- a worker the model is told about, holding a lock file
        // for a session that never existed, whose first `save_message` dies on the constraint
        // instead. [`Self::fork_session_into`] reads its own count for the same reason.
        if !inserted {
            // Nothing was written, so the claim protects nothing and its file is garbage from the
            // moment it exists. The id is a fresh v4, so no one else can hold it.
            drop(lock);
            let path = self.lock_dir.join(format!("{}.lock", session_id));
            if let Err(error) = std::fs::remove_file(&path) {
                tracing::debug!("could not remove {}: {}", path.display(), error);
            }
            return Err(MekaError::Database(format!(
                "cannot spawn a sub-agent of session {}: it no longer exists",
                parent
            )));
        }

        Ok((session_id, lock))
    }

    /// The recorded spawn terms for a sub-agent session, or `None` for a top-level session.
    pub async fn load_subagent_spec(&self, session_id: Uuid) -> Result<Option<String>> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection
                    .query_row(
                        "SELECT subagent_spec_json FROM sessions WHERE id = ?1",
                        rusqlite::params![session_id.to_string()],
                        |row| row.get::<_, Option<String>>(0),
                    )
                    .optional()
                    .map(Option::flatten)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to load sub-agent spec: {}", error))
            })
    }

    /// Copy `source`'s conversation into a brand-new top-level session and return it, or `Ok(None)`
    /// when `source` doesn't exist (callers map that to their own not-found shape). The whole copy
    /// is one transaction, so a failure leaves no half-built session behind.
    ///
    /// What travels: the event log verbatim (per-event timestamps included), `tool_outputs`,
    /// `permission`, `capabilities_json`, the cumulative stats, and `cwd` / `additional_roots`
    /// unless [`ForkOverrides`] replaces them.
    ///
    /// What deliberately does not:
    ///
    /// - **`created_at` / `updated_at`**, both stamped to now. Retention GC deletes by `updated_at`
    ///   at every agent startup, so inheriting the source's would let a fork of an old session be
    ///   swept before its first turn.
    /// - **`parent_session_id`**, left NULL. That column means "sub-agent parent", and
    ///   [`Self::list_sessions`] hides rows that have one, so reusing it for fork lineage would
    ///   make every fork invisible to `meka session list`.
    /// - **Sub-agent children.** A child links to its parent only through `parent_session_id`,
    ///   while the sub-agent's *result* already sits in the parent's own event log as a tool
    ///   result, so the copy is self-contained without them. This is the intended divergence from
    ///   [`Self::import_sessions`], which copies the tree because an archive should restore whole.
    /// - **`subagent_spec_json`**, left NULL for the same reason as `parent_session_id`: a fork is
    ///   top-level, and spawn terms on a session nothing spawned would describe a relationship that
    ///   no longer exists. [`Self::import_sessions`] does carry it, because there the relationship
    ///   is carried too.
    ///
    /// Copying in SQL rather than through the export/import structs is deliberate: routing a fork
    /// through that envelope is precisely how `additional_roots` came to be silently dropped. The
    /// column list below lives next to the schema, and `fork_copies_every_session_column` fails
    /// when a new column appears without a decision about it.
    pub async fn fork_session(
        &self,
        source: Uuid,
        overrides: ForkOverrides,
    ) -> Result<Option<CreatedSession>> {
        self.fork_session_into(Uuid::new_v4(), source, overrides)
            .await
    }

    /// Fork, taking the copy's lock *before* its row exists.
    ///
    /// The door every host should use. A fork committed the copy and then locked it, which is the
    /// same commit-then-claim window [`Self::create_session_locked`] was written to close, in the
    /// same width: a concurrent `meka session delete --all` enumerates the copy, takes the lock
    /// nobody holds, and deletes it -- and `fork_and_lock` then locks the vanished id successfully
    /// and hands the caller a session whose next turn dies on a foreign-key violation. Under ACP it
    /// is quieter still, because `load_events` returns empty and the editor is handed a silently
    /// blank fork.
    ///
    /// `Ok(None)` still means the source is gone; the claim is released *and* its file removed,
    /// since nothing was written for it to protect.
    pub async fn fork_session_locked(
        &self,
        source: Uuid,
        overrides: ForkOverrides,
    ) -> Result<Option<(CreatedSession, std::result::Result<FileLock, MekaError>)>> {
        let new_id = Uuid::new_v4();
        let lock = self.claim_a_fresh_id(new_id);
        let Some(created) = self.fork_session_into(new_id, source, overrides).await? else {
            // Nothing was written, so the file this claim created is garbage the moment it exists.
            // Left behind it accumulates once per fork of an unknown id -- reachable from a client,
            // since ACP's `session/fork` answers `invalid_params` on that path -- and the sweep
            // that would collect it only runs at `open()` and after a delete. The id is
            // a fresh v4, so nothing else can be holding this file.
            drop(lock);
            let path = self.lock_dir.join(format!("{}.lock", new_id));
            if let Err(error) = std::fs::remove_file(&path) {
                tracing::debug!("could not remove {}: {}", path.display(), error);
            }
            return Ok(None);
        };
        Ok(Some((created, lock)))
    }

    /// The copy itself, on an id the caller has already minted (and may already have locked).
    async fn fork_session_into(
        &self,
        new_id: Uuid,
        source: Uuid,
        overrides: ForkOverrides,
    ) -> Result<Option<CreatedSession>> {
        let created_at = chrono::Utc::now().to_rfc3339();
        let cwd_override = overrides.cwd.map(|path| path.display().to_string());
        // A flag rather than a nested `Option`: "inherit" and "override with no roots" both encode
        // to SQL NULL, so `COALESCE` alone can't tell them apart and would resurrect the source's
        // roots when a caller explicitly asked for none.
        let (override_roots, roots_override) = match overrides.additional_roots {
            Some(roots) => (true, encode_additional_roots(&roots)?),
            None => (false, None),
        };

        let created_at_for_db = created_at.clone();
        let inserted = self
            .connection
            .call(move |connection| -> rusqlite::Result<bool> {
                let txn = connection.transaction()?;
                let source_id = source.to_string();
                let new_id_string = new_id.to_string();

                // The enclosing transaction is what makes the three statements below a consistent
                // snapshot: the first `INSERT` takes SQLite's write lock, so no other connection
                // can append an event to the source between the row copy and the message copy.
                let rows = txn.execute(
                    "INSERT INTO sessions (
                         id, created_at, updated_at, parent_session_id, cwd, permission,
                         capabilities_json, token_id, additional_roots_json, provider,
                         stat_turns,
                         stat_input_tokens, stat_output_tokens,
                         stat_cache_creation_input_tokens, stat_cache_read_input_tokens,
                         stat_redactions, stat_redacted_images, stat_redacted_bytes
                     )
                     SELECT ?1, ?2, ?2, NULL, COALESCE(?3, cwd), permission,
                            capabilities_json, ?4,
                            CASE WHEN ?5 THEN ?6 ELSE additional_roots_json END, provider,
                            stat_turns,
                            stat_input_tokens, stat_output_tokens,
                            stat_cache_creation_input_tokens, stat_cache_read_input_tokens,
                            stat_redactions, stat_redacted_images, stat_redacted_bytes
                     FROM sessions WHERE id = ?7",
                    rusqlite::params![
                        new_id_string,
                        created_at_for_db,
                        cwd_override,
                        overrides.token_id,
                        override_roots,
                        roots_override,
                        source_id,
                    ],
                )?;
                if rows == 0 {
                    // No source row: nothing was inserted, so the rollback is a formality.
                    txn.rollback()?;
                    return Ok(false);
                }

                txn.execute(
                    "INSERT INTO messages (session_id, role, content, created_at)
                     SELECT ?1, role, content, created_at
                     FROM messages WHERE session_id = ?2 ORDER BY id ASC",
                    rusqlite::params![new_id_string, source_id],
                )?;
                txn.execute(
                    "INSERT INTO tool_outputs (session_id, name, content, created_at)
                     SELECT ?1, name, content, created_at
                     FROM tool_outputs WHERE session_id = ?2",
                    rusqlite::params![new_id_string, source_id],
                )?;

                txn.commit()?;
                Ok(true)
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to fork session: {}", error)))?;

        if !inserted {
            return Ok(None);
        }
        Ok(Some(CreatedSession {
            id: new_id,
            created_at,
        }))
    }

    /// Acquire an exclusive OS file lock on the session. Returns a [`FileLock`] handle whose
    /// lifetime owns the lock; drop it (or let the process exit) to release.
    ///
    /// The session must already exist in the database. Returns [`MekaError::SessionLocked`] if
    /// another live process holds the lock.
    pub fn lock_session(&self, session_id: Uuid) -> Result<FileLock> {
        let path = self.lock_dir.join(format!("{}.lock", session_id));
        try_lock_file(&path)?.ok_or(MekaError::SessionLocked(session_id))
    }

    /// Best-effort removal of `<lock_dir>/<uuid>.lock` files whose session no longer exists.
    /// `lock_session` creates these files but never deletes them; the OS releases the *lock* on
    /// process exit, yet the empty file remains. A file for a UUID that isn't in the `sessions`
    /// table is pure garbage.
    ///
    /// Housekeeping only: never fails the caller. A DB-query failure is a recoverable fallback
    /// (`warn!`); a per-file unlink failure (e.g. a root-owned file left by a container run) is
    /// expected and logged at `debug!`.
    ///
    /// Nothing is unlinked without first taking its lock. The rule used to be "unlink any file
    /// whose id is not in the live set", justified by a session's row being committed before its
    /// lock file is acquired -- which constrains the *creator*, not the sweeper: this `SELECT` can
    /// finish before a row commits while the `read_dir` below runs after that session's lock file
    /// exists. Measured against a running `meka serve`, **21 of 401** live sessions had their lock
    /// file unlinked, after which `meka -r <id> --oneshot` attached to a session `serve` still held
    /// and wrote a full turn into it. Unlinking does not release a held `flock`, so the two
    /// processes then held locks on different inodes and neither could see the other.
    ///
    /// Taking the lock answers the question directly rather than inferring it, and it is the only
    /// thing here that does. An earlier version of this comment argued that a lock file comes into
    /// existence only *after* its row has committed, so a fresh `session_exists` was the whole
    /// answer. That is no longer true and was not worth relying on:
    /// [`Self::create_session_locked`] now takes the lock *before* the insert, precisely so a row
    /// cannot be visible to a sweep while unheld. A file with no row is therefore either genuine
    /// garbage from an earlier run or a session being created this instant, and only the `flock`
    /// can tell them apart.
    ///
    /// `session_exists` runs first anyway, because it is the cheaper question and the common answer
    /// is "the row is there, leave it alone" -- no reason to open and lock a file to learn that.
    ///
    /// One window is left and is accepted rather than closed. [`try_lock_file`] opens and then
    /// locks, two syscalls with nothing between; a sweep that `read_dir`s inside that gap can take
    /// a creator's lock and unlink its file, leaving that session to run unheld. It is
    /// sub-microsecond and it needs the sweep to be running at that instant (only at
    /// [`SessionManager::open`] and after a delete, with a `session_exists` round trip between its
    /// `read_dir` and the `flock`).
    ///
    /// What it costs is not merely a guarantee. A creator whose file is unlinked keeps its `flock`
    /// on an inode with no name and then commits its row, so from that moment the session has a
    /// visible row and nothing on disk to claim it -- and the next sweep re-creates the path, locks
    /// it, and deletes the row mid-turn. That is the same loss, reached one step later. The trade
    /// is still strongly favourable, because the window it replaced was thousands of times wider
    /// and needed no second coincidence at all, but it is a smaller chance of the same outcome
    /// rather than a lesser outcome.
    ///
    /// Those counts are from Unix. Windows `LockFileEx` will not let an open file be unlinked, so
    /// the same bug takes a different shape there -- the `remove_file` fails rather than succeeding
    /// and stranding two holders on separate inodes. Proving the file is unheld before unlinking it
    /// is the right answer on both.
    async fn prune_orphan_lock_files(&self) {
        let live_ids: std::collections::HashSet<String> = match self
            .connection
            .call(|connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare("SELECT id FROM sessions")?;
                let ids = statement
                    .query_map([], |row| row.get::<_, String>(0))?
                    .collect::<rusqlite::Result<std::collections::HashSet<String>>>()?;
                Ok(ids)
            })
            .await
        {
            Ok(ids) => ids,
            Err(error) => {
                tracing::warn!("lock-file prune: failed to list sessions: {}", error);
                return;
            }
        };

        let entries = match std::fs::read_dir(&self.lock_dir) {
            Ok(entries) => entries,
            Err(error) => {
                tracing::debug!(
                    "lock-file prune: cannot read {}: {}",
                    self.lock_dir.display(),
                    error
                );
                return;
            }
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("lock") {
                continue;
            }
            // Only touch files whose stem is a UUID; never delete an unrelated file someone
            // dropped into the lock directory.
            let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            // `schema.lock` shares this directory and is not a session's. It survives the UUID
            // check below by accident -- "schema" does not parse as one -- and the accident is
            // worth not relying on: unlinking a file does not release a held `flock`, so a swept
            // schema lock would let the next process create a different inode, and both would enter
            // `memory::store`'s FTS-trigger reconciliation believing they held it, each deciding
            // from a `sqlite_master` read the other is free to invalidate before the decision is
            // acted on. Named explicitly so a future lock called anything UUID-shaped does not
            // quietly inherit the hazard.
            if stem == SCHEMA_LOCK_STEM || stem.starts_with(PROVIDER_LOCK_PREFIX) {
                continue;
            }
            let Ok(id) = Uuid::parse_str(stem) else {
                continue;
            };
            if live_ids.contains(stem) {
                continue;
            }
            // Re-read, because the snapshot was taken before `read_dir` and a row committed since
            // would not be in it. Before the lock rather than after, so a creator that has opened
            // this file and not yet locked it is not raced for it. Cheap: only a genuinely
            // orphaned file gets this far, and `unwrap_or(true)` keeps a failed read on the
            // sparing side.
            if self.session_exists(id).await.unwrap_or(true) {
                continue;
            }
            // The claim, not a guess about one. A file whose lock is held belongs to a live
            // process whatever the `sessions` table says about its row.
            let Ok(claim) = self.lock_session(id) else {
                continue;
            };
            // Released before the unlink so the descriptor this process holds is not the one being
            // removed from under it.
            drop(claim);
            if let Err(error) = std::fs::remove_file(&path) {
                tracing::debug!(
                    "lock-file prune: cannot remove {}: {}",
                    path.display(),
                    error
                );
            }
        }
    }

    /// Persist a single event from the conversation log. Events are
    /// encoded into the existing `messages(role, content, …)` table:
    ///
    /// - `Event::Append(message)` writes one row with the message's role (`user` / `assistant` /
    ///   `tool_results`).
    /// - `Event::CompactBoundary { … }` writes one row with the pseudo-role `compact_boundary` and
    ///   a JSON-serialized envelope in `content`.
    pub async fn save_event(
        &self,
        session_id: Uuid,
        event: &crate::conversation::Event,
    ) -> Result<()> {
        let (role, content) = encode_event_for_db(event)
            .map_err(|error| MekaError::Database(format!("failed to encode event: {}", error)))?;
        self.save_message(session_id, &role, &content).await
    }

    /// Persist a batch of events atomically in one SQLite transaction.  The agent loop
    /// uses this to save the assistant message + the matching tool-results message
    /// together.  Without the transaction, a failure on the tool-results row would leave
    /// the assistant message persisted with `tool_use` blocks but no matching tool
    /// results, corrupting the conversation for subsequent turns.  The transaction
    /// guarantees either both rows commit or neither does.
    ///
    /// `events` MUST be non-empty; an empty batch is a no-op. `updated_at` is bumped once
    /// at the end of the batch (not once per row) so the row reflects the batch's commit
    /// time rather than the order events were appended.
    pub async fn save_events_atomic(
        &self,
        session_id: Uuid,
        events: Vec<crate::conversation::Event>,
    ) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }
        // Encode all events upfront so a serialization failure aborts before any DB I/O.
        let mut encoded: Vec<(String, String)> = Vec::with_capacity(events.len());
        for event in &events {
            let pair = encode_event_for_db(event).map_err(|error| {
                MekaError::Database(format!("failed to encode event: {}", error))
            })?;
            encoded.push(pair);
        }
        let now = chrono::Utc::now().to_rfc3339();
        let session_id_str = session_id.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let txn = connection.transaction()?;
                {
                    let mut insert = txn.prepare(
                        "INSERT INTO messages (session_id, role, content, created_at) \
                         VALUES (?1, ?2, ?3, ?4)",
                    )?;
                    for (role, content) in &encoded {
                        insert.execute(rusqlite::params![session_id_str, role, content, now])?;
                    }
                }
                txn.execute(
                    "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![now, session_id_str],
                )?;
                txn.commit()?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to save event batch atomically: {}", error))
            })
    }

    /// Persist a set of imported sessions (a root plus its sub-agent descendants) in a single
    /// transaction: the `sessions` rows (preserving `created_at` and cumulative stats, but never
    /// the `token_id` fingerprint), each session's event log (preserving per-event timestamps), and
    /// its `tool_outputs`. `records` MUST be ordered parents-first so every `new_parent_id`
    /// references an already-inserted row (the `parent_session_id` foreign key is enforced).
    /// All-or-nothing: any failure rolls back the whole import, leaving no partial tree.
    ///
    /// `updated_at` is stamped to the import time rather than restored from the export. Retention
    /// GC deletes by `updated_at` ([`Self::delete_expired_sessions`], run at startup when
    /// `[session].retention_days` is set), so restoring the original value meant an archive older
    /// than the window was swept on the next launch, before anyone could resume it. `created_at`
    /// still carries the original for provenance.
    pub async fn import_sessions(&self, records: Vec<ImportSessionRecord>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        // Encode every event up front so a serialization failure aborts before any DB I/O.
        struct EncodedSession {
            id: String,
            parent_id: Option<String>,
            created_at: String,
            cwd: Option<String>,
            permission: Option<String>,
            capabilities_json: Option<String>,
            additional_roots_json: Option<String>,
            subagent_spec_json: Option<String>,
            provider: String,
            stats: crate::stats::SessionStatsSnapshot,
            events: Vec<(String, String, String)>,
            tool_outputs: Vec<(String, String)>,
        }
        let imported_at = chrono::Utc::now().to_rfc3339();
        let mut encoded = Vec::with_capacity(records.len());
        for record in records {
            let mut events = Vec::with_capacity(record.events.len());
            for (at, event) in &record.events {
                let (role, content) = encode_event_for_db(event).map_err(|error| {
                    MekaError::Database(format!("failed to encode event: {}", error))
                })?;
                events.push((role, content, at.clone()));
            }
            encoded.push(EncodedSession {
                id: record.new_id.to_string(),
                parent_id: record.new_parent_id.map(|id| id.to_string()),
                created_at: record.created_at,
                cwd: record.cwd,
                permission: record.permission,
                capabilities_json: record.capabilities_json,
                additional_roots_json: encode_additional_roots(&record.additional_roots)?,
                subagent_spec_json: record.subagent_spec_json,
                provider: record.provider,
                stats: record.stats,
                events,
                tool_outputs: record.tool_outputs,
            });
        }
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let txn = connection.transaction()?;
                for session in &encoded {
                    txn.execute(
                        "INSERT INTO sessions (
                             id, created_at, updated_at, parent_session_id, cwd, permission,
                             capabilities_json, additional_roots_json, subagent_spec_json,
                             provider,
                             stat_turns, stat_input_tokens, stat_output_tokens,
                             stat_cache_creation_input_tokens, stat_cache_read_input_tokens,
                             stat_redactions, stat_redacted_images, stat_redacted_bytes
                         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
                        rusqlite::params![
                            session.id,
                            session.created_at,
                            imported_at,
                            session.parent_id,
                            session.cwd,
                            session.permission,
                            session.capabilities_json,
                            session.additional_roots_json,
                            session.subagent_spec_json,
                            session.provider,
                            session.stats.turns as i64,
                            session.stats.input_tokens as i64,
                            session.stats.output_tokens as i64,
                            session.stats.cache_creation_input_tokens as i64,
                            session.stats.cache_read_input_tokens as i64,
                            session.stats.redactions as i64,
                            session.stats.redacted_images as i64,
                            session.stats.redacted_bytes as i64,
                        ],
                    )?;
                    {
                        let mut insert_event = txn.prepare(
                            "INSERT INTO messages (session_id, role, content, created_at) \
                             VALUES (?1, ?2, ?3, ?4)",
                        )?;
                        for (role, content, created_at) in &session.events {
                            insert_event.execute(rusqlite::params![
                                session.id,
                                role,
                                content,
                                created_at
                            ])?;
                        }
                    }
                    {
                        let mut insert_output = txn.prepare(
                            "INSERT INTO tool_outputs (session_id, name, content, created_at) \
                             VALUES (?1, ?2, ?3, ?4)",
                        )?;
                        for (name, content) in &session.tool_outputs {
                            insert_output.execute(rusqlite::params![
                                session.id,
                                name,
                                content,
                                session.created_at
                            ])?;
                        }
                    }
                }
                txn.commit()?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to import sessions: {}", error)))
    }

    /// Load every event for a session in chronological order. The `user`, `assistant`,
    /// `tool_results` and `user_blocks` roles are what `encode_event_for_db` writes for an
    /// `Event::Append`, so reading them back as `Event::Append` closes that round trip rather than
    /// falling back to anything; `compact_boundary` and `repair` rows are deserialized from their
    /// JSON envelope. Unknown roles are skipped with a warning.
    pub async fn load_events(&self, session_id: Uuid) -> Result<Vec<crate::conversation::Event>> {
        let stored = self.load_messages(session_id).await?;
        let mut events = Vec::with_capacity(stored.len());
        for row in stored {
            match decode_event_from_row(&row) {
                Ok(Some(event)) => events.push(event),
                Ok(None) => {
                    tracing::warn!("dropping unparseable session row (role={})", row.role);
                }
                Err(error) => {
                    tracing::warn!(
                        "failed to decode session row (role={}): {}",
                        row.role,
                        error
                    );
                }
            }
        }
        Ok(events)
    }

    /// Variant of [`Self::load_events`] that also returns the persisted `created_at` timestamp for
    /// each event. Used by the HTTP `GET /v1/sessions/{id}/messages` endpoint to surface
    /// per-message creation timestamps on `MessageView` per the spec's resource model.
    /// Order matches `load_events` exactly: chronological by insert id.
    pub async fn load_events_with_timestamps(
        &self,
        session_id: Uuid,
    ) -> Result<Vec<(String, crate::conversation::Event)>> {
        let stored = self.load_messages(session_id).await?;
        let mut events = Vec::with_capacity(stored.len());
        for row in stored {
            match decode_event_from_row(&row) {
                Ok(Some(event)) => events.push((row.created_at, event)),
                Ok(None) => {
                    tracing::warn!("dropping unparseable session row (role={})", row.role);
                }
                Err(error) => {
                    tracing::warn!(
                        "failed to decode session row (role={}): {}",
                        row.role,
                        error
                    );
                }
            }
        }
        Ok(events)
    }

    /// Load a session together with every descendant sub-agent session (recursively via
    /// `parent_session_id`), ordered root-first (breadth-first by depth). Used by JSON session
    /// export to capture an entire agent tree, and the root-first order lets an importer insert
    /// parents before children so the `parent_session_id` foreign key is always satisfied. Returns
    /// an empty vec when the root session doesn't exist.
    pub async fn load_session_tree(&self, root: Uuid) -> Result<Vec<SessionMetaRow>> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(
                    "WITH RECURSIVE tree(id, depth) AS (
                         SELECT id, 0 FROM sessions WHERE id = ?1
                         UNION ALL
                         SELECT s.id, tree.depth + 1
                         FROM sessions s JOIN tree ON s.parent_session_id = tree.id
                     )
                     SELECT s.id, s.parent_session_id, s.created_at, s.updated_at,
                            s.cwd, s.permission, s.capabilities_json, s.additional_roots_json,
                            s.subagent_spec_json, s.provider
                     FROM sessions s JOIN tree ON s.id = tree.id
                     ORDER BY tree.depth ASC, s.created_at ASC, s.id ASC",
                )?;
                let parse_uuid = |value: String| {
                    Uuid::parse_str(&value)
                        .map_err(|error| rusqlite::Error::InvalidParameterName(error.to_string()))
                };
                let rows = statement.query_map(rusqlite::params![root.to_string()], |row| {
                    let id = parse_uuid(row.get::<_, String>(0)?)?;
                    let parent_id = match row.get::<_, Option<String>>(1)? {
                        Some(value) => Some(parse_uuid(value)?),
                        None => None,
                    };
                    Ok(SessionMetaRow {
                        id,
                        parent_id,
                        created_at: row.get(2)?,
                        updated_at: row.get(3)?,
                        cwd: row.get(4)?,
                        permission: row.get(5)?,
                        capabilities_json: row.get(6)?,
                        additional_roots: decode_additional_roots(
                            row.get::<_, Option<String>>(7)?.as_deref(),
                        ),
                        subagent_spec_json: row.get(8)?,
                        provider: row.get(9)?,
                    })
                })?;
                let mut out = Vec::new();
                for row in rows {
                    out.push(row?);
                }
                Ok(out)
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to load session tree: {}", error)))
    }

    /// Persist a single row into the `messages` table. Internal helper for [`Self::save_event`];
    /// external consumers go through the event API. Tests still call this directly to populate
    /// fixtures.
    pub(super) async fn save_message(
        &self,
        session_id: Uuid,
        role: &str,
        content: &str,
    ) -> Result<()> {
        let role = role.to_string();
        let content = content.to_string();
        let now = chrono::Utc::now().to_rfc3339();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO messages (session_id, role, content, created_at) VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![session_id.to_string(), role, content, now],
                )?;

                connection.execute(
                    "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![
                        chrono::Utc::now().to_rfc3339(),
                        session_id.to_string()
                    ],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to save message: {}", error)))
    }

    /// Persist the cumulative `/status` counters onto the session row so they survive resume. The
    /// caller treats this as best-effort (a failed write must never fail a turn).
    pub async fn save_session_stats(
        &self,
        session_id: Uuid,
        stats: &crate::stats::SessionStatsSnapshot,
    ) -> Result<()> {
        // SQLite has no u64; counts never realistically exceed i64::MAX, so cast on the way in/out.
        let stats = stats.clone();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE sessions SET
                         stat_turns = ?2,
                         stat_input_tokens = ?3,
                         stat_output_tokens = ?4,
                         stat_cache_creation_input_tokens = ?5,
                         stat_cache_read_input_tokens = ?6,
                         stat_redactions = ?7,
                         stat_redacted_images = ?8,
                         stat_redacted_bytes = ?9
                     WHERE id = ?1",
                    rusqlite::params![
                        session_id.to_string(),
                        stats.turns as i64,
                        stats.input_tokens as i64,
                        stats.output_tokens as i64,
                        stats.cache_creation_input_tokens as i64,
                        stats.cache_read_input_tokens as i64,
                        stats.redactions as i64,
                        stats.redacted_images as i64,
                        stats.redacted_bytes as i64,
                    ],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to save session stats: {}", error))
            })
    }

    /// Load the persisted cumulative stats for a session, used to seed `SessionStats` on resume.
    /// Returns all-zero when the session row doesn't exist yet (fresh session).
    pub async fn load_session_stats(
        &self,
        session_id: Uuid,
    ) -> Result<crate::stats::SessionStatsSnapshot> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let result = connection.query_row(
                    "SELECT stat_turns, stat_input_tokens, stat_output_tokens,
                            stat_cache_creation_input_tokens, stat_cache_read_input_tokens,
                            stat_redactions, stat_redacted_images, stat_redacted_bytes
                     FROM sessions WHERE id = ?1",
                    rusqlite::params![session_id.to_string()],
                    |row| {
                        Ok(crate::stats::SessionStatsSnapshot {
                            turns: row.get::<_, i64>(0)? as u64,
                            input_tokens: row.get::<_, i64>(1)? as u64,
                            output_tokens: row.get::<_, i64>(2)? as u64,
                            cache_creation_input_tokens: row.get::<_, i64>(3)? as u64,
                            cache_read_input_tokens: row.get::<_, i64>(4)? as u64,
                            redactions: row.get::<_, i64>(5)? as u64,
                            redacted_images: row.get::<_, i64>(6)? as u64,
                            redacted_bytes: row.get::<_, i64>(7)? as u64,
                        })
                    },
                );
                match result {
                    Ok(snapshot) => Ok(snapshot),
                    Err(rusqlite::Error::QueryReturnedNoRows) => {
                        Ok(crate::stats::SessionStatsSnapshot::default())
                    }
                    Err(error) => Err(error),
                }
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to load session stats: {}", error))
            })
    }

    /// Fetch raw rows for a session. Internal helper for [`Self::load_events`]; external consumers
    /// go through the event API.
    async fn load_messages(&self, session_id: Uuid) -> Result<Vec<StoredMessage>> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(
                    "SELECT role, content, created_at FROM messages WHERE session_id = ?1 ORDER BY id ASC",
                )?;

                let messages = statement
                    .query_map(rusqlite::params![session_id.to_string()], |row| {
                        Ok(StoredMessage {
                            role: row.get(0)?,
                            content: row.get(1)?,
                            created_at: row.get(2)?,
                        })
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;

                Ok(messages)
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to load messages: {}", error)))
    }

    /// How many times this session has been compacted, i.e. its compaction *generation*.
    ///
    /// Read from the database rather than the in-memory log because
    /// [`crate::conversation::Conversation::prune_compacted_events`] drains every event preceding
    /// the most recent boundary, so the log in memory holds at most one no matter how many
    /// compactions have run. Every boundary is still its own row here.
    ///
    /// Worth surfacing to the model: a fourth summary-of-a-summary has lost far more than a first,
    /// and an agent that knows its generation can compensate by writing to memory more readily.
    pub async fn count_compactions(&self, session_id: Uuid) -> Result<u64> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.query_row(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?1 AND role = ?2",
                    rusqlite::params![session_id.to_string(), COMPACT_BOUNDARY_ROLE],
                    |row| row.get::<_, i64>(0),
                )
            })
            .await
            .map(|count| count.max(0) as u64)
            .map_err(|error| MekaError::Database(format!("failed to count compactions: {}", error)))
    }

    pub async fn last_session_id(&self) -> Result<Option<Uuid>> {
        self.connection
            .call(|connection| -> rusqlite::Result<_> {
                let result: std::result::Result<String, _> = connection.query_row(
                    "SELECT id FROM sessions ORDER BY updated_at DESC LIMIT 1",
                    [],
                    |row| row.get(0),
                );

                match result {
                    Ok(id_str) => {
                        let uuid = Uuid::parse_str(&id_str).map_err(|error| {
                            rusqlite::Error::InvalidParameterName(error.to_string())
                        })?;
                        Ok(Some(uuid))
                    }
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(error) => Err(error),
                }
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to get last session: {}", error)))
    }

    pub async fn session_exists(&self, session_id: Uuid) -> Result<bool> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let count: i64 = connection.query_row(
                    "SELECT COUNT(*) FROM sessions WHERE id = ?1",
                    rusqlite::params![session_id.to_string()],
                    |row| row.get(0),
                )?;
                Ok(count > 0)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to check session existence: {}", error))
            })
    }

    /// Resolve a session-ID prefix (e.g. `d64`) to the matching full UUIDs.
    ///
    /// Used by `meka -c <prefix>` so the user doesn't have to type the whole UUID. Capped at 16
    /// matches; ordered most-recent-first so the caller's "ambiguous prefix" listing leads with the
    /// session the user most likely meant.
    ///
    /// Anything outside the UUID alphabet (`0-9a-fA-F-`) returns an empty list, both because such
    /// a prefix can't match any real session ID and to keep SQL `LIKE` wildcards (`%`, `_`) from
    /// sneaking through.
    pub async fn find_sessions_by_prefix(&self, prefix: &str) -> Result<Vec<Uuid>> {
        if prefix.is_empty() || !prefix.chars().all(|c| c.is_ascii_hexdigit() || c == '-') {
            return Ok(Vec::new());
        }
        let pattern = format!("{}%", prefix);
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(
                    "SELECT id FROM sessions WHERE id LIKE ?1 \
                     ORDER BY updated_at DESC LIMIT 16",
                )?;
                let rows = statement.query_map(rusqlite::params![pattern], |row| {
                    let id: String = row.get(0)?;
                    Ok(id)
                })?;
                let mut ids = Vec::new();
                for row in rows {
                    if let Ok(uuid) = Uuid::parse_str(&row?) {
                        ids.push(uuid);
                    }
                }
                Ok(ids)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to find sessions by prefix: {}", error))
            })
    }

    /// List sessions, most-recent first. When `include_children` is `false`, sub-agent sessions
    /// (rows with non-NULL `parent_session_id`) are hidden; they're persisted for audit/debug but
    /// shouldn't clutter the user's view of their own conversations. Set to `true` to surface them,
    /// e.g. via `meka session list --include-children`.
    ///
    /// `cwd_filter`, if `Some`, restricts the result set to sessions whose persisted `cwd` matches
    /// the given path. Rows with NULL `cwd` are excluded: a session created by
    /// `create_session(None, "test-profile".to_string())` recorded no cwd to match against.
    ///
    /// `cursor`, if `Some`, is a previous `next_cursor` value from this method; rows are returned
    /// strictly *after* the cursor in `(updated_at, id) DESC` order. Returns `(rows, next_cursor)`;
    /// `next_cursor` is `Some` iff there is at least one more row past `limit`. Invalid cursors
    /// are rejected with [`MekaError::Database`].
    pub async fn list_sessions(
        &self,
        limit: u32,
        include_children: bool,
        cwd_filter: Option<&Path>,
        cursor: Option<&str>,
    ) -> Result<(Vec<SessionSummary>, Option<String>)> {
        let cursor_decoded = match cursor {
            Some(token) => Some(decode_list_cursor(token)?),
            None => None,
        };
        let cwd_filter_string = cwd_filter.map(|path| path.display().to_string());

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut clauses: Vec<&str> = Vec::new();
                if !include_children {
                    clauses.push("s.parent_session_id IS NULL");
                }
                if cwd_filter_string.is_some() {
                    clauses.push("s.cwd = :cwd");
                }
                if cursor_decoded.is_some() {
                    // Keyset on (updated_at, id) DESC: strictly past the cursor row. Tie-break on
                    // id keeps pagination stable when multiple sessions share an updated_at.
                    clauses.push(
                        "(s.updated_at < :cursor_updated_at \
                          OR (s.updated_at = :cursor_updated_at AND s.id < :cursor_id))",
                    );
                }
                let where_clause = if clauses.is_empty() {
                    String::new()
                } else {
                    format!("WHERE {}", clauses.join(" AND "))
                };
                let query = format!(
                    "SELECT s.id, s.created_at, s.updated_at, s.cwd, s.permission, s.capabilities_json, s.additional_roots_json, s.token_id, s.parent_session_id, s.provider,
                            COALESCE(
                              (SELECT content FROM messages
                               WHERE session_id = s.id AND role = 'user'
                               ORDER BY id ASC LIMIT 1),
                              ''
                            ) AS preview
                     FROM sessions s
                     {}
                     ORDER BY s.updated_at DESC, s.id DESC
                     LIMIT :limit",
                    where_clause,
                );
                let mut statement = connection.prepare(&query)?;

                // Fetch one extra row to detect whether a next page exists without a second COUNT
                // query.
                let fetch_limit: i64 = i64::from(limit).saturating_add(1);
                let mut params: Vec<(&str, &dyn rusqlite::ToSql)> = Vec::new();
                params.push((":limit", &fetch_limit));
                if let Some(ref cwd) = cwd_filter_string {
                    params.push((":cwd", cwd));
                }
                if let Some((ref updated_at, ref id)) = cursor_decoded {
                    params.push((":cursor_updated_at", updated_at));
                    params.push((":cursor_id", id));
                }

                let rows = statement.query_map(params.as_slice(), |row| {
                    let id_str: String = row.get(0)?;
                    let created_at: String = row.get(1)?;
                    let updated_at: String = row.get(2)?;
                    let cwd: Option<String> = row.get(3)?;
                    let permission: Option<String> = row.get(4)?;
                    let capabilities_json: Option<String> = row.get(5)?;
                    let additional_roots_json: Option<String> = row.get(6)?;
                    let token_id: Option<String> = row.get(7)?;
                    let parent_id: Option<String> = row.get(8)?;
                    let provider: String = row.get(9)?;
                    let preview: String = row.get(10)?;
                    Ok((
                        id_str,
                        created_at,
                        updated_at,
                        cwd,
                        permission,
                        capabilities_json,
                        additional_roots_json,
                        token_id,
                        parent_id,
                        provider,
                        preview,
                    ))
                })?;

                let mut summaries = Vec::new();
                for row in rows {
                    let (
                        id_str,
                        created_at,
                        updated_at,
                        cwd,
                        permission,
                        capabilities_json,
                        additional_roots_json,
                        token_id,
                        parent_id,
                        provider,
                        preview,
                    ) = row?;
                    let id = Uuid::parse_str(&id_str).map_err(|error| {
                        rusqlite::Error::InvalidParameterName(error.to_string())
                    })?;
                    let preview = truncate_preview(&preview, 80);
                    summaries.push(SessionSummary {
                        id,
                        created_at,
                        updated_at,
                        preview,
                        cwd: cwd.map(PathBuf::from),
                        permission,
                        provider,
                        capabilities_json,
                        additional_roots: decode_additional_roots(additional_roots_json.as_deref()),
                        token_id,
                        parent_id: parent_id.as_deref().and_then(|raw| Uuid::parse_str(raw).ok()),
                    });
                }
                Ok(summaries)
            })
            .await
            .map(|mut rows| {
                let next_cursor = if rows.len() > limit as usize {
                    rows.truncate(limit as usize);
                    rows.last()
                        .map(|row| encode_list_cursor(&row.updated_at, &row.id.to_string()))
                } else {
                    None
                };
                (rows, next_cursor)
            })
            .map_err(|error| MekaError::Database(format!("failed to list sessions: {}", error)))
    }

    /// Fetch a single session by id without scanning the full list. Returns `Ok(None)` if the
    /// session doesn't exist. Used by ACP's `session/load` to verify the requested session exists
    /// and to surface its persisted cwd back to the client.
    pub async fn session_info(&self, id: Uuid) -> Result<Option<SessionSummary>> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(
                    "SELECT s.id, s.created_at, s.updated_at, s.cwd, s.permission, s.capabilities_json, s.additional_roots_json, s.token_id, s.parent_session_id, s.provider,
                            COALESCE(
                              (SELECT content FROM messages
                               WHERE session_id = s.id AND role = 'user'
                               ORDER BY id ASC LIMIT 1),
                              ''
                            ) AS preview
                     FROM sessions s
                     WHERE s.id = ?1",
                )?;
                let mut rows = statement.query_map(rusqlite::params![id.to_string()], |row| {
                    let id_str: String = row.get(0)?;
                    let created_at: String = row.get(1)?;
                    let updated_at: String = row.get(2)?;
                    let cwd: Option<String> = row.get(3)?;
                    let permission: Option<String> = row.get(4)?;
                    let capabilities_json: Option<String> = row.get(5)?;
                    let additional_roots_json: Option<String> = row.get(6)?;
                    let token_id: Option<String> = row.get(7)?;
                    let parent_id: Option<String> = row.get(8)?;
                    let provider: String = row.get(9)?;
                    let preview: String = row.get(10)?;
                    Ok((
                        id_str,
                        created_at,
                        updated_at,
                        cwd,
                        permission,
                        capabilities_json,
                        additional_roots_json,
                        token_id,
                        parent_id,
                        provider,
                        preview,
                    ))
                })?;
                match rows.next() {
                    Some(row) => {
                        let (
                            id_str,
                            created_at,
                            updated_at,
                            cwd,
                            permission,
                            capabilities_json,
                            additional_roots_json,
                            token_id,
                            parent_id,
                            provider,
                            preview,
                        ) = row?;
                        let id = Uuid::parse_str(&id_str).map_err(|error| {
                            rusqlite::Error::InvalidParameterName(error.to_string())
                        })?;
                        Ok(Some(SessionSummary {
                            id,
                            created_at,
                            updated_at,
                            preview: truncate_preview(&preview, 80),
                            cwd: cwd.map(PathBuf::from),
                            permission,
                            provider,
                            additional_roots: decode_additional_roots(
                                additional_roots_json.as_deref(),
                            ),
                            capabilities_json,
                            token_id,
                            parent_id: parent_id
                                .as_deref()
                                .and_then(|raw| Uuid::parse_str(raw).ok()),
                        }))
                    }
                    None => Ok(None),
                }
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to fetch session: {}", error)))
    }

    /// Run `PRAGMA wal_checkpoint(TRUNCATE)` to flush the SQLite write-ahead log into the main
    /// database file. Called from `meka serve`'s graceful-shutdown path so a `SIGTERM` followed
    /// by a fresh `meka` process invocation doesn't see a long WAL replay on open. Errors are
    /// non-fatal: SQLite recovers from an unflushed WAL on next open, so we log and continue.
    pub async fn checkpoint(&self) -> Result<()> {
        self.connection
            .call(|connection| -> rusqlite::Result<_> {
                connection.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("WAL checkpoint failed: {}", error)))
    }

    /// Backdate a session's `updated_at`, for tests that need one to look old to the retention
    /// sweep. Lives here because `connection` is private to this module, so tests in other modules
    /// have no other way to age a row.
    #[cfg(test)]
    pub(crate) async fn set_session_updated_at_for_test(
        &self,
        session_id: uuid::Uuid,
        updated_at: &str,
    ) -> Result<()> {
        let updated_at = updated_at.to_string();
        let rows = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![updated_at, session_id.to_string()],
                )
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to backdate session: {}", error))
            })?;
        // A typo'd id would otherwise backdate nothing and leave the test asserting against a
        // session that was never aged, which reads as the sweep failing to match.
        if rows != 1 {
            return Err(MekaError::Database(format!(
                "expected to backdate 1 session, updated {rows}"
            )));
        }
        Ok(())
    }

    /// Sweep sessions no turn has touched inside the retention window, leaving alone any that
    /// another meka process currently has open.
    ///
    /// The lock check is not a nicety. `updated_at` is bumped by turns only, and resuming a session
    /// does not touch it, so a REPL left sitting at its prompt past the window is a perfect
    /// candidate for deletion while a human is looking at it. Any start that goes through
    /// `async_main` runs this sweep, so a completely unrelated `meka` in another terminal used to
    /// announce `deleted 1 session(s)` and destroy the live one. What the operator saw was their
    /// next turn running against the provider and *then* failing on a foreign-key violation, with
    /// the answer paid for and lost, and every later turn in that REPL failing the same way.
    ///
    /// A locked *child* is not separately checked. The cascade would take one with its parent, but
    /// children are sub-agent rows and nothing ever locks those; a child that could be locked would
    /// need its own claim on the parent, which is not a shape that exists.
    pub async fn delete_expired_sessions(&self, retention_days: u64) -> Result<SessionSweep> {
        // Both steps can blow up on an absurd `retention_days`, and this takes user input straight
        // from `--older-than-days`, so a run of digits must not panic. `TimeDelta` overflows around
        // 10^11 days; subtracting from `Utc::now()` overflows far sooner, around 96.4 million. Both
        // fall back to a ~100-year window, which matches nothing and so keeps every session: the
        // sane reading of "delete anything older than forever".
        let now = chrono::Utc::now();
        #[allow(clippy::expect_used)]
        let fallback = now
            .checked_sub_signed(chrono::TimeDelta::days(36_500))
            .expect("100 years before now is representable");
        let cutoff = i64::try_from(retention_days)
            .ok()
            .and_then(chrono::TimeDelta::try_days)
            .and_then(|retention| now.checked_sub_signed(retention))
            .unwrap_or(fallback);
        let cutoff_str = cutoff.to_rfc3339();

        let expired: Vec<Uuid> = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                // FK CASCADE sweeps messages, tool_outputs, and any sub-agent child sessions of the
                // expired parents.
                //
                // A session with a scheduled job still ahead of it is *not* expired, whatever
                // `updated_at` says. Only turns bump that column -- `complete_claim` and
                // `claim_occurrence` touch `scheduled_jobs` alone -- so a gated watcher
                // that evaluates every tick but rarely fires looks untouched for as
                // long as it stays quiet, which is exactly when it is working. The
                // cascade then took the job with the session, and the sweep
                // reported "deleted 1 session(s)" without ever mentioning
                // that a schedule went with it.
                //
                // Sparing only the row that *owns* the job is not enough. `parent_session_id`
                // carries `ON DELETE CASCADE`, so deleting a stale parent silently takes its
                // sub-agent children -- and a job created against a child (reachable over HTTP,
                // whose only gate is that the session exists) goes with them. The recursive term
                // walks parent links up from every job-owning session and spares that whole chain.
                //
                // Selected rather than deleted outright, because which of these rows may go is not
                // a question the database can answer: it depends on which of them another process
                // has open. [`Self::delete_the_unattached_among`] re-applies the same condition
                // inside the delete, so splitting one statement into two does not open a window
                // where a job created in between is cascaded away by a decision taken before it
                // existed.
                let mut statement = connection.prepare(&format!(
                    "SELECT id FROM sessions WHERE updated_at < ?1 AND {}",
                    NOT_SPOKEN_FOR_BY_A_SCHEDULE
                ))?;
                let ids = statement
                    .query_map(rusqlite::params![cutoff_str], |row| row.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<String>>>()?;
                Ok(ids)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to list expired sessions: {}", error))
            })?
            .into_iter()
            .filter_map(|id| Uuid::parse_str(&id).ok())
            .collect();

        self.delete_the_unattached_among(&expired, NOT_SPOKEN_FOR_BY_A_SCHEDULE)
            .await
    }

    /// Delete every one of `candidates` whose lock this process can take, and report what it left.
    ///
    /// The lock is held across the delete rather than probed and released, because the window
    /// between a probe and a `DELETE` is exactly long enough for another process to attach and
    /// start a turn on a row that is about to vanish underneath it.
    ///
    /// `still_eligible` is a SQL predicate re-applied inside the delete, so the statement decides
    /// on the rows as they are rather than on a list read earlier. Selecting candidates and then
    /// deleting them by id is two statements where there used to be one, and a condition checked
    /// only in the first is a condition that can stop being true in between: the retention sweep's
    /// "no schedule ahead of it" is the case that matters, because a job created against a
    /// sub-agent child in that gap would be cascaded away with a parent nothing has locked. A
    /// `&'static str` and never anything derived from input; `""` for a caller with no further
    /// condition, which is `--all`.
    ///
    /// Chunked because a lock is an open file descriptor: a sweep over ten thousand expired
    /// sessions would otherwise hold ten thousand at once and hit the process limit, turning a
    /// housekeeping pass into a hard failure. The chunk is the unit of both locking and deleting,
    /// so no lock is held longer than its own statement.
    async fn delete_the_unattached_among(
        &self,
        candidates: &[Uuid],
        still_eligible: &'static str,
    ) -> Result<SessionSweep> {
        /// Sessions locked and deleted per statement. Well under SQLite's parameter ceiling
        /// (32,766) and well under any sane descriptor limit.
        const CHUNK: usize = 100;

        let mut sweep = SessionSweep::default();
        for chunk in candidates.chunks(CHUNK) {
            let mut held = Vec::with_capacity(chunk.len());
            for id in chunk {
                match self.lock_session(*id) {
                    Ok(lock) => held.push((*id, lock)),
                    // Not "someone has it" but "we could not ask", which is the same answer here:
                    // a session this process cannot establish a claim on is one it must not
                    // delete. Counted as attached so the caller still reports incomplete coverage.
                    Err(_) => sweep.attached_elsewhere += 1,
                }
            }
            if held.is_empty() {
                continue;
            }
            let ids: Vec<String> = held.iter().map(|(id, _)| id.to_string()).collect();
            let deleted = self
                .connection
                .call(move |connection| -> rusqlite::Result<_> {
                    let placeholders = vec!["?"; ids.len()].join(",");
                    connection.execute(
                        &format!(
                            "DELETE FROM sessions WHERE id IN ({}){}",
                            placeholders,
                            match still_eligible {
                                "" => String::new(),
                                predicate => format!(" AND {}", predicate),
                            }
                        ),
                        rusqlite::params_from_iter(ids.iter()),
                    )
                })
                .await
                .map_err(|error| {
                    MekaError::Database(format!("failed to delete sessions: {}", error))
                })?;
            sweep.deleted += deleted as u64;
            // Before the sweep below, not after: `prune_orphan_lock_files` refuses to unlink a
            // file whose lock it cannot take, and this process is holding every one of these.
            drop(held);
        }
        self.prune_orphan_lock_files().await;
        Ok(sweep)
    }

    /// Update the persisted cwd for an existing session. Called by the ACP `session/load` /
    /// `session/resume` handlers when the client's `cwd` differs from the persisted value; the
    /// client wins so future `session/list` results reflect the live state. `cwd` is stored as the
    /// path's `to_string_lossy()` form (UTF-8 is the only column type SQLite has). Returns the
    /// number of rows updated (0 if the session id doesn't exist).
    ///
    /// Apply `permission`, `cwd` and `provider` updates in a single SQLite transaction so a DB
    /// failure between the writes can't leave a half-applied state on disk.  Any of them may be
    /// `None` to skip that field.  `updated_at` is recomputed inside the transaction so the
    /// timestamp matches the commit, not the call.  The individual `update_session_cwd` /
    /// `update_session_permission` / `set_recorded_provider` methods remain for callers that only
    /// need a single-column write.
    pub async fn update_session_metadata_atomic(
        &self,
        session_id: Uuid,
        new_permission: Option<String>,
        new_cwd: Option<std::path::PathBuf>,
        // A profile name, which is the whole binding: nothing else about a provider is recorded on
        // a session, so there is no second field that could be left behind pointing somewhere
        // else.
        new_provider: Option<String>,
    ) -> Result<()> {
        if new_permission.is_none() && new_cwd.is_none() && new_provider.is_none() {
            return Ok(());
        }
        let now = chrono::Utc::now().to_rfc3339();
        let id_string = session_id.to_string();
        let cwd_string = new_cwd.map(|cwd| cwd.to_string_lossy().into_owned());
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let txn = connection.transaction()?;
                if let Some(ref permission) = new_permission {
                    txn.execute(
                        "UPDATE sessions SET permission = ?1, updated_at = ?2 WHERE id = ?3",
                        rusqlite::params![permission, now, id_string],
                    )?;
                }
                if let Some(ref cwd) = cwd_string {
                    txn.execute(
                        "UPDATE sessions SET cwd = ?1, updated_at = ?2 WHERE id = ?3",
                        rusqlite::params![cwd, now, id_string],
                    )?;
                }
                if let Some(ref provider) = new_provider {
                    txn.execute(
                        "UPDATE sessions SET provider = ?1, updated_at = ?2 WHERE id = ?3",
                        rusqlite::params![provider, now, id_string],
                    )?;
                }
                txn.commit()?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!(
                    "failed to update session metadata atomically: {}",
                    error
                ))
            })
    }

    pub async fn update_session_cwd(
        &self,
        session_id: Uuid,
        cwd: &std::path::Path,
    ) -> Result<usize> {
        let cwd_string = cwd.to_string_lossy().into_owned();
        let id_string = session_id.to_string();
        // Bump `updated_at` alongside the target column so a re-attach after GC eviction
        // sees the post-PATCH timestamp instead of regressing to the stale pre-PATCH
        // value.
        let updated_at = chrono::Utc::now().to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE sessions SET cwd = ?1, updated_at = ?2 WHERE id = ?3",
                    rusqlite::params![cwd_string, updated_at, id_string],
                )
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to update session cwd: {}", error))
            })
    }

    /// Update the persisted permission level for a session. Called by the HTTP API's
    /// `PATCH /v1/sessions/{id}` so a permission flip survives GC-eviction + re-attach. Returns
    /// the number of rows updated (0 if the session id doesn't exist).
    pub async fn update_session_permission(
        &self,
        session_id: Uuid,
        permission: &str,
    ) -> Result<usize> {
        let permission = permission.to_string();
        let id_string = session_id.to_string();
        // Bump `updated_at` alongside the target column. See `update_session_cwd`.
        let updated_at = chrono::Utc::now().to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE sessions SET permission = ?1, updated_at = ?2 WHERE id = ?3",
                    rusqlite::params![permission, updated_at, id_string],
                )
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to update session permission: {}", error))
            })
    }

    /// Update the persisted capabilities JSON blob for a session. Symmetric counterpart to
    /// [`Self::update_session_permission`]. The blob's internal shape isn't validated here; the
    /// blob is expected to be a serialised `SessionCapabilities` value; nothing writes it yet.
    #[allow(
        dead_code,
        reason = "wired for future PATCH support; capability flips are rare"
    )]
    pub async fn update_session_capabilities(
        &self,
        session_id: Uuid,
        capabilities_json: &str,
    ) -> Result<usize> {
        let capabilities_json = capabilities_json.to_string();
        let id_string = session_id.to_string();
        let updated_at = chrono::Utc::now().to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE sessions SET capabilities_json = ?1, updated_at = ?2 WHERE id = ?3",
                    rusqlite::params![capabilities_json, updated_at, id_string],
                )
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to update session capabilities: {}", error))
            })
    }

    /// Replace a session's workspace roots (the ACP `additionalDirectories` list).
    ///
    /// Replace rather than merge, and an empty `roots` clears the column: per the ACP spec a
    /// non-empty list "is the complete resulting additional-root list", while omitted or empty
    /// means no additional roots are activated. The stored value is therefore the last activated
    /// set, which is exactly what `session/list` should report. A client reopening a session from a
    /// window that no longer has the second folder correctly narrows it.
    ///
    /// Unlike the sibling updaters this does *not* touch `updated_at`: activating roots is a
    /// property of how a session is being opened, not conversation activity, and bumping the
    /// timestamp would reorder the client's session list on every load.
    pub async fn update_session_roots(
        &self,
        session_id: Uuid,
        roots: &[std::path::PathBuf],
    ) -> Result<usize> {
        let encoded = encode_additional_roots(roots)?;
        let id_string = session_id.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE sessions SET additional_roots_json = ?1 WHERE id = ?2",
                    rusqlite::params![encoded, id_string],
                )
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to update session roots: {}", error))
            })
    }

    #[cfg(test)]
    pub async fn clear_messages(&self, session_id: Uuid) -> Result<()> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "DELETE FROM tool_outputs WHERE session_id = ?1",
                    rusqlite::params![session_id.to_string()],
                )?;

                connection.execute(
                    "DELETE FROM messages WHERE session_id = ?1",
                    rusqlite::params![session_id.to_string()],
                )?;

                connection.execute(
                    "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![chrono::Utc::now().to_rfc3339(), session_id.to_string()],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to clear messages: {}", error)))
    }

    /// Delete a session this caller already owns.
    ///
    /// Deliberately does *not* consult the session lock, because every caller here is holding it
    /// already or is acting on a row nothing can have locked: the HTTP handler evicting its own
    /// entry, the GC dropping a session it served, a sub-agent tool removing a child row it
    /// created moments ago. Taking the lock in those cases would *refuse* the caller its own
    /// session -- `flock` is per open file description rather than per process, so a second
    /// descriptor contends with the first -- and `try_write` is non-blocking, so what it produces
    /// is a spurious [`MekaError::SessionLocked`] rather than a hang.
    ///
    /// A caller acting on a session it has never met wants
    /// [`Self::delete_session_unless_attached`] instead.
    pub async fn delete_session(&self, session_id: Uuid) -> Result<bool> {
        let deleted = self.delete_session_row(session_id).await?;
        self.prune_orphan_lock_files().await;
        Ok(deleted)
    }

    /// The row half of a delete, without the lock-directory sweep, so a caller holding this
    /// session's lock can drop it before the sweep runs rather than blocking its own cleanup.
    async fn delete_session_row(&self, session_id: Uuid) -> Result<bool> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                // ON DELETE CASCADE on `messages.session_id`, `tool_outputs.session_id`, and
                // `sessions.parent_session_id` sweeps own-session rows + any sub-agent children +
                // their messages/tool_outputs in a single statement.
                let deleted = connection
                    .execute("DELETE FROM sessions WHERE id = ?1", rusqlite::params![
                        session_id.to_string()
                    ])?;
                Ok(deleted > 0)
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to delete session: {}", error)))
    }

    /// Delete a session, refusing with [`MekaError::SessionLocked`] if another meka process has it
    /// open. The door for a caller acting on a session it does not hold: `meka session delete`.
    ///
    /// `meka session delete <id>` against a live REPL used to exit 0 having said nothing at all --
    /// the count goes through `tracing::info!`, invisible at the default level -- while the row and
    /// its messages cascaded away underneath a conversation that carried on as if nothing had
    /// happened, until its next turn failed on a foreign-key violation.
    pub async fn delete_session_unless_attached(&self, session_id: Uuid) -> Result<bool> {
        let lock = self.lock_session(session_id)?;
        let deleted = self.delete_session_row(session_id).await?;
        // Released before the sweep: it will not unlink a file it cannot lock, and that file is
        // this one.
        drop(lock);
        self.prune_orphan_lock_files().await;
        Ok(deleted)
    }

    /// Delete every session no other meka process has open, and report what was left behind.
    pub async fn delete_all_sessions(&self) -> Result<SessionSweep> {
        let ids: Vec<Uuid> = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare("SELECT id FROM sessions")?;
                let ids = statement
                    .query_map([], |row| row.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<String>>>()?;
                Ok(ids)
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to list sessions: {}", error)))?
            .into_iter()
            .filter_map(|id| Uuid::parse_str(&id).ok())
            .collect();
        // No further condition: `--all` means every session nobody else has open, schedules
        // included. Sparing a job-owning session here would make the command quietly not mean what
        // it says.
        self.delete_the_unattached_among(&ids, "").await
    }

    pub async fn save_tool_output(
        &self,
        session_id: Uuid,
        name: &str,
        content: &str,
    ) -> Result<()> {
        let name = name.to_string();
        let content = content.to_string();
        let now = chrono::Utc::now().to_rfc3339();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT OR REPLACE INTO tool_outputs (session_id, name, content, created_at) \
                     VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![session_id.to_string(), name, content, now],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to save tool output: {}", error)))
    }

    pub async fn update_tool_output(
        &self,
        session_id: Uuid,
        name: &str,
        content: &str,
    ) -> Result<bool> {
        let name = name.to_string();
        let content = content.to_string();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let updated = connection.execute(
                    "UPDATE tool_outputs SET content = ?1 \
                     WHERE session_id = ?2 AND name = ?3",
                    rusqlite::params![content, session_id.to_string(), name],
                )?;
                Ok(updated > 0)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to update tool output: {}", error))
            })
    }

    pub async fn delete_tool_output(&self, session_id: Uuid, name: &str) -> Result<bool> {
        let name = name.to_string();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let deleted = connection.execute(
                    "DELETE FROM tool_outputs WHERE session_id = ?1 AND name = ?2",
                    rusqlite::params![session_id.to_string(), name],
                )?;
                Ok(deleted > 0)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to delete tool output: {}", error))
            })
    }

    pub async fn rename_tool_output(
        &self,
        session_id: Uuid,
        old: &str,
        new: &str,
    ) -> Result<RenameOutcome> {
        let old = old.to_string();
        let new = new.to_string();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                // Pre-check: target must not exist. `tokio_rusqlite` serializes connection access
                // so this and the UPDATE share a consistent view; the `PRIMARY KEY (session_id,
                // name)` constraint at the schema layer is the final backstop.
                let target_exists: i64 = connection.query_row(
                    "SELECT COUNT(*) FROM tool_outputs WHERE session_id = ?1 AND name = ?2",
                    rusqlite::params![session_id.to_string(), new],
                    |row| row.get(0),
                )?;
                if target_exists > 0 {
                    return Ok(RenameOutcome::TargetExists);
                }
                let renamed = connection.execute(
                    "UPDATE tool_outputs SET name = ?1 WHERE session_id = ?2 AND name = ?3",
                    rusqlite::params![new, session_id.to_string(), old],
                )?;
                Ok(if renamed > 0 {
                    RenameOutcome::Renamed
                } else {
                    RenameOutcome::NotFound
                })
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to rename tool output: {}", error))
            })
    }

    pub async fn list_tool_outputs(&self, session_id: Uuid) -> Result<Vec<ToolOutputSummary>> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(
                    "SELECT name, LENGTH(content), created_at \
                     FROM tool_outputs WHERE session_id = ?1 ORDER BY created_at ASC",
                )?;

                let rows = statement
                    .query_map(rusqlite::params![session_id.to_string()], |row| {
                        Ok(ToolOutputSummary {
                            name: row.get(0)?,
                            size: row.get::<_, i64>(1)? as usize,
                            created_at: row.get(2)?,
                        })
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;

                Ok(rows)
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to list tool outputs: {}", error)))
    }

    pub async fn load_tool_output(&self, session_id: Uuid, name: &str) -> Result<Option<String>> {
        let name = name.to_string();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let result = connection.query_row(
                    "SELECT content FROM tool_outputs \
                     WHERE session_id = ?1 AND name = ?2",
                    rusqlite::params![session_id.to_string(), name],
                    |row| row.get::<_, String>(0),
                );

                match result {
                    Ok(content) => Ok(Some(content)),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(error) => Err(error),
                }
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to load tool output: {}", error)))
    }

    pub async fn load_all_tool_outputs(&self, session_id: Uuid) -> Result<Vec<(String, String)>> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(
                    "SELECT name, content FROM tool_outputs \
                     WHERE session_id = ?1 ORDER BY created_at ASC",
                )?;

                let rows = statement
                    .query_map(rusqlite::params![session_id.to_string()], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;

                Ok(rows)
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to load tool outputs: {}", error)))
    }
}

/// Which kind of secret a `mcp_credentials` row holds, and therefore how its `secret` column is to
/// be read.
///
/// Stored rather than inferred from the value's shape. Sniffing would make every reader depend on
/// what rmcp's serialisation happens to look like this week, which is a fact about someone else's
/// system and would go stale with nothing to notice.
///
/// Three variants rather than "secret or not", because a server can hold two at once and they are
/// not interchangeable: a confidential `auth = "oauth"` client keeps its long-lived
/// [`Self::ClientSecret`] *and* the [`Self::OAuth`] bundle obtained with it, and a token refresh
/// must rewrite only the second. That is what `(server_name, kind)` keys the table on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpCredentialKind {
    /// A static token sent verbatim as `Authorization: Bearer <token>`. It *is* the credential:
    /// nothing is exchanged for it and meka never rewrites it.
    Bearer,
    /// The client half of an OAuth client's identity, presented to an authorization server to
    /// obtain an access token. Long-lived, and meka never rewrites it either.
    ClientSecret,
    /// An rmcp OAuth bundle, obtained by the authorization-code flow and refreshed in place by the
    /// adapter. The only kind meka itself rewrites.
    OAuth,
}

impl McpCredentialKind {
    /// The stored discriminator. Values are part of the schema, so they are written out here rather
    /// than derived from the variant names, which are free to be renamed.
    fn as_str(self) -> &'static str {
        match self {
            Self::Bearer => "bearer",
            Self::ClientSecret => "client_secret",
            Self::OAuth => "oauth",
        }
    }

    /// What to call this kind when telling the user about it. Deliberately not [`Self::as_str`]:
    /// that one is a schema value and must not drift to suit a sentence.
    pub fn label(self) -> &'static str {
        match self {
            Self::Bearer => "bearer token",
            Self::ClientSecret => "client secret",
            Self::OAuth => "OAuth tokens",
        }
    }
}

#[derive(Clone)]
pub struct TokenStore {
    connection: Arc<Connection>,
    /// Where the per-profile credential locks live, shared with the session locks because they are
    /// the same kind of thing: a claim one process makes on a name, released by the OS when it
    /// exits. See [`TokenStore::try_lock_provider_credential`].
    lock_dir: PathBuf,
    /// Keeps an in-memory store's lock directory alive for as long as any handle on that store is,
    /// exactly as [`SessionManager`] does. Without it a `TokenStore` outliving its manager would
    /// be locking files under a directory that had already been removed.
    _ephemeral_lock_dir: Option<Arc<EphemeralLockDir>>,
}

impl TokenStore {
    /// Try to take the lock that serialises one provider profile's credential rotation across
    /// processes. `None` means another meka is already refreshing that profile.
    ///
    /// Separate from the session locks beside it because the thing being protected is different: a
    /// session lock says who owns a conversation, this says who is allowed to spend a refresh
    /// token. Two processes refreshing the same profile both present the token the other is about
    /// to invalidate, and against an issuer with a reuse window both succeed -- leaving the
    /// database holding the *older* of the two, superseded, with the next launch getting
    /// `invalid_grant` and nothing naming why.
    ///
    /// The profile name is hashed rather than used directly: it is a TOML table key with no charset
    /// rule behind it, so `[providers."a/b"]` would otherwise name a path. A readable prefix is
    /// kept so an operator listing the directory can see what these files are.
    pub fn try_lock_provider_credential(&self, profile: &str) -> Result<Option<FileLock>> {
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(profile.as_bytes());
        let readable: String = profile
            .chars()
            .filter(|character| character.is_ascii_alphanumeric() || *character == '-')
            .take(24)
            .collect();
        let path = self.lock_dir.join(format!(
            "{}{}-{:x}.lock",
            PROVIDER_LOCK_PREFIX,
            readable,
            // The digest's leading four bytes: injective on `u32`, so two profiles never share a
            // file, and short enough that the name stays readable. Rendered by `{:x}`, which drops
            // leading zeros, so this is up to eight hex digits rather than always eight.
            u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]])
        ));
        try_lock_file(&path)
    }

    /// Load the stored credential (API key or OAuth bundle) for a provider profile, keyed by the
    /// user-chosen profile name. The credential is stored as a serialized [`AuthCredential`].
    pub async fn load_provider_credential(&self, profile: &str) -> Result<Option<AuthCredential>> {
        let profile = profile.to_string();
        let json: Option<String> = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                let result = connection.query_row(
                    "SELECT credentials_json FROM provider_credentials WHERE profile = ?1",
                    rusqlite::params![profile],
                    |row| row.get::<_, String>(0),
                );
                match result {
                    Ok(json) => Ok(Some(json)),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(error) => Err(error),
                }
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to load provider credential: {}", error))
            })?;

        match json {
            Some(json) => {
                let credential = serde_json::from_str(&json).map_err(|error| {
                    MekaError::Database(format!(
                        "failed to parse stored provider credential: {}",
                        error
                    ))
                })?;
                Ok(Some(credential))
            }
            None => Ok(None),
        }
    }

    /// A value that changes whenever a profile's stored credential does, without reading the
    /// credential.
    ///
    /// For a holder that built something *from* a credential and needs to know whether the thing it
    /// built is still the right one. [`crate::provider::ProviderRegistry`] is the caller: it keeps
    /// a built provider for reuse, and the writer that supersedes the credential is usually
    /// another process (`meka provider login` while `meka serve` runs), so a push cannot reach
    /// it. Comparing this against the value the provider was built from is the whole check.
    ///
    /// `None` for a profile with no stored credential, which is a distinct answer from any version:
    /// a memo built while one existed must not survive `meka provider remove`.
    ///
    /// The row's `updated_at` rather than a digest of the credential, because the point is to avoid
    /// pulling a secret into memory for a comparison. Every writer here stamps it
    /// ([`Self::save_provider_credential`], [`Self::replace_provider_credential`]), so a write the
    /// value of which happens to be unchanged still moves it, and the cost is a rebuild rather than
    /// a wrong answer.
    pub async fn provider_credential_version(&self, profile: &str) -> Result<Option<String>> {
        let profile = profile.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let result = connection.query_row(
                    "SELECT updated_at FROM provider_credentials WHERE profile = ?1",
                    rusqlite::params![profile],
                    |row| row.get::<_, String>(0),
                );
                match result {
                    Ok(version) => Ok(Some(version)),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(error) => Err(error),
                }
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!(
                    "failed to read provider credential version: {}",
                    error
                ))
            })
    }

    /// Replace a provider's credential only if the stored one is still what the caller derived its
    /// new value from.
    ///
    /// This is the door a *refresh* goes through, and the reason it is a compare-and-swap is that a
    /// refresh is not an assignment: it is a value computed from the old one, over a network round
    /// trip long enough for something else to have written. Two of those somethings are real. Two
    /// meka processes refreshing at once both present the same refresh token, and against an issuer
    /// with a reuse window *both* succeed -- so the blind upsert left the database holding
    /// whichever finished last, which is the token the issuer has already superseded. The next
    /// launch got `invalid_grant` with nothing naming why. And a `meka provider login`
    /// completing during a slow refresh was simply overwritten, silently, by a credential
    /// minted before it.
    ///
    /// Returns [`CredentialWrite::Superseded`] with what the row holds now, so the caller can
    /// decide whether to switch to it. Newer in write order is not the same as usable; see
    /// `provider::is_worth_adopting`.
    ///
    /// The comparison is on the stored JSON rather than a version column, for the same reason the
    /// memory store's body write is: every writer serialises through the same `serde` impl, so
    /// re-serialising a value read back out of the column reproduces its bytes exactly.
    pub async fn replace_provider_credential(
        &self,
        profile: &str,
        expected: &AuthCredential,
        credential: &AuthCredential,
    ) -> Result<CredentialWrite> {
        let expected_json = serde_json::to_string(expected).map_err(|error| {
            MekaError::Database(format!(
                "failed to serialize provider credential: {}",
                error
            ))
        })?;
        let json = serde_json::to_string(credential).map_err(|error| {
            MekaError::Database(format!(
                "failed to serialize provider credential: {}",
                error
            ))
        })?;
        let now = chrono::Utc::now().to_rfc3339();
        let profile_for_db = profile.to_string();
        let changed = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE provider_credentials SET credentials_json = ?3, updated_at = ?4 \
                     WHERE profile = ?1 AND credentials_json = ?2",
                    rusqlite::params![profile_for_db, expected_json, json, now],
                )
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to save provider credential: {}", error))
            })?;
        if changed == 1 {
            return Ok(CredentialWrite::Stored);
        }
        // Zero rows means the row moved, or that there is no row at all -- a profile whose
        // credential was deleted mid-refresh. Both are "somebody else decided what this profile
        // holds", and in neither case may a token minted from a superseded one be written back.
        match self.load_provider_credential(profile).await? {
            Some(current) => Ok(CredentialWrite::Superseded(Box::new(current))),
            None => Ok(CredentialWrite::Gone),
        }
    }

    /// Persist (or replace) the credential for a provider profile, keyed by profile name.
    ///
    /// The unconditional door, for a caller whose credential is not derived from a stored one: a
    /// fresh `meka provider login` or `provider add`, where overwriting whatever is there is the
    /// point. A refresh wants [`Self::replace_provider_credential`].
    pub async fn save_provider_credential(
        &self,
        profile: &str,
        credential: &AuthCredential,
    ) -> Result<()> {
        let profile = profile.to_string();
        let json = serde_json::to_string(credential).map_err(|error| {
            MekaError::Database(format!(
                "failed to serialize provider credential: {}",
                error
            ))
        })?;
        let now = chrono::Utc::now().to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO provider_credentials (profile, credentials_json, updated_at) \
                     VALUES (?1, ?2, ?3) \
                     ON CONFLICT(profile) DO UPDATE SET \
                         credentials_json = excluded.credentials_json, \
                         updated_at = excluded.updated_at",
                    rusqlite::params![profile, json, now],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to save provider credential: {}", error))
            })
    }

    /// Remove the stored credential for a provider profile (used by `provider remove`).
    pub async fn delete_provider_credential(&self, profile: &str) -> Result<()> {
        let profile = profile.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "DELETE FROM provider_credentials WHERE profile = ?1",
                    rusqlite::params![profile],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to delete provider credential: {}", error))
            })
    }

    /// Every profile name that has a stored credential, sorted.
    ///
    /// Credentials are keyed by profile name and nothing enforces that the name still exists in
    /// `config.toml`: deleting a `[providers.<name>]` block by hand leaves its API key or OAuth
    /// refresh token here indefinitely. Without this query the leftover cannot be named, so it
    /// cannot be reported or removed; `meka provider list` diffs the result against the configured
    /// profiles.
    pub async fn list_credential_profiles(&self) -> Result<Vec<String>> {
        self.connection
            .call(|connection| -> rusqlite::Result<_> {
                let mut statement = connection
                    .prepare("SELECT profile FROM provider_credentials ORDER BY profile")?;
                let profiles = statement
                    .query_map([], |row| row.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<String>>>()?;
                Ok(profiles)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to list provider credentials: {}", error))
            })
    }

    /// One server's stored secret of `kind`, or `None`.
    ///
    /// Matching on `kind` as well as the name is what makes "this server has no bearer" and "this
    /// server has an OAuth bundle" different answers, rather than handing an OAuth blob to a caller
    /// that would read it as a token.
    pub async fn load_mcp_credentials(
        &self,
        server_name: &str,
        kind: McpCredentialKind,
    ) -> Result<Option<String>> {
        let server_name = server_name.to_string();
        let kind = kind.as_str();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let result = connection.query_row(
                    "SELECT secret FROM mcp_credentials WHERE server_name = ?1 AND kind = ?2",
                    rusqlite::params![server_name, kind],
                    |row| row.get::<_, String>(0),
                );

                match result {
                    Ok(secret) => Ok(Some(secret)),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(error) => Err(error),
                }
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to load MCP credentials: {}", error))
            })
    }

    /// Replace an MCP server's credentials only if the stored ones are still the ones the caller
    /// last read. Returns whether the write landed.
    ///
    /// The same compare-and-swap as [`Self::replace_provider_credential`], for the same reason: two
    /// meka processes refreshing one server's OAuth token both write, and a blind upsert leaves the
    /// database holding whichever finished last rather than the one the issuer considers current.
    /// It arbitrates less here because rmcp owns the refresh and hands this adapter only a
    /// `load`/`save` pair, so the losing process keeps using its own token for the rest of its run.
    /// What it does guarantee is that the *stored* credential is never moved backwards, which is
    /// what the next process to start will load.
    ///
    /// Scoped to the `oauth` row because that is the only kind anything refreshes. A bearer and a
    /// client secret are written once by the user and read thereafter, so a compare-and-swap over
    /// them would arbitrate a race that cannot happen.
    pub async fn replace_mcp_credentials(
        &self,
        server_name: &str,
        expected_json: &str,
        json: &str,
    ) -> Result<bool> {
        let server_name = server_name.to_string();
        let expected_json = expected_json.to_string();
        let json = json.to_string();
        let now = chrono::Utc::now().to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE mcp_credentials SET secret = ?3, updated_at = ?4 \
                     WHERE server_name = ?1 AND secret = ?2 AND kind = 'oauth'",
                    rusqlite::params![server_name, expected_json, json, now],
                )
            })
            .await
            .map(|changed| changed == 1)
            .map_err(|error| {
                MekaError::Database(format!("failed to save MCP credentials: {}", error))
            })
    }

    /// Persist (or replace) one of an MCP server's secrets unconditionally, for a caller whose
    /// value is not derived from a stored one: `meka mcp add` / `login` reading from stdin, and the
    /// interactive authorisation flow depositing its first bundle.
    ///
    /// The conflict target is the whole key, so writing one kind leaves the server's other kinds
    /// untouched. That is what lets a confidential client hold its client secret while its OAuth
    /// bundle is replaced.
    pub async fn save_mcp_credentials(
        &self,
        server_name: &str,
        kind: McpCredentialKind,
        secret: &str,
    ) -> Result<()> {
        let server_name = server_name.to_string();
        let kind = kind.as_str();
        let secret = secret.to_string();
        let now = chrono::Utc::now().to_rfc3339();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO mcp_credentials (server_name, kind, secret, updated_at)
                     VALUES (?1, ?2, ?3, ?4)
                     ON CONFLICT(server_name, kind) DO UPDATE SET
                         secret = excluded.secret,
                         updated_at = excluded.updated_at",
                    rusqlite::params![server_name, kind, secret, now],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to save MCP credentials: {}", error))
            })
    }

    /// Drop just one of a server's secrets.
    ///
    /// `meka mcp login` clears the OAuth row before running the flow, so a stale bundle cannot be
    /// picked up mid-authorisation. It must leave the other kinds alone: a confidential client's
    /// `client_secret` is an *input* to the flow it is about to run, and clearing everything would
    /// delete the credential the login needs to succeed.
    pub async fn clear_mcp_credentials_of_kind(
        &self,
        server_name: &str,
        kind: McpCredentialKind,
    ) -> Result<()> {
        let server_name = server_name.to_string();
        let kind = kind.as_str();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "DELETE FROM mcp_credentials WHERE server_name = ?1 AND kind = ?2",
                    rusqlite::params![server_name, kind],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to clear MCP credentials: {}", error))
            })
    }

    /// Drop every secret this server has, whatever kind. `meka mcp logout` and `remove`.
    pub async fn clear_mcp_credentials(&self, server_name: &str) -> Result<()> {
        let server_name = server_name.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "DELETE FROM mcp_credentials WHERE server_name = ?1",
                    rusqlite::params![server_name],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to clear MCP credentials: {}", error))
            })
    }

    /// Whether this server has a stored credential at all, whatever kind.
    ///
    /// `meka mcp remove` asks this rather than loading, because it is about to delete every kind
    /// and only needs to know there is something to delete. Loading with a guessed kind would make
    /// a server that authenticates the other way look like a name that does not exist.
    pub async fn has_mcp_credentials(&self, server_name: &str) -> Result<bool> {
        let server_name = server_name.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let count: i64 = connection.query_row(
                    "SELECT COUNT(*) FROM mcp_credentials WHERE server_name = ?1",
                    rusqlite::params![server_name],
                    |row| row.get(0),
                )?;
                Ok(count > 0)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to look up MCP credentials: {}", error))
            })
    }

    /// Every MCP server name that has a stored credential of any kind, sorted.
    ///
    /// The counterpart to [`Self::list_credential_profiles`], and stale for the same reason:
    /// deleting an `[[mcp.servers]]` entry by hand strands its secret here. `meka mcp list` diffs
    /// the result against the configured servers.
    ///
    /// Kind-agnostic on purpose: an orphaned bearer is exactly as much of a leak as an orphaned
    /// OAuth bundle, and the report exists to name what is still lying around.
    pub async fn list_mcp_credential_servers(&self) -> Result<Vec<String>> {
        self.connection
            .call(|connection| -> rusqlite::Result<_> {
                // DISTINCT because the table is keyed by `(server_name, kind)`: a confidential
                // OAuth client holds two rows and a bearer beside them would make three, and this
                // answers "which servers have a secret", not "how many secrets are there".
                let mut statement = connection.prepare(
                    "SELECT DISTINCT server_name FROM mcp_credentials ORDER BY server_name",
                )?;
                let servers = statement
                    .query_map([], |row| row.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<String>>>()?;
                Ok(servers)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to list MCP credentials: {}", error))
            })
    }
}

impl SessionManager {
    pub fn token_store(&self) -> TokenStore {
        TokenStore {
            connection: Arc::clone(&self.connection),
            lock_dir: self.lock_dir.clone(),
            _ephemeral_lock_dir: self._ephemeral_lock_dir.clone(),
        }
    }

    pub fn schedule_store(&self) -> crate::schedule::ScheduleStore {
        crate::schedule::ScheduleStore::new(Arc::clone(&self.connection))
    }

    /// Handle on the memory store. See [`crate::memory::store`] for why it shares this database
    /// rather than owning one: meka has one database, and a second would be a new thing to back
    /// up, lock and explain for the sake of two tables.
    pub fn memory_store(&self, enabled: bool) -> Arc<crate::memory::MemoryStore> {
        crate::memory::MemoryStore::from_connection(Arc::clone(&self.connection), enabled)
    }

    pub fn background_store(&self) -> crate::background::BackgroundStore {
        crate::background::BackgroundStore::new(Arc::clone(&self.connection))
    }
}

/// Strip `<context>...</context>` tags from a stored user message, returning only the actual user
/// input.
pub fn strip_context_tags(text: &str) -> &str {
    const CLOSING_TAG: &str = "</context>";
    if let Some(end) = text.find(CLOSING_TAG) {
        let after = &text[end + CLOSING_TAG.len()..];
        after.trim_start_matches('\n')
    } else {
        text
    }
}

/// Pseudo-role written to the `messages` table's `role` column for `Event::CompactBoundary` rows,
/// distinct from every role an `Event::Append` uses.
const COMPACT_BOUNDARY_ROLE: &str = "compact_boundary";

/// Pseudo-role for a `Role::User` message that carries non-text blocks (input images). Its full
/// `Vec<ContentBlock>` is stored as JSON, because flattening to `text_content()` (as the plain
/// `user` role does) would silently drop the images. Text-only user turns stay plaintext under
/// `user` so `list_sessions`'s raw-content preview subquery keeps working. A `role`-column
/// pseudo-role, mirroring [`COMPACT_BOUNDARY_ROLE`].
const USER_BLOCKS_ROLE: &str = "user_blocks";

/// Pseudo-role for `Event::Repair` rows, mirroring [`COMPACT_BOUNDARY_ROLE`]. The superseded
/// messages keep their own rows, so `meka session export` still shows what was replaced.
const REPAIR_ROLE: &str = "repair";

/// Encode an [`crate::conversation::Event`] into the `(role, content)` columns of the `messages`
/// table. `Event::Append` writes the message's natural role; `Event::CompactBoundary` and
/// `Event::Repair` write a JSON envelope under their pseudo-role.
fn encode_event_for_db(
    event: &crate::conversation::Event,
) -> std::result::Result<(String, String), serde_json::Error> {
    use crate::{
        conversation::Event,
        provider::{ContentBlock, Role},
    };

    match event {
        Event::Append(message) => {
            let (role, content) = match message.role {
                Role::User => {
                    if message
                        .content
                        .iter()
                        .any(|block| matches!(block, ContentBlock::ToolResult { .. }))
                    {
                        ("tool_results", serde_json::to_string(&message.content)?)
                    } else if message
                        .content
                        .iter()
                        .any(|block| !matches!(block, ContentBlock::Text { .. }))
                    {
                        // A user turn carrying non-text blocks (input images) can't be flattened to
                        // plain text without losing them, so persist the full block list as JSON.
                        (USER_BLOCKS_ROLE, serde_json::to_string(&message.content)?)
                    } else {
                        ("user", message.text_content())
                    }
                }
                Role::Assistant => ("assistant", serde_json::to_string(&message.content)?),
            };
            Ok((role.to_string(), content))
        }
        Event::CompactBoundary { .. } => {
            let content = serde_json::to_string(event)?;
            Ok((COMPACT_BOUNDARY_ROLE.to_string(), content))
        }
        Event::Repair { .. } => {
            let content = serde_json::to_string(event)?;
            Ok((REPAIR_ROLE.to_string(), content))
        }
    }
}

/// Decode one persisted row back into an [`crate::conversation::Event`]. Returns `Ok(None)` when
/// the row's role is unrecognised (forward- compat for new variants).
fn decode_event_from_row(
    row: &StoredMessage,
) -> std::result::Result<Option<crate::conversation::Event>, serde_json::Error> {
    use crate::{
        conversation::Event,
        provider::{ContentBlock, Message, Role},
    };

    match row.role.as_str() {
        "user" => Ok(Some(Event::Append(Message::user(&row.content)))),
        "assistant" => match serde_json::from_str::<Vec<ContentBlock>>(&row.content) {
            Ok(content) => Ok(Some(Event::Append(Message {
                role: Role::Assistant,
                content,
            }))),
            Err(_) => {
                // Deliberately softer than the `tool_results` and `user_blocks` arms below, which
                // drop an unparseable row. Corruption here costs the turn's `tool_use` blocks, and
                // dropping the row entirely would take the assistant's prose with them. Keeping the
                // raw string as text leaves the transcript readable and the turn boundary intact.
                Ok(Some(Event::Append(Message::assistant_text(&row.content))))
            }
        },
        "tool_results" => match serde_json::from_str::<Vec<ContentBlock>>(&row.content) {
            Ok(content) => Ok(Some(Event::Append(Message {
                role: Role::User,
                content,
            }))),
            Err(error) => Err(error),
        },
        role if role == USER_BLOCKS_ROLE => {
            match serde_json::from_str::<Vec<ContentBlock>>(&row.content) {
                Ok(content) => Ok(Some(Event::Append(Message {
                    role: Role::User,
                    content,
                }))),
                Err(error) => Err(error),
            }
        }
        role if role == COMPACT_BOUNDARY_ROLE || role == REPAIR_ROLE => {
            let event: Event = serde_json::from_str(&row.content)?;
            Ok(Some(event))
        }
        _ => Ok(None),
    }
}

/// Pagination cursor for [`SessionManager::list_sessions`]: encodes the `(updated_at, id)` of the
/// last row in a page as base64-url JSON. The shape is opaque to clients; they only round-trip it
/// back as `next_cursor`.
#[derive(Serialize, Deserialize)]
struct ListSessionsCursor {
    #[serde(rename = "u")]
    updated_at: String,
    #[serde(rename = "i")]
    id: String,
}

fn encode_list_cursor(updated_at: &str, id: &str) -> String {
    use base64::Engine;
    let payload = ListSessionsCursor {
        updated_at: updated_at.to_string(),
        id: id.to_string(),
    };
    // `ListSessionsCursor` is two owned `String`s; `serde_json::to_vec` on a struct of plain
    // strings cannot fail. The `expect` documents the invariant.
    #[allow(clippy::expect_used)]
    let json = serde_json::to_vec(&payload)
        .expect("ListSessionsCursor is two owned Strings; serialization cannot fail");
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json)
}

fn decode_list_cursor(token: &str) -> Result<(String, String)> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(token)
        .map_err(|error| MekaError::Database(format!("invalid list cursor: {}", error)))?;
    let cursor: ListSessionsCursor = serde_json::from_slice(&bytes)
        .map_err(|error| MekaError::Database(format!("invalid list cursor: {}", error)))?;
    Ok((cursor.updated_at, cursor.id))
}

/// Encode an additional-root list for the `additional_roots_json` column. `None` for the empty case
/// keeps "no extra roots" as NULL, so the column has one representation for one meaning rather than
/// NULL and `[]` both appearing.
fn encode_additional_roots(roots: &[PathBuf]) -> Result<Option<String>> {
    if roots.is_empty() {
        return Ok(None);
    }
    serde_json::to_string(roots).map(Some).map_err(|error| {
        MekaError::Database(format!("failed to encode additional roots: {}", error))
    })
}

/// Decode the persisted `additional_roots_json` column into workspace roots.
///
/// NULL (every session that never carried extra roots) and unparseable JSON both yield an empty
/// list. Failing soft is right here: a session whose root list can't be read is still perfectly
/// usable as a single-root session, and refusing to load it would be a far worse outcome than
/// silently narrowing its search scope.
fn decode_additional_roots(json: Option<&str>) -> Vec<PathBuf> {
    json.and_then(|raw| serde_json::from_str::<Vec<PathBuf>>(raw).ok())
        .unwrap_or_default()
}

/// Derive a short, single-line preview from a stored user message: strip the agent's `<context>`
/// preamble, take the first line, and cap it at `max_chars` (appending `…` when cut). Used both for
/// the `session/list` preview column and the ACP live session title.
pub(crate) fn truncate_preview(text: &str, max_chars: usize) -> String {
    let text = strip_context_tags(text);
    let first_line = text.lines().next().unwrap_or("");
    if first_line.chars().count() <= max_chars {
        first_line.to_string()
    } else {
        let truncated: String = first_line.chars().take(max_chars).collect();
        format!("{}…", truncated.trim_end())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn test_manager() -> SessionManager {
        SessionManager::open(Some(Path::new(":memory:")), &Default::default())
            .await
            .expect("failed to open in-memory database")
    }

    /// The regression this whole arrangement exists for: a session created on one profile must
    /// still name that profile afterwards, so resuming it does not silently move the conversation.
    #[tokio::test]
    async fn a_session_keeps_the_provider_it_was_created_on() {
        let manager = test_manager().await;
        let id = manager
            .create_session(None, "openaiprof".to_string())
            .await
            .expect("create");

        assert_eq!(
            manager.recorded_provider(id).await.expect("read"),
            Some("openaiprof".to_string())
        );
    }

    /// Repinning is what every "change the provider" surface lands on, and it has to stick.
    #[tokio::test]
    async fn a_session_can_be_moved_to_another_provider() {
        let manager = test_manager().await;
        let id = manager
            .create_session(None, "openaiprof".to_string())
            .await
            .expect("create");

        assert!(
            manager
                .set_recorded_provider(id, "claudeprof")
                .await
                .expect("repin"),
            "the row was there to update"
        );
        assert_eq!(
            manager.recorded_provider(id).await.expect("read"),
            Some("claudeprof".to_string())
        );
    }

    /// A session that is gone reads as absent rather than as a profile named nothing, so a caller
    /// can tell "no such session" from "this session runs on ''".
    #[tokio::test]
    async fn a_missing_session_has_no_recorded_provider() {
        let manager = test_manager().await;
        assert_eq!(
            manager
                .recorded_provider(Uuid::new_v4())
                .await
                .expect("read"),
            None
        );
        assert!(
            !manager
                .set_recorded_provider(Uuid::new_v4(), "anything")
                .await
                .expect("update"),
            "a repin that matched no row says so rather than reporting success"
        );
    }

    /// A child records the profile it was given, not the one on its parent's row.
    ///
    /// The two are the same on every ordinary path, and this used to be a `SELECT … provider FROM
    /// sessions` that made them the same by construction. They come apart for exactly as long as a
    /// repin that could not take the runtime lock, which is what ACP's `session/set_config_option`
    /// does mid-turn: the row moves, the agent does not, and a worker spawned in that window ran on
    /// one account while its row claimed the other. Whoever calls this owes it the profile the
    /// worker is actually built on; `agent_spawn` reads both from the same cell.
    ///
    /// Asserted with the two deliberately different, because passing the parent's own profile
    /// cannot tell the new behaviour from the old.
    #[tokio::test]
    async fn a_child_session_records_the_profile_it_was_given_not_its_parents_row() {
        let manager = test_manager().await;
        let parent = manager
            .create_session(None, "openaiprof".to_string())
            .await
            .expect("parent");
        let (child, _lock) = manager
            .create_child_session(parent, None, None, "repinned-midturn".to_string())
            .await
            .expect("child");

        assert_eq!(
            manager.recorded_provider(child).await.expect("read"),
            Some("repinned-midturn".to_string()),
            "the caller's profile, which is the one the worker runs on"
        );
        assert_eq!(
            manager.recorded_provider(parent).await.expect("read"),
            Some("openaiprof".to_string()),
            "and the parent's own row is left alone"
        );
    }

    /// `MEKA_DATA_DIR` has to be absolute, for a sharper version of the reason `MEKA_CONFIG_DIR`
    /// does: `meka.db` holds every provider credential, so a relative value gives one credential
    /// store per directory meka is launched from, none of them the one the user set up.
    ///
    /// The comment justifying the config-directory hardening asserted this sibling "already refuses
    /// both". It refused only the empty value.
    #[tokio::test]
    async fn a_relative_data_dir_is_refused_rather_than_joined_to_the_cwd() {
        let _env = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        let previous = std::env::var_os("MEKA_DATA_DIR");
        // SAFETY: `MEKA_DATA_DIR` is process-global; the lock above serialises every test that
        // touches it, and the original value is restored before the guard drops.
        unsafe { std::env::set_var("MEKA_DATA_DIR", "relative/data") };
        let relative = default_database_path();

        let absolute_dir = std::env::temp_dir().join("meka-data-dir-test");
        unsafe { std::env::set_var("MEKA_DATA_DIR", &absolute_dir) };
        let absolute = default_database_path();

        unsafe {
            match previous {
                Some(value) => std::env::set_var("MEKA_DATA_DIR", value),
                None => std::env::remove_var("MEKA_DATA_DIR"),
            }
        }

        let relative = relative.expect("a rejected override still resolves a platform default");
        assert!(
            !relative.starts_with("relative"),
            "a relative override must not be joined to the cwd; got {}",
            relative.display(),
        );
        assert_eq!(
            absolute.expect("an absolute override is honoured"),
            absolute_dir.join("meka.db"),
            "and an absolute one is still used verbatim",
        );
    }

    /// Populate a session with the full spread of state a fork has to carry.
    async fn seeded_session(manager: &SessionManager) -> Uuid {
        use crate::{conversation::Event, provider::Message};

        let id = manager
            .create_session(
                Some(PathBuf::from("/work/main")),
                "test-profile".to_string(),
            )
            .await
            .expect("create session");
        // Enough alternating turns that a copy which reordered the conversation would show it
        // rather than coincidentally matching. This does not prove the `ORDER BY id` in
        // `fork_session` is load-bearing: SQLite's index scan yields rowid order anyway, so the
        // clause states the requirement rather than repairing a scramble.
        for turn in 0..20 {
            for event in [
                Event::Append(Message::user(format!("ask {}", turn))),
                Event::Append(Message::assistant_text(format!("reply {}", turn))),
            ] {
                manager.save_event(id, &event).await.expect("save event");
            }
        }
        manager
            .save_tool_output(id, "tool_1_output", "scratch")
            .await
            .expect("tool output");
        manager
            .update_session_roots(id, &[PathBuf::from("/work/shared")])
            .await
            .expect("roots");
        manager
            .save_session_stats(id, &crate::stats::SessionStatsSnapshot {
                turns: 3,
                input_tokens: 4242,
                ..Default::default()
            })
            .await
            .expect("stats");
        id
    }

    #[tokio::test]
    async fn fork_copies_the_conversation_and_leaves_the_source_alone() {
        let manager = test_manager().await;
        let source = seeded_session(&manager).await;

        let forked = manager
            .fork_session(source, ForkOverrides::default())
            .await
            .expect("fork")
            .expect("source exists");
        assert_ne!(forked.id, source);

        let copy_events = manager.load_events(forked.id).await.expect("copy events");
        let source_events = manager.load_events(source).await.expect("source events");
        assert_eq!(
            serde_json::to_string(&copy_events).expect("serialize copy"),
            serde_json::to_string(&source_events).expect("serialize source"),
            "the copy starts from the source's exact conversation"
        );
        assert_eq!(
            manager
                .load_all_tool_outputs(forked.id)
                .await
                .expect("copy outputs"),
            manager
                .load_all_tool_outputs(source)
                .await
                .expect("source outputs"),
            "scratchpad entries are referenced by name from tool inputs, so they must travel"
        );

        let copy = manager
            .session_info(forked.id)
            .await
            .expect("info")
            .expect("row");
        let original = manager
            .session_info(source)
            .await
            .expect("info")
            .expect("row");
        assert_eq!(copy.cwd, original.cwd);
        assert_eq!(copy.additional_roots, original.additional_roots);
        assert_eq!(copy.preview, original.preview);
        let copy_stats = manager
            .load_session_stats(forked.id)
            .await
            .expect("copy stats");
        assert_eq!(copy_stats.turns, 3);
        assert_eq!(copy_stats.input_tokens, 4242);
        assert!(
            copy.token_id.is_none(),
            "the bearer-token fingerprint is never inherited"
        );

        // The source is untouched: same event count, and still a listable top-level session.
        assert_eq!(
            manager.load_events(source).await.expect("source").len(),
            40,
            "forking must not mutate the source"
        );
    }

    /// Regression: retention GC deletes by `updated_at` and runs at every agent startup, so a fork
    /// that inherited a stale timestamp was swept before its first turn.
    #[tokio::test]
    async fn fork_stamps_fresh_timestamps_so_retention_gc_spares_it() {
        let manager = test_manager().await;
        let source = seeded_session(&manager).await;

        let stale = (chrono::Utc::now() - chrono::TimeDelta::days(100)).to_rfc3339();
        let source_string = source.to_string();
        manager
            .connection
            .call(move |connection| {
                connection.execute(
                    "UPDATE sessions SET created_at = ?1, updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![stale, source_string],
                )
            })
            .await
            .expect("age the source");

        let forked = manager
            .fork_session(source, ForkOverrides::default())
            .await
            .expect("fork")
            .expect("source exists");

        let deleted = manager
            .delete_expired_sessions(90)
            .await
            .expect("retention sweep");
        assert_eq!(deleted.deleted, 1, "only the stale source is swept");
        assert!(
            manager.session_exists(forked.id).await.expect("exists"),
            "the fork must survive the sweep that removes its stale source"
        );
    }

    /// A child links to its parent only through `parent_session_id`, and the sub-agent's *result*
    /// already sits in the parent's own event log, so the copy is self-contained without it.
    #[tokio::test]
    async fn fork_does_not_copy_sub_agent_children() {
        let manager = test_manager().await;
        let source = seeded_session(&manager).await;
        // Bound rather than discarded only because the tuple's second element is a `Result` that
        // must be used. Nothing here needs the child's lock: `fork_session` and `load_session_tree`
        // take none.
        let _child = manager
            .create_child_session(source, None, None, "test-profile".to_string())
            .await
            .expect("child");

        let forked = manager
            .fork_session(source, ForkOverrides::default())
            .await
            .expect("fork")
            .expect("source exists");

        let tree = manager.load_session_tree(forked.id).await.expect("tree");
        assert_eq!(tree.len(), 1, "the fork stands alone");
        assert_eq!(
            tree[0].parent_id, None,
            "a fork is top-level: `parent_session_id` means sub-agent parent, and list_sessions \
             hides rows that have one"
        );
        assert_eq!(
            manager
                .load_session_tree(source)
                .await
                .expect("source tree")
                .len(),
            2,
            "the source keeps its child"
        );
    }

    #[tokio::test]
    async fn fork_overrides_replace_the_workspace() {
        let manager = test_manager().await;
        let source = seeded_session(&manager).await;

        let forked = manager
            .fork_session(source, ForkOverrides {
                cwd: Some(PathBuf::from("/elsewhere")),
                additional_roots: Some(vec![PathBuf::from("/elsewhere/docs")]),
                token_id: Some("token-fingerprint".to_string()),
            })
            .await
            .expect("fork")
            .expect("source exists");

        let copy = manager
            .session_info(forked.id)
            .await
            .expect("info")
            .expect("row");
        assert_eq!(copy.cwd, Some(PathBuf::from("/elsewhere")));
        assert_eq!(copy.additional_roots, vec![PathBuf::from(
            "/elsewhere/docs"
        )]);
        assert_eq!(copy.token_id.as_deref(), Some("token-fingerprint"));
    }

    /// `Some(vec![])` means "activate no additional roots", which is what ACP's fork request sends
    /// when `additionalDirectories` is omitted. It must not be confused with "inherit": both encode
    /// to SQL NULL, so a `COALESCE` alone would resurrect the source's roots.
    #[tokio::test]
    async fn fork_can_override_additional_roots_to_empty() {
        let manager = test_manager().await;
        let source = seeded_session(&manager).await;

        let forked = manager
            .fork_session(source, ForkOverrides {
                additional_roots: Some(Vec::new()),
                ..Default::default()
            })
            .await
            .expect("fork")
            .expect("source exists");

        let copy = manager
            .session_info(forked.id)
            .await
            .expect("info")
            .expect("row");
        assert!(
            copy.additional_roots.is_empty(),
            "an explicit empty override must narrow the workspace, not inherit it"
        );
        assert_eq!(
            manager
                .session_info(source)
                .await
                .expect("info")
                .expect("row")
                .additional_roots,
            vec![PathBuf::from("/work/shared")],
            "and it must not disturb the source"
        );
    }

    /// Several clients forking one session at once must each get a whole, distinct copy. The copy
    /// spans three statements, so an interleaving that let a second fork observe the first's
    /// half-built session would show up as a short or empty event log.
    #[tokio::test]
    async fn concurrent_forks_of_one_source_each_get_a_complete_copy() {
        let manager = test_manager().await;
        let source = seeded_session(&manager).await;
        let expected = manager.load_events(source).await.expect("source").len();

        let mut handles = Vec::new();
        for _ in 0..8 {
            let manager = manager.clone();
            handles.push(tokio::spawn(async move {
                manager
                    .fork_session(source, ForkOverrides::default())
                    .await
                    .expect("fork")
                    .expect("source exists")
                    .id
            }));
        }
        let mut ids = Vec::new();
        for handle in handles {
            ids.push(handle.await.expect("join"));
        }

        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(
            unique.len(),
            ids.len(),
            "every fork must be its own session"
        );
        for id in &ids {
            assert_eq!(
                manager.load_events(*id).await.expect("copy").len(),
                expected,
                "every concurrent fork must carry the whole conversation"
            );
        }
    }

    /// A fork is a top-level session, not a child, so deleting what it was forked from must not
    /// take it with it. Had fork reused `parent_session_id` for lineage, the FK cascade would
    /// destroy every fork the moment its source was deleted.
    #[tokio::test]
    async fn deleting_the_source_leaves_the_fork_intact() {
        let manager = test_manager().await;
        let source = seeded_session(&manager).await;
        let expected = manager.load_events(source).await.expect("source").len();

        let forked = manager
            .fork_session(source, ForkOverrides::default())
            .await
            .expect("fork")
            .expect("source exists");
        assert!(manager.delete_session(source).await.expect("delete"));

        assert!(manager.session_exists(forked.id).await.expect("exists"));
        assert_eq!(
            manager.load_events(forked.id).await.expect("copy").len(),
            expected,
            "the fork keeps its own copy of the conversation"
        );
        assert_eq!(
            manager
                .load_all_tool_outputs(forked.id)
                .await
                .expect("outputs")
                .len(),
            1,
        );
    }

    /// Deleting a fork must sweep the rows it copied. `discard_failed_fork` in the ACP handler
    /// relies on this to undo a fork whose runtime wouldn't build; if the cascade missed, a failed
    /// fork would leave an orphaned transcript with no session row pointing at it.
    #[tokio::test]
    async fn deleting_a_fork_removes_the_rows_it_copied() {
        let manager = test_manager().await;
        let source = seeded_session(&manager).await;
        let forked = manager
            .fork_session(source, ForkOverrides::default())
            .await
            .expect("fork")
            .expect("source exists");

        assert!(manager.delete_session(forked.id).await.expect("delete"));
        assert!(
            manager
                .load_events(forked.id)
                .await
                .expect("events")
                .is_empty(),
            "the copied event log must go with the row"
        );
        assert!(
            manager
                .load_all_tool_outputs(forked.id)
                .await
                .expect("outputs")
                .is_empty(),
            "and so must the copied scratchpad entries"
        );
        // The source is untouched by the fork's deletion.
        assert!(
            !manager
                .load_events(source)
                .await
                .expect("source")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn fork_of_an_unknown_session_is_none() {
        let manager = test_manager().await;
        assert!(
            manager
                .fork_session(Uuid::new_v4(), ForkOverrides::default())
                .await
                .expect("fork")
                .is_none(),
            "callers map None to their own not-found shape rather than parsing an error"
        );
    }

    /// Drift guard for [`SessionManager::fork_session`], whose `INSERT ... SELECT` names every
    /// column explicitly. A new column silently omitted there would be dropped from every fork,
    /// which is exactly how `additional_roots` came to be lost by export/import. If this fails,
    /// decide whether the new column should be copied, reset, or overridden, then update both the
    /// fork statement and this list.
    #[tokio::test]
    async fn fork_copies_every_session_column() {
        let manager = test_manager().await;
        let columns = manager
            .connection
            .call(|connection| {
                let mut statement =
                    connection.prepare("SELECT name FROM pragma_table_info('sessions')")?;
                let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
                rows.collect::<rusqlite::Result<Vec<String>>>()
            })
            .await
            .expect("read schema");

        assert_eq!(columns, vec![
            "id",
            "created_at",
            "updated_at",
            "parent_session_id",
            "cwd",
            "permission",
            "capabilities_json",
            "token_id",
            "additional_roots_json",
            // Deliberately not copied by a fork; see `fork_session`'s doc comment.
            "subagent_spec_json",
            "stat_turns",
            "stat_input_tokens",
            "stat_output_tokens",
            "stat_cache_creation_input_tokens",
            "stat_cache_read_input_tokens",
            "stat_redactions",
            "stat_redacted_images",
            "stat_redacted_bytes",
            // Last, and after the stats, because a migration appended it and the fresh path
            // replays that same step rather than creating the column inline, so both
            // orders agree.
            //
            // Copied by a fork: a fork continues the same conversation and so has to keep running
            // on the same thing. Resetting it would make forking one more door that switches
            // provider without saying so.
            "provider",
        ]);
    }

    /// The sibling of `fork_copies_every_session_column`, for the door that had no such guard and
    /// was therefore the one that forgot: `import_sessions` wrote no profile at all, so every
    /// imported session landed on the empty profile no configuration can name, and the resume hint
    /// the command printed named a session that could not resume.
    #[tokio::test]
    async fn import_writes_the_sessions_profile() {
        let manager = test_manager().await;
        let id = Uuid::new_v4();
        manager
            .import_sessions(vec![ImportSessionRecord {
                new_id: id,
                new_parent_id: None,
                created_at: chrono::Utc::now().to_rfc3339(),
                cwd: None,
                permission: None,
                capabilities_json: None,
                additional_roots: Vec::new(),
                subagent_spec_json: None,
                provider: "work".to_string(),
                stats: crate::stats::SessionStatsSnapshot::default(),
                events: Vec::new(),
                tool_outputs: Vec::new(),
            }])
            .await
            .expect("import");

        assert_eq!(
            manager.recorded_provider(id).await.expect("read"),
            Some("work".to_string())
        );
    }

    /// A `PATCH /v1/sessions/{id}` naming a provider moves the row, in the same statement that
    /// carries permission and cwd, so a client that changed all three cannot end up with a session
    /// that took some of them.
    #[tokio::test]
    async fn moving_a_session_atomically_rewrites_its_profile() {
        let manager = test_manager().await;
        let created = manager
            .create_session(None, "alpha".to_string())
            .await
            .expect("create");

        manager
            .update_session_metadata_atomic(created, None, None, Some("beta".to_string()))
            .await
            .expect("move the session");

        assert_eq!(
            manager.recorded_provider(created).await.expect("read"),
            Some("beta".to_string())
        );
    }

    #[tokio::test]
    async fn sessions_on_a_provider_are_counted_for_the_removal_warning() {
        let manager = test_manager().await;
        for profile in ["work", "work", "side"] {
            manager
                .create_session(None, profile.to_string())
                .await
                .expect("create");
        }
        assert_eq!(
            manager
                .count_sessions_on_provider("work")
                .await
                .expect("count"),
            2
        );
        assert_eq!(
            manager
                .count_sessions_on_provider("gone")
                .await
                .expect("count"),
            0
        );
    }

    /// The count is what `meka provider remove` warns with, beside advice that only applies to a
    /// top-level session. A sub-agent row copies its parent's binding, so counting children made
    /// the warning cite a number many times what `meka session list` shows.
    #[tokio::test]
    async fn a_provider_count_leaves_out_the_workers_a_session_spawned() {
        let manager = test_manager().await;
        let parent = manager
            .create_session(None, "work".to_string())
            .await
            .expect("create parent");
        for _ in 0..3 {
            let (_id, lock) = manager
                .create_child_session(parent, None, None, "test-profile".to_string())
                .await
                .expect("spawn a worker");
            lock.expect("claim the worker's lock");
        }
        assert_eq!(
            manager
                .count_sessions_on_provider("work")
                .await
                .expect("count"),
            1,
            "three workers on the parent's profile are not three more sessions to move"
        );
    }

    /// A spawn against a parent that is gone must fail, not hand back an id with no row behind it.
    ///
    /// The statement is an `INSERT … SELECT` even though the child is now *told* its provider
    /// rather than copying it, and this is why: that form selects nothing and succeeds where the
    /// `VALUES` it replaced was refused by `parent_session_id`'s foreign key. Unchecked, the model
    /// is told about a worker
    /// that does not exist, a lock file is held for it, and the failure resurfaces as a raw
    /// constraint violation on the worker's first saved message.
    #[tokio::test]
    async fn spawning_from_a_session_that_is_gone_is_refused() {
        let manager = test_manager().await;
        let Err(error) = manager
            .create_child_session(Uuid::new_v4(), None, None, "test-profile".to_string())
            .await
        else {
            panic!("a missing parent must refuse the spawn");
        };
        assert!(
            error.to_string().contains("no longer exists"),
            "the refusal must name what went wrong: {error}"
        );
    }

    #[tokio::test]
    async fn additional_roots_round_trip_through_the_database() {
        let manager = test_manager().await;
        let id = manager
            .create_session(
                Some(PathBuf::from("/work/main")),
                "test-profile".to_string(),
            )
            .await
            .expect("create session");

        // A fresh session has none: the column starts NULL.
        let summary = manager.session_info(id).await.expect("info").expect("row");
        assert!(summary.additional_roots.is_empty());

        let roots = vec![PathBuf::from("/work/shared"), PathBuf::from("/work/docs")];
        manager
            .update_session_roots(id, &roots)
            .await
            .expect("store roots");
        let summary = manager.session_info(id).await.expect("info").expect("row");
        assert_eq!(summary.additional_roots, roots);

        // `session/list` reports the same set, since that is what a client rebuilds a workspace
        // from when picking a session out of its history.
        let (listed, _cursor) = manager
            .list_sessions(10, false, None, None)
            .await
            .expect("list");
        let row = listed
            .iter()
            .find(|row| row.id == id)
            .expect("session should be listed");
        assert_eq!(row.additional_roots, roots);
    }

    /// Load and resume carry the complete resulting list, so an empty one clears rather than
    /// merges: reopening a session from a window that no longer has the second folder has to
    /// narrow the session, not silently keep searching a folder the user removed.
    #[tokio::test]
    async fn empty_additional_roots_clears_the_stored_list() {
        let manager = test_manager().await;
        let id = manager
            .create_session(
                Some(PathBuf::from("/work/main")),
                "test-profile".to_string(),
            )
            .await
            .expect("create session");
        manager
            .update_session_roots(id, &[PathBuf::from("/work/shared")])
            .await
            .expect("store roots");

        manager
            .update_session_roots(id, &[])
            .await
            .expect("clear roots");

        let summary = manager.session_info(id).await.expect("info").expect("row");
        assert!(
            summary.additional_roots.is_empty(),
            "an empty list must clear, not merge"
        );
    }

    /// Unparseable JSON must not make a session unloadable. Like NULL it means a single root, which
    /// is what such a session is anyway.
    #[test]
    fn decode_additional_roots_fails_soft() {
        assert!(decode_additional_roots(None).is_empty());
        assert!(decode_additional_roots(Some("not json")).is_empty());
        assert_eq!(decode_additional_roots(Some(r#"["/a","/b"]"#)), vec![
            PathBuf::from("/a"),
            PathBuf::from("/b"),
        ]);
    }

    /// Regression: every in-memory `open()` mints a temp lock dir, and nothing used to remove it,
    /// so a few hundred `cargo test` runs left thousands of stray directories in the system temp
    /// dir.
    #[tokio::test]
    async fn in_memory_lock_dir_is_removed_when_the_last_clone_drops() {
        let manager = test_manager().await;
        let lock_dir = manager.lock_dir.clone();
        assert!(lock_dir.exists(), "open() should have created the lock dir");

        // A clone still holding it must keep the directory alive: sub-agents and tool builders
        // clone the manager, and a lock taken through one of them has to keep working.
        let clone = manager.clone();
        drop(manager);
        assert!(
            lock_dir.exists(),
            "the dir must outlive the first clone to drop"
        );

        drop(clone);
        assert!(
            !lock_dir.exists(),
            "the last clone dropping should remove '{}'",
            lock_dir.display()
        );
    }

    /// The on-disk counterpart must survive: its `locks/` directory lives beside the database and
    /// outlives the process.
    #[tokio::test]
    async fn on_disk_lock_dir_survives_the_manager() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let db_path = temp_dir.path().join("meka.db");
        let manager = SessionManager::open(Some(&db_path), &Default::default())
            .await
            .expect("open on-disk database");
        let lock_dir = manager.lock_dir.clone();
        drop(manager);
        assert!(
            lock_dir.exists(),
            "an on-disk lock dir must not be swept away with the manager"
        );
    }

    /// Persist one of every event variant via `save_event` and read it back through `load_events`.
    /// Verifies the encoding/decoding round trip, including the JSON envelope used for
    /// `CompactBoundary`, matches the in-memory shape.
    #[tokio::test]
    async fn test_save_and_load_events_round_trip() {
        use std::collections::HashSet;

        use crate::{
            conversation::Event,
            provider::{ContentBlock, Message, Role, ToolResultContent},
        };

        let manager = test_manager().await;
        let sid = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");

        let user_event = Event::Append(Message::user("hello"));
        let assistant_event = Event::Append(Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "thinking aloud".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "u1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/x"}),
                },
            ],
        });
        let tool_result_event = Event::Append(Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "u1".to_string(),
                content: vec![ToolResultContent::Text {
                    text: "ok".to_string(),
                }],
                is_error: false,
            }],
        });
        let snapshot: HashSet<String> = ["mcp__notion__fetch".to_string()].into_iter().collect();
        let boundary_event = Event::CompactBoundary {
            summary: Message::user("[summary]"),
            replaced_count: 3,
            loaded_tools_snapshot: snapshot,
        };

        let repair_event = Event::Repair {
            replaced_count: 2,
            messages: vec![Message::assistant_text("[degraded]")],
        };

        for event in [
            &user_event,
            &assistant_event,
            &tool_result_event,
            &boundary_event,
            &repair_event,
        ] {
            manager.save_event(sid, event).await.expect("save event");
        }

        let loaded = manager.load_events(sid).await.expect("load events");
        assert_eq!(loaded.len(), 5);

        match &loaded[0] {
            Event::Append(m) => assert_eq!(m.text_content(), "hello"),
            _ => panic!("expected user Append"),
        }
        match &loaded[1] {
            Event::Append(m) => {
                assert_eq!(m.role, Role::Assistant);
                assert_eq!(m.content.len(), 2);
                assert!(matches!(&m.content[1], ContentBlock::ToolUse { id, .. } if id == "u1"));
            }
            _ => panic!("expected assistant Append"),
        }
        match &loaded[2] {
            Event::Append(m) => {
                assert_eq!(m.role, Role::User);
                assert!(matches!(
                    &m.content[0],
                    ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == "u1"
                ));
            }
            _ => panic!("expected tool_results Append"),
        }
        match &loaded[3] {
            Event::CompactBoundary {
                replaced_count,
                loaded_tools_snapshot,
                summary,
            } => {
                assert_eq!(*replaced_count, 3);
                assert!(loaded_tools_snapshot.contains("mcp__notion__fetch"));
                assert_eq!(summary.text_content(), "[summary]");
            }
            _ => panic!("expected CompactBoundary"),
        }
        match &loaded[4] {
            Event::Repair {
                replaced_count,
                messages,
            } => {
                assert_eq!(*replaced_count, 2);
                assert_eq!(messages.len(), 1);
                assert_eq!(messages[0].text_content(), "[degraded]");
            }
            _ => panic!("expected Repair"),
        }
    }

    /// A user turn carrying an input image is persisted under the `user_blocks` role as full JSON
    /// so the image survives the round trip, while a text-only user turn still stores as plaintext
    /// under `user` (keeping `list_sessions`'s raw-content preview intact).
    #[tokio::test]
    async fn test_user_input_image_round_trips_via_user_blocks_role() {
        use crate::{
            conversation::Event,
            provider::{ContentBlock, ImageSource, Message, Role},
        };

        let manager = test_manager().await;
        let sid = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");

        let image = ImageSource {
            source_type: "base64".to_string(),
            media_type: "image/png".to_string(),
            data: "aGVsbG8=".to_string(),
        };
        let image_event = Event::Append(Message::user_with_images("look at this", vec![image]));
        let text_event = Event::Append(Message::user("plain text only"));
        manager
            .save_event(sid, &image_event)
            .await
            .expect("save image event");
        manager
            .save_event(sid, &text_event)
            .await
            .expect("save text event");

        // Storage roles: the image-bearing turn is JSON under `user_blocks`; the text-only turn
        // stays plaintext under `user`.
        let rows = manager.load_messages(sid).await.expect("load messages");
        assert_eq!(rows[0].role, "user_blocks");
        assert_eq!(rows[1].role, "user");
        assert_eq!(rows[1].content, "plain text only");

        // The image block survives the decode round trip.
        let loaded = manager.load_events(sid).await.expect("load events");
        match &loaded[0] {
            Event::Append(message) => {
                assert_eq!(message.role, Role::User);
                assert_eq!(message.content.len(), 2);
                assert!(matches!(
                    &message.content[0],
                    ContentBlock::Text { text } if text == "look at this"
                ));
                assert!(matches!(
                    &message.content[1],
                    ContentBlock::Image { source }
                        if source.data == "aGVsbG8=" && source.media_type == "image/png"
                ));
            }
            other => panic!("expected user Append, got {other:?}"),
        }
        match &loaded[1] {
            Event::Append(message) => assert_eq!(message.text_content(), "plain text only"),
            other => panic!("expected user Append, got {other:?}"),
        }
    }

    /// The plain `user` and `assistant` roles are what `encode_event_for_db` writes for an
    /// `Event::Append`, so `load_events` must hand every such row back as one: this is the primary
    /// live path through the decoder, not a fallback.
    #[tokio::test]
    async fn test_load_events_decodes_stored_append_rows() {
        use crate::conversation::Event;

        let manager = test_manager().await;
        let sid = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");

        // Written by hand rather than through `save_event`, so what is under test is the decoder
        // alone: nothing here would notice the encoder changing which role it writes.
        // `test_user_input_image_round_trips_via_user_blocks_role` is the test that closes that
        // loop.
        manager
            .save_message(sid, "user", "first")
            .await
            .expect("save user");
        let assistant_blocks = serde_json::json!([
            {"type": "text", "text": "answer"}
        ])
        .to_string();
        manager
            .save_message(sid, "assistant", &assistant_blocks)
            .await
            .expect("save assistant");

        let events = manager.load_events(sid).await.expect("load events");
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|e| matches!(e, Event::Append(_))));
    }

    /// A row with an unknown role should be skipped (with a warning) so a future schema bump that
    /// adds new event variants doesn't crash older binaries reading newer DBs.
    #[tokio::test]
    async fn test_load_events_skips_unknown_role() {
        let manager = test_manager().await;
        let sid = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        manager
            .save_message(sid, "user", "real")
            .await
            .expect("save real row");
        manager
            .save_message(sid, "future_event_kind", "{}")
            .await
            .expect("save unknown row");
        let events = manager.load_events(sid).await.expect("load events");
        assert_eq!(events.len(), 1);
    }

    /// Regression test for the umask-dependent permission bug: the session database file stores
    /// OAuth tokens and MCP credentials, so it must be readable by the owner only (0600) and the
    /// surrounding directory by the owner only (0700), regardless of the user's umask.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_session_db_file_mode() {
        use std::os::unix::fs::PermissionsExt;

        let temp_dir = tempfile::tempdir().expect("tempdir");
        let db_path = temp_dir.path().join("data").join("meka.db");

        let _manager = SessionManager::open(Some(&db_path), &Default::default())
            .await
            .expect("open session");

        let db_mode = std::fs::metadata(&db_path)
            .expect("stat db")
            .permissions()
            .mode();
        assert_eq!(
            db_mode & 0o777,
            0o600,
            "db file should be 0600 (got {:o})",
            db_mode & 0o777
        );

        let dir_mode = std::fs::metadata(db_path.parent().expect("parent"))
            .expect("stat dir")
            .permissions()
            .mode();
        assert_eq!(
            dir_mode & 0o777,
            0o700,
            "data dir should be 0700 (got {:o})",
            dir_mode & 0o777
        );

        let lock_mode = std::fs::metadata(db_path.parent().expect("parent").join("locks"))
            .expect("stat lock dir")
            .permissions()
            .mode();
        assert_eq!(
            lock_mode & 0o777,
            0o700,
            "lock dir should be 0700 (got {:o})",
            lock_mode & 0o777
        );
    }

    #[tokio::test]
    async fn test_create_session() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed to create session");
        assert!(
            manager
                .session_exists(session_id)
                .await
                .expect("failed to check")
        );
    }

    #[tokio::test]
    async fn test_session_stats_persist_round_trip() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");

        // A fresh row starts at all-zero (columns default to 0).
        let fresh = manager
            .load_session_stats(session_id)
            .await
            .expect("load fresh");
        assert_eq!(fresh.turns, 0);
        assert_eq!(fresh.input_tokens, 0);

        let snapshot = crate::stats::SessionStatsSnapshot {
            turns: 5,
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 10,
            cache_read_input_tokens: 200,
            redactions: 2,
            redacted_images: 3,
            redacted_bytes: 4096,
        };
        manager
            .save_session_stats(session_id, &snapshot)
            .await
            .expect("save stats");

        let loaded = manager
            .load_session_stats(session_id)
            .await
            .expect("load stats");
        assert_eq!(loaded.turns, 5);
        assert_eq!(loaded.input_tokens, 100);
        assert_eq!(loaded.output_tokens, 50);
        assert_eq!(loaded.cache_creation_input_tokens, 10);
        assert_eq!(loaded.cache_read_input_tokens, 200);
        assert_eq!(loaded.redactions, 2);
        assert_eq!(loaded.redacted_images, 3);
        assert_eq!(loaded.redacted_bytes, 4096);

        // An unknown session id is not an error; it reads as all-zero.
        let unknown = manager
            .load_session_stats(uuid::Uuid::new_v4())
            .await
            .expect("load unknown");
        assert_eq!(unknown.turns, 0);
    }

    #[tokio::test]
    async fn test_save_and_load_messages() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed to create session");

        manager
            .save_message(session_id, "user", "hello")
            .await
            .expect("failed to save message");
        manager
            .save_message(session_id, "assistant", "hi there")
            .await
            .expect("failed to save message");

        let messages = manager
            .load_messages(session_id)
            .await
            .expect("failed to load messages");

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "hello");
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(messages[1].content, "hi there");
    }

    #[tokio::test]
    async fn test_last_session_id() {
        let manager = test_manager().await;
        assert!(manager.last_session_id().await.expect("failed").is_none());

        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed to create session");
        let last = manager
            .last_session_id()
            .await
            .expect("failed to get last session");
        assert_eq!(last, Some(session_id));
    }

    #[tokio::test]
    async fn test_find_sessions_by_prefix_empty_db() {
        let manager = test_manager().await;
        let matches = manager
            .find_sessions_by_prefix("abc")
            .await
            .expect("failed prefix lookup");
        assert!(matches.is_empty());
    }

    #[tokio::test]
    async fn test_find_sessions_by_prefix_unique_match() {
        let manager = test_manager().await;
        let id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed to create session");
        // First 8 hex chars (before the first dash), guaranteed unique for a freshly-generated
        // random UUID with only one row in the DB.
        let prefix: String = id.to_string().chars().take(8).collect();
        let matches = manager
            .find_sessions_by_prefix(&prefix)
            .await
            .expect("failed prefix lookup");
        assert_eq!(matches, vec![id]);
    }

    #[tokio::test]
    async fn test_find_sessions_by_prefix_no_match() {
        let manager = test_manager().await;
        manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed to create session");
        let matches = manager
            .find_sessions_by_prefix("ffffffff")
            .await
            .expect("failed prefix lookup");
        // Real UUIDs are random; collision with this prefix is astronomically unlikely but
        // theoretically possible; re-create a session if so.
        assert!(matches.is_empty() || matches.len() == 1);
    }

    #[tokio::test]
    async fn test_find_sessions_by_prefix_rejects_non_hex_chars() {
        let manager = test_manager().await;
        manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed to create session");
        // SQL `%` and `_` wildcards must not slip through as prefix chars.
        for bad in ["%", "_", "abc%", "ab_c", "g0g0", "x123"] {
            let matches = manager
                .find_sessions_by_prefix(bad)
                .await
                .expect("failed prefix lookup");
            assert!(
                matches.is_empty(),
                "non-hex prefix {:?} should match nothing",
                bad
            );
        }
    }

    #[tokio::test]
    async fn test_find_sessions_by_prefix_empty_prefix_matches_nothing() {
        let manager = test_manager().await;
        manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed to create session");
        let matches = manager
            .find_sessions_by_prefix("")
            .await
            .expect("failed prefix lookup");
        assert!(
            matches.is_empty(),
            "empty prefix must not match every session"
        );
    }

    #[tokio::test]
    async fn test_session_locking_acquire_and_release() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed to create session");

        let lock = manager
            .lock_session(session_id)
            .expect("failed to lock session");

        // While the lock handle is alive, a second attempt must fail.
        match manager.lock_session(session_id) {
            Err(MekaError::SessionLocked(id)) => assert_eq!(id, session_id),
            other => panic!("expected SessionLocked, got {:?}", other.map(|_| "Ok(_)")),
        }

        // Dropping the handle releases the OS lock; re-acquisition succeeds.
        drop(lock);
        let _lock2 = manager
            .lock_session(session_id)
            .expect("failed to re-acquire session lock after drop");
    }

    #[tokio::test]
    async fn test_prune_orphan_lock_files() {
        let manager = test_manager().await;
        let live = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let live_lock = manager.lock_dir.join(format!("{}.lock", live));
        let orphan_lock = manager.lock_dir.join(format!("{}.lock", Uuid::new_v4()));
        let stray = manager.lock_dir.join("not-a-uuid.lock");
        std::fs::write(&live_lock, "").expect("write live lock");
        std::fs::write(&orphan_lock, "").expect("write orphan lock");
        std::fs::write(&stray, "").expect("write stray file");

        manager.prune_orphan_lock_files().await;

        assert!(live_lock.exists(), "live session's lock file must be kept");
        assert!(!orphan_lock.exists(), "orphan lock file must be removed");
        assert!(stray.exists(), "non-UUID file must be left untouched");
    }

    /// Opening a fresh database while another connection holds its write lock must wait, and must
    /// come out of it in WAL: converting a rollback journal takes an exclusive lock, and a database
    /// left unconverted is a permanent contention problem, not a slow start.
    ///
    /// What this does *not* isolate is the retry loop, and the reason is worth stating rather than
    /// leaving to be rediscovered. `rusqlite` installs a five-second busy timeout at open, so any
    /// hold shorter than that is waited out on the first attempt and the retry never runs. The case
    /// the retry exists for is the one where SQLite declines to consult the busy handler for this
    /// pragma at all -- observed at a couple of launches per few hundred, and not reproducible on
    /// demand. So this pins the property (a contended open waits and gets WAL) and the retry's own
    /// arm is covered by argument, not by a test.
    ///
    /// The writer is a plain `rusqlite` connection holding `BEGIN EXCLUSIVE`, which is what an
    /// unrelated process mid-transaction looks like from outside.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_a_first_open_waits_out_a_writer_instead_of_failing() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let db_path = temp_dir.path().join("meka.db");

        // A rollback-journal database with something in it, so the open below has a real conversion
        // to do rather than a no-op on an empty file.
        let blocker = rusqlite::Connection::open(&db_path).expect("open");
        blocker
            .execute_batch("CREATE TABLE placeholder (id INTEGER);")
            .expect("seed");
        blocker.execute_batch("BEGIN EXCLUSIVE;").expect("hold");

        let released = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(600));
            blocker.execute_batch("COMMIT;").expect("release");
        });

        let manager = SessionManager::open(Some(&db_path), &Default::default())
            .await
            .expect("a contended first open must wait, not fail");
        released.join().expect("the writer thread finishes");

        let mode: String = manager
            .connection
            .call(|connection| {
                connection.query_row("PRAGMA journal_mode", [], |row| row.get::<_, String>(0))
            })
            .await
            .expect("read the journal mode");
        assert_eq!(
            mode.to_lowercase(),
            "wal",
            "and must actually get WAL, not settle for the rollback journal"
        );
    }

    /// The prune's own question, asked of the lock rather than inferred from a snapshot.
    ///
    /// The old rule -- unlink any file whose id is not in the live set -- was justified by a
    /// session's row being committed before its lock file is acquired. That constrains the
    /// creator, not the sweeper: the `SELECT` can finish before a row commits while the `read_dir`
    /// runs after that session's lock file exists. Against a running `meka serve`, 21 of 401 live
    /// sessions lost their lock file, after which a second process attached to a session `serve`
    /// still held and wrote a whole turn into it.
    ///
    /// The id here is deliberately absent from `sessions`, which is exactly what the old rule
    /// called an orphan. Holding the lock is the only thing standing between it and deletion.
    #[tokio::test]
    async fn test_prune_spares_a_lock_file_that_is_held() {
        let manager = test_manager().await;
        let unrecorded = Uuid::new_v4();
        let path = manager.lock_dir.join(format!("{}.lock", unrecorded));
        let held = manager.lock_session(unrecorded).expect("take the lock");

        manager.prune_orphan_lock_files().await;
        assert!(
            path.exists(),
            "a file whose lock is held belongs to a live process, whatever the sessions table says"
        );

        // And the same file, once nobody holds it, is the garbage the sweep exists for.
        drop(held);
        manager.prune_orphan_lock_files().await;
        assert!(!path.exists(), "an unheld orphan is still swept");
    }

    #[tokio::test]
    async fn test_delete_session_removes_lock_file() {
        let manager = test_manager().await;
        let session = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let lock_path = manager.lock_dir.join(format!("{}.lock", session));
        std::fs::write(&lock_path, "").expect("write lock");

        manager.delete_session(session).await.expect("delete");
        assert!(
            !lock_path.exists(),
            "deleting a session must remove its lock file"
        );
    }

    /// A session another meka process has open is not one this process may delete.
    ///
    /// `meka session delete <id>` against a live REPL exited 0 having said nothing at all: the
    /// count goes through `tracing::info!`, invisible at the default level. The row and its
    /// messages cascaded away underneath a conversation that carried on as though nothing had
    /// happened, until its next turn ran against the provider and *then* failed on a foreign-key
    /// violation -- tokens spent, answer lost, and every later turn in that REPL failing the same
    /// way with no recovery.
    #[tokio::test]
    async fn test_deleting_a_session_another_process_holds_is_refused() {
        let manager = test_manager().await;
        let session = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let _held = manager.lock_session(session).expect("hold the session");

        match manager.delete_session_unless_attached(session).await {
            Err(MekaError::SessionLocked(id)) => assert_eq!(id, session),
            other => panic!("expected SessionLocked, got {:?}", other.map(|_| "Ok(_)")),
        }
        assert!(
            manager.session_exists(session).await.expect("exists"),
            "the refusal must leave the conversation alone, not merely report one"
        );
    }

    /// The sweep that runs on every start, against a session someone is sitting in.
    ///
    /// Only turns bump `updated_at` -- resuming does not touch it -- so a REPL left at its prompt
    /// past the retention window looks expired while a human is looking at it. Any `meka` start
    /// that goes through `async_main` runs this sweep, so an unrelated invocation in another
    /// terminal announced `deleted 1 session(s)` and destroyed the live one.
    #[tokio::test]
    async fn test_the_retention_sweep_spares_a_session_that_is_open() {
        let manager = test_manager().await;
        let open = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let stale = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let long_ago = (chrono::Utc::now() - chrono::Duration::days(90)).to_rfc3339();
        for session in [open, stale] {
            manager
                .set_session_updated_at_for_test(session, &long_ago)
                .await
                .expect("backdate");
        }
        let _held = manager.lock_session(open).expect("hold the session");

        let sweep = manager
            .delete_expired_sessions(30)
            .await
            .expect("retention sweep");

        assert_eq!(sweep.deleted, 1, "the session nobody has open still goes");
        assert_eq!(
            sweep.attached_elsewhere, 1,
            "and the sweep has to be able to say what it left, or a count of deletions reads as \
             'everything matched went'"
        );
        assert!(manager.session_exists(open).await.expect("exists"));
        assert!(!manager.session_exists(stale).await.expect("exists"));
    }

    /// A session's lock is taken *before* its row is written, not after.
    ///
    /// The window between the two is what `meka session delete --all` fell into: it enumerates
    /// `SELECT id FROM sessions` at delete time, so it saw a row committed microseconds earlier
    /// whose creator had not yet reached `flock`, took the lock nobody held, and cascaded the
    /// conversation away underneath the process creating it. A contention run measured **42 lost
    /// turns in 11,948** with four creators against two sweep loops, each ending
    /// `FOREIGN KEY constraint failed` with the user's prompt gone.
    ///
    /// Observed by ordering rather than by racing, deliberately. The window is microseconds wide,
    /// so a test that waits for the row and then sweeps passes just as happily with the fix
    /// reverted -- the first version of this test did exactly that. Volume does find it, but at
    /// roughly one event per six hundred turns, which is a coin flip rather than a guard. Breaking
    /// the insert instead makes the ordering directly visible: if the lock comes first, the
    /// attempt leaves a lock file behind even though no row was ever written, and if it comes
    /// second there is nothing in the directory at all.
    #[tokio::test]
    async fn a_session_is_locked_before_its_row_is_written() {
        let manager = test_manager().await;
        let locks_before = std::fs::read_dir(&manager.lock_dir)
            .expect("read the lock dir")
            .count();
        // Renamed rather than dropped, so the failure is a plain "no such table" from the insert
        // rather than anything the foreign keys have an opinion about.
        manager
            .connection
            .call(|connection| connection.execute_batch("ALTER TABLE sessions RENAME TO hidden;"))
            .await
            .expect("hide the table");

        let refused = manager
            .create_session_locked(None, None, None, None, "test-profile".to_string())
            .await;

        assert!(
            refused.is_err(),
            "the premise: with no `sessions` table the row cannot be written"
        );
        assert_eq!(
            std::fs::read_dir(&manager.lock_dir)
                .expect("read the lock dir")
                .count(),
            locks_before + 1,
            "a creation that never wrote a row must still have taken its lock first; nothing in \
             the lock directory means the row went first, and a row that lands before its lock is \
             one a sweep can take"
        );
    }

    /// A fork takes the copy's lock before the copy's row exists.
    ///
    /// Forking committed the copy and *then* locked it, which is the same commit-then-claim window
    /// [`SessionManager::create_session_locked`] closed, in the same width. A concurrent
    /// `meka session delete --all` enumerates the copy, takes the lock nobody holds, and deletes
    /// it -- after which the fork locks the vanished id successfully and hands its caller a session
    /// whose next turn dies on a foreign-key violation. Under ACP it is quieter still:
    /// `load_events` returns empty and the editor gets a silently blank fork.
    ///
    /// Observed by ordering rather than by racing, for the same reason
    /// [`a_session_is_locked_before_its_row_is_written`] is: the window is microseconds and nothing
    /// that polls can see it. Breaking the copy makes the order visible -- if the claim comes
    /// first, a lock file is left behind for a row that was never written, and if it comes second
    /// there is nothing in the directory at all.
    ///
    /// Deliberately *not* the missing-source path, which would be the more obvious probe: that one
    /// now removes its own file, precisely because a fork of an unknown id is client-reachable and
    /// would otherwise accumulate one per attempt.
    #[tokio::test]
    async fn a_fork_is_locked_before_the_copy_exists() {
        let manager = test_manager().await;
        let source = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let locks_before = std::fs::read_dir(&manager.lock_dir)
            .expect("read the lock dir")
            .count();
        // Renamed rather than dropped, so the copy fails on a plain "no such table" rather than on
        // anything the foreign keys have an opinion about.
        manager
            .connection
            .call(|connection| connection.execute_batch("ALTER TABLE sessions RENAME TO hidden;"))
            .await
            .expect("hide the table");

        let refused = manager
            .fork_session_locked(source, ForkOverrides::default())
            .await;

        assert!(
            refused.is_err(),
            "the premise: with no `sessions` table the copy cannot be written"
        );
        assert_eq!(
            std::fs::read_dir(&manager.lock_dir)
                .expect("read the lock dir")
                .count(),
            locks_before + 1,
            "a fork that never wrote a row must still have claimed its id first; nothing in the \
             lock directory means the row would have gone first, and a copy that lands before its \
             lock is one a sweep can take"
        );
    }

    /// The sweep decides on the rows as they are, not on a list read a moment earlier.
    ///
    /// Selecting candidates and then deleting them by id is two statements where there was one, so
    /// a condition checked only in the first can stop being true in between. The one that matters
    /// is "no schedule ahead of it": `parent_session_id` cascades, so a job created against a
    /// sub-agent child in that gap would be swept away with a parent nothing has locked, and the
    /// lock cannot stand in for the check because it is the *parent* being deleted and the
    /// *child* that acquired the job.
    ///
    /// Driven through [`SessionManager::delete_the_unattached_among`] directly, with a candidate
    /// that already owns a job, because the gap itself is microseconds wide and not something a
    /// test can sit inside. What it pins is the property that closes it: the predicate is in the
    /// delete, not only in the select.
    #[tokio::test]
    async fn the_sweep_re_checks_its_own_condition_inside_the_delete() {
        let manager = test_manager().await;
        let scheduled = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let ordinary = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        manager
            .schedule_store()
            .create_scheduled_job(&crate::schedule::ScheduledJob {
                attempts: 0,
                id: "job-1".to_string(),
                session_id: scheduled,
                schedule: crate::schedule::Schedule::parse_every("1h").expect("parses"),
                prompt: "check the thing".to_string(),
                gate: None,
                created_at: chrono::Utc::now(),
                last_fired_at: None,
                next_fire_at: chrono::Utc::now() + chrono::Duration::hours(1),
            })
            .await
            .expect("create the job");

        let sweep = manager
            .delete_the_unattached_among(&[scheduled, ordinary], NOT_SPOKEN_FOR_BY_A_SCHEDULE)
            .await
            .expect("sweep");

        assert_eq!(sweep.deleted, 1, "only the one with nothing ahead of it");
        assert!(
            manager.session_exists(scheduled).await.expect("exists"),
            "a session a job still depends on must survive a delete it was listed for"
        );
        assert!(!manager.session_exists(ordinary).await.expect("exists"));
    }

    /// `meka session delete --all` is the same rule with no window: everything except what someone
    /// else is using.
    #[tokio::test]
    async fn test_delete_all_spares_a_session_that_is_open() {
        let manager = test_manager().await;
        let open = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let other = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let _held = manager.lock_session(open).expect("hold the session");

        let sweep = manager.delete_all_sessions().await.expect("delete all");

        assert_eq!(sweep.deleted, 1);
        assert_eq!(sweep.attached_elsewhere, 1);
        assert!(manager.session_exists(open).await.expect("exists"));
        assert!(!manager.session_exists(other).await.expect("exists"));
    }

    #[tokio::test]
    async fn test_open_prunes_orphan_lock_files() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let db_path = temp_dir.path().join("meka.db");
        let lock_dir = temp_dir.path().join("locks");
        std::fs::create_dir_all(&lock_dir).expect("create locks dir");
        let orphan = lock_dir.join(format!("{}.lock", Uuid::new_v4()));
        std::fs::write(&orphan, "").expect("write orphan");

        // A fresh DB has no sessions, so the planted file is an orphan.
        let _manager = SessionManager::open(Some(&db_path), &Default::default())
            .await
            .expect("open should succeed");
        assert!(
            !orphan.exists(),
            "open() must prune pre-existing orphan lock files"
        );
    }

    #[tokio::test]
    async fn test_session_not_found() {
        let manager = test_manager().await;
        let fake_id = Uuid::new_v4();
        assert!(
            !manager
                .session_exists(fake_id)
                .await
                .expect("failed to check")
        );
    }

    #[tokio::test]
    async fn test_multiple_sessions() {
        let manager = test_manager().await;
        let session1 = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed");
        let session2 = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed");

        manager
            .save_message(session1, "user", "msg1")
            .await
            .expect("failed");
        manager
            .save_message(session2, "user", "msg2")
            .await
            .expect("failed");

        let messages1 = manager.load_messages(session1).await.expect("failed");
        let messages2 = manager.load_messages(session2).await.expect("failed");

        assert_eq!(messages1.len(), 1);
        assert_eq!(messages1[0].content, "msg1");
        assert_eq!(messages2.len(), 1);
        assert_eq!(messages2[0].content, "msg2");
    }

    #[tokio::test]
    async fn test_delete_expired_sessions() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed");
        manager
            .save_message(session_id, "user", "hello")
            .await
            .expect("failed");

        // Backdate the session to 100 days ago
        let old_date = (chrono::Utc::now() - chrono::TimeDelta::days(100)).to_rfc3339();
        manager
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![old_date, session_id.to_string()],
                )?;
                Ok(())
            })
            .await
            .expect("failed to backdate");

        let deleted = manager
            .delete_expired_sessions(30)
            .await
            .expect("failed to delete");
        assert_eq!(deleted.deleted, 1);
        assert!(!manager.session_exists(session_id).await.expect("failed"));

        let messages = manager.load_messages(session_id).await.expect("failed");
        assert!(messages.is_empty());
    }

    /// The FK cascade must not take a job-owning child with its stale parent.
    ///
    /// A session holding a scheduled job is not idle, whatever its `updated_at` says: only turns
    /// bump that column, so a gated watcher that evaluates every tick and rarely fires looks
    /// untouched precisely while it is doing its job.
    ///
    /// Sparing only the row named by `scheduled_jobs.session_id` left the guard half-built:
    /// `parent_session_id` carries `ON DELETE CASCADE`, so a top-level session that has gone quiet
    /// still deletes its sub-agent children, and a job created against a child -- which the HTTP
    /// surface allows, gating only on the session existing -- went with them. The sweep then
    /// reported one deletion and said nothing about the schedule it destroyed.
    #[tokio::test]
    async fn retention_spares_the_parent_of_a_child_that_has_a_scheduled_job() {
        let manager = test_manager().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create parent");
        let child = manager
            .create_child_session(parent, None, None, "test-profile".to_string())
            .await
            .expect("create child")
            .0;

        let job = crate::schedule::ScheduledJob {
            attempts: 0,
            id: uuid::Uuid::new_v4().to_string(),
            session_id: child,
            schedule: crate::schedule::Schedule::parse_every("30m").expect("parses"),
            prompt: "check the build".to_string(),
            gate: None,
            created_at: chrono::Utc::now(),
            last_fired_at: None,
            next_fire_at: chrono::Utc::now() + chrono::Duration::minutes(30),
        };
        manager
            .schedule_store()
            .create_scheduled_job(&job)
            .await
            .expect("save job");

        // Only the parent looks ancient; the child is what owns the future.
        let ancient = (chrono::Utc::now() - chrono::Duration::days(400)).to_rfc3339();
        manager
            .set_session_updated_at_for_test(parent, &ancient)
            .await
            .expect("backdate");

        manager.delete_expired_sessions(30).await.expect("sweep");

        assert!(
            manager.session_exists(child).await.expect("exists"),
            "the child owning the job was cascaded away with its stale parent"
        );
        assert!(
            manager.session_exists(parent).await.expect("exists"),
            "the parent must be spared too, since deleting it is what takes the child"
        );
    }

    /// Retention must leave a session alone while it still owns a scheduled job.
    ///
    /// It did not: the cascade took the job along with the session, and the sweep reported only
    /// "deleted N session(s)".
    #[tokio::test]
    async fn retention_spares_a_session_that_still_has_a_scheduled_job() {
        let manager = test_manager().await;
        let watcher = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let plain = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let job = crate::schedule::ScheduledJob {
            attempts: 0,
            id: uuid::Uuid::new_v4().to_string(),
            session_id: watcher,
            schedule: crate::schedule::Schedule::parse_every("30m").expect("parses"),
            prompt: "check the build".to_string(),
            gate: None,
            created_at: chrono::Utc::now(),
            last_fired_at: None,
            next_fire_at: chrono::Utc::now() + chrono::Duration::minutes(30),
        };
        manager
            .schedule_store()
            .create_scheduled_job(&job)
            .await
            .expect("save job");

        // Both look ancient by `updated_at`; only one of them has a future.
        let ancient = (chrono::Utc::now() - chrono::Duration::days(400)).to_rfc3339();
        for id in [watcher, plain] {
            manager
                .set_session_updated_at_for_test(id, &ancient)
                .await
                .expect("backdate");
        }

        manager.delete_expired_sessions(30).await.expect("sweep");

        assert!(
            manager.session_exists(watcher).await.expect("exists"),
            "a session with a pending job must survive retention"
        );
        assert!(
            !manager.session_exists(plain).await.expect("exists"),
            "a genuinely idle session should still be swept"
        );
        assert_eq!(
            manager
                .schedule_store()
                .list_scheduled_jobs(watcher)
                .await
                .expect("list")
                .len(),
            1,
            "the job must survive with its session"
        );
    }

    #[tokio::test]
    async fn test_delete_expired_sessions_keeps_recent() {
        let manager = test_manager().await;
        let old_session = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed");
        let new_session = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed");

        manager
            .save_message(old_session, "user", "old")
            .await
            .expect("failed");
        manager
            .save_message(new_session, "user", "new")
            .await
            .expect("failed");

        // Backdate only the old session
        let old_date = (chrono::Utc::now() - chrono::TimeDelta::days(100)).to_rfc3339();
        manager
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![old_date, old_session.to_string()],
                )?;
                Ok(())
            })
            .await
            .expect("failed to backdate");

        let deleted = manager
            .delete_expired_sessions(30)
            .await
            .expect("failed to delete");
        assert_eq!(deleted.deleted, 1);
        assert!(!manager.session_exists(old_session).await.expect("failed"));
        assert!(manager.session_exists(new_session).await.expect("failed"));
    }

    /// `--older-than-days` puts a raw number in the user's hands, so a mistyped run of digits must
    /// not panic. `TimeDelta` overflows near 10^11 days and `Utc::now() - delta` near 96.4 million,
    /// so both bounds need covering; either way nothing is old enough to match.
    #[tokio::test]
    async fn test_delete_expired_sessions_survives_absurd_windows() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        manager
            .save_message(session_id, "user", "hello")
            .await
            .expect("save");

        for days in [96_500_000, u64::MAX] {
            let deleted = manager
                .delete_expired_sessions(days)
                .await
                .expect("must not panic or error");
            assert_eq!(deleted.deleted, 0, "{days} days should match nothing");
        }
        assert!(manager.session_exists(session_id).await.expect("exists"));
    }

    #[tokio::test]
    async fn test_clear_messages() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed");

        manager
            .save_message(session_id, "user", "hello")
            .await
            .expect("failed");
        manager
            .save_message(session_id, "assistant", "hi")
            .await
            .expect("failed");

        let messages = manager.load_messages(session_id).await.expect("failed");
        assert_eq!(messages.len(), 2);

        manager
            .clear_messages(session_id)
            .await
            .expect("failed to clear");

        let messages = manager.load_messages(session_id).await.expect("failed");
        assert!(messages.is_empty());

        // Session itself should still exist
        assert!(manager.session_exists(session_id).await.expect("failed"));
    }

    #[test]
    fn test_strip_context_tags_with_context() {
        let input = "<context>\n[Environment context]\nWorking directory: /tmp\nDate: Mon\n</context>\n\nhello world";
        assert_eq!(strip_context_tags(input), "hello world");
    }

    #[test]
    fn test_strip_context_tags_without_context() {
        let input = "hello world";
        assert_eq!(strip_context_tags(input), "hello world");
    }

    #[test]
    fn test_strip_context_tags_empty_after_context() {
        let input = "<context>\nstuff\n</context>\n\n";
        assert_eq!(strip_context_tags(input), "");
    }

    #[test]
    fn test_truncate_preview_with_context_tags() {
        let input = "<context>\n[Environment context]\nWorking directory: /tmp\n</context>\n\nfind all Rust files";
        assert_eq!(truncate_preview(input, 80), "find all Rust files");
    }

    #[test]
    fn test_truncate_preview_without_context_tags() {
        let input = "find all Rust files";
        assert_eq!(truncate_preview(input, 80), "find all Rust files");
    }

    #[test]
    fn test_truncate_preview_untagged_multiline_takes_first_line() {
        let input = "[Environment context]\nWorking directory: /tmp\n\nfind all Rust files";
        assert_eq!(truncate_preview(input, 80), "[Environment context]");
    }

    #[test]
    fn test_truncate_preview_with_context_tags_long_input() {
        let long_input = format!("<context>\nstuff\n</context>\n\n{}", "x".repeat(100));
        let preview = truncate_preview(&long_input, 80);
        assert!(preview.ends_with('…'));
        assert!(preview.len() <= 84); // 80 chars + "…"
    }

    // End-to-end regression tests for `meka session list`'s preview.
    // These tests mock the complete pipeline that produces the
    // `Preview` column: build the turn-context block the agent
    // actually sends, prepend it to a user prompt the way
    // `agent::Agent::run_turn` does, persist via `save_message`,
    // then call `list_sessions` and assert the preview matches the
    // raw user prompt. Any future change to:
    //   - `context::build_turn_context`'s output shape
    //   - `agent::Agent::run_turn`'s "prefix block, then user input" format
    //   - `save_message` storage
    //   - `list_sessions`'s SQL / preview rendering
    //   - `strip_context_tags` / `truncate_preview`
    // that breaks the preview will fail one of these tests.
    //
    // The preview has regressed several times historically; these guards exist so the next breakage
    // is caught by CI, not a user.

    /// Reconstructs what `agent::Agent::run_turn` passes to `save_message` for a fresh user turn:
    /// the `<context>...</context>` block followed by `\n\n` and the user's raw prompt. Kept
    /// structurally identical to the real call-site (see `src/agent.rs::run_turn` →
    /// `augmented_input = format!("{}\n\n{}", block, user_input)`).
    fn mock_run_turn_user_message(
        permission: crate::permission::Permission,
        user_input: &str,
    ) -> String {
        let block = crate::context::build_turn_context(
            permission,
            &crate::tools::todo::TodoState::default(),
            std::path::Path::new("."),
            &[],
            "",
            // A populated budget, matching the real call site: the section lands inside the
            // `<context>` block, so `strip_context_tags` must carry it away with everything else.
            Some(crate::context::ContextBudget {
                used: 42_000,
                window: 200_000,
                compact_at_percent: Some(80),
                generation: 0,
            }),
            &[],
            // Likewise the resume notice: it heads the block, so it is the first thing
            // `strip_context_tags` has to swallow rather than mistake for the user's prompt.
            true,
        );
        format!("{}\n\n{}", block, user_input)
    }

    #[tokio::test]
    async fn test_list_sessions_preview_is_user_prompt_not_context_wrapper() {
        // The canonical regression: user types a prompt, turn runs, `meka session list` must show
        // the prompt, not `<context>`, not the permission/environment metadata.
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create_session");

        let user_prompt = "find all Rust files under src/";
        let stored = mock_run_turn_user_message(crate::permission::Permission::Read, user_prompt);
        manager
            .save_message(session_id, "user", &stored)
            .await
            .expect("save_message");

        let (summaries, _next_cursor) = manager
            .list_sessions(10, false, None, None)
            .await
            .expect("list_sessions");
        let summary = summaries
            .iter()
            .find(|s| s.id == session_id)
            .expect("session missing from list");

        assert_eq!(
            summary.preview, user_prompt,
            "preview regressed: expected user prompt, got {:?}",
            summary.preview
        );
        assert!(
            !summary.preview.contains("<context>"),
            "wrapper leaked into preview: {:?}",
            summary.preview
        );
        assert!(
            !summary.preview.contains("[Permission context]"),
            "permission metadata leaked into preview: {:?}",
            summary.preview
        );
    }

    #[tokio::test]
    async fn test_list_sessions_preview_covers_all_permission_levels() {
        // The context block's shape differs per permission level (`none` omits the [Environment
        // context] entirely, `workspace` adds a write-boundary paragraph to it). Every level should
        // still surface the user's prompt cleanly. "all permission levels" in the name is a claim,
        // so the list has to actually hold all of them: `workspace` was missing here, and it is the
        // level whose block grew a new section.
        let manager = test_manager().await;
        for (label, permission) in &[
            ("none", crate::permission::Permission::None),
            ("read", crate::permission::Permission::Read),
            ("workspace", crate::permission::Permission::Workspace),
            ("ask", crate::permission::Permission::Ask),
            ("unrestricted", crate::permission::Permission::Unrestricted),
        ] {
            let session_id = manager
                .create_session(None, "test-profile".to_string())
                .await
                .expect("create_session");
            let prompt = format!("ask at {} level", label);
            let stored = mock_run_turn_user_message(*permission, &prompt);
            manager
                .save_message(session_id, "user", &stored)
                .await
                .expect("save_message");

            let (summaries, _next_cursor) = manager
                .list_sessions(100, false, None, None)
                .await
                .expect("list_sessions");
            let summary = summaries
                .iter()
                .find(|s| s.id == session_id)
                .unwrap_or_else(|| panic!("session missing for level {}", label));
            assert_eq!(
                summary.preview, prompt,
                "preview mismatch at permission level {}",
                label
            );
        }
    }

    #[tokio::test]
    async fn test_list_sessions_preview_truncates_long_prompt_with_ellipsis() {
        // Long prompts are capped at 80 chars with a trailing ellipsis. The cap must apply to the
        // user's prompt, not the wrapper.
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create_session");

        let long_prompt = "a".repeat(150);
        let stored = mock_run_turn_user_message(crate::permission::Permission::Read, &long_prompt);
        manager
            .save_message(session_id, "user", &stored)
            .await
            .expect("save_message");

        let (summaries, _next_cursor) = manager
            .list_sessions(10, false, None, None)
            .await
            .expect("list_sessions");
        let summary = summaries.iter().find(|s| s.id == session_id).unwrap();

        assert!(
            summary.preview.starts_with("aaa"),
            "preview should start with the user's content, not the wrapper: {:?}",
            summary.preview
        );
        assert!(
            summary.preview.ends_with('…'),
            "long preview should end with ellipsis: {:?}",
            summary.preview
        );
        assert!(summary.preview.chars().count() <= 81);
    }

    #[tokio::test]
    async fn test_list_sessions_preview_is_first_user_turn_not_later() {
        // Multiple turns in one session: preview must be the FIRST user prompt, not a later one.
        // `ORDER BY id ASC LIMIT 1` guarantees this; guard against that being changed.
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create_session");

        for (i, prompt) in ["first prompt", "second prompt", "third prompt"]
            .iter()
            .enumerate()
        {
            let stored = mock_run_turn_user_message(crate::permission::Permission::Read, prompt);
            manager
                .save_message(session_id, "user", &stored)
                .await
                .expect("save_message");
            // Interleave an assistant reply: real sessions alternate.
            manager
                .save_message(session_id, "assistant", &format!("reply {}", i))
                .await
                .expect("save_message");
        }

        let (summaries, _next_cursor) = manager
            .list_sessions(10, false, None, None)
            .await
            .expect("list_sessions");
        let summary = summaries.iter().find(|s| s.id == session_id).unwrap();
        assert_eq!(summary.preview, "first prompt");
    }

    #[tokio::test]
    async fn test_list_sessions_preview_multiline_shows_first_line() {
        // Multi-line user prompts collapse to the first line in the list view. The remaining lines
        // are not leaked.
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create_session");

        let stored = mock_run_turn_user_message(
            crate::permission::Permission::Read,
            "line one is the preview\nline two should not appear\nline three either",
        );
        manager
            .save_message(session_id, "user", &stored)
            .await
            .expect("save_message");

        let (summaries, _next_cursor) = manager
            .list_sessions(10, false, None, None)
            .await
            .expect("list_sessions");
        let summary = summaries.iter().find(|s| s.id == session_id).unwrap();
        assert_eq!(summary.preview, "line one is the preview");
    }

    #[tokio::test]
    async fn test_list_sessions_preview_independent_per_session() {
        // Each session's preview is its own first user turn: no cross-contamination from neighbour
        // sessions.
        let manager = test_manager().await;
        let a = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create_session");
        let b = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create_session");
        let c = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create_session");

        for (sid, prompt) in [(a, "alpha"), (b, "beta"), (c, "gamma")] {
            let stored = mock_run_turn_user_message(crate::permission::Permission::Read, prompt);
            manager
                .save_message(sid, "user", &stored)
                .await
                .expect("save_message");
        }

        let (summaries, _next_cursor) = manager
            .list_sessions(10, false, None, None)
            .await
            .expect("list_sessions");
        let preview_of = |id: uuid::Uuid| {
            summaries
                .iter()
                .find(|s| s.id == id)
                .map(|s| s.preview.clone())
                .unwrap_or_default()
        };
        assert_eq!(preview_of(a), "alpha");
        assert_eq!(preview_of(b), "beta");
        assert_eq!(preview_of(c), "gamma");
    }

    #[tokio::test]
    async fn test_list_sessions_preview_empty_session_has_empty_preview() {
        // A session with zero user messages (e.g. created but Ctrl-C'd before first dispatch) falls
        // back to an empty string; it should not panic or render `<no user msg>` scaffolding.
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create_session");

        let (summaries, _next_cursor) = manager
            .list_sessions(10, false, None, None)
            .await
            .expect("list_sessions");
        let summary = summaries.iter().find(|s| s.id == session_id).unwrap();
        assert_eq!(summary.preview, "");
    }

    #[tokio::test]
    async fn test_list_sessions_preview_compacted_session() {
        // After `/compact`, the agent clears messages and inserts a single new user message
        // starting with `[Conversation summary from session compaction]`. That has no `<context>`
        // wrapper; `list_sessions` should surface the summary's first line, not an empty preview.
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create_session");

        let summary_msg = "[Conversation summary from session compaction]\n\nSummary text here\n\n\
             [Post-compaction context]\n\n…";
        manager
            .save_message(session_id, "user", summary_msg)
            .await
            .expect("save_message");

        let (summaries, _next_cursor) = manager
            .list_sessions(10, false, None, None)
            .await
            .expect("list_sessions");
        let summary = summaries.iter().find(|s| s.id == session_id).unwrap();
        assert_eq!(
            summary.preview, "[Conversation summary from session compaction]",
            "compacted session should surface the summary marker as preview"
        );
    }

    #[tokio::test]
    async fn test_list_sessions_preview_unwrapped_user_message() {
        // The `<context>` block is added by `Agent::run_turn`, not by storage. A `user` row written
        // by any other path carries none -- `import_sessions` replays whatever an archive held --
        // and then the stored string IS the prompt, so the preview equals it.
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create_session");
        manager
            .save_message(session_id, "user", "prompt without any wrapper")
            .await
            .expect("save_message");

        let (summaries, _next_cursor) = manager
            .list_sessions(10, false, None, None)
            .await
            .expect("list_sessions");
        let summary = summaries.iter().find(|s| s.id == session_id).unwrap();
        assert_eq!(summary.preview, "prompt without any wrapper");
    }

    #[tokio::test]
    async fn test_session_metadata_round_trips() {
        let manager = test_manager().await;

        // The NULL shape the REPL and ACP get from `create_session_locked`, which they call with
        // permission, capabilities and token id all `None`: unset, so the re-attach helper falls
        // back to the process default.
        let plain = manager
            .create_session(
                Some(std::path::PathBuf::from("/tmp/plain")),
                "test-profile".to_string(),
            )
            .await
            .expect("create plain");
        let plain_info = manager
            .session_info(plain)
            .await
            .expect("session_info")
            .expect("plain row");
        assert_eq!(plain_info.permission, None);
        assert_eq!(plain_info.capabilities_json, None);

        // Metadata path: persisted permission + capabilities + token_id round-trip verbatim.
        let with_meta = manager
            .create_session_with_metadata(
                Some(std::path::PathBuf::from("/tmp/meta")),
                Some("read".to_string()),
                Some(r#"{"supports_reasoning_stream":true}"#.to_string()),
                Some("token_fp_1234".to_string()),
                "test-profile".to_string(),
            )
            .await
            .expect("create with metadata");
        let meta_info = manager
            .session_info(with_meta.id)
            .await
            .expect("session_info")
            .expect("meta row");
        assert_eq!(meta_info.permission.as_deref(), Some("read"));
        assert_eq!(
            meta_info.capabilities_json.as_deref(),
            Some(r#"{"supports_reasoning_stream":true}"#)
        );
        assert_eq!(
            meta_info.token_id.as_deref(),
            Some("token_fp_1234"),
            "token_id round-trips through the DB"
        );
        // The DB-returned `created_at` matches what session_info reads back.
        assert_eq!(meta_info.created_at, with_meta.created_at);

        // `update_session_permission` flips the persisted value.
        let updated = manager
            .update_session_permission(with_meta.id, "workspace")
            .await
            .expect("update permission");
        assert_eq!(updated, 1);
        let after_flip = manager
            .session_info(with_meta.id)
            .await
            .expect("session_info")
            .expect("post-flip row");
        assert_eq!(after_flip.permission.as_deref(), Some("workspace"));
    }

    // Child-session tests: parent→sub-agent linkage, cascade-on-delete, and `meka session list`
    // filter behavior.

    #[tokio::test]
    async fn test_create_child_session_writes_parent_id() {
        let manager = test_manager().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create parent");
        let child = manager
            .create_child_session(parent, None, None, "test-profile".to_string())
            .await
            .expect("create child")
            .0;

        // Cross-check the column via list_sessions(include_children=true).
        let (summaries, _next_cursor) = manager
            .list_sessions(100, true, None, None)
            .await
            .expect("list_sessions");
        let ids: Vec<_> = summaries.iter().map(|s| s.id).collect();
        assert!(ids.contains(&parent), "parent missing from listing");
        assert!(ids.contains(&child), "child missing from listing");
    }

    #[tokio::test]
    async fn test_list_sessions_default_hides_children() {
        let manager = test_manager().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create parent");
        let _child = manager
            .create_child_session(parent, None, None, "test-profile".to_string())
            .await
            .expect("create child")
            .0;

        let (default_view, _) = manager
            .list_sessions(10, false, None, None)
            .await
            .expect("list");
        let ids: Vec<_> = default_view.iter().map(|s| s.id).collect();
        assert_eq!(ids.len(), 1);
        assert!(ids.contains(&parent), "parent should still be visible");

        let (full_view, _) = manager
            .list_sessions(10, true, None, None)
            .await
            .expect("list");
        assert_eq!(full_view.len(), 2);
    }

    #[tokio::test]
    async fn test_create_session_round_trips_cwd_through_session_info() {
        let manager = test_manager().await;
        let cwd = PathBuf::from("/home/agent/proj-a");
        let sid = manager
            .create_session(Some(cwd.clone()), "test-profile".to_string())
            .await
            .expect("create");

        let info = manager
            .session_info(sid)
            .await
            .expect("session_info")
            .expect("present");
        assert_eq!(info.cwd, Some(cwd));
    }

    #[tokio::test]
    async fn test_session_info_returns_none_for_unknown_id() {
        let manager = test_manager().await;
        let absent = manager
            .session_info(Uuid::new_v4())
            .await
            .expect("session_info");
        assert!(absent.is_none());
    }

    #[tokio::test]
    async fn test_list_sessions_filters_by_cwd() {
        let manager = test_manager().await;
        let cwd_a = PathBuf::from("/home/agent/proj-a");
        let cwd_b = PathBuf::from("/home/agent/proj-b");
        let a = manager
            .create_session(Some(cwd_a.clone()), "test-profile".to_string())
            .await
            .expect("create a");
        let _b = manager
            .create_session(Some(cwd_b.clone()), "test-profile".to_string())
            .await
            .expect("create b");

        let (only_a, next) = manager
            .list_sessions(10, false, Some(&cwd_a), None)
            .await
            .expect("list filtered");
        assert_eq!(only_a.len(), 1);
        assert_eq!(only_a[0].id, a);
        assert!(
            next.is_none(),
            "single result must not advertise a next page"
        );

        let (all, _) = manager
            .list_sessions(10, false, None, None)
            .await
            .expect("list unfiltered");
        assert_eq!(all.len(), 2, "unfiltered must include both sessions");
    }

    #[tokio::test]
    async fn test_list_sessions_cwd_filter_excludes_null_cwd_rows() {
        // A session created by `create_session(None, "test-profile".to_string())` recorded no cwd,
        // so it can never match a cwd filter: NULL is not equal to a TEXT value in SQL.
        let manager = test_manager().await;
        let cwd = PathBuf::from("/home/agent/proj");
        let with_cwd = manager
            .create_session(Some(cwd.clone()), "test-profile".to_string())
            .await
            .expect("create with cwd");
        let _without_cwd = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create without cwd");

        let (filtered, _) = manager
            .list_sessions(10, false, Some(&cwd), None)
            .await
            .expect("list");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, with_cwd);
    }

    #[tokio::test]
    async fn test_list_sessions_pagination_cursor_round_trips() {
        let manager = test_manager().await;
        // Create five sessions; cap each page at 2. Walking forward must visit all five exactly
        // once with monotonically older updated_at.
        let mut ids = Vec::new();
        for _ in 0..5 {
            let id = manager
                .create_session(None, "test-profile".to_string())
                .await
                .expect("create");
            // `created_at`/`updated_at` use chrono::Utc::now(); pause to ensure each row's
            // timestamp is strictly newer (RFC3339 millisecond resolution).
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
            ids.push(id);
        }

        let mut walked = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let (page, next) = manager
                .list_sessions(2, false, None, cursor.as_deref())
                .await
                .expect("list");
            for summary in &page {
                walked.push(summary.id);
            }
            if next.is_none() {
                break;
            }
            cursor = next;
            assert!(walked.len() <= 5, "infinite pagination loop");
        }
        // The walk emits sessions newest-first; the creation order is oldest-first, so reverse to
        // compare.
        ids.reverse();
        assert_eq!(walked, ids, "pagination must visit every row in order");
    }

    #[tokio::test]
    async fn test_list_sessions_invalid_cursor_returns_error() {
        let manager = test_manager().await;
        let result = manager
            .list_sessions(10, false, None, Some("not_base64_at_all!!"))
            .await;
        assert!(
            result.is_err(),
            "garbage cursor must be rejected rather than silently ignored"
        );
    }

    #[tokio::test]
    async fn test_delete_session_cascades_to_children() {
        let manager = test_manager().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create parent");
        let child = manager
            .create_child_session(parent, None, None, "test-profile".to_string())
            .await
            .expect("create child")
            .0;

        // Populate the child with a message and a tool_output so the cascade has something to clean
        // up. This proves the descendant deletions run, not just the parent row.
        manager
            .save_message(child, "user", "hello from sub-agent")
            .await
            .expect("save_message");
        manager
            .save_tool_output(child, "fixture", "tool body")
            .await
            .expect("save_tool_output");

        let deleted = manager.delete_session(parent).await.expect("delete parent");
        assert!(deleted);
        assert!(
            !manager.session_exists(parent).await.expect("exists check"),
            "parent should be gone"
        );
        assert!(
            !manager.session_exists(child).await.expect("exists check"),
            "child should be cascaded"
        );
        assert!(
            manager
                .load_tool_output(child, "fixture")
                .await
                .expect("load")
                .is_none(),
            "child's tool_output should be gone"
        );
    }

    // MCP TokenStore tests. Exercise the methods backing `meka mcp login/logout`. In-memory DB
    // keeps each case hermetic.

    #[tokio::test]
    async fn mcp_credentials_round_trip() {
        let manager = test_manager().await;
        let store = manager.token_store();

        assert!(
            store
                .load_mcp_credentials("srv", crate::session::McpCredentialKind::OAuth)
                .await
                .expect("load absent")
                .is_none(),
            "no credentials should exist yet"
        );

        store
            .save_mcp_credentials(
                "srv",
                crate::session::McpCredentialKind::OAuth,
                r#"{"tokens":{"access_token":"at1"}}"#,
            )
            .await
            .expect("save");
        assert_eq!(
            store
                .load_mcp_credentials("srv", crate::session::McpCredentialKind::OAuth)
                .await
                .expect("load")
                .as_deref(),
            Some(r#"{"tokens":{"access_token":"at1"}}"#)
        );

        // Upsert: second save replaces the first.
        store
            .save_mcp_credentials(
                "srv",
                crate::session::McpCredentialKind::OAuth,
                r#"{"tokens":{"access_token":"at2"}}"#,
            )
            .await
            .expect("save again");
        assert_eq!(
            store
                .load_mcp_credentials("srv", crate::session::McpCredentialKind::OAuth)
                .await
                .expect("load")
                .as_deref(),
            Some(r#"{"tokens":{"access_token":"at2"}}"#)
        );

        store.clear_mcp_credentials("srv").await.expect("clear");
        assert!(
            store
                .load_mcp_credentials("srv", crate::session::McpCredentialKind::OAuth)
                .await
                .expect("load after clear")
                .is_none()
        );
    }

    #[tokio::test]
    async fn mcp_credentials_are_scoped_per_server() {
        let manager = test_manager().await;
        let store = manager.token_store();
        store
            .save_mcp_credentials("a", crate::session::McpCredentialKind::OAuth, "alpha")
            .await
            .expect("save a");
        store
            .save_mcp_credentials("b", crate::session::McpCredentialKind::OAuth, "beta")
            .await
            .expect("save b");
        store.clear_mcp_credentials("a").await.expect("clear a");
        assert!(
            store
                .load_mcp_credentials("a", crate::session::McpCredentialKind::OAuth)
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(
            store
                .load_mcp_credentials("b", crate::session::McpCredentialKind::OAuth)
                .await
                .unwrap()
                .as_deref(),
            Some("beta")
        );
    }

    /// One server is one name in the listing, however many secrets it holds.
    ///
    /// The listing is a set of server names, and the composite key makes it tempting to forget
    /// that: a confidential OAuth client has two rows, and `meka mcp list` would name it twice in
    /// the orphaned-credential report, which reads as two different strandings to clean up.
    #[tokio::test]
    async fn a_server_with_several_kinds_is_listed_once() {
        use crate::session::McpCredentialKind;

        let manager = test_manager().await;
        let store = manager.token_store();
        for kind in [
            McpCredentialKind::Bearer,
            McpCredentialKind::ClientSecret,
            McpCredentialKind::OAuth,
        ] {
            store
                .save_mcp_credentials("api", kind, "not-a-real-secret")
                .await
                .expect("save");
        }
        store
            .save_mcp_credentials("docs", McpCredentialKind::OAuth, "not-a-real-secret")
            .await
            .expect("save");

        assert_eq!(
            store.list_mcp_credential_servers().await.expect("list"),
            vec!["api".to_string(), "docs".to_string()],
            "three kinds on one server is still one name"
        );
    }

    /// The case `PRIMARY KEY (server_name, kind)` exists for.
    ///
    /// A confidential `auth = "oauth"` client holds two secrets at once: the long-lived client
    /// secret it authenticates *with*, and the bundle it obtained. Refreshing the bundle must not
    /// touch the secret. Under a `server_name`-only key these collide, and the server would work
    /// until its first token refresh and fail from then on.
    #[tokio::test]
    async fn one_server_holds_a_client_secret_and_an_oauth_bundle_at_once() {
        use crate::session::McpCredentialKind;

        let manager = test_manager().await;
        let store = manager.token_store();

        store
            .save_mcp_credentials(
                "api",
                McpCredentialKind::ClientSecret,
                "cs-not-a-real-secret",
            )
            .await
            .expect("save client secret");
        store
            .save_mcp_credentials("api", McpCredentialKind::OAuth, r#"{"access_token":"at1"}"#)
            .await
            .expect("save bundle");

        // A refresh: rmcp's adapter compare-and-swaps the bundle it last read.
        assert!(
            store
                .replace_mcp_credentials(
                    "api",
                    r#"{"access_token":"at1"}"#,
                    r#"{"access_token":"at2"}"#,
                )
                .await
                .expect("refresh"),
            "the refresh should have matched the stored bundle"
        );

        assert_eq!(
            store
                .load_mcp_credentials("api", McpCredentialKind::OAuth)
                .await
                .expect("load bundle")
                .as_deref(),
            Some(r#"{"access_token":"at2"}"#),
            "the refresh should have moved the bundle"
        );
        assert_eq!(
            store
                .load_mcp_credentials("api", McpCredentialKind::ClientSecret)
                .await
                .expect("load client secret")
                .as_deref(),
            Some("cs-not-a-real-secret"),
            "the refresh must not have touched the client secret"
        );
    }

    /// The two strings a kind carries answer to different masters: `as_str` is schema and must
    /// never move, `label` is prose and is free to. Printing the schema token where the prose
    /// belongs is the drift worth pinning, since it reads as almost right.
    #[test]
    fn a_kind_says_one_thing_to_the_schema_and_another_to_the_user() {
        use crate::session::McpCredentialKind;

        for (kind, stored, shown) in [
            (McpCredentialKind::Bearer, "bearer", "bearer token"),
            (
                McpCredentialKind::ClientSecret,
                "client_secret",
                "client secret",
            ),
            (McpCredentialKind::OAuth, "oauth", "OAuth tokens"),
        ] {
            assert_eq!(kind.as_str(), stored, "the stored discriminator moved");
            assert_eq!(kind.label(), shown, "the label the user reads moved");
            assert_ne!(
                kind.label(),
                kind.as_str(),
                "a label that is the schema value is a schema token leaking into the UI"
            );
        }
    }

    /// `mcp login` clears before authorising, and must clear only what it is about to replace.
    #[tokio::test]
    async fn clearing_one_kind_leaves_the_others() {
        use crate::session::McpCredentialKind;

        let manager = test_manager().await;
        let store = manager.token_store();
        for (kind, secret) in [
            (McpCredentialKind::Bearer, "bearer-not-a-real-token"),
            (McpCredentialKind::ClientSecret, "cs-not-a-real-secret"),
            (McpCredentialKind::OAuth, r#"{"access_token":"at1"}"#),
        ] {
            store
                .save_mcp_credentials("api", kind, secret)
                .await
                .expect("save");
        }

        store
            .clear_mcp_credentials_of_kind("api", McpCredentialKind::OAuth)
            .await
            .expect("clear oauth");

        assert!(
            store
                .load_mcp_credentials("api", McpCredentialKind::OAuth)
                .await
                .expect("load oauth")
                .is_none()
        );
        assert!(
            store
                .load_mcp_credentials("api", McpCredentialKind::Bearer)
                .await
                .expect("load bearer")
                .is_some(),
            "clearing the OAuth bundle must leave the bearer"
        );
        assert!(
            store
                .load_mcp_credentials("api", McpCredentialKind::ClientSecret)
                .await
                .expect("load client secret")
                .is_some(),
            "clearing the OAuth bundle must leave the client secret, which the login needs"
        );
        assert!(
            store
                .has_mcp_credentials("api")
                .await
                .expect("has any after clearing one kind"),
            "two kinds remain, so the server still has credentials"
        );
    }

    #[tokio::test]
    async fn test_oauth_token_round_trip_preserves_all_fields() {
        let manager = SessionManager::open(Some(Path::new(":memory:")), &Default::default())
            .await
            .expect("memory store");
        let store = manager.token_store();

        let credential = AuthCredential::OAuthToken {
            access_token: "access-1".to_string(),
            refresh_token: Some("refresh-1".to_string()),
            expires_at: Some(1_700_000_000_000),
            account_id: Some("account-abc".to_string()),
        };

        store
            .save_provider_credential("chatgpt-subscription", &credential)
            .await
            .expect("save");

        let loaded = store
            .load_provider_credential("chatgpt-subscription")
            .await
            .expect("load")
            .expect("present");

        match loaded {
            AuthCredential::OAuthToken {
                access_token,
                refresh_token,
                expires_at,
                account_id,
            } => {
                assert_eq!(access_token, "access-1");
                assert_eq!(refresh_token.as_deref(), Some("refresh-1"));
                assert_eq!(expires_at, Some(1_700_000_000_000));
                assert_eq!(account_id.as_deref(), Some("account-abc"));
            }
            _ => panic!("expected OAuthToken"),
        }
    }

    #[tokio::test]
    async fn test_oauth_token_round_trip_account_id_optional() {
        // Claude OAuth doesn't populate `account_id`; make sure round-tripping a `None` value
        // works without losing other fields.
        let manager = SessionManager::open(Some(Path::new(":memory:")), &Default::default())
            .await
            .expect("memory store");
        let store = manager.token_store();

        let credential = AuthCredential::OAuthToken {
            access_token: "claude-token".to_string(),
            refresh_token: None,
            expires_at: None,
            account_id: None,
        };

        store
            .save_provider_credential("claude", &credential)
            .await
            .expect("save");

        let loaded = store
            .load_provider_credential("claude")
            .await
            .expect("load");

        match loaded {
            Some(AuthCredential::OAuthToken {
                access_token,
                account_id,
                ..
            }) => {
                assert_eq!(access_token, "claude-token");
                assert!(account_id.is_none());
            }
            _ => panic!("expected OAuthToken with account_id=None"),
        }
    }

    /// Two providers can persist independently with different `account_id` values. This test
    /// verifies the provider PK keeps chatgpt-subscription and a hypothetical future OAuth provider
    /// isolated.
    #[tokio::test]
    async fn test_oauth_token_two_providers_independent() {
        let manager = SessionManager::open(Some(Path::new(":memory:")), &Default::default())
            .await
            .expect("memory store");
        let store = manager.token_store();

        let codex_credential = AuthCredential::OAuthToken {
            access_token: "codex-access".to_string(),
            refresh_token: Some("codex-refresh".to_string()),
            expires_at: Some(2_000_000_000_000),
            account_id: Some("workspace-1".to_string()),
        };
        let claude_credential = AuthCredential::OAuthToken {
            access_token: "claude-access".to_string(),
            refresh_token: Some("claude-refresh".to_string()),
            expires_at: Some(3_000_000_000_000),
            account_id: None,
        };

        store
            .save_provider_credential("chatgpt-subscription", &codex_credential)
            .await
            .expect("save codex");
        store
            .save_provider_credential("claude", &claude_credential)
            .await
            .expect("save claude");

        let codex_loaded = store
            .load_provider_credential("chatgpt-subscription")
            .await
            .expect("load codex")
            .expect("present");
        let claude_loaded = store
            .load_provider_credential("claude")
            .await
            .expect("load claude")
            .expect("present");

        if let AuthCredential::OAuthToken { account_id, .. } = codex_loaded {
            assert_eq!(account_id.as_deref(), Some("workspace-1"));
        } else {
            panic!("expected OAuthToken");
        }
        if let AuthCredential::OAuthToken { account_id, .. } = claude_loaded {
            assert!(account_id.is_none());
        } else {
            panic!("expected OAuthToken");
        }
    }

    #[tokio::test]
    async fn test_api_key_credential_round_trip() {
        let manager = SessionManager::open(Some(Path::new(":memory:")), &Default::default())
            .await
            .expect("memory store");
        let store = manager.token_store();

        let credential = AuthCredential::ApiKey("sk-secret-123".to_string());
        store
            .save_provider_credential("personal", &credential)
            .await
            .expect("save");

        let loaded = store
            .load_provider_credential("personal")
            .await
            .expect("load")
            .expect("present");

        match loaded {
            AuthCredential::ApiKey(key) => assert_eq!(key, "sk-secret-123"),
            _ => panic!("expected ApiKey"),
        }
    }

    #[tokio::test]
    async fn test_delete_provider_credential_removes_entry() {
        let manager = SessionManager::open(Some(Path::new(":memory:")), &Default::default())
            .await
            .expect("memory store");
        let store = manager.token_store();

        store
            .save_provider_credential("work", &AuthCredential::ApiKey("key".to_string()))
            .await
            .expect("save");
        store
            .delete_provider_credential("work")
            .await
            .expect("delete");

        assert!(
            store
                .load_provider_credential("work")
                .await
                .expect("load")
                .is_none(),
            "credential must be gone after delete"
        );
        // Deleting a missing profile is a no-op, not an error.
        store
            .delete_provider_credential("work")
            .await
            .expect("delete missing is a no-op");
    }

    /// A refresh may only replace the credential it was derived from.
    ///
    /// Two meka processes refreshing at once both present the same refresh token, and against an
    /// issuer with a reuse window both succeed. The blind upsert this replaces left the database
    /// holding whichever finished last -- the token the issuer had already superseded -- and the
    /// symptom arrived at the *next* launch as `invalid_grant` with nothing naming the cause. The
    /// loser adopts the winner's credential instead, so the store and the issuer agree.
    #[tokio::test]
    async fn test_a_refresh_cannot_overwrite_a_credential_it_did_not_read() {
        let manager = SessionManager::open(Some(Path::new(":memory:")), &Default::default())
            .await
            .expect("memory store");
        let store = manager.token_store();
        let original = AuthCredential::ApiKey("original".to_string());
        store
            .save_provider_credential("work", &original)
            .await
            .expect("save");

        // The winner: still holds what it read, so its write lands.
        let winner = AuthCredential::ApiKey("winner".to_string());
        match store
            .replace_provider_credential("work", &original, &winner)
            .await
            .expect("swap")
        {
            CredentialWrite::Stored => {}
            other => panic!("expected the first write to land, got {:?}", other),
        }

        // The loser: derived from the same original, which the row no longer holds.
        let loser = AuthCredential::ApiKey("loser".to_string());
        match store
            .replace_provider_credential("work", &original, &loser)
            .await
            .expect("swap")
        {
            CredentialWrite::Superseded(current) => match *current {
                AuthCredential::ApiKey(key) => assert_eq!(
                    key, "winner",
                    "the loser must be handed what the row holds, to adopt rather than retry"
                ),
                other => panic!("expected an ApiKey, got {:?}", other),
            },
            other => panic!("expected the second write to be refused, got {:?}", other),
        }

        match store
            .load_provider_credential("work")
            .await
            .expect("load")
            .expect("still stored")
        {
            AuthCredential::ApiKey(key) => assert_eq!(key, "winner"),
            other => panic!("expected an ApiKey, got {:?}", other),
        }
    }

    /// A profile disconnected mid-refresh must stay disconnected. Re-creating the row would put
    /// back an account the user had just removed, and it would come back working.
    #[tokio::test]
    async fn test_a_refresh_does_not_resurrect_a_removed_profile() {
        let manager = SessionManager::open(Some(Path::new(":memory:")), &Default::default())
            .await
            .expect("memory store");
        let store = manager.token_store();
        let original = AuthCredential::ApiKey("original".to_string());
        store
            .save_provider_credential("work", &original)
            .await
            .expect("save");
        store
            .delete_provider_credential("work")
            .await
            .expect("remove");

        match store
            .replace_provider_credential("work", &original, &AuthCredential::ApiKey("new".into()))
            .await
            .expect("swap")
        {
            CredentialWrite::Gone => {}
            other => panic!(
                "expected the write to find nothing to replace, got {:?}",
                other
            ),
        }
        assert!(
            store
                .load_provider_credential("work")
                .await
                .expect("load")
                .is_none(),
            "the profile must stay removed"
        );
    }

    /// The MCP half of the same rule. rmcp hands the adapter a bare `save`, so what it compares
    /// against is the credentials it last *read* -- and a write derived from something the row no
    /// longer holds would move the stored credential backwards onto a token another process has
    /// already replaced.
    #[tokio::test]
    async fn test_an_mcp_refresh_cannot_overwrite_credentials_it_did_not_read() {
        let manager = SessionManager::open(Some(Path::new(":memory:")), &Default::default())
            .await
            .expect("memory store");
        let store = manager.token_store();
        store
            .save_mcp_credentials(
                "docs",
                crate::session::McpCredentialKind::OAuth,
                "{\"token\":\"original\"}",
            )
            .await
            .expect("save");

        assert!(
            store
                .replace_mcp_credentials(
                    "docs",
                    "{\"token\":\"original\"}",
                    "{\"token\":\"first\"}"
                )
                .await
                .expect("swap"),
            "the writer that still holds what it read wins"
        );
        assert!(
            !store
                .replace_mcp_credentials(
                    "docs",
                    "{\"token\":\"original\"}",
                    "{\"token\":\"second\"}"
                )
                .await
                .expect("swap"),
            "and the writer holding a superseded copy is refused"
        );
        assert_eq!(
            store
                .load_mcp_credentials("docs", crate::session::McpCredentialKind::OAuth)
                .await
                .expect("load")
                .as_deref(),
            Some("{\"token\":\"first\"}"),
            "the stored credential must never move backwards"
        );
    }

    /// The lock that keeps two processes from spending the same refresh token at once.
    ///
    /// Per profile, not per store: one profile refreshing must not stall an unrelated one, which is
    /// the whole reason the file is named after the profile rather than the database.
    #[tokio::test]
    async fn test_the_credential_lock_is_per_profile() {
        let manager = SessionManager::open(Some(Path::new(":memory:")), &Default::default())
            .await
            .expect("memory store");
        let store = manager.token_store();

        let held = store
            .try_lock_provider_credential("work")
            .expect("ask")
            .expect("nobody holds it");
        assert!(
            store
                .try_lock_provider_credential("work")
                .expect("ask")
                .is_none(),
            "a second holder of the same profile must be refused"
        );
        assert!(
            store
                .try_lock_provider_credential("personal")
                .expect("ask")
                .is_some(),
            "a different profile is a different lock"
        );

        drop(held);
        assert!(
            store
                .try_lock_provider_credential("work")
                .expect("ask")
                .is_some(),
            "and it is released when the holder goes"
        );
    }

    /// A profile name is a TOML table key with no charset rule behind it, so it cannot be a file
    /// name directly: `[providers."../../etc/passwd"]` would otherwise name a path. Stripping is
    /// what keeps the file inside the lock directory, and the hash is what keeps two names that
    /// strip alike from sharing a lock -- one profile's refresh blocking an unrelated one's, for as
    /// long as it takes.
    #[tokio::test]
    async fn test_the_credential_lock_handles_a_profile_name_that_is_not_a_file_name() {
        let manager = SessionManager::open(Some(Path::new(":memory:")), &Default::default())
            .await
            .expect("memory store");
        let store = manager.token_store();

        let _escaping = store
            .try_lock_provider_credential("../../etc/passwd")
            .expect("ask")
            .expect("nobody holds it");
        let inside: Vec<_> = std::fs::read_dir(&manager.lock_dir)
            .expect("read the lock dir")
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(PROVIDER_LOCK_PREFIX)
            })
            .collect();
        assert_eq!(
            inside.len(),
            1,
            "a path-shaped profile name must still land inside the lock directory"
        );

        // `a/b` and `a.b` both strip to `ab`, so without the hash they would be one lock.
        let _first = store
            .try_lock_provider_credential("a/b")
            .expect("ask")
            .expect("nobody holds it");
        assert!(
            store
                .try_lock_provider_credential("a.b")
                .expect("ask")
                .is_some(),
            "two profiles that strip to the same readable name must not share a lock"
        );
    }

    /// The listing queries exist so a credential whose config entry was deleted by hand can still
    /// be named. Nothing else in the codebase enumerates either table, so an unlisted row is an
    /// invisible one: a live API key or OAuth refresh token no surface can report or remove.
    #[tokio::test]
    async fn test_credential_listings_name_every_stored_row() {
        let manager = SessionManager::open(Some(Path::new(":memory:")), &Default::default())
            .await
            .expect("memory store");
        let store = manager.token_store();

        assert!(
            store
                .list_credential_profiles()
                .await
                .expect("list empty")
                .is_empty()
        );
        assert!(
            store
                .list_mcp_credential_servers()
                .await
                .expect("list empty")
                .is_empty()
        );

        store
            .save_provider_credential("work", &AuthCredential::ApiKey("key".to_string()))
            .await
            .expect("save work");
        store
            .save_provider_credential("archive", &AuthCredential::ApiKey("key".to_string()))
            .await
            .expect("save archive");
        store
            .save_mcp_credentials(
                "linear",
                crate::session::McpCredentialKind::OAuth,
                r#"{"tokens":{"access_token":"at"}}"#,
            )
            .await
            .expect("save linear");

        // Sorted, so the reported order doesn't depend on insertion order.
        assert_eq!(store.list_credential_profiles().await.expect("list"), vec![
            "archive".to_string(),
            "work".to_string()
        ]);
        assert_eq!(
            store.list_mcp_credential_servers().await.expect("list"),
            vec!["linear".to_string()]
        );

        // The two tables are independent: clearing one must not hide rows in the other.
        store
            .delete_provider_credential("work")
            .await
            .expect("delete");
        assert_eq!(store.list_credential_profiles().await.expect("list"), vec![
            "archive".to_string()
        ]);
        assert_eq!(
            store.list_mcp_credential_servers().await.expect("list"),
            vec!["linear".to_string()]
        );
    }

    async fn job_fixture(
        manager: &SessionManager,
        session_id: Uuid,
        schedule: crate::schedule::Schedule,
        gate: Option<crate::schedule::Gate>,
    ) -> crate::schedule::ScheduledJob {
        let created_at = chrono::Utc::now();
        let next_fire_at = schedule.next_after(created_at).expect("has a next fire");
        let job = crate::schedule::ScheduledJob {
            attempts: 0,
            id: Uuid::new_v4().to_string(),
            session_id,
            schedule,
            prompt: "check the deploy".to_string(),
            gate,
            created_at,
            last_fired_at: None,
            next_fire_at,
        };
        manager
            .schedule_store()
            .create_scheduled_job(&job)
            .await
            .expect("create scheduled job");
        job
    }

    #[tokio::test]
    async fn test_scheduled_job_round_trips_through_the_database() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let written = job_fixture(
            &manager,
            session_id,
            crate::schedule::Schedule::parse_every("30m").expect("parses"),
            Some(crate::schedule::Gate {
                probe: crate::schedule::GateProbe::Shell {
                    command: "gh pr checks 123".to_string(),
                },
                predicate: crate::schedule::GatePredicate::Changed,
                last_output: None,
                permission: crate::permission::Permission::Unrestricted,
            }),
        )
        .await;

        let jobs = manager
            .schedule_store()
            .list_scheduled_jobs(session_id)
            .await
            .expect("list jobs");
        assert_eq!(jobs.len(), 1);
        let read = &jobs[0];
        assert_eq!(read.id, written.id);
        assert_eq!(read.prompt, "check the deploy");
        assert_eq!(read.schedule.spec(), written.schedule.spec());
        let gate = read.gate.as_ref().expect("gate survived the round trip");
        assert_eq!(gate.probe, crate::schedule::GateProbe::Shell {
            command: "gh pr checks 123".to_string(),
        });
        assert_eq!(gate.predicate, crate::schedule::GatePredicate::Changed);
    }

    async fn task_fixture(
        manager: &SessionManager,
        session_id: Uuid,
        label: &str,
    ) -> crate::background::BackgroundTask {
        let task = crate::background::BackgroundTask {
            id: Uuid::new_v4().to_string(),
            session_id,
            tool_name: "execute_command".to_string(),
            label: label.to_string(),
            status: crate::background::TaskStatus::Running,
            outcome: None,
            scratchpad_name: None,
            started_at: chrono::Utc::now(),
            finished_at: None,
            delivered_at: None,
        };
        manager
            .background_store()
            .start_background_task(&task)
            .await
            .expect("start background task");
        task
    }

    #[tokio::test]
    async fn test_background_task_round_trips_through_the_database() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let written = task_fixture(&manager, session_id, "cargo test --all").await;

        let running = manager
            .background_store()
            .list_running_background_tasks(session_id)
            .await
            .expect("list running");
        assert_eq!(running.len(), 1);
        assert_eq!(running[0].id, written.id);
        assert_eq!(running[0].label, "cargo test --all");

        manager
            .background_store()
            .finish_background_task(
                &written.id,
                crate::background::TaskStatus::Completed,
                Some("42 passed".to_string()),
                Some("task_log".to_string()),
            )
            .await
            .expect("finish");

        let undelivered = manager
            .background_store()
            .list_undelivered_background_tasks(session_id)
            .await
            .expect("list undelivered");
        assert_eq!(undelivered.len(), 1);
        assert_eq!(
            undelivered[0].status,
            crate::background::TaskStatus::Completed
        );
        assert_eq!(undelivered[0].outcome.as_deref(), Some("42 passed"));
        assert_eq!(undelivered[0].scratchpad_name.as_deref(), Some("task_log"));
        assert!(undelivered[0].finished_at.is_some());
        assert!(
            manager
                .background_store()
                .list_running_background_tasks(session_id)
                .await
                .expect("list running")
                .is_empty()
        );
    }

    /// The whole point of `delivered_at`: an outcome reaches the conversation once, including
    /// across a restart that re-runs the delivery poll.
    #[tokio::test]
    async fn test_a_delivered_outcome_is_not_delivered_again() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let task = task_fixture(&manager, session_id, "sleep 60").await;
        manager
            .background_store()
            .finish_background_task(
                &task.id,
                crate::background::TaskStatus::Completed,
                None,
                None,
            )
            .await
            .expect("finish");

        manager
            .background_store()
            .mark_background_tasks_delivered(std::slice::from_ref(&task.id))
            .await
            .expect("mark delivered");

        assert!(
            manager
                .background_store()
                .list_undelivered_background_tasks(session_id)
                .await
                .expect("list undelivered")
                .is_empty()
        );
    }

    /// A task in flight when the process died would otherwise leave the agent waiting on a report
    /// that can never arrive, having usually already promised one.
    #[tokio::test]
    async fn test_the_sweep_retires_tasks_left_running_by_a_dead_process() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        task_fixture(&manager, session_id, "sleep 600").await;

        let swept = manager
            .background_store()
            .sweep_interrupted_background_tasks(session_id)
            .await
            .expect("sweep");
        assert_eq!(swept, 1);

        let undelivered = manager
            .background_store()
            .list_undelivered_background_tasks(session_id)
            .await
            .expect("list undelivered");
        assert_eq!(undelivered.len(), 1);
        assert_eq!(
            undelivered[0].status,
            crate::background::TaskStatus::Interrupted,
            "the agent must be told the work stopped, not left waiting"
        );

        // Idempotent: a second attach must not re-report what the first already retired.
        assert_eq!(
            manager
                .background_store()
                .sweep_interrupted_background_tasks(session_id)
                .await
                .expect("second sweep"),
            0
        );
    }

    /// The shape a `--oneshot` resume hits: the previous process died mid-task, so the sweep
    /// produces an outcome that no task in *this* process is waiting on. A host that only looked
    /// for outcomes when it had started something itself would answer the prompt and exit with that
    /// report still sitting undelivered.
    #[tokio::test]
    async fn test_a_swept_outcome_is_pending_for_a_process_that_started_nothing() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        task_fixture(&manager, session_id, "sleep 400").await;

        // A fresh process takes the session: it holds no handles, and the sweep is all it knows.
        manager
            .background_store()
            .sweep_interrupted_background_tasks(session_id)
            .await
            .expect("sweep");

        let ready = manager
            .background_store()
            .list_undelivered_background_tasks(session_id)
            .await
            .expect("list undelivered");
        assert_eq!(ready.len(), 1, "the report must be waiting to be collected");
        assert_eq!(ready[0].status, crate::background::TaskStatus::Interrupted);
    }

    /// A cancelled task whose work happens to finish a moment later must not overwrite the
    /// cancellation and report success.
    #[tokio::test]
    async fn test_the_first_terminal_write_wins() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let task = task_fixture(&manager, session_id, "sleep 600").await;

        manager
            .background_store()
            .finish_background_task(
                &task.id,
                crate::background::TaskStatus::Cancelled,
                None,
                None,
            )
            .await
            .expect("cancel");
        manager
            .background_store()
            .finish_background_task(
                &task.id,
                crate::background::TaskStatus::Completed,
                Some("finished anyway".to_string()),
                None,
            )
            .await
            .expect("late completion is accepted but ignored");

        let undelivered = manager
            .background_store()
            .list_undelivered_background_tasks(session_id)
            .await
            .expect("list undelivered");
        assert_eq!(
            undelivered[0].status,
            crate::background::TaskStatus::Cancelled
        );
        assert!(undelivered[0].outcome.is_none());
    }

    #[tokio::test]
    async fn test_resolve_background_task_by_prefix() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let task = task_fixture(&manager, session_id, "sleep 60").await;

        let found = manager
            .background_store()
            .resolve_background_task(session_id, &task.id[..8])
            .await
            .expect("resolve")
            .expect("matched");
        assert_eq!(found.id, task.id);

        assert!(
            manager
                .background_store()
                .resolve_background_task(session_id, "zzzzzzzz")
                .await
                .expect("resolve")
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_deleting_a_session_cascades_to_its_background_tasks() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        task_fixture(&manager, session_id, "sleep 600").await;

        manager
            .delete_session(session_id)
            .await
            .expect("delete session");

        assert!(
            manager
                .background_store()
                .list_background_tasks(session_id)
                .await
                .expect("list")
                .is_empty()
        );
    }

    /// Jobs are keyed to the conversation that asked for them, so deleting the session must not
    /// leave a scheduler entry that fires into nothing.
    #[tokio::test]
    async fn test_deleting_a_session_cascades_to_its_scheduled_jobs() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        job_fixture(
            &manager,
            session_id,
            crate::schedule::Schedule::parse_every("30m").expect("parses"),
            None,
        )
        .await;

        manager
            .delete_session(session_id)
            .await
            .expect("delete session");

        let due = manager
            .schedule_store()
            .list_due_scheduled_jobs(chrono::Utc::now() + chrono::Duration::days(365))
            .await
            .expect("list due");
        assert!(due.is_empty(), "the job should have cascaded away");
    }

    /// `meka schedule list` and `cancel` work from a job id, so they need every job regardless of
    /// when it is due. An earlier version approximated that with "due within the next century",
    /// which left a job scheduled past that horizon invisible and therefore uncancellable.
    #[tokio::test]
    async fn test_list_all_includes_jobs_beyond_any_due_horizon() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let far_future = chrono::Utc::now() + chrono::Duration::days(365 * 200);
        job_fixture(
            &manager,
            session_id,
            crate::schedule::Schedule::At(far_future),
            None,
        )
        .await;

        assert!(
            manager
                .schedule_store()
                .list_due_scheduled_jobs(chrono::Utc::now() + chrono::Duration::days(365 * 100))
                .await
                .expect("list due")
                .is_empty(),
            "fixture must actually sit beyond the horizon the old query used"
        );
        assert_eq!(
            manager
                .schedule_store()
                .list_all_scheduled_jobs()
                .await
                .expect("list all")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn test_list_due_selects_only_jobs_whose_time_has_come() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        job_fixture(
            &manager,
            session_id,
            crate::schedule::Schedule::parse_every("1h").expect("parses"),
            None,
        )
        .await;

        let none_yet = manager
            .schedule_store()
            .list_due_scheduled_jobs(chrono::Utc::now())
            .await
            .expect("list due");
        assert!(none_yet.is_empty(), "not due for another hour");

        let later = manager
            .schedule_store()
            .list_due_scheduled_jobs(chrono::Utc::now() + chrono::Duration::hours(2))
            .await
            .expect("list due");
        assert_eq!(later.len(), 1);
    }

    /// The anchoring rule at the storage layer: claiming an occurrence records when the job is next
    /// due and stamping the fire records when it fired, and a subsequent read reconstructs the same
    /// anchor the in-memory scheduler held. Without the `last_fired_at` half, a restart would
    /// re-anchor on `created_at` and replay every occurrence since.
    #[tokio::test]
    async fn test_stamping_a_fire_persists_the_anchor_for_the_next_process() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let job = job_fixture(
            &manager,
            session_id,
            crate::schedule::Schedule::parse_every("1h").expect("parses"),
            None,
        )
        .await;

        let fired_at = chrono::Utc::now();
        let next = job.schedule.next_after(fired_at).expect("has a next fire");
        assert!(
            manager
                .schedule_store()
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "this-process",
                    fired_at,
                    fired_at + chrono::Duration::hours(1),
                )
                .await
                .expect("claim the occurrence"),
            "the occurrence is unclaimed, so this process takes it"
        );
        manager
            .schedule_store()
            .complete_claim(&job.id, "this-process", Some(next), Some(fired_at), None)
            .await
            .expect("record the delivery");

        let reloaded = manager
            .schedule_store()
            .list_scheduled_jobs(session_id)
            .await
            .expect("list jobs");
        let reloaded = reloaded.first().expect("job still present");
        assert!(reloaded.last_fired_at.is_some());
        assert_eq!(
            reloaded.anchor(),
            reloaded.last_fired_at.unwrap_or_default()
        );
        // One hour on from the fire, not from creation.
        assert_eq!(
            (reloaded.next_fire_at - fired_at).num_seconds(),
            chrono::Duration::hours(1).num_seconds()
        );
    }

    #[tokio::test]
    async fn test_cancel_resolves_an_id_prefix_and_reports_ambiguity() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let job = job_fixture(
            &manager,
            session_id,
            crate::schedule::Schedule::parse_every("30m").expect("parses"),
            None,
        )
        .await;

        assert!(
            manager
                .schedule_store()
                .cancel_scheduled_job(session_id, "nomatch")
                .await
                .expect("cancel runs")
                .is_none()
        );

        let cancelled = manager
            .schedule_store()
            .cancel_scheduled_job(session_id, job.short_id())
            .await
            .expect("cancel runs")
            .expect("prefix matched");
        assert_eq!(cancelled, job.id);
        assert!(
            manager
                .schedule_store()
                .list_scheduled_jobs(session_id)
                .await
                .expect("list jobs")
                .is_empty()
        );
    }

    /// A half-written gate is corruption, not "no gate": treating it as an ungated job would
    /// silently promote a watcher into an unconditional timer that fires every interval.
    #[tokio::test]
    async fn test_a_row_with_a_half_written_gate_is_skipped_not_downgraded() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let job = job_fixture(
            &manager,
            session_id,
            crate::schedule::Schedule::parse_every("30m").expect("parses"),
            Some(crate::schedule::Gate {
                probe: crate::schedule::GateProbe::Shell {
                    command: "true".to_string(),
                },
                predicate: crate::schedule::GatePredicate::Succeeded,
                last_output: None,
                permission: crate::permission::Permission::Unrestricted,
            }),
        )
        .await;

        // Prove the row is readable before the corruption, so an empty result afterwards can only
        // be the decoder rejecting it.
        assert_eq!(
            manager
                .schedule_store()
                .list_scheduled_jobs(session_id)
                .await
                .expect("list jobs")
                .len(),
            1
        );

        let id = job.id.clone();
        manager
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE scheduled_jobs SET gate_spec = NULL WHERE id = ?1",
                    rusqlite::params![id],
                )?;
                Ok(())
            })
            .await
            .expect("corrupt the row");

        assert!(
            manager
                .schedule_store()
                .list_scheduled_jobs(session_id)
                .await
                .expect("list jobs")
                .is_empty(),
            "the corrupt row is skipped rather than read as ungated"
        );
    }

    /// A name already taken is stepped over rather than reused, so an earlier backup survives a
    /// second attempt.
    #[test]
    fn a_backup_name_already_in_use_is_not_reused() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let database_path = temp_dir.path().join("meka.db");
        assert_eq!(
            free_backup_path(&database_path, 1).expect("a free name"),
            temp_dir.path().join("meka.db.v1.bak"),
            "the unsuffixed name when nothing is there"
        );

        std::fs::write(
            temp_dir.path().join("meka.db.v1.bak"),
            b"an earlier attempt",
        )
        .expect("plant");
        assert_eq!(
            free_backup_path(&database_path, 1).expect("a free name"),
            temp_dir.path().join("meka.db.v1.bak.1")
        );
        std::fs::write(temp_dir.path().join("meka.db.v1.bak.1"), b"and another").expect("plant");
        assert_eq!(
            free_backup_path(&database_path, 1).expect("a free name"),
            temp_dir.path().join("meka.db.v1.bak.2")
        );
    }

    /// `Path::exists` follows symlinks and so reports `false` for a dangling one, which would hand
    /// back a name that is really a redirection out of the data directory. The whole credential
    /// store went with it when this was measured.
    #[cfg(unix)]
    #[test]
    fn a_dangling_symlink_does_not_count_as_a_free_backup_name() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let database_path = temp_dir.path().join("meka.db");
        let taken = temp_dir.path().join("meka.db.v1.bak");
        std::os::unix::fs::symlink(temp_dir.path().join("nowhere"), &taken).expect("symlink");
        assert!(
            !taken.exists(),
            "the premise: the link dangles, so `exists` says no"
        );

        assert_eq!(
            free_backup_path(&database_path, 1).expect("a free name"),
            temp_dir.path().join("meka.db.v1.bak.1"),
            "a dangling symlink occupies the name as surely as a file does"
        );
    }

    /// Debris from an interrupted upgrade must not wedge the next one.
    ///
    /// The copy is staged at `<name>.partial`, so a crash between creating it and renaming it into
    /// place leaves that file behind. Checking only the target name then reused it, `create_new`
    /// failed `EEXIST`, and *every* later start refused with the store still unmigrated: one Ctrl-C
    /// during the upgrade this release is built around bricked meka permanently. Reproduced against
    /// the real binary before this guard existed.
    #[test]
    fn a_staging_file_left_by_a_crash_does_not_block_the_next_attempt() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let database_path = temp_dir.path().join("meka.db");
        std::fs::write(
            temp_dir.path().join("meka.db.v1.bak.partial"),
            b"an interrupted copy",
        )
        .expect("plant the debris");

        assert_eq!(
            free_backup_path(&database_path, 1).expect("a usable name is still found"),
            temp_dir.path().join("meka.db.v1.bak.1"),
            "the pair is taken, so the next pair is used rather than failing forever"
        );
    }

    /// Running out of names is an error, not a fallback to the occupied one.
    ///
    /// It used to be safe to return the base name, because `VACUUM INTO` refuses a non-empty
    /// target. Staging moved the write to `.partial` and the final step to `std::fs::rename`, which
    /// refuses nothing, so the old fallthrough silently overwrote the *oldest* backup: the one most
    /// likely to be the one that mattered.
    #[test]
    fn running_out_of_backup_names_refuses_rather_than_overwriting_the_oldest() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let database_path = temp_dir.path().join("meka.db");
        std::fs::write(
            temp_dir.path().join("meka.db.v1.bak"),
            b"the irreplaceable one",
        )
        .expect("plant");
        for suffix in 1..1000 {
            std::fs::write(
                temp_dir.path().join(format!("meka.db.v1.bak.{suffix}")),
                b"x",
            )
            .expect("plant");
        }

        let error = free_backup_path(&database_path, 1).expect_err("no name is available");
        assert!(
            error.to_string().contains("Nothing has been changed"),
            "{error}"
        );
        assert_eq!(
            std::fs::read(temp_dir.path().join("meka.db.v1.bak")).expect("still there"),
            b"the irreplaceable one",
            "the oldest copy must survive"
        );
    }

    /// Best-effort, and both halves matter: it clears a copy that never finished, and it stays
    /// quiet when there is nothing to clear, because the caller is already returning the error
    /// that stopped the migration.
    #[test]
    fn an_incomplete_backup_is_cleared_and_a_missing_one_is_not_an_error() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let staging = temp_dir.path().join("meka.db.v1.bak.partial");
        std::fs::write(&staging, b"half a copy").expect("plant");
        remove_partial_backup(&staging);
        assert!(!staging.exists(), "the incomplete copy is gone");
        remove_partial_backup(&staging);
    }

    /// When no backup can be taken, the migration refuses and the store is left where it was.
    ///
    /// Reached here by exhausting every name, which is the one way to make the copy impossible that
    /// a test can force deterministically. Two earlier attempts are worth recording because both
    /// stopped working, and for good reasons: a read-only data directory does not work because
    /// `SessionManager::open` calls `restrict_permissions(parent, 0o700)` on the way in and widens
    /// it back; occupying the staging name with a directory no longer works because
    /// `free_backup_path` now treats a taken `.partial` as a taken pair and steps past it, which is
    /// the fix for the wedge described on that function.
    #[tokio::test]
    async fn a_migration_that_cannot_be_backed_up_refuses_and_changes_nothing() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let database_path = temp_dir.path().join("meka.db");
        {
            let connection = rusqlite::Connection::open(&database_path).expect("open");
            connection
                .execute_batch(crate::session::migrations::baseline_for_test())
                .expect("a store with work pending");
        }
        std::fs::write(temp_dir.path().join("meka.db.v1.bak"), b"the oldest copy").expect("plant");
        for suffix in 1..1000 {
            std::fs::write(
                temp_dir.path().join(format!("meka.db.v1.bak.{suffix}")),
                b"x",
            )
            .expect("plant");
        }

        let error = SessionManager::open(Some(&database_path), &Default::default())
            .await
            .err()
            .expect("the migration refuses rather than proceeding without a backup");
        assert!(
            error.to_string().contains("Nothing has been changed"),
            "{error}"
        );

        let connection = rusqlite::Connection::open(&database_path).expect("reopen");
        let version: i64 = connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .expect("version");
        assert_eq!(version, 0, "no migration ran");
        let old_column: i64 = connection
            .query_row(
                "SELECT count(*) FROM pragma_table_info('scheduled_jobs') WHERE name = 'gate_command'",
                [],
                |row| row.get(0),
            )
            .expect("count");
        assert_eq!(old_column, 1, "the conversion did not start");
        assert_eq!(
            std::fs::read(temp_dir.path().join("meka.db.v1.bak")).expect("still there"),
            b"the oldest copy",
            "and the existing copies are untouched"
        );
    }

    /// A store carried forward keeps a copy of what it was, without the user having asked.
    ///
    /// The instruction the retired script's documentation had to give ("take a backup first") is
    /// the one a user skips, and it is only needed on the one run that might go wrong. Doing it
    /// here means it happened.
    #[tokio::test]
    async fn an_upgraded_store_leaves_a_copy_of_what_it_was() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let database_path = temp_dir.path().join("meka.db");
        {
            // A store as an older meka left one: the baseline shape, and nothing in `user_version`.
            let connection = rusqlite::Connection::open(&database_path).expect("open");
            connection
                .execute_batch(crate::session::migrations::baseline_for_test())
                .expect("the baseline builds");
        }

        let manager = SessionManager::open(Some(&database_path), &Default::default())
            .await
            .expect("the store migrates on open");
        drop(manager);

        let backup = temp_dir.path().join("meka.db.v1.bak");
        assert!(
            backup.exists(),
            "the pre-migration copy should be beside the store"
        );
        let copy = rusqlite::Connection::open(&backup).expect("the copy opens");
        let version: i64 = copy
            .query_row("SELECT * FROM pragma_user_version", [], |row| row.get(0))
            .expect("the copy has a version");
        assert_eq!(
            version, 0,
            "the copy must identify itself as what it was, so restoring it migrates once rather \
             than being mistaken for a current store"
        );
    }

    /// [`parse_number`]'s own contract, which its callers only partly exercise.
    ///
    /// Both call sites reject zero themselves, so nothing downstream can tell `Some(0)` from `None`
    /// here, and a mutation sweep found exactly that: widening the leading-zero guard's
    /// `digits.len() > 1` to `>= 1`, which makes a bare `0` a non-match, survived the entire suite.
    /// It is worth pinning anyway, because this function's job is to read back what `Display` wrote
    /// and `Display` writes zero as `0`. A guard that is only correct because of who happens to
    /// call it is the kind that drifts.
    #[test]
    fn a_number_is_parsed_exactly_as_display_would_have_written_it() {
        assert_eq!(parse_number(b"0"), Some(0), "`Display` writes zero as `0`");
        assert_eq!(parse_number(b"1"), Some(1));
        assert_eq!(parse_number(b"999"), Some(999));
        assert_eq!(
            parse_number(u32::MAX.to_string().as_bytes()),
            Some(u32::MAX)
        );
        assert_eq!(parse_number(b""), None, "no digits at all is not a number");
        assert_eq!(parse_number(b"01"), None, "`Display` never pads");
        assert_eq!(parse_number(b"007"), None);
        assert_eq!(
            parse_number(b"4294967296"),
            None,
            "one past `u32::MAX` is a non-match rather than a wrap"
        );
        assert_eq!(parse_number(b"99999999999999999999"), None);
    }

    /// Exactly which names this module will delete, stated as a table.
    ///
    /// The cheapest guard on the whole change, and the one that matters most: everything else
    /// decides *when* to prune, and this decides *what*. A match that is too generous deletes a
    /// file meka did not write, which is the one outcome the feature must never have.
    ///
    /// `meka.db.vault.bak` earns its row. It is the near-miss the digit requirement exists for:
    /// drop that requirement and `.v` followed by anything at all becomes a backup.
    #[test]
    fn only_names_this_module_writes_are_recognised_as_backups() {
        let store = std::ffi::OsStr::new("meka.db");
        let matches = |name: &str| is_backup_name(store, std::ffi::OsStr::new(name));

        for name in ["meka.db.v1.bak", "meka.db.v42.bak", "meka.db.v1.bak.7"] {
            assert!(matches(name), "{name} is a name free_backup_path builds");
        }
        for name in [
            "meka.db",
            "meka.db-wal",
            "meka.db-shm",
            "meka.db.mine.bak",
            // `.v` then a word, not a version.
            "meka.db.vault.bak",
            // Staging is deliberately spared: `free_backup_path` reads it to step past a name.
            "meka.db.v1.bak.partial",
            // A version or suffix has to be digits, and the suffix has to end the name.
            "meka.db.v.bak",
            "meka.db.v1.bak.",
            "meka.db.v1.bak.x",
            "meka.db.v1.bak.1.2",
            // Digits are not enough: these are shapes `free_backup_path` cannot emit, so a file
            // wearing one belongs to whoever made it. `.bak.0` is the near-miss that matters,
            // being how a person numbering their own archive from zero would name it.
            "meka.db.v1.bak.0",
            "meka.db.v0.bak",
            "meka.db.v01.bak",
            "meka.db.v1.bak.01",
            "meka.db.v1.bak.1000",
            "meka.db.v99999999999999999999.bak",
            // Another store's backup in a shared directory.
            "other.db.v1.bak",
            "notes.txt",
        ] {
            assert!(!matches(name), "{name} is not meka's to delete");
        }
    }

    /// One copy survives an upgrade, not one per release.
    ///
    /// The planted names are what a store carried through two earlier releases looks like, and they
    /// also force the interesting interaction: `free_backup_path` steps past both to `.bak.2`, so
    /// the copy being kept is *not* the name the older ones wear. A prune keyed to the name rather
    /// than to the path would take the wrong file.
    #[tokio::test]
    async fn only_the_newest_pre_migration_copy_survives() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let database_path = temp_dir.path().join("meka.db");
        {
            let connection = rusqlite::Connection::open(&database_path).expect("open");
            connection
                .execute_batch(crate::session::migrations::baseline_for_test())
                .expect("a store with work pending");
        }
        std::fs::write(temp_dir.path().join("meka.db.v1.bak"), b"older").expect("plant");
        std::fs::write(temp_dir.path().join("meka.db.v1.bak.1"), b"less old").expect("plant");

        drop(
            SessionManager::open(Some(&database_path), &Default::default())
                .await
                .expect("the store migrates on open"),
        );

        let kept = temp_dir.path().join("meka.db.v1.bak.2");
        assert!(kept.exists(), "the copy just taken is the one that stays");
        assert!(
            !temp_dir.path().join("meka.db.v1.bak").exists()
                && !temp_dir.path().join("meka.db.v1.bak.1").exists(),
            "and the copies it supersedes are gone"
        );
        let copy = rusqlite::Connection::open(&kept).expect("the survivor opens");
        let version: i64 = copy
            .query_row("SELECT * FROM pragma_user_version", [], |row| row.get(0))
            .expect("the copy has a version");
        assert_eq!(version, 0, "and it is the real copy, not a planted file");
    }

    /// Everything else beside the store is left exactly as it was.
    ///
    /// A data directory is the user's, and a migration that tidies it is a migration that deletes
    /// something one day. The `.partial` is here because sparing it is load-bearing rather than
    /// incidental: `free_backup_path` treats a taken staging name as a taken pair, which is what
    /// makes it step to `.bak.1` below.
    #[tokio::test]
    async fn a_file_meka_did_not_write_is_left_alone() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let database_path = temp_dir.path().join("meka.db");
        {
            let connection = rusqlite::Connection::open(&database_path).expect("open");
            connection
                .execute_batch(crate::session::migrations::baseline_for_test())
                .expect("a store with work pending");
        }
        let bystanders = [
            "meka.db.mine.bak",
            // `.v` with no version at all: the near-miss that a match without the digit
            // requirement swallows, which is why it is here and not only in the table test.
            "meka.db.v.bak",
            "meka.db.vault.bak",
            "other.db.v1.bak",
            "notes.txt",
            "meka.db.v1.bak.partial",
        ];
        for name in bystanders {
            std::fs::write(temp_dir.path().join(name), name.as_bytes()).expect("plant");
        }

        drop(
            SessionManager::open(Some(&database_path), &Default::default())
                .await
                .expect("the store migrates on open"),
        );

        for name in bystanders {
            let path = temp_dir.path().join(name);
            assert_eq!(
                std::fs::read(&path).expect("still there").as_slice(),
                name.as_bytes(),
                "{name} is not meka's to delete"
            );
        }
        assert!(
            temp_dir.path().join("meka.db.v1.bak.1").exists(),
            "and the copy stepped past the occupied staging name, as it always did"
        );
    }

    /// A prune that cannot delete is a warning, not a failed upgrade.
    ///
    /// Driven directly rather than through `SessionManager::open`, because open widens the data
    /// directory back to `0700` on its way in (`restrict_permissions`), so a read-only parent
    /// cannot be staged through it. The property under test belongs to the helper anyway: it must
    /// return, having done what it could, whatever the filesystem says.
    #[cfg(unix)]
    #[test]
    fn a_prune_that_cannot_delete_still_returns() {
        use std::os::unix::fs::PermissionsExt;

        let temp_dir = tempfile::tempdir().expect("tempdir");
        let directory = temp_dir.path().join("data");
        std::fs::create_dir(&directory).expect("create");
        let database_path = directory.join("meka.db");
        std::fs::write(&database_path, b"store").expect("plant");
        let keep = directory.join("meka.db.v2.bak");
        std::fs::write(&keep, b"newest").expect("plant");
        let doomed = directory.join("meka.db.v1.bak");
        std::fs::write(&doomed, b"older").expect("plant");

        let original = std::fs::metadata(&directory).expect("stat").permissions();
        std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o500))
            .expect("make it read-only");
        prune_older_backups(&database_path, &keep);
        std::fs::set_permissions(&directory, original).expect("restore so the tempdir can go");

        assert!(
            doomed.exists(),
            "it could not delete, which is the point: it must not have panicked or unwound either"
        );
    }

    /// Nothing is pruned unless a fresh copy actually landed.
    ///
    /// The failure this needs is a narrow one: it has to happen *after* `free_backup_path` has
    /// chosen a name and *before* the rename, since anything earlier or later cannot tell a
    /// prune-then-copy ordering from the copy-then-prune one. A store path that is not valid UTF-8
    /// is exactly that: `back_up_before_migrating` gets as far as `staging.to_str()` and refuses
    /// there, with a name already picked and no copy written.
    ///
    /// An earlier version of this module's comments claimed no such test existed and rested the
    /// guarantee on control flow alone. Control flow is still what enforces it; this is the test
    /// that would notice if someone rearranged that.
    #[cfg(unix)]
    #[tokio::test]
    async fn nothing_is_pruned_when_no_fresh_copy_landed() {
        use std::{ffi::OsString, os::unix::ffi::OsStringExt};

        let temp_dir = tempfile::tempdir().expect("tempdir");
        // A directory whose name is not valid UTF-8, so every path under it is not either.
        let mut name = OsString::from_vec(b"not-utf8-\xff".to_vec());
        name.push("");
        let directory = temp_dir.path().join(name);
        std::fs::create_dir(&directory).expect("create");
        let database_path = directory.join("meka.db");
        {
            let connection = rusqlite::Connection::open(&database_path).expect("open");
            connection
                .execute_batch(crate::session::migrations::baseline_for_test())
                .expect("a store with work pending");
        }
        let planted = directory.join("meka.db.v1.bak");
        std::fs::write(&planted, b"the only copy").expect("plant");

        let error = SessionManager::open(Some(&database_path), &Default::default())
            .await
            .err()
            .expect("a backup that cannot be written refuses the migration");
        assert!(
            error.to_string().contains("not valid UTF-8"),
            "the refusal should be the one this test aims at, not some other: {error}"
        );
        assert_eq!(
            std::fs::read(&planted).expect("still there"),
            b"the only copy",
            "a prune that ran before the copy landed would have left the user with none"
        );
    }

    /// A directory that is not there at all is not an error, for the same reason.
    #[test]
    fn a_prune_with_nowhere_to_look_is_not_an_error() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let missing = temp_dir.path().join("gone").join("meka.db");
        prune_older_backups(
            &missing,
            &temp_dir.path().join("gone").join("meka.db.v1.bak"),
        );
    }

    /// A backup protects data that already exists, so neither a first run nor a subsequent one
    /// leaves a copy: the first has nothing to lose, and the second has nothing to do. Copying on
    /// every start would double the store on disk and make opening meka cost more as the
    /// conversation grows.
    #[tokio::test]
    async fn a_store_with_nothing_to_preserve_is_not_backed_up() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let database_path = temp_dir.path().join("meka.db");
        drop(
            SessionManager::open(Some(&database_path), &Default::default())
                .await
                .expect("first open"),
        );
        drop(
            SessionManager::open(Some(&database_path), &Default::default())
                .await
                .expect("second open"),
        );

        let copies: Vec<_> = std::fs::read_dir(temp_dir.path())
            .expect("readable")
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".bak"))
            .collect();
        assert!(copies.is_empty(), "expected no backups, found {copies:?}");
    }

    /// Two hosts starting together against one unmigrated store. The schema lock is what makes the
    /// loser re-read the winner's answer instead of acting on its own stale one, and a migration
    /// applied twice is how a store gets a duplicated column or a half-converted table.
    ///
    /// Spawned rather than `join!`ed, and that is not a style choice. `initialize_schema` holds a
    /// *blocking* file lock across an `await`, so two opens driven by one task deadlock: the second
    /// blocks the thread that the first needs in order to be polled again and release. Separate
    /// tasks put them on separate workers, which is also what two real hosts are.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn two_hosts_opening_one_unmigrated_store_migrate_it_once() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let database_path = temp_dir.path().join("meka.db");
        {
            let connection = rusqlite::Connection::open(&database_path).expect("open");
            connection
                .execute_batch(crate::session::migrations::baseline_for_test())
                .expect("the baseline builds");
        }

        let one = tokio::spawn({
            let database_path = database_path.clone();
            async move { SessionManager::open(Some(&database_path), &Default::default()).await }
        });
        let other = tokio::spawn({
            let database_path = database_path.clone();
            async move { SessionManager::open(Some(&database_path), &Default::default()).await }
        });
        let first = one
            .await
            .expect("the task finishes")
            .expect("one host opens");
        let second = other
            .await
            .expect("the task finishes")
            .expect("the other host opens too");

        let version: i64 = first
            .connection
            .call(|connection| {
                connection.query_row("SELECT * FROM pragma_user_version", [], |row| row.get(0))
            })
            .await
            .expect("a version");
        assert!(version > 0, "the store was migrated");
        // One migration, not two: a second pass would have tried to add columns that now exist.
        let claim_columns: i64 = second
            .connection
            .call(|connection| {
                connection.query_row(
                    "SELECT count(*) FROM pragma_table_info('scheduled_jobs') WHERE name = 'claimed_by'",
                    [],
                    |row| row.get(0),
                )
            })
            .await
            .expect("column count");
        assert_eq!(claim_columns, 1);
    }
}
