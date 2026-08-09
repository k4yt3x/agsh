//! SQLite-backed session store. Persists messages, large tool outputs (so they can be referenced
//! from the conversation by handle), OAuth tokens, and MCP credentials. Per-session mutual
//! exclusion is provided by an OS-level file lock ([`SessionLock`]) so the kernel reclaims it
//! whenever the holder dies: no PID-aliveness check, no risk of stale locks.
//!
//! On Unix the data directory (`0700`), lock directory (`0700`), and the database file itself
//! (`0600`) are tightened after creation so the persisted OAuth tokens, MCP credentials, and
//! conversation content aren't readable by other local users regardless of the user's umask.

use std::{
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
    sync::Arc,
};

use fd_lock::{RwLock as FdRwLock, RwLockWriteGuard as FdRwLockWriteGuard};
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
    /// Workspace roots beyond `cwd`. Empty for every non-ACP session and for rows written before
    /// the column existed, both of which were single-root.
    pub additional_roots: Vec<PathBuf>,
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
    /// Working directory captured at `create_session` time. `None` for legacy rows from before the
    /// `cwd` column was added; ACP-facing code falls back to the process cwd for display.
    pub cwd: Option<std::path::PathBuf>,
    /// Permission level captured at session creation, or `None` for legacy rows / sessions created
    /// without a per-session permission (the REPL and ACP paths derive permission from process
    /// config, not per-session). The HTTP API persists this so `POST /v1/sessions` with an
    /// explicit `permission` field survives GC-eviction + re-attach.
    pub permission: Option<String>,
    /// JSON-encoded per-session capability flags (currently just
    /// `{"supports_reasoning_stream": bool}`). Same NULL-for-legacy semantics as `permission`.
    pub capabilities_json: Option<String>,
    /// Workspace roots beyond `cwd`, from an ACP client's `additionalDirectories`. Empty for every
    /// non-ACP session and for legacy rows written before the column existed, both of which were
    /// single-root.
    pub additional_roots: Vec<PathBuf>,
    /// SHA-256 fingerprint of the bearer token that created this session. `None` for legacy
    /// rows and for sessions not created via the HTTP API.
    pub token_id: Option<String>,
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
        // Any `.lock` files still inside belong to this manager's own sessions; a `SessionLock`
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

/// RAII handle for an exclusive per-session OS file lock. Holding this value keeps the underlying
/// lock file descriptor open; dropping it (including when the process exits or panics) closes the
/// FD, which causes the kernel to release the `flock`/`LockFileEx` lock automatically. There is no
/// "stale lock" failure mode; even `SIGKILL` is safe.
///
/// Internally this is a self-referential struct: `guard` borrows from `*lock` (a `Box` for stable
/// heap address). The explicit [`Drop`] impl drops `guard` before `lock` regardless of field
/// declaration order, the safety invariant of the lifetime transmute used during construction.
pub struct SessionLock {
    guard: std::mem::ManuallyDrop<FdRwLockWriteGuard<'static, File>>,
    lock: std::mem::ManuallyDrop<Box<FdRwLock<File>>>,
}

impl Drop for SessionLock {
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

fn default_database_path() -> Result<PathBuf> {
    // `MEKA_DATA_DIR` is the cross-platform override, the only env var that works on every OS,
    // mirroring how `MEKA_CONFIG_DIR` overrides the config directory. The value points at the
    // `meka` data dir itself (the parent that contains `meka.db`). Useful for tests, portable
    // installs, and isolating per-project state from the global one.
    if let Ok(value) = std::env::var("MEKA_DATA_DIR")
        && !value.is_empty()
    {
        return Ok(PathBuf::from(value).join("meka.db"));
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

impl SessionManager {
    pub async fn open(path: Option<&Path>) -> Result<Self> {
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
        // in `CREATE TABLE` are decorative without this. Set before `initialize_schema` so the
        // migration's DELETE/ALTER statements run with enforcement active. Must run outside any
        // transaction to take effect.
        connection
            .call(|connection| -> rusqlite::Result<_> {
                // WAL lets the REPL's history connection read without blocking the agent's writes;
                // `busy_timeout` makes both connections wait briefly instead of erroring under
                // contention. (On `:memory:` the journal_mode request is silently ignored.)
                connection.execute_batch(
                    "PRAGMA journal_mode = WAL;\n\
                     PRAGMA busy_timeout = 5000;\n\
                     PRAGMA foreign_keys = ON;",
                )?;
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
        manager.initialize_schema().await?;
        manager.prune_orphan_lock_files().await;
        Ok(manager)
    }

    /// Resolved path to the on-disk database (or `:memory:`). The REPL opens a second synchronous
    /// connection here for persistent input history (see [`crate::history::PromptHistory`]).
    pub fn database_path(&self) -> &Path {
        &self.database_path
    }

    async fn initialize_schema(&self) -> Result<()> {
        self.connection
            .call(|connection| -> rusqlite::Result<_> {
                connection.execute_batch(
                    "CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        parent_session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
                        cwd TEXT,
                        permission TEXT,
                        capabilities_json TEXT,
                        token_id TEXT,
                        additional_roots_json TEXT
                    );

                    CREATE INDEX IF NOT EXISTS idx_sessions_updated_at
                        ON sessions(updated_at);

                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_messages_session_id
                        ON messages(session_id);

                    -- Provider credentials (API keys and OAuth bundles) are keyed by the
                    -- user-chosen profile name and stored as a serialized AuthCredential. The
                    -- pre-0.27 `oauth_tokens` table is dropped; users re-authenticate via
                    -- `meka provider add`.
                    DROP TABLE IF EXISTS oauth_tokens;

                    CREATE TABLE IF NOT EXISTS provider_credentials (
                        profile TEXT PRIMARY KEY,
                        credentials_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS mcp_oauth_credentials (
                        server_name TEXT PRIMARY KEY,
                        credentials_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    ",
                )?;

                // Migration: recreate tool_outputs if it has the old integer-ID schema.
                let has_old_schema: bool = connection
                    .query_row(
                        "SELECT COUNT(*) FROM pragma_table_info('tool_outputs') WHERE name = 'id'",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap_or(0)
                    > 0;
                if has_old_schema {
                    connection.execute_batch("DROP TABLE tool_outputs")?;
                }

                // Migration: drop the legacy `sessions.locked_by` column. Locks are now OS file
                // locks managed via `SessionLock`, so any value left in this column is meaningless
                // and a stale PID can permanently lock a session if the column survives.
                let has_locked_by: bool = connection
                    .query_row(
                        "SELECT COUNT(*) FROM pragma_table_info('sessions') WHERE name = 'locked_by'",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap_or(0)
                    > 0;
                if has_locked_by {
                    connection.execute_batch("ALTER TABLE sessions DROP COLUMN locked_by")?;
                }

                connection.execute_batch(
                    "CREATE TABLE IF NOT EXISTS tool_outputs (
                        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                        name TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (session_id, name)
                    );

                    CREATE TABLE IF NOT EXISTS mcp_auth_cache (
                        server_name TEXT PRIMARY KEY,
                        needs_auth INTEGER NOT NULL,
                        cached_at INTEGER NOT NULL
                    );

                    DROP TABLE IF EXISTS model_context_cache;

                    CREATE TABLE IF NOT EXISTS model_metadata_cache (
                        provider_profile TEXT NOT NULL,
                        model TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        source TEXT NOT NULL,
                        fetched_at INTEGER NOT NULL,
                        PRIMARY KEY (provider_profile, model)
                    );",
                )?;

                // Migration: add `sessions.parent_session_id` so sub-agent sessions can be linked
                // back to the parent that spawned them. Primary sessions store NULL. The cascade-FK
                // is attached later in the rebuild migration below; this step only guarantees the
                // column exists. Index is created unconditionally afterwards so fresh DBs (column
                // from CREATE TABLE) and migrated DBs (column from ALTER TABLE) both end up with
                // it.
                let has_parent_col: bool = connection
                    .query_row(
                        "SELECT COUNT(*) FROM pragma_table_info('sessions') WHERE name = 'parent_session_id'",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap_or(0)
                    > 0;
                if !has_parent_col {
                    connection
                        .execute_batch("ALTER TABLE sessions ADD COLUMN parent_session_id TEXT")?;
                }
                connection.execute_batch(
                    "CREATE INDEX IF NOT EXISTS idx_sessions_parent
                         ON sessions(parent_session_id)",
                )?;

                // Migration: add `sessions.cwd` so ACP's `session/list` can report each session's
                // working directory and `session/load` has a stored cwd to validate the client's
                // request against. Existing rows get NULL; the ACP handlers fall back to the
                // process cwd for those entries so legacy sessions stay listable.
                let has_cwd_col: bool = connection
                    .query_row(
                        "SELECT COUNT(*) FROM pragma_table_info('sessions') WHERE name = 'cwd'",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap_or(0)
                    > 0;
                if !has_cwd_col {
                    connection.execute_batch("ALTER TABLE sessions ADD COLUMN cwd TEXT")?;
                }

                // Migration: drop the legacy `sessions.metadata` column. It was reserved for future
                // use but never populated by any codepath, so it's pure schema noise.
                let has_metadata: bool = connection
                    .query_row(
                        "SELECT COUNT(*) FROM pragma_table_info('sessions') WHERE name = 'metadata'",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap_or(0)
                    > 0;
                if has_metadata {
                    connection.execute_batch("ALTER TABLE sessions DROP COLUMN metadata")?;
                }

                // Migration: rebuild sessions / messages / tool_outputs so their
                // session-referencing FKs carry `ON DELETE CASCADE`, and so `parent_session_id` has
                // a FK at all (it was added as a plain TEXT column). SQLite can't ALTER a column's
                // constraints, so we follow the documented redefinition procedure: disable FK
                // enforcement, recreate each table with the final schema, copy rows, drop old,
                // rename, then re-enable enforcement. Wrapped in a transaction so a crash
                // mid-migration leaves the original tables intact.
                //
                // Detection key: the `messages` table SQL contains the literal `ON DELETE CASCADE`
                // after the migration runs.
                let has_cascade: bool = connection
                    .query_row(
                        "SELECT COUNT(*) FROM sqlite_master \
                         WHERE type = 'table' AND name = 'messages' \
                           AND sql LIKE '%ON DELETE CASCADE%'",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap_or(0)
                    > 0;
                if !has_cascade {
                    // PRAGMA foreign_keys only takes effect outside any transaction. Toggle, run
                    // rebuild, restore (even on failure) so the connection isn't left in a state
                    // where FKs are silently off for subsequent queries.
                    connection.execute_batch("PRAGMA foreign_keys = OFF")?;
                    let migration_result = (|| -> rusqlite::Result<()> {
                        let txn = connection.transaction()?;
                        txn.execute_batch(
                            "CREATE TABLE sessions_new (
                                 id TEXT PRIMARY KEY,
                                 created_at TEXT NOT NULL,
                                 updated_at TEXT NOT NULL,
                                 parent_session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
                                 cwd TEXT
                             );
                             INSERT INTO sessions_new(id, created_at, updated_at, parent_session_id, cwd)
                                 SELECT id, created_at, updated_at, parent_session_id, cwd FROM sessions;
                             DROP TABLE sessions;
                             ALTER TABLE sessions_new RENAME TO sessions;
                             CREATE INDEX idx_sessions_updated_at ON sessions(updated_at);
                             CREATE INDEX idx_sessions_parent ON sessions(parent_session_id);

                             CREATE TABLE messages_new (
                                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                                 session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                                 role TEXT NOT NULL,
                                 content TEXT NOT NULL,
                                 created_at TEXT NOT NULL
                             );
                             INSERT INTO messages_new(id, session_id, role, content, created_at)
                                 SELECT id, session_id, role, content, created_at FROM messages;
                             DROP TABLE messages;
                             ALTER TABLE messages_new RENAME TO messages;
                             CREATE INDEX idx_messages_session_id ON messages(session_id);

                             CREATE TABLE tool_outputs_new (
                                 session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                                 name TEXT NOT NULL,
                                 content TEXT NOT NULL,
                                 created_at TEXT NOT NULL,
                                 PRIMARY KEY (session_id, name)
                             );
                             INSERT INTO tool_outputs_new(session_id, name, content, created_at)
                                 SELECT session_id, name, content, created_at FROM tool_outputs;
                             DROP TABLE tool_outputs;
                             ALTER TABLE tool_outputs_new RENAME TO tool_outputs;",
                        )?;
                        txn.commit()
                    })();
                    connection.execute_batch("PRAGMA foreign_keys = ON")?;
                    migration_result?;
                }

                // Migration: add `sessions.permission` and `sessions.capabilities_json`. Both back
                // the HTTP API's per-session permission / capability flags so a GC-evicted session
                // can be re-attached with the same shape the client originally created it with.
                // Legacy rows (REPL / ACP / pre-0.27 HTTP) get NULL, and the re-attach helper falls
                // back to the process default. Runs after the cascade-FK rebuild because the
                // rebuild's `sessions_new` schema doesn't include these columns; the ALTER ADD
                // here applies to both fresh DBs (no-op) and rebuilt DBs alike.
                let has_permission_col: bool = connection
                    .query_row(
                        "SELECT COUNT(*) FROM pragma_table_info('sessions') WHERE name = 'permission'",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap_or(0)
                    > 0;
                if !has_permission_col {
                    connection
                        .execute_batch("ALTER TABLE sessions ADD COLUMN permission TEXT")?;
                }
                let has_capabilities_col: bool = connection
                    .query_row(
                        "SELECT COUNT(*) FROM pragma_table_info('sessions') WHERE name = 'capabilities_json'",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap_or(0)
                    > 0;
                if !has_capabilities_col {
                    connection.execute_batch(
                        "ALTER TABLE sessions ADD COLUMN capabilities_json TEXT",
                    )?;
                }

                // Migration: add `sessions.token_id` to attribute HTTP-created sessions back
                // to their creating bearer token. NULL for legacy rows and for REPL/ACP
                // sessions.
                let has_token_id_col: bool = connection
                    .query_row(
                        "SELECT COUNT(*) FROM pragma_table_info('sessions') WHERE name = 'token_id'",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap_or(0)
                    > 0;
                if !has_token_id_col {
                    connection.execute_batch("ALTER TABLE sessions ADD COLUMN token_id TEXT")?;
                }

                // Migration: add `sessions.additional_roots_json` so an ACP client's
                // `additionalDirectories` survives a restart and `session/list` can report the
                // workspace shape a session was last used with. NULL on legacy rows decodes to an
                // empty list, i.e. single-root, which is what those sessions were.
                //
                // Runs after the cascade-FK rebuild for the same reason as the three migrations
                // above: the rebuild's `sessions_new` schema lists only id/created_at/updated_at/
                // parent_session_id/cwd, so it silently drops any column added before it. Adding
                // this one next to the `cwd` migration looked natural and was wrong: on a pre-0.24
                // database the column was created, then dropped by the rebuild, and every query
                // naming it failed for the rest of that process's life.
                let has_roots_col: bool = connection
                    .query_row(
                        "SELECT COUNT(*) FROM pragma_table_info('sessions') WHERE name = 'additional_roots_json'",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap_or(0)
                    > 0;
                if !has_roots_col {
                    connection.execute_batch(
                        "ALTER TABLE sessions ADD COLUMN additional_roots_json TEXT",
                    )?;
                }

                // Migration: per-session cumulative stat columns (surfaced by `/status`) so the
                // running totals survive resume instead of restarting at zero each process. Added as
                // a set; the presence of `stat_turns` gates the whole batch.
                let has_stat_cols: bool = connection
                    .query_row(
                        "SELECT COUNT(*) FROM pragma_table_info('sessions') WHERE name = 'stat_turns'",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap_or(0)
                    > 0;
                if !has_stat_cols {
                    connection.execute_batch(
                        "ALTER TABLE sessions ADD COLUMN stat_turns INTEGER NOT NULL DEFAULT 0;
                         ALTER TABLE sessions ADD COLUMN stat_input_tokens INTEGER NOT NULL DEFAULT 0;
                         ALTER TABLE sessions ADD COLUMN stat_output_tokens INTEGER NOT NULL DEFAULT 0;
                         ALTER TABLE sessions ADD COLUMN stat_cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0;
                         ALTER TABLE sessions ADD COLUMN stat_cache_read_input_tokens INTEGER NOT NULL DEFAULT 0;
                         ALTER TABLE sessions ADD COLUMN stat_redactions INTEGER NOT NULL DEFAULT 0;
                         ALTER TABLE sessions ADD COLUMN stat_redacted_images INTEGER NOT NULL DEFAULT 0;
                         ALTER TABLE sessions ADD COLUMN stat_redacted_bytes INTEGER NOT NULL DEFAULT 0;",
                    )?;
                }

                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to initialize schema: {}", error)))
    }

    /// Create a new session, optionally recording its working directory. `cwd` is persisted as an
    /// absolute path string; pass `None` only for code paths that genuinely have no cwd context
    /// (legacy/test fixtures, `meka tools list`). Production paths (the REPL and `meka acp`) pass
    /// the agent's current cwd so `session/list` can surface it later.
    pub async fn create_session(&self, cwd: Option<std::path::PathBuf>) -> Result<Uuid> {
        self.create_session_with_metadata(cwd, None, None, None)
            .await
            .map(|created| created.id)
    }

    /// Like [`Self::create_session`] but also persists the HTTP API's per-session metadata
    /// (`permission` level, `capabilities_json` blob, and `token_id` fingerprint). The REPL
    /// and ACP paths derive permission from process config and don't have a bearer token, so
    /// they keep calling the unparameterised `create_session`; only the HTTP server's
    /// `POST /v1/sessions` handler reaches for this overload.
    pub async fn create_session_with_metadata(
        &self,
        cwd: Option<std::path::PathBuf>,
        permission: Option<String>,
        capabilities_json: Option<String>,
        token_id: Option<String>,
    ) -> Result<CreatedSession> {
        let session_id = Uuid::new_v4();
        let created_at = chrono::Utc::now().to_rfc3339();
        let cwd_string = cwd.map(|path| path.display().to_string());

        let created_at_for_db = created_at.clone();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO sessions (id, created_at, updated_at, cwd, permission, capabilities_json, token_id)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    rusqlite::params![
                        session_id.to_string(),
                        created_at_for_db,
                        created_at_for_db,
                        cwd_string,
                        permission,
                        capabilities_json,
                        token_id,
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
    /// `spawn_agent` so sub-agent conversations persist as children of the parent for auditing.
    /// Cascades on parent delete (see [`Self::delete_session`]). The optional `cwd` is the parent's
    /// cwd snapshot at spawn time.
    pub async fn create_child_session(
        &self,
        parent: Uuid,
        cwd: Option<std::path::PathBuf>,
    ) -> Result<Uuid> {
        let session_id = Uuid::new_v4();
        let now = chrono::Utc::now().to_rfc3339();
        let cwd_string = cwd.map(|path| path.display().to_string());

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO sessions (id, created_at, updated_at, parent_session_id, cwd)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    rusqlite::params![
                        session_id.to_string(),
                        now,
                        now,
                        parent.to_string(),
                        cwd_string,
                    ],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to create child session: {}", error))
            })?;

        Ok(session_id)
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
        let new_id = Uuid::new_v4();
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
                         capabilities_json, token_id, additional_roots_json, stat_turns,
                         stat_input_tokens, stat_output_tokens,
                         stat_cache_creation_input_tokens, stat_cache_read_input_tokens,
                         stat_redactions, stat_redacted_images, stat_redacted_bytes
                     )
                     SELECT ?1, ?2, ?2, NULL, COALESCE(?3, cwd), permission,
                            capabilities_json, ?4,
                            CASE WHEN ?5 THEN ?6 ELSE additional_roots_json END, stat_turns,
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

    /// Acquire an exclusive OS file lock on the session. Returns a [`SessionLock`] handle whose
    /// lifetime owns the lock; drop it (or let the process exit) to release.
    ///
    /// The session must already exist in the database. Returns [`MekaError::SessionLocked`] if
    /// another live process holds the lock.
    pub fn lock_session(&self, session_id: Uuid) -> Result<SessionLock> {
        let path = self.lock_dir.join(format!("{}.lock", session_id));
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .map_err(|error| {
                MekaError::Database(format!(
                    "failed to open session lock file '{}': {}",
                    path.display(),
                    error
                ))
            })?;

        let mut lock = Box::new(FdRwLock::new(file));
        let guard = match lock.try_write() {
            Ok(guard) => guard,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                return Err(MekaError::SessionLocked(session_id));
            }
            Err(error) => {
                return Err(MekaError::Database(format!(
                    "failed to acquire session lock '{}': {}",
                    path.display(),
                    error
                )));
            }
        };

        // SAFETY: `guard` borrows from `*lock`. We move the box (not the RwLock inside it) into the
        // returned `SessionLock`, so the RwLock's heap address is stable for as long as the box
        // lives. The explicit `Drop` impl on `SessionLock` drops `guard` before `lock`, so the
        // borrow never outlives the borrowee.
        let guard: FdRwLockWriteGuard<'static, File> = unsafe { std::mem::transmute(guard) };

        Ok(SessionLock {
            guard: std::mem::ManuallyDrop::new(guard),
            lock: std::mem::ManuallyDrop::new(lock),
        })
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
    /// Deleting a *held* lock file for an already-deleted session is benign (no process starts a
    /// deleted session), and the sweep never races a live one: a session's row is committed before
    /// its lock file is acquired (see `main.rs`), so a live UUID is always in the set this query
    /// returns.
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
            if Uuid::parse_str(stem).is_err() || live_ids.contains(stem) {
                continue;
            }
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
    ///
    /// No schema migration: legacy databases (predating this commit) only contain `Event::Append`
    /// rows; loading them via [`Self::load_events`] yields the same events the in-memory log
    /// produced before.
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
    /// GC deletes by `updated_at` ([`Self::delete_expired_sessions`], run at every agent startup),
    /// so restoring the original value meant an archive older than `retention_days` was swept on
    /// the next launch, before anyone could resume it. `created_at` still carries the original for
    /// provenance.
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
                             capabilities_json, additional_roots_json, stat_turns,
                             stat_input_tokens, stat_output_tokens,
                             stat_cache_creation_input_tokens, stat_cache_read_input_tokens,
                             stat_redactions, stat_redacted_images, stat_redacted_bytes
                         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
                        rusqlite::params![
                            session.id,
                            session.created_at,
                            imported_at,
                            session.parent_id,
                            session.cwd,
                            session.permission,
                            session.capabilities_json,
                            session.additional_roots_json,
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

    /// Load every event for a session in chronological order. Legacy rows (role ∈ {`user`,
    /// `assistant`, `tool_results`}) are reconstructed as `Event::Append`; rows with role
    /// `compact_boundary` are deserialized from the JSON envelope. Unknown roles are skipped with a
    /// warning.
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
                            s.cwd, s.permission, s.capabilities_json, s.additional_roots_json
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
    /// the given path (rows with NULL `cwd` are excluded; legacy rows can't be filtered by cwd
    /// they never recorded).
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
                    "SELECT s.id, s.created_at, s.updated_at, s.cwd, s.permission, s.capabilities_json, s.additional_roots_json, s.token_id,
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
                    let preview: String = row.get(8)?;
                    Ok((
                        id_str,
                        created_at,
                        updated_at,
                        cwd,
                        permission,
                        capabilities_json,
                        additional_roots_json,
                        token_id,
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
                        capabilities_json,
                        additional_roots: decode_additional_roots(additional_roots_json.as_deref()),
                        token_id,
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
                    "SELECT s.id, s.created_at, s.updated_at, s.cwd, s.permission, s.capabilities_json, s.additional_roots_json, s.token_id,
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
                    let preview: String = row.get(8)?;
                    Ok((
                        id_str,
                        created_at,
                        updated_at,
                        cwd,
                        permission,
                        capabilities_json,
                        additional_roots_json,
                        token_id,
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
                            additional_roots: decode_additional_roots(
                                additional_roots_json.as_deref(),
                            ),
                            capabilities_json,
                            token_id,
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

    pub async fn delete_expired_sessions(&self, retention_days: u64) -> Result<u64> {
        // `TimeDelta::days` panics on out-of-range input, so route through `try_days`: an absurdly
        // large `retention_days` falls back to a ~100-year window, which deletes nothing: the
        // intended "retain everything" outcome.
        let retention = i64::try_from(retention_days)
            .ok()
            .and_then(chrono::TimeDelta::try_days)
            .unwrap_or_else(|| chrono::TimeDelta::days(36_500));
        let cutoff = chrono::Utc::now() - retention;
        let cutoff_str = cutoff.to_rfc3339();

        let deleted = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                // FK CASCADE sweeps messages, tool_outputs, and any sub-agent child sessions of the
                // expired parents.
                let deleted = connection.execute(
                    "DELETE FROM sessions WHERE updated_at < ?1",
                    rusqlite::params![cutoff_str],
                )?;
                Ok(deleted as u64)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to delete expired sessions: {}", error))
            })?;
        self.prune_orphan_lock_files().await;
        Ok(deleted)
    }

    /// Update the persisted cwd for an existing session. Called by the ACP `session/load` /
    /// `session/resume` handlers when the client's `cwd` differs from the persisted value; the
    /// client wins so future `session/list` results reflect the live state. `cwd` is stored as the
    /// path's `to_string_lossy()` form (UTF-8 is the only column type SQLite has). Returns the
    /// number of rows updated (0 if the session id doesn't exist).
    ///
    /// Apply both `permission` and `cwd` updates in a single SQLite transaction so a DB
    /// failure between the two writes can't leave a half-applied state on disk.  Either
    /// column may be `None` to skip that field.  `updated_at` is recomputed inside the
    /// transaction so the timestamp matches the commit, not the call.  The individual
    /// `update_session_cwd` / `update_session_permission` methods remain for callers
    /// that only need a single-column write.
    pub async fn update_session_metadata_atomic(
        &self,
        session_id: Uuid,
        new_permission: Option<String>,
        new_cwd: Option<std::path::PathBuf>,
    ) -> Result<()> {
        if new_permission.is_none() && new_cwd.is_none() {
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
    /// HTTP handler is the only writer and serialises a `SessionCapabilities` value.
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

    pub async fn delete_session(&self, session_id: Uuid) -> Result<bool> {
        let deleted = self
            .connection
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
            .map_err(|error| MekaError::Database(format!("failed to delete session: {}", error)))?;
        self.prune_orphan_lock_files().await;
        Ok(deleted)
    }

    pub async fn delete_all_sessions(&self) -> Result<u64> {
        let deleted = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                // FK CASCADE clears messages and tool_outputs.
                let deleted = connection.execute("DELETE FROM sessions", [])?;
                Ok(deleted as u64)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to delete all sessions: {}", error))
            })?;
        self.prune_orphan_lock_files().await;
        Ok(deleted)
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

    /// Delete the oldest sessions until total `messages.content` size is at or below `max_bytes`.
    /// `active_ids` is the set of session ids the caller knows to be currently in use; they are
    /// excluded from the eviction sweep so the caller's in-flight `save_event` calls don't trip on
    /// foreign-key violations after a deletion. Pass an empty set when there are no live sessions
    /// (typical at startup, when this is called before any session is opened).
    pub async fn enforce_storage_limit(
        &self,
        max_bytes: u64,
        active_ids: &std::collections::HashSet<String>,
    ) -> Result<u64> {
        // Take the caller's snapshot once and move it into the blocking task so the SQL closure can
        // match against it without re-locking anything.
        let active: Vec<String> = active_ids.iter().cloned().collect();
        let deleted = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut deleted: u64 = 0;

                loop {
                    let total_bytes: i64 = connection.query_row(
                        "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM messages",
                        [],
                        |row| row.get(0),
                    )?;

                    if u64::try_from(total_bytes).unwrap_or(0) <= max_bytes {
                        break;
                    }

                    // Build the `NOT IN (?, ?, ...)` placeholder list dynamically. SQLite's
                    // parameter index uses 1-based values; we feed each id positionally below.
                    let placeholders = (1..=active.len())
                        .map(|index| format!("?{}", index))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let query = if placeholders.is_empty() {
                        "SELECT id FROM sessions ORDER BY updated_at ASC LIMIT 1".to_string()
                    } else {
                        format!(
                            "SELECT id FROM sessions WHERE id NOT IN ({}) \
                             ORDER BY updated_at ASC LIMIT 1",
                            placeholders
                        )
                    };
                    let params = rusqlite::params_from_iter(active.iter());
                    let oldest_id: std::result::Result<String, _> =
                        connection.query_row(&query, params, |row| row.get(0));

                    match oldest_id {
                        Ok(session_id) => {
                            // FK CASCADE sweeps the session's messages and tool_outputs along with
                            // the session row.
                            connection.execute(
                                "DELETE FROM sessions WHERE id = ?1",
                                rusqlite::params![session_id],
                            )?;
                            deleted += 1;
                        }
                        // No eligible row left: either the DB is empty, or every remaining session
                        // is in the active set. Either way, we can't reclaim more without touching
                        // live state.
                        Err(rusqlite::Error::QueryReturnedNoRows) => break,
                        Err(error) => return Err(error),
                    }
                }

                Ok(deleted)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to enforce storage limit: {}", error))
            })?;
        self.prune_orphan_lock_files().await;
        Ok(deleted)
    }
}

#[derive(Clone)]
pub struct TokenStore {
    connection: Arc<Connection>,
}

impl TokenStore {
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

    /// Persist (or replace) the credential for a provider profile, keyed by profile name.
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

    pub async fn load_mcp_credentials(&self, server_name: &str) -> Result<Option<String>> {
        let server_name = server_name.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let result = connection.query_row(
                    "SELECT credentials_json FROM mcp_oauth_credentials WHERE server_name = ?1",
                    rusqlite::params![server_name],
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
                MekaError::Database(format!("failed to load MCP credentials: {}", error))
            })
    }

    pub async fn save_mcp_credentials(&self, server_name: &str, json: &str) -> Result<()> {
        let server_name = server_name.to_string();
        let json = json.to_string();
        let now = chrono::Utc::now().to_rfc3339();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO mcp_oauth_credentials (server_name, credentials_json, updated_at)
                     VALUES (?1, ?2, ?3)
                     ON CONFLICT(server_name) DO UPDATE SET
                         credentials_json = excluded.credentials_json,
                         updated_at = excluded.updated_at",
                    rusqlite::params![server_name, json, now],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to save MCP credentials: {}", error))
            })
    }

    pub async fn clear_mcp_credentials(&self, server_name: &str) -> Result<()> {
        let server_name = server_name.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "DELETE FROM mcp_oauth_credentials WHERE server_name = ?1",
                    rusqlite::params![server_name],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to clear MCP credentials: {}", error))
            })
    }

    /// Load a cached needs-auth verdict for an MCP server, or `None` if there is no entry or it is
    /// older than `ttl`.
    pub async fn load_auth_probe(
        &self,
        server_name: &str,
        ttl: std::time::Duration,
    ) -> Result<Option<bool>> {
        let server_name = server_name.to_string();
        let ttl_seconds = ttl.as_secs() as i64;
        let now = chrono::Utc::now().timestamp();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let result = connection.query_row(
                    "SELECT needs_auth, cached_at FROM mcp_auth_cache WHERE server_name = ?1",
                    rusqlite::params![server_name],
                    |row| {
                        let needs_auth: i64 = row.get(0)?;
                        let cached_at: i64 = row.get(1)?;
                        Ok((needs_auth != 0, cached_at))
                    },
                );
                match result {
                    // Strict `<` so a zero-duration TTL behaves as "never cache" instead of "cache
                    // for the rest of this second".
                    Ok((needs_auth, cached_at)) if now - cached_at < ttl_seconds => {
                        Ok(Some(needs_auth))
                    }
                    Ok(_) => Ok(None),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(error) => Err(error),
                }
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to load MCP auth probe cache: {}", error))
            })
    }

    /// Persist a needs-auth verdict for an MCP server (TTL is enforced at load time, so we just
    /// record the current timestamp here).
    pub async fn save_auth_probe(&self, server_name: &str, needs_auth: bool) -> Result<()> {
        let server_name = server_name.to_string();
        let now = chrono::Utc::now().timestamp();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO mcp_auth_cache (server_name, needs_auth, cached_at)
                     VALUES (?1, ?2, ?3)
                     ON CONFLICT(server_name) DO UPDATE SET
                         needs_auth = excluded.needs_auth,
                         cached_at = excluded.cached_at",
                    rusqlite::params![server_name, if needs_auth { 1_i64 } else { 0_i64 }, now],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to save MCP auth probe cache: {}", error))
            })
    }

    /// Remove any cached needs-auth verdict for an MCP server.
    pub async fn clear_auth_probe(&self, server_name: &str) -> Result<()> {
        let server_name = server_name.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "DELETE FROM mcp_auth_cache WHERE server_name = ?1",
                    rusqlite::params![server_name],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to clear MCP auth probe cache: {}", error))
            })
    }

    /// Load the cached [`crate::provider::ModelInfo`] for `(profile, model)`. Returns
    /// `(info, source, age_seconds)` where `source` is `"api"` or `"resolver"`; the caller applies
    /// the per-source TTL. `None` when no row exists or the stored JSON can't be parsed (treated as
    /// a miss so the resolver re-probes and overwrites it).
    pub async fn load_model_metadata(
        &self,
        profile: &str,
        model: &str,
    ) -> Result<Option<(crate::provider::ModelInfo, String, i64)>> {
        let profile = profile.to_string();
        let model = model.to_string();
        let now = chrono::Utc::now().timestamp();
        let row = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                let result = connection.query_row(
                    "SELECT metadata_json, source, fetched_at FROM model_metadata_cache
                     WHERE provider_profile = ?1 AND model = ?2",
                    rusqlite::params![profile, model],
                    |row| {
                        let metadata_json: String = row.get(0)?;
                        let source: String = row.get(1)?;
                        let fetched_at: i64 = row.get(2)?;
                        Ok((metadata_json, source, fetched_at))
                    },
                );
                match result {
                    Ok(row) => Ok(Some(row)),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(error) => Err(error),
                }
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to load model metadata cache: {}", error))
            })?;
        let Some((metadata_json, source, fetched_at)) = row else {
            return Ok(None);
        };
        match serde_json::from_str::<crate::provider::ModelInfo>(&metadata_json) {
            Ok(info) => Ok(Some((info, source, now - fetched_at))),
            Err(error) => {
                tracing::debug!("ignoring unparseable model-metadata cache row: {}", error);
                Ok(None)
            }
        }
    }

    /// Upsert the cached [`crate::provider::ModelInfo`] for `(profile, model)`, stamping
    /// `fetched_at = now`. The whole struct is stored as JSON so one probe caches every attribute
    /// (context window and effort levels) in a single row.
    ///
    /// `preserve_fresh_api_secs` guards against a negative marker clobbering a healthy positive row
    /// written by a concurrent process: when `Some(ttl)`, the write is skipped if the existing row
    /// is an `api` row younger than `ttl` seconds. `None` (positive `api` writes) always wins.
    pub async fn save_model_metadata(
        &self,
        profile: &str,
        model: &str,
        info: &crate::provider::ModelInfo,
        source: &str,
        preserve_fresh_api_secs: Option<i64>,
    ) -> Result<()> {
        let metadata_json = serde_json::to_string(info).map_err(|error| {
            MekaError::Database(format!("failed to serialize model metadata: {}", error))
        })?;
        let profile = profile.to_string();
        let model = model.to_string();
        let source = source.to_string();
        let now = chrono::Utc::now().timestamp();
        // A row older than this cutoff is not a fresh `api` row, so the guard won't block. `None`
        // uses `i64::MAX`, which no `fetched_at` can exceed, so the update always applies.
        let fresh_api_cutoff = preserve_fresh_api_secs.map_or(i64::MAX, |ttl| now - ttl);
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO model_metadata_cache
                         (provider_profile, model, metadata_json, source, fetched_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)
                     ON CONFLICT(provider_profile, model) DO UPDATE SET
                         metadata_json = excluded.metadata_json,
                         source = excluded.source,
                         fetched_at = excluded.fetched_at
                     WHERE NOT (model_metadata_cache.source = 'api'
                         AND model_metadata_cache.fetched_at > ?6)",
                    rusqlite::params![profile, model, metadata_json, source, now, fresh_api_cutoff],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to save model metadata cache: {}", error))
            })
    }
}

impl SessionManager {
    pub fn token_store(&self) -> TokenStore {
        TokenStore {
            connection: Arc::clone(&self.connection),
        }
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

/// Pseudo-role used in the `messages` table for `Event::CompactBoundary` rows. Coexists with the
/// legacy `user`/`assistant`/`tool_results` roles without a schema migration.
const COMPACT_BOUNDARY_ROLE: &str = "compact_boundary";

/// Pseudo-role for a `Role::User` message that carries non-text blocks (input images). Its full
/// `Vec<ContentBlock>` is stored as JSON, because flattening to `text_content()` (as the plain
/// `user` role does) would silently drop the images. Text-only user turns stay plaintext under
/// `user` so `list_sessions`'s raw-content preview subquery keeps working. Added without a schema
/// migration, mirroring [`COMPACT_BOUNDARY_ROLE`].
const USER_BLOCKS_ROLE: &str = "user_blocks";

/// Encode an [`crate::conversation::Event`] into the `(role, content)` columns of the `messages`
/// table. `Event::Append` writes the message's natural role; `Event::CompactBoundary` writes a JSON
/// envelope under the [`COMPACT_BOUNDARY_ROLE`] pseudo-role.
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
                // Legacy or malformed JSON: fall back to text.
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
        role if role == COMPACT_BOUNDARY_ROLE => {
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
/// keeps "no extra roots" as NULL, matching legacy rows, so the column has one representation for
/// one meaning rather than NULL and `[]` both appearing.
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
/// NULL (legacy rows, and every session that never carried extra roots) and unparseable JSON both
/// yield an empty list. Failing soft is right here: a session whose root list can't be read is
/// still perfectly usable as a single-root session, and refusing to load it would be a far worse
/// outcome than silently narrowing its search scope.
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
        SessionManager::open(Some(Path::new(":memory:")))
            .await
            .expect("failed to open in-memory database")
    }

    /// Populate a session with the full spread of state a fork has to carry.
    async fn seeded_session(manager: &SessionManager) -> Uuid {
        use crate::{conversation::Event, provider::Message};

        let id = manager
            .create_session(Some(PathBuf::from("/work/main")))
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
        assert_eq!(deleted, 1, "only the stale source is swept");
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
        manager
            .create_child_session(source, None)
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
            "stat_turns",
            "stat_input_tokens",
            "stat_output_tokens",
            "stat_cache_creation_input_tokens",
            "stat_cache_read_input_tokens",
            "stat_redactions",
            "stat_redacted_images",
            "stat_redacted_bytes",
        ]);
    }

    #[tokio::test]
    async fn additional_roots_round_trip_through_the_database() {
        let manager = test_manager().await;
        let id = manager
            .create_session(Some(PathBuf::from("/work/main")))
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
            .create_session(Some(PathBuf::from("/work/main")))
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

    /// Legacy rows predate the column, and unparseable JSON shouldn't make a session unloadable:
    /// both mean "single root", which is what those sessions always were.
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
        let manager = SessionManager::open(Some(&db_path))
            .await
            .expect("open on-disk database");
        let lock_dir = manager.lock_dir.clone();
        drop(manager);
        assert!(
            lock_dir.exists(),
            "an on-disk lock dir must not be swept away with the manager"
        );
    }

    #[tokio::test]
    async fn test_model_metadata_cache_roundtrip() {
        use crate::provider::ModelInfo;
        let store = test_manager().await.token_store();
        // Miss before any write.
        assert!(
            store
                .load_model_metadata("chatgpt", "gpt-5.6-sol")
                .await
                .unwrap()
                .is_none()
        );
        // Save + read back the whole ModelInfo (context window AND effort levels) and source.
        let info = ModelInfo {
            context_window: Some(1_050_000),
            effort_levels: Some(vec![
                "low".to_string(),
                "high".to_string(),
                "xhigh".to_string(),
            ]),
        };
        store
            .save_model_metadata("chatgpt", "gpt-5.6-sol", &info, "api", None)
            .await
            .unwrap();
        let (cached, source, age) = store
            .load_model_metadata("chatgpt", "gpt-5.6-sol")
            .await
            .unwrap()
            .expect("row present");
        assert_eq!(cached, info);
        assert_eq!(source, "api");
        assert!(age >= 0);
        // Composite key isolation: a different model (same profile) is a miss.
        assert!(
            store
                .load_model_metadata("chatgpt", "other-model")
                .await
                .unwrap()
                .is_none()
        );
        // Upsert overwrites the metadata + source in place (here a negative all-None marker).
        let negative = ModelInfo {
            context_window: None,
            effort_levels: None,
        };
        store
            .save_model_metadata("chatgpt", "gpt-5.6-sol", &negative, "resolver", None)
            .await
            .unwrap();
        let (cached, source, _) = store
            .load_model_metadata("chatgpt", "gpt-5.6-sol")
            .await
            .unwrap()
            .expect("row present");
        assert_eq!(cached, negative);
        assert_eq!(source, "resolver");
    }

    #[tokio::test]
    async fn test_negative_marker_preserves_fresh_api_row() {
        use crate::provider::ModelInfo;
        let store = test_manager().await.token_store();
        let positive = ModelInfo {
            context_window: Some(321_000),
            effort_levels: None,
        };
        store
            .save_model_metadata("p", "m", &positive, "api", None)
            .await
            .unwrap();
        // A negative marker must NOT clobber a still-fresh `api` row (guard: preserve within 1h).
        let marker = ModelInfo {
            context_window: None,
            effort_levels: None,
        };
        store
            .save_model_metadata("p", "m", &marker, "resolver", Some(3600))
            .await
            .unwrap();
        let (cached, source, _) = store
            .load_model_metadata("p", "m")
            .await
            .unwrap()
            .expect("row present");
        assert_eq!(source, "api", "fresh api row preserved");
        assert_eq!(cached, positive);
        // A positive `api` write always wins (`None`), refreshing even a fresh row.
        let refreshed = ModelInfo {
            context_window: Some(400_000),
            effort_levels: None,
        };
        store
            .save_model_metadata("p", "m", &refreshed, "api", None)
            .await
            .unwrap();
        let (cached, source, _) = store
            .load_model_metadata("p", "m")
            .await
            .unwrap()
            .expect("row present");
        assert_eq!(source, "api");
        assert_eq!(cached, refreshed);
        // The guard is api-specific: a negative marker still overwrites an existing non-api row.
        store
            .save_model_metadata("p", "n", &marker, "resolver", None)
            .await
            .unwrap();
        store
            .save_model_metadata("p", "n", &marker, "resolver", Some(3600))
            .await
            .unwrap();
        let (_, source, _) = store
            .load_model_metadata("p", "n")
            .await
            .unwrap()
            .expect("row present");
        assert_eq!(source, "resolver");
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
        let sid = manager.create_session(None).await.expect("create session");

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

        for event in [
            &user_event,
            &assistant_event,
            &tool_result_event,
            &boundary_event,
        ] {
            manager.save_event(sid, event).await.expect("save event");
        }

        let loaded = manager.load_events(sid).await.expect("load events");
        assert_eq!(loaded.len(), 4);

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
        let sid = manager.create_session(None).await.expect("create session");

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

    /// Legacy databases (predating PR 2) only contain rows with the `user` / `assistant` /
    /// `tool_results` roles, no `compact_boundary`. `load_events` must hydrate every legacy row as
    /// an `Event::Append` so resume works without a schema migration.
    #[tokio::test]
    async fn test_load_events_legacy_rows_as_append() {
        use crate::conversation::Event;

        let manager = test_manager().await;
        let sid = manager.create_session(None).await.expect("create session");

        // Simulate a pre-PR-2 session by writing rows the legacy way.
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
        let sid = manager.create_session(None).await.expect("create session");
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

        let _manager = SessionManager::open(Some(&db_path))
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
            .create_session(None)
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
        let session_id = manager.create_session(None).await.expect("create session");

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
            .create_session(None)
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
            .create_session(None)
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
            .create_session(None)
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
            .create_session(None)
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
            .create_session(None)
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
            .create_session(None)
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
            .create_session(None)
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
        let live = manager.create_session(None).await.expect("create");

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

    #[tokio::test]
    async fn test_delete_session_removes_lock_file() {
        let manager = test_manager().await;
        let session = manager.create_session(None).await.expect("create");
        let lock_path = manager.lock_dir.join(format!("{}.lock", session));
        std::fs::write(&lock_path, "").expect("write lock");

        manager.delete_session(session).await.expect("delete");
        assert!(
            !lock_path.exists(),
            "deleting a session must remove its lock file"
        );
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
        let _manager = SessionManager::open(Some(&db_path))
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
        let session1 = manager.create_session(None).await.expect("failed");
        let session2 = manager.create_session(None).await.expect("failed");

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
        let session_id = manager.create_session(None).await.expect("failed");
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
        assert_eq!(deleted, 1);
        assert!(!manager.session_exists(session_id).await.expect("failed"));

        let messages = manager.load_messages(session_id).await.expect("failed");
        assert!(messages.is_empty());
    }

    #[tokio::test]
    async fn test_delete_expired_sessions_keeps_recent() {
        let manager = test_manager().await;
        let old_session = manager.create_session(None).await.expect("failed");
        let new_session = manager.create_session(None).await.expect("failed");

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
        assert_eq!(deleted, 1);
        assert!(!manager.session_exists(old_session).await.expect("failed"));
        assert!(manager.session_exists(new_session).await.expect("failed"));
    }

    #[tokio::test]
    async fn test_enforce_storage_limit() {
        let manager = test_manager().await;
        let session1 = manager.create_session(None).await.expect("failed");

        // Add enough content to exceed a small limit
        let large_content = "x".repeat(1000);
        manager
            .save_message(session1, "user", &large_content)
            .await
            .expect("failed");

        // Backdate session1 so it's the oldest
        let old_date = (chrono::Utc::now() - chrono::TimeDelta::days(10)).to_rfc3339();
        manager
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![old_date, session1.to_string()],
                )?;
                Ok(())
            })
            .await
            .expect("failed to backdate");

        let session2 = manager.create_session(None).await.expect("failed");
        manager
            .save_message(session2, "user", "small")
            .await
            .expect("failed");

        // Set a limit smaller than the total, but larger than session2 alone
        let no_active: std::collections::HashSet<String> = std::collections::HashSet::new();
        let deleted = manager
            .enforce_storage_limit(500, &no_active)
            .await
            .expect("failed to enforce");
        assert_eq!(deleted, 1);
        assert!(!manager.session_exists(session1).await.expect("failed"));
        assert!(manager.session_exists(session2).await.expect("failed"));
    }

    #[tokio::test]
    async fn test_clear_messages() {
        let manager = test_manager().await;
        let session_id = manager.create_session(None).await.expect("failed");

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
    fn test_truncate_preview_old_format_backward_compat() {
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
        );
        format!("{}\n\n{}", block, user_input)
    }

    #[tokio::test]
    async fn test_list_sessions_preview_is_user_prompt_not_context_wrapper() {
        // The canonical regression: user types a prompt, turn runs, `meka session list` must show
        // the prompt, not `<context>`, not the permission/environment metadata.
        let manager = test_manager().await;
        let session_id = manager.create_session(None).await.expect("create_session");

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
        // The context block's shape differs per permission level (Write omits the [Environment
        // context], None differs again). Every level should still surface the user's prompt
        // cleanly.
        let manager = test_manager().await;
        for (label, permission) in &[
            ("none", crate::permission::Permission::None),
            ("read", crate::permission::Permission::Read),
            ("ask", crate::permission::Permission::Ask),
            ("write", crate::permission::Permission::Write),
        ] {
            let session_id = manager.create_session(None).await.expect("create_session");
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
        let session_id = manager.create_session(None).await.expect("create_session");

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
        let session_id = manager.create_session(None).await.expect("create_session");

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
        let session_id = manager.create_session(None).await.expect("create_session");

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
        let a = manager.create_session(None).await.expect("create_session");
        let b = manager.create_session(None).await.expect("create_session");
        let c = manager.create_session(None).await.expect("create_session");

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
        let session_id = manager.create_session(None).await.expect("create_session");

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
        let session_id = manager.create_session(None).await.expect("create_session");

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
    async fn test_list_sessions_preview_legacy_unwrapped_user_message() {
        // Backward-compat: older sessions (or future non-wrapper message paths) whose user content
        // has no `<context>` block at all; the stored string IS the prompt, and the preview equals
        // it.
        let manager = test_manager().await;
        let session_id = manager.create_session(None).await.expect("create_session");
        manager
            .save_message(session_id, "user", "legacy prompt without any wrapper")
            .await
            .expect("save_message");

        let (summaries, _next_cursor) = manager
            .list_sessions(10, false, None, None)
            .await
            .expect("list_sessions");
        let summary = summaries.iter().find(|s| s.id == session_id).unwrap();
        assert_eq!(summary.preview, "legacy prompt without any wrapper");
    }

    #[tokio::test]
    async fn test_enforce_storage_limit_no_deletion_needed() {
        let manager = test_manager().await;
        let session_id = manager.create_session(None).await.expect("failed");
        manager
            .save_message(session_id, "user", "small")
            .await
            .expect("failed");

        let no_active: std::collections::HashSet<String> = std::collections::HashSet::new();
        let deleted = manager
            .enforce_storage_limit(1_000_000, &no_active)
            .await
            .expect("failed to enforce");
        assert_eq!(deleted, 0);
        assert!(manager.session_exists(session_id).await.expect("failed"));
    }

    /// Active sessions must survive eviction even when they're the oldest by `updated_at`. The
    /// eviction loop should walk through younger eligible rows and only stop when the budget is met
    /// or no inactive sessions remain.
    #[tokio::test]
    async fn test_enforce_storage_limit_skips_active_sessions() {
        let manager = test_manager().await;
        let oldest = manager.create_session(None).await.expect("create oldest");
        let large = "x".repeat(1000);
        manager
            .save_message(oldest, "user", &large)
            .await
            .expect("save oldest");

        // Bump a younger session to also push the budget over.
        let younger = manager.create_session(None).await.expect("create younger");
        manager
            .save_message(younger, "user", &large)
            .await
            .expect("save younger");

        // Mark the oldest as active; it must be skipped even though it would otherwise be the
        // natural eviction target.
        let mut active = std::collections::HashSet::new();
        active.insert(oldest.to_string());

        let deleted = manager
            .enforce_storage_limit(500, &active)
            .await
            .expect("enforce");
        assert_eq!(deleted, 1);
        assert!(
            manager
                .session_exists(oldest)
                .await
                .expect("session_exists"),
            "active session must not be evicted",
        );
        assert!(
            !manager
                .session_exists(younger)
                .await
                .expect("session_exists"),
            "inactive younger session should have been evicted",
        );
    }

    // Regression: upgrading a pre-0.24 on-disk DB (no parent_session_id column, FKs without ON
    // DELETE CASCADE, `metadata` still present) must succeed end-to-end with data preserved and the
    // post-migration schema in place. Previously the initial `CREATE INDEX ... ON
    // sessions(parent_session_id)` ran before the ADD COLUMN step and bombed with "no such column:
    // parent_session_id".
    #[tokio::test]
    async fn test_migration_from_pre_0_24_schema() {
        use rusqlite::Connection as RusqliteConnection;

        let temp_dir = tempfile::tempdir().expect("tempdir");
        let db_path = temp_dir.path().join("meka.db");
        let session_id = Uuid::new_v4();
        let now = chrono::Utc::now().to_rfc3339();

        // Stage a pre-0.24 schema with a session + message + tool_output.
        {
            let conn = RusqliteConnection::open(&db_path).expect("rusqlite open");
            conn.execute_batch(
                "CREATE TABLE sessions (
                     id TEXT PRIMARY KEY,
                     created_at TEXT NOT NULL,
                     updated_at TEXT NOT NULL,
                     metadata TEXT
                 );
                 CREATE TABLE messages (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     session_id TEXT NOT NULL,
                     role TEXT NOT NULL,
                     content TEXT NOT NULL,
                     created_at TEXT NOT NULL,
                     FOREIGN KEY (session_id) REFERENCES sessions(id)
                 );
                 CREATE INDEX idx_messages_session_id ON messages(session_id);
                 CREATE TABLE tool_outputs (
                     session_id TEXT NOT NULL,
                     name TEXT NOT NULL,
                     content TEXT NOT NULL,
                     created_at TEXT NOT NULL,
                     PRIMARY KEY (session_id, name),
                     FOREIGN KEY (session_id) REFERENCES sessions(id)
                 );",
            )
            .expect("stage pre-0.24 schema");
            conn.execute(
                "INSERT INTO sessions(id, created_at, updated_at, metadata) \
                 VALUES (?1, ?2, ?2, NULL)",
                rusqlite::params![session_id.to_string(), &now],
            )
            .expect("insert session");
            conn.execute(
                "INSERT INTO messages(session_id, role, content, created_at) \
                 VALUES (?1, 'user', 'preserved', ?2)",
                rusqlite::params![session_id.to_string(), &now],
            )
            .expect("insert message");
            conn.execute(
                "INSERT INTO tool_outputs(session_id, name, content, created_at) \
                 VALUES (?1, 'scratch', 'body', ?2)",
                rusqlite::params![session_id.to_string(), &now],
            )
            .expect("insert tool_output");
        }

        let manager = SessionManager::open(Some(&db_path))
            .await
            .expect("migration should succeed for pre-0.24 schema");

        // Data preserved.
        assert!(
            manager.session_exists(session_id).await.expect("exists"),
            "session row should survive migration"
        );
        let preserved = manager
            .load_tool_output(session_id, "scratch")
            .await
            .expect("load tool_output");
        assert_eq!(preserved.as_deref(), Some("body"));

        // Schema reshaped to 0.24+ form.
        let (has_metadata, has_parent, messages_cascade, tool_outputs_cascade): (
            i64,
            i64,
            i64,
            i64,
        ) = manager
            .connection
            .call(move |conn| -> rusqlite::Result<_> {
                let has_metadata = conn.query_row(
                    "SELECT COUNT(*) FROM pragma_table_info('sessions') WHERE name='metadata'",
                    [],
                    |row| row.get::<_, i64>(0),
                )?;
                let has_parent = conn.query_row(
                    "SELECT COUNT(*) FROM pragma_table_info('sessions') \
                     WHERE name='parent_session_id'",
                    [],
                    |row| row.get::<_, i64>(0),
                )?;
                let messages_cascade = conn.query_row(
                    "SELECT COUNT(*) FROM sqlite_master \
                     WHERE type='table' AND name='messages' \
                       AND sql LIKE '%ON DELETE CASCADE%'",
                    [],
                    |row| row.get::<_, i64>(0),
                )?;
                let tool_outputs_cascade = conn.query_row(
                    "SELECT COUNT(*) FROM sqlite_master \
                     WHERE type='table' AND name='tool_outputs' \
                       AND sql LIKE '%ON DELETE CASCADE%'",
                    [],
                    |row| row.get::<_, i64>(0),
                )?;
                Ok((
                    has_metadata,
                    has_parent,
                    messages_cascade,
                    tool_outputs_cascade,
                ))
            })
            .await
            .expect("schema introspection");

        assert_eq!(has_metadata, 0, "metadata column should be dropped");
        assert_eq!(has_parent, 1, "parent_session_id column should be added");
        assert_eq!(
            messages_cascade, 1,
            "messages FK should carry ON DELETE CASCADE"
        );
        assert_eq!(
            tool_outputs_cascade, 1,
            "tool_outputs FK should carry ON DELETE CASCADE"
        );
    }

    #[tokio::test]
    async fn test_session_metadata_round_trips() {
        let manager = test_manager().await;

        // Legacy path: `create_session` leaves permission + capabilities NULL so the re-attach
        // helper falls back to the process default.
        let legacy = manager
            .create_session(Some(std::path::PathBuf::from("/tmp/legacy")))
            .await
            .expect("create legacy");
        let legacy_info = manager
            .session_info(legacy)
            .await
            .expect("session_info")
            .expect("legacy row");
        assert_eq!(legacy_info.permission, None);
        assert_eq!(legacy_info.capabilities_json, None);

        // Metadata path: persisted permission + capabilities + token_id round-trip verbatim.
        let with_meta = manager
            .create_session_with_metadata(
                Some(std::path::PathBuf::from("/tmp/meta")),
                Some("read".to_string()),
                Some(r#"{"supports_reasoning_stream":true}"#.to_string()),
                Some("token_fp_1234".to_string()),
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
            .update_session_permission(with_meta.id, "write")
            .await
            .expect("update permission");
        assert_eq!(updated, 1);
        let after_flip = manager
            .session_info(with_meta.id)
            .await
            .expect("session_info")
            .expect("post-flip row");
        assert_eq!(after_flip.permission.as_deref(), Some("write"));
    }

    // Child-session tests: parent→sub-agent linkage, cascade-on-delete, and `meka session list`
    // filter behavior.

    #[tokio::test]
    async fn test_create_child_session_writes_parent_id() {
        let manager = test_manager().await;
        let parent = manager.create_session(None).await.expect("create parent");
        let child = manager
            .create_child_session(parent, None)
            .await
            .expect("create child");

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
        let parent = manager.create_session(None).await.expect("create parent");
        let _child = manager
            .create_child_session(parent, None)
            .await
            .expect("create child");

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
            .create_session(Some(cwd.clone()))
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
            .create_session(Some(cwd_a.clone()))
            .await
            .expect("create a");
        let _b = manager
            .create_session(Some(cwd_b.clone()))
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
    async fn test_list_sessions_cwd_filter_excludes_legacy_null_rows() {
        // Sessions created via `create_session(None)` simulate legacy rows: they can't match any
        // cwd filter (NULL is never equal to a TEXT value in SQL).
        let manager = test_manager().await;
        let cwd = PathBuf::from("/home/agent/proj");
        let with_cwd = manager
            .create_session(Some(cwd.clone()))
            .await
            .expect("create with cwd");
        let _legacy = manager.create_session(None).await.expect("create legacy");

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
            let id = manager.create_session(None).await.expect("create");
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
        let parent = manager.create_session(None).await.expect("create parent");
        let child = manager
            .create_child_session(parent, None)
            .await
            .expect("create child");

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

    // MCP TokenStore tests. Exercise the methods backing `meka mcp login/logout` and the auth-probe
    // cache that skips unauthenticated connects after a 401. In-memory DB keeps each case hermetic.

    #[tokio::test]
    async fn mcp_credentials_round_trip() {
        let manager = test_manager().await;
        let store = manager.token_store();

        assert!(
            store
                .load_mcp_credentials("srv")
                .await
                .expect("load absent")
                .is_none(),
            "no credentials should exist yet"
        );

        store
            .save_mcp_credentials("srv", r#"{"tokens":{"access_token":"at1"}}"#)
            .await
            .expect("save");
        assert_eq!(
            store
                .load_mcp_credentials("srv")
                .await
                .expect("load")
                .as_deref(),
            Some(r#"{"tokens":{"access_token":"at1"}}"#)
        );

        // Upsert: second save replaces the first.
        store
            .save_mcp_credentials("srv", r#"{"tokens":{"access_token":"at2"}}"#)
            .await
            .expect("save again");
        assert_eq!(
            store
                .load_mcp_credentials("srv")
                .await
                .expect("load")
                .as_deref(),
            Some(r#"{"tokens":{"access_token":"at2"}}"#)
        );

        store.clear_mcp_credentials("srv").await.expect("clear");
        assert!(
            store
                .load_mcp_credentials("srv")
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
            .save_mcp_credentials("a", "alpha")
            .await
            .expect("save a");
        store
            .save_mcp_credentials("b", "beta")
            .await
            .expect("save b");
        store.clear_mcp_credentials("a").await.expect("clear a");
        assert!(store.load_mcp_credentials("a").await.unwrap().is_none());
        assert_eq!(
            store.load_mcp_credentials("b").await.unwrap().as_deref(),
            Some("beta")
        );
    }

    #[tokio::test]
    async fn auth_probe_round_trip_and_scoping() {
        let manager = test_manager().await;
        let store = manager.token_store();
        let ttl = std::time::Duration::from_secs(3600);

        assert!(store.load_auth_probe("srv", ttl).await.unwrap().is_none());

        store.save_auth_probe("srv", true).await.expect("save true");
        assert_eq!(store.load_auth_probe("srv", ttl).await.unwrap(), Some(true));

        // Upsert: flip the verdict.
        store
            .save_auth_probe("srv", false)
            .await
            .expect("save false");
        assert_eq!(
            store.load_auth_probe("srv", ttl).await.unwrap(),
            Some(false)
        );

        // A different server name is unaffected.
        assert!(store.load_auth_probe("other", ttl).await.unwrap().is_none());

        store.clear_auth_probe("srv").await.expect("clear");
        assert!(store.load_auth_probe("srv", ttl).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn auth_probe_honours_ttl() {
        let manager = test_manager().await;
        let store = manager.token_store();
        store.save_auth_probe("srv", true).await.expect("save");
        // A TTL of 0 seconds must treat any entry as stale.
        assert!(
            store
                .load_auth_probe("srv", std::time::Duration::from_secs(0))
                .await
                .unwrap()
                .is_none(),
            "entry should be stale under zero TTL"
        );
        // A large TTL still returns the same entry.
        assert_eq!(
            store
                .load_auth_probe("srv", std::time::Duration::from_secs(3600))
                .await
                .unwrap(),
            Some(true)
        );
    }

    #[tokio::test]
    async fn test_oauth_token_round_trip_preserves_all_fields() {
        let manager = SessionManager::open(Some(Path::new(":memory:")))
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
            .save_provider_credential("openai-codex", &credential)
            .await
            .expect("save");

        let loaded = store
            .load_provider_credential("openai-codex")
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
        let manager = SessionManager::open(Some(Path::new(":memory:")))
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
    /// verifies the provider PK keeps openai-codex and a hypothetical future OAuth provider
    /// isolated.
    #[tokio::test]
    async fn test_oauth_token_two_providers_independent() {
        let manager = SessionManager::open(Some(Path::new(":memory:")))
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
            .save_provider_credential("openai-codex", &codex_credential)
            .await
            .expect("save codex");
        store
            .save_provider_credential("claude", &claude_credential)
            .await
            .expect("save claude");

        let codex_loaded = store
            .load_provider_credential("openai-codex")
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
        let manager = SessionManager::open(Some(Path::new(":memory:")))
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
        let manager = SessionManager::open(Some(Path::new(":memory:")))
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
}
