//! The schema ledger: the only place in meka that knows a previous version of the store existed.
//!
//! Three rules hold this module apart from the rest of the program, and `AGENTS.md` states them as
//! hard rules because the whole benefit depends on them.
//!
//! **Nothing outside this module may know that an older meka existed.** No fallback reader for a
//! superseded shape, no `#[serde(alias)]` for a renamed field, no deprecation notice, no branch
//! whose condition is "was this row written by an older release". Every other reader assumes the
//! current schema unconditionally, and may do so because [`apply`] has already run by the time any
//! of them sees the store. That assumption is the thing being bought here. Scattered backwards
//! compatibility grows with every reader times every shape it has to tolerate and never goes away;
//! a migration converts once and leaves exactly one shape in the world.
//!
//! **A migration is frozen when it ships, and so are its dependencies.** It may not call meka's own
//! code. It reads and writes raw rows, and inlines whatever the logic meant at the time it was
//! written. [`gates_become_kind_and_spec`] builds its JSON by hand rather than through
//! `Gate::spec`, because a migration that borrowed that function would quietly start doing
//! something else the day the gate types are refactored, years after the users it ran for stopped
//! being able to notice.
//!
//! It may, though, *receive* a fact it cannot work out for itself. That is what [`Context`] carries
//! and what separates it from the ban above: a function can change meaning under a frozen step, a
//! `String` cannot. [`sessions_name_their_provider`] needs the profile a session with none recorded
//! should adopt, `config.toml` is the only place that knows, and this module must not read it.
//!
//! **A migration must be safe to run twice.** `user_version` lives in the file header, and the
//! standard SQLite round trip drops it: `sqlite3 old.db .dump | sqlite3 new.db` produces a store
//! with the right schema and a version of 0 (plain `VACUUM` keeps it; `.dump` does not). Such a
//! store is classified by shape, which can only answer "fresh" or "at the baseline", so every step
//! after the baseline runs again over data that already has them applied.
//!
//! [`gates_become_kind_and_spec`] survives that because it was written to: it guards each
//! `ADD COLUMN` on the column's absence and returns early when `gate_command` is already gone. A
//! plain `Step::Sql("ALTER TABLE … ADD COLUMN x")` in the same position would fail with
//! `duplicate column name` and refuse the store on every start, with no way forward. Prefer
//! `CREATE … IF NOT EXISTS`, guard `ALTER TABLE` on the current column set, and make data
//! conversions test for the shape they are converting *from* rather than assuming it.
//!
//! What is *not* banned elsewhere: guards against hand-editing, corruption and bugs. Those name no
//! release and are equally true of a store meka created five minutes ago, so they stay where the
//! data is read. `gate_kind` and `gate_spec` must both be set or both be null; an unparseable
//! `gate_permission` fails closed. Deleting those would turn fail-closed paths into fail-open ones,
//! and has nothing to do with versioning.
//!
//! The version lives in `PRAGMA user_version`, a 32-bit slot in the file header that SQLite
//! reserves for applications and never touches itself. It holds the number of migrations applied,
//! so head is `MIGRATIONS.len()`. It is transactional, so the DDL and the version bump commit or
//! fail together and no interruption can leave the schema ahead of the number that describes it.
//! `VACUUM INTO` copies it, so a restored backup identifies itself correctly. Note that SQLite's
//! own `PRAGMA schema_version` is a different thing entirely, an internal DDL counter that moves on
//! its own and differs between logically identical stores; it is not usable for this and must not
//! be written.

use crate::error::{MekaError, Result};

/// One step in the ledger. Append only: see the module docs on freezing.
struct Migration {
    /// Identifies the step in logs and in the frozen-prefix test. Never reused.
    name: &'static str,
    step: Step,
}

enum Step {
    /// Statements with no decisions in them.
    Sql(&'static str),
    /// A conversion SQL cannot express. Takes the transaction so it commits with everything else.
    Rust(fn(&rusqlite::Transaction<'_>) -> rusqlite::Result<()>),
    /// A conversion that also needs a fact only the caller can supply. See [`Context`].
    Contextual(fn(&rusqlite::Transaction<'_>, &Context) -> rusqlite::Result<()>),
}

/// Facts a migration cannot work out for itself, supplied by the caller.
///
/// This is the one loosening of rule 2, and the line it draws is between *receiving* data and
/// *calling* code to get it. `Gate::spec` can start meaning something else the day the gate types
/// are refactored, years after the stores it ran against stopped being able to notice; a `String`
/// cannot. So a migration may take one of these and may not reach for a function.
///
/// **Append only, like the ledger itself.** A shipped step's inputs are part of what is frozen
/// about it: adding a field for a later step is additive and safe, but renaming or removing one
/// silently rewrites what an already-run step would do.
#[derive(Debug, Clone, Default)]
pub struct Context {
    /// The profile a session with no recorded one adopts, or empty when nothing resolves.
    ///
    /// Empty is not a sentinel the readers know about. No profile can be named `""` in a way that
    /// resolves, so an empty value lands on the same "recorded profile is not configured" refusal
    /// a deleted profile produces, which every reader already has to handle.
    pub(crate) default_provider: String,
    /// Why `default_provider` is empty, when it is: because nothing resolved, or because the
    /// caller could not read `config.toml` at all.
    ///
    /// The two are not the same answer and must not produce the same write. "Nothing resolved" is
    /// a fact about a config meka read; "could not read it" is meka knowing nothing, and
    /// stamping every existing session against no profile on the strength of a parse error is
    /// irreversible, since the step runs once and `user_version` moves with it. So a step that
    /// would act on the value refuses instead, which aborts the transaction and leaves the
    /// store exactly as it was for a later run with a readable config.
    ///
    /// A *separate* field rather than making `default_provider` an `Option`, because `Context` is
    /// append-only for the same reason the ledger is: a shipped step's inputs are part of what is
    /// frozen about it, and changing the type of one would silently rewrite what an already-run
    /// step would have done.
    pub(crate) config_unreadable: bool,
}

impl Context {
    /// What a caller that has resolved a profile hands over. `None` means none resolved.
    pub(crate) fn adopting(profile: Option<&str>) -> Self {
        Self {
            default_provider: profile.unwrap_or_default().to_string(),
            config_unreadable: false,
        }
    }

    /// What a caller hands over when `config.toml` could not be parsed or read.
    ///
    /// Deliberately still openable: a store already at head runs no step that consults this, so the
    /// commands that exist to *repair* an unreadable config -- `meka mcp remove`, `meka provider
    /// remove`, which edit the raw document through `toml_edit` and never parse it -- keep working.
    /// Only a store that would actually adopt a profile is refused.
    pub(crate) fn on_unreadable_config() -> Self {
        Self {
            default_provider: String::new(),
            config_unreadable: true,
        }
    }
}

/// Every migration, in order. **Append only, and never edit a shipped entry**: users who already
/// ran it will not run it again, so an edit changes what new stores get and nothing else, which is
/// a divergence no test downstream of it can see. `the_released_prefix_is_frozen` fails the build
/// on any change to an entry that has shipped.
const MIGRATIONS: &[Migration] = &[
    Migration {
        name: "baseline_0_42",
        step: Step::Sql(BASELINE_0_42),
    },
    Migration {
        name: "gates_become_kind_and_spec",
        step: Step::Rust(gates_become_kind_and_spec),
    },
    Migration {
        name: "sessions_name_their_provider",
        step: Step::Contextual(sessions_name_their_provider),
    },
    Migration {
        name: "sessions_record_their_model_overrides",
        step: Step::Rust(sessions_record_their_model_overrides),
    },
    Migration {
        name: "mcp_credentials_hold_every_kind",
        step: Step::Rust(mcp_credentials_hold_every_kind),
    },
    Migration {
        name: "scheduled_jobs_forget_isolation",
        step: Step::Rust(scheduled_jobs_forget_isolation),
    },
];

/// What [`plan`] decided, and what [`apply`] will do about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Plan {
    /// The version the store is at, after classifying an unversioned one.
    pub(crate) from: u32,
    /// The version it will be at once [`apply`] returns.
    pub(crate) head: u32,
}

impl Plan {
    /// Whether anything would be written. The overwhelmingly common answer is `false`, and the
    /// caller uses it to skip both the backup and the transaction rather than paying for a
    /// no-op write on every process start.
    pub(crate) fn has_work(&self) -> bool {
        self.from < self.head
    }
}

/// Decide what this store needs, writing nothing.
///
/// Must be called with the schema lock held, and its answer used under the same lock: two processes
/// starting together would otherwise both read "needs migrating" and both try.
pub(crate) fn plan(connection: &rusqlite::Connection) -> Result<Plan> {
    let head = MIGRATIONS.len() as u32;
    let stored = user_version(connection)?;
    // A store from a newer meka. Refused rather than migrated, because the steps that would bring
    // it here do not exist in this binary and running the ones that do would be inventing a
    // downgrade.
    if stored > head {
        return Err(MekaError::Database(format!(
            "this store is at schema version {} and this meka only knows {}, so it was written by \
             a newer release. Nothing has been changed. Upgrade meka, or point MEKA_DATA_DIR at a \
             different store",
            stored, head
        )));
    }
    let from = if stored <= RETIRED_INITIALISED_FLAG {
        classify_by_shape(connection)?
    } else {
        stored
    };
    Ok(Plan { from, head })
}

/// `1` does not mean what this ledger would mean by it, so it is never taken at face value.
///
/// meka used to carry a different schema system, removed in 0.42, which stamped
/// `PRAGMA user_version = 1` as a one-shot "this database has been initialised" flag rather than as
/// a step counter. Every store that any release up to and including 0.41 finished opening still
/// carries it, which is most stores in existence and was true of the first real one this was tested
/// against.
///
/// Read as a ledger version it says "the baseline is applied", and for a 0.42-shaped store that
/// happens to be true. For a **0.41**-shaped store it is false and the consequence is severe:
/// trusting it skips [`classify_by_shape`], which is where the refusal for that shape lives, so the
/// gate conversion runs against a table with no `gate_permission`, drops the two columns it did
/// read, and commits. The store is then stamped at head, so nothing will revisit it, and every
/// later read of `scheduled_jobs` fails with `no such column: gate_permission`. Reproduced end to
/// end before this guard existed; the only way back was the backup.
///
/// Distrusting it costs nothing, because this ledger cannot produce a `1`: [`apply`] always stamps
/// `MIGRATIONS.len()`, which `the_ledger_can_never_stamp_the_retired_flag` pins at two or more.
const RETIRED_INITIALISED_FLAG: u32 = 1;

/// Apply everything [`plan`] found pending, and stamp the new version, in one transaction.
///
/// Foreign keys are suspended for the duration, and this is the reason the two halves are split
/// across two functions. `PRAGMA foreign_keys` is a **no-op inside a transaction**, so it has to be
/// set before `BEGIN`, which the transaction-owning half cannot do. SQLite's documented procedure
/// for the table changes `ALTER TABLE` cannot express -- changing a column's type, adding or
/// removing `NOT NULL`, changing a default, dropping a constraint -- is to build a new table, copy,
/// drop the old, and rename, and it requires enforcement off. With it on, `DROP TABLE sessions`
/// cascades through `messages`, `tool_outputs`, `scheduled_jobs` and `background_tasks`, deleting
/// the entire conversation history inside a transaction that then commits successfully. Measured:
/// one child row before, zero after, with the pragma reading `1` throughout because the attempt to
/// turn it off was ignored. `PRAGMA defer_foreign_keys` does not help.
///
/// Neither shipped step rebuilds a table, so this changes nothing today. It is here now because
/// `apply`'s transaction boundary is itself a shipped decision: the first migration that needs a
/// rebuild would otherwise have to change it, and would probably not notice why it had to.
/// [`apply_steps`] runs `foreign_key_check` before committing, since nothing was enforcing
/// references while the steps ran.
pub(crate) fn apply(
    connection: &mut rusqlite::Connection,
    plan: Plan,
    context: &Context,
) -> Result<()> {
    if !plan.has_work() {
        return Ok(());
    }
    let applied = with_foreign_keys_suspended(connection, |connection| {
        apply_steps(connection, plan, context)
    })?;
    // After the commit, so a migration that rolled back is not reported as applied.
    for name in applied {
        tracing::info!("applied schema migration '{}'", name);
    }
    Ok(())
}

/// Run `work` with foreign-key enforcement off, and restore it whichever way that goes.
///
/// Separated from [`apply`] so the property can be tested rather than argued for. The thing that
/// has to be true is "a step that rebuilds a table does not cascade-delete its children", and no
/// shipped migration rebuilds one, so with the suspension inlined there was nothing a test could
/// reach: the mutation sweep could only confirm the *count* was read, never that the guard worked.
/// A closure lets a test hand in the rebuild that does not exist in `MIGRATIONS`.
///
/// The restore is not optional and not best-effort. This connection goes on to serve the whole
/// process, and every write after this point expects enforcement to be live; a failure to put it
/// back is reported even when the migration itself succeeded, and logged even when the migration
/// failed too and its error is the one returned.
///
/// A panic inside a step is the one path that skips the restore. No shipped step can panic (none
/// indexes, slices or unwraps), and a future one must not either; that is part of what "a migration
/// is frozen and self-contained" buys.
fn with_foreign_keys_suspended<T>(
    connection: &mut rusqlite::Connection,
    work: impl FnOnce(&mut rusqlite::Connection) -> Result<T>,
) -> Result<T> {
    connection
        .execute_batch("PRAGMA foreign_keys = OFF;")
        .map_err(|error| {
            MekaError::Database(format!(
                "failed to suspend foreign keys for the schema migration: {}. Nothing has been \
                 changed",
                error
            ))
        })?;
    let outcome = work(connection);
    let restored = connection.execute_batch("PRAGMA foreign_keys = ON;");
    // Said out loud even when the migration is the thing that failed. Returning only the migration
    // error is right -- it is the more useful message and the reason the caller is unwinding -- but
    // dropping this one silently would hide that the connection is now unsafe as well.
    if let Err(error) = &restored {
        tracing::error!(
            "foreign keys could not be re-enabled after the schema migration: {}. Restart meka \
             rather than continuing with enforcement off",
            error
        );
    }
    let outcome = outcome?;
    restored.map_err(|error| {
        MekaError::Database(format!(
            "the schema migration committed but foreign keys could not be re-enabled on this \
             connection: {}. Restart meka rather than continuing with enforcement off",
            error
        ))
    })?;
    Ok(outcome)
}

/// The transaction half. `Immediate`, so the write lock is taken at `BEGIN` rather than on the
/// first write: under WAL a deferred transaction that upgrades later can return `SQLITE_BUSY`
/// without consulting the busy handler at all, which is the same reason
/// [`crate::memory::store`] gives for its own writes.
fn apply_steps(
    connection: &mut rusqlite::Connection,
    plan: Plan,
    context: &Context,
) -> Result<Vec<&'static str>> {
    let transaction = connection
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(|error| {
            MekaError::Database(format!("failed to begin the schema migration: {}", error))
        })?;
    // Counted before as well as after, so what fails the migration is damage *it* caused rather
    // than damage it inherited. A store that already carries a dangling reference, which takes
    // hand-editing to arrange because enforcement is on for every normal write, would otherwise be
    // refused every start forever with no way forward.
    let dangling_before = count_dangling_references(&transaction)?;
    let mut applied = Vec::new();
    for (index, migration) in MIGRATIONS.iter().enumerate().skip(plan.from as usize) {
        match &migration.step {
            Step::Sql(sql) => transaction.execute_batch(sql),
            Step::Rust(step) => step(&transaction),
            Step::Contextual(step) => step(&transaction, context),
        }
        .map_err(|error| {
            MekaError::Database(format!(
                "schema migration {} ('{}') failed: {}. The store is unchanged",
                index + 1,
                migration.name,
                error
            ))
        })?;
        applied.push(migration.name);
    }
    // The price of suspending enforcement: a step that orphaned a row would otherwise commit it
    // silently, and the damage would only surface much later as a row pointing at a parent that is
    // not there.
    let dangling_after = count_dangling_references(&transaction)?;
    if dangling_after > dangling_before {
        return Err(MekaError::Database(format!(
            "the schema migration would have left {} row(s) referring to a parent that is not \
             there, so it was rolled back. The store is unchanged",
            dangling_after - dangling_before
        )));
    }
    if dangling_before > 0 {
        tracing::warn!(
            "this store already carried {} row(s) referring to a parent that is not there. The \
             migration did not add to them and has not removed them",
            dangling_before
        );
    }
    set_user_version(&transaction, plan.head)?;
    transaction.commit().map_err(|error| {
        MekaError::Database(format!(
            "failed to commit the schema migration: {}. The store is unchanged",
            error
        ))
    })?;
    Ok(applied)
}

/// How many rows point at a parent row that is not there.
///
/// A full scan of every foreign key in the store, so it is run twice per migration and not at all
/// when there is nothing to do. `pragma_foreign_key_check` reports one row per violation; only the
/// count is wanted here, because the migration is refused wholesale either way.
fn count_dangling_references(transaction: &rusqlite::Transaction<'_>) -> Result<i64> {
    transaction
        .query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |row| {
            row.get(0)
        })
        .map_err(|error| {
            MekaError::Database(format!(
                "failed to check foreign keys during the schema migration: {}. The store is \
                 unchanged",
                error
            ))
        })
}

fn user_version(connection: &rusqlite::Connection) -> Result<u32> {
    connection
        .query_row("SELECT * FROM pragma_user_version", [], |row| {
            row.get::<_, i64>(0)
        })
        // Clamped rather than trusted: the column is a signed 32-bit slot anyone can write, and a
        // negative value would otherwise wrap into a huge version and read as "newer than this
        // binary".
        .map(|version| version.max(0) as u32)
        .map_err(|error| {
            MekaError::Database(format!("failed to read the store's schema version: {}", error))
        })
}

fn set_user_version(transaction: &rusqlite::Transaction<'_>, version: u32) -> Result<()> {
    // Formatted rather than bound because pragmas do not accept bind parameters at all
    // (`PRAGMA user_version = ?` is a parse error). Not an injection surface, and not a candidate
    // for being "fixed" into a bound parameter later: `version` is a `u32` this module computed
    // from `MIGRATIONS.len()` and no caller can influence it.
    transaction
        .execute_batch(&format!("PRAGMA user_version = {};", version))
        .map_err(|error| {
            MekaError::Database(format!(
                "failed to record the new schema version: {}",
                error
            ))
        })
}

/// Decide what a store already is by looking at it, for the versions that cannot be trusted.
///
/// Reached when `user_version` is 0 (never stamped) or [`RETIRED_INITIALISED_FLAG`] (stamped by a
/// system that meant something else by it). The answer is stamped by [`apply`], so this runs once
/// per store and never again: it is the whole of the adoption problem, and it is why the ban on
/// version knowledge everywhere else costs nothing.
///
/// Markers first, each the column or table that the release in question introduced. `sessions` has
/// existed for as long as meka has had a store, so its absence means there is nothing of meka's
/// here. `gate_permission` is what 0.42 added, so a `scheduled_jobs` without it is exactly 0.41.
///
/// Then completeness, because returning `1` asserts that *everything* the baseline creates is
/// already there, and the old code that would have quietly filled a gap is gone: it ran
/// `CREATE TABLE IF NOT EXISTS` for every object on every open. It also ran them as six separate
/// statements in autocommit, so a first run of any pre-0.43 release interrupted partway leaves
/// exactly this state, with `background_tasks` and the memory tables likeliest because they were
/// last. Without the check such a store is stamped at head with a table still missing, which
/// nothing will ever revisit; measured, `meka session list` succeeded and left it that way.
/// Refusing names what is wrong instead.
fn classify_by_shape(connection: &rusqlite::Connection) -> Result<u32> {
    // Not a meka store: an empty file, or one carrying tables meka did not write. Nothing to carry
    // forward either way, so build the schema alongside whatever is already there, which is what
    // every release before this one did to such a file too.
    if table_columns(connection, "sessions")?.is_empty() {
        return Ok(0);
    }
    let columns = table_columns(connection, "scheduled_jobs")?;
    if columns.is_empty() {
        return Err(MekaError::Database(
            "this store has tables but no `scheduled_jobs`, so it predates 0.42 and this meka \
             cannot bring it forward. Nothing has been changed. Run the 0.42 release against it \
             once, then this one"
                .to_string(),
        ));
    }
    if !columns.iter().any(|column| column == "gate_permission") {
        return Err(MekaError::Database(
            "this store is in the 0.41 shape, which this meka cannot bring forward. Nothing has \
             been changed. Run `migrate-0.41-to-0.42.py`, attached to the 0.42 release, once; \
             every upgrade after that is automatic"
                .to_string(),
        ));
    }
    let mut missing = Vec::new();
    for names in BASELINE_OBJECTS {
        let mut found = false;
        for name in *names {
            let present: i64 = connection
                .query_row(
                    "SELECT count(*) FROM sqlite_master WHERE name = ?1",
                    [name],
                    |row| row.get(0),
                )
                .map_err(|error| {
                    MekaError::Database(format!("failed to inspect the store's objects: {}", error))
                })?;
            if present > 0 {
                found = true;
                break;
            }
        }
        // Reported under the baseline's own name. A store that reaches here has none of the later
        // ones either, and naming a table this meka would create under a different name would send
        // the reader looking for the wrong thing in their backup.
        if !found && let Some(baseline) = names.first() {
            missing.push(*baseline);
        }
    }
    if !missing.is_empty() {
        return Err(MekaError::Database(format!(
            "this store is missing {}, which every release from 0.42 creates, so it was probably \
             left half-built by an interrupted first run. Nothing has been changed. Restore it \
             from a backup, or move it aside and let meka build a new one",
            missing.join(", ")
        )));
    }
    Ok(1)
}

/// Every **table** [`BASELINE_0_42`] creates, by the name it appears under in `sqlite_master`.
///
/// Tables only, deliberately. A missing table means the store is not at the baseline and the
/// classification would be a lie; a missing *index* means the same queries return the same
/// answers more slowly, and refusing to start over one would be worse than the problem. The
/// seven indexes the baseline creates are therefore not checked, and not repaired either:
/// the old `CREATE INDEX IF NOT EXISTS` on every open used to put a dropped one back, and
/// nothing does now. That is a real if small regression, accepted because the alternative is
/// the declare-and-heal pattern the ledger exists to replace.
///
/// Read by [`classify_by_shape`] to check that a store claiming to be at the baseline really is.
/// Deliberately a separate list rather than parsed out of the SQL: it is the *question* asked of an
/// old store, which is frozen for the same reason the migration is, whereas the SQL is the answer
/// given to a new one. A later migration that adds a table does not belong here.
///
/// Each entry is every name the object may legitimately appear under, first the baseline's. More
/// than one only where a later step **renamed** a table the baseline created, because the question
/// this list asks is whether the store was fully built rather than left half written, and a rename
/// does not make it less so. A store that was carried forward and then lost its `user_version`
/// shows the new name, and refusing it would tell the user their complete store was half-built.
/// This is the one list allowed to know that, for the reason [`classify_by_shape`] gives about
/// itself.
const BASELINE_OBJECTS: &[&[&str]] = &[
    &["background_tasks"],
    // Renamed by `mcp_credentials_hold_every_kind` once it held more than OAuth bundles.
    &["mcp_oauth_credentials", "mcp_credentials"],
    &["memories"],
    &["memories_fts"],
    &["messages"],
    &["provider_credentials"],
    &["scheduled_jobs"],
    &["sessions"],
    &["tool_outputs"],
];

fn table_columns(connection: &rusqlite::Connection, table: &str) -> Result<Vec<String>> {
    let mut statement = connection
        .prepare("SELECT name FROM pragma_table_info(?1)")
        .map_err(|error| {
            MekaError::Database(format!("failed to inspect `{}`: {}", table, error))
        })?;
    let names = statement
        .query_map([table], |row| row.get::<_, String>(0))
        .and_then(|rows| rows.collect::<rusqlite::Result<Vec<_>>>())
        .map_err(|error| {
            MekaError::Database(format!("failed to inspect `{}`: {}", table, error))
        })?;
    Ok(names)
}

/// Every table, index and virtual table as 0.42 left them.
///
/// A fresh install runs this and then every step after it, so the schema a new store gets is
/// produced by the same code path an upgraded one goes through. That replay is the point: two
/// separate definitions of "the current schema" drift, and
/// `a_fresh_store_and_an_upgraded_one_have_the_same_schema` is what proves this one cannot.
///
/// The FTS *triggers* are deliberately absent. `crate::memory::store`'s `sync_triggers` owns their
/// creation and reconciles them on every open, and a `CREATE TRIGGER IF NOT EXISTS` here would put
/// a trigger that had gone missing back before that comparison could notice, which is a silent
/// index desync that module documents having reproduced.
const BASELINE_0_42: &str = "
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        parent_session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
        cwd TEXT,
        permission TEXT,
        capabilities_json TEXT,
        token_id TEXT,
        additional_roots_json TEXT,
        subagent_spec_json TEXT,
        stat_turns INTEGER NOT NULL DEFAULT 0,
        stat_input_tokens INTEGER NOT NULL DEFAULT 0,
        stat_output_tokens INTEGER NOT NULL DEFAULT 0,
        stat_cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
        stat_cache_read_input_tokens INTEGER NOT NULL DEFAULT 0,
        stat_redactions INTEGER NOT NULL DEFAULT 0,
        stat_redacted_images INTEGER NOT NULL DEFAULT 0,
        stat_redacted_bytes INTEGER NOT NULL DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at);

    CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);

    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);

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

    CREATE TABLE IF NOT EXISTS tool_outputs (
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (session_id, name)
    );

    CREATE TABLE IF NOT EXISTS scheduled_jobs (
        id                TEXT PRIMARY KEY,
        session_id        TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        kind              TEXT NOT NULL,
        spec              TEXT NOT NULL,
        prompt            TEXT NOT NULL,
        gate_command      TEXT,
        gate_fire         TEXT,
        gate_last_output  TEXT,
        gate_permission   TEXT,
        isolated          INTEGER NOT NULL DEFAULT 0,
        created_at        TEXT NOT NULL,
        last_fired_at     TEXT,
        next_fire_at      TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_next_fire ON scheduled_jobs(next_fire_at);

    CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_session ON scheduled_jobs(session_id);

    CREATE TABLE IF NOT EXISTS background_tasks (
        id                TEXT PRIMARY KEY,
        session_id        TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        tool_name         TEXT NOT NULL,
        label             TEXT NOT NULL,
        status            TEXT NOT NULL,
        outcome           TEXT,
        scratchpad_name   TEXT,
        started_at        TEXT NOT NULL,
        finished_at       TEXT,
        delivered_at      TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_background_tasks_session_status
        ON background_tasks(session_id, status);

    CREATE TABLE IF NOT EXISTS memories (
        id           INTEGER PRIMARY KEY,
        name         TEXT NOT NULL UNIQUE COLLATE NOCASE,
        description  TEXT NOT NULL,
        tags         TEXT NOT NULL DEFAULT '',
        body         TEXT NOT NULL DEFAULT '',
        priority     INTEGER NOT NULL DEFAULT 5,
        recorded_at  TEXT NOT NULL,
        updated_at   TEXT NOT NULL,
        read_count   INTEGER NOT NULL DEFAULT 0,
        last_read_at TEXT
    );

    CREATE INDEX IF NOT EXISTS memories_rank ON memories(priority, recorded_at DESC);

    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        name,
        description,
        tags,
        body,
        content = 'memories',
        content_rowid = 'id',
        tokenize = 'porter unicode61'
    );
";

/// 0.43: a gate becomes `gate_kind` plus a JSON `gate_spec`, and a due occurrence is leased.
///
/// Both old values were written only by meka, so the predicate mapping is a rename rather than a
/// guess.
fn gates_become_kind_and_spec(transaction: &rusqlite::Transaction<'_>) -> rusqlite::Result<()> {
    let columns: Vec<String> = {
        let mut statement = transaction.prepare("SELECT name FROM pragma_table_info(?1)")?;
        let rows = statement.query_map(["scheduled_jobs"], |row| row.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    let has = |column: &str| columns.iter().any(|existing| existing == column);

    for (column, statement) in [
        (
            "gate_kind",
            "ALTER TABLE scheduled_jobs ADD COLUMN gate_kind TEXT",
        ),
        (
            "gate_spec",
            "ALTER TABLE scheduled_jobs ADD COLUMN gate_spec TEXT",
        ),
        (
            "claimed_by",
            "ALTER TABLE scheduled_jobs ADD COLUMN claimed_by TEXT",
        ),
        (
            "claimed_until",
            "ALTER TABLE scheduled_jobs ADD COLUMN claimed_until TEXT",
        ),
        (
            "attempts",
            "ALTER TABLE scheduled_jobs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0",
        ),
    ] {
        if !has(column) {
            transaction.execute_batch(statement)?;
        }
    }

    // A store whose gates are already in the new shape, which is what a hand-run of the retired
    // script leaves behind. The lease columns above are still worth reaching, because an early
    // build of that script added the gate columns without them.
    if !has("gate_command") {
        return Ok(());
    }

    let mut unconvertible: Vec<String> = Vec::new();
    {
        let mut statement = transaction.prepare(
            "SELECT id, gate_command, gate_fire FROM scheduled_jobs \
             WHERE gate_command IS NOT NULL OR gate_fire IS NOT NULL",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
            ))
        })?;
        for row in rows {
            let (id, command, fire) = row?;
            let converted = match (command, fire) {
                (Some(command), Some(fire)) => predicate_for(&fire).map(|predicate| {
                    // Built through `serde_json` rather than by hand because a gate command is
                    // arbitrary user text: quotes, newlines and non-ASCII all have to survive, and
                    // hand-rolled escaping is where that goes wrong. The *shape* is inlined
                    // deliberately, mirroring `GateSpec` as it stands today without borrowing it;
                    // see the module docs on frozen dependencies.
                    serde_json::json!({ "shell": { "command": command }, "when": predicate })
                        .to_string()
                }),
                _ => None,
            };
            match converted {
                Some(spec) => {
                    transaction.execute(
                        "UPDATE scheduled_jobs SET gate_kind = 'shell', gate_spec = ?2 \
                         WHERE id = ?1",
                        rusqlite::params![id, spec],
                    )?;
                }
                // A row 0.42 refused to load: a half-written gate, or a `gate_fire` value it did
                // not recognise. Preserved in exactly the state it was already in rather than
                // guessed at or deleted, by setting `gate_kind` and leaving `gate_spec` null, which
                // the reader's existing corrupt-row rule refuses the same way 0.42's did. Getting
                // this wrong is expensive and silent in one specific direction: leaving both null
                // reads as *no gate at all*, which turns a watcher that never fired into a timer
                // that fires every interval.
                None => {
                    transaction.execute(
                        "UPDATE scheduled_jobs SET gate_kind = 'shell', gate_spec = NULL \
                         WHERE id = ?1",
                        rusqlite::params![&id],
                    )?;
                    unconvertible.push(id);
                }
            }
        }
    }

    transaction.execute_batch(
        "ALTER TABLE scheduled_jobs DROP COLUMN gate_command;
         ALTER TABLE scheduled_jobs DROP COLUMN gate_fire;",
    )?;

    if !unconvertible.is_empty() {
        tracing::warn!(
            "{} scheduled job(s) had a gate that could not be read and were left inert, exactly as \
             the previous release left them: {}. They will not fire, and they will not appear in \
             `meka schedule list` or be reachable by `meka schedule cancel`. Recreate them if you \
             still want them; the pre-migration backup has the originals",
            unconvertible.len(),
            unconvertible.join(", ")
        );
    }
    Ok(())
}

/// 0.44: a session records the provider profile it runs with.
///
/// The column is `NOT NULL` because every session created from here on has a resolved profile at
/// the moment it is written, so the readers get to assume one unconditionally. Existing rows have
/// no such fact to recover, and this is the only place allowed to invent one.
///
/// The value is whatever the caller resolved as the default. When nothing resolved, the sole stored
/// credential is a better guess than nothing: a store with sessions in it was used, and a single
/// credential is almost certainly the profile that ran them. Two or more cannot be told apart from
/// here, so those fall through to the empty string, which no configured profile can be named and so
/// resolves to the refusal a deleted profile already produces.
fn sessions_name_their_provider(
    transaction: &rusqlite::Transaction<'_>,
    context: &Context,
) -> rusqlite::Result<()> {
    let columns = {
        let mut statement = transaction.prepare("SELECT name FROM pragma_table_info(?1)")?;
        let rows = statement.query_map(["sessions"], |row| row.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<Vec<String>>>()?
    };
    // Guarded rather than bare, so a store that lost its `user_version` and replays this step does
    // not fail with `duplicate column name` and refuse to open on every start afterwards.
    if !columns.iter().any(|column| column == "provider") {
        transaction
            .execute_batch("ALTER TABLE sessions ADD COLUMN provider TEXT NOT NULL DEFAULT ''")?;
    }

    // Refused, not guessed. An unreadable `config.toml` means the caller could not tell us what
    // these sessions should adopt, and this step gets one attempt: it runs once, `user_version`
    // moves with it, and every row it stamps against no profile stays that way. Erroring aborts the
    // transaction, so the store is untouched and a later run with a readable config migrates it
    // correctly. The `ALTER TABLE` above is inside the same transaction and goes with it.
    if context.config_unreadable {
        let carried: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM sessions WHERE provider = ''",
            [],
            |row| row.get(0),
        )?;
        if carried > 0 {
            // `InvalidParameterName` carries an arbitrary message, which is the idiom `session.rs`
            // already uses to surface a non-SQLite failure through a `rusqlite::Result`. Its
            // `Display` prefixes the variant name, which reads oddly here but is the price of the
            // one variant available without enabling a rusqlite feature for a single error path.
            return Err(rusqlite::Error::InvalidParameterName(format!(
                "cannot record a provider for {} carried-forward session(s) while config.toml \
                 cannot be read; fix the file and start meka again",
                carried
            )));
        }
    }

    let adopted = if context.default_provider.is_empty() {
        sole_credential(transaction)?.unwrap_or_default()
    } else {
        context.default_provider.clone()
    };
    if adopted.is_empty() {
        // Only when there is something to say. Warning unconditionally fired on the literal first
        // run of a fresh install, where the sessions table is empty and there is nothing to leave
        // without a provider, and on every `meka provider add` before a default exists. A warning
        // that appears when nothing is wrong teaches the reader to ignore the one that matters.
        let stranded: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM sessions WHERE provider = ''",
            [],
            |row| row.get(0),
        )?;
        if stranded > 0 {
            tracing::warn!(
                "no provider profile could be resolved, so {} existing session(s) were left \
                 without one. Configure a provider, then resume such a session with --provider \
                 once to say which it runs on; that is what records it",
                stranded
            );
        }
        return Ok(());
    }
    // Only the rows that have nothing, which is what makes a replay a no-op rather than a rewrite
    // of a session someone has since moved to another profile.
    let adopting = transaction.execute(
        "UPDATE sessions SET provider = ?1 WHERE provider = ''",
        rusqlite::params![&adopted],
    )?;
    // Said out loud, because for anyone with more than one account this is a guess. The alternative
    // was leaving these sessions unresumable, so adopting is right; doing it silently is not, since
    // a session that actually ran on the other account now names this one and resuming it bills
    // here. `info!` and not `warn!` because it is a one-time lifecycle signpost that is simply
    // correct for a single-profile install, and a warning that cries wolf for most readers is one
    // the rest learn to skip.
    if adopting > 0 {
        tracing::info!(
            "recorded provider profile '{}' on {} session(s) that predate meka recording one. \
             Resume any that ran on a different profile once with --provider <name> to correct it",
            adopted,
            adopting
        );
    }
    Ok(())
}

/// Add the two columns that hold a session's per-session model and endpoint.
///
/// Nullable with no backfill, and that is the correct value rather than a shortcut: absent means
/// "whatever the profile says", which is exactly what every session written before these columns
/// existed did. There is nothing to convert.
fn sessions_record_their_model_overrides(
    transaction: &rusqlite::Transaction<'_>,
) -> rusqlite::Result<()> {
    let columns = {
        let mut statement = transaction.prepare("SELECT name FROM pragma_table_info(?1)")?;
        let rows = statement.query_map(["sessions"], |row| row.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<Vec<String>>>()?
    };
    // Guarded for the reason the module docs give: a store that lost its `user_version` replays
    // every step after the baseline, and a bare `ADD COLUMN` would then refuse it forever.
    if !columns.iter().any(|column| column == "model_override") {
        transaction.execute_batch("ALTER TABLE sessions ADD COLUMN model_override TEXT")?;
    }
    if !columns.iter().any(|column| column == "base_url_override") {
        transaction.execute_batch("ALTER TABLE sessions ADD COLUMN base_url_override TEXT")?;
    }
    Ok(())
}

/// One table for every MCP secret, not only the OAuth ones.
///
/// `mcp_oauth_credentials` was named for the only kind it could hold. A server's static bearer and
/// its `client_secret` stopped being config keys in this release and land here beside the OAuth
/// bundles, so the name had become a lie about its own contents.
///
/// `kind` is what a reader consults to know how to interpret `secret`, and it exists so that
/// nothing has to guess from the value's shape. Sniffing would be the same mistake as version
/// sniffing: a rule about what some other component's JSON happens to look like, which goes stale
/// with nothing to notice. The column is `secret` rather than the old `credentials_json` because
/// only one of the three kinds is JSON; a bearer and a client secret are the string itself.
///
/// **The key is `(server_name, kind)`, and that is the whole reason this is a rebuild rather than a
/// rename.** One server can hold two secrets at once: `McpAuthConfig::OAuth` takes an optional
/// `client_secret`, so a confidential client has a long-lived secret *and* the refreshable bundle
/// obtained with it. Under the old `server_name` primary key those two collide, and the first token
/// refresh silently overwrites the client secret. That server then works until the refresh and
/// fails afterwards, which is about the worst shape a failure can take. SQLite cannot alter a
/// primary key, so the table is built anew and the rows copied across.
///
/// Every row that already exists came from the authorization-code flow, which is the only one that
/// ever persisted anything, so they all copy over as `oauth` and there is nothing to work out.
fn mcp_credentials_hold_every_kind(
    transaction: &rusqlite::Transaction<'_>,
) -> rusqlite::Result<()> {
    let table_exists = |name: &str| -> rusqlite::Result<bool> {
        let count: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
            [name],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    };

    // Guarded on the target for the reason the module docs give: a store that lost its
    // `user_version` replays every step after the baseline, and a second pass would otherwise fail
    // on `CREATE TABLE` and refuse that store on every start afterwards.
    if table_exists("mcp_credentials")? {
        return Ok(());
    }
    transaction.execute_batch(
        "CREATE TABLE mcp_credentials (
            server_name TEXT NOT NULL,
            kind        TEXT NOT NULL,
            secret      TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            PRIMARY KEY (server_name, kind)
        )",
    )?;
    // Absent only for a store built by a release that never created it, which the baseline check in
    // `classify_by_shape` has already ruled out for anything this reaches. Guarded anyway, because
    // a step that assumes its predecessor's output is the assumption the replay rule exists to
    // break.
    if table_exists("mcp_oauth_credentials")? {
        transaction.execute_batch(
            "INSERT INTO mcp_credentials (server_name, kind, secret, updated_at)
             SELECT server_name, 'oauth', credentials_json, updated_at FROM mcp_oauth_credentials",
        )?;
        transaction.execute_batch("DROP TABLE mcp_oauth_credentials")?;
    }
    Ok(())
}

/// 0.44: a scheduled job always fires in the session that created it.
///
/// `isolated` let a job run its turn in a fresh top-level session instead of the conversation that
/// created it. The flag was honoured only by `meka serve`; the REPL and ACP each drive one
/// conversation per session and ran such a job in the open one, with a warning. It is gone, so
/// every host now does what two of the three already did.
///
/// Nothing is converted, because there is nothing to convert: a job's identity is its session, its
/// schedule and its prompt, and only where the turn landed changes. A row that carried `1` keeps
/// firing on exactly its old schedule, into the conversation it belongs to.
///
/// Dropping the column rather than leaving it inert is what earns every reader the right to stop
/// asking. The row decoder addresses columns by position, so a dead column is not free: it stays in
/// every `SELECT` list and every index below it counts around it, which is a standing invitation to
/// an off-by-one in a place where the symptom is a timestamp that will not parse.
fn scheduled_jobs_forget_isolation(
    transaction: &rusqlite::Transaction<'_>,
) -> rusqlite::Result<()> {
    let columns: Vec<String> = {
        let mut statement = transaction.prepare("SELECT name FROM pragma_table_info(?1)")?;
        let rows = statement.query_map(["scheduled_jobs"], |row| row.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    // Guarded for the reason the module docs give: a store that lost its `user_version` replays
    // every step after the baseline, and a bare `DROP COLUMN` on the second pass fails with
    // `no such column` and refuses that store on every start afterwards.
    if columns.iter().any(|column| column == "isolated") {
        transaction.execute_batch("ALTER TABLE scheduled_jobs DROP COLUMN isolated")?;
    }
    Ok(())
}

/// The profile named by the only stored credential, when there is exactly one.
///
/// Ambiguity is not resolved by picking: with two credentials there is no evidence here about which
/// of them ran any given session, and a confident wrong answer is worse than an admitted absence.
fn sole_credential(transaction: &rusqlite::Transaction<'_>) -> rusqlite::Result<Option<String>> {
    let mut statement = transaction.prepare("SELECT profile FROM provider_credentials LIMIT 2")?;
    let profiles = statement
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<String>>>()?;
    match profiles.as_slice() {
        [only] => Ok(Some(only.clone())),
        _ => Ok(None),
    }
}

fn predicate_for(fire: &str) -> Option<&'static str> {
    match fire {
        "on-change" => Some("changed"),
        "on-success" => Some("succeeded"),
        _ => None,
    }
}

/// Build a store at head, for a test that needs the schema without a whole
/// [`crate::session::SessionManager`].
///
/// Deliberately [`plan`] then [`apply`], the same pair production calls, rather than a private loop
/// over `MIGRATIONS`. A second way to build the schema is a second thing that can be right while
/// the real one is wrong, and the tests that would notice are exactly the ones using this.
#[cfg(test)]
pub(crate) fn create_for_test(connection: &mut rusqlite::Connection) -> Result<()> {
    let plan = plan(connection)?;
    apply(connection, plan, &Context::default())
}

/// The baseline DDL, for a test that needs to plant a store shaped the way an older meka left one.
#[cfg(test)]
pub(crate) fn baseline_for_test() -> &'static str {
    BASELINE_0_42
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> rusqlite::Connection {
        rusqlite::Connection::open_in_memory().expect("an in-memory database")
    }

    /// The schema as SQLite understands it, rather than as it was typed.
    ///
    /// Compared structurally because the two paths this file cares about cannot produce identical
    /// text even when they are identical schemas: one runs `CREATE TABLE`, the other runs that plus
    /// `ALTER TABLE ADD`/`DROP COLUMN`, and SQLite rewrites `sqlite_master.sql` differently for
    /// each. `pragma_table_info` answers what the columns actually are, which is the thing that has
    /// to match.
    fn fingerprint(connection: &rusqlite::Connection) -> String {
        let objects: Vec<(String, String)> = {
            let mut statement = connection
                .prepare(
                    "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' \
                     ORDER BY type, name",
                )
                .expect("sqlite_master is readable");
            let rows = statement
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                .expect("sqlite_master rows");
            rows.collect::<rusqlite::Result<_>>()
                .expect("sqlite_master rows")
        };
        let mut lines = Vec::new();
        for (kind, name) in objects {
            lines.push(format!("{} {}", kind, name));
            if kind != "table" {
                continue;
            }
            let mut statement = connection
                .prepare(
                    "SELECT cid, name, type, \"notnull\", ifnull(dflt_value, ''), pk \
                     FROM pragma_table_info(?1) ORDER BY cid",
                )
                .expect("pragma_table_info is queryable");
            let columns = statement
                .query_map([&name], |row| {
                    Ok(format!(
                        "  {} {} {} notnull={} default={} pk={}",
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, i64>(5)?,
                    ))
                })
                .expect("column rows")
                .collect::<rusqlite::Result<Vec<_>>>()
                .expect("column rows");
            lines.extend(columns);
        }
        lines.join("\n")
    }

    /// A store as the 0.42 binary left one: the baseline shape, and nothing in `user_version`.
    fn store_as_0_42_left_it() -> rusqlite::Connection {
        let connection = fresh();
        connection
            .execute_batch(BASELINE_0_42)
            .expect("the baseline builds");
        connection
    }

    fn plant_job(
        connection: &rusqlite::Connection,
        id: &str,
        command: Option<&str>,
        fire: Option<&str>,
    ) {
        connection
            .execute(
                "INSERT INTO sessions (id, created_at, updated_at) VALUES ('s', 'now', 'now') \
                 ON CONFLICT(id) DO NOTHING",
                [],
            )
            .expect("a session to hang the job from");
        connection
            .execute(
                "INSERT INTO scheduled_jobs \
                 (id, session_id, kind, spec, prompt, gate_command, gate_fire, gate_permission, \
                  created_at, next_fire_at) \
                 VALUES (?1, 's', 'every', '30s', 'p', ?2, ?3, 'unrestricted', 'now', 'later')",
                rusqlite::params![id, command, fire],
            )
            .expect("a planted job");
    }

    fn gate_of(connection: &rusqlite::Connection, id: &str) -> (Option<String>, Option<String>) {
        connection
            .query_row(
                "SELECT gate_kind, gate_spec FROM scheduled_jobs WHERE id = ?1",
                [id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("the row survives")
    }

    /// A fresh install runs the whole chain; an existing store starts partway along it. Those are
    /// two different paths through the same ledger and they must land in the same place. What this
    /// catches is a migration that behaves differently depending on whether the objects it touches
    /// were created a moment ago or were already there: `gates_become_kind_and_spec` has exactly
    /// such a branch in its `has("gate_command")` early return, and a future step guarded the same
    /// way is the likely place for it to happen again.
    ///
    /// It does **not** catch a column added to `BASELINE_0_42` instead of appended as a new
    /// migration, because the 0.42 fixture below is built from that same constant, so an edit lands
    /// in both paths and they agree. Verified by making that exact edit: this test passed and
    /// `the_released_prefix_is_frozen` failed. That one is the guard for edits; this one is the
    /// guard for divergence.
    #[test]
    fn a_fresh_store_and_an_upgraded_one_have_the_same_schema() {
        let mut built_from_scratch = fresh();
        create_for_test(&mut built_from_scratch).expect("a fresh store reaches head");

        let mut carried_forward = store_as_0_42_left_it();
        let plan = plan(&carried_forward).expect("a 0.42 store is classified");
        assert_eq!(plan.from, 1, "the baseline is version 1, not a fresh store");
        apply(&mut carried_forward, plan, &Context::default()).expect("the remaining steps apply");

        assert_eq!(
            fingerprint(&built_from_scratch),
            fingerprint(&carried_forward),
            "a store built by the whole chain and one carried forward through part of it must end \
             up with the same schema"
        );
    }

    /// Editing a migration that has already shipped changes what new stores get and nothing else,
    /// because the users who ran it will never run it again. Reordering or renaming one does the
    /// same. None of that is visible downstream, so it is checked here rather than hoped for.
    ///
    /// A `Rust` step's body is not covered, and cannot be: it is a function pointer, so only its
    /// name and position are pinned. That matters more than it sounds, because
    /// `gates_become_kind_and_spec` carries five `ADD COLUMN` and two `DROP COLUMN` statements
    /// -- more DDL than anything here covers apart from the baseline. Editing one of those
    /// (a default, say) passes this test *and* the convergence test, since both paths run the
    /// edited step and therefore agree. The conversion tests below pin the gate JSON, not the
    /// DDL. Treat a `Rust` step's body as guarded by review alone.
    #[test]
    fn the_released_prefix_is_frozen() {
        // FNV-1a, written out rather than taken from `DefaultHasher`, whose output Rust explicitly
        // does not promise to keep stable across releases. A test that changes its own expectation
        // when the toolchain moves is not a guard.
        fn digest(input: &str) -> u64 {
            let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
            for byte in input.as_bytes() {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            hash
        }
        /// What has shipped, and only that. A **prefix**, deliberately: appending a migration is
        /// the supported thing to do and must not fail this test, or the repair ritual becomes
        /// "paste the new vector from the failure", which silently re-freezes an edited entry
        /// alongside the appended one and launders exactly what this guards.
        const SHIPPED: &[(&str, u64)] = &[
            ("baseline_0_42", 9890918125805624612_u64),
            ("gates_become_kind_and_spec", 16184223490636562176_u64),
        ];
        let current: Vec<(&str, u64)> = MIGRATIONS
            .iter()
            .map(|migration| {
                let body = match &migration.step {
                    Step::Sql(sql) => digest(sql),
                    // No body to hash. The name still pins its position in the order.
                    Step::Rust(_) | Step::Contextual(_) => 0,
                };
                (migration.name, digest(migration.name) ^ body)
            })
            .collect();
        assert!(
            current.len() >= SHIPPED.len(),
            "a shipped migration was removed; the ledger is append-only"
        );
        assert_eq!(
            &current[..SHIPPED.len()],
            SHIPPED,
            "a shipped migration changed. Append a *new* migration instead of editing one users \
             have already run; do not repair this by pasting the current values over `SHIPPED`"
        );
    }

    /// Rule 2 from the module docs, enforced rather than remembered. A migration that reached for
    /// `Gate::spec` would keep passing every test in the suite and quietly start doing something
    /// else the day those types are refactored.
    ///
    /// `super::` is checked as well as `crate::`, and that is not belt-and-braces. This module is a
    /// child of `session`, so every item in meka is reachable as `super::super::…`; a scan for
    /// `crate::` alone accepts `super::super::schedule::Gate::spec`, which is precisely the call
    /// the module docs single out as forbidden. Verified against the earlier form of this test,
    /// which passed it.
    ///
    /// Split on `mod tests` rather than on the first `#[cfg(test)]` for the same reason: that
    /// attribute also marks test-only helpers, and one placed above a migration would truncate the
    /// scanned region to nothing and pass vacuously.
    #[test]
    fn no_migration_calls_meka_s_own_code() {
        let source = include_str!("migrations.rs");
        // `"\nmod tests {"` rather than `"mod tests {"`: the module is declared at column zero, so
        // this cannot be truncated by a doc comment that happens to contain the literal. The
        // unanchored form could be, and the sanity check below would not have noticed, because it
        // anchors on a function that sits *above* where such a comment would go.
        let production = source
            .split("\nmod tests {")
            .next()
            .expect("splitting always yields a first part");
        assert!(
            production.contains("fn gates_become_kind_and_spec"),
            "the scanned region no longer covers the migrations, so this test proves nothing"
        );
        for line in production.lines() {
            let code = line.trim_start();
            if code.starts_with("//") || !(code.contains("crate::") || code.contains("super::")) {
                continue;
            }
            assert_eq!(
                code, "use crate::error::{MekaError, Result};",
                "a migration may not reach into meka's own code, by any path; see the module docs"
            );
        }
    }

    fn plant_session(connection: &rusqlite::Connection, id: &str) {
        connection
            .execute(
                "INSERT INTO sessions (id, created_at, updated_at) VALUES (?1, 'now', 'now')",
                [id],
            )
            .expect("a session to carry forward");
    }

    fn plant_credential(connection: &rusqlite::Connection, profile: &str) {
        connection
            .execute(
                "INSERT INTO provider_credentials (profile, credentials_json, updated_at) \
                 VALUES (?1, '{}', 'now')",
                [profile],
            )
            .expect("a credential");
    }

    fn provider_of(connection: &rusqlite::Connection, id: &str) -> String {
        connection
            .query_row("SELECT provider FROM sessions WHERE id = ?1", [id], |row| {
                row.get(0)
            })
            .expect("the row survives")
    }

    /// The ordinary upgrade: whatever the caller resolved is what existing sessions adopt.
    #[test]
    fn an_existing_session_adopts_the_resolved_default() {
        let mut connection = store_as_0_42_left_it();
        plant_session(&connection, "carried");
        let plan = plan(&connection).expect("classified");
        apply(&mut connection, plan, &Context::adopting(Some("work"))).expect("migrated");

        assert_eq!(provider_of(&connection, "carried"), "work");
    }

    /// With nothing resolved, one stored credential is the only evidence in the store about which
    /// profile ran these sessions, and it is better than admitting nothing.
    #[test]
    fn a_sole_credential_stands_in_when_no_default_resolves() {
        let mut connection = store_as_0_42_left_it();
        plant_session(&connection, "carried");
        plant_credential(&connection, "the-only-one");
        let plan = plan(&connection).expect("classified");
        apply(&mut connection, plan, &Context::default()).expect("migrated");

        assert_eq!(provider_of(&connection, "carried"), "the-only-one");
    }

    /// Two credentials say nothing about which ran any given session, so nothing is claimed. The
    /// empty value is not special-cased anywhere: it simply does not name a configured profile.
    #[test]
    fn two_credentials_are_not_guessed_between() {
        let mut connection = store_as_0_42_left_it();
        plant_session(&connection, "carried");
        plant_credential(&connection, "one");
        plant_credential(&connection, "two");
        let plan = plan(&connection).expect("classified");
        apply(&mut connection, plan, &Context::default()).expect("migrated");

        assert_eq!(
            provider_of(&connection, "carried"),
            "",
            "an ambiguous store is left saying nothing rather than guessing"
        );
    }

    /// The caller's answer beats the store's, because config is the current statement of intent and
    /// a credential can outlive the profile that used it.
    #[test]
    fn the_resolved_default_beats_a_sole_credential() {
        let mut connection = store_as_0_42_left_it();
        plant_session(&connection, "carried");
        plant_credential(&connection, "stale");
        let plan = plan(&connection).expect("classified");
        apply(&mut connection, plan, &Context::adopting(Some("current"))).expect("migrated");

        assert_eq!(provider_of(&connection, "carried"), "current");
    }

    /// Rule 3. A store that lost its `user_version` replays every step, and this one must not fail
    /// with `duplicate column name` nor overwrite a session that has since moved profile.
    #[test]
    fn a_replay_neither_fails_nor_rewrites_a_chosen_provider() {
        let mut connection = store_as_0_42_left_it();
        plant_session(&connection, "carried");
        let first = plan(&connection).expect("classified");
        apply(&mut connection, first, &Context::adopting(Some("first"))).expect("migrated");

        connection
            .execute(
                "UPDATE sessions SET provider = 'chosen' WHERE id = 'carried'",
                [],
            )
            .expect("the user repins it");
        connection
            .execute_batch("PRAGMA user_version = 0;")
            .expect("the round trip that drops the version");

        let replayed = plan(&connection).expect("classified by shape");
        apply(
            &mut connection,
            replayed,
            &Context::adopting(Some("second")),
        )
        .expect("the replay must not fail");

        assert_eq!(
            provider_of(&connection, "carried"),
            "chosen",
            "a replay only fills what is empty"
        );
    }

    /// The OAuth rows a 0.42 store holds arrive in the new table under kind `oauth`, and the table
    /// they came from is gone. Nothing outside this module may know it ever existed, which is only
    /// true if the copy is complete.
    #[test]
    fn every_oauth_credential_carries_over_and_the_old_table_goes() {
        let mut connection = store_as_0_42_left_it();
        connection
            .execute_batch(
                "INSERT INTO mcp_oauth_credentials (server_name, credentials_json, updated_at) \
                 VALUES ('docs', '{\"access_token\":\"at1\"}', '2026-01-01T00:00:00Z'), \
                        ('api',  '{\"access_token\":\"at2\"}', '2026-01-02T00:00:00Z')",
            )
            .expect("two authorised servers");

        let plan = plan(&connection).expect("classified");
        apply(&mut connection, plan, &Context::adopting(Some("p"))).expect("migrated");

        let mut statement = connection
            .prepare("SELECT server_name, kind, secret, updated_at FROM mcp_credentials ORDER BY 1")
            .expect("the new table exists");
        let rows: Vec<(String, String, String, String)> = statement
            .query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })
            .expect("query")
            .collect::<rusqlite::Result<_>>()
            .expect("rows");
        assert_eq!(
            rows,
            vec![
                (
                    "api".to_string(),
                    "oauth".to_string(),
                    "{\"access_token\":\"at2\"}".to_string(),
                    "2026-01-02T00:00:00Z".to_string(),
                ),
                (
                    "docs".to_string(),
                    "oauth".to_string(),
                    "{\"access_token\":\"at1\"}".to_string(),
                    "2026-01-01T00:00:00Z".to_string(),
                ),
            ],
            "every row carries over, at kind oauth, with its timestamp"
        );

        let old: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master \
                 WHERE type = 'table' AND name = 'mcp_oauth_credentials'",
                [],
                |row| row.get(0),
            )
            .expect("query");
        assert_eq!(old, 0, "the table named for one kind is gone");
    }

    /// Rule 3, for the step that rebuilds a table rather than altering one. A bare `CREATE TABLE`
    /// here would fail with `table … already exists` and refuse the store on every start.
    #[test]
    fn a_replay_neither_fails_nor_discards_a_stored_credential() {
        let mut connection = store_as_0_42_left_it();
        connection
            .execute_batch(
                "INSERT INTO mcp_oauth_credentials (server_name, credentials_json, updated_at) \
                 VALUES ('docs', '{\"access_token\":\"at1\"}', '2026-01-01T00:00:00Z')",
            )
            .expect("one authorised server");
        let first = plan(&connection).expect("classified");
        apply(&mut connection, first, &Context::adopting(Some("first"))).expect("migrated");

        // A secret acquired after the migration, of a kind the old table could not hold.
        connection
            .execute_batch(
                "INSERT INTO mcp_credentials (server_name, kind, secret, updated_at) \
                 VALUES ('docs', 'client_secret', 'cs-not-a-real-secret', '2026-02-01T00:00:00Z')",
            )
            .expect("a client secret beside the bundle");
        connection
            .execute_batch("PRAGMA user_version = 0;")
            .expect("the round trip that drops the version");

        let replayed = plan(&connection).expect("classified by shape");
        apply(
            &mut connection,
            replayed,
            &Context::adopting(Some("second")),
        )
        .expect("the replay must not fail");

        let kinds: Vec<String> = connection
            .prepare("SELECT kind FROM mcp_credentials WHERE server_name = 'docs' ORDER BY 1")
            .expect("prepare")
            .query_map([], |row| row.get(0))
            .expect("query")
            .collect::<rusqlite::Result<_>>()
            .expect("rows");
        assert_eq!(
            kinds,
            vec!["client_secret".to_string(), "oauth".to_string()],
            "a replay leaves both secrets where they were"
        );
    }

    /// Every job survives losing the column, keeps its schedule, and is still readable afterwards.
    ///
    /// The column sat at index 9 of a positionally-decoded row, so dropping it moves four fields
    /// under the reader. Reading the schedule and the fire time back out is what would catch a
    /// half-applied shift: a timestamp read from the wrong column does not parse.
    #[test]
    fn a_job_survives_losing_the_isolated_column() {
        let mut connection = store_as_0_42_left_it();
        connection
            .execute_batch(
                "INSERT INTO sessions (id, created_at, updated_at) \
                 VALUES ('s1', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z');
                 INSERT INTO scheduled_jobs \
                   (id, session_id, kind, spec, prompt, isolated, created_at, next_fire_at) \
                 VALUES ('j-solo', 's1', 'every', '1h', 'check the feed', 1, \
                         '2026-01-01T00:00:00Z', '2026-01-01T01:00:00Z'), \
                        ('j-joined', 's1', 'every', '2h', 'check the other feed', 0, \
                         '2026-01-01T00:00:00Z', '2026-01-01T02:00:00Z')",
            )
            .expect("one job of each kind");

        let planned = plan(&connection).expect("classified");
        apply(&mut connection, planned, &Context::adopting(Some("p"))).expect("migrated");

        let columns: Vec<String> = connection
            .prepare("SELECT name FROM pragma_table_info('scheduled_jobs')")
            .expect("prepare")
            .query_map([], |row| row.get(0))
            .expect("query")
            .collect::<rusqlite::Result<_>>()
            .expect("rows");
        assert!(
            !columns.iter().any(|column| column == "isolated"),
            "the column is gone: {columns:?}"
        );

        let rows: Vec<(String, String, String)> = connection
            .prepare("SELECT id, spec, next_fire_at FROM scheduled_jobs ORDER BY id")
            .expect("prepare")
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .expect("query")
            .collect::<rusqlite::Result<_>>()
            .expect("rows");
        assert_eq!(
            rows,
            vec![
                (
                    "j-joined".to_string(),
                    "2h".to_string(),
                    "2026-01-01T02:00:00Z".to_string()
                ),
                (
                    "j-solo".to_string(),
                    "1h".to_string(),
                    "2026-01-01T01:00:00Z".to_string()
                ),
            ],
            "both jobs keep their schedule and their next fire; the one that was isolated is now \
             an ordinary job on the same session"
        );
    }

    /// Rule 3 for the column drop. A bare `ALTER TABLE … DROP COLUMN` would fail with
    /// `no such column: isolated` on the second pass and refuse the store on every start.
    #[test]
    fn dropping_the_isolated_column_replays_cleanly() {
        let mut connection = store_as_0_42_left_it();
        connection
            .execute_batch(
                "INSERT INTO sessions (id, created_at, updated_at) \
                 VALUES ('s1', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z');
                 INSERT INTO scheduled_jobs \
                   (id, session_id, kind, spec, prompt, isolated, created_at, next_fire_at) \
                 VALUES ('j1', 's1', 'every', '1h', 'watch it', 1, \
                         '2026-01-01T00:00:00Z', '2026-01-01T01:00:00Z')",
            )
            .expect("a job");
        let first = plan(&connection).expect("classified");
        apply(&mut connection, first, &Context::adopting(Some("p"))).expect("migrated");

        connection
            .execute_batch("PRAGMA user_version = 0;")
            .expect("the round trip that drops the version");
        let replayed = plan(&connection).expect("classified by shape");
        apply(&mut connection, replayed, &Context::adopting(Some("p")))
            .expect("the replay must not fail");

        let surviving: i64 = connection
            .query_row("SELECT COUNT(*) FROM scheduled_jobs", [], |row| row.get(0))
            .expect("query");
        assert_eq!(surviving, 1, "and the job is still there");
    }

    /// The override columns replay for the same reason the provider column does, and the failure
    /// mode is worse: a bare `ADD COLUMN` would refuse the store on every start, permanently.
    #[test]
    fn a_replay_neither_fails_nor_clears_a_chosen_model() {
        let mut connection = store_as_0_42_left_it();
        plant_session(&connection, "carried");
        let first = plan(&connection).expect("classified");
        apply(&mut connection, first, &Context::adopting(Some("first"))).expect("migrated");

        connection
            .execute(
                "UPDATE sessions SET model_override = 'pinned' WHERE id = 'carried'",
                [],
            )
            .expect("the user pins a model");
        connection
            .execute_batch("PRAGMA user_version = 0;")
            .expect("the round trip that drops the version");

        let replayed = plan(&connection).expect("classified by shape");
        apply(
            &mut connection,
            replayed,
            &Context::adopting(Some("second")),
        )
        .expect("the replay must not fail");

        let model: Option<String> = connection
            .query_row(
                "SELECT model_override FROM sessions WHERE id = 'carried'",
                [],
                |row| row.get(0),
            )
            .expect("read back");
        assert_eq!(
            model.as_deref(),
            Some("pinned"),
            "a replay adds the columns and converts nothing"
        );
    }

    /// The step speaks exactly when it has something to say, and says which profile it chose.
    ///
    /// Both halves matter and both were wrong. It warned on the literal first run of a fresh
    /// install, where there are no sessions to leave without a provider, which teaches a reader to
    /// skip the message that does matter. And it adopted a profile for every carried-forward
    /// session in silence: for anyone with two accounts that is a guess, and a session that ran on
    /// the other one now names this one and bills here when resumed.
    #[test]
    fn the_step_reports_the_profile_it_adopted_and_only_then() {
        #[derive(Clone)]
        struct Capture(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl std::io::Write for Capture {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .extend_from_slice(buf);
                Ok(buf.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Capture {
            type Writer = Self;

            fn make_writer(&'a self) -> Self::Writer {
                self.clone()
            }
        }

        // INFO rather than WARN, because the adoption notice is a lifecycle signpost and the
        // empty-profile complaint above it is a warning; capturing both keeps one test honest
        // about which level each uses.
        let migrate = |plant: bool, default: Option<&str>| -> String {
            let capture = Capture(std::sync::Arc::new(std::sync::Mutex::new(Vec::new())));
            let buffer = std::sync::Arc::clone(&capture.0);
            let subscriber = tracing_subscriber::fmt()
                .with_writer(capture)
                .with_max_level(tracing::Level::INFO)
                .finish();
            tracing::subscriber::with_default(subscriber, || {
                let mut connection = store_as_0_42_left_it();
                if plant {
                    plant_session(&connection, "carried");
                }
                let plan = plan(&connection).expect("classified");
                apply(&mut connection, plan, &Context::adopting(default)).expect("migrated");
            });
            String::from_utf8_lossy(
                &buffer
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            )
            .into_owned()
        };

        let text = migrate(true, Some("work"));
        assert!(
            text.contains("recorded provider profile 'work'") && text.contains(" 1 session"),
            "the adoption must name the profile and how many rows it touched: {text}"
        );

        // Nothing carried forward: the step still runs and still has nothing to report.
        let text = migrate(false, Some("work"));
        assert!(
            !text.contains("recorded provider profile"),
            "a store with no sessions has nothing to adopt: {text}"
        );
        let text = migrate(false, None);
        assert!(
            !text.contains("left without one"),
            "a fresh install must not be warned about sessions it does not have: {text}"
        );
    }

    #[test]
    fn each_fire_mode_becomes_its_predicate() {
        for (fire, expected) in [("on-change", "changed"), ("on-success", "succeeded")] {
            let mut connection = store_as_0_42_left_it();
            plant_job(&connection, "job", Some("gh pr checks"), Some(fire));
            let plan = plan(&connection).expect("classified");
            apply(&mut connection, plan, &Context::default()).expect("converted");

            let (kind, spec) = gate_of(&connection, "job");
            assert_eq!(kind.as_deref(), Some("shell"));
            let spec: serde_json::Value =
                serde_json::from_str(&spec.expect("a spec")).expect("valid JSON");
            assert_eq!(spec["shell"]["command"], "gh pr checks");
            assert_eq!(spec["when"], expected);
        }
    }

    /// A command is arbitrary user text and has to survive byte for byte. The corpus is the one the
    /// retired Python script used, for the same reason it chose it.
    #[test]
    fn a_command_with_quotes_newlines_and_non_ascii_survives() {
        let awkward = "curl -f 'https://x' # naïve — ünïcødé ✓ 日本語\nsecond \"line\"\\";
        let mut connection = store_as_0_42_left_it();
        plant_job(&connection, "job", Some(awkward), Some("on-change"));
        let plan = plan(&connection).expect("classified");
        apply(&mut connection, plan, &Context::default()).expect("converted");

        let (_, spec) = gate_of(&connection, "job");
        let spec: serde_json::Value =
            serde_json::from_str(&spec.expect("a spec")).expect("valid JSON");
        assert_eq!(spec["shell"]["command"], awkward);
    }

    /// The expensive failure, and the reason the unconvertible row is written the way it is.
    ///
    /// 0.42 refused to load these, so they never fired. Leaving both gate columns null would read
    /// as *no gate at all*, turning a watcher that never fired into a timer that fires every
    /// interval. Setting `gate_kind` without `gate_spec` keeps them refused by the reader's own
    /// corrupt-row rule, which names no version and would say the same of a hand-edited row.
    #[test]
    fn a_gate_that_cannot_be_converted_stays_refused_rather_than_becoming_ungated() {
        let mut connection = store_as_0_42_left_it();
        plant_job(
            &connection,
            "unknown-fire",
            Some("echo hi"),
            Some("on-tuesday"),
        );
        plant_job(&connection, "half-written", Some("echo hi"), None);
        plant_job(&connection, "no-command", None, Some("on-change"));
        let plan = plan(&connection).expect("classified");
        apply(&mut connection, plan, &Context::default()).expect("converted");

        for id in ["unknown-fire", "half-written", "no-command"] {
            let (kind, spec) = gate_of(&connection, id);
            assert_eq!(kind.as_deref(), Some("shell"), "{id} keeps a gate kind");
            assert_eq!(
                spec, None,
                "{id} must stay unreadable, because a null kind *and* spec reads as ungated"
            );
        }
    }

    #[test]
    fn the_retired_columns_are_gone_and_the_lease_columns_are_present() {
        let mut connection = store_as_0_42_left_it();
        let plan = plan(&connection).expect("classified");
        apply(&mut connection, plan, &Context::default()).expect("converted");

        let columns = table_columns(&connection, "scheduled_jobs").expect("columns");
        for gone in ["gate_command", "gate_fire"] {
            assert!(
                !columns.iter().any(|c| c == gone),
                "{gone} should be dropped"
            );
        }
        for present in [
            "gate_kind",
            "gate_spec",
            "claimed_by",
            "claimed_until",
            "attempts",
        ] {
            assert!(
                columns.iter().any(|c| c == present),
                "{present} should exist"
            );
        }
    }

    /// Enforcement is suspended for the steps, so `apply` has to put it back. A connection left
    /// with foreign keys off would go on serving the whole process, silently accepting writes the
    /// schema forbids.
    #[test]
    fn foreign_keys_are_suspended_for_the_steps_and_restored_afterwards() {
        let mut connection = store_as_0_42_left_it();
        connection
            .execute_batch("PRAGMA foreign_keys = ON;")
            .expect("enforcement on, as `SessionManager::open` leaves it");
        let plan = plan(&connection).expect("classified");
        apply(&mut connection, plan, &Context::default()).expect("migrated");

        let enforcing: i64 = connection
            .query_row("PRAGMA foreign_keys", [], |row| row.get(0))
            .expect("readable");
        assert_eq!(
            enforcing, 1,
            "foreign keys must be back on after a migration"
        );
    }

    /// The count is what decides whether a migration is refused, so a constant in its place
    /// disarms the guard while every other test still passes. Mutation-checked: `Ok(0)`, `Ok(1)`
    /// and `Ok(-1)` all survived the suite until this existed, because the comparison that reads it
    /// is `after > before` and any constant makes that false.
    #[test]
    fn dangling_references_are_counted_rather_than_assumed() {
        let mut connection = store_as_0_42_left_it();
        let transaction = connection.transaction().expect("a transaction");
        assert_eq!(
            count_dangling_references(&transaction).expect("count"),
            0,
            "a store with no rows has nothing dangling"
        );
        drop(transaction);

        // Only reachable with enforcement off, which is how these rows come to exist at all.
        connection
            .execute_batch(
                "PRAGMA foreign_keys = OFF;
                 INSERT INTO scheduled_jobs (id, session_id, kind, spec, prompt, created_at, \
                     next_fire_at) \
                 VALUES ('a', 'no-such-session', 'every', '6h', 'p', 'now', 'later');
                 INSERT INTO scheduled_jobs (id, session_id, kind, spec, prompt, created_at, \
                     next_fire_at) \
                 VALUES ('b', 'no-such-session', 'every', '6h', 'p', 'now', 'later');",
            )
            .expect("two orphaned rows");
        let transaction = connection.transaction().expect("a transaction");
        assert_eq!(
            count_dangling_references(&transaction).expect("count"),
            2,
            "both orphans are counted, not just noticed"
        );
    }

    /// The warning is the only signal a user gets that a job was left inert, and the only place the
    /// ids appear. Mutation-checked: deleting the `!` so it fires on the empty case survived the
    /// suite until this existed, because nothing read the log.
    #[test]
    fn only_a_store_with_unreadable_gates_is_warned_about() {
        crate::render::log_capture::start();

        let mut clean = store_as_0_42_left_it();
        plant_job(&clean, "fine", Some("gh pr checks"), Some("on-change"));
        let clean_plan = plan(&clean).expect("classified");
        apply(&mut clean, clean_plan, &Context::default()).expect("converted");
        assert!(
            !crate::render::log_capture::warnings().contains("left inert"),
            "a store whose gates all convert must not be warned about: {}",
            crate::render::log_capture::warnings()
        );

        let mut damaged = store_as_0_42_left_it();
        plant_job(&damaged, "unreadable", Some("echo hi"), Some("on-tuesday"));
        let damaged_plan = plan(&damaged).expect("classified");
        apply(&mut damaged, damaged_plan, &Context::default()).expect("converted");
        let warnings = crate::render::log_capture::warnings();
        assert!(warnings.contains("left inert"), "{warnings}");
        assert!(
            warnings.contains("unreadable"),
            "the warning must name the row, since nothing else will: {warnings}"
        );
    }

    /// The whole point of suspending enforcement, and until [`with_foreign_keys_suspended`] was
    /// split out there was no way to reach it: no shipped migration rebuilds a table, so nothing
    /// could exercise the case the suspension exists for.
    ///
    /// `sessions` is the parent of `messages`, `tool_outputs`, `scheduled_jobs` and
    /// `background_tasks`, every one `ON DELETE CASCADE`. SQLite's documented procedure for the
    /// table changes `ALTER TABLE` cannot express ends in `DROP TABLE sessions`, and with
    /// enforcement on that deletes the entire conversation history inside a transaction that then
    /// commits successfully. Measured before the fix: one child row before, zero after.
    #[test]
    fn a_step_that_rebuilds_a_parent_table_does_not_delete_its_children() {
        let mut connection = store_as_0_42_left_it();
        connection
            .execute_batch(
                "PRAGMA foreign_keys = ON;
                 INSERT INTO sessions (id, created_at, updated_at) VALUES ('s', 'x', 'x');
                 INSERT INTO messages (session_id, role, content, created_at) \
                     VALUES ('s', 'user', 'keep me', 'x');",
            )
            .expect("a session with a message hanging off it");

        // The rebuild `MIGRATIONS` does not contain, handed in the way a future migration would
        // perform it.
        let rebuilt = with_foreign_keys_suspended(&mut connection, |connection| {
            let transaction = connection
                .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
                .expect("a transaction");
            transaction
                .execute_batch(
                    "CREATE TABLE sessions_new (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, \
                         updated_at TEXT NOT NULL);
                     INSERT INTO sessions_new SELECT id, created_at, updated_at FROM sessions;
                     DROP TABLE sessions;
                     ALTER TABLE sessions_new RENAME TO sessions;",
                )
                .expect("the rebuild runs");
            transaction.commit().expect("and commits");
            Ok(())
        });
        rebuilt.expect("the wrapper returns cleanly");

        let survivors: i64 = connection
            .query_row("SELECT count(*) FROM messages", [], |row| row.get(0))
            .expect("count the messages");
        assert_eq!(
            survivors, 1,
            "rebuilding the parent must not cascade; with enforcement on this reads 0 and the \
             conversation is gone"
        );
        let enforcing: i64 = connection
            .query_row("PRAGMA foreign_keys", [], |row| row.get(0))
            .expect("readable");
        assert_eq!(enforcing, 1, "and enforcement is back on afterwards");
    }

    /// A store round-tripped through `sqlite3 .dump` keeps its schema and loses its version, so
    /// every step after the baseline replays over data that already has it. Plain `VACUUM` keeps
    /// the version; `.dump` does not, and that round trip is what people reach for to repair or
    /// move a database.
    ///
    /// Survivable only because `gates_become_kind_and_spec` guards each `ALTER TABLE` and returns
    /// early once `gate_command` is gone. This pins that, so the day a step is written without
    /// those guards it fails here rather than on a stranger's machine, permanently.
    #[test]
    fn a_store_that_lost_its_version_replays_without_damage() {
        let mut connection = fresh();
        create_for_test(&mut connection).expect("a store at head");
        connection
            .execute(
                "INSERT INTO sessions (id, created_at, updated_at) VALUES ('s', 'now', 'now')",
                [],
            )
            .expect("a session");
        connection
            .execute(
                "INSERT INTO scheduled_jobs (id, session_id, kind, spec, prompt, gate_kind, \
                 gate_spec, created_at, next_fire_at) \
                 VALUES ('j', 's', 'every', '6h', 'p', 'shell', ?1, 'now', 'later')",
                [r#"{"shell":{"command":"gh pr checks"},"when":"changed"}"#],
            )
            .expect("an already-converted job");
        let before = fingerprint(&connection);

        // What `.dump` into a fresh database leaves: the schema, the rows, and no version.
        connection
            .execute_batch("PRAGMA user_version = 0;")
            .expect("lose the version");

        let replayed = plan(&connection).expect("classified by shape");
        assert_eq!(
            replayed.from, 1,
            "shape says the baseline is applied, which it is"
        );
        apply(&mut connection, replayed, &Context::default()).expect("the replay must not fail");

        assert_eq!(before, fingerprint(&connection), "the schema is unchanged");
        let (kind, spec) = gate_of(&connection, "j");
        assert_eq!(kind.as_deref(), Some("shell"));
        assert_eq!(
            spec.as_deref(),
            Some(r#"{"shell":{"command":"gh pr checks"},"when":"changed"}"#),
            "the already-converted gate is left alone rather than converted twice"
        );
    }

    /// Damage a store already carries must not block it forever. Enforcement is on for every normal
    /// write, so arranging this takes hand-editing, and refusing the migration would leave such a
    /// user with no way forward at all.
    #[test]
    fn a_dangling_reference_that_predates_the_migration_does_not_block_it() {
        let mut connection = store_as_0_42_left_it();
        // Written with enforcement off, which is the only way this row can exist.
        connection
            .execute_batch(
                "PRAGMA foreign_keys = OFF;
                 INSERT INTO scheduled_jobs (id, session_id, kind, spec, prompt, created_at, \
                     next_fire_at) \
                 VALUES ('orphan', 'no-such-session', 'every', '6h', 'p', 'now', 'later');",
            )
            .expect("an orphaned row");

        let plan = plan(&connection).expect("classified");
        apply(&mut connection, plan, &Context::default())
            .expect("the migration is not blocked by inherited damage");
        assert_eq!(
            user_version(&connection).expect("version"),
            MIGRATIONS.len() as u32
        );
    }

    /// Returning 1 asserts the whole baseline is present. The old code filled a gap silently on
    /// every open; nothing does now, so a store missing a table has to be named rather than
    /// stamped at head with the table still absent.
    #[test]
    fn a_half_built_store_is_named_rather_than_stamped_as_current() {
        let connection = store_as_0_42_left_it();
        connection
            .execute_batch("DROP TABLE background_tasks;")
            .expect("a store an interrupted first run could leave");

        let error = plan(&connection).expect_err("refused");
        let message = error.to_string();
        assert!(message.contains("background_tasks"), "{message}");
        assert!(message.contains("Nothing has been changed"), "{message}");
    }

    #[test]
    fn a_second_run_finds_nothing_to_do() {
        let mut connection = store_as_0_42_left_it();
        plant_job(&connection, "job", Some("echo hi"), Some("on-change"));
        let first = plan(&connection).expect("classified");
        apply(&mut connection, first, &Context::default()).expect("converted");
        let before = fingerprint(&connection);

        let second = plan(&connection).expect("classified again");
        assert!(!second.has_work(), "a converted store has nothing pending");
        apply(&mut connection, second, &Context::default()).expect("a no-op");
        assert_eq!(before, fingerprint(&connection), "nothing moved");
        assert_eq!(
            user_version(&connection).expect("version"),
            MIGRATIONS.len() as u32
        );
    }

    /// The store a hand-run of the retired script left behind: gates already converted, and on an
    /// early build of it, no lease columns. It has to converge here rather than be refused.
    #[test]
    fn a_store_converted_by_hand_still_reaches_head() {
        let mut connection = store_as_0_42_left_it();
        connection
            .execute_batch(
                "ALTER TABLE scheduled_jobs ADD COLUMN gate_kind TEXT;
                 ALTER TABLE scheduled_jobs ADD COLUMN gate_spec TEXT;
                 ALTER TABLE scheduled_jobs DROP COLUMN gate_command;
                 ALTER TABLE scheduled_jobs DROP COLUMN gate_fire;",
            )
            .expect("a hand conversion");
        let plan = plan(&connection).expect("classified");
        apply(&mut connection, plan, &Context::default()).expect("the lease columns still arrive");

        let columns = table_columns(&connection, "scheduled_jobs").expect("columns");
        for present in ["claimed_by", "claimed_until", "attempts"] {
            assert!(
                columns.iter().any(|c| c == present),
                "{present} should exist"
            );
        }
    }

    #[test]
    fn a_0_41_store_is_refused_by_name_rather_than_converted() {
        let connection = fresh();
        connection
            .execute_batch(
                "CREATE TABLE sessions (id TEXT PRIMARY KEY);
                 CREATE TABLE scheduled_jobs (id TEXT PRIMARY KEY, gate_command TEXT, \
                     gate_fire TEXT);",
            )
            .expect("a 0.41 shape");
        let error = plan(&connection).expect_err("0.41 is refused");
        let message = error.to_string();
        assert!(message.contains("migrate-0.41-to-0.42.py"), "{message}");
        assert!(message.contains("Nothing has been changed"), "{message}");
    }

    /// The whole basis for distrusting a stored `1`, pinned so it cannot quietly stop being true.
    /// If the ledger could ever stamp that value, `plan` would be second-guessing a store this
    /// build itself wrote, and the shape probe would run on every start forever.
    #[test]
    fn the_ledger_can_never_stamp_the_retired_flag() {
        assert!(
            MIGRATIONS.len() as u32 > RETIRED_INITIALISED_FLAG,
            "`apply` stamps `MIGRATIONS.len()`, so a ledger of one step would write the same \
             number the retired schema system used as its initialised flag, and nothing could tell \
             the two apart afterwards"
        );
    }

    /// The bug [`RETIRED_INITIALISED_FLAG`] exists for, found by running against a real store.
    ///
    /// Every meka up to 0.41 stamped `user_version = 1` on a store it had finished initialising, so
    /// a 0.41 store in the wild carries it. Taking that at face value skipped the shape probe,
    /// which is where this refusal lives; the gate conversion then ran against a table with no
    /// `gate_permission`, dropped the columns it had read, committed, and stamped head. Reproduced
    /// end to end: every later read failed with `no such column: gate_permission`, and because the
    /// store was stamped current nothing would ever revisit it.
    #[test]
    fn a_0_41_store_carrying_the_retired_stamp_is_still_refused() {
        let connection = fresh();
        connection
            .execute_batch(
                "CREATE TABLE sessions (id TEXT PRIMARY KEY);
                 CREATE TABLE scheduled_jobs (id TEXT PRIMARY KEY, gate_command TEXT, \
                     gate_fire TEXT);
                 PRAGMA user_version = 1;",
            )
            .expect("a 0.41 store as that release left one");
        let error =
            plan(&connection).expect_err("refused, not converted into something unreadable");
        assert!(
            error.to_string().contains("migrate-0.41-to-0.42.py"),
            "{error}"
        );
    }

    /// The other half: distrusting the stamp must not cost a 0.42 store its upgrade. The shape
    /// probe reaches the same answer the stamp claimed, so this one converts normally.
    #[test]
    fn a_0_42_store_carrying_the_retired_stamp_migrates_normally() {
        let mut connection = store_as_0_42_left_it();
        connection
            .execute_batch("PRAGMA user_version = 1;")
            .expect("the stamp every release up to 0.41 left");
        plant_job(&connection, "job", Some("gh pr checks"), Some("on-change"));

        let plan = plan(&connection).expect("classified by shape rather than by the stamp");
        assert_eq!(
            plan.from, 1,
            "the shape says the baseline is applied, and it is"
        );
        apply(&mut connection, plan, &Context::default()).expect("converted");
        let (kind, spec) = gate_of(&connection, "job");
        assert_eq!(kind.as_deref(), Some("shell"));
        assert!(
            spec.is_some(),
            "the gate converted rather than being left unreadable"
        );
    }

    #[test]
    fn a_store_from_a_newer_meka_is_refused_and_left_alone() {
        let mut connection = fresh();
        create_for_test(&mut connection).expect("a fresh store");
        connection
            .execute_batch("PRAGMA user_version = 99;")
            .expect("a version from the future");
        let before = fingerprint(&connection);

        let error = plan(&connection).expect_err("a newer store is refused");
        assert!(error.to_string().contains("newer release"), "{error}");
        assert_eq!(before, fingerprint(&connection), "nothing was touched");
        assert_eq!(user_version(&connection).expect("version"), 99);
    }

    /// A file meka did not write is not a store to carry forward. Every release before this one
    /// created its tables alongside whatever was already there, and so does this.
    #[test]
    fn a_database_that_is_not_a_meka_store_is_built_rather_than_refused() {
        let mut connection = fresh();
        connection
            .execute_batch("CREATE TABLE placeholder (id INTEGER);")
            .expect("an unrelated table");
        let plan = plan(&connection).expect("classified as new");
        assert_eq!(
            plan.from, 0,
            "there is nothing of meka's here to carry forward"
        );
        apply(&mut connection, plan, &Context::default()).expect("the schema is built");
        assert!(
            !table_columns(&connection, "sessions")
                .expect("columns")
                .is_empty()
        );
    }

    /// An unreadable `config.toml` must not be turned into "no profile" and stamped.
    ///
    /// The adopt step runs once and `user_version` moves with it, so a value derived from a parse
    /// error is written to every carried-forward session and never revisited: one typo, and 476
    /// sessions permanently record no profile with nothing said. "Nothing resolved" and "I could
    /// not read the file" are different answers and only the first may be adopted.
    ///
    /// The refusal is conditional on there being rows to stamp, which is what keeps `meka mcp
    /// remove` and `meka provider remove` usable: those edit the raw document through `toml_edit`
    /// and are how a user repairs such a config, so a store with nothing to migrate must still
    /// open.
    #[test]
    fn an_unreadable_config_refuses_to_stamp_carried_sessions_but_not_an_empty_store() {
        let mut carrying = store_as_0_42_left_it();
        carrying
            .execute_batch(
                "INSERT INTO sessions (id, created_at, updated_at) VALUES \
                 ('11111111-1111-4111-8111-111111111111', '2020-01-01', '2020-01-01')",
            )
            .expect("a session that predates meka recording a provider");
        let before = fingerprint(&carrying);

        let carrying_plan = plan(&carrying).expect("classified");
        assert!(carrying_plan.has_work());
        let error = apply(
            &mut carrying,
            carrying_plan,
            &Context::on_unreadable_config(),
        )
        .expect_err("a store with rows to stamp must refuse");
        assert!(
            error.to_string().contains("config.toml"),
            "the refusal must name what has to be fixed: {error}"
        );
        assert_eq!(
            before,
            fingerprint(&carrying),
            "the store must be exactly as it was, so a later run can migrate it correctly"
        );
        assert_eq!(
            user_version(&carrying).expect("version"),
            0,
            "and the version must not move, or the step never runs again"
        );

        // Nothing to stamp, so nothing to get wrong: the repair commands keep working.
        let mut empty = store_as_0_42_left_it();
        let empty_plan = plan(&empty).expect("classified");
        apply(&mut empty, empty_plan, &Context::on_unreadable_config())
            .expect("a store with no carried sessions opens on an unreadable config");
        assert_eq!(
            user_version(&empty).expect("version"),
            MIGRATIONS.len() as u32,
            "and reaches head"
        );
    }

    /// One transaction, so a step that fails leaves the version and the tables where they were.
    /// Without this the store can end up carrying half a migration with nothing recording that.
    #[test]
    fn a_failing_step_leaves_the_store_untouched() {
        let mut connection = store_as_0_42_left_it();
        plant_job(&connection, "job", Some("gh pr checks"), Some("on-change"));
        // A store that classifies cleanly as 0.42 but that the gate conversion cannot finish: the
        // trigger aborts the `UPDATE` that writes the converted spec, after the step has already
        // added five columns. That partial state is exactly what the single transaction has to
        // undo, so forcing the failure *mid-step* is the point rather than an accident of setup.
        connection
            .execute_batch(
                "CREATE TRIGGER refuse_the_conversion BEFORE UPDATE ON scheduled_jobs \
                 BEGIN SELECT RAISE(ABORT, 'this store will not take the update'); END;",
            )
            .expect("a store the next step cannot get through");
        let before = fingerprint(&connection);

        let plan = plan(&connection).expect("classified");
        assert!(plan.has_work());
        let error = apply(&mut connection, plan, &Context::default()).expect_err("the step fails");
        assert!(
            error.to_string().contains("The store is unchanged"),
            "{error}"
        );
        assert_eq!(
            before,
            fingerprint(&connection),
            "the tables are as they were"
        );
        assert_eq!(
            user_version(&connection).expect("version"),
            0,
            "the version must not move without the schema it describes"
        );
        assert!(
            table_columns(&connection, "scheduled_jobs")
                .expect("columns")
                .iter()
                .any(|column| column == "gate_command"),
            "the columns the failed step meant to drop are still there"
        );
    }
}
