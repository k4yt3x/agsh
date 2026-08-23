//! The memory store: the `memories` table, its FTS5 index, and the ranked retrieval over both.
//!
//! **This is the source of truth.** Memories used to be Markdown files with this database as a
//! derived mirror, and keeping two mutable copies in step is where the subsystem's defects lived:
//! mtime precision, a dirty set that reached discovery but not the index, a sync-versus-publish
//! race, usage rows pruned against the wrong set, and an entire apparatus for reporting files that
//! would not parse. One copy removes all of it. See `docs/book/src/usage/memory.md` for the
//! user-facing consequence, which is that `meka memory export` replaces `grep`.
//!
//! # What SQLite is doing that this module no longer has to
//!
//! - **The FTS index is external-content** (`content='memories'`), so three triggers keep it in
//!   step and there is no sync function to get wrong. It stays derived and disposable even though
//!   the content no longer is: `INSERT INTO memories_fts(memories_fts) VALUES('rebuild')` repairs
//!   it from the table at any time.
//! - **`name` is `UNIQUE COLLATE NOCASE`**, which is the case-collision check.
//! - **Omit-to-keep is in the upsert**, not in a Rust branch. See [`MemoryStore::write`].
//! - **A transaction is the read-modify-write lock**, across processes, so no write door needs an
//!   `flock`. The one in-process mutex that remains is [`MemoryStore::lock_duplicate_check`], which
//!   guards an advisory read-then-report, not the write itself.
//!
//! # Three traps
//!
//! **`bm25()` returns a negative number, and more negative is better.** [`Ranking::score`] negates
//! before multiplying by the importance weights; without that the multiplication orders the results
//! backwards, and it does so silently.
//!
//! **A model-supplied query is not FTS5 syntax.** `*`, `"`, `NEAR` and `OR` are all operators, and
//! a query containing them would either error or quietly mean something else. [`Terms`] tokenizes
//! on the Rust side and re-emits each token as a quoted string literal, so the model's words are
//! only ever words.
//!
//! **Never `INSERT OR REPLACE` into `memories`, and never `UPDATE` its `id`.** Both are a
//! delete-then-insert that can change the rowid, which unanchors every row of the external-content
//! index. `ON CONFLICT DO UPDATE` is the only upsert shape allowed here. A column added to
//! `memories` and left out of the three triggers desyncs the index just as silently, and FTS5's
//! own `integrity-check` will *not* tell you: on an external-content table it verifies the index's
//! internal structure and never compares it to the content. [`MemoryStore::integrity_check`]
//! therefore counts documents as well, [`repair_a_desynced_index`] runs that same comparison at
//! every open and repairs what it finds, `meka memory verify` is how a user runs it by hand, and
//! `rebuild_index` is the only real repair.
//!
//! # A fourth, learned the hard way
//!
//! **`sqlite_master.sql` is not the text you submitted.** SQLite strips `IF NOT EXISTS` and drops
//! the trailing `;`, so a build that compares its own literal against the stored one to decide
//! whether the schema has drifted concludes "drifted" every single time. This module did exactly
//! that, and the cost was a full index rebuild on every process start, hidden behind a doc comment
//! asserting the opposite. See [`canonical_trigger_sql`]. The general lesson is the one the header
//! above keeps repeating in other forms: a property nothing tests is a property nothing holds.

use std::{
    sync::Arc,
    time::{Duration, SystemTime},
};

use rusqlite::OptionalExtension;
use tokio_rusqlite::Connection;

use crate::{
    error::{MekaError, Result},
    memory::Memory,
    store::{DEFAULT_PRIORITY, MAX_PRIORITY},
};

/// Column indices in `memories_fts`, in declaration order. Named because `snippet()` and `bm25()`
/// take positional column arguments and a bare `3` at a call site is unreadable and unsearchable.
const COLUMN_BODY: i64 = 3;

/// `bm25` weights, one per indexed column: name, description, tags, body.
///
/// A hit on the *name* is close to an exact-recall event -- the model half-remembered what it
/// called the note -- so it outranks everything. The description is the one line the model wrote to
/// stand on its own in the index, which makes it a better signal than prose buried in a body.
const WEIGHT_NAME: f64 = 10.0;
const WEIGHT_DESCRIPTION: f64 = 3.0;
const WEIGHT_TAGS: f64 = 2.0;
const WEIGHT_BODY: f64 = 1.0;

/// How many BM25-ranked candidates are pulled back before the importance weights re-order them.
///
/// Two-stage retrieval: relevance alone picks the pool, then [`Ranking::score`] applies priority,
/// usage and freshness within it. The weights span roughly 0.5x to 6x, which reorders neighbours
/// but will not lift the thousandth-best match to the top, so a pool this size costs a little work
/// and buys the whole practical effect. Anything dropped here was dropped for having no textual
/// relevance, which is the one criterion the model's query actually stated.
const CANDIDATE_POOL: usize = 200;

/// Priority at or below which a memory is a standing directive and does not decay. Read with
/// [`FRESHNESS_HALF_LIFE_DAYS`].
const STANDING_PRIORITY_MAX: u8 = 1;

/// How far the declared priority can move a score, as a multiplier span above 1.0. A priority-0
/// memory is worth `1.0 + PRIORITY_WEIGHT_SPAN` times a priority-9 one at equal relevance.
const PRIORITY_WEIGHT_SPAN: f64 = 2.0;

/// Read count at which the usage bonus saturates. Logarithmic, so the first few recalls move the
/// needle and the hundredth does not.
const USAGE_SATURATION_READS: f64 = 20.0;

/// Half-life of a non-standing memory's freshness, in days.
///
/// Deliberately long, and deliberately *not applied to standing directives at all*. A two-year-old
/// priority-0 rule is exactly as binding as a new one, while a two-year-old priority-8 note
/// probably is not, so decay is keyed on what the memory claims to be rather than applied flat.
/// This is the one place the design departs from the usual recency term, and it is the difference
/// between a memory store and a feed.
const FRESHNESS_HALF_LIFE_DAYS: f64 = 365.0;

/// Floor on the freshness multiplier. An old memory is demoted, never buried: it is still the
/// answer when nothing newer matches.
const FRESHNESS_FLOOR: f64 = 0.5;

/// How many characters the prefix tier may trim from a query term on its second attempt, and the
/// shortest a term may be trimmed to. See [`Terms::trimmed_prefix_match_expressions`].
const PREFIX_TRIM_CHARS: usize = 5;
const PREFIX_MIN_CHARS: usize = 4;

/// Distinct terms one search may carry.
///
/// The substring tier builds one `OR` clause per term against four columns, and SQLite caps an
/// expression tree at depth 1000; the full-text tiers OR the same terms into one `MATCH`. Far more
/// than any real question needs -- the tool asks for synonyms, not for a document -- and low enough
/// that pasting five paragraphs in as a "query" degrades to searching its first hundred words
/// rather than erroring.
const MAX_TERMS: usize = 100;

/// How much of a body [`MemoryStore::substring_search`] carries as its excerpt. FTS5's `snippet()`
/// is only available under a `MATCH`, and this scan is what runs when there is none.
const SUBSTRING_SNIPPET_CHARS: usize = 160;

/// A window of `chars` characters from `body` containing the first of `terms` that appears in it.
///
/// The renderer presents this under a preamble saying the text contains the search term, so an
/// excerpt that does not is a statement the tool makes and the body contradicts. Taking the body's
/// opening satisfied that only when the match happened to be near the start, and reliably failed
/// for the case this tier exists for: a long CJK body, which `unicode61` tokenises as a single
/// token so `MATCH` cannot reach it, where the match sits wherever the author put it.
///
/// A quarter of the window leads the match, so what comes back reads as an excerpt rather than as
/// a fragment starting mid-word. When no term is in the body -- the row matched on its name,
/// description or tags -- the opening is the honest answer and there is nothing to centre on.
///
/// Case-insensitively, and specifically *ASCII* case-insensitively, because that is what SQLite's
/// `LIKE` did to select this row. Anything else would look for a match the query never made.
fn excerpt_around_a_match(body: &str, terms: &[String], chars: usize) -> String {
    // The first term that hits, not the earliest hit among all of them: both satisfy the contract,
    // and only one of them stops scanning.
    //
    // Every term is tried, with no cap. A cap was here briefly and was a mistake: `Terms::parse`
    // preserves query order across all `queries` entries, so three phrasings ending in the one
    // non-ASCII word put that word past any small ceiling -- and a row whose body is entirely CJK
    // reaches this tier precisely because no full-text tier can see it, matches on `LIKE`, and
    // then got an excerpt from the body's opening under a preamble asserting it contained the
    // term. The cost the cap was buying back is real but bounded by `MAX_TERMS`, and it is the
    // wrong thing to spend a false statement on.
    let found = terms
        .iter()
        .find_map(|term| find_ignoring_ascii_case(body, term));
    let lead = chars / 4;
    let start = match found {
        Some(offset) => body[..offset].chars().count().saturating_sub(lead),
        None => 0,
    };
    body.chars().skip(start).take(chars).collect()
}

/// Byte offset of the first ASCII-case-insensitive occurrence of `needle`, at a character
/// boundary.
///
/// Byte-wise rather than lowercasing the haystack, because the haystack is a whole memory body and
/// this runs once per candidate row: a store of 200 KB notes would otherwise allocate a second copy
/// of each one to answer a question about a 160-character window. The boundary check is what keeps
/// the offset sliceable -- an ASCII needle cannot match inside a multi-byte character, but a
/// non-ASCII one can start part-way through.
fn find_ignoring_ascii_case(haystack: &str, needle: &str) -> Option<usize> {
    let (hay, need) = (haystack.as_bytes(), needle.as_bytes());
    if need.is_empty() || need.len() > hay.len() {
        return None;
    }
    (0..=hay.len() - need.len()).find(|&start| {
        haystack.is_char_boundary(start)
            && hay[start..start + need.len()].eq_ignore_ascii_case(need)
    })
}

/// Create the memory tables. Idempotent, and invoked from
/// [`crate::session::SessionManager::initialize_schema`] so schema ownership stays in one place.
///
/// Not creation alone. The `CREATE`s are followed by [`sync_triggers`], which replaces the FTS
/// triggers and rebuilds the index when the stored definitions differ from this build's, and then
/// by [`repair_a_desynced_index`] -- unless that rebuild just ran, in which case the probe would
/// only re-ask a question one pass of the table has already answered. Both reconcile a derived
/// index against the table it is derived from, which is why they sit on the open path instead of in
/// a command the user has to remember to run. Nothing beyond the objects named here is touched.
pub(crate) fn create_tables(connection: &rusqlite::Connection) -> rusqlite::Result<()> {
    connection.execute_batch(
        // `id` rather than an implicit rowid because the FTS index anchors to it by name, and an
        // anchor you cannot see is one a later `ALTER` can move.
        "CREATE TABLE IF NOT EXISTS memories (
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

         -- Serves the one ordering the `[Memory]` index renders in, so the per-turn read is a
         -- scan of an index rather than a sort of the table.
         CREATE INDEX IF NOT EXISTS memories_rank
             ON memories(priority, recorded_at DESC);

         CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
             name,
             description,
             tags,
             body,
             content = 'memories',
             content_rowid = 'id',
             tokenize = 'porter unicode61'
         );",
    )?;
    // The triggers are created by [`sync_triggers`] and nowhere else.
    //
    // A `CREATE TRIGGER IF NOT EXISTS` pass here first looked harmless and was the opposite: a
    // trigger that had gone *missing*, however it went missing, was silently put back before
    // `sync_triggers` read `sqlite_master`, so the comparison matched, no rebuild ran, and every
    // write that landed while it was absent stayed out of the index for good.
    // `repair_a_desynced_index` cannot see that either, because a missed *update* leaves the
    // document counts equal. Reproduced end to end: an edit made with `memories_au` dropped was
    // still unfindable after a normal restart, with `meka memory verify` reporting the store sound.
    //
    // Letting `sync_triggers` own creation costs nothing on a fresh database -- it finds none of
    // the three, takes the replacement path, and creates all three plus a rebuild of an empty table
    // inside one transaction, which is better than three autocommit statements anyway.
    //
    // Skipped when that swap rebuilt: the index is then one pass old, and probing it would only ask
    // a question just answered.
    if sync_triggers(connection)? {
        return Ok(());
    }
    repair_a_desynced_index(connection)
}

/// The three triggers that uphold the external-content contract, and the single place their text
/// lives so [`sync_triggers`] compares against exactly what [`create_tables`] writes.
///
/// Every mutation of `memories` must be mirrored, and a delete must be told the *old* values so
/// FTS can find the posting list to remove. A column added to the table and not added to all three
/// of these desyncs the index in silence.
const TRIGGER_DEFINITIONS: [(&str, &str); 3] = [
    (
        "memories_ai",
        "CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
             INSERT INTO memories_fts(rowid, name, description, tags, body)
             VALUES (new.id, new.name, new.description, new.tags, new.body);
         END;",
    ),
    (
        "memories_ad",
        "CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
             INSERT INTO memories_fts(memories_fts, rowid, name, description, tags, body)
             VALUES ('delete', old.id, old.name, old.description, old.tags, old.body);
         END;",
    ),
    (
        // Gated on the indexed columns actually changing. `record_read` bumps `read_count` on
        // every `memory_read`, and an ungated trigger answered that by deleting and re-inserting
        // all four columns -- two full posting-list rewrites of a body that had not changed, on
        // the one operation a large memory performs most often.
        //
        // Every indexed column has to appear in the `WHEN` or an edit to the one left out stops
        // reaching the index. `id` is deliberately absent: it is the `content_rowid`, and the
        // module header forbids updating it at all.
        //
        // `name` is compared `COLLATE BINARY` because `IS NOT` otherwise inherits the column's
        // `NOCASE` and a case-only rename would not fire. That is defence in depth rather than a
        // live guarantee, and worth saying so: `unicode61` folds case when it tokenises, so
        // `Policy` and `policy` produce identical index entries and skipping the re-index for a
        // case-only change is currently unobservable through search or `integrity-check`. It stays
        // because the comparison should follow the *content*, not the tokenizer's opinion of it --
        // a future tokenizer change, or a case-sensitive column added beside this one, would make
        // the difference real, and nothing would fail in between.
        "memories_au",
        "CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories
         WHEN old.name COLLATE BINARY IS NOT new.name COLLATE BINARY
           OR old.description IS NOT new.description
           OR old.tags        IS NOT new.tags
           OR old.body        IS NOT new.body
         BEGIN
             INSERT INTO memories_fts(memories_fts, rowid, name, description, tags, body)
             VALUES ('delete', old.id, old.name, old.description, old.tags, old.body);
             INSERT INTO memories_fts(rowid, name, description, tags, body)
             VALUES (new.id, new.name, new.description, new.tags, new.body);
         END;",
    ),
];

/// One trigger definition reduced to the form SQLite will hand back, so this build's text and the
/// stored text can be compared at all.
///
/// **`sqlite_master.sql` is not the statement as submitted.** SQLite removes `IF NOT EXISTS` and
/// drops the trailing `;`, so comparing a [`TRIGGER_DEFINITIONS`] literal against it directly never
/// matched -- not on a stale store, not on a fresh one, in no database state whatsoever. That made
/// [`sync_triggers`]'s early return dead code and turned every single process open into three
/// dropped triggers, three recreated ones and a full `'rebuild'` of the index. Measured on a
/// 20,000-memory store: 306 ms for `meka memory get <name>`, a point lookup, against 3 ms for a
/// command that never opens the store. Two independent reviews found it the same way, which is
/// what a claim with no test behind it is worth.
fn canonical_trigger_sql(sql: &str) -> String {
    // Whitespace is collapsed as well: this file's indentation is not something a definition should
    // be considered stale over.
    sql.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace("CREATE TRIGGER IF NOT EXISTS ", "CREATE TRIGGER ")
        .trim_end_matches(';')
        .trim_end()
        .to_string()
}

/// Replace the FTS triggers when a previous build left different ones behind. Reports whether it
/// rebuilt the index, so the caller can skip a probe that just became redundant.
///
/// The reconciliation `CREATE TRIGGER IF NOT EXISTS` cannot do, without the window that dropping
/// them unconditionally opens. An earlier attempt did drop and recreate all three on every open,
/// which was worse than the problem: the six DDL statements are six auto-commit transactions, so
/// every other connection saw the table with triggers missing between them and a write landing in
/// a gap never reached the index. Measured at 4 lost inserts in 120 concurrent writes, 12 lost
/// updates in 360, and 16 lost through a live `meka serve` -- and the lost-update class is the one
/// [`MemoryStore::integrity_check`] cannot see. A `SIGKILL` mid-sequence left the triggers gone
/// permanently, which `IF NOT EXISTS` alone can never do.
///
/// Three properties, all load-bearing:
///
/// - **It compares first and writes nothing when the definitions already match**, which is every
///   open after the first. A store that is already correct is never briefly wrong, and a process
///   start costs one `sqlite_master` read rather than six DDL transactions. Comparing needs
///   [`canonical_trigger_sql`], because the stored text is not the submitted text; without it this
///   property reads as true and is false on every run.
/// - **The replacement is one transaction.** SQLite's DDL is transactional, so a concurrent
///   connection sees the old set or the new set and never a partial one, and a crash rolls back.
/// - **The rebuild is inside that transaction.** It was a separate statement after the commit, and
///   a crash in between left triggers that matched over an index built by the *old* ones -- so
///   every later open took the early return, the rebuild never ran, and search answered from
///   superseded text for good. Nothing detects that: the document counts still agree, FTS5's
///   `integrity-check` passes, and `meka memory verify` reports the store sound. Reproduced.
///
/// The `sqlite_master` read sits outside the transaction, which is safe because
/// [`crate::session::SessionManager::initialize_schema`] holds an exclusive OS file lock across all
/// schema work: no second meka process can be between this check and its replacement.
fn sync_triggers(connection: &rusqlite::Connection) -> rusqlite::Result<bool> {
    let existing: std::collections::HashMap<String, String> = {
        let mut statement = connection.prepare(
            "SELECT name, sql FROM sqlite_master
             WHERE type = 'trigger' AND name IN ('memories_ai', 'memories_ad', 'memories_au')",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                canonical_trigger_sql(&row.get::<_, String>(1)?),
            ))
        })?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    if TRIGGER_DEFINITIONS
        .iter()
        .all(|(name, sql)| existing.get(*name) == Some(&canonical_trigger_sql(sql)))
    {
        return Ok(false);
    }

    tracing::info!("replacing the memory search index triggers with this build's definitions");
    // `Immediate`, matching [`MemoryStore::write`] and for the same reason it gives: a deferred
    // transaction takes its write lock on the first write rather than at `BEGIN`, and under WAL
    // that upgrade can return `SQLITE_BUSY` without consulting the busy handler at all. This one
    // drops three triggers and rebuilds the whole index, so the process it would lose to is a
    // `meka serve` mid-write -- not another process's schema work, which the schema `flock`
    // already excludes. Rare path, clean rollback, and no reason to differ from its sibling.
    let transaction =
        rusqlite::Transaction::new_unchecked(connection, rusqlite::TransactionBehavior::Immediate)?;
    for (name, _) in TRIGGER_DEFINITIONS {
        transaction.execute_batch(&format!("DROP TRIGGER IF EXISTS {name};"))?;
    }
    for (_, sql) in TRIGGER_DEFINITIONS {
        transaction.execute_batch(sql)?;
    }
    // The index cannot be trusted once a definition has changed: whatever the old triggers did, or
    // failed to do, is already in it. A rebuild is one pass over the table and the only thing that
    // makes the new definitions true of the rows already there -- and it commits with them, or not
    // at all.
    transaction.execute_batch("INSERT INTO memories_fts(memories_fts) VALUES('rebuild');")?;
    transaction.commit()?;
    Ok(true)
}

/// How many documents the table holds and how many the index believes it holds, read together.
///
/// One statement, so both counts come from one snapshot. As two `query_row` calls in autocommit
/// mode they straddled any concurrent commit, which is a spurious disagreement on a healthy store
/// -- and this number decides whether to rebuild.
fn document_counts(connection: &rusqlite::Connection) -> rusqlite::Result<(i64, i64)> {
    connection.query_row(
        // `memories_fts_docsize` holds one row per *indexed* document, so it counts what the index
        // believes; `memories` is what is true.
        "SELECT (SELECT count(*) FROM memories), (SELECT count(*) FROM memories_fts_docsize)",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )
}

/// Rebuild the index at open when it does not hold one document per stored memory.
///
/// The index is derived and disposable, so a disagreement has exactly one correct response and no
/// reason to wait for a human to run `meka memory verify`. This catches the case [`sync_triggers`]
/// cannot: a `memories_fts` dropped and recreated empty by a partial restore, which is otherwise
/// silent and permanent -- search simply stops finding things.
///
/// Cheap enough to run unconditionally: measured at 20,000 memories, the two counts are 0.27 ms and
/// 0.11 ms, against 280 ms for the rebuild that only a genuinely broken store pays.
///
/// It cannot see a *changed* document whose update trigger did not fire, because the counts still
/// agree. Nothing can, short of rebuilding; see [`MemoryStore::integrity_check`].
fn repair_a_desynced_index(connection: &rusqlite::Connection) -> rusqlite::Result<()> {
    let (stored, indexed) = document_counts(connection)?;
    if stored == indexed {
        return Ok(());
    }
    tracing::warn!(
        "the memory search index holds {} documents but the store holds {}; rebuilding it from \
         the store. No memory is lost: the index is derived.",
        indexed,
        stored
    );
    connection.execute_batch("INSERT INTO memories_fts(memories_fts) VALUES('rebuild');")
}

/// A model-supplied query, reduced to tokens that cannot be FTS5 syntax.
///
/// Splitting on anything non-alphanumeric already strips every operator character; quoting each
/// token on the way back out is what stops a *word* like `or` or `near` being read as one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Terms(Vec<String>);

impl Terms {
    /// Tokenize one or more phrasings into a single term set. Duplicate tokens collapse, so
    /// supplying "terse", "terseness" and "be terse" does not triple-count the shared word.
    ///
    /// Capped at [`MAX_TERMS`]. The tool tells the model that supplying synonyms costs nothing,
    /// which is true of a handful and not of a thousand: the substring tier builds one `OR` clause
    /// per term, and a few pasted paragraphs pushed the expression past SQLite's depth limit so the
    /// model got a raw `Expression tree is too large (maximum depth 1000)` back from what it had
    /// been encouraged to do.
    pub(crate) fn parse(queries: &[String]) -> Self {
        let mut terms: Vec<String> = Vec::new();
        for query in queries {
            for token in query.split(|character: char| {
                !character.is_alphanumeric() && character != '_' && character != '\''
            }) {
                if terms.len() >= MAX_TERMS {
                    return Self(terms);
                }
                let token = token.trim_matches('\'');
                if token.is_empty() {
                    continue;
                }
                let lowered = token.to_lowercase();
                if !terms.contains(&lowered) {
                    terms.push(lowered);
                }
            }
        }
        Self(terms)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The `MATCH` expression for exact (stemmed) matching: every term OR'd.
    ///
    /// OR rather than AND because the caller is encouraged to pass several phrasings of the same
    /// question, and requiring all of them would return nothing whenever one guess was wrong.
    /// Ranking is what separates a document matching four terms from one matching one.
    pub(crate) fn match_expression(&self) -> String {
        self.render(false)
    }

    /// The same, with each term prefix-expanded (`"pref"*`), for the typo-and-truncation retry.
    pub(crate) fn prefix_match_expression(&self) -> String {
        self.render(true)
    }

    /// Prefix expressions with each term shortened by one more character each time, longest first.
    ///
    /// The prefix tier only ever worked in one direction: `Tok*` reaches `Tokyo` because the stored
    /// word is the longer one. The symmetric case is just as ordinary and nothing covered it -- a
    /// live model searched `deployment` against a memory tagged `deploy` whose body says `Deploys`,
    /// and got "No memories matched". Every tier missed it: SQLite's porter strips `-s`, `-ing` and
    /// `-ed` but not `-ment`, `deployment*` does not match `deploy` because the star is on the
    /// wrong end, `LIKE '%deployment%'` cannot match a shorter word, and the edit distance is 4
    /// against a threshold of 3. The model said it did not believe the empty result; it was right.
    ///
    /// Progressive rather than one fixed cut, because the ending's length is not fixed either:
    /// `deployment` needs four characters off and `deployments` five. Longest prefix first, so the
    /// most specific query that can answer is the one that does. Bounded by [`PREFIX_TRIM_CHARS`],
    /// past which a prefix stops being a near-miss and starts being a different word, and floored
    /// at [`PREFIX_MIN_CHARS`] so a short term is never reduced to noise.
    pub(crate) fn trimmed_prefix_match_expressions(&self) -> Vec<String> {
        let mut expressions = Vec::new();
        for trim in 1..=PREFIX_TRIM_CHARS {
            let trimmed: Vec<String> = self
                .0
                .iter()
                .map(|term| {
                    let keep = term
                        .chars()
                        .count()
                        .saturating_sub(trim)
                        .max(PREFIX_MIN_CHARS);
                    term.chars().take(keep).collect()
                })
                .collect();
            // Every term already at the floor: shortening further changes nothing, and repeating
            // the same query is a round trip that cannot answer differently.
            if trimmed == self.0
                || expressions
                    .last()
                    .is_some_and(|last| *last == Terms(trimmed.clone()).render(true))
            {
                continue;
            }
            expressions.push(Terms(trimmed).render(true));
        }
        expressions
    }

    fn render(&self, prefix: bool) -> String {
        self.0
            .iter()
            .map(|term| {
                // FTS5 escapes a double quote inside a string literal by doubling it. Tokens
                // cannot contain one after `parse`, but the escape stays: the invariant lives in
                // another function, and a future tokenizer change should not become an injection.
                let quoted = format!("\"{}\"", term.replace('"', "\"\""));
                if prefix { format!("{quoted}*") } else { quoted }
            })
            .collect::<Vec<_>>()
            .join(" OR ")
    }

    /// The terms as plain words, for the edit-distance fallback that runs when FTS finds nothing.
    pub(crate) fn words(&self) -> &[String] {
        &self.0
    }
}

/// One search result, before the caller decides how much of it to render.
#[derive(Debug, Clone)]
pub struct Hit {
    pub name: String,
    pub description: String,
    pub body: String,
    /// An excerpt of the body around the match, or its opening when the match was in the
    /// description or the name.
    pub snippet: String,
    pub priority: u8,
    pub recorded: SystemTime,
    pub read_count: u32,
    /// The composed [`Ranking::score`]. Higher is better, unlike the raw `bm25` it derives from.
    pub score: f64,
}

/// What one search found: the ranked hits the caller asked for, and how many there were before the
/// cut.
///
/// `matched` is separate from `hits.len()` so a truncated result can say so. It is itself bounded
/// by [`CANDIDATE_POOL`], which `pool_exhausted` reports: at that point "200" is the size of the
/// window rather than a count of the store, and stating it as a total would be a second, quieter
/// version of the same lie.
#[derive(Debug, Default)]
pub struct SearchResults {
    pub hits: Vec<Hit>,
    pub matched: usize,
    pub pool_exhausted: bool,
}

/// The three multipliers that turn textual relevance into "which of these did you actually mean".
///
/// Split out from the query so each factor is separately testable and so the weights live in one
/// readable place rather than inside a SQL string.
pub(crate) struct Ranking;

impl Ranking {
    /// `relevance x priority x usage x freshness`, all of them positive and increasing-is-better.
    ///
    /// `bm25` is negated on the way in. FTS5 returns a *negative* number whose magnitude grows with
    /// relevance, so multiplying it by a positive weight ranks the least relevant result first, and
    /// nothing about the output looks wrong.
    pub(crate) fn score(bm25: f64, priority: u8, read_count: u32, age: Duration) -> f64 {
        let relevance = -bm25;
        relevance
            * Self::priority_weight(priority)
            * Self::usage_weight(read_count)
            * Self::freshness(priority, age)
    }

    /// What the agent said this was worth when it wrote it. Spans `1.0 ..= 1.0 + span`.
    fn priority_weight(priority: u8) -> f64 {
        let inverted = f64::from(MAX_PRIORITY.saturating_sub(priority)) / f64::from(MAX_PRIORITY);
        1.0 + PRIORITY_WEIGHT_SPAN * inverted
    }

    /// What the agent has since *done* with it. Spans `1.0 ..= 2.0`.
    ///
    /// The counterweight to a priority chosen once and never revised. `meka memory list`'s
    /// distribution exists because priorities drift toward 0 over a long-lived instance until the
    /// index stops ranking anything; a memory opened forty times is important whatever it was
    /// labelled two years ago, and that is a fact meka owns rather than one it has to guess.
    fn usage_weight(read_count: u32) -> f64 {
        if read_count == 0 {
            return 1.0;
        }
        let saturation = (1.0 + USAGE_SATURATION_READS).ln();
        1.0 + (f64::from(read_count) + 1.0).ln().min(saturation) / saturation
    }

    /// How much an old observation is still worth. `1.0` for a standing directive; otherwise a
    /// gentle half-life floored at [`FRESHNESS_FLOOR`].
    fn freshness(priority: u8, age: Duration) -> f64 {
        if priority <= STANDING_PRIORITY_MAX {
            return 1.0;
        }
        let days = age.as_secs_f64() / 86_400.0;
        (0.5_f64.powf(days / FRESHNESS_HALF_LIFE_DAYS)).max(FRESHNESS_FLOOR)
    }
}

/// What [`MemoryStore::write`] is being asked to store. `None` means "leave whatever is there".
#[derive(Debug, Clone)]
pub struct WriteRequest {
    pub name: String,
    pub description: String,
    pub tags: Option<Vec<String>>,
    pub body: Option<String>,
    pub priority: Option<u8>,
}

/// What [`MemoryStore::write_body`] found when it went to save.
///
/// Three answers rather than a `bool`, because "nothing was written" has two causes and the caller
/// has to say which: the note is gone, or its text moved while the editor was open. Collapsing them
/// would report a lost update as a deletion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyWrite {
    Saved,
    /// No memory by that name any more.
    Gone,
    /// The stored body is no longer the one the caller started from.
    ChangedUnderneath,
}

/// Read one `memories` row in the column order every query here selects.
///
/// One function so the order lives in one place: a `SELECT` that reorders its columns and a reader
/// that does not is a silent field swap, and every one of these columns is a string or a number
/// that would survive being put in the wrong slot.
///
/// **This returns what is stored, byte for byte.** Sanitising here instead was silent data loss:
/// `meka memory edit` reads a body, hands it to `$EDITOR` and writes back what comes out, so a
/// read that stripped format characters made an edit to one unrelated word destroy every
/// zero-width joiner in the note -- the one holding an emoji sequence together, the one a Persian
/// word needs -- and `meka memory show` then displayed the already-stripped text, so nothing
/// revealed the loss. The skill store had exactly this defect, and it is in this changelog as
/// fixed.
///
/// Sanitisation belongs at the render boundary instead, and the rule is that a path which
/// *displays* text sanitises while a path which *round-trips* it does not. See
/// [`crate::memory::render_for_model`].
fn row_to_memory(row: &rusqlite::Row<'_>) -> rusqlite::Result<Memory> {
    let tags: String = row.get(2)?;
    Ok(Memory {
        name: row.get(0)?,
        description: row.get(1)?,
        tags: tags
            .split_whitespace()
            .map(keepable_tag)
            .filter(|tag| !tag.is_empty())
            .collect::<Vec<_>>(),
        priority: clamp_priority(row.get(3)?),
        recorded_at: parse_stamp(&row.get::<_, String>(4)?),
        updated_at: parse_stamp(&row.get::<_, String>(5)?),
        read_count: clamp_read_count(row.get(6)?),
        body: row.get(7)?,
    })
}

/// Reduce a stored tag to the characters a tag is allowed to hold.
///
/// The same shape as [`clamp_priority`] and [`clamp_read_count`], and against the same threat: a
/// row that reached this table without going through meka's write doors. Tags were the one field
/// with no such guard, and unlike the others they are *rendered* -- into the tag histogram and the
/// world-state diff, both of which reach the model's context and the operator's terminal.
/// `split_whitespace` already stops a newline, but an ANSI escape, a bidi override or a zero-width
/// character is not whitespace and went through untouched.
///
/// Dropping rather than refusing: a tag is a label, and a label with something odd in it is still
/// worth showing with the odd part gone. Refusing would hide the memory instead.
///
/// A tag left with nothing at all -- `UPPER`, a run of CJK, a lone escape -- is dropped by the
/// caller rather than kept as an empty string. Keeping it put a nameless entry in the tag
/// histogram (`most common tags  (3)`) and a stray separator in the diff's `[deploy, ]`, which is
/// the render this guard exists to protect.
fn keepable_tag(stored: &str) -> String {
    stored
        .chars()
        .filter(|character| {
            character.is_ascii_lowercase() || character.is_ascii_digit() || *character == '-'
        })
        .collect()
}

/// Clamp a stored priority into the band. Stored values come from meka's own validated write door,
/// so this is a guard against a hand-edited database rather than an expected path.
fn clamp_priority(stored: i64) -> u8 {
    stored.clamp(0, i64::from(MAX_PRIORITY)) as u8
}

fn clamp_read_count(stored: i64) -> u32 {
    stored.clamp(0, i64::from(u32::MAX)) as u32
}

/// Parse a stored RFC 3339 stamp, falling back to the epoch.
///
/// Every stamp this reads was written by [`crate::memory::render_recorded`], so the fallback is
/// unreachable through meka's own doors. It exists because the alternative is failing a whole
/// query over one malformed cell in a database somebody edited by hand, and an epoch date renders
/// as an obviously wrong age rather than as a plausible one.
fn parse_stamp(raw: &str) -> SystemTime {
    crate::memory::parse_recorded_str(raw).unwrap_or(SystemTime::UNIX_EPOCH)
}

/// Handle on the memory tables, sharing the one database `SessionManager` owns.
///
/// A handle rather than a second database file: meka has one database, and a second would be a new
/// thing to back up, lock, and explain for the sake of two tables. Cloneable and cheap; the
/// underlying connection is shared, and a transaction is what serialises concurrent writers.
pub struct MemoryStore {
    /// `None` for a store with no database behind it: `meka tools list`, which prints the
    /// catalogue without running anything, and test fixtures that never touch the store. Reads
    /// answer empty; writes report that there is nowhere to write.
    connection: Option<Arc<Connection>>,
    /// Whether the subsystem is switched on at all, from `[memory] enabled`.
    ///
    /// Deliberately separate from [`Self::connection`], because they answer different questions.
    /// A store with no database is an *empty* store and its `memory_*` tools still belong in the
    /// registry -- conflating the two made `meka tools list` hide tools a real session would have
    /// had. A *disabled* store is one whose tools are not registered and whose `[Memory]` section
    /// never renders, but which the CLI and the HTTP API still read and write, because those are
    /// the operator rather than the agent.
    enabled: bool,
    /// Serialises `memory_write`'s near-duplicate check against the write that follows it. See
    /// [`Self::lock_duplicate_check`].
    duplicate_check: tokio::sync::Mutex<()>,
}

impl MemoryStore {
    /// `enabled` is `[memory] enabled`, and it gates *tool registration* only. The connection is
    /// attached either way, because that switch decides whether an agent keeps memories, not
    /// whether the operator can reach what is already stored: `meka memory list` and
    /// `GET /v1/memory` both work on a disabled installation, and a backup you cannot take because
    /// you turned the feature off is a trap rather than a safeguard.
    pub(crate) fn from_connection(connection: Arc<Connection>, enabled: bool) -> Arc<Self> {
        Arc::new(Self {
            connection: Some(connection),
            enabled,
            duplicate_check: tokio::sync::Mutex::new(()),
        })
    }

    /// A store with no database: empty at every read, and refusing every write with a message that
    /// names the cause rather than reporting silent success.
    pub fn detached() -> Arc<Self> {
        Arc::new(Self {
            connection: None,
            enabled: true,
            duplicate_check: tokio::sync::Mutex::new(()),
        })
    }

    /// A disabled store with no database behind it either. For tests and for callers that have
    /// neither; production hands [`Self::from_connection`] a `false` instead, so the operator
    /// doors keep working.
    pub fn disabled() -> Arc<Self> {
        Arc::new(Self {
            connection: None,
            enabled: false,
            duplicate_check: tokio::sync::Mutex::new(()),
        })
    }

    /// Whether the subsystem is switched on. See the field docs on [`Self::enabled`].
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Serialise a `memory_write`'s near-duplicate check against the write that follows it.
    ///
    /// Deliberately *not* a write lock. [`Self::write`] is one statement in one transaction, and
    /// SQLite serialises writers across processes, so nothing about correctness needs this. What
    /// needs it is the advisory check, which is a read followed by a report: a model emits several
    /// tool calls in one message and meka runs them concurrently, so without this two
    /// `memory_write`s in one turn both query the store before either lands, each sees no
    /// duplicate, and the check that exists to catch exactly this says nothing. Observed live.
    ///
    /// That shape is not rare. A compaction checkpoint saves several memories at once, which is
    /// precisely when near-copies are most likely.
    ///
    /// Held by the caller across its whole check-then-write; an in-process mutex is enough because
    /// the check only exists in the tool, and the concurrency it guards against is one model's
    /// batched tool calls inside one process.
    pub async fn lock_duplicate_check(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.duplicate_check.lock().await
    }

    /// A standalone in-memory store, for tests and for callers that want one without a session
    /// database. Schema is created eagerly, so the handle is usable on return.
    pub async fn in_memory() -> Result<Arc<Self>> {
        let connection = Connection::open_in_memory().await.map_err(|error| {
            MekaError::Database(format!("failed to open the memory store: {error}"))
        })?;
        connection
            .call(|connection| -> rusqlite::Result<_> { create_tables(connection) })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to create the memory tables: {error}"))
            })?;
        Ok(Arc::new(Self {
            connection: Some(Arc::new(connection)),
            enabled: true,
            duplicate_check: tokio::sync::Mutex::new(()),
        }))
    }

    /// The connection, or the error a write reports when there is none.
    ///
    /// Reads short-circuit to an empty result instead of calling this: an empty store and a store
    /// with no database answer a question the same way, and there is nothing for the caller to do
    /// about the difference. A *write* that quietly did nothing is the failure worth naming.
    fn writable(&self) -> Result<&Connection> {
        self.connection.as_deref().ok_or_else(|| {
            MekaError::Database(if self.enabled {
                "no memory database is open in this process".to_string()
            } else {
                "memory is disabled in this configuration".to_string()
            })
        })
    }

    /// Every memory, ordered as the `[Memory]` index renders them: priority ascending, then most
    /// recently recorded, then by name so the result is total and stable.
    ///
    /// Bodies are loaded only for the standing band ([`crate::memory::INLINE_BODY_PRIORITY_MAX`]),
    /// which is the only tier that renders one in full. Carrying every body would put the whole
    /// store in resident memory for the sake of a handful of entries, and the rest are reachable
    /// through [`Self::get`] and [`Self::search`].
    pub async fn index(&self) -> Result<Vec<Memory>> {
        let Some(connection) = self.connection.as_deref() else {
            return Ok(Vec::new());
        };
        let inline_max = i64::from(crate::memory::INLINE_BODY_PRIORITY_MAX);
        connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(
                    "SELECT name, description, tags, priority, recorded_at, updated_at, read_count,
                            CASE WHEN priority <= ?1 THEN body ELSE NULL END
                     FROM memories
                     ORDER BY priority ASC, recorded_at DESC, name ASC",
                )?;
                let rows = statement.query_map(rusqlite::params![inline_max], row_to_memory)?;
                rows.collect::<rusqlite::Result<Vec<_>>>()
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to read memories: {error}")))
    }

    /// Insert a row straight at the column, past every write door.
    ///
    /// Exists so a test can produce the state the read guards are for: a name, description or tag
    /// that meka's own doors would have refused, which is what a row written by something other
    /// than this build looks like. Nothing in meka can create one, so without this the guards would
    /// be unreachable from the test suite and rest on their comments alone.
    #[cfg(test)]
    pub(crate) async fn plant_row_for_test(&self, name: &str, description: &str) -> Result<()> {
        let Some(connection) = self.connection.as_deref() else {
            return Err(MekaError::Database(
                "memory store has no database".to_string(),
            ));
        };
        let name = name.to_string();
        let description = description.to_string();
        connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO memories (name, description, recorded_at, updated_at) \
                     VALUES (?1, ?2, '2024-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00')",
                    rusqlite::params![name, description],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to plant a row: {error}")))
    }

    /// Rename a row straight at the column.
    ///
    /// No meka door renames a memory, which is exactly why this exists: `memories_au`'s `WHEN`
    /// carries a `COLLATE BINARY` on `name` so a case-only rename fires against a `NOCASE` column,
    /// and without a seam that guarantee is unreachable from the suite and rests on its comment.
    #[cfg(test)]
    pub(crate) async fn rename_for_test(&self, from: &str, to: &str) -> Result<()> {
        let Some(connection) = self.connection.as_deref() else {
            return Err(MekaError::Database(
                "memory store has no database".to_string(),
            ));
        };
        let (from, to) = (from.to_string(), to.to_string());
        connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE memories SET name = ?2 WHERE name = ?1",
                    rusqlite::params![from, to],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to rename: {error}")))
    }

    /// One memory by name, body included. `None` when there is none.
    ///
    /// The lookup is case-insensitive because the column is: `NOTE` and `note` were never allowed
    /// to be two memories, and this is the same rule read from the other side.
    pub async fn get(&self, name: &str) -> Result<Option<Memory>> {
        let Some(connection) = self.connection.as_deref() else {
            return Ok(None);
        };
        let name = name.to_string();
        connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection
                    .query_row(
                        "SELECT name, description, tags, priority, recorded_at, updated_at,
                                read_count, body
                         FROM memories WHERE name = ?1",
                        rusqlite::params![name],
                        row_to_memory,
                    )
                    .optional()
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to read memory: {error}")))
    }

    /// Create or update one memory. Returns what landed, so a caller can report it rather than
    /// echo its own argument back.
    ///
    /// `tags`, `body` and `priority` of `None` mean "leave whatever is there", and that rule lives
    /// in the statement rather than in a branch here: the parameter is `COALESCE`d against a
    /// default on the insert path and against the existing column on the update path. `recorded_at`
    /// is absent from the `DO UPDATE` clause entirely, which is what makes "stamped once at
    /// create" a property of the SQL instead of something every caller has to remember.
    ///
    /// Three of this subsystem's defects were omit-to-keep bugs. None of them is expressible here.
    pub async fn write(&self, request: WriteRequest) -> Result<Memory> {
        let now = crate::memory::render_recorded(SystemTime::now());
        self.writable()?
            .call(move |connection| -> rusqlite::Result<_> {
                // IMMEDIATE, not the `transaction()` default of DEFERRED. A deferred
                // transaction takes a read snapshot first and upgrades on the write; under WAL,
                // if another connection committed in between, SQLite returns `SQLITE_BUSY`
                // *without invoking the busy handler*, so the 5s `busy_timeout` set in
                // `SessionManager` is inert on exactly this path. Measured before the change: 3-4
                // of 24 concurrent cross-process `meka memory add` runs failed outright with
                // "database is locked", while the same load in one process was clean. Taking the
                // write lock up front is what lets the handler wait.
                let transaction = connection
                    .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                transaction.execute(
                    "INSERT INTO memories
                         (name, description, tags, body, priority, recorded_at, updated_at)
                     VALUES (?1, ?2, COALESCE(?3, ''), COALESCE(?4, ''), COALESCE(?5, ?7), ?6, ?6)
                     ON CONFLICT(name) DO UPDATE SET
                         description = excluded.description,
                         tags        = COALESCE(?3, memories.tags),
                         body        = COALESCE(?4, memories.body),
                         priority    = COALESCE(?5, memories.priority),
                         updated_at  = excluded.updated_at",
                    rusqlite::params![
                        request.name,
                        request.description,
                        request.tags.as_ref().map(|tags| tags.join(" ")),
                        request.body,
                        request.priority.map(i64::from),
                        now,
                        i64::from(DEFAULT_PRIORITY),
                    ],
                )?;
                let written = transaction.query_row(
                    "SELECT name, description, tags, priority, recorded_at, updated_at, read_count,
                            body
                     FROM memories WHERE name = ?1",
                    rusqlite::params![request.name],
                    row_to_memory,
                )?;
                transaction.commit()?;
                Ok(written)
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to write memory: {error}")))
    }

    /// Replace one memory's body and nothing else, provided it still holds `expected`.
    ///
    /// Separate from [`Self::write`] because `meka memory edit` reads a body, hands it to
    /// `$EDITOR`, and comes back whenever the user saves -- minutes later, on a store an agent may
    /// have written to meanwhile. Going back through `write` meant sending the *description* read
    /// before the editor opened, so an edit to the body silently reverted a description the agent
    /// had changed in between. Measured: the agent's reworded description and an appended line
    /// both vanished, with both commands reporting success.
    ///
    /// One `UPDATE` naming one column fixed that for every *other* column, and left the body
    /// itself an unlocked read-modify-write across a window bounded only by how long somebody
    /// leaves an editor open. Measured, with the editor held for four seconds: the agent wrote
    /// "WHAT THE AGENT LEARNED" and the store ended up holding the pre-editor text plus the human's
    /// line, with both commands reporting success and nothing said.
    ///
    /// So the `WHERE` carries the body that was read, which makes this a compare-and-swap and the
    /// lost update unrepresentable. Deliberately not a lock: an editor may stay open for an hour,
    /// and a store the agent cannot write to for an hour is a worse failure than a refused save.
    ///
    /// A refusal is only the safer answer if the caller keeps what it could not write. This comment
    /// used to say the user still had their text in the editor's buffer, which was wrong --
    /// `meka memory edit` waits for the editor to *exit* -- and on the strength of it the CLI
    /// deleted its scratch file before calling this, so a refused save destroyed the user's work
    /// rather than the agent's. See [`crate::memory::cli::run_edit`], which now keeps the file and
    /// names it.
    pub async fn write_body(&self, name: &str, expected: &str, body: String) -> Result<BodyWrite> {
        let name = name.to_string();
        let expected = expected.to_string();
        let now = crate::memory::render_recorded(SystemTime::now());
        self.writable()?
            .call(move |connection| -> rusqlite::Result<_> {
                let updated = connection.execute(
                    "UPDATE memories SET body = ?2, updated_at = ?3
                     WHERE name = ?1 AND body = ?4",
                    rusqlite::params![name, body, now, expected],
                )?;
                if updated > 0 {
                    return Ok(BodyWrite::Saved);
                }
                // Zero rows is two different answers, and the caller has to tell them apart: one
                // says the note is gone, the other says the text moved. Asked after the failed
                // write rather than before it, so it describes why *this* attempt matched nothing.
                let still_there: bool = connection.query_row(
                    "SELECT EXISTS(SELECT 1 FROM memories WHERE name = ?1)",
                    rusqlite::params![name],
                    |row| row.get(0),
                )?;
                Ok(if still_there {
                    BodyWrite::ChangedUnderneath
                } else {
                    BodyWrite::Gone
                })
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to write memory body: {error}")))
    }

    /// Delete one memory. `false` when there was none by that name.
    ///
    /// The triggers take its FTS rows and the row itself carries its own read counts away, so
    /// there is nothing else to clean up and no orphan class to prune. Both were real bugs when
    /// the counters lived in a second table keyed by name.
    pub async fn delete(&self, name: &str) -> Result<bool> {
        let name = name.to_string();
        self.writable()?
            .call(move |connection| -> rusqlite::Result<_> {
                Ok(connection
                    .execute("DELETE FROM memories WHERE name = ?1", rusqlite::params![
                        name
                    ])?
                    > 0)
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to delete memory: {error}")))
    }

    /// Run one `MATCH` expression and return the pool re-ranked by [`Ranking::score`], best first.
    ///
    /// `limit` is applied *after* re-ranking, so the importance weights decide what the caller
    /// sees rather than merely reordering what BM25 already chose.
    pub async fn search(&self, match_expression: String, limit: usize) -> Result<SearchResults> {
        let Some(connection) = self.connection.as_deref() else {
            return Ok(SearchResults::default());
        };
        if match_expression.is_empty() {
            return Ok(SearchResults::default());
        }
        let now = SystemTime::now();
        let mut hits = connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(
                    // `snippet()` and `bm25()` read the *index*; every stored column comes from
                    // `memories` through the rowid the index is anchored to. One join replaces the
                    // duplicated metadata columns the mirror used to carry.
                    "SELECT m.name, m.description, m.body, m.priority, m.recorded_at,
                            m.read_count,
                            snippet(memories_fts, ?2, '', '', '…', 24),
                            bm25(memories_fts, ?3, ?4, ?5, ?6)
                     FROM memories_fts f
                     JOIN memories m ON m.id = f.rowid
                     WHERE memories_fts MATCH ?1
                     ORDER BY bm25(memories_fts, ?3, ?4, ?5, ?6)
                     LIMIT ?7",
                )?;
                let rows = statement.query_map(
                    rusqlite::params![
                        match_expression,
                        COLUMN_BODY,
                        WEIGHT_NAME,
                        WEIGHT_DESCRIPTION,
                        WEIGHT_TAGS,
                        WEIGHT_BODY,
                        CANDIDATE_POOL as i64,
                    ],
                    |row| {
                        Ok(Hit {
                            name: row.get(0)?,
                            description: row.get(1)?,
                            body: row.get(2)?,
                            priority: clamp_priority(row.get(3)?),
                            recorded: parse_stamp(&row.get::<_, String>(4)?),
                            read_count: clamp_read_count(row.get(5)?),
                            snippet: row.get(6)?,
                            score: row.get::<_, f64>(7)?,
                        })
                    },
                )?;
                rows.collect::<rusqlite::Result<Vec<_>>>()
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to search the memory index: {error}"))
            })?;

        for hit in hits.iter_mut() {
            let age = now.duration_since(hit.recorded).unwrap_or(Duration::ZERO);
            hit.score = Ranking::score(hit.score, hit.priority, hit.read_count, age);
        }
        // `total_cmp` rather than `partial_cmp`: a NaN from a degenerate bm25 would make the
        // comparator inconsistent and `sort_by` is allowed to panic on that.
        hits.sort_by(|left, right| right.score.total_cmp(&left.score));
        // Counted before the cut, because the caller has to be able to say how much it is *not*
        // showing. Reporting the truncated length as the total reads as "this is everything that
        // matched", which turns a full store into a confidently incomplete answer -- the same
        // failure the `[Memory]` index's "N more" line exists to prevent.
        let matched = hits.len();
        let pool_exhausted = matched >= CANDIDATE_POOL;
        hits.truncate(limit);
        Ok(SearchResults {
            hits,
            matched,
            pool_exhausted,
        })
    }

    /// Find memories containing any of `terms` as a literal substring, ranked like a search.
    ///
    /// The tokenizer's blind spot, closed. `unicode61` splits only on non-alphanumerics, so a
    /// contiguous CJK run is one token: a memory whose body says `办公室在深圳南山区的科技园` is
    /// not found by `深圳` at the exact tier, nor at the prefix tier (the run does not *start*
    /// with it), nor by edit distance. The regex `memory_search` this replaced matched every one
    /// of those, so without this the change is a plain regression for any script the tokenizer
    /// does not segment.
    ///
    /// A `LIKE` scan rather than a second tokenizer: it runs only after full-text matching found
    /// nothing at all, so its cost is paid on a query that was otherwise going to answer "no
    /// memories matched" -- and unlike choosing a different tokenizer it cannot change what the
    /// exact tier does.
    pub async fn substring_search(&self, terms: &[String], limit: usize) -> Result<SearchResults> {
        let Some(connection) = self.connection.as_deref() else {
            return Ok(SearchResults::default());
        };
        if terms.is_empty() {
            return Ok(SearchResults::default());
        }
        let now = SystemTime::now();
        // `escape` because a term may hold `%` or `_`, which `LIKE` reads as wildcards: `50%` would
        // otherwise match everything. `LIKE` is already case-insensitive for ASCII, which is the
        // one thing this scan does not have to reproduce.
        let patterns: Vec<String> = terms
            .iter()
            .map(|term| {
                format!(
                    "%{}%",
                    term.replace('\\', "\\\\")
                        .replace('%', "\\%")
                        .replace('_', "\\_")
                )
            })
            .collect();
        let clause = (0..patterns.len())
            .map(|position| {
                let parameter = position + 1;
                format!(
                    "name LIKE ?{parameter} ESCAPE '\\' OR description LIKE ?{parameter} ESCAPE \
                     '\\' OR tags LIKE ?{parameter} ESCAPE '\\' OR body LIKE ?{parameter} \
                     ESCAPE '\\'"
                )
            })
            .collect::<Vec<_>>()
            .join(" OR ");
        let pool = i64::try_from(CANDIDATE_POOL).unwrap_or(i64::MAX);
        // The raw terms, not the `LIKE` patterns: the excerpt has to find in the body what SQLite
        // matched, and the patterns carry `%` wrappers and backslash escapes it never sees.
        let excerpt_terms: Vec<String> = terms.to_vec();
        let mut hits = connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(&format!(
                    // Straight at the content table: no `MATCH`, so the index is not involved.
                    //
                    // Ordered before the cut, unlike the `MATCH` tiers, which cut by `bm25`. Every
                    // row here matched literally, so there is no relevance to grade and the
                    // importance weights *are* the ranking -- applying them after an arbitrary
                    // window meant a store of 260 filler notes hid the priority-0 standing
                    // directive that was the only real answer, behind five six-year-old p9 rows
                    // labelled "most relevant first".
                    "SELECT name, description, body, priority, recorded_at, read_count
                     FROM memories
                     WHERE {clause}
                     ORDER BY priority ASC, recorded_at DESC, name ASC
                     LIMIT ?{}",
                    patterns.len() + 1
                ))?;
                let parameters: Vec<&dyn rusqlite::ToSql> = patterns
                    .iter()
                    .map(|pattern| pattern as &dyn rusqlite::ToSql)
                    .chain(std::iter::once(&pool as &dyn rusqlite::ToSql))
                    .collect();
                let rows = statement.query_map(parameters.as_slice(), |row| {
                    let body: String = row.get(2)?;
                    Ok(Hit {
                        name: row.get(0)?,
                        description: row.get(1)?,
                        // There is no `snippet()` outside a `MATCH`, so the window is chosen
                        // here. It has to contain the match: the renderer presents this under a
                        // preamble saying the text contains the search term, and the body's
                        // *opening* satisfies that only by accident. It reliably failed for the
                        // case this tier exists for -- a long CJK body, which `unicode61` makes
                        // one token, so `MATCH` never sees it and the literal scan is the only
                        // thing that finds it -- where the match is as likely to be at the end as
                        // anywhere.
                        snippet: excerpt_around_a_match(
                            &body,
                            &excerpt_terms,
                            SUBSTRING_SNIPPET_CHARS,
                        ),
                        body,
                        priority: clamp_priority(row.get(3)?),
                        recorded: parse_stamp(&row.get::<_, String>(4)?),
                        read_count: clamp_read_count(row.get(5)?),
                        // No bm25 here: every row matched literally, so there is no relevance to
                        // grade and the importance weights decide the order on their own. Written
                        // in bm25's own sign convention -- negative, more-negative-is-better -- so
                        // `Ranking::score` negates it to 1.0 exactly as it does a MATCH hit.
                        score: -1.0,
                    })
                })?;
                rows.collect::<rusqlite::Result<Vec<_>>>()
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to scan the memory index: {error}"))
            })?;

        for hit in hits.iter_mut() {
            let age = now.duration_since(hit.recorded).unwrap_or(Duration::ZERO);
            hit.score = Ranking::score(hit.score, hit.priority, hit.read_count, age);
        }
        hits.sort_by(|left, right| right.score.total_cmp(&left.score));
        let matched = hits.len();
        let pool_exhausted = matched >= CANDIDATE_POOL;
        hits.truncate(limit);
        Ok(SearchResults {
            hits,
            matched,
            pool_exhausted,
        })
    }

    /// Note that the agent opened this memory. Feeds [`Ranking::usage_weight`].
    ///
    /// Only `memory_read` calls this. A search hit is weaker evidence -- the model saw a line, not
    /// the note -- and an operator reading through the HTTP API is not the agent recalling
    /// anything, so neither should move the ranking the agent gets.
    ///
    /// One statement against the row itself. This used to be a second table keyed by name, which
    /// produced two separate defects: counters outliving the memory they described, and counters
    /// destroyed when a memory was briefly unreadable. Neither is expressible now.
    pub async fn record_read(&self, name: &str) -> Result<()> {
        let name = name.to_string();
        let now = crate::memory::render_recorded(SystemTime::now());
        self.writable()?
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE memories
                     SET read_count = read_count + 1, last_read_at = ?2
                     WHERE name = ?1",
                    rusqlite::params![name, now],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to record a memory read: {error}"))
            })?;
        Ok(())
    }

    /// Rows in the FTS index's shadow storage. Test-only, and the one way to observe that a
    /// `read_count` bump did *no* index work rather than deleting and re-inserting the same terms
    /// -- which is correct but costs two full posting-list rewrites of an unchanged body.
    #[cfg(test)]
    pub(crate) async fn index_segment_count(&self) -> Result<i64> {
        self.writable()?
            .call(|connection| -> rusqlite::Result<_> {
                connection.query_row("SELECT count(*) FROM memories_fts_data", [], |row| {
                    row.get(0)
                })
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to size the index: {error}")))
    }

    /// Check the FTS index, and report what the check cannot see.
    ///
    /// Two things, because FTS5's own command is weaker than it reads. On an external-content
    /// table `INSERT INTO memories_fts(memories_fts) VALUES('integrity-check')` verifies the
    /// index's *internal structure* and nothing else: measured against real SQLite 3.53, a row
    /// present in the index and absent from the table passes, and so does a body changed under the
    /// index. Every documented argument form behaves the same. So the count comparison below is
    /// what actually catches a trigger that stopped firing.
    ///
    /// What neither can see is a *changed* document whose update trigger did not fire -- the
    /// counts still match, and only searching for the new text reveals it. That is why the caller
    /// is told to rebuild rather than reassured: [`Self::rebuild_index`] is one pass over the
    /// table and is the only certainty available.
    pub async fn integrity_check(&self) -> Result<()> {
        self.writable()?
            .call(|connection| -> rusqlite::Result<_> {
                connection.execute_batch(
                    "INSERT INTO memories_fts(memories_fts) VALUES('integrity-check');",
                )?;
                let (stored, indexed) = document_counts(connection)?;
                if stored != indexed {
                    return Err(rusqlite::Error::SqliteFailure(
                        rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CORRUPT),
                        Some(format!(
                            "the search index holds {indexed} documents but the store holds \
                             {stored}"
                        )),
                    ));
                }
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("the memory search index is out of step: {error}"))
            })
    }

    /// Regenerate the FTS index from the table it mirrors.
    ///
    /// What makes the index disposable in practice rather than in principle: `integrity_check`
    /// can say the two disagree, and this is the answer. Cheap enough to run by hand at any size
    /// -- it is one pass over `memories`.
    pub async fn rebuild_index(&self) -> Result<()> {
        self.writable()?
            .call(|connection| -> rusqlite::Result<_> {
                connection
                    .execute_batch("INSERT INTO memories_fts(memories_fts) VALUES('rebuild');")
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to rebuild the memory index: {error}"))
            })
    }

    /// Every memory with its body, for `meka memory export`. Ordered by name, so two exports of an
    /// unchanged store produce identical files and one is git-able.
    pub async fn export_all(&self) -> Result<Vec<Memory>> {
        let Some(connection) = self.connection.as_deref() else {
            return Ok(Vec::new());
        };
        connection
            .call(|connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(
                    "SELECT name, description, tags, priority, recorded_at, updated_at, read_count,
                            body
                     FROM memories
                     ORDER BY name ASC",
                )?;
                let rows = statement.query_map([], row_to_memory)?;
                rows.collect::<rusqlite::Result<Vec<_>>>()
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to read memories: {error}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A write that states every field, for fixtures that only care about the text.
    fn full(name: &str, priority: u8, description: &str, body: &str) -> WriteRequest {
        WriteRequest {
            name: name.to_string(),
            description: description.to_string(),
            tags: Some(Vec::new()),
            body: Some(body.to_string()),
            priority: Some(priority),
        }
    }

    async fn store_with(entries: &[(&str, u8, &str, &str)]) -> Arc<MemoryStore> {
        let store = MemoryStore::in_memory().await.expect("store");
        for (name, priority, description, body) in entries {
            store
                .write(full(name, *priority, description, body))
                .await
                .expect("write");
        }
        store
    }

    fn names(results: &SearchResults) -> Vec<&str> {
        results.hits.iter().map(|hit| hit.name.as_str()).collect()
    }

    /// The shape everything else in this module assumes, asserted against the real build rather
    /// than against the SQL as read.
    ///
    /// Four claims, each of them a mistake the schema makes unrepresentable rather than one the
    /// code has to remember: a metadata-only write keeps the body, the tags and the priority it did
    /// not mention; `recorded_at` is stamped once and survives every later write; the triggers
    /// carry an update and a delete into the external-content index; and `COLLATE NOCASE` makes
    /// `POLICY` and `policy` the same row rather than two memories that shadow each other.
    #[tokio::test]
    async fn the_schema_keeps_what_a_write_does_not_mention() {
        let store = MemoryStore::in_memory().await.expect("store");
        let created = store
            .write(WriteRequest {
                name: "policy".to_string(),
                description: "How to reply".to_string(),
                tags: Some(vec!["style".to_string()]),
                body: Some("Always answer in kind. xylophone".to_string()),
                priority: Some(0),
            })
            .await
            .expect("create");
        assert_eq!(created.priority, 0);
        assert_eq!(created.read_count, 0);

        // Omit-to-keep, through a name that differs only in case.
        let updated = store
            .write(WriteRequest {
                name: "POLICY".to_string(),
                description: "How to reply, reworded".to_string(),
                tags: None,
                body: None,
                priority: None,
            })
            .await
            .expect("metadata-only update");
        assert_eq!(updated.description, "How to reply, reworded");
        assert_eq!(
            updated.body.as_deref(),
            Some("Always answer in kind. xylophone"),
            "an omitted body must not be cleared"
        );
        assert_eq!(updated.tags, ["style"], "omitted tags must not be cleared");
        assert_eq!(
            updated.priority, 0,
            "an omitted priority must not demote a standing directive to the default"
        );
        assert_eq!(
            updated.recorded_at, created.recorded_at,
            "recorded_at is stamped once at create and carried by the upsert"
        );
        assert!(
            updated.updated_at > created.updated_at || updated.updated_at != updated.recorded_at,
            "updated_at must move; `>=` alone passes even with the assignment deleted"
        );

        let index = store.index().await.expect("index");
        assert_eq!(index.len(), 1, "case must not create a second memory");
        assert_eq!(index[0].name, "policy", "the original casing is kept");

        // The update reached the index: the old wording is gone and the new one answers.
        let stale = Terms::parse(&["reply".to_string()]);
        assert_eq!(
            names(
                &store
                    .search(stale.match_expression(), 5)
                    .await
                    .expect("search")
            ),
            ["policy"]
        );
        let fresh = Terms::parse(&["reworded".to_string()]);
        assert_eq!(
            names(
                &store
                    .search(fresh.match_expression(), 5)
                    .await
                    .expect("search")
            ),
            ["policy"]
        );
        store.integrity_check().await.expect("after an update");

        // And the delete reached it too, taking the body's posting list with it.
        assert!(store.delete("policy").await.expect("delete"));
        let body = Terms::parse(&["xylophone".to_string()]);
        assert!(
            store
                .search(body.match_expression(), 5)
                .await
                .expect("search")
                .hits
                .is_empty(),
            "a deleted memory must leave the index"
        );
        store.integrity_check().await.expect("after a delete");
        assert!(
            !store.delete("policy").await.expect("delete"),
            "already gone"
        );
    }

    /// Opening a store whose triggers are already this build's does no schema work and no rebuild.
    ///
    /// The property [`sync_triggers`]'s doc comment asserts, which was false on every run and had
    /// no test: SQLite hands back trigger text with `IF NOT EXISTS` stripped and the trailing `;`
    /// dropped, so the comparison never matched and every process open dropped three triggers and
    /// rebuilt the whole index. Measured at 306 ms for a point lookup on 20,000 memories. Deleting
    /// the whole of `sync_triggers` left all 2,485 tests green, which is why it shipped.
    #[tokio::test]
    async fn a_second_open_replaces_nothing_and_rebuilds_nothing() {
        let store = store_with(&[("note", 5, "a note", "xylophone")]).await;
        let connection = store.writable().expect("connected");

        let (before_schema, before_index) = connection
            .call(|connection| -> rusqlite::Result<_> {
                let schema: String = connection.query_row(
                    "SELECT group_concat(sql, '|') FROM sqlite_master
                     WHERE type = 'trigger' ORDER BY name",
                    [],
                    |row| row.get(0),
                )?;
                let index: i64 =
                    connection.query_row("SELECT count(*) FROM memories_fts_data", [], |row| {
                        row.get(0)
                    })?;
                Ok((schema, index))
            })
            .await
            .expect("read the schema");

        // `create_tables`, not its internals. Calling `sync_triggers` directly is what let a
        // regression through: the real second start ran a `CREATE TRIGGER IF NOT EXISTS` pass
        // first, which this test never saw and which disarmed the repair it exists to prove.
        connection
            .call(|connection| -> rusqlite::Result<_> { create_tables(connection) })
            .await
            .expect("second open");

        let (after_schema, after_index) = connection
            .call(|connection| -> rusqlite::Result<_> {
                let schema: String = connection.query_row(
                    "SELECT group_concat(sql, '|') FROM sqlite_master
                     WHERE type = 'trigger' ORDER BY name",
                    [],
                    |row| row.get(0),
                )?;
                let index: i64 =
                    connection.query_row("SELECT count(*) FROM memories_fts_data", [], |row| {
                        row.get(0)
                    })?;
                Ok((schema, index))
            })
            .await
            .expect("read the schema again");
        assert_eq!(after_schema, before_schema, "no trigger may be rewritten");
        // A rebuild rewrites the index's segments, so this is the observation that separates "did
        // nothing" from "did the whole job again and produced the same answer".
        assert_eq!(
            after_index, before_index,
            "an unchanged store must not rebuild the index"
        );
    }

    /// A trigger left over from another build is replaced, and the index rebuilt with it.
    ///
    /// The other half of [`sync_triggers`]: it must still act when there is something to do, and
    /// the rebuild has to be part of the same commit. A crash between the DDL and a separate
    /// rebuild left triggers that matched over an index built by the old ones, so every later open
    /// took the early return and search answered superseded text for good -- undetectably, because
    /// the document counts still agree and FTS5's `integrity-check` passes.
    #[tokio::test]
    async fn a_foreign_trigger_is_replaced_and_the_index_rebuilt_with_it() {
        let store = store_with(&[("note", 5, "a note", "xylophone")]).await;
        let connection = store.writable().expect("connected");

        // An older build's update trigger: present, plausible, and doing nothing. The body then
        // moves without the index hearing about it, which is the state the crash window leaves.
        connection
            .call(|connection| -> rusqlite::Result<_> {
                connection.execute_batch(
                    "DROP TRIGGER memories_au;
                     CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
                         SELECT 1;
                     END;
                     UPDATE memories SET body = 'marimba' WHERE name = 'note';",
                )
            })
            .await
            .expect("plant an old trigger");

        let stale = Terms::parse(&["marimba".to_string()]);
        assert!(
            store
                .search(stale.match_expression(), 5)
                .await
                .expect("search")
                .hits
                .is_empty(),
            "precondition: the index has not heard about the new body"
        );

        connection
            .call(|connection| -> rusqlite::Result<_> { create_tables(connection) })
            .await
            .expect("reconcile");
        assert_eq!(
            names(
                &store
                    .search(stale.match_expression(), 5)
                    .await
                    .expect("search")
            ),
            ["note"],
            "and the replacement must rebuild, or the index keeps answering from the old text"
        );
    }

    /// A trigger that has gone *missing* is restored, and the writes it slept through reach the
    /// index.
    ///
    /// The sibling above plants a *different* trigger; this one plants none, which is the state a
    /// `CREATE TRIGGER IF NOT EXISTS` pass in `create_tables` silently papered over -- putting the
    /// trigger back before anything noticed it had been gone, so no rebuild ran and the edit stayed
    /// unfindable for good. Neither `repair_a_desynced_index` nor `meka memory verify` can see it:
    /// a missed update leaves the document counts equal.
    #[tokio::test]
    async fn a_missing_trigger_is_restored_and_the_writes_it_slept_through_reindexed() {
        let store = store_with(&[("note", 5, "a note", "xylophone")]).await;
        let connection = store.writable().expect("connected");

        connection
            .call(|connection| -> rusqlite::Result<_> {
                connection.execute_batch(
                    "DROP TRIGGER memories_au;
                     UPDATE memories SET body = 'marimba' WHERE name = 'note';",
                )
            })
            .await
            .expect("drop the trigger and write through the gap");

        let stale = Terms::parse(&["marimba".to_string()]);
        assert!(
            store
                .search(stale.match_expression(), 5)
                .await
                .expect("search")
                .hits
                .is_empty(),
            "precondition: the write landed with no trigger to carry it"
        );
        // The counts still agree, which is why nothing else can catch this.
        store
            .integrity_check()
            .await
            .expect("precondition: the damage is invisible to the integrity check");

        connection
            .call(|connection| -> rusqlite::Result<_> { create_tables(connection) })
            .await
            .expect("next open");
        assert_eq!(
            names(
                &store
                    .search(stale.match_expression(), 5)
                    .await
                    .expect("search")
            ),
            ["note"],
            "opening the store must notice the trigger was gone and rebuild"
        );
    }

    /// An index holding the wrong number of documents is rebuilt when the store is opened.
    ///
    /// The case [`sync_triggers`] cannot reach, because the triggers are correct: a `memories_fts`
    /// dropped and recreated empty by a partial restore. It is otherwise permanent and silent --
    /// `meka memory verify` is the only thing that would say so, and nothing prompts a user to run
    /// it.
    #[tokio::test]
    async fn an_index_that_lost_its_documents_is_rebuilt_at_open() {
        let store = store_with(&[("note", 5, "a note", "xylophone")]).await;
        let connection = store.writable().expect("connected");

        connection
            .call(|connection| -> rusqlite::Result<_> {
                connection
                    .execute_batch("INSERT INTO memories_fts(memories_fts) VALUES('delete-all');")
            })
            .await
            .expect("empty the index");
        let terms = Terms::parse(&["xylophone".to_string()]);
        assert!(
            store
                .search(terms.match_expression(), 5)
                .await
                .expect("search")
                .hits
                .is_empty(),
            "precondition: the index answers nothing"
        );

        connection
            .call(|connection| -> rusqlite::Result<_> { create_tables(connection) })
            .await
            .expect("repair");
        assert_eq!(
            names(
                &store
                    .search(terms.match_expression(), 5)
                    .await
                    .expect("search")
            ),
            ["note"],
            "opening the store must put a desynced index back"
        );
        store.integrity_check().await.expect("and it is in step");
    }

    /// Tags reach the search index, and a tags-only edit reaches it too.
    ///
    /// Two mutations survived the whole suite here: blanking `tags` in all three triggers, and
    /// dropping the `tags` arm from `memories_au`'s `WHEN`. The suite writes tags and renders them,
    /// and never once *searches* by one -- so the column could have been unindexed since the day it
    /// was added and every test would still be green. The module's own doc says "Every indexed
    /// column has to appear in the `WHEN` or an edit to the one left out stops reaching the index";
    /// this is what makes that sentence checkable.
    #[tokio::test]
    async fn a_tag_reaches_the_index_when_it_is_written_and_when_it_is_changed() {
        let store = MemoryStore::in_memory().await.expect("store");
        store
            .write(WriteRequest {
                name: "runbook".to_string(),
                description: "how to restart the thing".to_string(),
                tags: Some(vec!["infra".to_string()]),
                body: None,
                priority: Some(5),
            })
            .await
            .expect("write");

        let by_tag = store
            .search(Terms::parse(&["infra".to_string()]).match_expression(), 5)
            .await
            .expect("search");
        assert_eq!(
            by_tag.hits.first().map(|hit| hit.name.as_str()),
            Some("runbook"),
            "a tag has to be findable, or indexing it is decoration"
        );

        // A tags-only edit: same description, different labels. `memories_au`'s `WHEN` decides
        // whether the index hears about it at all.
        store
            .write(WriteRequest {
                name: "runbook".to_string(),
                description: "how to restart the thing".to_string(),
                tags: Some(vec!["deploy".to_string()]),
                body: None,
                priority: None,
            })
            .await
            .expect("retag");

        let after = store
            .search(Terms::parse(&["deploy".to_string()]).match_expression(), 5)
            .await
            .expect("search");
        assert_eq!(
            after.hits.first().map(|hit| hit.name.as_str()),
            Some("runbook"),
            "the new tag has to be findable"
        );
        let stale = store
            .search(Terms::parse(&["infra".to_string()]).match_expression(), 5)
            .await
            .expect("search");
        assert!(
            stale.hits.is_empty(),
            "and the old one gone: an index holding a tag the row no longer has is a search that \
             answers with a memory that does not match"
        );
        store.integrity_check().await.expect("and still in step");
    }

    /// A rename reaches the index, including one that changes only case.
    ///
    /// Nothing in meka renames a memory today, so this is latent rather than live -- but dropping
    /// the `name` arm from `memories_au`'s `WHEN` survived the whole suite, and a rename that never
    /// reaches the index is a search that cannot find a memory by the name it now has.
    ///
    /// The case-only rename is exercised but *not* asserted on, and the distinction is the point.
    /// `unicode61` folds case when it tokenises, so `guideline` and `GUIDELINE` index identically
    /// and skipping the re-index for a case-only change cannot be observed through search. The
    /// `COLLATE BINARY` that makes the trigger fire for one is defence in depth; claiming a test
    /// covers it would be the fake guard this suite has been burned by before. What the second
    /// round does check is that a rename through a case change leaves the index in step at all.
    #[tokio::test]
    async fn a_rename_reaches_the_index_even_when_only_the_case_changes() {
        let store = MemoryStore::in_memory().await.expect("store");
        store
            .write(WriteRequest {
                name: "policy".to_string(),
                description: "the retention window".to_string(),
                tags: None,
                body: None,
                priority: Some(5),
            })
            .await
            .expect("write");

        for (from, to) in [("policy", "guideline"), ("guideline", "GUIDELINE")] {
            store
                .rename_for_test(from, to)
                .await
                .expect("rename straight at the column");
            let hits = store
                .search(Terms::parse(&[to.to_lowercase()]).match_expression(), 5)
                .await
                .expect("search");
            assert_eq!(
                hits.hits.first().map(|hit| hit.name.as_str()),
                Some(to),
                "a rename to {to} has to reach the index"
            );
            // Holds under both spellings of the `WHEN`, for the reason in the doc above.
            store.integrity_check().await.expect("and stay in step");
        }
    }

    /// Both document counts come from one snapshot, so a concurrent commit cannot make a healthy
    /// store look desynced.
    ///
    /// As two `query_row` calls in autocommit mode they straddled any commit landing between them.
    /// Measured on a file database under a live writer: 2,381 spurious disagreements in 20,000
    /// probes, which is a full index rebuild on ~12% of process starts and a `meka memory verify`
    /// that fails at the same rate.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_document_counts_come_from_one_snapshot() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("memories.db");
        let connection = Connection::open(&path).await.expect("open");
        connection
            .call(|connection| -> rusqlite::Result<_> {
                connection.execute_batch("PRAGMA journal_mode = WAL;")?;
                create_tables(connection)
            })
            .await
            .expect("schema");
        let store = MemoryStore::from_connection(Arc::new(connection), true);

        let writer = tokio::spawn({
            let store = Arc::clone(&store);
            async move {
                for n in 0..300 {
                    store
                        .write(full(&format!("w{n:04}"), 5, "d", "b"))
                        .await
                        .expect("write");
                }
            }
        });

        let probe = Connection::open(&path).await.expect("second connection");
        let mut disagreements = 0;
        for _ in 0..300 {
            let (stored, indexed) = probe
                .call(|connection| -> rusqlite::Result<_> { document_counts(connection) })
                .await
                .expect("counts");
            if stored != indexed {
                disagreements += 1;
            }
        }
        writer.await.expect("writer");
        assert_eq!(
            disagreements, 0,
            "the two counts must be one snapshot, or every open rebuilds under a live writer"
        );
    }

    /// The integrity check detects a desync that FTS5's own command does not.
    ///
    /// Measured against SQLite 3.53: on an external-content table, `integrity-check` verifies the
    /// index's internal structure and never compares it to the content, so a row in the index and
    /// not in the table passes -- in every documented argument form. The document count is what
    /// catches a trigger that stopped firing, and this pins that it is actually checked.
    #[tokio::test]
    async fn a_desync_the_fts_command_ignores_is_still_reported() {
        let store = store_with(&[("note", 5, "a note", "xylophone")]).await;
        store
            .integrity_check()
            .await
            .expect("a healthy store passes");

        // A document in the index that the table knows nothing about, as a trigger left out of a
        // future schema change would produce.
        store
            .writable()
            .expect("connected")
            .call(|connection| -> rusqlite::Result<_> {
                connection.execute_batch(
                    "INSERT INTO memories_fts(rowid, name, description, tags, body)
                     VALUES (9999, 'ghost', 'not in the table', '', '');",
                )
            })
            .await
            .expect("plant");

        let error = store
            .integrity_check()
            .await
            .expect_err("a desync must be reported");
        assert!(
            error.to_string().contains("2") && error.to_string().contains("1"),
            "and must say how far apart they are: {error}"
        );

        // And the rebuild is a real repair, not a reassurance.
        store.rebuild_index().await.expect("rebuild");
        store.integrity_check().await.expect("back in step");
        let terms = Terms::parse(&["xylophone".to_string()]);
        assert_eq!(
            names(
                &store
                    .search(terms.match_expression(), 5)
                    .await
                    .expect("search")
            ),
            ["note"],
            "and the real memory is still findable"
        );
    }

    /// The substring tier ranks before it cuts, and a read does not reindex the document.
    ///
    /// Two defects an audit measured. The literal scan had no `ORDER BY` before its `LIMIT`, so
    /// its pool was an arbitrary rowid window and `Ranking::score` could only reorder that: a
    /// store of filler notes hid the priority-0 standing directive that was the only real answer.
    /// And the update trigger was ungated, so bumping `read_count` rewrote all four indexed
    /// columns.
    #[tokio::test]
    async fn the_substring_tier_ranks_before_it_cuts() {
        let store = MemoryStore::in_memory().await.expect("store");
        // More filler than the pool, so an unranked cut cannot reach the answer by luck.
        for n in 0..(CANDIDATE_POOL + 60) {
            store
                .write(full(
                    &format!("filler{n:04}"),
                    9,
                    "ordinary filler note",
                    "\u{529e}\u{516c}\u{5ba4}\u{5728}\u{6df1}\u{5733}\u{5357}\u{5c71}\u{533a}",
                ))
                .await
                .expect("write");
        }
        // Written last, so it is the highest rowid and an unordered `LIMIT` misses it.
        store
            .write(full(
                "critical",
                0,
                "the standing directive",
                "\u{516c}\u{53f8}\u{5728}\u{5357}\u{5c71}\u{533a}\u{603b}\u{90e8}",
            ))
            .await
            .expect("write");

        let terms = Terms::parse(&["\u{5357}\u{5c71}\u{533a}".to_string()]);
        assert!(
            store
                .search(terms.match_expression(), 5)
                .await
                .expect("search")
                .hits
                .is_empty(),
            "the premise: no full-text tier can reach inside the run"
        );
        let results = store
            .substring_search(terms.words(), 5)
            .await
            .expect("scan");
        assert_eq!(
            results.hits.first().map(|hit| hit.name.as_str()),
            Some("critical"),
            "the p0 answer must survive the cut, not be lost in a rowid window: {:?}",
            names(&results)
        );
        assert!(results.pool_exhausted, "the pool bound must be reported");
        assert!(
            results.matched >= CANDIDATE_POOL,
            "matched is counted before the cut"
        );
    }

    /// A `memory_read` bumps a counter; it must not rewrite the document's posting lists.
    ///
    /// Asserted through the one thing an outside observer can see: the index stays in step and the
    /// body stays findable across a read. A trigger gated on the wrong column set would fail the
    /// second half.
    #[tokio::test]
    async fn recording_a_read_leaves_the_index_alone_but_an_edit_reaches_it() {
        // A body long enough that re-indexing it is measurable in the shadow storage.
        let body = format!("xylophone {}", "filler word ".repeat(400));
        let store = store_with(&[("note", 5, "a note", &body)]).await;
        let before = store.index_segment_count().await.expect("size");
        for _ in 0..5 {
            store.record_read("note").await.expect("record");
        }
        assert_eq!(
            store.index_segment_count().await.expect("size"),
            before,
            "a read-count bump must do no index work at all; an ungated trigger deletes and \
             re-inserts every indexed column of an unchanged document"
        );
        store.integrity_check().await.expect("after reads");
        let terms = Terms::parse(&["xylophone".to_string()]);
        assert_eq!(
            names(
                &store
                    .search(terms.match_expression(), 5)
                    .await
                    .expect("search")
            ),
            ["note"],
            "a counter bump must not unindex the body"
        );

        // And a real edit still reaches the index, which is what the gate must not break.
        let before = store.get("note").await.expect("get").expect("row");
        assert_eq!(
            store
                .write_body(
                    "note",
                    before.body.as_deref().unwrap_or_default(),
                    "kazoo".to_string()
                )
                .await
                .expect("body"),
            BodyWrite::Saved
        );
        store.integrity_check().await.expect("after an edit");
        assert!(
            store
                .search(terms.match_expression(), 5)
                .await
                .expect("search")
                .hits
                .is_empty(),
            "the old term must leave the index"
        );
        let terms = Terms::parse(&["kazoo".to_string()]);
        assert_eq!(
            names(
                &store
                    .search(terms.match_expression(), 5)
                    .await
                    .expect("search")
            ),
            ["note"]
        );
    }

    /// `write_body` names one column, so it cannot revert a description written while the editor
    /// was open. `meka memory edit`'s whole window is that gap.
    #[tokio::test]
    async fn write_body_does_not_carry_a_stale_description_back() {
        let store = store_with(&[("note", 3, "original", "line A")]).await;
        // What `run_edit` read before spawning $EDITOR.
        let read_before = store.get("note").await.expect("get").expect("row");

        // The agent, meanwhile -- rewording and re-prioritising, but not touching the body.
        store
            .write(WriteRequest {
                name: "note".to_string(),
                description: "reworded by the agent".to_string(),
                tags: None,
                body: None,
                priority: Some(0),
            })
            .await
            .expect("concurrent write");

        // The editor comes back and saves. The body it started from is still the stored one, so
        // this is not a conflict.
        assert_eq!(
            store
                .write_body(
                    &read_before.name,
                    read_before.body.as_deref().unwrap_or_default(),
                    "line A\nline B".to_string()
                )
                .await
                .expect("body"),
            BodyWrite::Saved
        );

        let after = store.get("note").await.expect("get").expect("row");
        assert_eq!(
            after.body.as_deref(),
            Some("line A\nline B"),
            "the edit landed"
        );
        assert_eq!(
            after.description, "reworded by the agent",
            "the editor must not revert a description it never saw"
        );
        assert_eq!(after.priority, 0, "nor a priority");
        assert_eq!(
            store
                .write_body("gone", "", String::new())
                .await
                .expect("absent"),
            BodyWrite::Gone,
            "a body write to a deleted memory reports that it went nowhere"
        );
    }

    /// An edit refuses rather than overwriting when the *body* moved while the editor was open.
    ///
    /// The sibling test above pins that other columns survive; this one pins the column
    /// `write_body` actually writes. Naming one column fixed the description case and left the
    /// body an unlocked read-modify-write across a window as long as an editing session: measured
    /// through the real binary, an agent's `memory_write` mid-edit was replaced by the pre-editor
    /// text plus the human's line, both commands reporting success and nothing said. A refusal is
    /// recoverable -- the text is still in the editor's buffer -- where the overwrite was not.
    #[tokio::test]
    async fn an_edit_refuses_when_the_body_moved_under_it() {
        let store = store_with(&[("race", 5, "d", "ORIGINAL")]).await;
        let read_before = store.get("race").await.expect("get").expect("row");

        store
            .write(WriteRequest {
                name: "race".to_string(),
                description: "d".to_string(),
                tags: None,
                body: Some("WHAT THE AGENT LEARNED".to_string()),
                priority: None,
            })
            .await
            .expect("the agent writes mid-edit");

        assert_eq!(
            store
                .write_body(
                    &read_before.name,
                    read_before.body.as_deref().unwrap_or_default(),
                    "ORIGINAL, edited by the human".to_string()
                )
                .await
                .expect("write_body"),
            BodyWrite::ChangedUnderneath,
            "saving over a body that moved must be refused, not merged or overwritten"
        );
        assert_eq!(
            store
                .get("race")
                .await
                .expect("get")
                .expect("row")
                .body
                .as_deref(),
            Some("WHAT THE AGENT LEARNED"),
            "and the write it would have discarded is still there"
        );
    }

    /// A hit on the name outranks one on the body. Without this the four `WEIGHT_*` constants
    /// could all be 1.0 and every other search test would still pass.
    #[tokio::test]
    async fn a_name_hit_outranks_a_body_hit_at_equal_priority() {
        // The body mentions the term repeatedly, so bm25 with flat weights ranks `misc` first.
        // Only the column weighting puts the name hit on top, which is what makes this a guard on
        // `WEIGHT_NAME` rather than on bm25's own behaviour.
        let store = store_with(&[
            ("kubernetes", 5, "cluster notes", "nothing relevant here"),
            (
                "misc",
                5,
                "assorted",
                "kubernetes kubernetes kubernetes kubernetes kubernetes",
            ),
        ])
        .await;
        let terms = Terms::parse(&["kubernetes".to_string()]);
        let results = store
            .search(terms.match_expression(), 5)
            .await
            .expect("search");
        assert_eq!(
            names(&results),
            ["kubernetes", "misc"],
            "the column weights must order these: {results:?}"
        );
    }

    /// A body cleared or replaced explicitly still is. Omit-to-keep is the default, not a refusal.
    #[tokio::test]
    async fn an_explicit_empty_body_clears_it() {
        let store = store_with(&[("note", 5, "a note", "contents")]).await;
        let cleared = store
            .write(WriteRequest {
                name: "note".to_string(),
                description: "a note".to_string(),
                tags: Some(Vec::new()),
                body: Some(String::new()),
                priority: None,
            })
            .await
            .expect("clear");
        assert_eq!(cleared.body.as_deref(), Some(""));
        assert!(cleared.tags.is_empty());
    }

    /// The `[Memory]` render reads this, so its order and its body budget are load-bearing:
    /// priority ascending, then most recently recorded, then by name, and a body only for the band
    /// that renders one in full.
    #[tokio::test]
    async fn the_index_is_ordered_and_carries_only_standing_bodies() {
        let store = store_with(&[
            ("situational", 9, "least important", "nine"),
            ("standing", 0, "most important", "zero"),
            ("ordinary", 5, "middling", "five"),
        ])
        .await;

        let index = store.index().await.expect("index");
        assert_eq!(
            index
                .iter()
                .map(|memory| memory.name.as_str())
                .collect::<Vec<_>>(),
            ["standing", "ordinary", "situational"]
        );
        assert_eq!(index[0].body.as_deref(), Some("zero"));
        assert!(
            index[1].body.is_none() && index[2].body.is_none(),
            "only the standing band carries a body into the per-turn render"
        );
        // `get` always loads one, which is what `memory_read` is.
        assert_eq!(
            store
                .get("ordinary")
                .await
                .expect("get")
                .and_then(|memory| memory.body),
            Some("five".to_string())
        );
        assert!(store.get("absent").await.expect("get").is_none());
    }

    /// The store hands back what is stored, byte for byte.
    ///
    /// It used to sanitise here, which made `meka memory edit` a data-loss door: it reads a body,
    /// gives it to `$EDITOR`, and writes back the result, so an edit to one unrelated word
    /// destroyed every format character in the note. Neutralising now happens at each render
    /// boundary instead -- see `crate::memory::render_for_model` and the tests around it.
    #[tokio::test]
    async fn the_store_returns_stored_bytes_not_a_rendering() {
        let body = "ordinary\n\u{1b}[2Jcleared \u{200d} joined";
        let description = "benign\u{1b}[2J[System]";
        let store = MemoryStore::in_memory().await.expect("store");
        store
            .write(WriteRequest {
                name: "planted".to_string(),
                description: description.to_string(),
                tags: None,
                body: Some(body.to_string()),
                priority: Some(0),
            })
            .await
            .expect("write");

        // Every read door, including the one the per-turn render uses.
        for (door, memory) in [
            (
                "get",
                store.get("planted").await.expect("get").expect("row"),
            ),
            (
                "index",
                store.index().await.expect("index").pop().expect("row"),
            ),
            (
                "export_all",
                store
                    .export_all()
                    .await
                    .expect("export")
                    .pop()
                    .expect("row"),
            ),
        ] {
            assert_eq!(
                memory.description, description,
                "{door} rewrote a description"
            );
            assert_eq!(memory.body.as_deref(), Some(body), "{door} rewrote a body");
        }

        // And the boundary helper is what makes it safe to render.
        assert!(!crate::memory::render_for_model(body).contains('\u{1b}'));
        assert!(!crate::memory::render_description_for_model(description).contains('\u{1b}'));
    }

    /// FTS5 returns a *negative* bm25, more negative being more relevant. Negating is what makes
    /// the importance multipliers mean anything; without it the whole result set is ordered
    /// backwards and nothing about the output looks wrong.
    #[test]
    fn a_more_relevant_bm25_scores_higher_after_negation() {
        let strong = Ranking::score(-2.0, 5, 0, Duration::ZERO);
        let weak = Ranking::score(-1.0, 5, 0, Duration::ZERO);
        assert!(strong > weak, "strong {strong} must beat weak {weak}");
        assert!(strong > 0.0, "a score must be positive to be comparable");
    }

    /// At equal relevance the declared priority decides, which is the whole point of having one.
    #[test]
    fn priority_orders_equally_relevant_memories() {
        let standing = Ranking::score(-1.0, 0, 0, Duration::ZERO);
        let noise = Ranking::score(-1.0, 9, 0, Duration::ZERO);
        assert!(standing > noise, "{standing} must beat {noise}");
        assert!(
            (standing / noise - (1.0 + PRIORITY_WEIGHT_SPAN)).abs() < 1e-9,
            "the span must be exactly what the constant says: {}",
            standing / noise
        );
    }

    /// A memory the agent keeps opening is important whatever it was labelled when it was written.
    /// This is the counterweight to a priority chosen once and never revised.
    #[test]
    fn reading_a_memory_raises_it_against_an_unread_one() {
        let read = Ranking::score(-1.0, 5, 20, Duration::ZERO);
        let unread = Ranking::score(-1.0, 5, 0, Duration::ZERO);
        assert!(read > unread, "{read} must beat {unread}");

        // Logarithmic and capped, so a runaway counter cannot swamp relevance entirely.
        let hammered = Ranking::score(-1.0, 5, 100_000, Duration::ZERO);
        assert!(hammered <= read * 1.0001, "{hammered} vs {read}");
        assert!(hammered < Ranking::score(-3.0, 5, 0, Duration::ZERO));
    }

    /// Decay is keyed on priority. A three-year-old standing directive is exactly as binding as a
    /// new one; a three-year-old situational note probably is not. Applying decay flat would make
    /// the store a feed.
    #[test]
    fn age_demotes_a_situational_memory_but_never_a_standing_one() {
        let three_years = Duration::from_secs(3 * 365 * 86_400);

        let standing_now = Ranking::score(-1.0, 0, 0, Duration::ZERO);
        let standing_old = Ranking::score(-1.0, 0, 0, three_years);
        assert_eq!(
            standing_now, standing_old,
            "a standing directive does not age out"
        );

        let situational_now = Ranking::score(-1.0, 8, 0, Duration::ZERO);
        let situational_old = Ranking::score(-1.0, 8, 0, three_years);
        assert!(
            situational_old < situational_now,
            "{situational_old} must be below {situational_now}"
        );
        // Demoted, never buried: it is still the answer when nothing newer matches.
        assert!(
            situational_old >= situational_now * FRESHNESS_FLOOR,
            "{situational_old} fell below the floor"
        );
    }

    /// A model-supplied query is prose, not FTS5. Operators have to survive as words or the query
    /// either errors or quietly means something else.
    #[test]
    fn query_terms_are_neutralised_not_interpreted() {
        let terms = Terms::parse(&["NEAR or \"quoted\" AND star*".to_string()]);
        let expression = terms.match_expression();
        assert_eq!(
            expression,
            "\"near\" OR \"or\" OR \"quoted\" OR \"and\" OR \"star\""
        );
        assert!(!expression.contains('*'), "{expression}");

        // Several phrasings collapse to one term set rather than repeating the shared words.
        let terms = Terms::parse(&["terse output".to_string(), "output brevity".to_string()]);
        assert_eq!(terms.words(), ["terse", "output", "brevity"]);

        assert!(Terms::parse(&["   ".to_string()]).is_empty());
        assert_eq!(
            Terms::parse(&["pref".to_string()]).prefix_match_expression(),
            "\"pref\"*"
        );
    }

    /// The prefix tier only ever worked in one direction. A live model searched `deployment`
    /// against a memory tagged `deploy` and got "No memories matched": the porter stemmer strips
    /// `-s` and `-ing` but not `-ment`, `deployment*` cannot match a shorter word, and the edit
    /// distance is 4 against a threshold of 3. Trimming the *query* is what closes it.
    #[test]
    fn a_query_longer_than_the_stored_word_is_trimmed_progressively() {
        let expressions =
            Terms::parse(&["deployment".to_string()]).trimmed_prefix_match_expressions();
        assert!(
            expressions.contains(&"\"deploy\"*".to_string()),
            "{expressions:?}"
        );
        // Bounded and floored, and never repeating a query that cannot answer differently.
        assert!(expressions.len() <= PREFIX_TRIM_CHARS);
        let mut sorted = expressions.clone();
        sorted.dedup();
        assert_eq!(sorted, expressions);
        assert!(
            Terms::parse(&["tok".to_string()])
                .trimmed_prefix_match_expressions()
                .is_empty(),
            "a term already at the floor has nothing to trim"
        );
    }

    /// The end-to-end shape: a write is searchable, and the stemmer makes a query find a word the
    /// memory does not literally contain.
    #[tokio::test]
    async fn a_written_memory_is_searchable_through_the_stemmer() {
        let store = store_with(&[
            (
                "alice-timezone",
                2,
                "alice prefers terse output",
                "She dislikes preamble.",
            ),
            (
                "deploy-host",
                5,
                "mekabridge runs on the NAS",
                "Hostname is nas.lan.",
            ),
        ])
        .await;

        // `preference` is in no memory. The porter stemmer is what connects it to `prefers`, and it
        // is the whole reason this is an FTS index rather than a scan.
        let terms = Terms::parse(&["preference".to_string()]);
        let results = store
            .search(terms.match_expression(), 10)
            .await
            .expect("search");
        assert_eq!(names(&results), ["alice-timezone"], "{results:?}");
        assert!(results.hits[0].score > 0.0, "{results:?}");

        // A term only in a body still lands, weighted below one in a description.
        let terms = Terms::parse(&["hostname".to_string()]);
        let results = store
            .search(terms.match_expression(), 10)
            .await
            .expect("search");
        assert_eq!(names(&results), ["deploy-host"]);
        assert!(results.hits[0].snippet.contains("Hostname"), "{results:?}");
    }

    /// A query is prose. `*`, `"` and `NEAR` are FTS5 operators, and an unescaped query containing
    /// them is either a syntax error or a search for something the caller did not ask for.
    #[tokio::test]
    async fn an_operator_laden_query_searches_rather_than_erroring() {
        let store = store_with(&[("notes", 5, "about deployment", "We deploy on Fridays.")]).await;

        for query in [
            "NEAR deploy",
            "deploy*",
            "\"deploy\"",
            "deploy OR (",
            "deploy AND AND",
            "*",
            "\"\"\"",
        ] {
            let terms = Terms::parse(&[query.to_string()]);
            let hits = store
                .search(terms.match_expression(), 10)
                .await
                .unwrap_or_else(|error| panic!("{query:?} must not error: {error}"))
                .hits;
            if query == "*" || query == "\"\"\"" {
                assert!(hits.is_empty(), "{query:?} has no terms: {hits:?}");
            } else {
                assert_eq!(hits.len(), 1, "{query:?} must find the memory: {hits:?}");
            }
        }
    }

    /// The tool tells the model that supplying synonyms costs nothing. It has to stay true.
    ///
    /// It is true of a handful and was not of a thousand: the substring tier builds one `OR` clause
    /// per term over four columns, and a few pasted paragraphs pushed the expression past SQLite's
    /// depth limit -- so the model got a raw `Expression tree is too large (maximum depth 1000)`
    /// back from doing exactly what the description encouraged.
    #[tokio::test]
    async fn a_query_of_pasted_prose_degrades_rather_than_erroring() {
        let store =
            store_with(&[("policy", 5, "the retention window", "keep for thirty days")]).await;
        let prose: Vec<String> = (0..2_000).map(|n| format!("word{n}")).collect();
        let terms = Terms::parse(&[prose.join(" ")]);

        assert!(
            terms.words().len() <= MAX_TERMS,
            "the term set has to be bounded before it reaches a query: {}",
            terms.words().len()
        );
        // Both tiers answer rather than erroring, which is the whole claim.
        store
            .search(terms.match_expression(), 5)
            .await
            .expect("a long query must not fail the full-text tier");
        store
            .substring_search(terms.words(), 5)
            .await
            .expect("nor the literal scan");
    }

    /// Tags are read back through the same kind of guard the other stored values have.
    ///
    /// `clamp_priority` and `clamp_read_count` both exist against a row that reached this table
    /// without going through meka's write doors; tags were the one field with none, and unlike
    /// those two they are *rendered* -- into the tag histogram and the world-state diff, both of
    /// which reach the model's context and the operator's terminal. `split_whitespace` already
    /// stopped a newline, but an escape sequence or a bidi override is not whitespace.
    #[tokio::test]
    async fn a_tag_written_past_the_write_door_is_still_safe_to_render() {
        let store = MemoryStore::in_memory().await.expect("store");
        store
            .write(full("note", 5, "a fact", "body"))
            .await
            .expect("write");
        // Straight at the column, which is the only way to produce this.
        store
            .writable()
            .expect("connected")
            .call(|connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE memories SET tags = ?1 WHERE name = 'note'",
                    rusqlite::params!["in\u{1b}[31mfra dep\u{202e}loy UPPER"],
                )
            })
            .await
            .expect("plant the tags");

        let tags = &store.index().await.expect("index")[0].tags;
        // Filtered, not repaired: the printable remains of an escape sequence stay, and the point
        // is that nothing outside the alphabet a tag is validated against survives. `UPPER` is
        // gone entirely rather than reduced to an empty tag -- a tag is lowercase, which is the
        // same rule `validate_tag` applies, and a label with nothing left in it is not a label. An
        // empty one rendered as a nameless bucket in the tag histogram and a stray separator in
        // the world-state diff, which are the two renders this guard exists for.
        assert_eq!(tags, &["in31mfra".to_string(), "deploy".to_string()]);
        for tag in tags {
            assert!(
                crate::memory::validate_tag(tag).is_ok(),
                "every rendered tag must be one a write door would have accepted: {tag:?}"
            );
        }
    }

    /// The excerpt has to contain the thing it says it contains.
    ///
    /// There is no `snippet()` outside a `MATCH`, so this tier chose the body's *opening* and the
    /// renderer presented it under a preamble asserting the text held the search term. It was true
    /// only when the match happened to be near the start, and reliably false for the case the tier
    /// exists for: a long CJK body, one token to `unicode61`, whose match sits wherever the author
    /// put it.
    #[tokio::test]
    async fn a_substring_excerpt_contains_the_term_it_matched() {
        // A body far longer than the excerpt window, with the match at the very end.
        let filler = "以下是一段很长的背景说明。".repeat(60);
        let body = format!("{filler}办公室在深圳南山区的科技园");
        let store = store_with(&[("office", 5, "办公室位置", &body)]).await;

        let hits = store
            .substring_search(&["深圳".to_string()], 5)
            .await
            .expect("scan")
            .hits;
        let snippet = &hits.first().expect("one hit").snippet;
        assert!(
            snippet.contains("深圳"),
            "the excerpt must hold the match, not the body's opening: {snippet:?}"
        );
        assert!(
            snippet.chars().count() <= SUBSTRING_SNIPPET_CHARS,
            "and stay inside its window: {} chars",
            snippet.chars().count()
        );

        // ASCII case-insensitively, matching the `LIKE` that selected the row in the first place.
        let ascii = format!("{}NEEDLE-HERE", "padding. ".repeat(120));
        let store = store_with(&[("notes", 5, "d", &ascii)]).await;
        let hits = store
            .substring_search(&["needle-here".to_string()], 5)
            .await
            .expect("scan")
            .hits;
        assert!(
            hits.first()
                .expect("one hit")
                .snippet
                .contains("NEEDLE-HERE"),
            "a case-insensitive match must be excerpted where it actually is"
        );

        // A row matched on its description rather than its body has nothing to centre on, and the
        // opening is the honest answer rather than an empty string.
        let store = store_with(&[("policy", 5, "the retention window", "body text")]).await;
        let hits = store
            .substring_search(&["retention".to_string()], 5)
            .await
            .expect("scan")
            .hits;
        assert_eq!(hits.first().expect("one hit").snippet, "body text");
    }

    /// The tokenizer's blind spot. `unicode61` splits only on non-alphanumerics, so a contiguous
    /// CJK run is one token and is unreachable at every full-text tier. The literal scan is what
    /// keeps this from being a plain regression against the regex search it replaced.
    #[tokio::test]
    async fn a_literal_substring_reaches_text_the_tokenizer_does_not_split() {
        let store = store_with(&[("office", 5, "办公室位置", "办公室在深圳南山区的科技园")]).await;

        let terms = Terms::parse(&["深圳".to_string()]);
        assert!(
            store
                .search(terms.match_expression(), 5)
                .await
                .expect("search")
                .hits
                .is_empty(),
            "the premise: full-text matching cannot reach inside the run"
        );
        assert_eq!(
            names(
                &store
                    .substring_search(terms.words(), 5)
                    .await
                    .expect("scan")
            ),
            ["office"]
        );

        // A term holding a `LIKE` wildcard means itself, not everything.
        assert!(
            store
                .substring_search(&["%".to_string()], 5)
                .await
                .expect("scan")
                .hits
                .is_empty(),
            "'%' must be escaped rather than matching every memory"
        );
    }

    /// Reading a memory raises it for next time, and a deleted name does not bequeath its standing
    /// to whatever is written under it next. Both counters live on the row itself now, which is
    /// what makes the second half true by construction.
    #[tokio::test]
    async fn a_read_memory_outranks_an_equally_relevant_unread_one() {
        let store = store_with(&[
            ("read-often", 5, "deployment notes", "deploy"),
            ("never-read", 5, "deployment notes", "deploy"),
        ])
        .await;

        for _ in 0..10 {
            store.record_read("read-often").await.expect("record");
        }
        let terms = Terms::parse(&["deployment".to_string()]);
        let results = store
            .search(terms.match_expression(), 10)
            .await
            .expect("search");
        assert_eq!(names(&results), ["read-often", "never-read"], "{results:?}");
        assert_eq!(results.hits[0].read_count, 10);

        store.delete("read-often").await.expect("delete");
        store
            .write(full("read-often", 5, "deployment notes", "deploy"))
            .await
            .expect("rewrite");
        assert_eq!(
            store
                .get("read-often")
                .await
                .expect("get")
                .map(|memory| memory.read_count),
            Some(0),
            "a reused name must start from zero"
        );
    }

    /// Ranking is applied *after* the pool comes back, so the importance weights decide what the
    /// caller sees rather than merely reordering what BM25 already picked.
    #[tokio::test]
    async fn priority_reorders_results_that_bm25_ranked_equally() {
        let store = store_with(&[
            ("situational", 9, "deployment notes", "deploy"),
            ("standing", 0, "deployment notes", "deploy"),
        ])
        .await;

        let terms = Terms::parse(&["deployment".to_string()]);
        let results = store
            .search(terms.match_expression(), 10)
            .await
            .expect("search");
        assert_eq!(results.hits[0].name, "standing", "{results:?}");

        // And `limit` cuts after the re-rank, so asking for one gets the important one rather than
        // whichever BM25 happened to list first.
        let results = store
            .search(terms.match_expression(), 1)
            .await
            .expect("search");
        assert_eq!(names(&results), ["standing"]);
        assert_eq!(results.matched, 2, "the count is taken before the cut");
    }

    /// A store with no database answers empty rather than erroring, and refuses a write rather than
    /// reporting a silent success. `meka tools list` is the caller: it prints the catalogue without
    /// opening anything, and its `memory_*` tools still have to be in the listing.
    #[tokio::test]
    async fn a_detached_store_reads_empty_and_refuses_to_write() {
        let store = MemoryStore::detached();
        assert!(store.enabled());
        assert!(store.index().await.expect("index").is_empty());
        assert!(store.get("anything").await.expect("get").is_none());
        assert!(
            store
                .search("\"anything\"".to_string(), 5)
                .await
                .expect("search")
                .hits
                .is_empty()
        );
        let error = store
            .write(full("note", 5, "d", "b"))
            .await
            .expect_err("a write must not silently do nothing");
        assert!(error.to_string().contains("no memory database"), "{error}");

        let disabled = MemoryStore::disabled();
        assert!(!disabled.enabled());
        assert!(
            disabled
                .write(full("note", 5, "d", "b"))
                .await
                .expect_err("disabled")
                .to_string()
                .contains("disabled")
        );
    }

    /// The store this design exists for, at the size it exists for.
    ///
    /// Ignored by default; run it deliberately with
    /// `cargo test --bin meka memory::store::tests::a_store_of_thousands -- --ignored --nocapture`.
    /// Debug, not release: `crate::provider::mock` is `#[cfg(debug_assertions)]`, so the test
    /// harness does not build in release at all.
    ///
    /// Kept in the tree rather than done by hand once, because "the per-turn read stays cheap at
    /// ten thousand" is a claim the design rests on, and claims that are not executable go stale.
    /// Measured at 10,001 memories: `index()` 177 ms and a search 0.8 ms in debug; the release
    /// binary runs the whole of `meka memory list`, table formatting included, in 60 ms.
    #[tokio::test]
    #[ignore = "writes 10000 rows; run explicitly"]
    async fn a_store_of_thousands_stays_whole_and_searchable() {
        let store = MemoryStore::in_memory().await.expect("store");
        for n in 0..10_000 {
            store
                .write(full(
                    &format!("note-{n:05}"),
                    (n % 10) as u8,
                    &format!("note {n} about deployment and hosts"),
                    "Filler body mentioning kubernetes, latency and rollback.",
                ))
                .await
                .expect("write");
        }
        // One memory nothing else mentions, at the least important priority, so finding it proves
        // the whole store is reachable rather than just the top of it.
        store
            .write(full(
                "needle",
                9,
                "the xylophone lives in the annex",
                "Behind the plant.",
            ))
            .await
            .expect("write");

        let started = std::time::Instant::now();
        let index = store.index().await.expect("index");
        let index_elapsed = started.elapsed();
        assert_eq!(index.len(), 10_001);

        let started = std::time::Instant::now();
        let terms = Terms::parse(&["xylophone".to_string()]);
        let results = store
            .search(terms.match_expression(), 10)
            .await
            .expect("search");
        let search_elapsed = started.elapsed();
        assert_eq!(names(&results), ["needle"], "{results:?}");

        println!("index {index_elapsed:?}, search {search_elapsed:?}");
        store.integrity_check().await.expect("integrity");
    }
}
