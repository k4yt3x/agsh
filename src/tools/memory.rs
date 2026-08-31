//! The `memory_*` tools: the agent's read/write access to its own durable notes
//! ([`crate::memory`]).
//!
//! All four gate at [`Permission::Read`], matching `scratchpad` and `todo`: these write to a store
//! meka owns in its own database, not to the user's tree, and the motivating deployment runs at
//! read permission permanently. Gating them at `workspace` would mean an agent that can never
//! remember anything, which defeats the feature.
//!
//! [`crate::memory::validate_memory_name`] is checked at every door that *writes* a name. The name
//! is no longer a path, so it is no longer a file-write primitive, but it is what
//! `meka memory export` turns into a file name and it is text the model reads in every turn's
//! index.
//!
//! The doors that only *look a name up* -- `memory_read` and `memory_delete` -- check
//! [`crate::memory::validate_memory_lookup`] instead, which requires only that the name is not
//! empty. It bounded length too, until that turned out to re-create this same wedge one length
//! short; the cost that bound existed for is bounded in `did_you_mean_hint` now.
//! Applying the write rule to them meant a row that reached the column past the tools was listed
//! to the model in the `[Memory]` index and then refused by every door that could have opened or
//! removed it, with `meka memory export` refusing the whole store on its account.

use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::{Tool, ToolOutput};
use crate::{
    error::{MekaError, Result},
    memory::{
        self, MemoryStore,
        store::{SearchResults, Terms, WriteRequest},
    },
    permission::Permission,
    provider::ToolDefinition,
};

/// Turn a store error into the tool's own failure, so the model sees which call failed.
fn tool_error(tool_name: &str, error: impl std::fmt::Display) -> MekaError {
    MekaError::ToolExecution {
        tool_name: tool_name.to_string(),
        message: error.to_string(),
    }
}

fn require_str<'a>(input: &'a serde_json::Value, key: &str, tool_name: &str) -> Result<&'a str> {
    input[key]
        .as_str()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("missing '{}' parameter", key),
        })
}

pub(super) struct MemoryWriteTool {
    pub memories: Arc<MemoryStore>,
}

#[async_trait]
impl Tool for MemoryWriteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "memory_write".to_string(),
            description: "Save a durable note that outlives this session. Use when you learn \
                something that will still matter in a later conversation: who someone is, how \
                they want you to work, a standing decision, or where something external lives. \
                Writing to a name that already exists updates it, so this is also how you \
                correct or refine an existing memory, or change just its priority: omit body and \
                whatever the memory already said is kept. Do not save what is derivable from the \
                code, git history, or the current conversation. The description is what you will \
                see in every future session, so make it stand on its own."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Identifier, letters/digits/-/_ only (e.g. 'alice-timezone')"
                    },
                    "description": {
                        "type": "string",
                        "description": "One line stating the fact itself, shown in every future \
                                        session's memory index. Required when creating a memory; \
                                        omit it to leave an existing memory's description untouched"
                    },
                    "priority": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 9,
                        "description": "Lower sorts higher in the index. 0 is a standing \
                                        directive and is the only tier whose body is in your \
                                        context every turn, so put a rule you must always follow \
                                        there; 1 also always applies but is listed by description \
                                        like the rest, 2-4 durable facts, 5 default, 6-9 \
                                        situational or short-lived"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lowercase labels ([a-z0-9-], at most 10), e.g. ['infra', \
                                        'deploy']. Indexed as words, so memory_search finds them. \
                                        Omit to leave an existing memory's tags untouched; pass \
                                        [] to clear them"
                    },
                    "body": {
                        "type": "string",
                        "description": "Optional detail, loaded only when memory_read is called. \
                                        Omit it to leave an existing memory's body untouched; \
                                        pass an empty string to clear it"
                    }
                },
                "required": ["name"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let name = require_str(&input, "name", "memory_write")?;
        memory::validate_memory_name(name).map_err(|message| MekaError::ToolExecution {
            tool_name: "memory_write".to_string(),
            message,
        })?;
        // Omit-to-keep, like `body`, `tags` and `priority`. It was required, which forced an agent
        // refining a stored note to resend a description it cannot actually see: the only copy in
        // its context is the index's, elided to 500 characters. So refining the *body* of a memory
        // whose description ran to 900 characters rewrote that description as 503 ending in `...`,
        // silently, on a call that never mentioned it. The store resolves the absence in SQL, and
        // refuses it when the write would create the memory rather than update one.
        //
        // A present-but-not-a-string `description` is refused rather than read as absent, for the
        // reason spelled out on `body` below.
        let description = match input.get("description") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::String(text)) => {
                if text.trim().is_empty() {
                    return Err(MekaError::ToolExecution {
                        tool_name: "memory_write".to_string(),
                        message: "'description' cannot be empty; omit it entirely to keep the \
                                  description a memory already has"
                            .to_string(),
                    });
                }
                // Whitespace is not the same question as "renders as nothing". Format characters
                // are not whitespace, so a description of three zero-width spaces got past every
                // write door and then rendered as blank in the index the model reads every turn.
                if !memory::description_says_something(text) {
                    return Err(MekaError::ToolExecution {
                        tool_name: "memory_write".to_string(),
                        message: "'description' renders as nothing once formatting characters are \
                             stripped"
                            .to_string(),
                    });
                }
                Some(text.clone())
            }
            Some(value) => {
                return Err(MekaError::ToolExecution {
                    tool_name: "memory_write".to_string(),
                    message: format!("'description' must be a string, got {}", value),
                });
            }
        };
        // An absent `body` means "leave it alone", not "make it empty". The schema has always
        // marked the field optional, so the call that changes only a priority is one the tool
        // invites -- and rendering the absence as `""` made that call silently delete everything
        // the memory said. `Some("")` is still an explicit request to clear it.
        //
        // A present-but-not-a-string `body` is refused rather than read as absent. `as_str`
        // returning `None` put `["line one", "line two"]` down the omit-to-keep path, so a new
        // memory was created with an empty body and reported as a plain success -- the model
        // believes it saved text that is not stored. `priority` in this same handler already
        // hard-errors on exactly this shape.
        let body = match input.get("body") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::String(text)) => Some(text.clone()),
            Some(value) => {
                return Err(MekaError::ToolExecution {
                    tool_name: "memory_write".to_string(),
                    message: format!("'body' must be a string, got {}", value),
                });
            }
        };
        // Same omit-to-keep rule as `body`, for the same reason: a call that changes only a
        // description would otherwise strip labels it never mentioned. `[]` still clears them.
        //
        // And the same refusal, for a sharper version of the same reason: `filter_map` dropped
        // every non-string element silently, so `tags: [1, 2]` collapsed to `[]` -- which *is* the
        // documented "clear them" signal. A malformed argument therefore erased labels the caller
        // never asked to remove, and the confirmation did not mention tags at all.
        let tags: Option<Vec<String>> = match input.get("tags") {
            Some(serde_json::Value::Array(values)) => {
                let mut collected = Vec::with_capacity(values.len());
                for value in values {
                    let Some(tag) = value.as_str() else {
                        return Err(MekaError::ToolExecution {
                            tool_name: "memory_write".to_string(),
                            message: format!(
                                "every entry in 'tags' must be a string, got {}",
                                value
                            ),
                        });
                    };
                    collected.push(tag.to_string());
                }
                Some(collected)
            }
            // A bare string where an array is declared, as `memory_search` also tolerates.
            Some(serde_json::Value::String(single)) => Some(vec![single.clone()]),
            None | Some(serde_json::Value::Null) => None,
            Some(value) => {
                return Err(MekaError::ToolExecution {
                    tool_name: "memory_write".to_string(),
                    message: format!("'tags' must be a list of strings, got {}", value),
                });
            }
        };
        // Clamped rather than rejected, because this door's caller is a model rather than a
        // person: `meka memory add --priority 99` and `PUT /v1/memory` both refuse out of range,
        // and a human reads the error and retypes, where a model spends a turn on it. Read as
        // `i64` so a negative clamps to 0 instead of failing the `as_u64` cast outright.
        //
        // `None` means "leave it alone", which the upsert resolves in SQL. Reading the absence as
        // the default demoted a priority-0 standing directive to an ordinary note every time the
        // agent reworded it -- taking it out of the always-in-context tier on the call whose
        // purpose was to keep it accurate, and saying nothing.
        let priority = match input.get("priority") {
            Some(serde_json::Value::Null) | None => None,
            Some(value) => {
                let raw = value.as_i64().ok_or_else(|| MekaError::ToolExecution {
                    tool_name: "memory_write".to_string(),
                    message: format!("'priority' must be a whole number, got {}", value),
                })?;
                Some(memory::parse_priority(Some(raw), name))
            }
        };

        // Held from here until after the write, so a second `memory_write` issued in the same turn
        // queues behind this one rather than racing it. Not for the write's sake -- the upsert is
        // one transaction -- but for the near-duplicate check below, which is a read.
        let _duplicate_guard = self.memories.lock_duplicate_check().await;

        // Normalised before the write so the doors agree on what a tag is: `Infra` and ` infra `
        // are the same label, and a duplicate would otherwise be stored twice in one column.
        let tags = match tags {
            Some(tags) => Some(
                memory::normalize_tags(&tags)
                    .map_err(|message| tool_error("memory_write", message))?,
            ),
            None => None,
        };

        // Read before the write, since the write is what creates the row. Without this the
        // confirmation would claim to have kept the body of a memory that had none, on the very
        // call that created it. Whether there is a body to *keep*, not merely a row: testing
        // existence alone made an update to a body-less memory report ", keeping the existing
        // body" -- a claim about content that does not exist, on the one line whose whole job is
        // to distinguish a metadata update from a rewrite.
        let kept_existing_body = body.is_none()
            && self
                .memories
                .get(name)
                .await
                .map_err(|error| tool_error("memory_write", error))?
                .and_then(|existing| existing.body)
                .is_some_and(|existing| !existing.trim().is_empty());

        // Also before the write, so the incoming description is compared against the store as it
        // was rather than against itself.
        //
        // Only when one was given. A write that omits the description is not proposing any text to
        // be a near-duplicate of, and running the check against the stored description would match
        // the memory against itself and report every such update as a duplicate of the note it is
        // updating.
        let duplicate = match description.as_deref() {
            Some(description) => self.near_duplicate_of(name, description).await,
            None => None,
        };

        // One statement, one transaction, and no `flock`: two writes to one name are two upserts,
        // and SQLite serialises them. Omit-to-keep is in the SQL, so there is no read here to go
        // stale between the check above and this.
        let written = self
            .memories
            .write(WriteRequest {
                name: name.to_string(),
                // Normalised to one line here rather than only at `PUT /v1/memory`, so all three
                // write doors store the same thing. `meka memory export` normalises on the way
                // out, so a description holding a newline came back collapsed after a round trip
                // and the docs' "byte-exact" claim was false for exactly the doors an agent uses.
                description: description
                    .as_deref()
                    .map(crate::store::normalize_description),
                tags,
                body,
                priority,
            })
            .await
            .map_err(|error| tool_error("memory_write", error))?;

        tracing::info!("saved memory '{}'", written.name);
        Ok(ToolOutput::text(
            format!(
                // Not "it will appear in your memory index from the next turn on", which was an
                // unconditional promise the index cannot keep: it renders a prefix of at most 200
                // entries under an 8 KB budget, so in the store this change is sized for a new
                // low-priority note sorts behind everything and is listed nowhere. Search reaches
                // it at any store size, which is the guarantee actually on offer.
                "Saved memory '{}' (priority {}){}. It is in your memory store from the next turn \
                 on, and `memory_search` will find it whatever the index has room to list.{}",
                written.name,
                // What landed, not what was asked for: an omitted priority is resolved from the
                // stored row, so echoing the argument would report a number the store does not
                // carry.
                written.priority,
                // Stated so the two calls are distinguishable from the result alone: a metadata
                // update and a rewrite otherwise report identically, and the difference is the
                // whole body of the note.
                if kept_existing_body {
                    ", keeping the existing body"
                } else {
                    ""
                },
                match duplicate {
                    Some(existing) => format!(
                        "\n\nNote: '{existing}' already says something very similar. If this is \
                         the same fact, call memory_write on '{existing}' instead and delete \
                         '{name}' -- two near-copies both stay in the index for ever and neither \
                         supersedes the other."
                    ),
                    None => String::new(),
                }
            ),
            false,
        ))
    }
}

impl MemoryWriteTool {
    /// The name of an existing memory whose description says close to the same thing, if any.
    ///
    /// Advisory. It never blocks and never rewrites: this is the agent's decision to make, and the
    /// only failure worth preventing is the silent one where a store accumulates a hundred
    /// near-copies because nothing ever pointed out the ninety-nine. Mem0 resolves the same
    /// problem with a background model deciding ADD versus UPDATE; handing the observation to the
    /// agent that is already holding the context is cheaper and better informed.
    async fn near_duplicate_of(&self, name: &str, description: &str) -> Option<String> {
        // The cheap disqualifier first: a two-word description shares words with everything.
        let incoming = Terms::parse(&[description.to_string()]);
        if incoming.words().len() < DUPLICATE_MIN_TERMS {
            return None;
        }

        let hits = self
            .memories
            .search(incoming.match_expression(), 5)
            .await
            .inspect_err(|error| {
                tracing::debug!("duplicate check skipped: {}", error);
            })
            .ok()?
            .hits;

        for hit in hits {
            // Rewriting a memory under its own name is an update, which is the thing this is
            // trying to encourage, not a duplicate. Compared case-insensitively because the column
            // is: writing `POLICY` over an existing `policy` updates that one row, and a
            // case-sensitive check then reported the row it had just written as a near-copy and
            // told the model to `memory_delete` it -- which, resolving NOCASE, would have deleted
            // the memory rather than a duplicate of it.
            if hit.name.eq_ignore_ascii_case(name) {
                continue;
            }
            let existing = Terms::parse(std::slice::from_ref(&hit.description));
            let shared = incoming
                .words()
                .iter()
                .filter(|word| existing.words().contains(word))
                .count();
            let overlap = shared as f64 / incoming.words().len() as f64;
            if overlap >= DUPLICATE_TERM_OVERLAP {
                return Some(hit.name);
            }
        }
        None
    }
}

pub(super) struct MemoryReadTool {
    pub memories: Arc<MemoryStore>,
}

#[async_trait]
impl Tool for MemoryReadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "memory_read".to_string(),
            description: "Load one saved memory in full. The memory index in your context lists \
                each memory's name and description; call this when a memory's description \
                suggests it holds detail you need."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the memory, as listed in the memory index"
                    }
                },
                "required": ["name"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let name = require_str(&input, "name", "memory_read")?;
        // Validated here as at every other door, which the module doc has always claimed and this
        // one did not do. Two things follow from the omission: an invalid name was reported as
        // merely absent, and the miss path below loads the whole index and runs an edit distance
        // per stored name with no bound on the argument -- measured at 11 s for a 200,000-character
        // name against 200 memories, and 48 s against 20,000, synchronously on a runtime worker
        // with the cancellation token ignored. Capping the name at 64 characters closes that.
        memory::validate_memory_lookup(name).map_err(|message| MekaError::ToolExecution {
            tool_name: "memory_read".to_string(),
            message,
        })?;
        let entry = match self
            .memories
            .get(name)
            .await
            .map_err(|error| tool_error("memory_read", error))?
        {
            Some(entry) => entry,
            None => {
                // A half-remembered name is the common miss at scale, and on its own it ends the
                // line. Pointing at the near-miss costs nothing and is the same recovery an unknown
                // *tool* name already gets.
                let index = self
                    .memories
                    .index()
                    .await
                    .map_err(|error| tool_error("memory_read", error))?;
                return Err(tool_error(
                    "memory_read",
                    format!(
                        "no memory named '{}'.{}",
                        name,
                        crate::tools::did_you_mean_hint(
                            name,
                            index.iter().map(|entry| entry.name.as_str())
                        )
                    ),
                ));
            }
        };

        // Counted here and nowhere else. A search hit is weaker evidence -- the model saw a line,
        // not the note -- and an operator reading through the HTTP API is not the agent recalling
        // anything, so neither moves the ranking the agent gets. Best-effort: a counter that fails
        // to increment must not fail the read.
        if let Err(error) = self.memories.record_read(&entry.name).await {
            tracing::warn!(
                "could not record a read of memory '{}': {}",
                entry.name,
                error
            );
        }

        // Age is stated on the way out, not just in the index: a memory is a point-in-time
        // observation, and detail that was true months ago is exactly what gets asserted as
        // current fact without a nudge.
        let age = memory::render_age(entry.recorded_at, std::time::SystemTime::now());
        // Bounded, and said when it is. This was the one memory render with no ceiling on it, so a
        // 200 KB note arrived whole into the window -- and the two search tiers both go to the
        // trouble of excerpting precisely so this call is the deliberate way to spend that. A cut
        // the reader is not told about is the worse half: a truncated note reads as a complete one
        // whose author simply stopped.
        let body = memory::render_for_model(entry.body.as_deref().unwrap_or_default());
        let body = body.trim();
        let rendered = match body.chars().count() {
            // Not an error and not silence. A memory whose whole content is its description is a
            // normal thing to write; rendering nothing after the header left the model to read the
            // gap as a failed load and call again.
            0 => "(no body: this memory's description is all of it)".to_string(),
            length if length > READ_BODY_MAX_CHARS => format!(
                "{}\n\n[Body truncated: {} of {} characters shown.]",
                crate::context::clip_chars(body, READ_BODY_MAX_CHARS),
                READ_BODY_MAX_CHARS,
                length
            ),
            _ => body.to_string(),
        };
        Ok(ToolOutput::text(
            format!(
                "# {}\n\n{}\n\nSaved {}. This is what you recorded then, not live state; verify \
                 before relying on it.\n\n{}",
                entry.name,
                memory::render_description_for_model(&entry.description),
                age,
                rendered
            ),
            false,
        ))
    }
}

/// Default number of ranked entries `memory_search` returns.
///
/// An order of magnitude below the shared `MAX_SEARCH_MATCHES`, which sized a list of grep
/// *lines*. An entry
/// here carries a description, an excerpt and often a whole short body, so ten of them is already a
/// substantial read and a hundred would crowd out the turn that asked for them.
const DEFAULT_SEARCH_LIMIT: usize = 10;
const MAX_SEARCH_LIMIT: usize = 25;

/// A hit whose body is at most this long is shown in full instead of an excerpt.
///
/// Most memories are shorter than this, which is the point: the common recall becomes one round
/// trip rather than a search followed by a `memory_read` per hit.
const INLINE_BODY_MAX_CHARS: usize = 800;

/// Ceiling on a body `memory_read` renders.
///
/// Generous, because this call is how the agent deliberately spends context on one note and a
/// bound that fights that is worse than none. It exists because *no* bound at all was the one
/// unbudgeted memory render: both search tiers excerpt so that this is the considered way to load a
/// note in full, and a 200 KB body reaching the window through it undid the whole arrangement.
const READ_BODY_MAX_CHARS: usize = 16_000;

/// Ceiling on the whole rendered result, inlined bodies included.
const SEARCH_RESULT_MAX_BYTES: usize = 6_144;

/// Fraction of an incoming description's words that must already appear in an existing memory's
/// description before the write says so.
///
/// Term overlap rather than a `bm25` threshold, because bm25 is a corpus statistic: the same pair
/// of descriptions scores differently in a store of ten and a store of ten thousand, so a fixed
/// cut-off would fire constantly in one and never in the other. Overlap means the same thing at
/// every size, and it is the thing that can be explained in the message.
const DUPLICATE_TERM_OVERLAP: f64 = 0.6;

/// Below this many words an overlap ratio is noise: two three-word descriptions sharing two words
/// are usually unrelated notes that both mention the same noun.
const DUPLICATE_MIN_TERMS: usize = 3;

/// Largest edit distance the last-resort tier accepts between a query term and a word in a
/// memory's name or description.
///
/// Two for anything five characters or longer, because the most common typo is a *transposition*
/// (`Tokoy` for `Tokyo`) and Levenshtein charges two for one. A threshold of one -- which is what
/// scaling by length alone gives a five-letter word, and what
/// [`crate::tools::did_you_mean_hint`] uses for tool names -- misses the single most likely way a
/// remembered word comes out wrong. Short words stay at one, where two edits would match almost
/// anything.
fn fuzzy_threshold(term: &str) -> usize {
    let length = term.chars().count();
    if length < 5 {
        return 1;
    }
    (length / 3).clamp(2, 4)
}

/// Which tier answered, so the result can say so.
///
/// Stated in the output rather than kept internal: a prefix or edit-distance answer is a *guess*
/// about what the caller meant, and a model that cannot tell it apart from an exact hit will
/// report the guess as a recalled fact.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tier {
    Exact,
    Prefix,
    Substring,
    Fuzzy,
}

impl Tier {
    fn preamble(self) -> &'static str {
        match self {
            Tier::Exact => "",
            Tier::Prefix => {
                "No exact matches. These are prefix matches, so treat them as \
                             near-misses rather than what you asked for.\n\n"
            }
            Tier::Substring => {
                "No word matches. These contain what you asked for as a literal \
                             substring, which is how text the word splitter does not divide -- \
                             Chinese, Japanese, Thai, an identifier, a serial number -- is found.\n\n"
            }
            Tier::Fuzzy => {
                "No full-text matches. These are the closest memory names and \
                            descriptions by spelling, so they may be unrelated.\n\n"
            }
        }
    }
}

pub(super) struct MemorySearchTool {
    pub memories: Arc<MemoryStore>,
}

impl MemorySearchTool {
    /// The last-resort tier: rank by how close a query term comes to a word in each memory's name
    /// or description. Runs only when full-text matching found nothing at all, so its cost is paid
    /// on a query that was otherwise going to return an empty result.
    ///
    /// Names and descriptions only, because that is all the index carries. Substring matching,
    /// which needs the *body* too, is [`memory::store::MemoryStore::substring_search`] and runs as
    /// its own tier ahead of this one: doing it here read the body of no memory at all, so a CJK
    /// note was found by a word in its description and invisible by a word in its text.
    fn fuzzy_by_spelling(
        index: &[memory::Memory],
        terms: &[String],
        limit: usize,
    ) -> (Vec<(usize, memory::Memory)>, usize) {
        let mut scored: Vec<(usize, memory::Memory)> = Vec::new();
        for entry in index.iter() {
            let haystack = format!("{} {}", entry.name, entry.description).to_lowercase();
            let mut best = usize::MAX;
            for term in terms {
                let threshold = fuzzy_threshold(term);
                // Hoisted: this is invariant across the word loop, and the pre-filter below exists
                // precisely to be O(1) per pair. Computing it inside cost ~13x on the path whose
                // own comment cites 14 s and 93 s measurements.
                let term_chars = term.chars().count();
                for word in haystack.split(|c: char| !c.is_alphanumeric() && c != '_') {
                    if word.is_empty() {
                        continue;
                    }
                    // Two strings whose lengths differ by more than the threshold cannot be
                    // within it, so this rejects them without building the matrix. Without it a
                    // pasted blob -- a stack trace, base64, a URL, exactly what produces an
                    // all-tiers miss and reaches this code -- cost 14 s against 200 memories and
                    // 93 s against 20,000, on a runtime worker, uncancellable.
                    //
                    // Counted in *characters*, because `fuzzy_threshold` and `edit_distance` both
                    // are. Comparing `len()` measured bytes against a character threshold, which
                    // is the same number only for ASCII: for any other script the filter threw
                    // away candidates that were inside the threshold. `東京都` against a stored
                    // `東京` is one edit apart and three bytes apart, so a store holding it
                    // answered "No memories matched" where the same shape in Latin script found
                    // its near-miss.
                    if term_chars.abs_diff(word.chars().count()) > threshold {
                        continue;
                    }
                    let distance = crate::tools::edit_distance(term, word);
                    if distance <= threshold {
                        best = best.min(distance);
                    }
                }
            }
            if best != usize::MAX {
                scored.push((best, entry.clone()));
            }
        }
        scored.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.name.cmp(&right.1.name))
        });
        // Returned alongside the cut list, because everything past `limit` is something the caller
        // is not being shown. Truncating and saying nothing reads as "this is the whole candidate
        // set", which is the failure the ranked tier's own "N further" line exists to prevent --
        // and it matters more here, where the tie-break is alphabetical and forty candidates can
        // sit at one distance.
        let candidates = scored.len();
        scored.truncate(limit);
        (scored, candidates)
    }
}

#[async_trait]
impl Tool for MemorySearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "memory_search".to_string(),
            description: "Search every saved memory, including ones too old or low-priority to \
                appear in your index. Pass several phrasings in `queries`; they are searched \
                together, so \"terse\", \"brevity\" and \"verbosity\" in one call all find the \
                same memory. Matching is case-insensitive and handles word endings."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "One or more phrasings of what you are looking for. Supplying \
                                        synonyms costs nothing and is the best way to find a memory \
                                        whose wording you do not remember."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_SEARCH_LIMIT,
                        "description": format!("Maximum memories to return. Default: {DEFAULT_SEARCH_LIMIT}.")
                    }
                },
                "required": ["queries"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        // A bare string where an array is declared is a mistake models make constantly, and the
        // recovery is free. Rejecting it would cost a turn to say something the tool could simply
        // have understood.
        // Two distinct failures, two distinct messages. "Missing" is for a parameter that is not
        // there or is the wrong type; a parameter that *is* there but yields nothing searchable is
        // a different mistake and needs a different fix, and reporting it as missing sends the
        // caller looking for a bug in how it built the call rather than at what it asked for.
        let queries: Option<Vec<String>> = match input.get("queries") {
            Some(serde_json::Value::String(single)) => Some(vec![single.clone()]),
            Some(serde_json::Value::Array(values)) => Some(
                values
                    .iter()
                    .filter_map(|value| value.as_str().map(str::to_string))
                    .collect(),
            ),
            _ => None,
        };
        let Some(queries) = queries else {
            return Err(MekaError::ToolExecution {
                tool_name: "memory_search".to_string(),
                message: "missing 'queries' parameter".to_string(),
            });
        };
        // Coerced rather than ignored. `as_u64` alone returned `None` for `"3"`, `3.0` and `-1`
        // alike, and `unwrap_or` then silently substituted 10: the model asked for three results,
        // got ten, and nothing in the output said the parameter had been discarded. The same
        // handler already goes out of its way to accept a bare string where `queries` declares an
        // array, so meeting a numeric string here is the consistent behaviour, not a new leniency.
        let limit = match input.get("limit") {
            None | Some(serde_json::Value::Null) => DEFAULT_SEARCH_LIMIT,
            Some(value) => {
                let number = value
                    .as_i64()
                    .or_else(|| value.as_f64().map(|value| value as i64))
                    .or_else(|| value.as_str().and_then(|text| text.trim().parse().ok()))
                    .ok_or_else(|| MekaError::ToolExecution {
                        tool_name: "memory_search".to_string(),
                        message: format!("'limit' must be a whole number, got {}", value),
                    })?;
                number.clamp(1, MAX_SEARCH_LIMIT as i64) as usize
            }
        };

        let store = self.memories.as_ref();
        let terms = Terms::parse(&queries);
        if terms.is_empty() {
            return Err(MekaError::ToolExecution {
                tool_name: "memory_search".to_string(),
                message: "'queries' contained no searchable words; try different wording"
                    .to_string(),
            });
        }

        // Four tiers, tried in order and reported by name. Exact handles word endings through the
        // stemmer, prefix handles a truncation or a trailing typo, substring handles text the
        // tokenizer does not split into words at all, and spelling handles the rest.
        let mut tier = Tier::Exact;
        let mut results = store.search(terms.match_expression(), limit).await?;
        if results.hits.is_empty() {
            tier = Tier::Prefix;
            results = store.search(terms.prefix_match_expression(), limit).await?;
        }
        // Still the prefix tier, in the other direction: the query may be the *longer* derived
        // form. Longest prefix first, stopping at the first that answers, so the most specific
        // query that can find anything is the one that does. See
        // `Terms::trimmed_prefix_match_expressions`.
        if results.hits.is_empty() {
            for expression in terms.trimmed_prefix_match_expressions() {
                results = store.search(expression, limit).await?;
                if !results.hits.is_empty() {
                    break;
                }
            }
        }
        if results.hits.is_empty() {
            tier = Tier::Substring;
            results = store.substring_search(terms.words(), limit).await?;
        }

        let now = std::time::SystemTime::now();
        if !results.hits.is_empty() {
            return Ok(ToolOutput::text(
                render_hits(tier, &results, limit, now),
                false,
            ));
        }

        // Only now is the whole index loaded: the tiers above answer from SQL, and this one is
        // the last resort that has to scan.
        let index = self
            .memories
            .index()
            .await
            .map_err(|error| tool_error("memory_search", error))?;
        let (fuzzy, candidates) = Self::fuzzy_by_spelling(&index, terms.words(), limit);
        if fuzzy.is_empty() {
            return Ok(ToolOutput::text(
                format!(
                    "No memories matched {:?}. Try broader words, or several phrasings of the \
                     same idea in one call.",
                    queries
                ),
                false,
            ));
        }
        Ok(ToolOutput::text(
            render_fuzzy(&fuzzy, candidates, now),
            false,
        ))
    }
}

/// Render ranked hits: one entry per memory, with its body inlined when short enough.
fn render_hits(
    tier: Tier,
    results: &SearchResults,
    limit: usize,
    now: std::time::SystemTime,
) -> String {
    use std::fmt::Write as _;

    let hits = &results.hits;
    let mut out = String::from(tier.preamble());
    // The number of *matches*, not the number rendered. Reporting the truncated length as the
    // total reads as "this is everything that matched", which is how a full store becomes a
    // confidently incomplete answer.
    let _ = writeln!(
        out,
        "{}{} matching {}, most relevant first:\n",
        if results.pool_exhausted {
            "at least "
        } else {
            ""
        },
        results.matched,
        if results.matched == 1 {
            "memory"
        } else {
            "memories"
        }
    );
    let mut shown = 0;
    let mut budget_bound = false;
    for hit in hits {
        let mut entry = String::new();
        let _ = writeln!(
            entry,
            "- **{}** (p{}, recorded {}, read {}x)",
            hit.name,
            hit.priority,
            memory::render_age(hit.recorded, now),
            hit.read_count
        );
        // Elided, like every other rendered description. Descriptions are deliberately unbounded
        // at parse time, and the always-emit-the-first-entry rule below then lets one memory spend
        // the whole turn's budget on its description alone -- measured at 100 KB from a 6 KB
        // ceiling. `render_fuzzy` and the `[Memory]` index both already guard this.
        let _ = writeln!(
            entry,
            "  {}",
            crate::store::elide_description_for_index(&memory::render_description_for_model(
                &hit.description
            ))
        );
        let body = hit.body.trim();
        if !body.is_empty() {
            // The whole body when it is short, which most are: that turns the common recall into
            // one call instead of a search plus a `memory_read` per hit. Otherwise the excerpt
            // around the match, which is what a follow-up read is for.
            //
            // The excerpt is clipped as well as excerpted. `snippet()` bounds by *tokens*, not
            // bytes, and `unicode61` makes a whole CJK paragraph -- or a base64 blob, or one
            // unbroken path -- a single token, so the "excerpt" came back as the entire column: a
            // 200 KB body arrived whole through the branch that exists to avoid sending it.
            let rendered = if body.chars().count() <= INLINE_BODY_MAX_CHARS {
                memory::render_for_model(body)
            } else {
                format!(
                    "… {} …",
                    memory::render_for_model(&crate::context::clip_chars(
                        &hit.snippet,
                        INLINE_BODY_MAX_CHARS
                    ))
                )
            };
            for line in rendered.lines() {
                let _ = writeln!(entry, "  {}", line);
            }
        }
        entry.push('\n');

        // Always emit the first entry: one pathological memory longer than the budget should still
        // be visible rather than collapsing the result to a bare count.
        if shown > 0 && out.len() + entry.len() > SEARCH_RESULT_MAX_BYTES {
            budget_bound = true;
            break;
        }
        out.push_str(&entry);
        shown += 1;
    }

    // Against `matched`, not `hits.len()`: entries dropped by `limit` and entries dropped by the
    // byte budget are both things the caller is not seeing, and counting only the latter would
    // understate the remainder by exactly the amount `limit` removed.
    let hidden = results.matched.saturating_sub(shown);
    if hidden > 0 {
        let _ = writeln!(
            out,
            "{}{hidden} further match(es) not shown here{}",
            // Hedged for the same reason the header is. `matched` is capped at the candidate pool,
            // so once the pool is full this number is a floor, not a count: a store of ten
            // thousand reported "197 further" when nine thousand seven hundred were hidden.
            if results.pool_exhausted {
                "at least "
            } else {
                ""
            },
            // Naming the constraint that actually bound, and only offering a remedy that can
            // work. "raise `limit`" was printed when the byte budget had done the cutting, and
            // again when `limit` was already at `MAX_SEARCH_LIMIT` -- both times handing the model
            // a remedy that provably returns the identical result. A `limit` above the maximum is
            // clamped on the way in, so a caller who asked for 100 and got 25 is in the second
            // case without having done anything wrong.
            if budget_bound {
                "; the result hit its size limit, so narrow the query."
            } else if limit >= MAX_SEARCH_LIMIT {
                "; `limit` is already at its maximum, so narrow the query."
            } else {
                "; raise `limit` or narrow the query."
            }
        );
    }
    let _ = write!(out, "Call `memory_read` for the full text of any of these.");
    out
}

/// Render the spelling-distance tier. These carry no body: they are candidates, not answers, and
/// padding a guess with content invites it to be read as one.
fn render_fuzzy(
    scored: &[(usize, memory::Memory)],
    candidates: usize,
    now: std::time::SystemTime,
) -> String {
    use std::fmt::Write as _;

    let mut out = String::from(Tier::Fuzzy.preamble());
    let mut rendered_count = 0;
    for (shown, (_, entry)) in scored.iter().enumerate() {
        // Budgeted like `render_hits`, and elided like every other index line. Descriptions are
        // not bounded at parse time, and this is the tier that is explicitly "candidates, not
        // answers" -- spending the turn's context on a guess is the wrong way round.
        let line = format!(
            "- **{}** (p{}, recorded {}, read {}x): {}\n",
            entry.name,
            entry.priority,
            memory::render_age(entry.recorded_at, now),
            entry.read_count,
            crate::store::elide_description_for_index(&memory::render_description_for_model(
                &entry.description
            ))
        );
        if shown > 0 && out.len() + line.len() > SEARCH_RESULT_MAX_BYTES {
            break;
        }
        out.push_str(&line);
        rendered_count += 1;
    }
    // This tier truncated in two places -- `limit` in `fuzzy_by_spelling` and the byte budget
    // above -- and said neither, so forty candidates at the same edit distance rendered nineteen
    // and read as the whole set. The tie-break past `limit` is alphabetical, which is arbitrary
    // enough that not saying so is the difference between a shortlist and a wrong answer.
    let hidden = candidates.saturating_sub(rendered_count);
    if hidden > 0 {
        let _ = write!(
            out,
            "\n{hidden} further candidate(s) scored the same or worse and are not shown.\n"
        );
    }
    let _ = write!(
        out,
        "\nCall `memory_read` to see whether one of these is what you meant."
    );
    out
}

pub(super) struct MemoryDeleteTool {
    pub memories: Arc<MemoryStore>,
}

#[async_trait]
impl Tool for MemoryDeleteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "memory_delete".to_string(),
            description: "Delete a saved memory permanently. Use when something you recorded has \
                turned out to be wrong or no longer applies. To revise a memory rather than drop \
                it, call memory_write with the same name instead."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the memory to delete"
                    }
                },
                "required": ["name"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let name = require_str(&input, "name", "memory_delete")?;
        memory::validate_memory_lookup(name).map_err(|message| MekaError::ToolExecution {
            tool_name: "memory_delete".to_string(),
            message,
        })?;

        // One `DELETE`, and the row carries its own read counts away with it. Separate steps
        // against a second table keyed by name let the doors disagree twice: one leaks the counters
        // the others clear, and one refuses a file the others remove.
        if !self
            .memories
            .delete(name)
            .await
            .map_err(|error| tool_error("memory_delete", error))?
        {
            return Err(tool_error(
                "memory_delete",
                format!("no memory named '{}'", name),
            ));
        }

        tracing::info!("deleted memory '{}'", name);
        Ok(ToolOutput::text(
            format!("Deleted memory '{}'.", name),
            false,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real store behind an in-memory SQLite database, created and torn down per test, so these
    /// exercise the same path a session does.
    async fn store() -> Arc<MemoryStore> {
        MemoryStore::in_memory().await.expect("in-memory store")
    }

    fn output_text(output: &ToolOutput) -> String {
        output
            .content
            .iter()
            .map(|block| match block {
                crate::provider::ToolResultContent::Text { text } => text.clone(),
                _ => String::new(),
            })
            .collect()
    }

    #[tokio::test]
    async fn omitting_a_description_keeps_the_stored_one() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
        };

        // Longer than the index elides to, which is the case that made the loss invisible.
        let long = format!("the retention window is {} days", "very ".repeat(150));
        write
            .execute(
                serde_json::json!({"name": "retention", "description": long, "body": "first"}),
                CancellationToken::new(),
            )
            .await
            .expect("create");

        // A later call that means to touch only the body.
        write
            .execute(
                serde_json::json!({"name": "retention", "body": "second"}),
                CancellationToken::new(),
            )
            .await
            .expect("refine");

        let stored = memories
            .get("retention")
            .await
            .expect("read")
            .expect("still there");
        assert_eq!(
            stored.description,
            crate::store::normalize_description(&long),
            "refining the body must not rewrite the description"
        );
        assert_eq!(stored.body.as_deref(), Some("second"));

        // Nothing to keep: creating a memory still needs one, and the refusal says so.
        let missing = write
            .execute(
                serde_json::json!({"name": "brand-new", "body": "text"}),
                CancellationToken::new(),
            )
            .await;
        let message = match missing {
            Err(error) => error.to_string(),
            Ok(output) => output_text(&output),
        };
        assert!(
            message.contains("description is required to create it"),
            "creating without a description must say which case this is: {message}"
        );
        // The shape, not just the substring. Carried on a `rusqlite::Error::InvalidParameterName`
        // the model receives `database error: failed to write memory: Error("Invalid parameter
        // name: no memory named ...")`, a user-input mistake dressed as a database fault, with a
        // prefix inviting a retry under a different `name`. The substring assertion above passed
        // the whole time.
        assert!(
            !message.contains("database error") && !message.contains("Invalid parameter name"),
            "a missing description is a refusal, not a database fault: {message}"
        );
    }

    /// The spelling tier's length pre-filter counts characters, as the distance it guards does.
    ///
    /// An omitted `description` keeps the stored one, and is refused when there is none to keep.
    ///
    /// The field was required, and the only copy of a description an agent can see is the index's,
    /// elided to 500 characters. So the ordinary act of refining a note -- rewriting its body,
    /// changing its priority -- forced the model to resend a description it could only reconstruct
    /// from that elision, and a description longer than the cap came back as 503 characters ending
    /// in `...`. Nothing reported it: the write succeeded and the confirmation named the memory.
    /// It compared `term.len()` -- bytes -- against a threshold `fuzzy_threshold` and
    /// `edit_distance` both express in characters. The two are the same number only for ASCII, so
    /// for every other script the filter discarded candidates that were inside the threshold and
    /// the tier answered "No memories matched". `東京都` is one edit from a stored `東京` and three
    /// bytes from it; the Latin-script shape of the same query found its near-miss all along,
    /// which is what made this invisible.
    #[tokio::test]
    async fn the_spelling_tier_measures_a_near_miss_in_characters_not_bytes() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
        };
        for (name, description) in [("office", "東京 rollout plan"), ("abc-note", "abc rollout")]
        {
            write
                .execute(
                    serde_json::json!({ "name": name, "description": description }),
                    CancellationToken::new(),
                )
                .await
                .expect("write");
        }
        let search = MemorySearchTool { memories };

        // The Latin control: one edit, one byte, found before and after.
        let latin = output_text(
            &search
                .execute(
                    serde_json::json!({"queries": ["abcd"]}),
                    CancellationToken::new(),
                )
                .await
                .expect("search"),
        );
        assert!(latin.contains("abc-note"), "the premise: {latin}");

        // One edit, three bytes. Every earlier tier is blind to it: `unicode61` makes a CJK run a
        // single token so no `MATCH` can reach it, and `LIKE '%東京都%'` cannot match a shorter
        // stored string. The spelling tier is the only thing left.
        let cjk = output_text(
            &search
                .execute(
                    serde_json::json!({"queries": ["東京都"]}),
                    CancellationToken::new(),
                )
                .await
                .expect("search"),
        );
        assert!(
            cjk.contains("office"),
            "a near-miss must be measured in the same unit as the threshold that admits it: {cjk}"
        );
    }

    /// A description made only of formatting characters is not a description.
    ///
    /// Every write door asked `trim().is_empty()`, which is a question about whitespace; zero-width
    /// spaces are not whitespace. Three of them were accepted, reported as a plain success, and
    /// then rendered as nothing at all -- `- **name** (p5, today): ` in the index the model reads
    /// every turn, a blank cell in `meka memory list`, a blank line in `memory_search`. The model
    /// is told it saved a note that says nothing.
    #[tokio::test]
    async fn a_description_of_only_formatting_characters_is_refused() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
        };

        let refused = write
            .execute(
                serde_json::json!({"name": "invisible", "description": "\u{200b}\u{200b}\u{200b}"}),
                CancellationToken::new(),
            )
            .await
            .expect_err("a description that renders as nothing must not be stored");
        assert!(
            refused.to_string().contains("renders as nothing"),
            "and the refusal has to name the reason: {refused}"
        );
        assert!(
            memories.get("invisible").await.expect("get").is_none(),
            "nothing was written"
        );
    }

    /// `memory_read` opens a row whose name meka's own write door would have refused -- for its
    /// character class *or* its length.
    ///
    /// The write rule is not this door's business, and applying it here wedged the store. A row
    /// whose name reached the column past the tools is listed to the model in the `[Memory]` index
    /// every turn, and was then refused by `memory_read`, `memory_delete`, `meka memory remove`
    /// and `DELETE /v1/memory/{name}` alike, while `meka memory export` refused the whole store on
    /// its account -- and the remedy that refusal printed validated the name too, so it failed
    /// identically. Nothing meka shipped could open or remove it.
    ///
    /// The character class went first. The 64-character cap stayed behind, bounding the miss
    /// path's edit-distance cost by refusing the argument, and re-created the same wedge one
    /// length short: a name of 65 characters was equally beyond reach. The cost is bounded in
    /// `did_you_mean_hint` now, so this door applies no write rule at all.
    #[tokio::test]
    async fn memory_read_opens_a_name_its_write_door_would_have_refused() {
        let memories = store().await;
        let read = MemoryReadTool {
            memories: memories.clone(),
        };

        // Past the write door's 64-character cap, so nothing can create it.
        let long = "l".repeat(120);
        memories
            .plant_row_for_test(&long, "written straight to the column")
            .await
            .expect("plant the long row");
        let opened = output_text(
            &read
                .execute(
                    serde_json::json!({ "name": long }),
                    CancellationToken::new(),
                )
                .await
                .expect("a stored name must be reachable however long it is"),
        );
        assert!(
            opened.contains("written straight to the column"),
            "the long-named row must open: {opened}"
        );

        // Straight at the column, which is the only way to produce this: meka's own write doors
        // refuse this name, so only a hand-edited store or another tool can have put it there.
        memories
            .plant_row_for_test("hand.edited", "put here past the write doors")
            .await
            .expect("plant the row");

        let found = output_text(
            &read
                .execute(
                    serde_json::json!({ "name": "hand.edited" }),
                    CancellationToken::new(),
                )
                .await
                .expect("a row the index shows the model must be one it can open"),
        );
        assert!(found.contains("put here past the write doors"), "{found}");

        // And it can be got rid of, which is what makes the store unwedgeable.
        assert!(
            MemoryDeleteTool {
                memories: memories.clone(),
            }
            .execute(
                serde_json::json!({ "name": "hand.edited" }),
                CancellationToken::new()
            )
            .await
            .is_ok(),
            "and one it can remove"
        );
        assert!(
            memories.get("hand.edited").await.expect("get").is_none(),
            "the row is really gone, not merely reported gone"
        );
    }

    /// `memory_write` stores a one-line description, as the CLI and the HTTP door do.
    ///
    /// Only `PUT /v1/memory` normalised, so a model that put a newline in a description -- which
    /// nothing stops it doing -- had it stored verbatim, and `meka memory export` then wrote it
    /// collapsed. The round trip changed the text for the one door an agent actually uses.
    #[tokio::test]
    async fn a_written_description_is_stored_as_one_line() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
        };
        write
            .execute(
                serde_json::json!({
                    "name": "note",
                    "description": "first line\nsecond   line",
                }),
                CancellationToken::new(),
            )
            .await
            .expect("write");
        assert_eq!(
            memories
                .get("note")
                .await
                .expect("get")
                .expect("present")
                .description,
            "first line second line",
            "the description must be stored as the export would write it"
        );
    }

    /// `body` has always been optional, so a priority change is a call the schema invites.
    /// Rendering the absence as an empty body deletes everything the memory said.
    #[tokio::test]
    async fn test_write_without_a_body_keeps_the_existing_one() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
        };

        // Creating a memory without a body keeps nothing, and must not say it did.
        let output = write
            .execute(
                serde_json::json!({"name": "bare", "description": "No body"}),
                CancellationToken::new(),
            )
            .await
            .expect("create without a body");
        assert!(
            !output_text(&output).contains("keeping"),
            "{}",
            output_text(&output)
        );

        write
            .execute(
                serde_json::json!({
                    "name": "policy",
                    "description": "How to reply",
                    "body": "Always answer in kind."
                }),
                CancellationToken::new(),
            )
            .await
            .expect("initial write");

        let output = write
            .execute(
                serde_json::json!({
                    "name": "policy",
                    "description": "How to reply",
                    "priority": 0
                }),
                CancellationToken::new(),
            )
            .await
            .expect("metadata-only write");
        // Said out loud, so the two calls are distinguishable from their results alone.
        assert!(
            output_text(&output).contains("keeping the existing body"),
            "{}",
            output_text(&output)
        );

        let read = MemoryReadTool { memories };
        let output = read
            .execute(
                serde_json::json!({"name": "policy"}),
                CancellationToken::new(),
            )
            .await
            .expect("read");
        assert!(
            output_text(&output).contains("Always answer in kind."),
            "{}",
            output_text(&output)
        );

        // Clearing is still possible; it just has to be asked for.
        write
            .execute(
                serde_json::json!({
                    "name": "policy",
                    "description": "How to reply",
                    "body": ""
                }),
                CancellationToken::new(),
            )
            .await
            .expect("explicit clear");
        let output = read
            .execute(
                serde_json::json!({"name": "policy"}),
                CancellationToken::new(),
            )
            .await
            .expect("read");
        assert!(
            !output_text(&output).contains("Always answer in kind."),
            "{}",
            output_text(&output)
        );
    }

    /// The description was sanitised because it renders every turn; the body was not. Being read
    /// on demand does not make an escape sequence or a forged section heading any less effective
    /// once it arrives, and a body is model-authored text that goes straight back into a model's
    /// context.
    #[tokio::test]
    async fn test_read_sanitises_a_stored_body() {
        let memories = store().await;
        MemoryWriteTool {
            memories: memories.clone(),
        }
        .execute(
            serde_json::json!({
                "name": "planted",
                "description": "benign",
                "body": "ordinary line\n\u{1b}[2Jcleared",
            }),
            CancellationToken::new(),
        )
        .await
        .expect("write");

        let read = MemoryReadTool { memories };
        let output = read
            .execute(
                serde_json::json!({"name": "planted"}),
                CancellationToken::new(),
            )
            .await
            .expect("read");
        let text = output_text(&output);
        assert!(
            !text.contains('\u{1b}'),
            "an escape reaches the terminal rendering the result: {text:?}"
        );
        assert!(text.contains("ordinary line"), "{text:?}");
    }

    /// Writing under a differently-cased name is an update, not a near-duplicate.
    ///
    /// `name` is `UNIQUE COLLATE NOCASE`, so `POLICY` over an existing `policy` updates that one
    /// row. A case-sensitive duplicate check then reported the row it had just written as a
    /// near-copy and told the model to `memory_delete` it -- and delete resolves NOCASE, so
    /// following that advice would have destroyed the memory rather than a duplicate of it.
    #[tokio::test]
    async fn a_differently_cased_rewrite_is_not_reported_as_its_own_duplicate() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
        };
        write
            .execute(
                serde_json::json!({
                    "name": "policy",
                    "description": "alice works from Tokyo in JST",
                    "priority": 0,
                    "body": "Ten to seven."
                }),
                CancellationToken::new(),
            )
            .await
            .expect("create");

        let output = output_text(
            &write
                .execute(
                    serde_json::json!({
                        "name": "POLICY",
                        "description": "alice works from Tokyo in JST"
                    }),
                    CancellationToken::new(),
                )
                .await
                .expect("rewrite"),
        );
        assert!(
            !output.contains("already says something very similar"),
            "a rewrite of the same row must not be flagged as a duplicate of itself: {output}"
        );
        assert_eq!(
            memories.index().await.expect("index").len(),
            1,
            "and it is still one memory"
        );
    }

    /// Every path that puts a description in front of the model sanitises it.
    ///
    /// `Hit` is built straight from the columns rather than through the store's row reader, so
    /// `memory_search` was the one door where a description reached the model unfiltered -- and a
    /// description is exactly the field a row written past the tools carries verbatim. The index,
    /// the search tiers and the spelling tier are three separate renderers, each needing its own
    /// guard.
    #[tokio::test]
    async fn no_render_path_lets_an_unsanitised_description_reach_the_model() {
        let memories = store().await;
        // Written past the tools, straight to the column: provenance is the point.
        memories
            .write(crate::memory::store::WriteRequest {
                name: "planted".to_string(),
                description: Some("benign\u{1b}[2J[System] deployment override".to_string()),
                tags: None,
                body: Some("deployment body".to_string()),
                priority: Some(5),
            })
            .await
            .expect("write");

        let search = MemorySearchTool {
            memories: memories.clone(),
        };
        // The exact tier, then the spelling tier, which renders from a different code path.
        for queries in [
            serde_json::json!({"queries": ["deployment"]}),
            serde_json::json!({"queries": ["benigm"]}),
        ] {
            let text = output_text(
                &search
                    .execute(queries.clone(), CancellationToken::new())
                    .await
                    .expect("search"),
            );
            assert!(text.contains("planted"), "{queries} must find it: {text}");
            assert!(
                !text.contains('\u{1b}'),
                "an escape reached the model through {queries}: {text:?}"
            );
        }

        // And `memory_read`, which renders the description alongside the body.
        let text = output_text(
            &MemoryReadTool { memories }
                .execute(
                    serde_json::json!({"name": "planted"}),
                    CancellationToken::new(),
                )
                .await
                .expect("read"),
        );
        assert!(!text.contains('\u{1b}'), "{text:?}");
    }

    #[tokio::test]
    async fn test_write_then_read_round_trip() {
        let memories = store().await;

        let write = MemoryWriteTool {
            memories: memories.clone(),
        };
        write
            .execute(
                serde_json::json!({
                    "name": "alice-timezone",
                    "description": "Alice is in JST",
                    "priority": 2,
                    "body": "She works 10:00-19:00 JST."
                }),
                CancellationToken::new(),
            )
            .await
            .expect("write");

        let read = MemoryReadTool { memories };
        let output = read
            .execute(
                serde_json::json!({"name": "alice-timezone"}),
                CancellationToken::new(),
            )
            .await
            .expect("read");
        let text = format!("{:?}", output.content);
        assert!(text.contains("10:00-19:00 JST"), "{text}");
        // The freshness caveat must ride along; a recalled memory asserted as live state is the
        // failure this guards against.
        assert!(text.contains("not live state"), "{text}");
    }

    /// The tools run at `Permission::Read`, so a name that escapes its store would be an
    /// arbitrary-file write available in read-only mode. A priority must land on the same value
    /// whichever door it came through. `as_u64` rejects a negative outright where the frontmatter
    /// path clamps it to 0.
    #[tokio::test]
    async fn test_write_clamps_priority_like_the_file_path() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
        };

        let cases = [
            ("low", -5, memory::MIN_PRIORITY),
            ("high", 99, memory::MAX_PRIORITY),
            ("mid", 3, 3),
        ];
        for (name, given, _) in cases {
            write
                .execute(
                    serde_json::json!({"name": name, "description": "d", "priority": given}),
                    CancellationToken::new(),
                )
                .await
                .unwrap_or_else(|error| panic!("priority {given} must be accepted: {error}"));
        }

        for (name, given, expected) in cases {
            let entry = memories
                .get(name)
                .await
                .expect("get")
                .unwrap_or_else(|| panic!("'{name}' must have been written"));
            assert_eq!(entry.priority, expected, "priority {given}");
        }

        // A non-integer is still a hard error rather than a silent default.
        assert!(
            write
                .execute(
                    serde_json::json!({"name": "bad", "description": "d", "priority": "high"}),
                    CancellationToken::new()
                )
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_write_rejects_path_traversal() {
        let write = MemoryWriteTool {
            memories: store().await,
        };

        for bad in ["../escape", "a/b", "/abs", ".hidden"] {
            let result = write
                .execute(
                    serde_json::json!({"name": bad, "description": "x"}),
                    CancellationToken::new(),
                )
                .await;
            assert!(result.is_err(), "'{bad}' must be rejected");
        }
    }

    /// `memory_delete` removes a row whose name meka would not have written, and reports a name it
    /// simply does not hold as absent rather than malformed.
    ///
    /// This door is the reason the lookup/write split exists: a name meka would not write must
    /// still be removable, or the row is unreachable through every door at once and only raw
    /// `sqlite3` gets it out. The character class went first; the 64-character cap stayed and left
    /// the same wedge one length short, which is what this now covers.
    #[tokio::test]
    async fn memory_delete_removes_a_name_its_write_door_would_have_refused() {
        let memories = store().await;
        let delete = MemoryDeleteTool {
            memories: memories.clone(),
        };

        // Both shapes the write door refuses: the character class, and the length.
        for planted in ["../escape", &"l".repeat(120)] {
            memories
                .plant_row_for_test(planted, "written straight to the column")
                .await
                .expect("plant the row");
            delete
                .execute(
                    serde_json::json!({ "name": planted }),
                    CancellationToken::new(),
                )
                .await
                .unwrap_or_else(|error| panic!("'{planted}' must be removable: {error}"));
            assert!(
                memories.get(planted).await.expect("get").is_none(),
                "'{planted}' must be gone from the store"
            );
        }

        let absent = delete
            .execute(
                serde_json::json!({"name": "never-stored"}),
                CancellationToken::new(),
            )
            .await
            .expect_err("a name the store does not hold is still an error");
        assert!(
            absent.to_string().contains("no memory named"),
            "and it is a miss, not a malformed name: {absent}"
        );
    }

    /// The end-to-end recall the tool exists for: a query whose wording appears in no file still
    /// finds the memory, because the stemmer relates `preference` to `prefers`.
    #[tokio::test]
    async fn test_search_finds_a_memory_the_query_does_not_quote() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
        };
        write
            .execute(
                serde_json::json!({
                    "name": "deploy-host",
                    "description": "mekabridge runs on the NAS",
                    "priority": 3,
                    "body": "Hostname is nas.lan. The operator prefers ssh keys."
                }),
                CancellationToken::new(),
            )
            .await
            .expect("write");

        let search = MemorySearchTool { memories };
        let hit = search
            .execute(
                serde_json::json!({"queries": ["preference"]}),
                CancellationToken::new(),
            )
            .await
            .expect("search");
        let text = output_text(&hit);
        assert!(text.contains("deploy-host"), "{text}");
        // A short body is inlined, so the common recall is one call rather than a search plus a
        // read per hit.
        assert!(text.contains("nas.lan"), "{text}");
        // And the entry carries what is needed to judge it without another call.
        assert!(text.contains("p3"), "{text}");
        assert!(text.contains("read 0x"), "{text}");
    }

    /// Several phrasings in one call. This is the answer to "the model has to guess the words it
    /// used months ago": it does not have to guess right, only to guess several times.
    #[tokio::test]
    async fn test_search_accepts_several_phrasings_at_once() {
        let memories = store().await;
        MemoryWriteTool {
            memories: memories.clone(),
        }
        .execute(
            serde_json::json!({
                "name": "output-style",
                "description": "K4YT3X wants terse answers",
                "body": "No preamble, no recap."
            }),
            CancellationToken::new(),
        )
        .await
        .expect("write");

        let search = MemorySearchTool {
            memories: memories.clone(),
        };
        // "verbosity" is in no file; "terse" is. Supplying both finds it, and supplying only the
        // wrong one does not -- which is exactly why the parameter is a list.
        let hit = search
            .execute(
                serde_json::json!({"queries": ["verbosity", "terse", "brevity"]}),
                CancellationToken::new(),
            )
            .await
            .expect("search");
        assert!(
            output_text(&hit).contains("output-style"),
            "{:?}",
            hit.content
        );

        // A bare string where an array is declared is a mistake models make constantly, and
        // understanding it costs nothing.
        let hit = search
            .execute(
                serde_json::json!({"queries": "terse"}),
                CancellationToken::new(),
            )
            .await
            .expect("search");
        assert!(
            output_text(&hit).contains("output-style"),
            "{:?}",
            hit.content
        );
    }

    /// Every remedy the search offers has to be one that can work.
    ///
    /// "raise `limit`" was printed when `limit` was already at [`MAX_SEARCH_LIMIT`], and a caller
    /// who asked for more than that is clamped on the way in -- so the model was handed an action
    /// that provably returns the identical result, and a model that follows instructions takes it.
    #[tokio::test]
    async fn a_capped_search_does_not_advise_raising_a_limit_that_is_already_at_its_maximum() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
        };
        for index in 0..(MAX_SEARCH_LIMIT + 5) {
            write
                .execute(
                    serde_json::json!({
                        "name": format!("note-{index:02}"),
                        "description": format!("deployment note {index}"),
                    }),
                    CancellationToken::new(),
                )
                .await
                .expect("write");
        }
        let search = MemorySearchTool { memories };

        // Asking for more than the maximum is clamped, so this caller is at the ceiling without
        // having typed the ceiling.
        let capped = search
            .execute(
                serde_json::json!({"queries": ["deployment"], "limit": 100}),
                CancellationToken::new(),
            )
            .await
            .expect("search");
        let text = output_text(&capped);
        assert!(
            text.contains("further match"),
            "the premise: something was cut: {text}"
        );
        assert!(
            !text.contains("raise `limit`"),
            "`limit` is already at its maximum, so raising it is not a remedy: {text}"
        );

        // Below the ceiling it still is one.
        let room = search
            .execute(
                serde_json::json!({"queries": ["deployment"], "limit": 2}),
                CancellationToken::new(),
            )
            .await
            .expect("search");
        assert!(
            output_text(&room).contains("raise `limit`"),
            "{:?}",
            room.content
        );
    }

    /// `memory_read` is the one memory render with no ceiling on it, which is the whole reason both
    /// search tiers excerpt: this call is meant to be the deliberate way to spend context on a
    /// note. Unbounded, a 200 KB body arrived whole. A cut nobody is told about is the worse half:
    /// a truncated note reads as a complete one whose author stopped writing.
    #[tokio::test]
    async fn memory_read_bounds_a_long_body_and_says_when_there_is_none() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
        };
        write
            .execute(
                serde_json::json!({
                    "name": "enormous",
                    "description": "a very long note",
                    "body": "y".repeat(READ_BODY_MAX_CHARS * 2),
                }),
                CancellationToken::new(),
            )
            .await
            .expect("write");
        write
            .execute(
                serde_json::json!({"name": "terse", "description": "all of it is the description"}),
                CancellationToken::new(),
            )
            .await
            .expect("write");
        let read = MemoryReadTool {
            memories: memories.clone(),
        };

        let long = output_text(
            &read
                .execute(
                    serde_json::json!({"name": "enormous"}),
                    CancellationToken::new(),
                )
                .await
                .expect("read"),
        );
        assert!(
            long.chars().count() < READ_BODY_MAX_CHARS * 2,
            "the body must be bounded, got {} chars",
            long.chars().count()
        );
        assert!(
            long.contains("Body truncated"),
            "and the cut said out loud: {}",
            &long[long.len().saturating_sub(200)..]
        );

        let empty = output_text(
            &read
                .execute(
                    serde_json::json!({"name": "terse"}),
                    CancellationToken::new(),
                )
                .await
                .expect("read"),
        );
        assert!(
            empty.contains("no body"),
            "a memory with no body must say so rather than trail off into nothing: {empty}"
        );
    }

    /// A result cut down to `limit` has to say how much it is not showing. Reporting the rendered
    /// count as the total reads as "this is everything that matched", which is how a full store
    /// becomes a confidently incomplete answer -- the same failure the `[Memory]` index's "N more"
    /// line exists to prevent, and the search tool had it.
    #[tokio::test]
    async fn test_search_reports_matches_it_did_not_show() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
        };
        for index in 0..25 {
            write
                .execute(
                    serde_json::json!({
                        "name": format!("note-{index:02}"),
                        "description": format!("deployment note {index}"),
                    }),
                    CancellationToken::new(),
                )
                .await
                .expect("write");
        }

        let hit = MemorySearchTool { memories }
            .execute(
                serde_json::json!({"queries": ["deployment"], "limit": 3}),
                CancellationToken::new(),
            )
            .await
            .expect("search");
        let text = output_text(&hit);

        assert!(
            text.contains("25 matching memories"),
            "the header must count matches, not the three rendered: {text}"
        );
        assert!(
            text.contains("22 further match(es) not shown"),
            "the remainder must account for what `limit` removed: {text}"
        );
    }

    /// Three tiers, and the result says which one answered. A prefix or spelling match is a guess
    /// about what the caller meant, and a model that cannot tell it from an exact hit will report
    /// the guess as a recalled fact.
    #[tokio::test]
    async fn test_search_names_the_tier_that_answered() {
        let memories = store().await;
        MemoryWriteTool {
            memories: memories.clone(),
        }
        .execute(
            serde_json::json!({
                "name": "alice-timezone",
                "description": "alice works from Tokyo",
                "body": "Ten to seven, JST."
            }),
            CancellationToken::new(),
        )
        .await
        .expect("write");
        let search = MemorySearchTool { memories };

        // Exact: no caveat.
        let exact = output_text(
            &search
                .execute(
                    serde_json::json!({"queries": ["Tokyo"]}),
                    CancellationToken::new(),
                )
                .await
                .expect("search"),
        );
        assert!(exact.contains("alice-timezone"), "{exact}");
        assert!(!exact.contains("No exact matches"), "{exact}");

        // Prefix: a truncated word, reported as a near-miss.
        let prefix = output_text(
            &search
                .execute(
                    serde_json::json!({"queries": ["Tok"]}),
                    CancellationToken::new(),
                )
                .await
                .expect("search"),
        );
        assert!(prefix.contains("alice-timezone"), "{prefix}");
        assert!(prefix.contains("No exact matches"), "{prefix}");

        // Spelling: a genuine typo that no prefix covers, reported as possibly unrelated.
        let fuzzy = output_text(
            &search
                .execute(
                    serde_json::json!({"queries": ["Tokoy"]}),
                    CancellationToken::new(),
                )
                .await
                .expect("search"),
        );
        assert!(fuzzy.contains("alice-timezone"), "{fuzzy}");
        assert!(fuzzy.contains("closest memory names"), "{fuzzy}");

        // And a miss says what to do next rather than just failing.
        let miss = output_text(
            &search
                .execute(
                    serde_json::json!({"queries": ["xylophone"]}),
                    CancellationToken::new(),
                )
                .await
                .expect("search"),
        );
        assert!(miss.contains("No memories matched"), "{miss}");
        assert!(miss.contains("several phrasings"), "{miss}");
    }

    /// A half-remembered name is the common miss at scale. "No such memory" on its own ends the
    /// line; pointing at the near-miss is the same recovery an unknown tool name gets.
    #[tokio::test]
    async fn test_read_suggests_a_near_miss_name() {
        let memories = store().await;
        MemoryWriteTool {
            memories: memories.clone(),
        }
        .execute(
            serde_json::json!({"name": "alice-timezone", "description": "JST"}),
            CancellationToken::new(),
        )
        .await
        .expect("write");

        let read = MemoryReadTool { memories };
        let error = read
            .execute(
                serde_json::json!({"name": "alice-timezon"}),
                CancellationToken::new(),
            )
            .await
            .expect_err("a misspelled name is still a miss");
        let message = error.to_string();
        assert!(message.contains("no memory named"), "{message}");
        assert!(message.contains("alice-timezone"), "{message}");
    }

    /// Reading a memory raises it for next time, which is the counterweight to a priority the
    /// agent chose once and never revised.
    #[tokio::test]
    async fn test_reading_a_memory_records_it_and_lifts_its_rank() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
        };
        for name in ["read-often", "never-read"] {
            write
                .execute(
                    serde_json::json!({
                        "name": name,
                        "description": "deployment procedure",
                        "body": "Run the thing."
                    }),
                    CancellationToken::new(),
                )
                .await
                .expect("write");
        }

        let read = MemoryReadTool {
            memories: memories.clone(),
        };
        for _ in 0..5 {
            read.execute(
                serde_json::json!({"name": "read-often"}),
                CancellationToken::new(),
            )
            .await
            .expect("read");
        }

        let hit = MemorySearchTool { memories }
            .execute(
                serde_json::json!({"queries": ["deployment"], "limit": 1}),
                CancellationToken::new(),
            )
            .await
            .expect("search");
        let text = output_text(&hit);
        assert!(
            text.contains("read-often"),
            "usage must break a tie the priorities cannot: {text}"
        );
        assert!(text.contains("read 5x"), "{text}");
    }

    /// Advisory, never blocking. The failure worth preventing is the silent one, where a store
    /// grows a hundred near-copies because nothing ever mentioned the ninety-nine.
    #[tokio::test]
    async fn test_write_names_a_near_duplicate_without_refusing() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
        };
        write
            .execute(
                serde_json::json!({
                    "name": "alice-timezone",
                    "description": "alice works from Tokyo in JST"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("write");

        let output = write
            .execute(
                serde_json::json!({
                    "name": "alice-tz",
                    "description": "alice works from Tokyo in JST"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("a duplicate must still be written");
        let text = output_text(&output);
        assert!(text.contains("Saved memory 'alice-tz'"), "{text}");
        assert!(text.contains("alice-timezone"), "{text}");
        assert!(text.contains("very similar"), "{text}");

        // Take the agent's advice, then update the survivor. Rewriting a memory under its own
        // name is the thing the note asks for, so it must not then accuse the call of duplicating
        // itself -- which is a different check from "no duplicate exists", and the one that would
        // make the warning fire on every single update.
        MemoryDeleteTool {
            memories: memories.clone(),
        }
        .execute(
            serde_json::json!({"name": "alice-tz"}),
            CancellationToken::new(),
        )
        .await
        .expect("delete the duplicate");

        let output = write
            .execute(
                serde_json::json!({
                    "name": "alice-timezone",
                    "description": "alice works from Tokyo in JST",
                    "priority": 1
                }),
                CancellationToken::new(),
            )
            .await
            .expect("self-update");
        assert!(
            !output_text(&output).contains("very similar"),
            "{}",
            output_text(&output)
        );
    }

    /// A model emits several tool calls in one message and meka runs them concurrently, so two
    /// `memory_write`s in one turn both read the store *before* either lands: each sees no
    /// duplicate, and the check that exists to catch exactly this reports nothing.
    ///
    /// Found live, writing two identically-described memories in one turn. Not a rare shape: a
    /// compaction checkpoint saves several memories at once, which is when near-copies are most
    /// likely. Remove the write lock and this fails.
    #[tokio::test]
    async fn concurrent_writes_in_one_turn_still_notice_a_duplicate() {
        let memories = store().await;
        let write = Arc::new(MemoryWriteTool {
            memories: memories.clone(),
        });

        let call = |name: &'static str| {
            let write = write.clone();
            async move {
                write
                    .execute(
                        serde_json::json!({
                            "name": name,
                            "description": "Backups run nightly at 02:00 UTC"
                        }),
                        CancellationToken::new(),
                    )
                    .await
                    .expect("write")
            }
        };
        let (first, second) = tokio::join!(call("backup-window"), call("backup-timing"));

        // Both are written; the check never blocks. Whichever the lock ordered second must have
        // seen the first, so exactly one of the two results names a duplicate.
        let noted = [output_text(&first), output_text(&second)]
            .iter()
            .filter(|text| text.contains("already says something very similar"))
            .count();
        assert_eq!(
            noted,
            1,
            "one of the two concurrent writes must notice the other:\n{}\n{}",
            output_text(&first),
            output_text(&second)
        );
    }

    /// A deleted name must not bequeath its standing to whatever is written under it next.
    #[tokio::test]
    async fn test_delete_clears_the_usage_counters() {
        let memories = store().await;
        MemoryWriteTool {
            memories: memories.clone(),
        }
        .execute(
            serde_json::json!({"name": "transient", "description": "a note"}),
            CancellationToken::new(),
        )
        .await
        .expect("write");
        MemoryReadTool {
            memories: memories.clone(),
        }
        .execute(
            serde_json::json!({"name": "transient"}),
            CancellationToken::new(),
        )
        .await
        .expect("read");

        assert_eq!(
            memories
                .get("transient")
                .await
                .expect("get")
                .map(|entry| entry.read_count),
            Some(1)
        );

        MemoryDeleteTool {
            memories: memories.clone(),
        }
        .execute(
            serde_json::json!({"name": "transient"}),
            CancellationToken::new(),
        )
        .await
        .expect("delete");

        // Written again under the same name: the counter is a column on the row, so it went with
        // it. When it lived in a second table keyed by name, the new memory inherited the old
        // one's standing.
        MemoryWriteTool {
            memories: memories.clone(),
        }
        .execute(
            serde_json::json!({"name": "transient", "description": "a different note"}),
            CancellationToken::new(),
        )
        .await
        .expect("rewrite");
        assert_eq!(
            memories
                .get("transient")
                .await
                .expect("get")
                .map(|entry| entry.read_count),
            Some(0)
        );
    }

    #[tokio::test]
    async fn test_all_memory_tools_gate_at_read() {
        let memories = store().await;
        // Read permission is what mekabridge runs at; anything stricter means an agent that can
        // never remember.
        assert_eq!(
            MemoryWriteTool {
                memories: memories.clone()
            }
            .required_permission(),
            Permission::Read
        );
        assert_eq!(
            MemoryReadTool {
                memories: memories.clone()
            }
            .required_permission(),
            Permission::Read
        );
        assert_eq!(
            MemorySearchTool {
                memories: memories.clone()
            }
            .required_permission(),
            Permission::Read
        );
        assert_eq!(
            MemoryDeleteTool { memories }.required_permission(),
            Permission::Read
        );
    }

    /// A query that is a longer derived form than the stored word must still reach it.
    ///
    /// Found live: a model searched `deployment` against a memory tagged `deploy` whose body says
    /// `Deploys`, and got "No memories matched" -- then said it did not believe the result, which
    /// was the correct call. Every tier missed. SQLite's porter strips `-s`/`-ing`/`-ed` but not
    /// `-ment`; `deployment*` puts the star on the wrong end; `LIKE '%deployment%'` cannot match a
    /// shorter word; and the edit distance is 4 against a threshold of 3.
    #[tokio::test]
    async fn test_a_query_longer_than_the_stored_word_still_finds_it() {
        let memories = store().await;
        MemoryWriteTool {
            memories: memories.clone(),
        }
        .execute(
            serde_json::json!({
                "name": "deploy-host",
                "description": "mekabridge runs on nas.lan behind Caddy",
                "tags": ["deploy", "infra"],
                "body": "Deploys go out on Fridays.",
            }),
            CancellationToken::new(),
        )
        .await
        .expect("write");

        for query in ["deployment", "deployments"] {
            let out = output_text(
                &MemorySearchTool {
                    memories: memories.clone(),
                }
                .execute(
                    serde_json::json!({"queries": [query]}),
                    CancellationToken::new(),
                )
                .await
                .expect("search"),
            );
            assert!(out.contains("deploy-host"), "{query} found nothing: {out}");
            // Reported as a near-miss, not as an exact hit: it is still a guess about what the
            // caller meant.
            assert!(
                out.contains("prefix matches"),
                "{query} must say the prefix tier answered: {out}"
            );
        }
    }

    /// A phrase inside a body written in a script the tokenizer does not segment must be findable.
    ///
    /// `unicode61` splits only on non-alphanumerics, so a contiguous CJK run is one token: `深圳`
    /// matches neither exactly nor by prefix against a body containing it, and edit distance is far
    /// past threshold. The regex `memory_search` this replaced matched every line of the file,
    /// bodies included, so without the substring tier this is a plain regression -- and the earlier
    /// rescue covered names and descriptions only, which is the half a `Memory` happens to carry.
    #[tokio::test]
    async fn test_a_phrase_inside_a_cjk_body_is_still_found() {
        let memories = store().await;
        MemoryWriteTool {
            memories: memories.clone(),
        }
        .execute(
            serde_json::json!({
                "name": "office",
                "description": "office location note",
                "body": "办公室在深圳南山区的科技园",
            }),
            CancellationToken::new(),
        )
        .await
        .expect("write");

        for query in ["深圳", "南山区", "科技园"] {
            let out = output_text(
                &MemorySearchTool {
                    memories: memories.clone(),
                }
                .execute(
                    serde_json::json!({"queries": [query]}),
                    CancellationToken::new(),
                )
                .await
                .expect("search"),
            );
            assert!(out.contains("office"), "{query} found nothing: {out}");
            assert!(
                out.contains("literal substring"),
                "{query} must say which tier answered: {out}"
            );
        }
    }

    /// A `limit` the model spelled as a string or a float is honoured, and one that is neither a
    /// number nor a numeric string is refused rather than silently replaced by the default.
    ///
    /// `as_u64` returned `None` for `"3"`, `3.0` and `-1` alike and `unwrap_or` substituted 10: the
    /// call asked for three results, received ten, and nothing in the output said so.
    #[tokio::test]
    async fn test_a_limit_of_the_wrong_type_is_coerced_or_refused_never_ignored() {
        let memories = store().await;
        for index in 0..8 {
            MemoryWriteTool {
                memories: memories.clone(),
            }
            .execute(
                serde_json::json!({
                    "name": format!("note-{index}"),
                    "description": "a deployment note",
                }),
                CancellationToken::new(),
            )
            .await
            .expect("write");
        }
        let search = MemorySearchTool {
            memories: memories.clone(),
        };
        for limit in [
            serde_json::json!(2),
            serde_json::json!("2"),
            serde_json::json!(2.0),
        ] {
            let out = output_text(
                &search
                    .execute(
                        serde_json::json!({"queries": ["deployment"], "limit": limit}),
                        CancellationToken::new(),
                    )
                    .await
                    .expect("search"),
            );
            assert_eq!(
                out.matches("- **note-").count(),
                2,
                "limit {limit} was not honoured: {out}"
            );
        }
        let error = search
            .execute(
                serde_json::json!({"queries": ["deployment"], "limit": "many"}),
                CancellationToken::new(),
            )
            .await
            .expect_err("a non-numeric limit must be refused, not swapped for the default");
        assert!(error.to_string().contains("'limit'"), "{error}");
    }

    /// A malformed `tags` or `body` argument is refused rather than read as a request to clear.
    ///
    /// `filter_map` dropped every non-string element, so `tags: [1, 2]` collapsed to `[]` -- which
    /// *is* the documented "clear them" signal, so a malformed argument erased labels the caller
    /// never mentioned. `body: ["a", "b"]` fell down the omit-to-keep path and wrote an empty body
    /// while reporting plain success.
    #[tokio::test]
    async fn test_a_malformed_tags_or_body_argument_is_refused_not_read_as_a_clear() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
        };
        write
            .execute(
                serde_json::json!({
                    "name": "note",
                    "description": "a fact",
                    "tags": ["infra", "deploy"],
                    "body": "the detail",
                }),
                CancellationToken::new(),
            )
            .await
            .expect("write");

        for bad in [
            serde_json::json!({"name": "note", "description": "a fact", "tags": [1, 2]}),
            serde_json::json!({"name": "note", "description": "a fact", "tags": 7}),
            serde_json::json!({"name": "note", "description": "a fact", "body": ["a", "b"]}),
        ] {
            write
                .execute(bad.clone(), CancellationToken::new())
                .await
                .expect_err("must refuse rather than silently clear");
        }

        let stored = memories
            .get("note")
            .await
            .expect("get")
            .expect("still there");
        assert_eq!(stored.tags, ["deploy", "infra"], "tags must survive");
        assert_eq!(
            stored.body.as_deref().map(str::trim),
            Some("the detail"),
            "the body must survive"
        );
    }

    /// The two truncation lines have to name the constraint that actually bound, and hedge a count
    /// that is a floor.
    ///
    /// "raise `limit`" was printed even when `limit` was already at its maximum and the byte budget
    /// had done the cutting, which hands the model a remedy that provably returns the identical
    /// result. Verified through the byte budget, which is the reachable half of the pair.
    #[tokio::test]
    async fn test_a_truncated_result_names_the_limit_that_actually_bound() {
        let memories = store().await;
        for index in 0..MAX_SEARCH_LIMIT {
            MemoryWriteTool {
                memories: memories.clone(),
            }
            .execute(
                serde_json::json!({
                    "name": format!("note-{index}"),
                    "description": "a deployment note",
                    "body": "deployment ".repeat(70),
                }),
                CancellationToken::new(),
            )
            .await
            .expect("write");
        }
        let out = output_text(
            &MemorySearchTool {
                memories: memories.clone(),
            }
            .execute(
                serde_json::json!({"queries": ["deployment"], "limit": MAX_SEARCH_LIMIT}),
                CancellationToken::new(),
            )
            .await
            .expect("search"),
        );
        assert!(
            out.contains("further match(es) not shown"),
            "the budget cut entries and must say so: {out}"
        );
        assert!(
            out.contains("hit its size limit"),
            "and must not blame `limit`, which is already at its maximum: {out}"
        );
        assert!(
            !out.contains("raise `limit`"),
            "the remedy offered must be one that could work: {out}"
        );
    }

    /// One memory with a pathological description or body cannot spend the whole result budget.
    ///
    /// The always-emit-the-first-entry rule exists so a single oversized memory stays visible; it
    /// was instead the hole an unbounded one passed through. `snippet()` bounds by *tokens*, and a
    /// body that is one token comes back whole -- measured at 200 KB from a 6 KB ceiling.
    #[tokio::test]
    async fn test_one_oversized_memory_cannot_blow_the_result_budget() {
        let memories = store().await;
        MemoryWriteTool {
            memories: memories.clone(),
        }
        .execute(
            serde_json::json!({
                "name": "huge",
                // One token, so `snippet()` has nothing to cut on, and far past the inline cap.
                "description": format!("deployment {}", "a".repeat(60_000)),
                "body": "b".repeat(200_000),
            }),
            CancellationToken::new(),
        )
        .await
        .expect("write");

        let out = output_text(
            &MemorySearchTool {
                memories: memories.clone(),
            }
            .execute(
                serde_json::json!({"queries": ["deployment"]}),
                CancellationToken::new(),
            )
            .await
            .expect("search"),
        );
        assert!(out.contains("huge"), "the memory must still be reported");
        assert!(
            out.len() < SEARCH_RESULT_MAX_BYTES * 2,
            "one memory spent {} bytes against a {} ceiling",
            out.len(),
            SEARCH_RESULT_MAX_BYTES
        );
    }
}
