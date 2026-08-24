//! `find_files` tool: glob-pattern file discovery.

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::{
    Tool, ToolOutput,
    util::{WalkBudget, WalkStop, redirects_to_scratchpad, require_str},
};
use crate::{
    error::{MekaError, Result},
    permission::Permission,
    provider::ToolDefinition,
};

/// Default inline result cap when the agent isn't redirecting to the scratchpad and didn't pass an
/// explicit `limit`. Single source of truth for the description and the runtime default.
const DEFAULT_INLINE_RESULTS: usize = 500;

/// What one walk produced, including the parts the caller has to disclose: a walk that was cut
/// short reports fewer matches than exist, which is only safe if the output says so.
struct FindOutcome {
    matches: Vec<String>,
    total: usize,
    unreadable: usize,
    timed_out: bool,
    budget_secs: u64,
}

pub(super) struct FindFilesTool {
    pub cwd: crate::workspace::SharedCwd,
    /// Extra workspace roots swept when the caller names no explicit `path`. Empty outside a
    /// multi-root ACP session.
    pub roots: crate::workspace::SharedRoots,
}

#[async_trait]
impl Tool for FindFilesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "find_files".to_string(),
            description: format!(
                "Find files matching a glob pattern (e.g., '**/*.rs', 'src/*.txt'). \
                 Avoid overly broad searches: scanning a large tree can take \
                 a long time and will hit many directories the user has no \
                 read permission for, producing noisy errors. Start with the \
                 smallest `path` and most specific pattern that plausibly \
                 contains the answer; if that returns nothing, widen the \
                 `path` by one level or loosen the pattern, and repeat. Only \
                 fall back to a tree-wide scan if targeted attempts have all \
                 failed. Inline results default to {} entries; pass `limit` to \
                 raise the cap or `scratchpad` to collect them all. \
                 Multiple independent find_files calls in one assistant message \
                 run in parallel.",
                DEFAULT_INLINE_RESULTS,
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match files against. Prefer narrow patterns over broad ones like `**/*`."
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in. Omit to search every workspace root (the working directory plus any additional roots listed in the environment context). Set it to narrow to the smallest subtree that can answer the question."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": format!(
                            "Maximum results to return. Defaults to {} when output is inline, \
                             unbounded when `scratchpad` is set. Pass an explicit value to \
                             override either default.",
                            DEFAULT_INLINE_RESULTS,
                        )
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the output to the scratchpad under this name instead of returning it inline."
                    }
                },
                "required": ["pattern"]
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
        cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let pattern = require_str(&input, "pattern", "find_files")?;
        // An explicit `path` searches exactly that tree, resolved against the per-session cwd so
        // the search runs in the right place regardless of where the process was launched. With no
        // `path`, sweep every workspace root: in a multi-root ACP workspace, searching only `cwd`
        // silently misses whole folders the user can see in their editor.
        let base_paths = match input["path"].as_str() {
            Some(raw) => vec![crate::workspace::resolve_against_cwd(&self.cwd, raw)],
            None => crate::workspace::glob_roots(&self.cwd, &self.roots),
        };
        let mut full_patterns: Vec<String> = Vec::with_capacity(base_paths.len());
        for base in &base_paths {
            // Normalised through the type rather than by trimming a character. The old code did
            // `trim_end_matches('/')`, which leaves a Windows `C:\work\` alone and yields
            // `C:\work\/*.md`, whose meaning then depends on `glob`'s separator handling rather
            // than on anything meka decided. Re-collecting the components drops a trailing
            // separator on both platforms. A trailing separator is reachable: `resolve_against_cwd`
            // preserves one, and an ACP client may send it in `additionalDirectories`.
            let base: std::path::PathBuf = base.components().collect();
            // `glob` takes a `&str` and there is no byte-oriented entry point, so a root that is
            // not valid UTF-8 cannot be searched. Refused loudly. Left to `to_string_lossy` it
            // became a path that does not exist, `glob` yielded nothing, and `find_files` reported
            // "No files found" -- a definitive answer to a question it never actually asked.
            let Some(base) = base.to_str() else {
                return Err(MekaError::ToolExecution {
                    tool_name: "find_files".to_string(),
                    message: format!(
                        "workspace root '{}' is not valid UTF-8, and the glob matcher cannot accept it. Pass an explicit `path` inside a root that is, or rename the directory.",
                        base.display()
                    ),
                });
            };
            // The base is escaped, the caller's `pattern` is not: a root is a literal directory,
            // and a client is free to have one named `2024*` or `notes[1]`. Without this those
            // parse as wildcards, silently widening the search past the named roots.
            full_patterns.push(format!("{}/{}", glob::Pattern::escape(base), pattern));
        }

        // Cap precedence:
        //   1. explicit `limit` parameter: honoured verbatim
        //   2. no limit + `scratchpad` set: unbounded (preserves the "collect everything" escape
        //      hatch)
        //   3. otherwise: DEFAULT_INLINE_RESULTS
        let explicit_limit = input
            .get("limit")
            .and_then(|value| value.as_u64())
            .map(|value| usize::try_from(value).unwrap_or(usize::MAX));
        let cap = match explicit_limit {
            Some(limit) => limit,
            None if redirects_to_scratchpad(&input) => usize::MAX,
            None => DEFAULT_INLINE_RESULTS,
        };

        // One budget for the whole call, not one per root: `WalkBudget::new` stamps its deadline at
        // construction, so a per-root budget would silently give a four-root workspace four times
        // the ceiling. `run_walk` walks the roots in order under this single deadline.
        let budget = WalkBudget::new(cancellation.clone());
        let walk = tokio::task::spawn_blocking(move || run_walk(&full_patterns, cap, &budget));

        // Race the walk against the token rather than just awaiting it. The walk checks the same
        // token itself, but `glob`'s iterator does its directory reads inside `next()`, so a walk
        // that finds neither a match nor an error for a long stretch sits between checks and would
        // otherwise hold the turn open. Returning here detaches that thread; its own budget check
        // stops it shortly after.
        let outcome = tokio::select! {
            joined = walk => joined.map_err(|error| MekaError::ToolExecution {
                tool_name: "find_files".to_string(),
                message: format!("task join error: {}", error),
            })??,
            _ = cancellation.cancelled() => return Err(MekaError::Interrupted),
        };

        Ok(ToolOutput::text(render_outcome(&outcome, cap), false))
    }
}

/// The blocking half of `find_files`. Consults `budget` once per entry the glob iterator yields,
/// which is what makes an over-broad walk stoppable at all: nothing outside this loop can end it,
/// because a `spawn_blocking` task that has started cannot be aborted.
///
/// `full_patterns` holds one rooted glob per workspace root, walked in order under a single shared
/// `budget`. Counts accumulate across roots so the caller's cap and truncation message describe the
/// whole search rather than its last leg, and a timeout part-way through root two still reports the
/// matches root one produced.
fn run_walk(full_patterns: &[String], cap: usize, budget: &WalkBudget) -> Result<FindOutcome> {
    let mut matches: Vec<String> = Vec::new();
    // Roots are kept even when one nests inside another (see `glob_roots`), so a pattern that
    // crosses directories can match the same file from two of them: with roots `/work` and
    // `/work/main`, `**/*.md` finds `/work/main/README.md` under both. Deduplicate here rather
    // than by pruning roots, which is what would lose files. Skipped for a single pattern, which
    // cannot repeat a path, so the common case allocates nothing.
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    // Total continues past the storage cap so the truncation message can report the real number of
    // matches. Note that the cap bounds only what is stored: a pattern that matches nothing never
    // reaches it, so the cap is not a bound on the walk. `budget` is.
    let mut total: usize = 0;
    // A pattern rooted high in the tree crosses directories the user cannot read, one `GlobError`
    // each. Logging every one at `warn` is what turned a `/`-rooted walk into gigabytes of log
    // output, so they are counted here and reported once.
    let mut unreadable: usize = 0;
    let mut timed_out = false;

    'roots: for full_pattern in full_patterns {
        // Checked here as well as per entry, for the same reason `walk_directory` checks at the top
        // of its outer loop: a root whose glob yields nothing never enters the inner loop, so a run
        // of empty or missing roots would advance through the whole list without consulting the
        // budget once, ignoring both the deadline and a `session/cancel`.
        match budget.check() {
            Some(WalkStop::Cancelled) => return Err(MekaError::Interrupted),
            Some(WalkStop::TimedOut) => {
                timed_out = true;
                break 'roots;
            }
            None => {}
        }

        // Escaping the base means the root can no longer make the combined pattern invalid, so a
        // compile failure here is the caller's `pattern` and every root would fail the same way.
        // That is a caller mistake worth reporting loudly: degrading it to "no files found" would
        // hand the model a definitive-looking answer to a question that was never asked.
        let paths = glob::glob(full_pattern).map_err(|error| MekaError::ToolExecution {
            tool_name: "find_files".to_string(),
            message: format!("invalid glob pattern '{}': {}", full_pattern, error),
        })?;

        for entry in paths {
            match budget.check() {
                Some(WalkStop::Cancelled) => return Err(MekaError::Interrupted),
                Some(WalkStop::TimedOut) => {
                    timed_out = true;
                    break 'roots;
                }
                None => {}
            }
            match entry {
                Ok(path) => {
                    let rendered = path.display().to_string();
                    if full_patterns.len() > 1 && !seen.insert(rendered.clone()) {
                        continue;
                    }
                    total += 1;
                    if matches.len() < cap {
                        matches.push(rendered);
                    }
                }
                Err(error) => {
                    unreadable += 1;
                    tracing::debug!("glob error: {}", error);
                }
            }
        }
    }

    Ok(FindOutcome {
        matches,
        total,
        unreadable,
        timed_out,
        budget_secs: budget.budget_secs(),
    })
}

/// Render the walk for the model. A truncated, timed-out, or error-riddled walk has to say so even
/// when it found nothing: a bare "No files found" on a search that never finished reads as a
/// definitive answer, and the model acts on it as one.
fn render_outcome(outcome: &FindOutcome, cap: usize) -> String {
    let mut notes: Vec<String> = Vec::new();
    if outcome.total > outcome.matches.len() {
        notes.push(format!(
            "showed first {} of {} matches; refine `pattern` to narrow, pass `limit: <n>` to \
             raise the cap, or pass `scratchpad: \"name\"` to collect them all",
            cap, outcome.total,
        ));
    }
    if outcome.timed_out {
        notes.push(format!(
            "search was still running after {}s and was stopped, so this list is incomplete: \
             narrow `path` to a smaller subtree or make `pattern` more specific",
            outcome.budget_secs,
        ));
    }
    if outcome.unreadable > 0 {
        notes.push(format!(
            "{} path(s) could not be read and were skipped",
            outcome.unreadable,
        ));
    }

    let body = if outcome.matches.is_empty() {
        "No files found matching the pattern.".to_string()
    } else {
        outcome.matches.join("\n")
    };
    if notes.is_empty() {
        body
    } else {
        format!("{}\n\n... ({})", body, notes.join("; "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::tests::text_content;

    #[tokio::test]
    async fn test_find_files() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        std::fs::write(temp_dir.path().join("a.txt"), "").expect("failed");
        std::fs::write(temp_dir.path().join("b.txt"), "").expect("failed");
        std::fs::write(temp_dir.path().join("c.rs"), "").expect("failed");

        let tool = FindFilesTool {
            cwd: crate::workspace::test_cwd(),
            roots: crate::workspace::test_roots(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "*.txt",
                    "path": temp_dir.path().to_str().expect("path")
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("a.txt"));
        assert!(text_content(&result).contains("b.txt"));
        assert!(!text_content(&result).contains("c.rs"));
    }

    /// End-to-end guard for the root set `execute` picks. [`crate::workspace::search_roots`] and
    /// [`crate::workspace::glob_roots`] have identical signatures, so swapping one for the other
    /// still compiles and the `run_walk` tests (which take patterns, not roots) stay green.
    /// Only a tool-level search of a workspace whose roots nest catches it.
    #[tokio::test]
    async fn test_find_files_sweeps_cwd_when_an_additional_root_contains_it() {
        let top = tempfile::tempdir().expect("tempdir");
        let nested = top.path().join("main");
        std::fs::create_dir(&nested).expect("mkdir");
        std::fs::write(nested.join("README.md"), "").expect("write");
        std::fs::write(top.path().join("top.md"), "").expect("write");

        let tool = FindFilesTool {
            cwd: std::sync::Arc::new(std::sync::RwLock::new(nested.clone())),
            roots: std::sync::Arc::new(std::sync::RwLock::new(vec![top.path().to_path_buf()])),
        };
        let result = tool
            .execute(
                serde_json::json!({ "pattern": "*.md" }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        let text = text_content(&result);
        assert!(
            text.contains("README.md"),
            "a file in cwd must be found even when the workspace also names cwd's parent; got: {}",
            text,
        );
        assert!(text.contains("top.md"), "got: {}", text);
    }

    #[tokio::test]
    async fn test_find_files_inline_default_cap_reports_total() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        for i in 0..600 {
            std::fs::write(temp_dir.path().join(format!("f{}.txt", i)), "").expect("write");
        }

        let tool = FindFilesTool {
            cwd: crate::workspace::test_cwd(),
            roots: crate::workspace::test_roots(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "*.txt",
                    "path": temp_dir.path().to_str().expect("path")
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        let text = text_content(&result);
        assert!(
            text.contains("showed first 500 of 600 matches"),
            "expected real total in truncation message, got: {:.300}",
            text,
        );
    }

    #[tokio::test]
    async fn test_find_files_limit_overrides_default() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        for i in 0..600 {
            std::fs::write(temp_dir.path().join(format!("f{}.txt", i)), "").expect("write");
        }

        let tool = FindFilesTool {
            cwd: crate::workspace::test_cwd(),
            roots: crate::workspace::test_roots(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "*.txt",
                    "path": temp_dir.path().to_str().expect("path"),
                    "limit": 100
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        let text = text_content(&result);
        assert!(text.contains("showed first 100 of 600 matches"));
        // 100 entries plus the trailing truncation line.
        let path_lines = text.lines().filter(|line| line.ends_with(".txt")).count();
        assert_eq!(path_lines, 100);
    }

    #[tokio::test]
    async fn test_find_files_scratchpad_lifts_cap() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        for i in 0..600 {
            std::fs::write(temp_dir.path().join(format!("f{}.txt", i)), "").expect("write");
        }

        let tool = FindFilesTool {
            cwd: crate::workspace::test_cwd(),
            roots: crate::workspace::test_roots(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "*.txt",
                    "path": temp_dir.path().to_str().expect("path"),
                    "scratchpad": "paths"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        let text = text_content(&result);
        assert!(
            !text.contains("showed first"),
            "expected no truncation marker when scratchpad set, got: {:.200}...",
            text,
        );
        let line_count = text.lines().filter(|l| l.ends_with(".txt")).count();
        assert!(
            line_count >= 600,
            "expected >= 600 entries, got {}",
            line_count
        );
    }

    #[tokio::test]
    async fn test_find_files_explicit_limit_with_scratchpad_caps() {
        // Regression: an explicit `limit` should beat the scratchpad "unbounded" default; the
        // agent might legitimately want a bounded scratchpad collection.
        let temp_dir = tempfile::tempdir().expect("tempdir");
        for i in 0..600 {
            std::fs::write(temp_dir.path().join(format!("f{}.txt", i)), "").expect("write");
        }

        let tool = FindFilesTool {
            cwd: crate::workspace::test_cwd(),
            roots: crate::workspace::test_roots(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "*.txt",
                    "path": temp_dir.path().to_str().expect("path"),
                    "scratchpad": "paths",
                    "limit": 50
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        let text = text_content(&result);
        assert!(text.contains("showed first 50 of 600 matches"));
    }

    #[tokio::test]
    async fn test_find_files_cancelled_walk_is_interrupted() {
        // Regression: the tool used to ignore its cancellation token entirely, so an over-broad
        // walk could not be stopped by Ctrl+C, by ACP `session/cancel`, or by anything else.
        let temp_dir = tempfile::tempdir().expect("tempdir");
        for i in 0..50 {
            std::fs::write(temp_dir.path().join(format!("f{}.txt", i)), "").expect("write");
        }

        let tool = FindFilesTool {
            cwd: crate::workspace::test_cwd(),
            roots: crate::workspace::test_roots(),
        };
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let error = tool
            .execute(
                serde_json::json!({
                    "pattern": "*.txt",
                    "path": temp_dir.path().to_str().expect("path")
                }),
                cancellation,
            )
            .await
            .expect_err("a cancelled turn must not run the walk to completion");
        assert!(matches!(error, MekaError::Interrupted), "got: {}", error);
    }

    #[test]
    fn test_run_walk_stops_on_exhausted_budget() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        for i in 0..50 {
            std::fs::write(temp_dir.path().join(format!("f{}.txt", i)), "").expect("write");
        }
        let pattern = format!("{}/*.txt", temp_dir.path().to_string_lossy());

        let budget =
            WalkBudget::with_budget(CancellationToken::new(), std::time::Duration::from_secs(0));
        let outcome = run_walk(&[pattern], 500, &budget).expect("walk should return, not error");
        assert!(outcome.timed_out, "expired budget must stop the walk");
    }

    /// A malformed `pattern` is the caller's mistake and must surface as an error. Returning
    /// "no files found" instead would read as a definitive answer to a search that never ran.
    #[test]
    fn test_run_walk_errors_on_a_malformed_caller_pattern() {
        let temp = tempfile::tempdir().expect("tempdir");
        let pattern = format!(
            "{}/{}",
            glob::Pattern::escape(temp.path().to_string_lossy().trim_end_matches('/')),
            "[unterminated",
        );
        let budget = WalkBudget::new(CancellationToken::new());
        let Err(error) = run_walk(&[pattern], 500, &budget) else {
            panic!("a malformed caller pattern must not be swallowed");
        };
        assert!(
            format!("{}", error).contains("invalid glob pattern"),
            "got: {}",
            error
        );
    }

    /// A root is a literal directory. Without escaping, one named `notes[1]` would be read as a
    /// character class and match a sibling named `notes1` instead of the root the client asked for.
    ///
    /// Brackets rather than `*`: Windows reserves `*` in filenames, so the fixture could not even
    /// be created there, and a root containing one is unreachable on that platform anyway.
    /// Brackets are legal on both, so this keeps the escaping covered everywhere CI runs.
    #[test]
    fn test_glob_metacharacters_in_a_root_are_literal() {
        let temp = tempfile::tempdir().expect("tempdir");
        let literal = temp.path().join("notes[1]");
        let decoy = temp.path().join("notes1");
        std::fs::create_dir_all(&literal).expect("mkdir literal");
        std::fs::create_dir_all(&decoy).expect("mkdir decoy");
        std::fs::write(literal.join("a.txt"), "").expect("write");
        std::fs::write(decoy.join("b.txt"), "").expect("write");

        let pattern = format!(
            "{}/{}",
            glob::Pattern::escape(literal.to_string_lossy().trim_end_matches('/')),
            "*.txt",
        );
        let budget = WalkBudget::new(CancellationToken::new());
        let outcome = run_walk(&[pattern], 500, &budget).expect("walk");

        assert_eq!(outcome.total, 1, "must not match the decoy directory");
        assert!(outcome.matches[0].ends_with("a.txt"));
    }

    /// Multi-root walks accumulate into one outcome rather than reporting only the last root.
    #[test]
    fn test_run_walk_accumulates_across_roots() {
        let first = tempfile::tempdir().expect("tempdir");
        let second = tempfile::tempdir().expect("tempdir");
        std::fs::write(first.path().join("a.txt"), "").expect("write");
        std::fs::write(second.path().join("b.txt"), "").expect("write");
        let patterns = vec![
            format!("{}/*.txt", first.path().to_string_lossy()),
            format!("{}/*.txt", second.path().to_string_lossy()),
        ];

        let budget = WalkBudget::new(CancellationToken::new());
        let outcome = run_walk(&patterns, 500, &budget).expect("walk should return, not error");

        assert_eq!(outcome.total, 2, "both roots must be searched");
        assert!(outcome.matches.iter().any(|m| m.ends_with("a.txt")));
        assert!(outcome.matches.iter().any(|m| m.ends_with("b.txt")));
        assert!(!outcome.timed_out);
    }

    /// A file under `cwd` must still be found when the workspace also names an ancestor of `cwd`.
    /// `find_files` anchors the caller's pattern at each root and a glob's `*` does not cross `/`,
    /// so pruning the nested root (which is right for a descending walk, and what `search_roots`
    /// does) would answer `*.md` from the ancestor alone and report the file as missing.
    #[test]
    fn test_run_walk_finds_files_under_a_root_nested_in_another() {
        let top = tempfile::tempdir().expect("tempdir");
        let nested = top.path().join("main");
        std::fs::create_dir(&nested).expect("mkdir");
        std::fs::write(top.path().join("top.md"), "").expect("write");
        std::fs::write(nested.join("README.md"), "").expect("write");

        // The order `glob_roots` produces for cwd = <top>/main with an additional root of <top>.
        let patterns = vec![
            format!("{}/*.md", nested.to_string_lossy()),
            format!("{}/*.md", top.path().to_string_lossy()),
        ];
        let budget = WalkBudget::new(CancellationToken::new());
        let outcome = run_walk(&patterns, 500, &budget).expect("walk should return, not error");

        assert!(
            outcome.matches.iter().any(|m| m.ends_with("README.md")),
            "the file under the nested root must be found; got: {:?}",
            outcome.matches,
        );
        assert!(outcome.matches.iter().any(|m| m.ends_with("top.md")));
        assert_eq!(outcome.total, 2);
    }

    /// Keeping nested roots means a directory-crossing pattern can match one file from two of them.
    /// The dedupe is what makes that safe; without it the file is listed twice and burns two slots
    /// of the result cap.
    #[test]
    fn test_run_walk_reports_a_doubly_matched_file_once() {
        let top = tempfile::tempdir().expect("tempdir");
        let nested = top.path().join("main");
        std::fs::create_dir(&nested).expect("mkdir");
        std::fs::write(nested.join("README.md"), "").expect("write");

        let patterns = vec![
            format!("{}/**/*.md", top.path().to_string_lossy()),
            format!("{}/**/*.md", nested.to_string_lossy()),
        ];
        let budget = WalkBudget::new(CancellationToken::new());
        let outcome = run_walk(&patterns, 500, &budget).expect("walk should return, not error");

        assert_eq!(
            outcome.matches.len(),
            1,
            "a file reachable from two roots must be listed once; got: {:?}",
            outcome.matches,
        );
        assert_eq!(outcome.total, 1, "and must not be double-counted");
    }

    /// The budget spans the whole call rather than resetting per root. A budget already spent when
    /// the walk begins must stop it at the first root, not grant each root a fresh deadline: that
    /// is what would silently multiply the 60s ceiling by the workspace's root count.
    #[test]
    fn test_run_walk_shares_one_budget_across_roots() {
        let first = tempfile::tempdir().expect("tempdir");
        let second = tempfile::tempdir().expect("tempdir");
        std::fs::write(first.path().join("a.txt"), "").expect("write");
        std::fs::write(second.path().join("b.txt"), "").expect("write");
        let patterns = vec![
            format!("{}/*.txt", first.path().to_string_lossy()),
            format!("{}/*.txt", second.path().to_string_lossy()),
        ];

        let budget =
            WalkBudget::with_budget(CancellationToken::new(), std::time::Duration::from_secs(0));
        let outcome = run_walk(&patterns, 500, &budget).expect("walk should return, not error");

        assert!(outcome.timed_out, "expired budget must stop the walk");
        assert_eq!(
            outcome.total, 0,
            "an expired budget must not let a later root start a fresh deadline"
        );
    }

    #[test]
    fn test_render_outcome_discloses_timeout_with_no_matches() {
        // The dangerous shape: a walk that was cut short found nothing, and saying only "No files
        // found" would report that as a definitive answer.
        let outcome = FindOutcome {
            matches: Vec::new(),
            total: 0,
            unreadable: 3,
            timed_out: true,
            budget_secs: 60,
        };
        let rendered = render_outcome(&outcome, DEFAULT_INLINE_RESULTS);
        assert!(rendered.contains("No files found"), "got: {}", rendered);
        assert!(rendered.contains("incomplete"), "got: {}", rendered);
        assert!(rendered.contains("after 60s"), "got: {}", rendered);
        assert!(
            rendered.contains("3 path(s) could not be read"),
            "got: {}",
            rendered
        );
    }

    #[test]
    fn test_render_outcome_clean_walk_has_no_notes() {
        let outcome = FindOutcome {
            matches: vec!["/tmp/a.txt".to_string()],
            total: 1,
            unreadable: 0,
            timed_out: false,
            budget_secs: 60,
        };
        assert_eq!(
            render_outcome(&outcome, DEFAULT_INLINE_RESULTS),
            "/tmp/a.txt"
        );
    }
}
