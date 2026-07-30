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
    pub cwd: crate::agent::SharedCwd,
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
                        "description": "Directory to search in. Defaults to current directory. Prefer the smallest subtree that can answer the question."
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
        // Resolve the optional `path` against the agent's per-session cwd so the search runs in the
        // right tree regardless of where the process was launched. Absolute paths pass through
        // unchanged.
        let base_path = input["path"]
            .as_str()
            .map(|raw| crate::agent::resolve_against_cwd(&self.cwd, raw))
            .unwrap_or_else(|| crate::agent::cwd_snapshot(&self.cwd));
        let full_pattern = format!(
            "{}/{}",
            base_path.to_string_lossy().trim_end_matches('/'),
            pattern,
        );

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

        let budget = WalkBudget::new(cancellation.clone());
        let walk = tokio::task::spawn_blocking(move || run_walk(&full_pattern, cap, &budget));

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
fn run_walk(full_pattern: &str, cap: usize, budget: &WalkBudget) -> Result<FindOutcome> {
    let mut matches: Vec<String> = Vec::new();
    // Total continues past the storage cap so the truncation message can report the real number of
    // matches. Note that the cap bounds only what is stored: a pattern that matches nothing never
    // reaches it, so the cap is not a bound on the walk. `budget` is.
    let mut total: usize = 0;
    // A pattern rooted high in the tree crosses directories the user cannot read, one `GlobError`
    // each. Logging every one at `warn` is what turned a `/`-rooted walk into gigabytes of log
    // output, so they are counted here and reported once.
    let mut unreadable: usize = 0;
    let mut timed_out = false;

    let paths = glob::glob(full_pattern).map_err(|error| MekaError::ToolExecution {
        tool_name: "find_files".to_string(),
        message: format!("invalid glob pattern '{}': {}", full_pattern, error),
    })?;

    for entry in paths {
        match budget.check() {
            Some(WalkStop::Cancelled) => return Err(MekaError::Interrupted),
            Some(WalkStop::TimedOut) => {
                timed_out = true;
                break;
            }
            None => {}
        }
        match entry {
            Ok(path) => {
                total += 1;
                if matches.len() < cap {
                    matches.push(path.display().to_string());
                }
            }
            Err(error) => {
                unreadable += 1;
                tracing::debug!("glob error: {}", error);
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
            cwd: crate::agent::test_cwd(),
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

    #[tokio::test]
    async fn test_find_files_inline_default_cap_reports_total() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        for i in 0..600 {
            std::fs::write(temp_dir.path().join(format!("f{}.txt", i)), "").expect("write");
        }

        let tool = FindFilesTool {
            cwd: crate::agent::test_cwd(),
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
            cwd: crate::agent::test_cwd(),
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
            cwd: crate::agent::test_cwd(),
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
            cwd: crate::agent::test_cwd(),
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
            cwd: crate::agent::test_cwd(),
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
        let outcome = run_walk(&pattern, 500, &budget).expect("walk should return, not error");
        assert!(outcome.timed_out, "expired budget must stop the walk");
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
