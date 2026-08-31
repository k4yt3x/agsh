//! `search_contents` tool: ripgrep-style content search powered by the `grep-*` crates, with glob
//! filtering.
//!
//! It does **not** honour `.gitignore`, despite the name suggesting ripgrep's behaviour: the walk
//! here is a hand-rolled `read_dir` traversal whose only exclusions are dotfiles, `target` and
//! `node_modules`, and the `ignore` crate is not a dependency. Only the matcher comes from the
//! `grep-*` family.

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

/// Inline match cap when the agent isn't redirecting to the scratchpad. Single source of truth for
/// the description and the runtime cap.
const MAX_INLINE_MATCHES: usize = 100;

pub(super) struct SearchContentsTool {
    pub cwd: crate::workspace::SharedCwd,
    /// Extra workspace roots swept when the caller names no explicit `path`. Empty outside a
    /// multi-root ACP session.
    pub roots: crate::workspace::SharedRoots,
}

#[async_trait]
impl Tool for SearchContentsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "search_contents".to_string(),
            description: format!(
                "Search file contents using a regex pattern (powered by ripgrep). \
                 Avoid overly broad searches: scanning a large tree is slow \
                 and will hit many directories the user has no read permission \
                 for, producing noisy errors. Start with the smallest `path` \
                 and a tight `glob` filter that plausibly contains the match; \
                 if that returns nothing, widen the `path` by one level or \
                 loosen the `glob`, and repeat. Only fall back to a tree-wide \
                 scan if targeted attempts have all failed. Inline results are \
                 capped at {} matches; use the `scratchpad` parameter to \
                 collect an unbounded result set. Multiple independent \
                 search_contents calls in one assistant message run in parallel.",
                MAX_INLINE_MATCHES,
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in. Omit to search every workspace root (the working directory plus any additional roots listed in the environment context). Set it to narrow to the smallest subtree that can answer the question."
                    },
                    "glob": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g., '*.rs'). Strongly recommended when searching directories to avoid scanning unrelated files."
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
        let pattern = require_str(&input, "pattern", "search_contents")?;
        // An explicit `path` searches exactly that tree, resolved against the per-session cwd. With
        // no `path`, sweep every workspace root: in a multi-root ACP workspace, searching only
        // `cwd` silently misses whole folders the user can see in their editor.
        // Carried as `PathBuf` end to end. Rendering each root through `to_string_lossy` and
        // rebuilding it with `Path::new` replaced every non-UTF-8 byte with U+FFFD, so a working
        // directory whose name is not valid UTF-8 -- `mkdir $'proj\xff'` -- named a directory that
        // does not exist, and the tool reported the user's own cwd as missing under a spelling
        // they never typed.
        let search_paths: Vec<std::path::PathBuf> = match input["path"].as_str() {
            Some(raw) => vec![crate::workspace::resolve_against_cwd(&self.cwd, raw)],
            None => crate::workspace::search_roots(&self.cwd, &self.roots),
        };
        let file_glob = input["glob"].as_str().map(|s| s.to_string());
        // Cap match count for inline use; lift it when redirecting output to the scratchpad so the
        // agent can collect an unbounded result set.
        let max_results = if redirects_to_scratchpad(&input) {
            usize::MAX
        } else {
            MAX_INLINE_MATCHES
        };

        // One budget for the whole call, not one per root: `WalkBudget::new` stamps its deadline at
        // construction, so a per-root budget would silently multiply the ceiling by the root count.
        let budget = WalkBudget::new(cancellation.clone());
        let search = tokio::task::spawn_blocking(move || {
            search_with_grep(
                &pattern,
                &search_paths,
                file_glob.as_deref(),
                max_results,
                &budget,
            )
        });

        // Race the search against the token: a started `spawn_blocking` task cannot be aborted, so
        // awaiting it unconditionally would let a walk rooted high in the tree hold the turn open.
        // The search checks the same token per directory entry and stops on its own.
        let result = tokio::select! {
            joined = search => joined.map_err(|error| MekaError::ToolExecution {
                tool_name: "search_contents".to_string(),
                message: format!("task join error: {}", error),
            })??,
            _ = cancellation.cancelled() => return Err(MekaError::Interrupted),
        };

        Ok(ToolOutput::text(result, false))
    }
}

/// `search_paths` holds one workspace root per entry (or the single tree the caller named via
/// `path`), walked in order under a single shared `budget`. Results, the truncation cap, and the
/// timeout note all span the whole set, so the output describes the search rather than its last
/// leg.
fn search_with_grep(
    pattern: &str,
    search_paths: &[std::path::PathBuf],
    file_glob: Option<&str>,
    max_results: usize,
    budget: &WalkBudget,
) -> Result<String> {
    use grep_regex::RegexMatcherBuilder;

    // Cap the compiled-regex automaton and DFA cache sizes so an LLM-supplied pattern like
    // `a{10_000_000}` can't exhaust host memory during compile.
    const PATTERN_SIZE_LIMIT: usize = 1 << 20;
    const DFA_SIZE_LIMIT: usize = 1 << 20;

    let matcher = RegexMatcherBuilder::new()
        .size_limit(PATTERN_SIZE_LIMIT)
        .dfa_size_limit(DFA_SIZE_LIMIT)
        .build(pattern)
        .map_err(|error| MekaError::ToolExecution {
            tool_name: "search_contents".to_string(),
            message: format!("invalid or oversized regex '{}': {}", pattern, error),
        })?;

    let mut results = Vec::new();
    let mut timed_out = false;
    // A root that doesn't exist is skipped rather than fatal, because one stale entry in a
    // multi-root workspace shouldn't sink a search the other roots can answer. With a single
    // explicit `path` this reduces to today's behaviour exactly: nothing existed, so the error
    // below fires with the same message.
    let mut searched_any = false;
    // Compiled lazily and at most once. Deliberately inside the directory branch rather than
    // hoisted above the loop: hoisting would report an invalid `glob` for a path that doesn't
    // exist, or for a single file where the glob is irrelevant, changing single-root behaviour.
    let mut glob_pattern: Option<glob::Pattern> = None;
    // Roots left unsearched because the match cap filled up first. cwd is always root #1, so a
    // busy cwd would otherwise starve every other root and report only "truncated", which reads as
    // "the other folders had nothing" -- the exact failure multi-root support exists to prevent.
    let mut unsearched_roots = 0usize;
    let mut unreadable = 0usize;

    for search_path in search_paths {
        // Checked per root as well as inside `walk_directory`: a root that is a plain file, or that
        // doesn't exist, never reaches the walk, so a long list of them would advance without
        // consulting the budget once and ignore both the deadline and a `session/cancel`.
        match budget.check() {
            Some(WalkStop::Cancelled) => return Err(MekaError::Interrupted),
            Some(WalkStop::TimedOut) => {
                timed_out = true;
                break;
            }
            None => {}
        }

        let path = search_path.as_path();

        // Stop before the walk, not after, so the remaining roots are counted rather than silently
        // dropped. Checked after `exists`, not before: counting a stale root here would advertise
        // "pass `path` to search one of them directly" about a directory that is gone, and the
        // model spends a round trip discovering that.
        if results.len() > max_results {
            if path.exists() {
                unsearched_roots += 1;
            }
            continue;
        }

        if path.is_file() {
            searched_any = true;
            search_file(&matcher, path, &mut results, max_results)?;
        } else if path.is_dir() {
            searched_any = true;
            if glob_pattern.is_none()
                && let Some(g) = file_glob
            {
                glob_pattern =
                    Some(
                        glob::Pattern::new(g).map_err(|error| MekaError::ToolExecution {
                            tool_name: "search_contents".to_string(),
                            message: format!("invalid glob pattern '{}': {}", g, error),
                        })?,
                    );
            }
            if walk_directory(
                path,
                &matcher,
                &glob_pattern,
                &mut results,
                max_results,
                budget,
                &mut unreadable,
            )? {
                timed_out = true;
                break;
            }
        } else {
            continue;
        }
    }

    // Only claim the path is missing when we actually looked. A budget that expired before the
    // first root was examined leaves `searched_any` false while saying nothing about whether the
    // path exists, and "does not exist" is a definitive answer the model will act on.
    if !searched_any && !timed_out {
        return Err(MekaError::ToolExecution {
            tool_name: "search_contents".to_string(),
            message: format!(
                "path '{}' does not exist",
                search_paths
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join("', '")
            ),
        });
    }

    // The walk collects one match past the cap so "more exist" is distinguishable from "exactly
    // this many exist" without having to search the rest of the tree to find out.
    let truncated = results.len() > max_results;
    if truncated {
        results.truncate(max_results);
    }

    // A search that was cut short must say so even when it found nothing: a bare "No matches
    // found." on an unfinished search reads as a definitive answer.
    let mut notes: Vec<String> = Vec::new();
    if truncated {
        notes.push(format!("truncated, showing first {} matches", max_results));
    }
    if unsearched_roots > 0 {
        notes.push(format!(
            "{} workspace root(s) were not searched because the match cap filled first: pass \
             `path` to search one of them directly, or `scratchpad` to lift the cap",
            unsearched_roots,
        ));
    }
    if timed_out {
        notes.push(format!(
            "search was still running after {}s and was stopped, so these results are \
             incomplete: narrow `path` to a smaller subtree or add a tighter `glob`",
            budget.budget_secs(),
        ));
    }
    if unreadable > 0 {
        notes.push(format!(
            "{} director(ies) could not be read and were skipped, so a match inside them would \
             not appear here",
            unreadable,
        ));
    }

    let body = if results.is_empty() {
        "No matches found.".to_string()
    } else {
        results.join("\n")
    };
    if notes.is_empty() {
        Ok(body)
    } else {
        Ok(format!("{}\n\n... ({})", body, notes.join("; ")))
    }
}

/// Search one file, stopping once `results` holds one entry more than `max_results`: a single file
/// can hold millions of matching lines, so the cap has to bound collection here too, not only at
/// the end of the walk.
fn search_file(
    matcher: &grep_regex::RegexMatcher,
    path: &std::path::Path,
    results: &mut Vec<String>,
    max_results: usize,
) -> Result<()> {
    use grep_searcher::{Searcher, sinks::UTF8};

    let mut searcher = Searcher::new();
    if let Err(error) = searcher.search_path(
        matcher,
        path,
        UTF8(|line_number, line| {
            results.push(format!(
                "{}:{}:{}",
                path.display(),
                line_number,
                line.trim_end()
            ));
            Ok(results.len() <= max_results)
        }),
    ) {
        tracing::debug!("could not search {}: {}", path.display(), error);
    }

    Ok(())
}

/// Walk `directory`, searching every file that passes `glob_pattern`. Returns whether the walk was
/// stopped by the time budget; errors with [`MekaError::Interrupted`] when the turn was cancelled.
fn walk_directory(
    directory: &std::path::Path,
    matcher: &grep_regex::RegexMatcher,
    glob_pattern: &Option<glob::Pattern>,
    results: &mut Vec<String>,
    max_results: usize,
    budget: &WalkBudget,
    // Directories the walk could not open, counted so the caller can say so. A silent skip turns
    // `search_contents` over a tree with an unreadable subdirectory into a confident "No matches
    // found.", which is the definitive-sounding wrong answer the truncation and timeout notices
    // already exist to prevent. `find_files` has reported this all along.
    unreadable: &mut usize,
) -> Result<bool> {
    // Iterative traversal via an explicit work-stack: a recursive walk would overflow the call
    // stack on a pathologically deep directory tree.
    let mut pending: Vec<std::path::PathBuf> = vec![directory.to_path_buf()];

    while let Some(dir) = pending.pop() {
        // Checked here as well as per entry: a run of directories that all fail `read_dir` (a tree
        // the user has no permission for) never reaches the inner loop, and would otherwise grind
        // through the whole work-stack without consulting the budget once.
        match budget.check() {
            Some(WalkStop::Cancelled) => return Err(MekaError::Interrupted),
            Some(WalkStop::TimedOut) => return Ok(true),
            None => {}
        }

        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) => {
                tracing::debug!(
                    "search_contents: cannot read '{}': {}",
                    dir.display(),
                    error
                );
                *unreadable += 1;
                continue;
            }
        };

        for entry in entries {
            match budget.check() {
                Some(WalkStop::Cancelled) => return Err(MekaError::Interrupted),
                Some(WalkStop::TimedOut) => return Ok(true),
                None => {}
            }

            let Ok(entry) = entry else { continue };
            let path = entry.path();

            // `to_string_lossy`, not `to_str().unwrap_or("")`. A directory whose name is not
            // valid UTF-8 made `to_str` yield `None` and the fallback yield `""`, which does not
            // start with `.` -- so `.cache\xff` was the one shape that walked straight past the
            // skip this exists for. Lossy conversion never alters ASCII bytes, so the leading dot
            // and both literals below survive it intact.
            let file_name = path.file_name().unwrap_or_default().to_string_lossy();
            if file_name.starts_with('.') || file_name == "target" || file_name == "node_modules" {
                continue;
            }

            // `entry.file_type()` does not follow symlinks: a symlinked directory reports as a
            // symlink, not a dir, so it is never descended into. That removes any symlink-cycle
            // risk while still letting symlinked *files* be searched via the path-based `is_file()`
            // check below.
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                pending.push(path);
            } else if path.is_file() {
                if let Some(pattern) = glob_pattern
                    && !pattern.matches(&file_name)
                {
                    continue;
                }
                search_file(matcher, &path, results, max_results)?;
                // Stop walking once the cap is exceeded. Reading every remaining file on the
                // machine to fill a result set that is already being truncated is pure waste.
                if results.len() > max_results {
                    return Ok(false);
                }
            }
        }
    }

    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::tests::text_content;

    #[tokio::test]
    async fn test_search_contents() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        std::fs::write(
            temp_dir.path().join("test.txt"),
            "hello world\nfoo bar\nhello again\n",
        )
        .expect("failed");

        let tool = SearchContentsTool {
            cwd: crate::workspace::test_cwd(),
            roots: crate::workspace::test_roots(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "hello",
                    "path": temp_dir.path().to_str().expect("path")
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("hello world"));
        assert!(text_content(&result).contains("hello again"));
    }

    /// The counterpart to `find_files`' nested-root test, pinning why the two tools use different
    /// root sets. `search_contents` descends, so `search_roots` pruning `cwd` in favour of an
    /// ancestor genuinely loses nothing here. If that ever stops holding, this fails rather than
    /// the tool quietly reporting a file in `cwd` as absent.
    #[tokio::test]
    async fn test_search_contents_reaches_cwd_through_an_ancestor_root() {
        let top = tempfile::tempdir().expect("tempdir");
        let nested = top.path().join("main");
        std::fs::create_dir(&nested).expect("mkdir");
        std::fs::write(nested.join("README.md"), "needle here\n").expect("write");

        let tool = SearchContentsTool {
            cwd: std::sync::Arc::new(std::sync::RwLock::new(nested.clone())),
            roots: std::sync::Arc::new(std::sync::RwLock::new(vec![top.path().to_path_buf()])),
        };
        let result = tool
            .execute(
                serde_json::json!({ "pattern": "needle" }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        let text = text_content(&result);
        assert!(
            text.contains("needle here"),
            "a descending walk from the ancestor must still reach cwd; got: {}",
            text,
        );
        assert_eq!(
            text.matches("README.md").count(),
            1,
            "and must not report it twice; got: {}",
            text,
        );
    }

    #[tokio::test]
    async fn test_search_contents_deeply_nested_tree() {
        // Exercises the iterative work-stack traversal: a file buried many directory levels deep
        // must still be found. A recursive walk would recurse once per level; the iterative version
        // uses a heap stack.
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let mut deep = temp_dir.path().to_path_buf();
        for _ in 0..300 {
            deep.push("d");
        }
        std::fs::create_dir_all(&deep).expect("create nested tree");
        std::fs::write(deep.join("buried.txt"), "needle here\n").expect("write");

        let tool = SearchContentsTool {
            cwd: crate::workspace::test_cwd(),
            roots: crate::workspace::test_roots(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "needle",
                    "path": temp_dir.path().to_str().expect("path")
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("needle here"));
    }

    #[tokio::test]
    async fn test_search_contents_inline_capped_at_100() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        // One file with 150 matching lines.
        let content = (0..150).map(|_| "match\n").collect::<String>();
        std::fs::write(temp_dir.path().join("many.txt"), content).expect("write");

        let tool = SearchContentsTool {
            cwd: crate::workspace::test_cwd(),
            roots: crate::workspace::test_roots(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "match",
                    "path": temp_dir.path().to_str().expect("path")
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(text_content(&result).contains("truncated, showing first 100"));
    }

    #[tokio::test]
    async fn test_search_contents_invalid_glob_errors() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp_dir.path().join("a.txt"), "match").expect("write");

        let tool = SearchContentsTool {
            cwd: crate::workspace::test_cwd(),
            roots: crate::workspace::test_roots(),
        };
        let err = tool
            .execute(
                serde_json::json!({
                    "pattern": "match",
                    "path": temp_dir.path().to_str().expect("path"),
                    "glob": "[unclosed",
                }),
                CancellationToken::new(),
            )
            .await
            .expect_err("invalid glob must be rejected, not silently scan everything");
        let message = format!("{}", err);
        assert!(
            message.contains("invalid glob pattern"),
            "unexpected error: {}",
            message
        );
    }

    #[tokio::test]
    async fn test_search_contents_scratchpad_lifts_cap() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let content = (0..150).map(|_| "match\n").collect::<String>();
        std::fs::write(temp_dir.path().join("many.txt"), content).expect("write");

        let tool = SearchContentsTool {
            cwd: crate::workspace::test_cwd(),
            roots: crate::workspace::test_roots(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "match",
                    "path": temp_dir.path().to_str().expect("path"),
                    "scratchpad": "matches"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        let text = text_content(&result);
        assert!(
            !text.contains("truncated"),
            "expected no truncation marker when scratchpad set"
        );
        let match_lines = text.lines().filter(|l| l.contains("match")).count();
        assert!(
            match_lines >= 150,
            "expected >= 150 match lines, got {}",
            match_lines
        );
    }

    #[tokio::test]
    async fn test_search_contents_cancelled_search_is_interrupted() {
        // An ignored cancellation token leaves a search rooted high in the tree running to
        // completion no matter what the user does.
        let temp_dir = tempfile::tempdir().expect("tempdir");
        for i in 0..50 {
            std::fs::write(temp_dir.path().join(format!("f{}.txt", i)), "match\n").expect("write");
        }

        let tool = SearchContentsTool {
            cwd: crate::workspace::test_cwd(),
            roots: crate::workspace::test_roots(),
        };
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let error = tool
            .execute(
                serde_json::json!({
                    "pattern": "match",
                    "path": temp_dir.path().to_str().expect("path")
                }),
                cancellation,
            )
            .await
            .expect_err("a cancelled turn must not run the search to completion");
        assert!(matches!(error, MekaError::Interrupted), "got: {}", error);
    }

    /// A subdirectory the walk cannot open is not "no matches here", it is a part of the tree
    /// nobody looked at. Folding the two together turns a permissions error into a confident
    /// negative the model then answers from, which is the failure every other disclosure in this
    /// tool exists to prevent.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_directory_that_cannot_be_read_is_disclosed_not_counted_as_no_match() {
        use std::os::unix::fs::PermissionsExt;

        let temp_dir = tempfile::tempdir().expect("tempdir");
        let sealed = temp_dir.path().join("sealed");
        std::fs::create_dir(&sealed).expect("mkdir");
        std::fs::write(sealed.join("hit.txt"), "needle\n").expect("write");
        std::fs::set_permissions(&sealed, std::fs::Permissions::from_mode(0o000)).expect("seal");
        if std::fs::read_dir(&sealed).is_ok() {
            // Running as root, where the mode is advisory. Nothing to assert.
            return;
        }

        let tool = SearchContentsTool {
            cwd: crate::workspace::test_cwd(),
            roots: crate::workspace::test_roots(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "needle",
                    "path": temp_dir.path().to_str().expect("path")
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        // Restore the mode so the temp dir can be torn down.
        std::fs::set_permissions(&sealed, std::fs::Permissions::from_mode(0o700)).expect("unseal");

        let text = text_content(&result);
        assert!(
            text.contains("could not be read"),
            "the unreadable directory must be named as unsearched, got: {}",
            text
        );
    }

    /// cwd is always root #1, so a busy cwd filling the cap would otherwise starve every other
    /// root while the output said only "truncated" -- which reads as "the other folders had
    /// nothing", the exact failure multi-root support exists to prevent.
    #[test]
    fn test_search_discloses_roots_left_unsearched_when_the_cap_fills() {
        let busy = tempfile::tempdir().expect("tempdir");
        let other = tempfile::tempdir().expect("tempdir");
        // More matches than the cap, all in the first root.
        let body = (0..MAX_INLINE_MATCHES + 20)
            .map(|_| "needle\n")
            .collect::<String>();
        std::fs::write(busy.path().join("busy.txt"), body).expect("write");
        std::fs::write(other.path().join("other.txt"), "needle\n").expect("write");

        let budget = WalkBudget::new(CancellationToken::new());
        let output = search_with_grep(
            "needle",
            &[busy.path().to_path_buf(), other.path().to_path_buf()],
            None,
            MAX_INLINE_MATCHES,
            &budget,
        )
        .expect("search should return, not error");

        assert!(output.contains("truncated"), "got: {}", output);
        assert!(
            output.contains("1 workspace root(s) were not searched"),
            "an unsearched root must be disclosed, not implied by absence; got: {}",
            output
        );
        // The note is prose the model reads and acts on, so pin it as prose: a `\`-continued string
        // literal that loses its leading-whitespace escape silently ships a run of spaces
        // mid-sentence, which no substring assertion would notice.
        assert!(
            !output.contains("  "),
            "the disclosure must not contain runs of whitespace; got: {}",
            output
        );
    }

    /// A root that no longer exists must not be counted into the cap-filled disclosure. It was not
    /// skipped because the cap filled, and telling the model to `path`-search it directly buys a
    /// round trip that can only answer "does not exist".
    #[test]
    fn test_search_does_not_blame_the_cap_for_a_stale_root() {
        let busy = tempfile::tempdir().expect("tempdir");
        let body = (0..MAX_INLINE_MATCHES + 20)
            .map(|_| "needle\n")
            .collect::<String>();
        std::fs::write(busy.path().join("busy.txt"), body).expect("write");

        let budget = WalkBudget::new(CancellationToken::new());
        let output = search_with_grep(
            "needle",
            &[
                busy.path().to_path_buf(),
                std::path::PathBuf::from("/nonexistent-workspace-root"),
            ],
            None,
            MAX_INLINE_MATCHES,
            &budget,
        )
        .expect("search should return, not error");

        assert!(output.contains("truncated"), "got: {}", output);
        assert!(
            !output.contains("not searched"),
            "a stale root is not a root the cap starved; got: {}",
            output
        );
    }

    /// A budget that expired before any root was examined says nothing about whether the path
    /// exists, so it must not report "does not exist" -- a definitive answer the model acts on.
    #[test]
    fn test_expired_budget_reports_timeout_not_missing_path() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("a.txt"), "needle\n").expect("write");

        let budget =
            WalkBudget::with_budget(CancellationToken::new(), std::time::Duration::from_secs(0));
        let output = search_with_grep(
            "needle",
            &[temp.path().to_path_buf()],
            None,
            MAX_INLINE_MATCHES,
            &budget,
        )
        .expect("an expired budget must not be reported as a missing path");
        assert!(output.contains("still running"), "got: {}", output);
    }

    #[test]
    fn test_search_with_grep_discloses_timeout_with_no_matches() {
        // The dangerous shape: the budget expires before anything is found, and reporting a bare
        // "No matches found." would present an unfinished search as a definitive answer.
        let temp_dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp_dir.path().join("a.txt"), "needle\n").expect("write");

        let budget =
            WalkBudget::with_budget(CancellationToken::new(), std::time::Duration::from_secs(0));
        let output = search_with_grep(
            "needle",
            &[temp_dir.path().to_path_buf()],
            None,
            MAX_INLINE_MATCHES,
            &budget,
        )
        .expect("search should return, not error");

        assert!(output.contains("No matches found."), "got: {}", output);
        assert!(output.contains("incomplete"), "got: {}", output);
    }

    #[test]
    fn test_search_file_stops_collecting_past_the_cap() {
        // A single file can hold millions of matching lines; the cap has to bound collection here
        // and not only at the end of the walk.
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let content = (0..5_000).map(|_| "match\n").collect::<String>();
        let file_path = temp_dir.path().join("many.txt");
        std::fs::write(&file_path, content).expect("write");

        let matcher = grep_regex::RegexMatcherBuilder::new()
            .build("match")
            .expect("matcher");
        let mut results = Vec::new();
        search_file(&matcher, &file_path, &mut results, 10).expect("search");

        assert_eq!(
            results.len(),
            11,
            "expected the cap plus the one entry that proves more exist"
        );
    }
}
