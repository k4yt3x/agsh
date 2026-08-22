//! The `skill_*` tools: the agent's access to the installed skill store ([`crate::skills`]).
//!
//! All four gate at [`Permission::Read`], for the reason spelled out on [`crate::tools::memory`]:
//! `Write` in meka means "may modify the user's tree", and these write to a store meka owns under
//! its own config directory. `agent_spawn` is a read-tier tool too, so the dispatcher deployment
//! these exist for runs at read permission permanently; gating them at `Write` would withhold them
//! from the only configuration that wants them.
//!
//! `skill_write` and `skill_delete` are registered only when `[skills] agent_managed` is on, and
//! never for a sub-agent. The authorization lives in that flag rather than in the permission tier.

use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::{
    Tool, ToolOutput,
    util::{MAX_SEARCH_MATCHES, compile_user_regex, require_str},
};
use crate::{
    error::{MekaError, Result},
    permission::Permission,
    provider::ToolDefinition,
    skills::{self, SkillCache},
};

pub(super) struct SkillReadTool {
    /// Shared skill cache with the agent. Dispatch reads through `current().await` so the tool
    /// sees any auto-reloads that happened during the turn.
    pub skills: Arc<SkillCache>,
}

#[async_trait]
impl Tool for SkillReadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "skill_read".to_string(),
            description: "Load the full content of a named skill. Skills are knowledge \
                          files that document procedures, tools, and non-standard \
                          knowledge. Call this tool with the skill name (as listed in \
                          the conversation context) to get its full instructions."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the skill to load"
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
        let name = require_str(&input, "name", "skill_read")?;
        let skills = self.skills.current().await;

        let skill = match skills.find(&name) {
            Some(skill) => skill,
            // "No such skill" and "it is right there and meka cannot read it" call for opposite
            // responses, and both used to arrive as "not found" -- so a model handed a procedure
            // whose file has a typo in its frontmatter was told the procedure does not exist, and
            // would go on to improvise one. `memory_read` was changed to stop telling exactly this
            // lie; this is the same fix on the sibling store.
            None => {
                let hint = match skills.skip_reason(&name) {
                    Some(reason) => format!(
                        "Error: skill '{}' exists on disk but could not be read: {}. Tell the user; \
                         they need to fix that file. Do not substitute your own version of it.",
                        name, reason
                    ),
                    None => {
                        let available: Vec<&str> =
                            skills.skills.iter().map(|s| s.name.as_str()).collect();
                        let hint = if available.is_empty() {
                            "No skills are installed.".to_string()
                        } else {
                            format!("Available skills: {}", available.join(", "))
                        };
                        format!("Error: skill '{}' not found. {}", name, hint)
                    }
                };
                return Ok(ToolOutput::text(hint, true));
            }
        };

        let body =
            skills::load_skill_body(skill)
                .await
                .map_err(|error| MekaError::ToolExecution {
                    tool_name: "skill_read".to_string(),
                    message: error,
                })?;

        Ok(ToolOutput::text(body, false))
    }
}

/// Resolve the skills root, or fail naming the cause rather than reporting an empty store. Mirrors
/// `require_root` in [`crate::tools::memory`].
fn require_root(cache: &SkillCache, tool_name: &str) -> Result<std::path::PathBuf> {
    cache
        .root()
        .map(|root| root.to_path_buf())
        .ok_or_else(|| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: "skills are disabled or the meka config directory could not be resolved"
                .to_string(),
        })
}

/// Refuse to touch a directory that holds a `SKILL.md` discovery could not parse.
///
/// Absent from the index is not the same as absent from disk. Such a file is skipped, so neither
/// the model nor this tool can say what is in it, and its only copy is that file. Reporting it as
/// "not found" while it sits in the skills directory is the confusion [`skills::SkippedSkill`]
/// exists to prevent, so name the case instead -- and name the *reason*, which the file cannot.
///
/// Answered from the index rather than by probing the filesystem. Discovery has already read every
/// one of these files and recorded why each failed; re-deriving a weaker version of that with a
/// `is_file()` call could say "not a valid skill" but never why, and gave a second, later answer to
/// a question already settled.
///
/// The remedy is only offered when the file is one `meka skill remove` can reach. For a broken
/// skill under a read-only `extra_paths` root that command answers "not found", so pointing the
/// model at it sent the user round a loop; the refusal names the path instead.
///
/// [`skills::write_skill`] refuses the same case independently; this exists so the refusal arrives
/// as a readable tool result rather than a tool error, and so `skill_delete` gets it too.
fn reject_unreadable(
    name: &str,
    installed: &skills::SkillIndex,
    native_root: &std::path::Path,
) -> Option<ToolOutput> {
    // A skill that *loaded* is not here, and that is [`skills::SkillIndex`]'s disjointness
    // invariant rather than a check of this function's own. Without it, a working `deploy` in
    // meka's store beside a broken `deploy/` in a read-only root put the name in both halves,
    // and this refused to write a skill sitting in the index -- claiming its contents could not
    // be shown, and offering `meka skill remove deploy`, which reaches the working copy.
    // Re-checking `find` here would fix this door and leave the other readers of `skipped` to
    // each remember the same thing.
    //
    // A bare `skills/<name>/` with no `SKILL.md` is not here either, because discovery skips such a
    // directory silently rather than recording it: it is a half-finished `meka skill add`, a
    // partly-copied folder, or the residue of an interrupted write, and `write_skill` should
    // happily finish it. This comment claimed that was already true for a while when it was not;
    // `a_directory_with_no_skill_file_is_not_a_broken_skill` is what makes it so.
    let reason = installed.skip_reason(name)?;
    let remedy = match installed.location(name) {
        Some((root, source_dir)) if root != native_root => format!(
            "It lives at {}, which meka reads but does not write to, so ask the user to fix or \
             remove it there.",
            source_dir.display()
        ),
        _ => format!(
            "Use a different name, or ask the user to fix or remove it with \
             `meka skill remove {}`.",
            name
        ),
    };
    Some(ToolOutput::text(
        format!(
            "Error: '{}' exists on disk but its SKILL.md is not a valid skill ({}), so it is in no \
             index and its contents cannot be shown. Leaving it untouched rather than overwriting \
             something neither of us can see. {}",
            name, reason, remedy
        ),
        true,
    ))
}

pub(super) struct SkillSearchTool {
    pub skills: Arc<SkillCache>,
}

#[async_trait]
impl Tool for SkillSearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "skill_search".to_string(),
            description: "Search the full text of every installed skill by regex. Use when the \
                one-line descriptions in your skill index are not enough to tell which skill \
                covers something, or when the index says skills are not shown. Searches bodies as \
                well as frontmatter, so this finds skills whose description does not mention the \
                term you are looking for."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Rust regex matched against each line of every skill"
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
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let pattern = require_str(&input, "pattern", "skill_search")?;
        let regex = compile_user_regex(&pattern, "skill_search")?;
        let skills = self.skills.current().await;

        let mut matches = Vec::new();
        let mut truncated = false;
        for skill in skills.skills.iter() {
            let content = match tokio::fs::read_to_string(&skill.body_path).await {
                Ok(content) => content,
                Err(error) => {
                    tracing::warn!(
                        "skill_search skipping {}: {}",
                        skill.body_path.display(),
                        error
                    );
                    continue;
                }
            };
            for (index, line) in content.lines().enumerate() {
                if !regex.is_match(line) {
                    continue;
                }
                if matches.len() >= MAX_SEARCH_MATCHES {
                    truncated = true;
                    break;
                }
                matches.push(format!("{}:{}: {}", skill.name, index + 1, line.trim()));
            }
            if truncated {
                break;
            }
        }

        if matches.is_empty() {
            return Ok(ToolOutput::text(
                "No skills matched that pattern.".to_string(),
                false,
            ));
        }

        let mut out = matches.join("\n");
        if truncated {
            out.push_str(&format!(
                "\n\n(stopped at {} matches; narrow the pattern to see the rest)",
                MAX_SEARCH_MATCHES
            ));
        }
        Ok(ToolOutput::text(out, false))
    }
}

pub(super) struct SkillWriteTool {
    pub skills: Arc<SkillCache>,
}

#[async_trait]
impl Tool for SkillWriteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "skill_write".to_string(),
            description: "Create or update a skill: a reusable procedure written down so a later \
                session, or a sub-agent, can follow it without being told again. Writing to a name \
                that already exists updates it, so this is also how you refine one: omit body and \
                whatever the skill already documented is kept. Prefer a skill over a memory when \
                the content is a *method* rather than a fact, and especially when you would want \
                to hand it to a sub-agent, since `agent_spawn` can run a skill by name without \
                routing its text through your own context."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Identifier: lowercase letters, digits and hyphens (e.g. 'triage-build-failure')"
                    },
                    "description": {
                        "type": "string",
                        "description": "One line stating what the skill is for, shown in every \
                                        future session's skill index"
                    },
                    "priority": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 9,
                        "description": "Lower sorts higher in the index and survives truncation. \
                                        0-2 procedures you reach for constantly, 5 default, 6-9 \
                                        rarely relevant"
                    },
                    "body": {
                        "type": "string",
                        "description": "The procedure itself, loaded only when skill_read is \
                                        called or the skill is spawned. Omit it to leave an \
                                        existing skill's body untouched"
                    }
                },
                "required": ["name", "description"]
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
        let root = require_root(&self.skills, "skill_write")?;
        let name = require_str(&input, "name", "skill_write")?;
        // Before the join below, and again inside `write_skill`. Same layered guard `memory_write`
        // applies: these tools run at read permission, so the character class is what keeps this
        // from being an arbitrary-file-write primitive.
        skills::validate_skill_name(&name).map_err(|message| MekaError::ToolExecution {
            tool_name: "skill_write".to_string(),
            message,
        })?;
        let description = require_str(&input, "description", "skill_write")?;
        let body = input.get("body").and_then(serde_json::Value::as_str);
        let requested_priority = match input.get("priority") {
            Some(serde_json::Value::Null) | None => None,
            Some(value) => {
                let raw = value.as_i64().ok_or_else(|| MekaError::ToolExecution {
                    tool_name: "skill_write".to_string(),
                    message: format!("'priority' must be a whole number, got {}", value),
                })?;
                Some(crate::store::parse_priority(Some(raw), "skill", &name))
            }
        };

        let installed = self.skills.current().await;
        // Omitted means "leave it alone", the rule `PUT /v1/skills` already applies and this tool's
        // own description promises ("omit body and whatever the skill already documented is kept").
        // Reading the absence as the default silently demoted a prioritised skill every time the
        // agent refined its text -- and priority both orders the `[Skills]` index the model reads
        // and decides which entries the index cap drops, so the demotion can remove it from view.
        let priority = requested_priority.unwrap_or_else(|| {
            installed
                .find(&name)
                .map_or(crate::store::DEFAULT_PRIORITY, |skill| skill.priority)
        });
        // Unreadable first, because it is the more specific answer: a file that is both foreign and
        // unparseable needs its parse error named, and `reject_unreadable` carries the read-only
        // remedy for that case where the plain foreign refusal cannot carry the reason.
        if let Some(refusal) = reject_unreadable(&name, &installed, &root) {
            return Ok(refusal);
        }
        if let Some(refusal) = skills::refuse_foreign_write(&installed, &name, &root) {
            return Ok(ToolOutput::text(format!("Error: {}", refusal), true));
        }
        // Read before the write, since the write is what makes the file exist: otherwise the
        // confirmation would claim to have kept the body of a skill that had none.
        let kept_existing_body = body.is_none() && installed.find(&name).is_some();

        // On the blocking pool, for the same reason `memory_write` is: the write goes through
        // `write_file_atomic`, which `fsync`s, and a `fsync` parks the calling thread for as long
        // as the filesystem takes. On a runtime worker that is every other session's turn waiting.
        let written = {
            let root = root.clone();
            let name = name.clone();
            let description = description.clone();
            let body = body.map(str::to_string);
            tokio::task::spawn_blocking(move || {
                skills::write_skill(
                    &root,
                    &name,
                    &description,
                    priority,
                    Some(AGENT_AUTHOR),
                    body.as_deref(),
                )
            })
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "skill_write".to_string(),
                message: format!("write task failed: {}", error),
            })?
            .map_err(|message| MekaError::ToolExecution {
                tool_name: "skill_write".to_string(),
                message,
            })?
        };
        // The write is only visible to the next `current()` if the cache notices it, and a
        // `(mtime, size)` snapshot cannot see a same-tick rewrite of the same length. That is
        // not hypothetical here: the dispatcher flow writes a skill and hands it to
        // `agent_spawn(skill:)` milliseconds later, in the same turn.
        self.skills.invalidate().await;

        tracing::info!("saved skill to {}", written.body_path.display());
        Ok(ToolOutput::text(
            // The rank the *file* now carries, read back from the bytes rather than echoed from
            // the request. Deliberately promises reachability by name rather than a
            // place in the index: the index is capped, so a low-priority skill in a
            // large store may not be listed there, and `skill_read` / `agent_spawn`
            // work either way.
            format!(
                "Saved skill '{}' (priority {}){}. From the next turn on you can load it with \
                 skill_read, or hand it to a worker with agent_spawn(skill: \"{}\").",
                name,
                written.priority,
                if kept_existing_body {
                    ", keeping the existing body"
                } else {
                    ""
                },
                name
            ),
            false,
        ))
    }
}

/// Stamped into the `author` frontmatter of a skill the agent *creates*.
///
/// Only on creation: [`skills::write_skill`] keeps an existing `author`, so refining a skill you
/// wrote does not quietly reassign it. Informational only, using a field skills already had. It
/// exists so `meka skill list` is legible about where an entry came from, not as a guard: nothing
/// branches on it.
const AGENT_AUTHOR: &str = "meka (agent-authored)";

pub(super) struct SkillDeleteTool {
    pub skills: Arc<SkillCache>,
}

#[async_trait]
impl Tool for SkillDeleteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "skill_delete".to_string(),
            description: "Delete a skill permanently, including any files bundled alongside it. \
                Use when a procedure you wrote down has turned out to be wrong or no longer \
                applies. To revise a skill rather than drop it, call skill_write with the same \
                name instead."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the skill to delete"
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
        let root = require_root(&self.skills, "skill_delete")?;
        let name = require_str(&input, "name", "skill_delete")?;
        // Lookup rules: a skill whose name predates the spec is still listed and still readable, so
        // it has to be removable too.
        skills::validate_addressable_name(&name).map_err(|message| MekaError::ToolExecution {
            tool_name: "skill_delete".to_string(),
            message,
        })?;

        let installed = self.skills.current().await;
        // Unreadable first, for the reason `skill_write` gives.
        if let Some(refusal) = reject_unreadable(&name, &installed, &root) {
            return Ok(refusal);
        }
        if let Some(refusal) = skills::refuse_foreign_delete(&installed, &name, &root) {
            return Ok(ToolOutput::text(format!("Error: {}", refusal), true));
        }
        if installed.find(&name).is_none() {
            return Ok(ToolOutput::text(
                format!("Error: skill '{}' not found.", name),
                true,
            ));
        }

        let dir =
            skills::delete_skill(&root, &name).map_err(|message| MekaError::ToolExecution {
                tool_name: "skill_delete".to_string(),
                message,
            })?;
        // See the note in `skill_write`: the index must not keep listing a skill that is gone.
        self.skills.invalidate().await;

        tracing::info!("deleted skill {}", dir.display());
        Ok(ToolOutput::text(
            format!("Deleted skill '{}' and everything in its directory.", name),
            false,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn write_skill(root: &Path, name: &str, body: &str) {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).expect("create dir");
        std::fs::write(dir.join("SKILL.md"), body).expect("write SKILL.md");
    }

    #[tokio::test]
    async fn test_skill_tool_unknown_skill() {
        let tool = SkillReadTool {
            skills: SkillCache::for_root(None),
        };
        let result = tool
            .execute(
                serde_json::json!({"name": "nonexistent-skill-xyz"}),
                CancellationToken::new(),
            )
            .await
            .expect("should return Ok with error output");

        assert!(result.is_error);
        let text = crate::provider::ContentBlock::tool_result_text_content(&result.content);
        assert!(text.contains("not found"));
    }

    #[tokio::test]
    async fn test_skill_tool_missing_name() {
        let tool = SkillReadTool {
            skills: SkillCache::for_root(None),
        };
        let result = tool
            .execute(serde_json::json!({}), CancellationToken::new())
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_skill_tool_prepends_context_header() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "demo",
            "---\ndescription: x\n---\nRun helper.py to do the thing.\n",
        );
        let tool = SkillReadTool {
            skills: SkillCache::for_root(Some(temp.path().to_path_buf())),
        };
        let result = tool
            .execute(
                serde_json::json!({"name": "demo"}),
                CancellationToken::new(),
            )
            .await
            .expect("should load");

        assert!(!result.is_error);
        let text = crate::provider::ContentBlock::tool_result_text_content(&result.content);
        assert!(text.starts_with("Base directory for this skill and its bundled files:"));
        assert!(text.contains(&temp.path().join("demo").display().to_string()));
        assert!(text.contains("Run helper.py to do the thing."));
    }

    #[test]
    fn an_index_tells_absent_from_unreadable() {
        let skill = crate::skills::Skill {
            name: "foo".to_string(),
            source_dir: std::path::PathBuf::from("/tmp"),
            description: "desc".to_string(),
            license: None,
            compatibility: None,
            allowed_tools: None,
            priority: crate::store::DEFAULT_PRIORITY,
            metadata: None,
            extra: std::collections::BTreeMap::new(),
            conformance: crate::skills::Conformance::default(),
            body_path: std::path::PathBuf::from("/tmp/SKILL.md"),
            root: std::path::PathBuf::from("/tmp"),
        };
        let index = crate::skills::SkillIndex {
            skills: vec![skill],
            skipped: vec![crate::skills::SkippedSkill {
                name: "broken".to_string(),
                reason: "missing YAML frontmatter".to_string(),
                root: std::path::PathBuf::from("/tmp"),
            }],
        };
        assert!(index.find("foo").is_some());
        assert!(index.find("bar").is_none());
        // The distinction the whole index exists for: a name that is absent and a name whose file
        // is unreadable are different answers, and only one of them is "no such skill".
        assert_eq!(
            index.skip_reason("broken"),
            Some("missing YAML frontmatter")
        );
        assert_eq!(index.skip_reason("bar"), None);
        assert!(index.find("broken").is_none());
    }

    #[test]
    fn test_write_skill_helper() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "test",
            "---\ndescription: x\nwhen_to_use: y\n---\nbody\n",
        );
        assert!(temp.path().join("test/SKILL.md").exists());
    }

    fn cache_at(temp: &tempfile::TempDir) -> Arc<SkillCache> {
        SkillCache::for_root(Some(temp.path().to_path_buf()))
    }

    /// A skill from a read-only `extra_paths` root is not the agent's to change: the write would
    /// land in meka's own store and shadow it, so the tool would report an update that did not
    /// happen to the file every other client reads.
    #[tokio::test]
    async fn write_and_delete_refuse_a_skill_from_a_read_only_root() {
        let temp = tempfile::tempdir().expect("tempdir");
        let native = temp.path().join("native");
        let shared = temp.path().join("shared");
        std::fs::create_dir_all(&native).expect("native");
        write_skill(
            &shared,
            "borrowed",
            "---\ndescription: theirs\n---\nTHEIR PROCEDURE\n",
        );
        let skills = SkillCache::new(Some(native.clone()), vec![shared.clone()]);

        let write = SkillWriteTool {
            skills: skills.clone(),
        };
        let result = run(
            &write,
            serde_json::json!({"name": "borrowed", "description": "mine", "body": "MINE"}),
        )
        .await;
        assert!(result.is_error);
        assert!(
            text_of(&result).contains("does not write to"),
            "{}",
            text_of(&result)
        );

        let delete = SkillDeleteTool {
            skills: skills.clone(),
        };
        let result = run(&delete, serde_json::json!({"name": "borrowed"})).await;
        assert!(result.is_error);
        assert!(text_of(&result).contains("does not write to"));

        // Neither refusal touched the foreign file, and neither created a shadow copy.
        assert!(
            std::fs::read_to_string(shared.join("borrowed/SKILL.md"))
                .expect("still there")
                .contains("THEIR PROCEDURE")
        );
        assert!(
            !native.join("borrowed").exists(),
            "a shadowing copy must not be created in meka's own root"
        );

        // A name meka does own is unaffected.
        let result = run(
            &write,
            serde_json::json!({"name": "ours", "description": "d", "body": "b"}),
        )
        .await;
        assert!(!result.is_error, "{}", text_of(&result));
    }

    /// The read-only rule covers a foreign skill whose `SKILL.md` does not parse, and the refusal
    /// sends the reader to the file rather than to a command that cannot reach it.
    ///
    /// Both halves were wrong. Every door compared against the *loaded* skills, so an unparseable
    /// file in an `extra_paths` root was a name nothing had an opinion about and got shadowed
    /// silently -- the worst case to shadow, since the original is then reported nowhere at all.
    /// And the refusal that did fire named `meka skill remove`, which answers "not found" for a
    /// file meka does not own.
    #[tokio::test]
    async fn a_broken_skill_in_a_read_only_root_is_neither_shadowed_nor_misdirected() {
        let temp = tempfile::tempdir().expect("tempdir");
        let native = temp.path().join("native");
        let shared = temp.path().join("shared");
        std::fs::create_dir_all(&native).expect("native");
        write_skill(
            &shared,
            "wrecked",
            "---\ndescription: [unclosed\n---\nTHEIRS\n",
        );
        let skills = SkillCache::new(Some(native.clone()), vec![shared.clone()]);

        let write = SkillWriteTool {
            skills: skills.clone(),
        };
        let result = run(
            &write,
            serde_json::json!({"name": "wrecked", "description": "mine", "body": "MINE"}),
        )
        .await;
        assert!(result.is_error, "{}", text_of(&result));
        let text = text_of(&result);
        assert!(
            text.contains(&shared.join("wrecked").display().to_string()),
            "the refusal must name where the file is: {text}"
        );
        assert!(
            !text.contains("meka skill remove"),
            "that command cannot reach a read-only root: {text}"
        );
        assert!(
            !native.join("wrecked").exists(),
            "an unparseable foreign skill must not be shadowed either"
        );

        // A broken skill meka *does* own still gets the remedy that works for it.
        write_skill(&native, "ours-wrecked", "no frontmatter\n");
        skills.invalidate().await;
        let result = run(
            &write,
            serde_json::json!({"name": "ours-wrecked", "description": "mine"}),
        )
        .await;
        assert!(result.is_error);
        assert!(
            text_of(&result).contains("meka skill remove ours-wrecked"),
            "{}",
            text_of(&result)
        );
    }

    /// A name that loaded is writable, whatever a shadowed copy of it elsewhere looks like.
    ///
    /// Roots merge first-wins and the skip list records every failure, so meka's own working
    /// `deploy` and a broken `deploy/` in a read-only root put one name in both halves of the
    /// index. `reject_unreadable` answered from the skipped half alone, which refused every write
    /// and every delete of a skill plainly in the index -- telling the model its contents could not
    /// be shown, and offering `meka skill remove deploy`, which reaches the working copy. An agent
    /// that authored a skill could then neither refine nor remove it, for a file in a directory it
    /// does not own.
    #[tokio::test]
    async fn a_skill_that_loaded_is_writable_though_a_broken_copy_shadows_it() {
        let temp = tempfile::tempdir().expect("tempdir");
        let native = temp.path().join("native");
        let shared = temp.path().join("shared");
        write_skill(
            &native,
            "deploy",
            "---\nname: deploy\ndescription: mine and working\n---\nMINE\n",
        );
        write_skill(
            &shared,
            "deploy",
            "---\ndescription: [unclosed\n---\nTHEIRS\n",
        );
        let skills = SkillCache::new(Some(native.clone()), vec![shared.clone()]);
        let index = skills.current().await;
        assert!(index.find("deploy").is_some(), "the native copy wins");
        assert_eq!(
            index.skip_reason("deploy"),
            None,
            "and the shadowed broken copy must not also claim the name"
        );

        let write = SkillWriteTool {
            skills: skills.clone(),
        };
        let result = run(
            &write,
            serde_json::json!({"name": "deploy", "description": "refined", "body": "MINE2"}),
        )
        .await;
        assert!(!result.is_error, "{}", text_of(&result));
        assert!(
            std::fs::read_to_string(native.join("deploy/SKILL.md"))
                .expect("still there")
                .contains("MINE2"),
            "the write must land in meka's own root"
        );
        assert!(
            std::fs::read_to_string(shared.join("deploy/SKILL.md"))
                .expect("still there")
                .contains("THEIRS"),
            "and must not touch the read-only one"
        );

        skills.invalidate().await;
        let delete = SkillDeleteTool {
            skills: skills.clone(),
        };
        let result = run(&delete, serde_json::json!({"name": "deploy"})).await;
        assert!(!result.is_error, "{}", text_of(&result));
        assert!(!native.join("deploy").exists(), "removed from meka's store");
        assert!(shared.join("deploy").exists(), "left alone elsewhere");
    }

    /// A skill whose file is unreadable is reported as unreadable, not as absent.
    ///
    /// The two call for opposite responses and both used to arrive as "not found", so a model
    /// handed a procedure with a typo in its frontmatter was told the procedure does not exist --
    /// and the reasonable next move, improvising its own version, is the worst available one.
    /// `memory_read` was changed to stop telling this lie; skills kept telling it because discovery
    /// computed the reason and then threw it away.
    #[tokio::test]
    async fn read_says_a_broken_skill_is_broken_rather_than_missing() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(temp.path(), "broken", "no frontmatter at all\nKEEP ME\n");
        write_skill(temp.path(), "fine", "---\ndescription: d\n---\nbody\n");
        let skills = cache_at(&temp);

        let read = SkillReadTool {
            skills: skills.clone(),
        };
        let result = run(&read, serde_json::json!({"name": "broken"})).await;
        assert!(result.is_error);
        let text = text_of(&result);
        assert!(
            text.contains("could not be read"),
            "reported as missing: {text}"
        );
        assert!(
            !text.contains("not found"),
            "a file that is right there is not 'not found': {text}"
        );
        // And the reason, which is the part the model can act on by telling the user.
        assert!(text.contains("frontmatter"), "{text}");

        // A name that really is absent still gets the plain answer, with the available list.
        let result = run(&read, serde_json::json!({"name": "absent"})).await;
        let text = text_of(&result);
        assert!(text.contains("not found"), "{text}");
        assert!(text.contains("fine"), "{text}");
    }

    /// `skill_write` surfaces the store's refusal of a `metadata` it cannot record in.
    ///
    /// It used to write anyway and then explain, in the model's context, that the rank it asked for
    /// had not applied "because this skill's 'metadata' is not a map" -- a sentence about YAML
    /// shapes that only existed because three other places had quietly done something other than
    /// what was asked. Refusing says it once, to the party who can fix it.
    #[tokio::test]
    async fn write_surfaces_the_refusal_of_a_metadata_it_cannot_record_in() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "verbatim",
            "---\nname: verbatim\ndescription: original\nmetadata: none\n---\nBODY\n",
        );
        let skills = cache_at(&temp);
        let write = SkillWriteTool {
            skills: skills.clone(),
        };

        let error = write
            .execute(
                serde_json::json!({"name": "verbatim", "description": "refined", "priority": 1}),
                CancellationToken::new(),
            )
            .await
            .expect_err("must refuse rather than write and explain");
        assert!(error.to_string().contains("not a map"), "{error}");

        // The file is untouched, and an ordinary skill still reports the rank it was given.
        let result = run(
            &write,
            serde_json::json!({"name": "ordinary", "description": "d", "priority": 1}),
        )
        .await;
        assert!(
            text_of(&result).contains("(priority 1)"),
            "{}",
            text_of(&result)
        );
    }

    async fn run(tool: &dyn Tool, input: serde_json::Value) -> ToolOutput {
        tool.execute(input, CancellationToken::new())
            .await
            .expect("tool should return Ok")
    }

    fn text_of(output: &ToolOutput) -> String {
        crate::provider::ContentBlock::tool_result_text_content(&output.content)
    }

    #[tokio::test]
    async fn test_skill_write_creates_a_loadable_skill() {
        let temp = tempfile::tempdir().expect("tempdir");
        let skills = cache_at(&temp);
        let write = SkillWriteTool {
            skills: skills.clone(),
        };

        let result = run(
            &write,
            serde_json::json!({
                "name": "triage",
                "description": "How to triage a build failure",
                "priority": 2,
                "body": "1. Read the log.\n2. Bisect.\n"
            }),
        )
        .await;
        assert!(!result.is_error, "{}", text_of(&result));

        // Round-trips through discovery rather than just checking the bytes: what matters is that
        // the file this wrote is one the parser accepts, since a skill that fails to parse is
        // silently skipped and would look identical from the write side.
        let discovered = skills.current().await;
        let skill = discovered
            .skills
            .iter()
            .find(|skill| skill.name == "triage")
            .expect("written skill must be discoverable");
        assert_eq!(skill.description, "How to triage a build failure");
        assert_eq!(skill.priority, 2);
        assert_eq!(skill.author().as_deref(), Some(AGENT_AUTHOR));

        let read = SkillReadTool { skills };
        let body = text_of(&run(&read, serde_json::json!({"name": "triage"})).await);
        assert!(body.contains("1. Read the log."), "{}", body);
    }

    /// An omitted `body` is "leave it alone", not "make it empty". A call that only re-prioritises
    /// a skill is one the schema invites, and treating the absent field as an empty string would
    /// delete the whole procedure on exactly that call.
    #[tokio::test]
    async fn test_skill_write_without_body_keeps_the_existing_one() {
        let temp = tempfile::tempdir().expect("tempdir");
        let skills = cache_at(&temp);
        let write = SkillWriteTool {
            skills: skills.clone(),
        };

        run(
            &write,
            serde_json::json!({
                "name": "keep",
                "description": "first",
                "body": "PRECIOUS PROCEDURE"
            }),
        )
        .await;
        let result = run(
            &write,
            serde_json::json!({"name": "keep", "description": "second", "priority": 1}),
        )
        .await;
        assert!(text_of(&result).contains("keeping the existing body"));

        let read = SkillReadTool {
            skills: skills.clone(),
        };
        let body = text_of(&run(&read, serde_json::json!({"name": "keep"})).await);
        assert!(body.contains("PRECIOUS PROCEDURE"), "{}", body);

        let discovered = skills.current().await;
        let skill = discovered
            .skills
            .iter()
            .find(|s| s.name == "keep")
            .expect("keep");
        assert_eq!(skill.description, "second");
        assert_eq!(skill.priority, 1);
    }

    #[tokio::test]
    async fn test_skill_write_clears_the_body_on_an_explicit_empty_string() {
        let temp = tempfile::tempdir().expect("tempdir");
        let skills = cache_at(&temp);
        let write = SkillWriteTool {
            skills: skills.clone(),
        };

        run(
            &write,
            serde_json::json!({"name": "clear", "description": "d", "body": "GONE"}),
        )
        .await;
        run(
            &write,
            serde_json::json!({"name": "clear", "description": "d", "body": ""}),
        )
        .await;

        let read = SkillReadTool { skills };
        let body = text_of(&run(&read, serde_json::json!({"name": "clear"})).await);
        assert!(!body.contains("GONE"), "{}", body);
        // Pinned, not merely "GONE is absent": a skill *is* its body, so an emptied one falls back
        // to a bare heading rather than leaving `skill_read` with only the directory header.
        assert!(body.contains("# clear"), "{}", body);
    }

    /// The name is joined onto the skills root, so this is the guard that keeps a read-permission
    /// tool from writing anywhere on disk.
    #[tokio::test]
    async fn test_skill_write_rejects_a_traversing_name() {
        let temp = tempfile::tempdir().expect("tempdir");
        let write = SkillWriteTool {
            skills: cache_at(&temp),
        };
        let result = write
            .execute(
                serde_json::json!({"name": "../escape", "description": "d"}),
                CancellationToken::new(),
            )
            .await;
        assert!(result.is_err(), "a traversing name must not reach the disk");
        assert!(
            !temp
                .path()
                .parent()
                .is_some_and(|p| p.join("escape").exists())
        );
    }

    /// Bundled files are part of a skill, so a delete that left them behind would produce a broken
    /// half-skill that discovery keeps warning about.
    #[tokio::test]
    async fn test_skill_delete_removes_bundled_files_too() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(temp.path(), "bundled", "---\ndescription: x\n---\nbody\n");
        std::fs::write(temp.path().join("bundled/helper.sh"), "#!/bin/sh\n").expect("write helper");

        let delete = SkillDeleteTool {
            skills: cache_at(&temp),
        };
        let result = run(&delete, serde_json::json!({"name": "bundled"})).await;
        assert!(!result.is_error, "{}", text_of(&result));
        assert!(!temp.path().join("bundled").exists());
    }

    #[tokio::test]
    async fn test_skill_delete_reports_a_missing_skill() {
        let temp = tempfile::tempdir().expect("tempdir");
        let delete = SkillDeleteTool {
            skills: cache_at(&temp),
        };
        let result = run(&delete, serde_json::json!({"name": "absent"})).await;
        assert!(result.is_error);
        assert!(text_of(&result).contains("not found"));
    }

    /// A directory whose `SKILL.md` does not parse is absent from every index, so "not found" is a
    /// lie the user can disprove with `ls`. Both tools refuse it, and say which case it is.
    #[tokio::test]
    async fn test_both_tools_distinguish_a_broken_skill_from_a_missing_one() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(temp.path(), "broken", "no frontmatter at all\nKEEP ME\n");
        let skills = cache_at(&temp);

        let delete = SkillDeleteTool {
            skills: skills.clone(),
        };
        let result = run(&delete, serde_json::json!({"name": "broken"})).await;
        assert!(result.is_error);
        let text = text_of(&result);
        assert!(text.contains("not a valid skill"), "{text}");
        assert!(!text.contains("not found"), "{text}");

        let write = SkillWriteTool { skills };
        let result = run(
            &write,
            serde_json::json!({"name": "broken", "description": "d", "body": "new"}),
        )
        .await;
        assert!(result.is_error);
        assert!(text_of(&result).contains("not a valid skill"));

        assert!(
            std::fs::read_to_string(temp.path().join("broken/SKILL.md"))
                .expect("read")
                .contains("KEEP ME")
        );
    }

    /// Searching bodies is the whole point: a skill whose description says nothing about the term
    /// is exactly the one the pushed index cannot help with.
    #[tokio::test]
    async fn test_skill_search_matches_bodies_not_just_descriptions() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "deploy",
            "---\ndescription: Ship it\n---\nRun kubectl rollout status.\n",
        );
        write_skill(
            temp.path(),
            "unrelated",
            "---\ndescription: Something else\n---\nNothing to see.\n",
        );

        let search = SkillSearchTool {
            skills: cache_at(&temp),
        };
        let text = text_of(&run(&search, serde_json::json!({"pattern": "kubectl"})).await);
        assert!(text.contains("deploy:"), "{}", text);
        assert!(!text.contains("unrelated"), "{}", text);

        let text = text_of(&run(&search, serde_json::json!({"pattern": "zzz-no-match"})).await);
        assert!(text.contains("No skills matched"), "{}", text);
    }

    /// Priority arrives through three doors (frontmatter, CLI flag, this schema) and they do not
    /// agree by accident: a number outside the range is clamped, but a non-number is refused
    /// outright rather than silently becoming the default.
    #[tokio::test]
    async fn test_skill_write_clamps_a_wild_priority_and_refuses_a_non_number() {
        let temp = tempfile::tempdir().expect("tempdir");
        let skills = cache_at(&temp);
        let write = SkillWriteTool {
            skills: skills.clone(),
        };

        for (name, given, expected) in [("low", -5, 0u8), ("high", 99, 9)] {
            run(
                &write,
                serde_json::json!({"name": name, "description": "d", "priority": given}),
            )
            .await;
            let found = skills.current().await;
            let skill = found
                .skills
                .iter()
                .find(|s| s.name == name)
                .expect(name)
                .clone();
            assert_eq!(skill.priority, expected, "{name}");
        }

        let result = write
            .execute(
                serde_json::json!({"name": "words", "description": "d", "priority": "high"}),
                CancellationToken::new(),
            )
            .await;
        assert!(result.is_err(), "a non-number priority must not be guessed");
    }

    /// The tail matters as much as the matches: without it a truncated result reads as the whole
    /// answer, which is the same failure the capped index exists to avoid.
    #[tokio::test]
    async fn test_skill_search_reports_when_it_stopped_early() {
        let temp = tempfile::tempdir().expect("tempdir");
        let body: String = (0..MAX_SEARCH_MATCHES + 20)
            .map(|index| format!("needle line {index}\n"))
            .collect();
        write_skill(
            temp.path(),
            "haystack",
            &format!("---\ndescription: x\n---\n{}", body),
        );

        let search = SkillSearchTool {
            skills: cache_at(&temp),
        };
        let text = text_of(&run(&search, serde_json::json!({"pattern": "needle"})).await);
        assert_eq!(
            text.lines().filter(|l| l.contains("needle")).count(),
            MAX_SEARCH_MATCHES
        );
        assert!(text.contains("narrow the pattern"), "{text}");
    }

    #[tokio::test]
    async fn test_skill_search_rejects_an_invalid_regex() {
        let temp = tempfile::tempdir().expect("tempdir");
        let search = SkillSearchTool {
            skills: cache_at(&temp),
        };
        let result = search
            .execute(
                serde_json::json!({"pattern": "[unclosed"}),
                CancellationToken::new(),
            )
            .await;
        assert!(result.is_err());
    }

    /// A rootless cache means "nowhere to write to", which is a different failure from an empty
    /// store and has to say so rather than reporting success against a path that does not exist.
    #[tokio::test]
    async fn test_write_without_a_root_fails_with_a_reason() {
        let write = SkillWriteTool {
            skills: SkillCache::for_root(None),
        };
        let error = write
            .execute(
                serde_json::json!({"name": "x", "description": "d"}),
                CancellationToken::new(),
            )
            .await
            .expect_err("a rootless cache has nowhere to write");
        assert!(error.to_string().contains("disabled"), "{}", error);
    }

    /// An omitted priority keeps the one the skill already has, matching `PUT /v1/skills` and this
    /// tool's own "omit body and whatever the skill already documented is kept".
    ///
    /// Reading the absence as the default demoted a prioritised skill every time the agent refined
    /// its text. Priority orders the `[Skills]` index the model reads *and* decides which entries
    /// the index cap drops, so the demotion can take the skill out of view entirely.
    #[tokio::test]
    async fn test_skill_write_keeps_an_omitted_priority() {
        let temp = tempfile::tempdir().expect("tempdir");
        let skills = cache_at(&temp);
        let write = SkillWriteTool {
            skills: skills.clone(),
        };

        run(
            &write,
            serde_json::json!({"name": "ranked", "description": "first", "priority": 1}),
        )
        .await;
        let result = run(
            &write,
            serde_json::json!({"name": "ranked", "description": "refined"}),
        )
        .await;
        assert!(
            text_of(&result).contains("(priority 1)"),
            "the confirmation must state what landed: {}",
            text_of(&result)
        );

        let discovered = skills.current().await;
        let skill = discovered
            .skills
            .iter()
            .find(|skill| skill.name == "ranked")
            .expect("skill");
        assert_eq!(skill.priority, 1, "a refinement must not demote it");
        assert_eq!(skill.description, "refined");
    }
}
