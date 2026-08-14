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
    skills::{self, Skill, SkillCache},
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

        let skill = match find_skill(&skills, &name) {
            Some(skill) => skill,
            None => {
                let available: Vec<&str> = skills.iter().map(|skill| skill.name.as_str()).collect();
                let hint = if available.is_empty() {
                    "No skills are installed.".to_string()
                } else {
                    format!("Available skills: {}", available.join(", "))
                };
                return Ok(ToolOutput::text(
                    format!("Error: skill '{}' not found. {}", name, hint),
                    true,
                ));
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

fn find_skill<'a>(skills: &'a [Skill], name: &str) -> Option<&'a Skill> {
    skills.iter().find(|skill| skill.name == name)
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

/// Refuse to write or delete a skill that declares a `source_url`.
///
/// Those are upstream-managed: `meka skill update` re-fetches and overwrites them, so an agent edit
/// is not merely risky but *futile*, and would be reverted with no warning at the worst possible
/// moment. Returning the reason rather than a bare refusal is the point, since the agent can act on
/// it by choosing a different name.
///
/// This deliberately does not protect a hand-written skill with no `source_url`. Nothing can,
/// without provenance in the frontmatter, which meka rejected for memory on the grounds that a
/// store shared by a human and an agent should not sort its entries by who typed them. The real
/// guard is `[skills] agent_managed` being off by default.
fn reject_upstream_managed(skill: &Skill) -> Option<ToolOutput> {
    let source_url = skill.source_url.as_deref()?;
    Some(ToolOutput::text(
        format!(
            "Error: skill '{}' is managed upstream (source_url: {}). `meka skill update` \
             re-fetches it, so any change here would be silently reverted. Write to a different \
             name, or ask the user to change it at the source.",
            skill.name, source_url
        ),
        true,
    ))
}

/// Refuse to touch a directory that holds a `SKILL.md` discovery could not parse.
///
/// Absent from the index is not the same as absent from disk. Such a file is skipped with a warning
/// and appears in no index and no listing, so neither the model nor this tool can say what is in
/// it, and its only copy is that file. Reporting it as "not found" while it sits in the skills
/// directory is the confusion `[Memory]`'s skip reporting exists to prevent, so name the case
/// instead.
///
/// [`skills::write_skill`] refuses the same case independently; this exists so the refusal arrives
/// as a readable tool result rather than a tool error, and so `skill_delete` gets it too.
fn reject_unreadable(
    root: &std::path::Path,
    name: &str,
    installed: &[Skill],
) -> Option<ToolOutput> {
    // The *file*, not the directory. A bare `skills/<name>/` with no `SKILL.md` is a half-finished
    // `meka skill add`, a partly-copied folder, or the residue of an interrupted write: there is
    // nothing in it to lose, and `write_skill` would happily create the skill. Refusing on the
    // directory blocked a legitimate create and claimed a `SKILL.md` that was not there.
    if find_skill(installed, name).is_some() || !root.join(name).join("SKILL.md").is_file() {
        return None;
    }
    Some(ToolOutput::text(
        format!(
            "Error: '{}' exists on disk but its SKILL.md is not a valid skill, so it is in no \
             index and its contents cannot be shown. Leaving it untouched rather than overwriting \
             something neither of us can see. Use a different name, or ask the user to fix or \
             remove it with `meka skill remove {}`.",
            name, name
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
        for skill in skills.iter() {
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
                        "description": "Identifier, letters/digits/-/_ only (e.g. 'triage-build-failure')"
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
        let priority = match input.get("priority") {
            Some(serde_json::Value::Null) | None => crate::store::DEFAULT_PRIORITY,
            Some(value) => {
                let raw = value.as_i64().ok_or_else(|| MekaError::ToolExecution {
                    tool_name: "skill_write".to_string(),
                    message: format!("'priority' must be a whole number, got {}", value),
                })?;
                crate::store::parse_priority(Some(raw), "skill", &name)
            }
        };

        let installed = self.skills.current().await;
        if let Some(refusal) = reject_unreadable(&root, &name, &installed) {
            return Ok(refusal);
        }
        if let Some(existing) = find_skill(&installed, &name)
            && let Some(refusal) = reject_upstream_managed(existing)
        {
            return Ok(refusal);
        }

        // Read before the write, since the write is what makes the file exist: otherwise the
        // confirmation would claim to have kept the body of a skill that had none.
        let kept_existing_body = body.is_none() && installed.iter().any(|s| s.name == name);

        let path = skills::write_skill(
            &root,
            &name,
            &description,
            priority,
            Some(AGENT_AUTHOR),
            body,
        )
        .map_err(|message| MekaError::ToolExecution {
            tool_name: "skill_write".to_string(),
            message,
        })?;

        tracing::info!("saved skill to {}", path.display());
        Ok(ToolOutput::text(
            // Deliberately promises reachability by name rather than a place in the index. The
            // index is capped, so a low-priority skill in a large store may not be listed there,
            // and `skill_read` / `agent_spawn` work either way.
            format!(
                "Saved skill '{}' (priority {}){}. From the next turn on you can load it with \
                 skill_read, or hand it to a worker with agent_spawn(skill: \"{}\").",
                name,
                priority,
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
        skills::validate_skill_name(&name).map_err(|message| MekaError::ToolExecution {
            tool_name: "skill_delete".to_string(),
            message,
        })?;

        let installed = self.skills.current().await;
        if let Some(refusal) = reject_unreadable(&root, &name, &installed) {
            return Ok(refusal);
        }
        match find_skill(&installed, &name) {
            Some(existing) => {
                if let Some(refusal) = reject_upstream_managed(existing) {
                    return Ok(refusal);
                }
            }
            None => {
                return Ok(ToolOutput::text(
                    format!("Error: skill '{}' not found.", name),
                    true,
                ));
            }
        }

        let dir =
            skills::delete_skill(&root, &name).map_err(|message| MekaError::ToolExecution {
                tool_name: "skill_delete".to_string(),
                message,
            })?;

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
    fn test_find_skill() {
        let skill = Skill {
            name: "foo".to_string(),
            source_dir: std::path::PathBuf::from("/tmp"),
            description: "desc".to_string(),
            version: None,
            author: None,
            source_url: None,
            priority: crate::store::DEFAULT_PRIORITY,
            body_path: std::path::PathBuf::from("/tmp/SKILL.md"),
        };
        let skills = vec![skill];
        assert!(find_skill(&skills, "foo").is_some());
        assert!(find_skill(&skills, "bar").is_none());
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
            .iter()
            .find(|skill| skill.name == "triage")
            .expect("written skill must be discoverable");
        assert_eq!(skill.description, "How to triage a build failure");
        assert_eq!(skill.priority, 2);
        assert_eq!(skill.author.as_deref(), Some(AGENT_AUTHOR));

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
        let skill = discovered.iter().find(|s| s.name == "keep").expect("keep");
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

    /// `meka skill update` re-fetches anything with a `source_url`, so an agent edit there would be
    /// reverted with no warning. Both tools refuse, and say why.
    #[tokio::test]
    async fn test_write_and_delete_refuse_upstream_managed_skills() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "vendored",
            "---\ndescription: x\nsource_url: https://example.com/SKILL.md\n---\nUPSTREAM\n",
        );
        let skills = cache_at(&temp);

        let write = SkillWriteTool {
            skills: skills.clone(),
        };
        let result = run(
            &write,
            serde_json::json!({"name": "vendored", "description": "mine", "body": "MINE"}),
        )
        .await;
        assert!(result.is_error);
        assert!(text_of(&result).contains("managed upstream"));

        let delete = SkillDeleteTool {
            skills: skills.clone(),
        };
        let result = run(&delete, serde_json::json!({"name": "vendored"})).await;
        assert!(result.is_error);
        assert!(text_of(&result).contains("managed upstream"));

        // Neither refusal touched the file.
        assert!(
            std::fs::read_to_string(temp.path().join("vendored/SKILL.md"))
                .expect("still there")
                .contains("UPSTREAM")
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
            let skill = found.iter().find(|s| s.name == name).expect(name).clone();
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
}
