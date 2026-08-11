//! `skill` tool: loads a named skill body, prefixed with its base directory, so the skill's
//! instructions become available to the agent on demand.

use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::{Tool, ToolOutput, util::require_str};
use crate::{
    error::{MekaError, Result},
    permission::Permission,
    provider::ToolDefinition,
    skills::{self, Skill, SkillCache},
};

pub(super) struct SkillTool {
    /// Shared skill cache with the agent. Dispatch reads through `current().await` so the tool
    /// sees any auto-reloads that happened during the turn.
    pub skills: Arc<SkillCache>,
}

#[async_trait]
impl Tool for SkillTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "skill".to_string(),
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
        let name = require_str(&input, "name", "skill")?;
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
                    tool_name: "skill".to_string(),
                    message: error,
                })?;

        Ok(ToolOutput::text(body, false))
    }
}

fn find_skill<'a>(skills: &'a [Skill], name: &str) -> Option<&'a Skill> {
    skills.iter().find(|skill| skill.name == name)
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
        let tool = SkillTool {
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
        let tool = SkillTool {
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
        let tool = SkillTool {
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
}
