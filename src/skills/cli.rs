//! Handlers for the `meka skill <subcommand>` CLI: list, get, show, add, remove, update. Mirrors
//! the structure of [`crate::mcp::cli`]: each handler returns `Result<()>`, prints parseable data
//! to stdout (the user requested it; pipes / scripts read from there) and lifecycle / diagnostic
//! messages via `tracing` per the project's logging guidelines.

use std::path::Path;

use crate::{
    error::{MekaError, Result},
    skills,
};

const DESCRIPTION_TRUNCATE: usize = 40;

/// Argument bag for [`run_add`]. Borrowed so callers don't have to clone every field out of the
/// clap-derived `cli::SkillAction::Add` variant.
pub struct AddArgs<'a> {
    pub name: &'a str,
    pub description: Option<&'a str>,
    pub version: Option<&'a str>,
    pub author: Option<&'a str>,
    pub source_url: Option<&'a str>,
    pub from_file: Option<&'a Path>,
    pub force: bool,
    pub edit: bool,
}

/// `meka skill list`: print a tab-separated table of every installed skill. Empty case prints `(no
/// skills installed)` so scripts grepping the output don't get a confusing zero-byte result.
pub async fn run_list() -> Result<()> {
    let skills = skills::discover_skills();
    print_list(&skills);
    Ok(())
}

fn print_list(skills: &[skills::Skill]) {
    if skills.is_empty() {
        println!("(no skills installed)");
        return;
    }

    let rows: Vec<Vec<String>> = skills
        .iter()
        .map(|skill| {
            vec![
                skill.name.clone(),
                skill.version.clone().unwrap_or_else(|| "-".to_string()),
                truncate(&skill.description, DESCRIPTION_TRUNCATE),
                skill.source_dir.display().to_string(),
            ]
        })
        .collect();

    print!(
        "{}",
        crate::render::format_columns(&["Name", "Version", "Description", "Path"], &rows)
    );
}

/// `meka skill get <name>`: dump frontmatter as `key: value` lines.
pub async fn run_get(name: &str) -> Result<()> {
    let skill = require_skill(name)?;
    let body_bytes = std::fs::metadata(&skill.body_path)
        .map(|m| m.len())
        .unwrap_or(0);
    println!("name: {}", skill.name);
    println!("source_dir: {}", skill.source_dir.display());
    println!("body_path: {}", skill.body_path.display());
    println!("description: {}", skill.description);
    println!("version: {}", skill.version.as_deref().unwrap_or("(unset)"));
    println!("author: {}", skill.author.as_deref().unwrap_or("(unset)"));
    println!(
        "source_url: {}",
        skill.source_url.as_deref().unwrap_or("(unset)")
    );
    println!("body: {} bytes", body_bytes);
    Ok(())
}

/// `meka skill show <name>`: print the body as the agent receives it, i.e. the base-directory
/// header followed by the body verbatim.
pub async fn run_show(name: &str) -> Result<()> {
    let skill = require_skill(name)?;
    let body = skills::load_skill_body(&skill)
        .await
        .map_err(|error| MekaError::Config(format!("failed to load skill body: {}", error)))?;
    print!("{}", body);
    Ok(())
}

/// `meka skill add <name> [flags]`: scaffold a new skill directory.
pub async fn run_add(args: AddArgs<'_>) -> Result<()> {
    skills::validate_skill_name(args.name).map_err(MekaError::Config)?;

    let dir = skills::skill_dir_for(args.name)
        .ok_or_else(|| MekaError::Config("could not resolve meka config directory".to_string()))?;

    if dir.exists() {
        if !args.force {
            return Err(MekaError::Config(format!(
                "skill '{}' already exists at {}; pass --force to overwrite",
                args.name,
                dir.display()
            )));
        }
        tokio::fs::remove_dir_all(&dir).await.map_err(|error| {
            MekaError::Config(format!("failed to remove {}: {}", dir.display(), error))
        })?;
    }

    let body = build_skill_body(&args)?;

    tokio::fs::create_dir_all(&dir).await.map_err(|error| {
        MekaError::Config(format!(
            "failed to create skill dir {}: {}",
            dir.display(),
            error
        ))
    })?;

    let skill_md = dir.join("SKILL.md");
    tokio::fs::write(&skill_md, &body).await.map_err(|error| {
        MekaError::Config(format!("failed to write {}: {}", skill_md.display(), error))
    })?;

    tracing::info!("created skill '{}' at {}", args.name, skill_md.display());
    println!("{}", skill_md.display());

    if args.edit {
        if let Some(editor) = std::env::var_os("EDITOR")
            && !editor.is_empty()
        {
            let status = std::process::Command::new(&editor)
                .arg(&skill_md)
                .status()
                .map_err(|error| {
                    MekaError::Config(format!("failed to launch $EDITOR: {}", error))
                })?;
            if !status.success() {
                tracing::warn!("$EDITOR exited with non-zero status: {:?}", status.code());
            }
        } else {
            tracing::info!("--edit was requested but $EDITOR is unset; skipping");
        }
    }

    Ok(())
}

fn build_skill_body(args: &AddArgs<'_>) -> Result<String> {
    if let Some(path) = args.from_file {
        if args.description.is_some() {
            return Err(MekaError::Config(
                "--from-file is mutually exclusive with --description".to_string(),
            ));
        }
        let content = std::fs::read_to_string(path).map_err(|error| {
            MekaError::Config(format!("failed to read {}: {}", path.display(), error))
        })?;
        Ok(content)
    } else {
        let description = args.description.ok_or_else(|| {
            MekaError::Config(
                "--description is required (or pass --from-file to copy a template)".to_string(),
            )
        })?;
        Ok(skills::render_template(
            args.name,
            description,
            args.version,
            args.author,
            args.source_url,
        ))
    }
}

/// `meka skill remove <name>`: delete the skill directory. No prompt; matches `meka mcp remove`'s
/// convention.
pub async fn run_remove(name: &str) -> Result<()> {
    skills::validate_skill_name(name).map_err(MekaError::Config)?;
    let dir = skills::skill_dir_for(name)
        .ok_or_else(|| MekaError::Config("could not resolve meka config directory".to_string()))?;
    if !dir.exists() {
        return Err(MekaError::Config(format!(
            "skill '{}' not found at {}",
            name,
            dir.display()
        )));
    }
    tokio::fs::remove_dir_all(&dir).await.map_err(|error| {
        MekaError::Config(format!("failed to remove {}: {}", dir.display(), error))
    })?;
    tracing::info!("removed skill '{}'", name);
    Ok(())
}

/// Outcome of a single skill re-fetch.
#[derive(Debug)]
enum UpdateOutcome {
    /// Fetched content was byte-identical to what's on disk: nothing written (avoids bumping mtime
    /// / a spurious skill-cache reload).
    Unchanged,
    Updated {
        from: Option<String>,
        to: Option<String>,
    },
}

fn version_label(version: &Option<String>) -> &str {
    version.as_deref().unwrap_or("unversioned")
}

/// `meka skill update [<name>] [--all] [--yes]`: re-fetch skills that declare a `source_url` and
/// replace them on disk.
pub async fn run_update(name: Option<&str>, all: bool, yes: bool) -> Result<()> {
    match (name, all) {
        (Some(_), true) => Err(MekaError::Config(
            "pass either a skill name or --all, not both".to_string(),
        )),
        (None, false) => Err(MekaError::Config(
            "specify a skill name, or pass --all to update every skill".to_string(),
        )),
        (Some(name), false) => update_one(name).await,
        (None, true) => update_all(yes).await,
    }
}

async fn update_one(name: &str) -> Result<()> {
    let skill = require_skill(name)?;
    if skill.source_url.is_none() {
        return Err(MekaError::Config(format!(
            "skill '{}' has no source_url; nothing to update",
            name
        )));
    }
    match fetch_and_replace_skill(&skill).await? {
        UpdateOutcome::Unchanged => println!("{}: unchanged", name),
        UpdateOutcome::Updated { from, to } => println!(
            "{}: updated ({} -> {})",
            name,
            version_label(&from),
            version_label(&to),
        ),
    }
    Ok(())
}

async fn update_all(yes: bool) -> Result<()> {
    let updatable: Vec<skills::Skill> = skills::discover_skills()
        .into_iter()
        .filter(|skill| skill.source_url.is_some())
        .collect();

    if updatable.is_empty() {
        println!("(no skills declare a source_url)");
        return Ok(());
    }

    // Dry run: --all without --yes lists what would change and stops. This is the confirmation gate
    // for a bulk remote fetch.
    if !yes {
        println!("Skills that would be updated (re-run with --yes to apply):");
        for skill in &updatable {
            println!(
                "  {}\t{}\t{}",
                skill.name,
                version_label(&skill.version),
                skill.source_url.as_deref().unwrap_or(""),
            );
        }
        return Ok(());
    }

    // A per-skill failure is reported and does not abort the rest.
    for skill in &updatable {
        match fetch_and_replace_skill(skill).await {
            Ok(UpdateOutcome::Unchanged) => println!("{}: unchanged", skill.name),
            Ok(UpdateOutcome::Updated { from, to }) => println!(
                "{}: updated ({} -> {})",
                skill.name,
                version_label(&from),
                version_label(&to),
            ),
            Err(error) => eprintln!("{}: update failed: {}", skill.name, error),
        }
    }
    Ok(())
}

/// Fetch a skill's `source_url`, validate the response parses as a skill, and atomically replace
/// the on-disk `SKILL.md`. A failed fetch or a malformed response leaves the existing file
/// untouched; validation happens entirely in memory before any write.
async fn fetch_and_replace_skill(skill: &skills::Skill) -> Result<UpdateOutcome> {
    let url = skill
        .source_url
        .as_deref()
        .ok_or_else(|| MekaError::Config(format!("skill '{}' has no source_url", skill.name)))?;

    // Explicit scheme check for a clear error; `https_only(true)` below is the defense-in-depth
    // that also blocks an http downgrade on a redirect (GitHub-raw / gist URLs do redirect).
    if !url.starts_with("https://") {
        return Err(MekaError::Config(format!(
            "skill '{}' source_url must be https://, got: {}",
            skill.name, url
        )));
    }

    let client = reqwest::Client::builder()
        .https_only(true)
        .timeout(std::time::Duration::from_secs(30))
        .user_agent(crate::config::DEFAULT_WEB_USER_AGENT)
        .build()
        .map_err(|error| MekaError::Config(format!("failed to build HTTP client: {}", error)))?;

    let fetched = client
        .get(url)
        .send()
        .await
        .map_err(|error| {
            MekaError::Config(format!("fetch failed for '{}': {}", skill.name, error))
        })?
        .error_for_status()
        .map_err(|error| {
            MekaError::Config(format!("fetch failed for '{}': {}", skill.name, error))
        })?
        .text()
        .await
        .map_err(|error| {
            MekaError::Config(format!(
                "failed to read response body for '{}': {}",
                skill.name, error
            ))
        })?;

    // Validate in memory: never overwrite the on-disk skill with a 404 page, a non-skill file, or
    // malformed frontmatter.
    let parsed =
        skills::parse_skill_definition(&skill.name, &skill.source_dir, &skill.body_path, &fetched)
            .map_err(|error| {
                MekaError::Config(format!(
                    "fetched content for '{}' is not a valid skill: {}",
                    skill.name, error
                ))
            })?;

    let current = tokio::fs::read_to_string(&skill.body_path)
        .await
        .unwrap_or_else(|error| {
            tracing::warn!(
                "could not read existing skill body {}: {}",
                skill.body_path.display(),
                error
            );
            String::new()
        });
    if current == fetched {
        return Ok(UpdateOutcome::Unchanged);
    }

    crate::config::write_file_atomic(&skill.body_path, &fetched).map_err(|error| {
        MekaError::Config(format!(
            "failed to write {}: {}",
            skill.body_path.display(),
            error
        ))
    })?;

    Ok(UpdateOutcome::Updated {
        from: skill.version.clone(),
        to: parsed.version,
    })
}

pub(crate) fn require_skill(name: &str) -> Result<skills::Skill> {
    let skills = skills::discover_skills();
    skills
        .into_iter()
        .find(|skill| skill.name == name)
        .ok_or_else(|| MekaError::Config(format!("no skill named '{}'", name)))
}

fn truncate(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Holds the env-lock and clears `MEKA_CONFIG_DIR` when dropped, so the var never outlives the
    /// tempdir it points at. The `config.rs` users of the same lock unset it symmetrically; leaving
    /// it set here would mean two modules sharing one lock while disagreeing on what state they
    /// hand back to each other.
    struct ConfigDirGuard(#[allow(dead_code)] tokio::sync::MutexGuard<'static, ()>);

    impl Drop for ConfigDirGuard {
        fn drop(&mut self) {
            // SAFETY: still under the lock; the guard field is dropped after this.
            unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
        }
    }

    /// Acquire the env-lock and point `MEKA_CONFIG_DIR` at `temp`. The returned guard must be held
    /// by the caller for the lifetime of the test; dropping it clears the var and releases the lock
    /// so the next test can run.
    ///
    /// Uses the crate-wide [`crate::config::CONFIG_DIR_ENV_LOCK`] rather than a lock private to
    /// this module: the var is process-global, so a per-module lock leaves these tests racing the
    /// ones in `config.rs`, and losing that race points `run_add` at the developer's real config
    /// directory.
    async fn isolate_config_dir(temp: &tempfile::TempDir) -> ConfigDirGuard {
        let guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        // SAFETY: the mutex makes this access exclusive across tests in this process; no other code
        // reads the var while the lock is held. Matches the env-var override at
        // `src/config.rs:462-467`.
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", temp.path()) };
        ConfigDirGuard(guard)
    }

    fn add_args<'a>(name: &'a str, description: &'a str) -> AddArgs<'a> {
        AddArgs {
            name,
            description: Some(description),
            version: None,
            author: None,
            source_url: None,
            from_file: None,
            force: false,
            edit: false,
        }
    }

    #[tokio::test]
    async fn test_run_add_then_discover_round_trip() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        run_add(add_args("demo", "demo desc")).await.expect("add");

        let skills = skills::discover_skills();
        assert_eq!(skills.len(), 1);
        let skill = &skills[0];
        assert_eq!(skill.name, "demo");
        assert_eq!(skill.description, "demo desc");
    }

    #[tokio::test]
    async fn test_run_add_rejects_existing_without_force() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        run_add(add_args("dup", "first")).await.expect("first add");
        let err = run_add(add_args("dup", "second"))
            .await
            .expect_err("second add should fail");
        assert!(format!("{}", err).contains("already exists"));
    }

    #[tokio::test]
    async fn test_run_add_force_overwrites() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        run_add(add_args("over", "old")).await.expect("first add");
        let mut args = add_args("over", "new");
        args.force = true;
        run_add(args).await.expect("force overwrite");

        let skills = skills::discover_skills();
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].description, "new");
    }

    #[tokio::test]
    async fn test_run_add_with_source_url_round_trips() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let mut args = add_args("sourced", "a sourced skill");
        args.version = Some("1.0");
        args.author = Some("John Doe <john.doe@example.com>");
        args.source_url = Some("https://example.com/SKILL.md");
        run_add(args).await.expect("add");

        let skill = require_skill("sourced").expect("found");
        assert_eq!(skill.version.as_deref(), Some("1.0"));
        assert_eq!(
            skill.author.as_deref(),
            Some("John Doe <john.doe@example.com>")
        );
        assert_eq!(
            skill.source_url.as_deref(),
            Some("https://example.com/SKILL.md")
        );
    }

    #[tokio::test]
    async fn test_run_add_from_file_copies_verbatim() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let template = temp.path().join("template.md");
        let body = "---\ndescription: tpl desc\n---\n# Templated\n\nbody.\n";
        std::fs::write(&template, body).expect("write template");

        let args = AddArgs {
            name: "tpl",
            description: None,
            version: None,
            author: None,
            source_url: None,
            from_file: Some(&template),
            force: false,
            edit: false,
        };
        run_add(args).await.expect("from-file add");

        let skills = skills::discover_skills();
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].description, "tpl desc");
    }

    #[tokio::test]
    async fn test_run_add_from_file_rejects_with_description_flag() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let template = temp.path().join("template.md");
        std::fs::write(&template, "---\ndescription: x\n---\n").expect("write");

        let args = AddArgs {
            name: "tpl",
            description: Some("collides"),
            version: None,
            author: None,
            source_url: None,
            from_file: Some(&template),
            force: false,
            edit: false,
        };
        let err = run_add(args).await.expect_err("should reject");
        assert!(format!("{}", err).contains("mutually exclusive"));
    }

    #[tokio::test]
    async fn test_run_add_requires_description_without_from_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let args = AddArgs {
            name: "needs",
            description: None,
            version: None,
            author: None,
            source_url: None,
            from_file: None,
            force: false,
            edit: false,
        };
        let err = run_add(args).await.expect_err("should reject");
        assert!(format!("{}", err).contains("--description is required"));
    }

    #[tokio::test]
    async fn test_run_remove_deletes_directory() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        run_add(add_args("gone", "x")).await.expect("add");
        assert_eq!(skills::discover_skills().len(), 1);

        run_remove("gone").await.expect("remove");
        assert!(skills::discover_skills().is_empty());
    }

    #[tokio::test]
    async fn test_run_remove_errors_on_missing() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let err = run_remove("ghost").await.expect_err("should error");
        assert!(format!("{}", err).contains("not found"));
    }

    #[tokio::test]
    async fn test_run_get_errors_on_missing() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let err = run_get("ghost").await.expect_err("should error");
        assert!(format!("{}", err).contains("no skill named"));
    }

    #[tokio::test]
    async fn test_run_update_requires_name_or_all() {
        let err = run_update(None, false, false)
            .await
            .expect_err("should error");
        assert!(format!("{}", err).contains("specify a skill name"));
    }

    #[tokio::test]
    async fn test_run_update_rejects_name_and_all_together() {
        let err = run_update(Some("x"), true, false)
            .await
            .expect_err("should error");
        assert!(format!("{}", err).contains("not both"));
    }

    #[tokio::test]
    async fn test_run_update_errors_when_skill_has_no_source_url() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        run_add(add_args("local", "no source url"))
            .await
            .expect("add");
        let err = run_update(Some("local"), false, false)
            .await
            .expect_err("should error");
        assert!(format!("{}", err).contains("no source_url"));
    }

    #[tokio::test]
    async fn test_fetch_and_replace_rejects_non_https() {
        // The scheme check fires before any network call, so this needs no server. A non-https
        // source_url must error and never write.
        let skill = skills::Skill {
            name: "insecure".to_string(),
            source_dir: std::path::PathBuf::from("/tmp/insecure"),
            description: "x".to_string(),
            version: None,
            author: None,
            source_url: Some("http://example.com/SKILL.md".to_string()),
            body_path: std::path::PathBuf::from("/tmp/insecure/SKILL.md"),
        };
        let err = fetch_and_replace_skill(&skill)
            .await
            .expect_err("should reject http://");
        assert!(format!("{}", err).contains("https://"));
    }

    #[tokio::test]
    async fn test_run_show_renders_header_and_verbatim_body() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        run_add(add_args("subst", "desc")).await.expect("add");

        let dir = skills::skill_dir_for("subst").expect("dir resolves");
        let body = "---\ndescription: x\n---\nRun scripts/helper.sh\nLiteral ${MEKA_SKILL_DIR}\n";
        std::fs::write(dir.join("SKILL.md"), body).expect("rewrite");

        // run_show prints to stdout; exercise the loader directly, since capturing stdout in tests
        // is brittle.
        let skill = require_skill("subst").expect("found");
        let rendered = skills::load_skill_body(&skill).await.expect("load");
        // The header names the base directory; the body itself is untouched.
        assert!(rendered.contains(&dir.display().to_string()));
        assert!(rendered.contains("Run scripts/helper.sh"));
        assert!(rendered.contains("Literal ${MEKA_SKILL_DIR}"));
    }

    #[test]
    fn test_truncate_short() {
        assert_eq!(truncate("hello", 40), "hello");
    }

    #[test]
    fn test_truncate_long() {
        let long = "a".repeat(80);
        let truncated = truncate(&long, 40);
        assert_eq!(truncated.chars().count(), 40);
        assert!(truncated.ends_with('…'));
    }
}
