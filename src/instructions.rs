//! User-authored instructions: standing guidance appended to meka's own system prompt.
//!
//! Content, not configuration. Like [`crate::skills`] and [`crate::memory`], it lives at a
//! conventional path under the config directory rather than behind a key in `config.toml`, because
//! prose long enough to be worth writing is miserable to maintain inside a TOML string.
//!
//! Two shapes are accepted, mirroring the familiar `conf.d` convention: a single
//! `<config>/instructions.md`, or a `<config>/instructions/` directory whose `*.md` files are
//! concatenated in lexical order so a large set can be split (`00-style.md`, `10-security.md`, …).
//! The directory wins when both exist.
//!
//! Unlike skills and memory this is read **once**, at startup, and lands in the system prompt
//! rather than the per-turn `<context>` block. That is deliberate and follows from size: the system
//! prompt is the cached prefix, so a large instruction set is billed once and re-read from cache on
//! every later turn. Re-reading it per turn would either invalidate that prefix whenever the file
//! changed or push the text into the conversation, where it would compete with actual context. The
//! cost is that edits take effect on the next run -- which for `meka serve` and `meka acp` means
//! the next *restart*, since one process serves every session. Documented for users at
//! `docs/book/src/configuration/overview.md` § "When edits take effect", beside the same caveat
//! about `config.toml`.

use std::path::{Path, PathBuf};

/// Ceiling above which an instruction set is reported as suspiciously large. Not enforced: a big
/// preamble can be exactly what the user wants, and it is cached, so this warns rather than
/// truncates. Roughly 8k tokens, which is a noticeable slice of even a large window.
const LARGE_INSTRUCTIONS_TOKENS: u64 = 8_000;

/// Cap on how many `*.md` files one `instructions/` directory contributes, mirroring
/// [`crate::memory`]'s bound on an unpruned store. A directory past this is far more likely to be
/// pointed at the wrong place than to be intentional.
const MAX_INSTRUCTION_FILES: usize = 100;

/// `<config>/instructions.md`, the single-file form.
pub fn instructions_file() -> Option<PathBuf> {
    crate::config::meka_config_dir().map(|dir| dir.join("instructions.md"))
}

/// `<config>/instructions/`, the split form.
pub fn instructions_dir() -> Option<PathBuf> {
    crate::config::meka_config_dir().map(|dir| dir.join("instructions"))
}

/// Where a resolved instruction set came from, so `meka instructions show` can answer "why is the
/// model being told this" without the user guessing at precedence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstructionsSource {
    /// `--instructions`.
    Flag,
    /// `MEKA_INSTRUCTIONS`.
    Env,
    /// `MEKA_INSTRUCTIONS_FILE`, or the conventional path. Carries every file that contributed, in
    /// the order they were concatenated.
    Files(Vec<PathBuf>),
}

impl std::fmt::Display for InstructionsSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Flag => write!(formatter, "--instructions"),
            Self::Env => write!(formatter, "MEKA_INSTRUCTIONS"),
            Self::Files(paths) => {
                let rendered: Vec<String> = paths
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect();
                write!(formatter, "{}", rendered.join(", "))
            }
        }
    }
}

/// A resolved instruction set and where it came from.
#[derive(Debug, Clone)]
pub struct Instructions {
    pub text: String,
    pub source: InstructionsSource,
}

/// Read the conventional location. `<config>/instructions/` takes precedence over
/// `<config>/instructions.md`, so splitting a grown file is a rename rather than a migration.
///
/// A missing path is simply "no instructions" and returns `None`; an unreadable one warns and is
/// skipped, so a single bad file never hides the rest of a directory. Returns `None` for a set that
/// is entirely whitespace, matching the treatment of an empty `--instructions`.
pub fn discover() -> Option<Instructions> {
    if let Some(dir) = instructions_dir()
        && dir.is_dir()
        && let Some(found) = read_dir(&dir, FileFilter::MarkdownOnly)
    {
        return Some(found);
    }
    let file = instructions_file()?;
    read_file(&file).map(|text| Instructions {
        text,
        source: InstructionsSource::Files(vec![file]),
    })
}

/// Read one explicit path, for `MEKA_INSTRUCTIONS_FILE`. A directory is accepted and concatenated
/// the same way the conventional one is, so a ConfigMap mounted as a directory of keys works
/// without the caller having to know which shape meka expects.
///
/// Unlike [`discover`] a missing or unreadable path is an error rather than silence: the user named
/// it explicitly, so failing quietly would leave the agent running without the guidance they
/// believe they supplied.
pub fn read_explicit(path: &Path) -> crate::error::Result<Instructions> {
    if path.is_dir() {
        // Any regular file counts here, not just `*.md`. The user named this directory and nothing
        // else lives in it, whereas the conventional `instructions/` sits among meka's other
        // content and has to be choosier. It also happens to be what containers need: a
        // ConfigMap key is often just `instructions`, and requiring an extension would turn a
        // naming choice made elsewhere into a startup failure inside a pod.
        return read_dir(path, FileFilter::AnyRegularFile).ok_or_else(|| {
            crate::error::MekaError::Config(format!(
                "instructions directory '{}' has no readable files",
                path.display()
            ))
        });
    }
    let text = std::fs::read_to_string(path).map_err(|error| {
        crate::error::MekaError::Config(format!(
            "failed to read instructions file '{}': {}",
            path.display(),
            error
        ))
    })?;
    let text = text.trim().to_string();
    if text.is_empty() {
        return Err(crate::error::MekaError::Config(format!(
            "instructions file '{}' is empty",
            path.display()
        )));
    }
    Ok(Instructions {
        text,
        source: InstructionsSource::Files(vec![path.to_path_buf()]),
    })
}

/// Which files in a directory count as instructions.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FileFilter {
    /// The conventional `instructions/`, which shares the config directory with other things and so
    /// only takes `*.md`.
    MarkdownOnly,
    /// A directory the user named outright, where anything present was put there on purpose.
    AnyRegularFile,
}

/// Concatenate every readable file in `root` that `filter` accepts, lexically by file name so the
/// `NN-topic.md` idiom orders the result predictably. Returns `None` when nothing readable and
/// non-empty was found.
fn read_dir(root: &Path, filter: FileFilter) -> Option<Instructions> {
    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) => {
            tracing::warn!(
                "failed to read instructions directory '{}': {}",
                root.display(),
                error
            );
            return None;
        }
    };

    let mut paths: Vec<PathBuf> = entries
        .filter_map(|entry| match entry {
            Ok(entry) => Some(entry.path()),
            Err(error) => {
                tracing::warn!("skipping unreadable instructions entry: {}", error);
                None
            }
        })
        .filter(|path| {
            // `is_file` follows symlinks, which is what a Kubernetes ConfigMap mount is made of: a
            // `..data` symlink to a timestamped directory, plus one symlink per key. The
            // directories in that layout fail the check and drop out on their own.
            path.is_file()
                && (filter == FileFilter::AnyRegularFile
                    || path
                        .extension()
                        .and_then(|extension| extension.to_str())
                        .is_some_and(|extension| extension.eq_ignore_ascii_case("md")))
        })
        .collect();
    paths.sort();

    if paths.len() > MAX_INSTRUCTION_FILES {
        tracing::warn!(
            "instructions directory '{}' holds {} files; reading the first {}",
            root.display(),
            paths.len(),
            MAX_INSTRUCTION_FILES,
        );
        paths.truncate(MAX_INSTRUCTION_FILES);
    }

    let mut used: Vec<PathBuf> = Vec::new();
    let mut sections: Vec<String> = Vec::new();
    for path in paths {
        match read_file(&path) {
            Some(text) => {
                sections.push(text);
                used.push(path);
            }
            None => continue,
        }
    }

    if sections.is_empty() {
        return None;
    }
    Some(Instructions {
        // Blank line between files so two sets can't run their last and first lines together into
        // one accidental sentence.
        text: sections.join("\n\n"),
        source: InstructionsSource::Files(used),
    })
}

/// Read one file, trimmed. `None` for missing, unreadable, or blank, with a warning for the middle
/// case only: a missing file is the normal state for an install that never wrote one.
fn read_file(path: &Path) -> Option<String> {
    match std::fs::read_to_string(path) {
        Ok(text) => {
            let text = text.trim().to_string();
            (!text.is_empty()).then_some(text)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            tracing::warn!(
                "failed to read instructions file '{}': {}",
                path.display(),
                error
            );
            None
        }
    }
}

/// Report an instruction set large enough that the user should know it is there. Silent otherwise.
///
/// It rides the cached system-prompt prefix so the ongoing cost is small, but it still occupies
/// window that the conversation can't use, and a set this size is usually a surprise (a whole
/// document pasted in, or a directory pointed somewhere unintended) rather than a decision.
pub fn warn_if_large(instructions: &Instructions) {
    let estimate = crate::tokens::estimate_text(&instructions.text);
    if estimate > LARGE_INSTRUCTIONS_TOKENS {
        tracing::warn!(
            "instructions from {} are large (~{} tokens); they occupy that much of every request's \
             context window",
            instructions.source,
            estimate,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, body: &str) {
        std::fs::write(dir.join(name), body).expect("write fixture");
    }

    #[test]
    fn test_read_dir_concatenates_in_lexical_order() {
        let temp = tempfile::tempdir().expect("tempdir");
        write(temp.path(), "20-second.md", "second");
        write(temp.path(), "10-first.md", "first");
        write(temp.path(), "notes.txt", "ignored, not markdown");

        let found = read_dir(temp.path(), FileFilter::MarkdownOnly).expect("some");
        assert_eq!(found.text, "first\n\nsecond");
        match found.source {
            InstructionsSource::Files(paths) => {
                assert_eq!(paths.len(), 2, "only the *.md files contribute");
                assert!(paths[0].ends_with("10-first.md"));
            }
            other => panic!("expected Files, got {:?}", other),
        }
    }

    #[test]
    fn test_read_dir_skips_blank_files() {
        let temp = tempfile::tempdir().expect("tempdir");
        write(temp.path(), "10-real.md", "real content");
        write(temp.path(), "20-blank.md", "   \n\t\n");

        let found = read_dir(temp.path(), FileFilter::MarkdownOnly).expect("some");
        assert_eq!(found.text, "real content");
    }

    #[test]
    fn test_read_dir_on_empty_directory_is_none() {
        let temp = tempfile::tempdir().expect("tempdir");
        assert!(read_dir(temp.path(), FileFilter::MarkdownOnly).is_none());
    }

    /// An explicitly named path failing quietly would leave the agent running without the guidance
    /// the user believes they supplied, so it errors where discovery would shrug.
    #[test]
    fn test_read_explicit_errors_on_a_missing_path() {
        let temp = tempfile::tempdir().expect("tempdir");
        let missing = temp.path().join("nope.md");
        let error = read_explicit(&missing).expect_err("must not be silent");
        assert!(error.to_string().contains("nope.md"), "{error}");
    }

    #[test]
    fn test_read_explicit_errors_on_an_empty_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        write(temp.path(), "empty.md", "  \n ");
        let error = read_explicit(&temp.path().join("empty.md")).expect_err("must not be silent");
        assert!(error.to_string().contains("empty"), "{error}");
    }

    /// A ConfigMap mounts as a directory of keys, so an explicit path has to accept one.
    #[test]
    fn test_read_explicit_accepts_a_directory() {
        let temp = tempfile::tempdir().expect("tempdir");
        write(temp.path(), "10-a.md", "alpha");
        write(temp.path(), "20-b.md", "beta");

        let found = read_explicit(temp.path()).expect("ok");
        assert_eq!(found.text, "alpha\n\nbeta");
    }

    /// A ConfigMap key is frequently just `instructions`, with no extension. Requiring one would
    /// turn a naming choice made in someone else's YAML into a startup failure inside a pod.
    #[test]
    fn test_read_explicit_directory_takes_files_without_a_markdown_extension() {
        let temp = tempfile::tempdir().expect("tempdir");
        write(temp.path(), "instructions", "no extension here");

        assert_eq!(
            read_explicit(temp.path()).expect("ok").text,
            "no extension here"
        );
        // The conventional directory stays choosier: it shares the config directory with other
        // content, so a stray file there must not become instructions.
        assert!(read_dir(temp.path(), FileFilter::MarkdownOnly).is_none());
    }

    /// The shape Kubernetes actually produces: `..data` symlinked at a timestamped directory, and
    /// one symlink per key beside it. The directories have to drop out without being listed.
    #[test]
    #[cfg(unix)]
    fn test_read_explicit_handles_a_configmap_symlink_farm() {
        let temp = tempfile::tempdir().expect("tempdir");
        let versioned = temp.path().join("..2026_08_11_00_00_00.123456789");
        std::fs::create_dir(&versioned).expect("mkdir");
        std::fs::write(versioned.join("instructions.md"), "from the configmap").expect("write");
        std::os::unix::fs::symlink(&versioned, temp.path().join("..data")).expect("symlink data");
        std::os::unix::fs::symlink(
            versioned.join("instructions.md"),
            temp.path().join("instructions.md"),
        )
        .expect("symlink key");

        let found = read_explicit(temp.path()).expect("ok");
        assert_eq!(
            found.text, "from the configmap",
            "the key must be read exactly once, with the ..data directory skipped"
        );
    }

    #[test]
    fn test_read_file_trims_and_reports_blank_as_none() {
        let temp = tempfile::tempdir().expect("tempdir");
        write(temp.path(), "padded.md", "\n\n  body  \n\n");
        assert_eq!(
            read_file(&temp.path().join("padded.md")).as_deref(),
            Some("body")
        );
        write(temp.path(), "blank.md", "\n\t ");
        assert!(read_file(&temp.path().join("blank.md")).is_none());
        assert!(read_file(&temp.path().join("absent.md")).is_none());
    }
}
