//! Filesystem tools: `read_file`, `write_file`, and `edit_file`. Image files are returned as
//! multimodal Image content blocks (transcoding to PNG when needed). Writes are gated by the active
//! permission level.
//!
//! All I/O goes through the canonicalized path and, on Unix, uses `O_NOFOLLOW` on the final
//! `open(2)` so a symlink swap between the permission check and the I/O cannot redirect the
//! operation onto an unintended target.

use std::{path::Path, sync::Arc};

use async_trait::async_trait;
use base64::Engine;
use tokio::io::AsyncReadExt;
use tokio_util::sync::CancellationToken;

use super::{
    ReadTracker, Tool, ToolOutput,
    util::{MAX_SEARCH_MATCHES, canonicalize_for_tool, require_str, search_lines, truncate_string},
};
use crate::{
    error::{MekaError, Result},
    image::{ImageHandling, classify_extension, prepare_image_payload},
    permission::Permission,
    provider::{ImageSource, ToolDefinition, ToolResultContent},
};

/// Open a file for reading, refusing to follow a symlink on Unix. Callers pass a canonicalized
/// `PathBuf` so the check closes the canonicalize→open TOCTOU window: if the target was replaced by
/// a symlink after we canonicalized, the open errors out instead of silently redirecting.
async fn open_read_nofollow(path: &Path) -> std::io::Result<tokio::fs::File> {
    #[cfg(unix)]
    {
        tokio::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)
            .await
    }
    #[cfg(not(unix))]
    {
        tokio::fs::File::open(path).await
    }
}

/// Open a file for writing (create-or-truncate) refusing to follow a symlink. A safer default than
/// `tokio::fs::write` for paths that may race against a hostile rename. On Unix `O_NOFOLLOW` errors
/// on a symlinked final component; on Windows the equivalent is opening the reparse point itself
/// and rejecting it before any truncation happens.
async fn open_write_nofollow(path: &Path) -> std::io::Result<tokio::fs::File> {
    #[cfg(unix)]
    {
        tokio::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)
            .await
    }
    #[cfg(windows)]
    {
        // FILE_FLAG_OPEN_REPARSE_POINT opens the link itself rather than following it, so a
        // symlinked path yields a handle we can inspect. Truncation is deferred to `set_len`
        // *after* the symlink check so a rejected target is never destroyed.
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        let file = tokio::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
            .open(path)
            .await?;
        if file.metadata().await?.file_type().is_symlink() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "refusing to write through a symlink",
            ));
        }
        file.set_len(0).await?;
        Ok(file)
    }
    #[cfg(not(any(unix, windows)))]
    {
        tokio::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .await
    }
}

pub(super) async fn read_file_bytes(path: &Path) -> std::io::Result<Vec<u8>> {
    let mut file = open_read_nofollow(path).await?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).await?;
    Ok(buffer)
}

async fn read_file_to_string(path: &Path) -> std::io::Result<String> {
    let mut file = open_read_nofollow(path).await?;
    let mut buffer = String::new();
    file.read_to_string(&mut buffer).await?;
    Ok(buffer)
}

pub(super) async fn write_file_bytes(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    let mut file = open_write_nofollow(path).await?;
    file.write_all(bytes).await?;
    file.flush().await
}

/// Which filesystem one file tool call is operating through.
///
/// Decided **once per tool call**, not per RPC. `edit_file` reads a file and writes it back, and
/// the two halves have to agree: diffing against the editor's buffer and then writing to disk (or
/// the reverse) is how unsaved work gets silently overwritten.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileRoute {
    /// The client serves this path. Reads see unsaved buffers, and writes land in the editor's
    /// document model where its buffer state and undo history stay coherent.
    Delegated,
    /// No client delegate exists at all (the REPL, the HTTP server, a client without the `fs`
    /// capability). There is no editor view to diverge from, so nothing to disclose.
    Local,
    /// A client exists and was asked, and answered that it cannot serve this path. The call runs
    /// against the local filesystem and says so.
    LocalUnservable,
    /// The client served the *read* but offers no `fs.writeTextFile`, so the write had to go to
    /// disk anyway. Distinct from [`FileRoute::LocalUnservable`] because here the client does hold
    /// a buffer for the file, which the local write has just diverged from.
    LocalWriteUnsupported,
}

impl FileRoute {
    fn is_delegated(self) -> bool {
        self == FileRoute::Delegated
    }

    /// Trailer appended to a *write* result that ended up on the local filesystem while a client
    /// was present. The tool call still carries its `Diff` metadata either way, so the change is
    /// visible in the client's agent panel; what it is missing is the editor's own view of the
    /// file, and that is the part worth saying out loud.
    fn write_disclosure(self) -> &'static str {
        match self {
            FileRoute::LocalUnservable => {
                "\n\nNote: your editor declined to serve this path, so meka wrote it directly to \
                 disk. The change is in the diff for this tool call, but not in the editor's \
                 buffer or undo history."
            }
            FileRoute::LocalWriteUnsupported => {
                "\n\nNote: your editor serves this file but does not accept writes, so meka wrote \
                 it directly to disk. If you have the file open with unsaved changes, they now \
                 differ from what is on disk."
            }
            FileRoute::Delegated | FileRoute::Local => "",
        }
    }
}

/// Local `old_text` for `write_file`'s diff metadata. A missing file is a create, not a failure,
/// and any other read error only costs the diff its "before" side, so neither is propagated.
async fn local_old_text(target: &Path) -> Option<String> {
    match read_file_to_string(target).await {
        Ok(text) => Some(text),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            tracing::debug!(
                "write_file: local pre-read of '{}' failed ({}); diff metadata will omit old_text",
                target.display(),
                error,
            );
            None
        }
    }
}

/// Read a file off the local filesystem, reporting failures as `tool_name`'s error with the path
/// the caller supplied rather than the canonicalised one.
async fn read_local_text(canonical: &Path, tool_name: &str, path: &str) -> Result<String> {
    read_file_to_string(canonical)
        .await
        .map_err(|error| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("failed to read '{}': {}", path, error),
        })
}

/// Write `content` to `target` through `route`, returning the route the write *actually* took.
///
/// The returned route is not always the one passed in: a client may offer `fs.readTextFile`
/// without `fs.writeTextFile`, reading for us but expecting us to do the write ourselves, so a
/// delegated read can still end on disk. The disclosure has to be built from what happened rather
/// than from what was planned, or that case reports nothing while being the one where the client
/// holds a buffer the write has just diverged from.
async fn apply_write(
    frontend: &Arc<dyn crate::frontend::Frontend>,
    route: FileRoute,
    target: &Path,
    path: &str,
    content: &str,
    tool_name: &str,
) -> Result<FileRoute> {
    if route.is_delegated() {
        match frontend.delegate_fs_write(target, content).await {
            Some(Ok(())) => return Ok(FileRoute::Delegated),
            // The client serves this path, so a write failure is never a reason to write behind
            // its back: it may be about to show the user its own view of the file.
            Some(Err(error)) => {
                return Err(MekaError::ToolExecution {
                    tool_name: tool_name.to_string(),
                    message: format!("failed to write '{}': {}", path, error),
                });
            }
            None => {}
        }
    }

    write_file_bytes(target, content.as_bytes())
        .await
        .map_err(|error| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("failed to write '{}': {}", path, error),
        })?;

    Ok(match route {
        FileRoute::Delegated => FileRoute::LocalWriteUnsupported,
        already_local => already_local,
    })
}

pub(super) struct ReadFileTool {
    pub read_tracker: ReadTracker,
    pub cwd: crate::agent::SharedCwd,
    /// When the connected ACP client advertises `fs.read_text_file`, plain-text reads are
    /// delegated to the editor's hosted filesystem so it can serve the in-buffer view of the
    /// file rather than the on-disk bytes. `None` (no delegate) and a delegate failure that
    /// disowns the path both read locally; any other delegate failure is surfaced, since the
    /// client may hold unsaved changes the on-disk bytes would misrepresent.
    pub frontend: Arc<dyn crate::frontend::Frontend>,
}

#[async_trait]
impl Tool for ReadFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read_file".to_string(),
            description: format!(
                "Read the contents of a file at the given path. Supported raster \
                 image files (PNG, JPEG, GIF, WebP, BMP, TIFF, ICO, HDR, EXR, \
                 TGA, PNM, QOI, DDS, Farbfeld) are returned as a multimodal \
                 content block; non-native formats are transparently converted \
                 to PNG. Only read image files if the current model supports \
                 vision input. Provide `regex` to return matching lines (max {}) \
                 instead of a line range; `regex` ignores `offset`/`limit` and \
                 cannot be combined with image reads. Multiple independent \
                 read_file calls in one assistant message run in parallel \
                 \u{2014} batch them instead of reading files sequentially.",
                MAX_SEARCH_MATCHES,
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to read"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (0-based). Optional."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read. Optional."
                    },
                    "regex": {
                        "type": "string",
                        "description": format!(
                            "If provided, search the file with this regex pattern \
                             and return matching lines (max {} matches) instead of \
                             a line range. Skipped for image files.",
                            MAX_SEARCH_MATCHES,
                        )
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the output to the scratchpad under this name instead of returning it inline."
                    }
                },
                "required": ["path"]
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
        let path = input["path"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "read_file".to_string(),
                message: "missing 'path' parameter".to_string(),
            })?
            .to_string();

        let resolved = crate::agent::resolve_against_cwd(&self.cwd, &path);
        let canonical = canonicalize_for_tool("read_file", &resolved).await?;

        // Detect image files and return multimodal content, converting non-native formats (TIFF,
        // ICO, etc.) to PNG along the way.
        let extension = canonical
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase());

        let handling = extension
            .as_deref()
            .map(classify_extension)
            .unwrap_or(ImageHandling::Unsupported);

        if !matches!(handling, ImageHandling::Unsupported) {
            let data =
                read_file_bytes(&canonical)
                    .await
                    .map_err(|error| MekaError::ToolExecution {
                        tool_name: "read_file".to_string(),
                        message: format!("failed to read '{}': {}", path, error),
                    })?;

            // The extension only gates whether to *try* the image path; the bytes decide both the
            // media type and whether this is an image at all. A `.png` holding something else
            // falls through to the text read below instead of shipping a block the provider will
            // reject, and a `.png` holding a JPEG is labelled `image/jpeg`.
            let sniffed = crate::image::classify_bytes(&data);
            if !matches!(sniffed, ImageHandling::Unsupported) {
                let (media_type, payload) = match prepare_image_payload(sniffed, &data) {
                    Ok(pair) => pair,
                    Err(message) => {
                        return Ok(ToolOutput::text(
                            format!("Error: image '{}': {}", path, message),
                            true,
                        ));
                    }
                };

                let base64_data = base64::engine::general_purpose::STANDARD.encode(&payload);

                self.read_tracker.write().await.insert(canonical);

                return Ok(ToolOutput {
                    content: vec![
                        ToolResultContent::Text {
                            text: format!("[Image: {}]", path),
                        },
                        ToolResultContent::Image {
                            source: ImageSource {
                                source_type: "base64".to_string(),
                                media_type: media_type.to_string(),
                                data: base64_data,
                            },
                        },
                    ],
                    is_error: false,
                    scratchpad_hint: None,
                    frontend_metadata: None,
                });
            }
        }

        const DEFAULT_LINE_LIMIT: usize = 2000;

        let offset = input["offset"]
            .as_u64()
            .map(|value| usize::try_from(value).unwrap_or(usize::MAX));
        let limit = input["limit"]
            .as_u64()
            .map(|value| usize::try_from(value).unwrap_or(usize::MAX));
        let regex = input.get("regex").and_then(|v| v.as_str());

        // Plain text reads delegate to the editor when it offers `fs.read_text_file` (in-buffer
        // view wins over on-disk). Regex / image reads have no `fs/*` analogue; always local.
        if regex.is_none() {
            let delegate_line =
                offset.map(|o| u32::try_from(o.saturating_add(1)).unwrap_or(u32::MAX));
            // Mirror the local fallback's `DEFAULT_LINE_LIMIT` so the delegate path can't
            // accidentally pull an unbounded file into the agent's context when the caller passed
            // no limit. Without this, an `fs.read_text_file`-capable client (e.g. Zed) returns the
            // whole file while the local path would cap at 2000 lines with a truncation marker:
            // divergent behavior + context-window risk.
            let delegate_limit = Some(
                limit
                    .map(|l| u32::try_from(l).unwrap_or(u32::MAX))
                    .unwrap_or(DEFAULT_LINE_LIMIT as u32),
            );
            match self
                .frontend
                .delegate_fs_read(&canonical, delegate_line, delegate_limit)
                .await
            {
                Some(Ok(content)) => {
                    self.read_tracker.write().await.insert(canonical);
                    return Ok(ToolOutput::text(content, false));
                }
                // The client will not serve this path, so it holds no buffer for it either: the
                // local bytes are not a degraded substitute for the delegate's view, they are the
                // same view. Fall through and read them, rather than turning a readable file into
                // a tool error -- which is what made skills, prompts, and configuration
                // unreadable under ACP.
                Some(Err(error)) if error.is_unservable_path() => {
                    tracing::debug!(
                        "read_file: client cannot serve '{}' ({}); reading it locally",
                        canonical.display(),
                        error,
                    );
                }
                // Any other failure leaves open that the client owns this file and has unsaved
                // changes in it. Reading disk bytes would silently hand the model a stale view of
                // a file the user is editing, so surface the failure instead.
                Some(Err(error)) => {
                    return Err(MekaError::ToolExecution {
                        tool_name: "read_file".to_string(),
                        message: format!("failed to read '{}': {}", path, error),
                    });
                }
                None => {}
            }
        }

        let content =
            read_file_to_string(&canonical)
                .await
                .map_err(|error| MekaError::ToolExecution {
                    tool_name: "read_file".to_string(),
                    message: format!("failed to read '{}': {}", path, error),
                })?;

        if let Some(pattern) = regex {
            self.read_tracker.write().await.insert(canonical);
            return search_lines(&content, pattern, "read_file");
        }

        let total_lines = content.lines().count();
        let effective_offset = offset.unwrap_or(0);
        let effective_limit = limit.unwrap_or(DEFAULT_LINE_LIMIT);

        let result: String = content
            .lines()
            .skip(effective_offset)
            .take(effective_limit)
            .collect::<Vec<_>>()
            .join("\n");

        let result = if offset.is_none() && limit.is_none() && total_lines > DEFAULT_LINE_LIMIT {
            format!(
                "{}\n\n... (showing first {} of {} lines, use offset/limit to read more)",
                result, DEFAULT_LINE_LIMIT, total_lines,
            )
        } else {
            result
        };

        self.read_tracker.write().await.insert(canonical);

        Ok(ToolOutput::text(result, false))
    }
}

pub(super) struct EditFileTool {
    pub read_tracker: ReadTracker,
    pub cwd: crate::agent::SharedCwd,
    /// Read + write both go through the frontend so the editor can apply the edit in-buffer (Zed's
    /// apply-diff UI). `None` from the frontend means "fall back to local I/O".
    pub frontend: Arc<dyn crate::frontend::Frontend>,
}

#[async_trait]
impl Tool for EditFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "edit_file".to_string(),
            description: "Modify a file. Two modes: (1) Replace: provide \
                          'new_string' to swap 'old_string' for it (an empty \
                          'new_string' deletes 'old_string'). (2) Insert: provide \
                          'insert_before' or 'insert_after' to place content adjacent to \
                          'old_string' while preserving the anchor itself; useful when you \
                          only need to add lines without rewriting surrounding context. \
                          Exactly one of 'new_string', 'insert_before', 'insert_after' \
                          must be set. 'replace_all' applies the operation to every \
                          occurrence; if it is omitted and 'old_string' matches more than \
                          once, the edit is rejected so you can add context to disambiguate \
                          or set 'replace_all' deliberately. The file must have been \
                          read with read_file first unless 'force' is set to true. On \
                          success the response includes a small ±3-line snippet around \
                          the first edited site so you can confirm the change landed."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to edit"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to find (acts as anchor in insert modes)"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replace mode: the replacement string (an empty string deletes old_string). Mutually exclusive with insert_before/insert_after."
                    },
                    "insert_before": {
                        "type": "string",
                        "description": "Insert mode: text inserted immediately before 'old_string' (anchor preserved). Mutually exclusive with new_string/insert_after."
                    },
                    "insert_after": {
                        "type": "string",
                        "description": "Insert mode: text inserted immediately after 'old_string' (anchor preserved). Mutually exclusive with new_string/insert_before."
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "If true, apply to every occurrence. If false (default) and old_string matches more than once, the edit is rejected as ambiguous."
                    },
                    "force": {
                        "type": "boolean",
                        "description": "If true, bypass the requirement to read the file first. Defaults to false."
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the output to the scratchpad under this name instead of returning it inline."
                    }
                },
                "required": ["path", "old_string"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Write
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let path = require_str(&input, "path", "edit_file")?;
        let old_string = require_str(&input, "old_string", "edit_file")?;
        let replace_all = input["replace_all"].as_bool().unwrap_or(false);
        let force = input["force"].as_bool().unwrap_or(false);

        let new_string_opt = input.get("new_string").and_then(|v| v.as_str());
        let insert_before_opt = input.get("insert_before").and_then(|v| v.as_str());
        let insert_after_opt = input.get("insert_after").and_then(|v| v.as_str());

        let mode_count = [
            new_string_opt.is_some(),
            insert_before_opt.is_some(),
            insert_after_opt.is_some(),
        ]
        .into_iter()
        .filter(|present| *present)
        .count();

        if mode_count == 0 {
            return Ok(ToolOutput::text(
                "Error: provide one of 'new_string', 'insert_before', or 'insert_after'"
                    .to_string(),
                true,
            ));
        }
        if mode_count > 1 {
            return Ok(ToolOutput::text(
                "Error: 'new_string', 'insert_before', and 'insert_after' are mutually exclusive"
                    .to_string(),
                true,
            ));
        }

        let effective_new_string = if let Some(new) = new_string_opt {
            new.to_string()
        } else if let Some(prefix) = insert_before_opt {
            format!("{}{}", prefix, old_string)
        } else {
            // Safe by mode_count == 1 above.
            format!("{}{}", old_string, insert_after_opt.unwrap_or(""))
        };

        // Canonicalize once. All subsequent I/O goes through this path so a symlink swap between
        // the tracker check and the actual read/write can't redirect us onto a different file.
        let resolved = crate::agent::resolve_against_cwd(&self.cwd, &path);
        let canonical = canonicalize_for_tool("edit_file", &resolved).await?;

        if !force && !self.read_tracker.read().await.contains(&canonical) {
            return Ok(ToolOutput::text(
                format!(
                    "Error: file '{}' must be read before editing. \
                     Use read_file first, or set force=true to bypass.",
                    path
                ),
                true,
            ));
        }

        // The pre-read picks the route for the whole edit. Prefer the editor's in-buffer view: it
        // is the document the editor will apply the edit against.
        let (content, route) = match self.frontend.delegate_fs_read(&canonical, None, None).await {
            Some(Ok(text)) => (text, FileRoute::Delegated),
            // The client cannot serve this path, so it holds no buffer for it and will not
            // accept the write either. Run the whole edit locally rather than refusing: an
            // editor's project boundary should not decide which files the agent can edit.
            Some(Err(error)) if error.is_unservable_path() => {
                tracing::debug!(
                    "edit_file: client cannot serve '{}' ({}); editing it locally",
                    canonical.display(),
                    error,
                );
                (
                    read_local_text(&canonical, "edit_file", &path).await?,
                    FileRoute::LocalUnservable,
                )
            }
            // Any other failure has to short-circuit. The client may own this file and hold
            // unsaved changes; diffing against on-disk bytes and then writing the result
            // through the delegate would overwrite the user's unsaved work with an edit
            // computed from stale input.
            Some(Err(error)) => {
                return Err(MekaError::ToolExecution {
                    tool_name: "edit_file".to_string(),
                    message: format!("failed to read '{}': {}", path, error),
                });
            }
            None => (
                read_local_text(&canonical, "edit_file", &path).await?,
                FileRoute::Local,
            ),
        };

        if !content.contains(&old_string) {
            return Ok(ToolOutput::text(
                format!(
                    "Error: '{}' not found in '{}'",
                    truncate_string(&old_string, 100),
                    path
                ),
                true,
            ));
        }

        // Reject an ambiguous single-occurrence edit: silently editing the first of several
        // matches is almost never intended. Surface the count so the caller can add surrounding
        // context to disambiguate, or set `replace_all` to change every occurrence on purpose.
        let match_count = content.matches(&old_string).count();
        if !replace_all && match_count > 1 {
            return Ok(ToolOutput::text(
                format!(
                    "Error: '{}' matches {} times in '{}'; add surrounding context to make \
                     old_string unique, or set replace_all=true to change every occurrence",
                    truncate_string(&old_string, 100),
                    match_count,
                    path
                ),
                true,
            ));
        }

        // Record the byte offset of the first match in the *original* content; since `replacen` /
        // `replace` only mutate at-or-after this point, the byte offset is stable in the new
        // content and locates the first edit site for the response snippet.
        let first_match_byte = content.find(&old_string).unwrap_or(0);

        let (new_content, count) = if replace_all {
            (
                content.replace(&old_string, &effective_new_string),
                match_count,
            )
        } else {
            (content.replacen(&old_string, &effective_new_string, 1), 1)
        };

        // Write back through whichever filesystem produced `content`. Re-deciding here instead of
        // reusing the route is what would let a diff taken from the editor's buffer land on disk.
        let route = apply_write(
            &self.frontend,
            route,
            &canonical,
            &path,
            &new_content,
            "edit_file",
        )
        .await?;

        let snippet = build_context_snippet(&new_content, first_match_byte, 3);
        let trailer = if count > 1 {
            format!(" ... (showing context for first of {} occurrences)", count)
        } else {
            String::new()
        };

        Ok(ToolOutput::text(
            format!(
                "Successfully edited '{}': {} occurrence(s){}\n\n{}{}",
                path,
                count,
                trailer,
                snippet,
                route.write_disclosure(),
            ),
            false,
        )
        .with_metadata(crate::frontend::ToolOutputMetadata::Diff {
            path: canonical.clone(),
            old_text: Some(content),
            new_text: new_content,
        }))
    }
}

/// Render a ±`lines_around` snippet around the line containing `change_byte_offset` in `content`.
/// Each line is prefixed with a right-aligned 1-based line number and a `|` separator, and
/// truncated to 200 chars to keep the response compact.
fn build_context_snippet(content: &str, change_byte_offset: usize, lines_around: usize) -> String {
    let safe_offset = change_byte_offset.min(content.len());
    let line_index = content[..safe_offset]
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count();

    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return String::new();
    }
    let start = line_index.saturating_sub(lines_around);
    let end = (line_index + lines_around + 1).min(lines.len());

    let mut output = String::new();
    for (idx, line) in lines.iter().enumerate().take(end).skip(start) {
        let display = truncate_string(line, 200);
        output.push_str(&format!("{:>5} | {}\n", idx + 1, display));
    }
    output
}

pub(super) struct WriteFileTool {
    /// Shared with `ReadFileTool` / `EditFileTool`. After a successful write we insert the
    /// canonical target so a follow-up `edit_file` against the same path doesn't require a
    /// redundant `read_file` or `force: true`; the agent obviously knows the content it just
    /// wrote.
    pub read_tracker: ReadTracker,
    pub cwd: crate::agent::SharedCwd,
    /// Write step is delegated to the editor's filesystem so the apply-diff UI sees the new
    /// content alongside the `tool_call_update`'s diff. `None` from the frontend means "fall
    /// back to local write".
    pub frontend: Arc<dyn crate::frontend::Frontend>,
}

#[async_trait]
impl Tool for WriteFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "write_file".to_string(),
            description: "Create or overwrite a file with the given content.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the output to the scratchpad under this name instead of returning it inline."
                    }
                },
                "required": ["path", "content"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Write
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let path = require_str(&input, "path", "write_file")?;
        let content = require_str(&input, "content", "write_file")?;

        // The target file may not exist yet, so we canonicalize the *parent* directory and re-join
        // the filename. This pins the final open to a directory whose symlinks have been resolved,
        // closing the window where a symlink-pointing-at-a-parent swap could redirect the write.
        // The per-file `O_NOFOLLOW` in `write_file_bytes` then prevents a last-component symlink
        // swap.
        let file_path = crate::agent::resolve_against_cwd(&self.cwd, &path);
        let file_name = file_path
            .file_name()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "write_file".to_string(),
                message: format!("invalid path (no file name): '{}'", path),
            })?;
        let parent = file_path.parent().ok_or_else(|| MekaError::ToolExecution {
            tool_name: "write_file".to_string(),
            message: format!("invalid path (no parent): '{}'", path),
        })?;

        // Treat an empty parent (relative filename like "out.txt") as the current directory; this
        // matches the previous `tokio::fs::write` behavior for bare filenames.
        let parent_for_create: &Path = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        tokio::fs::create_dir_all(parent_for_create)
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "write_file".to_string(),
                message: format!("failed to create directories for '{}': {}", path, error),
            })?;

        let canonical_parent = canonicalize_for_tool("write_file", parent_for_create).await?;
        let target = canonical_parent.join(file_name);

        // Snapshot the existing content (if any) so frontends can render a proper diff. `None`
        // means the file did not exist (this is a create); we use the `not_found` ErrorKind to
        // distinguish from a permissions error so the latter surfaces normally.
        //
        // When the client offers `fs.read_text_file`, ask the editor for its view first; buffers
        // with unsaved changes give a more accurate `old_text` than the on-disk bytes. A delegate
        // error is non-fatal here (diff metadata is informational), so we fall back to the local
        // read on `Some(Err(_))` too.
        //
        // Some clients return `Ok("")` for files that don't exist
        // (rather than an error). To avoid reporting `old_text:
        // Some("")` for what is actually a fresh-file create, we
        // probe local metadata: if the file is absent on disk AND
        // the delegate returned an empty string, treat it as "new
        // file" (`None`). The probe is one stat; cost is negligible
        // and the heuristic is conservative: a truly-empty existing
        // file still loses `old_text`, but the diff content is
        // identical either way.
        //
        // The probe also picks the route for the write, the same way `edit_file`'s pre-read does,
        // and for a reason worth recording: a client may report an unservable path on a *read* but
        // not on a write. Zed does exactly this -- its `read_text_file` maps a path outside the
        // open project to `ResourceNotFound`, while its `write_text_file` returns a generic error
        // for the same path. Routing the write on its own error code would therefore never
        // recognise the case it exists for. The read is the reliable signal, and it is unambiguous
        // there: a file that does not exist *yet* inside the project still maps to a project path,
        // so it comes back as `Ok("")` rather than as not-found.
        let (old_text, route) = match self.frontend.delegate_fs_read(&target, None, None).await {
            Some(Ok(text)) => {
                let old = if text.is_empty() && !target.exists() {
                    None
                } else {
                    Some(text)
                };
                (old, FileRoute::Delegated)
            }
            Some(Err(error)) if error.is_unservable_path() => {
                tracing::debug!(
                    "write_file: client cannot serve '{}' ({}); writing it locally",
                    target.display(),
                    error,
                );
                (local_old_text(&target).await, FileRoute::LocalUnservable)
            }
            // The client did not disown the path, it just failed this probe. The write still goes
            // to it; only the diff's `old_text` is degraded, and that is informational.
            Some(Err(error)) => {
                tracing::debug!(
                    "write_file: pre-read of '{}' failed ({}); diff metadata will omit old_text",
                    target.display(),
                    error,
                );
                (local_old_text(&target).await, FileRoute::Delegated)
            }
            None => (local_old_text(&target).await, FileRoute::Local),
        };

        // Write through whichever filesystem the probe selected. Same shape as `edit_file`.
        let route = apply_write(
            &self.frontend,
            route,
            &target,
            &path,
            &content,
            "write_file",
        )
        .await?;

        // Record the canonical path so subsequent `edit_file` calls accept it without `force:
        // true`. We just produced the content, so the "must read first" safety check has nothing to
        // gain.
        self.read_tracker.write().await.insert(target.clone());

        Ok(ToolOutput::text(
            format!(
                "Successfully wrote {} bytes to '{}'{}",
                content.len(),
                path,
                route.write_disclosure(),
            ),
            false,
        )
        .with_metadata(crate::frontend::ToolOutputMetadata::Diff {
            path: target,
            old_text,
            new_text: content.to_string(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, sync::Arc};

    use tokio::sync::RwLock;

    use super::*;
    use crate::tools::tests::text_content;

    fn test_tracker() -> ReadTracker {
        Arc::new(RwLock::new(HashSet::new()))
    }

    /// A frontend whose hosted filesystem answers with a scripted outcome, so each branch of the
    /// routing rule can be driven directly.
    struct ScriptedDelegateFrontend {
        read: Option<std::result::Result<String, crate::frontend::FrontendError>>,
        write: Option<std::result::Result<(), crate::frontend::FrontendError>>,
        delegated_writes: std::sync::Mutex<Vec<std::path::PathBuf>>,
    }

    impl ScriptedDelegateFrontend {
        /// A client that will not serve the path at all, the way an editor answers for files
        /// outside the project it has open.
        fn unservable() -> Self {
            Self {
                read: Some(Err(crate::frontend::FrontendError::unservable_path(
                    "fs/read_text_file failed: Resource not found",
                ))),
                write: Some(Err(crate::frontend::FrontendError::unservable_path(
                    "fs/write_text_file failed: Resource not found",
                ))),
                delegated_writes: std::sync::Mutex::new(Vec::new()),
            }
        }

        /// A client that owns the path but failed this time round.
        fn transient() -> Self {
            Self {
                read: Some(Err(crate::frontend::FrontendError::new(
                    "fs/read_text_file failed: Internal error",
                ))),
                write: Some(Err(crate::frontend::FrontendError::new(
                    "fs/write_text_file failed: Internal error",
                ))),
                delegated_writes: std::sync::Mutex::new(Vec::new()),
            }
        }

        /// A client serving its in-buffer view, which differs from the on-disk bytes.
        fn serving(buffer: &str) -> Self {
            Self {
                read: Some(Ok(buffer.to_string())),
                write: Some(Ok(())),
                delegated_writes: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl crate::frontend::Frontend for ScriptedDelegateFrontend {
        async fn emit(&self, _event: crate::frontend::FrontendEvent) {}

        async fn request_permission(
            &self,
            _request: crate::frontend::PermissionRequest,
        ) -> crate::frontend::PermissionOutcome {
            crate::frontend::PermissionOutcome::Allow
        }

        async fn delegate_fs_read(
            &self,
            _path: &Path,
            _line: Option<u32>,
            _limit: Option<u32>,
        ) -> Option<std::result::Result<String, crate::frontend::FrontendError>> {
            self.read.clone()
        }

        async fn delegate_fs_write(
            &self,
            path: &Path,
            _content: &str,
        ) -> Option<std::result::Result<(), crate::frontend::FrontendError>> {
            self.delegated_writes
                .lock()
                .expect("lock")
                .push(path.to_path_buf());
            self.write.clone()
        }
    }

    #[tokio::test]
    async fn test_read_file() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, "line1\nline2\nline3\n").expect("failed to write");

        let tool = ReadFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({"path": file_path.to_str().expect("path")}),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("line1"));
        assert!(text_content(&result).contains("line3"));
    }

    #[tokio::test]
    async fn test_read_file_falls_back_when_client_cannot_serve_path() {
        // Regression: a delegate failure used to abort the read. Editors serve `fs/read_text_file`
        // only for the project they have open, so every skill, prompt, and config file read under
        // ACP became a hard tool error -- with the file sitting right there, readable.
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("outside-the-project.md");
        std::fs::write(&file_path, "on-disk contents\n").expect("failed to write");

        let tool = ReadFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(ScriptedDelegateFrontend::unservable()),
        };
        let result = tool
            .execute(
                serde_json::json!({"path": file_path.to_str().expect("path")}),
                CancellationToken::new(),
            )
            .await
            .expect("a path the client cannot serve must still be readable locally");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("on-disk contents"));
    }

    #[tokio::test]
    async fn test_read_file_surfaces_transient_delegate_failure() {
        // A client that owns the file and merely failed this time may be holding unsaved changes.
        // Reading disk bytes would hand the model a stale view of a file the user is editing.
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("owned-by-the-editor.md");
        std::fs::write(&file_path, "stale on-disk contents\n").expect("failed to write");

        let tool = ReadFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(ScriptedDelegateFrontend::transient()),
        };
        let error = tool
            .execute(
                serde_json::json!({"path": file_path.to_str().expect("path")}),
                CancellationToken::new(),
            )
            .await
            .expect_err("a transient delegate failure must not silently read stale bytes");
        assert!(error.to_string().contains("Internal error"), "{}", error);
    }

    #[tokio::test]
    async fn test_edit_file_runs_locally_when_client_cannot_serve_path() {
        // ACP must not be less capable than the terminal: an editor's project boundary should not
        // decide which files the agent is allowed to edit.
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("skill.md");
        std::fs::write(&file_path, "before\n").expect("write");

        let frontend = Arc::new(ScriptedDelegateFrontend::unservable());
        let tracker = test_tracker();
        tracker
            .write()
            .await
            .insert(std::fs::canonicalize(&file_path).expect("canonicalize"));
        let tool = EditFileTool {
            read_tracker: tracker,
            cwd: crate::agent::test_cwd(),
            frontend: frontend.clone(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "before",
                    "new_string": "after",
                }),
                CancellationToken::new(),
            )
            .await
            .expect("an unservable path must still be editable locally");

        assert!(!result.is_error);
        assert_eq!(
            std::fs::read_to_string(&file_path).expect("read back"),
            "after\n"
        );
        // The user has to be told the editor's buffer and undo history do not know about this.
        assert!(
            text_content(&result).contains("declined to serve this path"),
            "local write must be disclosed, got: {}",
            text_content(&result),
        );
    }

    #[tokio::test]
    async fn test_edit_file_surfaces_transient_delegate_failure() {
        // The data-loss shape: if the pre-read fell back to disk while the editor held unsaved
        // changes, the edit would be computed from stale bytes and then written through the
        // delegate, overwriting the user's unsaved work.
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("owned.md");
        std::fs::write(&file_path, "before\n").expect("write");

        let tracker = test_tracker();
        tracker
            .write()
            .await
            .insert(std::fs::canonicalize(&file_path).expect("canonicalize"));
        let tool = EditFileTool {
            read_tracker: tracker,
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(ScriptedDelegateFrontend::transient()),
        };
        let error = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "before",
                    "new_string": "after",
                }),
                CancellationToken::new(),
            )
            .await
            .expect_err("a transient pre-read failure must not produce a local edit");
        assert!(error.to_string().contains("Internal error"), "{}", error);
        assert_eq!(
            std::fs::read_to_string(&file_path).expect("read back"),
            "before\n",
            "the file must be untouched"
        );
    }

    #[tokio::test]
    async fn test_edit_file_writes_back_through_the_route_it_read_from() {
        // Route consistency: the edit was computed from the client's buffer, so it has to be
        // applied there. Writing it to disk instead would drop it the moment the buffer is saved.
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("open-in-editor.md");
        std::fs::write(&file_path, "stale disk bytes\n").expect("write");
        let canonical = std::fs::canonicalize(&file_path).expect("canonicalize");

        let frontend = Arc::new(ScriptedDelegateFrontend::serving("unsaved buffer\n"));
        let tracker = test_tracker();
        tracker.write().await.insert(canonical.clone());
        let tool = EditFileTool {
            read_tracker: tracker,
            cwd: crate::agent::test_cwd(),
            frontend: frontend.clone(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "unsaved buffer",
                    "new_string": "edited buffer",
                }),
                CancellationToken::new(),
            )
            .await
            .expect("edit against the client's buffer");

        assert!(!result.is_error);
        assert_eq!(
            frontend.delegated_writes.lock().expect("lock").as_slice(),
            &[canonical],
            "the edit must be applied where it was read from"
        );
        assert_eq!(
            std::fs::read_to_string(&file_path).expect("read back"),
            "stale disk bytes\n",
            "a delegated edit must not also touch the local file"
        );
        assert!(
            !text_content(&result).contains("declined to serve"),
            "a delegated write has nothing to disclose"
        );
    }

    #[tokio::test]
    async fn test_edit_file_discloses_local_write_when_client_cannot_write() {
        // A client may advertise `fs.readTextFile` without `fs.writeTextFile`: it reads for us but
        // expects us to do the write. The edit is then computed from its buffer and lands on disk,
        // so the file it is showing the user now differs from what is stored -- which the result
        // has to say, even though the read was delegated.
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("open.md");
        std::fs::write(&file_path, "stale disk bytes\n").expect("write");
        let canonical = std::fs::canonicalize(&file_path).expect("canonicalize");

        let frontend = Arc::new(ScriptedDelegateFrontend {
            read: Some(Ok("unsaved buffer\n".to_string())),
            // No `fs.writeTextFile` capability.
            write: None,
            delegated_writes: std::sync::Mutex::new(Vec::new()),
        });
        let tracker = test_tracker();
        tracker.write().await.insert(canonical);
        let tool = EditFileTool {
            read_tracker: tracker,
            cwd: crate::agent::test_cwd(),
            frontend,
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "unsaved buffer",
                    "new_string": "edited buffer",
                }),
                CancellationToken::new(),
            )
            .await
            .expect("edit must still apply");

        assert!(!result.is_error);
        assert_eq!(
            std::fs::read_to_string(&file_path).expect("read back"),
            "edited buffer\n"
        );
        assert!(
            text_content(&result).contains("does not accept writes"),
            "divergence from the client's buffer must be disclosed, got: {}",
            text_content(&result),
        );
    }

    #[tokio::test]
    async fn test_write_file_routes_on_the_read_probe_not_the_write_error() {
        // Zed reports an out-of-project path as `ResourceNotFound` on `read_text_file` but as a
        // generic error on `write_text_file`. Routing on the write's own error code would never
        // recognise the case, so the probe decides and the delegated write is not even attempted.
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("skills").join("new-skill.md");
        std::fs::create_dir_all(file_path.parent().expect("parent")).expect("mkdir");

        let frontend = Arc::new(ScriptedDelegateFrontend {
            read: Some(Err(crate::frontend::FrontendError::unservable_path(
                "fs/read_text_file failed: Resource not found",
            ))),
            // What Zed would answer if we asked: a generic error, not `ResourceNotFound`.
            write: Some(Err(crate::frontend::FrontendError::new(
                "fs/write_text_file failed: invalid path",
            ))),
            delegated_writes: std::sync::Mutex::new(Vec::new()),
        });
        let tool = WriteFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: frontend.clone(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "content": "# skill\n",
                }),
                CancellationToken::new(),
            )
            .await
            .expect("an out-of-project create must succeed under ACP");

        assert!(!result.is_error);
        assert_eq!(
            std::fs::read_to_string(&file_path).expect("read back"),
            "# skill\n"
        );
        assert!(
            frontend.delegated_writes.lock().expect("lock").is_empty(),
            "the write must not be attempted once the probe disowned the path"
        );
        assert!(
            text_content(&result).contains("declined to serve this path"),
            "local write must be disclosed, got: {}",
            text_content(&result),
        );
    }

    #[tokio::test]
    async fn test_write_file_falls_back_and_discloses_when_unservable() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("config.toml");

        let tool = WriteFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(ScriptedDelegateFrontend::unservable()),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "content": "key = 1\n",
                }),
                CancellationToken::new(),
            )
            .await
            .expect("an unservable path must still be writable locally");

        assert!(!result.is_error);
        assert_eq!(
            std::fs::read_to_string(&file_path).expect("read back"),
            "key = 1\n"
        );
        assert!(
            text_content(&result).contains("declined to serve this path"),
            "local write must be disclosed, got: {}",
            text_content(&result),
        );
    }

    #[tokio::test]
    async fn test_write_file_surfaces_transient_delegate_failure() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("owned.txt");

        let tool = WriteFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(ScriptedDelegateFrontend::transient()),
        };
        let error = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "content": "x\n",
                }),
                CancellationToken::new(),
            )
            .await
            .expect_err("a transient write failure must not become a local write");
        assert!(error.to_string().contains("Internal error"), "{}", error);
        assert!(!file_path.exists(), "nothing should have been written");
    }

    #[tokio::test]
    async fn test_read_file_with_offset_and_limit() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, "line0\nline1\nline2\nline3\nline4\n").expect("failed to write");

        let tool = ReadFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "offset": 1,
                    "limit": 2
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("line1"));
        assert!(text_content(&result).contains("line2"));
        assert!(!text_content(&result).contains("line0"));
        assert!(!text_content(&result).contains("line3"));
    }

    #[tokio::test]
    async fn test_write_and_read_file() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("output.txt");

        let write_tool = WriteFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let write_result = write_tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "content": "hello world"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");
        assert!(!write_result.is_error);

        let content = std::fs::read_to_string(&file_path).expect("failed to read");
        assert_eq!(content, "hello world");
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn test_write_file_rejects_symlink_on_windows() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let real = temp_dir.path().join("real.txt");
        std::fs::write(&real, "original").expect("seed real file");
        let link = temp_dir.path().join("link.txt");
        // Symlink creation needs Developer Mode / SeCreateSymbolicLink; skip rather than fail if
        // the runner can't create one.
        if std::os::windows::fs::symlink_file(&real, &link).is_err() {
            return;
        }

        let write_tool = WriteFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = write_tool
            .execute(
                serde_json::json!({
                    "path": link.to_str().expect("path"),
                    "content": "overwritten"
                }),
                CancellationToken::new(),
            )
            .await;

        assert_eq!(
            std::fs::read_to_string(&real).expect("read real"),
            "original",
            "symlink target was overwritten"
        );
        assert!(
            result.map(|output| output.is_error).unwrap_or(true),
            "write through a symlink should be rejected"
        );
    }

    #[tokio::test]
    async fn test_edit_file_after_write_no_force_needed() {
        // Regression: `write_file` should mark the target as read so a follow-up `edit_file`
        // doesn't require `force: true`.
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("write_then_edit.txt");
        let tracker = test_tracker();

        let write_tool = WriteFileTool {
            read_tracker: tracker.clone(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        write_tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "content": "hello world"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("write should succeed");

        let edit_tool = EditFileTool {
            read_tracker: tracker.clone(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let edit_result = edit_tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "world",
                    "new_string": "rust"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("edit should succeed without force");
        assert!(
            !edit_result.is_error,
            "edit after write should succeed without force, got: {}",
            text_content(&edit_result)
        );

        let content = std::fs::read_to_string(&file_path).expect("read");
        assert_eq!(content, "hello rust");
    }

    #[tokio::test]
    async fn test_edit_file() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "hello world").expect("failed to write");

        let tracker = test_tracker();
        // Read the file first to satisfy read-before-edit
        let read_tool = ReadFileTool {
            read_tracker: tracker.clone(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        read_tool
            .execute(
                serde_json::json!({"path": file_path.to_str().expect("path")}),
                CancellationToken::new(),
            )
            .await
            .expect("read should succeed");

        let tool = EditFileTool {
            read_tracker: tracker,
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "world",
                    "new_string": "rust"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        let content = std::fs::read_to_string(&file_path).expect("failed to read");
        assert_eq!(content, "hello rust");
    }

    #[tokio::test]
    async fn test_edit_file_replace_all() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "foo bar foo baz foo").expect("failed to write");

        let tool = EditFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "foo",
                    "new_string": "qux",
                    "replace_all": true,
                    "force": true
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("3 occurrence(s)"));
        let content = std::fs::read_to_string(&file_path).expect("failed to read");
        assert_eq!(content, "qux bar qux baz qux");
    }

    #[tokio::test]
    async fn test_edit_file_ambiguous_match_without_replace_all_errors() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "foo bar foo baz foo").expect("failed to write");

        let tool = EditFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "foo",
                    "new_string": "qux",
                    "force": true
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(result.is_error);
        assert!(
            text_content(&result).contains("matches 3 times"),
            "got: {}",
            text_content(&result)
        );
        // The file must be untouched when an ambiguous edit is rejected.
        let content = std::fs::read_to_string(&file_path).expect("failed to read");
        assert_eq!(content, "foo bar foo baz foo");
    }

    #[tokio::test]
    async fn test_edit_file_not_found_string() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "hello world").expect("failed to write");

        let tool = EditFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "nonexistent",
                    "new_string": "replacement",
                    "force": true
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(result.is_error);
        assert!(text_content(&result).contains("not found"));
    }

    #[tokio::test]
    async fn test_edit_without_read_fails() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "hello world").expect("failed to write");

        let tool = EditFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "world",
                    "new_string": "rust"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(result.is_error);
        assert!(text_content(&result).contains("must be read before editing"));
    }

    #[tokio::test]
    async fn test_edit_with_force_bypasses_read_check() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "hello world").expect("failed to write");

        let tool = EditFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "world",
                    "new_string": "rust",
                    "force": true
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        let content = std::fs::read_to_string(&file_path).expect("failed to read");
        assert_eq!(content, "hello rust");
    }

    #[tokio::test]
    async fn test_read_then_edit_succeeds() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "hello world").expect("failed to write");

        let tracker = test_tracker();

        let read_tool = ReadFileTool {
            read_tracker: tracker.clone(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        read_tool
            .execute(
                serde_json::json!({"path": file_path.to_str().expect("path")}),
                CancellationToken::new(),
            )
            .await
            .expect("read should succeed");

        let edit_tool = EditFileTool {
            read_tracker: tracker,
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = edit_tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "world",
                    "new_string": "rust"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
    }

    /// Regression test for the canonicalize/open TOCTOU fix: edit_file must honor the canonical
    /// path, not re-interpret the raw argument after the tracker check. Simulated here by
    /// read-tracking the resolved file, then swapping the symlink's target between read and edit.
    /// The edit must land on the original canonical file, never the new target.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_edit_file_symlink_swap_lands_on_canonical() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let real_a = temp_dir.path().join("a.txt");
        let real_b = temp_dir.path().join("b.txt");
        let link = temp_dir.path().join("link");
        std::fs::write(&real_a, "value-a").expect("write a");
        std::fs::write(&real_b, "value-b").expect("write b");
        std::os::unix::fs::symlink(&real_a, &link).expect("symlink");

        let tracker = test_tracker();

        let read_tool = ReadFileTool {
            read_tracker: tracker.clone(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        read_tool
            .execute(
                serde_json::json!({"path": link.to_str().expect("path")}),
                CancellationToken::new(),
            )
            .await
            .expect("read");

        // Attacker swaps symlink to point at real_b between read and edit.
        std::fs::remove_file(&link).expect("remove link");
        std::os::unix::fs::symlink(&real_b, &link).expect("swap symlink");

        let edit_tool = EditFileTool {
            read_tracker: tracker,
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = edit_tool
            .execute(
                serde_json::json!({
                    "path": link.to_str().expect("path"),
                    "old_string": "value-a",
                    "new_string": "overwritten",
                }),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        // Either the tracker rejects the new canonical target (expected, since `real_b` was never
        // read) or the O_NOFOLLOW open hits the swapped symlink and errors. Both outcomes are
        // acceptable; the critical invariant is that neither file is corrupted.
        assert!(
            result.is_error,
            "edit should be rejected after symlink swap, got: {}",
            text_content(&result)
        );
        assert_eq!(
            std::fs::read_to_string(&real_a).expect("read a"),
            "value-a",
            "original target must be untouched"
        );
        assert_eq!(
            std::fs::read_to_string(&real_b).expect("read b"),
            "value-b",
            "alternate target must be untouched"
        );
    }

    #[tokio::test]
    async fn test_read_file_regex_basic() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("rg.txt");
        std::fs::write(&file_path, "alpha\nbravo 42\ncharlie\ndelta 99\necho\n").expect("write");

        let tool = ReadFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "regex": r"\d+"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("regex search should succeed");

        assert!(!result.is_error);
        let text = text_content(&result);
        assert!(text.contains("2:bravo 42"));
        assert!(text.contains("4:delta 99"));
        assert!(!text.contains("alpha"));
        assert!(!text.contains("charlie"));
    }

    #[tokio::test]
    async fn test_read_file_regex_no_match() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("rg.txt");
        std::fs::write(&file_path, "alpha\nbravo\ncharlie\n").expect("write");

        let tool = ReadFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "regex": r"xyz\d+"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("No matches found"));
    }

    #[tokio::test]
    async fn test_read_file_regex_invalid_pattern_errors() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("rg.txt");
        std::fs::write(&file_path, "anything\n").expect("write");

        let tool = ReadFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "regex": "[invalid"
                }),
                CancellationToken::new(),
            )
            .await;
        let err = result.expect_err("invalid regex must surface as an error");
        assert!(err.to_string().contains("invalid or oversized regex"));
    }

    #[tokio::test]
    async fn test_read_file_regex_caps_at_max_matches() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("rg.txt");
        let mut body = String::new();
        for i in 0..150 {
            body.push_str(&format!("match-{}\n", i));
        }
        std::fs::write(&file_path, &body).expect("write");

        let tool = ReadFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "regex": "match-"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        let text = text_content(&result);
        assert!(
            text.contains("showing first 100 of 150 matches"),
            "expected truncation trailer; got: {}",
            text
        );
    }

    #[tokio::test]
    async fn test_edit_file_insert_before() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "anchor line\n").expect("write");

        let tool = EditFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "anchor",
                    "insert_before": "prefix-",
                    "force": true
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error, "got: {}", text_content(&result));
        let content = std::fs::read_to_string(&file_path).expect("read");
        assert_eq!(content, "prefix-anchor line\n");
    }

    #[tokio::test]
    async fn test_edit_file_insert_after() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "anchor line\n").expect("write");

        let tool = EditFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "anchor",
                    "insert_after": "-suffix",
                    "force": true
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error, "got: {}", text_content(&result));
        let content = std::fs::read_to_string(&file_path).expect("read");
        assert_eq!(content, "anchor-suffix line\n");
    }

    #[tokio::test]
    async fn test_edit_file_rejects_replace_and_insert_combined() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "anchor\n").expect("write");

        let tool = EditFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "anchor",
                    "new_string": "replaced",
                    "insert_after": "tail",
                    "force": true
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(result.is_error);
        assert!(text_content(&result).contains("mutually exclusive"));
    }

    #[tokio::test]
    async fn test_edit_file_rejects_both_insert_directions() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "anchor\n").expect("write");

        let tool = EditFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "anchor",
                    "insert_before": "head",
                    "insert_after": "tail",
                    "force": true
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(result.is_error);
        assert!(text_content(&result).contains("mutually exclusive"));
    }

    #[tokio::test]
    async fn test_edit_file_rejects_no_mode() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "anchor\n").expect("write");

        let tool = EditFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "anchor",
                    "force": true
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(result.is_error);
        assert!(
            text_content(&result).contains("provide one of"),
            "got: {}",
            text_content(&result)
        );
    }

    #[tokio::test]
    async fn test_edit_file_success_includes_context_snippet() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("edit.txt");
        let body = (1..=10)
            .map(|i| format!("line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&file_path, body).expect("write");

        let tool = EditFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "line 5",
                    "new_string": "FIVE",
                    "force": true
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        let text = text_content(&result);
        assert!(text.contains("Successfully edited"));
        // Context snippet shows the edited line plus ±3 around it.
        assert!(text.contains("FIVE"));
        assert!(text.contains("line 2"));
        assert!(text.contains("line 8"));
        assert!(!text.contains("line 1\n"));
        assert!(!text.contains("line 9\n"));
    }

    #[tokio::test]
    async fn test_edit_file_multi_match_trailer() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "x\nx\nx\n").expect("write");

        let tool = EditFileTool {
            read_tracker: test_tracker(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "x",
                    "new_string": "y",
                    "replace_all": true,
                    "force": true
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        let text = text_content(&result);
        assert!(text.contains("3 occurrence(s)"));
        assert!(text.contains("first of 3 occurrences"));
    }

    #[tokio::test]
    async fn test_read_file_a_edit_file_b_fails() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_a = temp_dir.path().join("a.txt");
        let file_b = temp_dir.path().join("b.txt");
        std::fs::write(&file_a, "content a").expect("failed to write");
        std::fs::write(&file_b, "content b").expect("failed to write");

        let tracker = test_tracker();

        let read_tool = ReadFileTool {
            read_tracker: tracker.clone(),
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        read_tool
            .execute(
                serde_json::json!({"path": file_a.to_str().expect("path")}),
                CancellationToken::new(),
            )
            .await
            .expect("read should succeed");

        let edit_tool = EditFileTool {
            read_tracker: tracker,
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = edit_tool
            .execute(
                serde_json::json!({
                    "path": file_b.to_str().expect("path"),
                    "old_string": "content",
                    "new_string": "modified"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(result.is_error);
        assert!(text_content(&result).contains("must be read before editing"));
    }
}
