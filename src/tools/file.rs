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
    ReadStamp, ReadTracker, Tool, ToolOutput,
    util::{MAX_SEARCH_MATCHES, canonicalize_for_tool, require_str, search_lines, truncate_string},
};
use crate::{
    error::{MekaError, Result},
    image::{ImageHandling, classify_extension, prepare_image_payload},
    permission::Permission,
    provider::{ImageSource, ToolDefinition, ToolResultContent},
};

/// Record `canonical` as read, stamped with what the file looks like right now.
///
/// A path that cannot be stated is simply not recorded, so the next `edit_file` asks for a re-read.
/// That is the right instruction: whatever stopped the stat will surface as a real error on the
/// second read, where it is legible, rather than as a silent edit against a file that moved.
async fn record_read(tracker: &ReadTracker, canonical: std::path::PathBuf) {
    if let Some(stamp) = ReadStamp::of_path(&canonical).await {
        tracker.write().await.insert(canonical, stamp);
    }
}

/// Record a read the frontend served, fingerprinting the text it gave us.
///
/// Not a disk stamp, because the two describe different documents. An editor serves its own copy of
/// every file it owns, saved or not, so a disk comparison is wrong in both directions: it fires
/// when the user saves a file nobody edited, and stays quiet when the user rewrites the buffer the
/// agent is about to edit. What *is* comparable is the next thing the editor serves, which
/// `edit_file` fetches anyway before editing.
async fn record_delegated_read(tracker: &ReadTracker, canonical: std::path::PathBuf, text: &str) {
    tracker
        .write()
        .await
        .insert(canonical, ReadStamp::of_delegated(text));
}

/// Record a file this tool has just written, stamped in whatever terms the route can answer for.
///
/// `content` is what was written, which on the delegated route is exactly what the editor now
/// holds, so the next edit compares against it and consecutive edits do not trip. Stamping a
/// delegated write from disk instead would leave the file looking unchanged until the user saved
/// and changed on the first save after that.
async fn record_write(
    tracker: &ReadTracker,
    canonical: std::path::PathBuf,
    route: FileRoute,
    content: &str,
) {
    if route.is_delegated() {
        record_delegated_read(tracker, canonical, content).await;
    } else {
        record_read(tracker, canonical).await;
    }
}

/// The complaint to return when the file moved under the agent since it read it, or `None` when the
/// read still stands.
///
/// Every comparison is like against like. A [`ReadStamp::Disk`] record is checked against the disk;
/// a [`ReadStamp::Delegated`] one against the text the frontend has just served. Crossing them is
/// what makes the check useless on an editor-hosted file: the editor serves its own copy of
/// everything it owns, saved or not, so disk state answers a different question than the one asked.
///
/// A record whose source no longer matches the route is passed rather than guessed at. That happens
/// when the editor adopts or disowns a file between the read and the edit, and neither source can
/// speak for the other; the next write re-stamps it in the current terms, so it self-corrects after
/// one call.
async fn stale_read_complaint(
    recorded: Option<ReadStamp>,
    route: FileRoute,
    content: &str,
    canonical: &Path,
    path: &str,
) -> Option<String> {
    match recorded? {
        // If the file cannot be stated now, say nothing: the edit's own write will produce the real
        // error, which is more legible than a staleness complaint about an unreachable file.
        disk @ ReadStamp::Disk { .. } if !route.is_delegated() => {
            (ReadStamp::of_path(canonical).await? != disk).then(|| {
                format!(
                    "Error: file '{}' changed on disk after you read it. Something else wrote to \
                     it (a shell command, another agent, or the user). Read it again before \
                     editing so you are not overwriting that change, or set force=true to edit \
                     anyway.",
                    path
                )
            })
        }
        served @ ReadStamp::Delegated { .. } if route.is_delegated() => {
            (ReadStamp::of_delegated(content) != served).then(|| {
                format!(
                    "Error: file '{}' changed in the editor after you read it. Someone edited the \
                     buffer, or the editor reloaded the file. Read it again before editing so you \
                     are not overwriting that change, or set force=true to edit anyway.",
                    path
                )
            })
        }
        // Said out loud at debug level rather than merely returning: a check that quietly declines
        // to run looks exactly like a check that ran and passed, and this whole comparison was
        // broken for a release by precisely that.
        mismatched => {
            tracing::debug!(
                "edit_file: '{}' was read from a different source than this edit ({:?} vs route \
                 {:?}); skipping the freshness check",
                canonical.display(),
                mismatched,
                route,
            );
            None
        }
    }
}

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

/// Per-path write locks, so two tool calls that mutate the same file cannot interleave.
///
/// Process-global rather than a field on the tools, because the hazard belongs to the file, not to
/// a registry: a sub-agent and its parent hold different `ToolRegistry` instances but write the
/// same disk, and a per-registry lock would let exactly the pair that shares a workspace race.
/// Keyed on the canonicalised path so two spellings of one file take the same lock.
///
/// The registry is a `std::sync::Mutex` holding only `Arc` clones -- no `.await` happens inside it,
/// so it never blocks the runtime -- while the per-path lock is a `tokio::sync::Mutex`, since it is
/// held across the read/modify/write awaits.
static WRITE_LOCKS: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, Arc<tokio::sync::Mutex<()>>>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Acquire the write lock for `canonical`, creating it on first use.
///
/// Entries whose only remaining owner is the map are dropped on the way past, which keeps it
/// bounded by the number of files being written *concurrently* rather than by every file the
/// session has ever touched.
async fn lock_path_for_write(canonical: &Path) -> tokio::sync::OwnedMutexGuard<()> {
    let lock = {
        let mut registry = match WRITE_LOCKS.lock() {
            Ok(guard) => guard,
            // A panic while holding this map leaves only `Arc` clones behind; the data cannot be
            // torn, so recovering is strictly better than propagating the poison into every later
            // write.
            Err(poisoned) => poisoned.into_inner(),
        };
        registry.retain(|_, held| Arc::strong_count(held) > 1);
        Arc::clone(
            registry
                .entry(canonical.to_path_buf())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
        )
    };
    lock.lock_owned().await
}

/// Resolve `path` to the file a write will land on, and take that file's write lock.
///
/// Shared by `write_file` and `scratchpad_save_file` because they must agree on both answers. They
/// did not: `scratchpad_save_file` carried its own copy of the path resolution and then called
/// [`write_file_bytes`] directly, so the two took no lock in common. Both are dispatched
/// concurrently from one assistant message and both compute a temp path from the same target, so a
/// `write_file` and a `scratchpad_save_file` naming one file could interleave their write-then-
/// rename and publish a spliced result.
///
/// The parent is created and canonicalized first, so the final open is pinned to a directory whose
/// symlinks are already resolved and a swap of some ancestor cannot redirect it. The full path is
/// then canonicalized when it resolves, which is what makes the lock key, the read-tracker key and
/// the bytes on disk name the same file as `read_file` and `edit_file` do; see the comment in
/// `WriteFileTool::execute` for what disagreeing about it cost. A path that does not resolve yet is
/// a create and keeps the joined form: there is no link to follow, and a dangling link is replaced
/// rather than followed, since writing through it would mean creating the file it names.
pub(super) async fn resolve_write_target(
    tool_name: &str,
    cwd: &crate::workspace::SharedCwd,
    path: &str,
) -> Result<(std::path::PathBuf, tokio::sync::OwnedMutexGuard<()>)> {
    let file_path = crate::workspace::resolve_against_cwd(cwd, path);
    let file_name = file_path
        .file_name()
        .ok_or_else(|| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("invalid path (no file name): '{}'", path),
        })?;
    let parent = file_path.parent().ok_or_else(|| MekaError::ToolExecution {
        tool_name: tool_name.to_string(),
        message: format!("invalid path (no parent): '{}'", path),
    })?;

    // An empty parent (a bare relative filename like "out.txt") is the current directory.
    let parent_for_create: &Path = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    tokio::fs::create_dir_all(parent_for_create)
        .await
        .map_err(|error| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("failed to create directories for '{}': {}", path, error),
        })?;

    let canonical_parent = canonicalize_for_tool(tool_name, parent_for_create).await?;
    let joined = canonical_parent.join(file_name);
    let target = match tokio::fs::canonicalize(&joined).await {
        Ok(resolved) => resolved,
        Err(_not_yet_a_file) => joined,
    };

    let guard = lock_path_for_write(&target).await;
    Ok((target, guard))
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

/// Ceiling on what one `read_file` will pull into memory.
///
/// Sits above `MAX_RESIDENT_OUTPUT_BYTES` (8 MiB) on purpose: a command's output is produced by a
/// process meka is already streaming and can spill, while a file is read whole in one call, and the
/// text ends up in the conversation where the window is the real limit long before this is. Any
/// file this large is one the model wants a slice of rather than the whole of.
const MAX_READ_FILE_BYTES: usize = 16 * 1024 * 1024;

/// Read a file's bytes, bounded by [`MAX_READ_FILE_BYTES`] exactly as the text path is.
///
/// The image branch of `read_file` selects on the *extension* alone, so a 3 GB `.tga` of non-image
/// data reached an unbounded `read_to_end` here, went fully resident, and only then failed
/// `classify_bytes` and fell through to the text read -- which applies the ceiling the image path
/// had already blown past. Same limit, same reason, on the sibling that missed it.
pub(super) async fn read_file_bytes(path: &Path) -> std::io::Result<Vec<u8>> {
    let file = open_read_nofollow(path).await?;
    let mut buffer = Vec::new();
    // One past the cap, so hitting it exactly is distinguishable from exceeding it.
    file.take(MAX_READ_FILE_BYTES as u64 + 1)
        .read_to_end(&mut buffer)
        .await?;
    if buffer.len() > MAX_READ_FILE_BYTES {
        return Err(std::io::Error::other(format!(
            "file is larger than the {} MiB read ceiling",
            MAX_READ_FILE_BYTES / (1024 * 1024)
        )));
    }
    Ok(buffer)
}

/// Read at most [`MAX_READ_FILE_BYTES`], erroring rather than filling memory.
///
/// The cancellation race at the call site makes a `/dev/zero` read *interruptible*; it does not
/// make it *bounded*, and an unattended `serve` or ACP session has nobody to press stop. A cap is
/// what turns "the process died" into "the tool said no", and the model can act on the second.
///
/// Reads one byte past the cap so hitting it exactly is distinguishable from exceeding it.
async fn read_file_to_string(path: &Path) -> std::io::Result<String> {
    let file = open_read_nofollow(path).await?;
    let mut buffer = String::new();
    let read = file
        .take(MAX_READ_FILE_BYTES as u64 + 1)
        .read_to_string(&mut buffer)
        .await?;
    if read > MAX_READ_FILE_BYTES {
        return Err(std::io::Error::other(format!(
            "file is larger than the {} MiB read_file ceiling; pass offset and limit to read it a \
             window at a time, or use execute_command with a tool that streams (head, tail, grep, \
             sed)",
            MAX_READ_FILE_BYTES / (1024 * 1024),
        )));
    }
    Ok(buffer)
}

/// Describe a window read out of a file too large to hold, in the same terms the in-memory path
/// uses, so the model cannot tell which route answered it apart from the extra ceiling note.
fn render_windowed_read(
    path: &str,
    window: String,
    offset: usize,
    total_lines: usize,
    cut_by_ceiling: bool,
) -> String {
    let shown_lines = if window.is_empty() {
        0
    } else {
        window.lines().count()
    };
    if shown_lines == 0 {
        // Two different facts, and the ceiling one used to be reported as the other. A minified
        // JSON blob or a base64 capture is one enormous line, so the window is empty because that
        // single line does not fit, not because the offset ran off the end -- and answering "offset
        // 0 is past the end of a file which has 1 line" is both self-contradictory and reads as
        // "unreadable" when the truth is "ask for it differently".
        if cut_by_ceiling {
            return format!(
                "(no lines: the first line at offset {} of '{}' is itself larger than the {} MiB \
                 read_file ceiling, so no whole line fits; use execute_command with a tool that \
                 slices by bytes, such as head -c or cut)",
                offset,
                path,
                MAX_READ_FILE_BYTES / (1024 * 1024),
            );
        }
        return format!(
            "(no lines: offset {} is past the end of '{}', which has {} line{})",
            offset,
            path,
            total_lines,
            if total_lines == 1 { "" } else { "s" },
        );
    }

    let last_shown = offset.saturating_add(shown_lines);
    let mut rendered = window;
    if cut_by_ceiling {
        rendered.push_str(&format!(
            "\n\n... (showing lines {}-{} of {}; the window stopped at the {} MiB read_file \
             ceiling, so ask for fewer lines to see the rest)",
            offset.saturating_add(1),
            last_shown,
            total_lines,
            MAX_READ_FILE_BYTES / (1024 * 1024),
        ));
    } else if last_shown < total_lines {
        rendered.push_str(&format!(
            "\n\n... (showing lines {}-{} of {}, use offset/limit to read more)",
            offset.saturating_add(1),
            last_shown,
            total_lines,
        ));
    }
    rendered
}

/// Read one line window out of a file, streaming past everything outside it.
///
/// Returns the window, the file's total line count, and whether the window itself was cut short by
/// the residency ceiling.
///
/// The ceiling bounds what this *keeps*, not how large a file it will look at, and that difference
/// is the point. Applying it to the file's size instead made a capture larger than the ceiling
/// unreadable by any means -- while `execute_command`'s own spill notice was telling the model the
/// file was "still reachable with `read_file`", and the shell docs said so too. A window is a
/// bounded amount of memory whatever the file's size, so refusing one bought nothing and cost the
/// model the only route to its own captured output.
async fn read_file_window(
    path: &Path,
    offset: usize,
    limit: usize,
) -> std::io::Result<(String, usize, bool)> {
    use tokio::io::AsyncBufReadExt;

    let file = open_read_nofollow(path).await?;
    let mut lines = tokio::io::BufReader::new(file).lines();
    let mut window = String::new();
    let mut total_lines = 0usize;
    let mut cut_by_ceiling = false;

    while let Some(line) = lines.next_line().await? {
        let inside_window = total_lines >= offset && total_lines - offset < limit;
        if inside_window && !cut_by_ceiling {
            // `+ 1` for the separator this line would carry.
            if window.len() + line.len() + 1 > MAX_READ_FILE_BYTES {
                cut_by_ceiling = true;
            } else {
                if !window.is_empty() {
                    window.push('\n');
                }
                window.push_str(&line);
            }
        }
        total_lines += 1;
    }

    Ok((window, total_lines, cut_by_ceiling))
}

/// Replace `path`'s contents atomically: write a sibling temp file, fsync it, then rename over the
/// target.
///
/// The previous shape opened the target with `truncate(true)` and streamed the new bytes in, so the
/// window between the truncate and the last write had the file at zero length with the original
/// content existing nowhere. A full disk, a kill, or a power loss in that window destroyed the
/// user's file and left `edit_file` reporting `No space left on device` about content that had been
/// in memory a microsecond earlier.
///
/// Rename preserves the `O_NOFOLLOW` guarantee rather than weakening it: `rename(2)` acts on the
/// directory entry, so a symlink swapped in at the final component is *replaced*, not written
/// through. The temp file is created in the target's own directory so the rename stays within one
/// filesystem (a cross-device rename fails with `EXDEV`).
///
/// On failure the temp file is removed; leaving `foo.txt.meka-tmp-…` beside a file the write did
/// not touch would be its own small confusion.
pub(super) async fn write_file_bytes(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file".to_string());
    // The pid keeps two meka processes writing the same path from colliding on the temp name; the
    // per-path lock only serialises within this one.
    let temp_path = parent.join(format!(".{}.meka-tmp-{}", file_name, std::process::id()));

    // Carry the target's mode across the rename.
    //
    // `rename(2)` replaces the inode, so the new file keeps the *temp* file's permissions, which
    // `open_write_nofollow` creates at `0o666 & ~umask` -- typically 0644. Before this function
    // wrote through a temp file it opened the target in place, so the mode was preserved for free.
    // Without this, `edit_file` on a 0600 secret returns it world-readable and reports success, and
    // on a 0755 script returns it non-executable. `config::write_file_atomic` sets its temp mode
    // explicitly for the same reason.
    //
    // Read before the write so a concurrent chmod loses the race rather than being half-applied.
    // A target that does not exist yet has no mode to carry; the umask default is correct there.
    #[cfg(unix)]
    let existing_mode = {
        use std::os::unix::fs::PermissionsExt;
        tokio::fs::metadata(path)
            .await
            .ok()
            .map(|metadata| metadata.permissions().mode())
    };

    // `open_write_nofollow` on the temp path: it is ours and should not exist, but a symlink
    // planted at that name would otherwise be written through.
    let write_result = async {
        let mut file = open_write_nofollow(&temp_path).await?;
        // Born at the target's mode, not merely renamed into it.
        //
        // The mode was applied to the temp file after `sync_all`, which is early enough that the
        // *published* path is never wrong -- but the temp file itself sits in the target's own
        // directory holding the full plaintext, and until that call it is `0o666 & ~umask`, i.e.
        // 0644 by default. Editing a 0600 secret in a 0755 directory therefore left its contents
        // world-readable for the whole write plus an fsync. Narrowing first and keeping the
        // post-write call means the window never opens, whichever order the filesystem completes
        // in.
        #[cfg(unix)]
        if let Some(mode) = existing_mode {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(std::fs::Permissions::from_mode(mode))
                .await?;
        }
        file.write_all(bytes).await?;
        file.flush().await?;
        // Durability before visibility: without this the rename can be recorded while the data is
        // still only in the page cache, so a crash leaves an intact directory entry pointing at an
        // empty or partial file -- the exact loss this function exists to prevent.
        file.sync_all().await
    }
    .await;

    if let Err(error) = write_result {
        if let Err(cleanup) = tokio::fs::remove_file(&temp_path).await {
            tracing::debug!(
                "failed to remove temp file '{}': {}",
                temp_path.display(),
                cleanup
            );
        }
        return Err(error);
    }

    // Applied to the temp file, before the rename, so the target is never briefly visible at the
    // wrong mode.
    #[cfg(unix)]
    if let Some(mode) = existing_mode {
        use std::os::unix::fs::PermissionsExt;
        if let Err(error) =
            tokio::fs::set_permissions(&temp_path, std::fs::Permissions::from_mode(mode)).await
        {
            // Refuse rather than publish the file at a looser mode than it had. The caller sees a
            // failed write on a target that still holds its original content, which is recoverable;
            // a silently world-readable secret is not.
            if let Err(cleanup) = tokio::fs::remove_file(&temp_path).await {
                tracing::debug!(
                    "failed to remove temp file '{}': {}",
                    temp_path.display(),
                    cleanup
                );
            }
            return Err(error);
        }
    }

    if let Err(error) = tokio::fs::rename(&temp_path, path).await {
        if let Err(cleanup) = tokio::fs::remove_file(&temp_path).await {
            tracing::debug!(
                "failed to remove temp file '{}': {}",
                temp_path.display(),
                cleanup
            );
        }
        return Err(error);
    }

    Ok(())
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
/// The file's current contents: `Ok(None)` when it does not exist, `Err` when it exists and cannot
/// be read.
///
/// The two used to collapse to `None`, and the staleness guard runs only on `Some` -- so a file
/// that exists but cannot be re-read (past the 16 MiB ceiling, or not valid UTF-8) skipped the
/// check entirely and was overwritten from a stale copy without a word. That is fail-open on
/// exactly the files where a blind overwrite costs most. `edit_file` refuses when it cannot verify;
/// this is the same posture, and `force` remains the escape hatch for both.
async fn local_old_text(target: &Path) -> std::result::Result<Option<String>, std::io::Error> {
    match read_file_to_string(target).await {
        Ok(text) => Ok(Some(text)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
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
            // Stopping the turn is not the client failing, and it is emphatically not a reason to
            // write the file locally instead: the editor may hold unsaved changes this content was
            // never diffed against.
            Some(Err(error)) if error.is_cancelled() => return Err(MekaError::Interrupted),
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
    pub cwd: crate::workspace::SharedCwd,
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
                        "description": "Maximum number of lines to read. Optional; defaults to 2000, and a read that is cut short says so."
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
        cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let path = input["path"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "read_file".to_string(),
                message: "missing 'path' parameter".to_string(),
            })?
            .to_string();

        let resolved = crate::workspace::resolve_against_cwd(&self.cwd, &path);
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
            // Raced against the token for the same reason the text read is: the bound above stops
            // this filling memory, but on a slow or blocking path (a fuse mount, a device file with
            // an image extension) it is still the thing standing between a stop keypress and the
            // turn ending.
            let data = tokio::select! {
                _ = cancellation.cancelled() => return Err(MekaError::Interrupted),
                read = read_file_bytes(&canonical) => read,
            }
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
                // Decoding and re-encoding a multi-megapixel image is tens of milliseconds of pure
                // CPU, and on the runtime it blocks every other task on that worker: a `serve`
                // process would stall unrelated sessions' streams behind one agent's screenshot.
                let prepared = tokio::task::spawn_blocking({
                    let data = data.clone();
                    move || prepare_image_payload(sniffed, &data)
                })
                .await
                .map_err(|error| MekaError::ToolExecution {
                    tool_name: "read_file".to_string(),
                    message: format!("image decode task failed: {}", error),
                })?;
                let (media_type, payload) = match prepared {
                    Ok(pair) => pair,
                    Err(message) => {
                        return Ok(ToolOutput::text(
                            format!("Error: image '{}': {}", path, message),
                            true,
                        ));
                    }
                };

                let base64_data = base64::engine::general_purpose::STANDARD.encode(&payload);

                record_read(&self.read_tracker, canonical).await;

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

        // Text reads delegate to the editor when it offers `fs.read_text_file`, so the model is
        // shown the document the editor will apply an edit against rather than the bytes under it.
        //
        // Including a regex read. There is no `fs/*` analogue for searching, but none is needed:
        // fetch the file through the same route and filter it here. Routing this one locally
        // instead searched the disk while `edit_file` went on to edit the buffer, and stamped the
        // read in terms `edit_file`'s freshness check could not compare against that buffer, so on
        // the common find-then-edit path the check silently skipped. Image reads stay local: they
        // are bytes, not text, and nothing edits them.
        // The delegate is asked for the whole document, never a window, and the windowing below is
        // applied to what it returns. Asking it for the window directly cost two things: the
        // freshness stamp recorded the *slice*, so every later `edit_file` compared it against the
        // whole buffer and reported a false "changed in the editor"; and a cut at exactly `limit`
        // lines was indistinguishable from a file that happened to end there, so it was returned
        // with no truncation notice at all. The local route reads the whole file too, so the two
        // now agree on both.
        let delegated = match self.frontend.delegate_fs_read(&canonical, None, None).await {
            Some(Ok(content)) => Some(content),
            // The client will not serve this path, so it holds no buffer for it either: the local
            // bytes are not a degraded substitute for the delegate's view, they are the same view.
            // Fall through and read them, rather than turning a readable file into a tool error --
            // which is what made skills, prompts, and configuration unreadable under ACP.
            Some(Err(error)) if error.is_unservable_path() => {
                tracing::debug!(
                    "read_file: client cannot serve '{}' ({}); reading it locally",
                    canonical.display(),
                    error,
                );
                None
            }
            // The turn was stopped while the client had the request. Reading locally instead would
            // be answering a question the user withdrew.
            Some(Err(error)) if error.is_cancelled() => return Err(MekaError::Interrupted),
            // Any other failure leaves open that the client owns this file and has unsaved changes
            // in it. Reading disk bytes would silently hand the model a stale view of a file the
            // user is editing, so surface the failure instead.
            Some(Err(error)) => {
                return Err(MekaError::ToolExecution {
                    tool_name: "read_file".to_string(),
                    message: format!("failed to read '{}': {}", path, error),
                });
            }
            None => None,
        };

        let content = match delegated {
            Some(content) => {
                record_delegated_read(&self.read_tracker, canonical.clone(), &content).await;
                content
            }
            None => {
                // A file past the ceiling is still readable a window at a time. Only this case
                // diverts: everything that fits is read whole, so the verbatim-CRLF return and the
                // freshness stamp below behave exactly as they did.
                if regex.is_none()
                    && (offset.is_some() || limit.is_some())
                    && tokio::fs::metadata(&canonical)
                        .await
                        .is_ok_and(|metadata| metadata.len() > MAX_READ_FILE_BYTES as u64)
                {
                    let start = offset.unwrap_or(0);
                    let span = limit.unwrap_or(DEFAULT_LINE_LIMIT);
                    let (window, total_lines, cut_by_ceiling) = tokio::select! {
                        _ = cancellation.cancelled() => return Err(MekaError::Interrupted),
                        result = read_file_window(&canonical, start, span) => {
                            result.map_err(|error| MekaError::ToolExecution {
                                tool_name: "read_file".to_string(),
                                message: format!("failed to read '{}': {}", path, error),
                            })?
                        }
                    };
                    // Deliberately not stamped for freshness: this read saw a window, and a stamp
                    // taken from one would make every later `edit_file` on the file report a false
                    // change. The whole-document stamping the delegated path does exists for the
                    // same reason, and a file this size is not an edit target anyway.
                    return Ok(ToolOutput::text(
                        render_windowed_read(&path, window, start, total_lines, cut_by_ceiling),
                        false,
                    ));
                }

                // Raced against cancellation because a path can be a device rather than a file:
                // `read_file("/dev/zero")` never returns, and without this the tool ignored the
                // turn's token and Ctrl+C alike, leaving the read running for the life of the
                // process while it consumed memory.
                let content = tokio::select! {
                    _ = cancellation.cancelled() => return Err(MekaError::Interrupted),
                    result = read_file_to_string(&canonical) => {
                        result.map_err(|error| MekaError::ToolExecution {
                            tool_name: "read_file".to_string(),
                            message: format!("failed to read '{}': {}", path, error),
                        })?
                    }
                };
                record_read(&self.read_tracker, canonical).await;
                content
            }
        };

        if let Some(pattern) = regex {
            return search_lines(&content, pattern, "read_file");
        }

        let total_lines = content.lines().count();
        let effective_offset = offset.unwrap_or(0);
        let effective_limit = limit.unwrap_or(DEFAULT_LINE_LIMIT);

        // A read that shows the whole file returns it verbatim, because the windowing below is also
        // a normalisation: `lines()` drops `\r` and `join("\n")` drops the trailing newline. The
        // model then copies an `old_string` out of LF text and `edit_file` cannot find it in the
        // CRLF file it is actually editing -- a "not found" whose cause is invisible in both the
        // read and the edit. Windowed reads still normalise; there is no way to slice lines and
        // keep their terminators without deciding which one each line ended with.
        if effective_offset == 0 && total_lines <= effective_limit {
            return Ok(ToolOutput::text(content, false));
        }

        let result: String = content
            .lines()
            .skip(effective_offset)
            .take(effective_limit)
            .collect::<Vec<_>>()
            .join("\n");

        // Disclose the cut whenever one happened, not only for a bare `read_file`.
        //
        // `effective_limit` defaults to `DEFAULT_LINE_LIMIT` regardless of whether `offset` was
        // given, but the notice used to be gated on *both* being absent -- so
        // `read_file({path, offset: 0})` on a 50,000-line log returned exactly 2,000 lines, with no
        // marker and no line count, and the model answered "the log contains no errors" from four
        // percent of the file. This is the failure the `find_files` / `search_contents` disclosures
        // exist to prevent; a definitive-sounding answer drawn from a silent truncation is worse
        // than an error.
        let shown_lines = result.lines().count();
        // An offset past the end returns nothing, and nothing reads as "the file is empty" -- which
        // is a different fact, and the one the model will act on. Say which it is.
        if shown_lines == 0 && effective_offset >= total_lines {
            return Ok(ToolOutput::text(
                format!(
                    "(no lines: offset {} is past the end of '{}', which has {} line{})",
                    effective_offset,
                    path,
                    total_lines,
                    if total_lines == 1 { "" } else { "s" },
                ),
                false,
            ));
        }
        let last_shown = effective_offset.saturating_add(shown_lines);
        let result = if last_shown < total_lines {
            format!(
                "{}\n\n... (showing lines {}-{} of {}, use offset/limit to read more)",
                result,
                effective_offset.saturating_add(1),
                last_shown,
                total_lines,
            )
        } else {
            result
        };

        Ok(ToolOutput::text(result, false))
    }
}

pub(super) struct EditFileTool {
    pub read_tracker: ReadTracker,
    pub cwd: crate::workspace::SharedCwd,
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
        let resolved = crate::workspace::resolve_against_cwd(&self.cwd, &path);
        let canonical = canonicalize_for_tool("edit_file", &resolved).await?;

        // Serialise everything from here to the write against other writers of this same file.
        //
        // An edit is read → check-freshness → modify → write with `.await` at every step, and the
        // agent dispatches all the tool calls in one assistant message concurrently
        // (`futures::future::join_all`). Two `edit_file` calls on one file therefore both read the
        // original, both pass the freshness gate (the tracker stamp is still the pre-edit one for
        // both), and the second write silently discards the first -- while *both* results report
        // success and show a context snippet proving the change landed. The freshness machinery
        // cannot catch this on its own: it compares against a stamp taken before either write.
        let _write_guard = lock_path_for_write(&canonical).await;

        // Bound the guard to a `let` rather than matching on it directly: a temporary in a match
        // scrutinee lives for the whole match, which would hold the lock across the `of_path` await
        // below.
        let recorded = self.read_tracker.read().await.get(&canonical).copied();
        if !force && recorded.is_none() {
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
            // The turn was stopped mid-request. Same short-circuit as below, reported as the stop
            // it is.
            Some(Err(error)) if error.is_cancelled() => return Err(MekaError::Interrupted),
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

        // Checked here, after the read, because a delegated record can only be checked against
        // what the same source serves now, and that is the text just fetched. Each stamp is
        // compared against its own kind of source; a record and a route that disagree (the editor
        // has since disowned the file, or adopted it) prove nothing about each other and pass.
        //
        // Deliberately not the "must be read" message below. The file *was* read, and the agent's
        // next move differs: re-read to see what changed, then decide whether the edit still
        // applies. Sending it to `read_file` for the wrong reason hides that something else is
        // writing here.
        if !force
            && let Some(stale) =
                stale_read_complaint(recorded, route, &content, &canonical, &path).await
        {
            return Ok(ToolOutput::text(stale, true));
        }

        if !content.contains(&old_string) {
            // Name the line endings when they are the likely cause. A windowed `read_file` shows
            // the model LF text, so an `old_string` spanning a line break will not match a CRLF
            // file, and nothing in a bare "not found" points at an invisible difference.
            let crlf_hint = if content.contains("\r\n")
                && old_string.contains('\n')
                && !old_string.contains("\r\n")
            {
                " (this file uses CRLF line endings; old_string uses LF)"
            } else {
                ""
            };
            return Ok(ToolOutput::text(
                format!(
                    "Error: '{}' not found in '{}'{}",
                    truncate_string(&old_string, 100),
                    path,
                    crlf_hint
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

        // Re-stamp: this edit is itself a change to the file, so without it the next `edit_file`
        // would compare against the pre-edit stamp and report the agent's own write as somebody
        // else's. Consecutive edits to one file are the common case, so that would be a constant
        // false alarm.
        record_write(&self.read_tracker, canonical.clone(), route, &new_content).await;

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
    pub cwd: crate::workspace::SharedCwd,
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
                    "force": {
                        "type": "boolean",
                        "description": "If true, overwrite an existing file even if it changed since it was read. Defaults to false."
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
        let force = input["force"].as_bool().unwrap_or(false);

        let (target, _write_guard) = resolve_write_target("write_file", &self.cwd, &path).await?;

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
                // The client served it, so there is nothing unreadable to refuse over.
                (Ok(old), FileRoute::Delegated)
            }
            Some(Err(error)) if error.is_unservable_path() => {
                tracing::debug!(
                    "write_file: client cannot serve '{}' ({}); writing it locally",
                    target.display(),
                    error,
                );
                (local_old_text(&target).await, FileRoute::LocalUnservable)
            }
            // Not a degraded probe: the user stopped the turn, so there is no write to do. Falling
            // through here would carry on and write the file after the stop.
            Some(Err(error)) if error.is_cancelled() => return Err(MekaError::Interrupted),
            // The client did not disown the path, it just failed this probe. The write still goes
            // to it; only the diff's `old_text` is degraded, and that is informational.
            Some(Err(error)) => {
                tracing::debug!(
                    "write_file: client pre-read of '{}' failed ({}); falling back to a local \
                     read for the diff",
                    target.display(),
                    error,
                );
                (local_old_text(&target).await, FileRoute::Delegated)
            }
            None => (local_old_text(&target).await, FileRoute::Local),
        };

        // A file that exists but could not be read is refused rather than silently overwritten.
        let old_text = match old_text {
            Ok(text) => text,
            Err(error) if force => {
                tracing::debug!(
                    "write_file: '{}' could not be pre-read ({}); force was set, writing anyway",
                    target.display(),
                    error,
                );
                None
            }
            Err(error) => {
                return Ok(ToolOutput::text(
                    format!(
                        "Error: '{}' exists but its current contents could not be read ({}), so \
                         meka cannot tell whether it changed since you last read it. Overwriting \
                         it blind could discard someone else's edit. Pass force to write anyway.",
                        path, error
                    ),
                    true,
                ));
            }
        };

        // Refuse to clobber a file that changed since the agent last read it.
        //
        // `edit_file` has always done this; `write_file` consulted the read tracker only to
        // *insert* into it, never to check. So the model reading `config.toml`, the user
        // editing and saving it, and the model then writing the whole file back from its
        // now-stale copy overwrote the user's change with no error, no warning and no
        // re-read prompt -- while the identical change routed through `edit_file` would
        // have been refused. Creating a new file stays unguarded (there is nothing to
        // lose), and `force` is the same escape hatch.
        if !force && let Some(existing) = old_text.as_deref() {
            let recorded = self.read_tracker.read().await.get(&target).copied();
            if let Some(stale) =
                stale_read_complaint(recorded, route, existing, &target, &path).await
            {
                return Ok(ToolOutput::text(stale, true));
            }
        }

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
        record_write(&self.read_tracker, target.clone(), route, &content).await;

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
    use std::{collections::HashMap, sync::Arc};

    use tokio::sync::RwLock;

    use super::*;
    use crate::tools::tests::text_content;

    fn test_tracker() -> ReadTracker {
        Arc::new(RwLock::new(HashMap::new()))
    }

    /// Seed the tracker as though `path` had just been read, stamped with its current state, so an
    /// `edit_file` under test passes the freshness gate the way a real read would leave it.
    async fn mark_read(tracker: &ReadTracker, path: &std::path::Path) -> std::path::PathBuf {
        let canonical = std::fs::canonicalize(path).expect("canonicalize");
        record_read(tracker, canonical.clone()).await;
        canonical
    }

    /// A frontend whose hosted filesystem answers with a scripted outcome, so each branch of the
    /// routing rule can be driven directly.
    ///
    /// The serving variant models a real editor rather than a fixed reply: it holds a buffer that
    /// reads return and writes update. Freshness on this route is judged by comparing what the
    /// editor served then against what it serves now, so a fixture that answered the same string
    /// forever could not tell a working check from an absent one.
    struct ScriptedDelegateFrontend {
        /// Consulted only when there is no buffer, which is how the failure fixtures answer.
        read: Option<std::result::Result<String, crate::frontend::FrontendError>>,
        write: Option<std::result::Result<(), crate::frontend::FrontendError>>,
        buffer: std::sync::Mutex<Option<String>>,
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
                buffer: std::sync::Mutex::new(None),
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
                buffer: std::sync::Mutex::new(None),
                delegated_writes: std::sync::Mutex::new(Vec::new()),
            }
        }

        /// A client serving its in-buffer view, which differs from the on-disk bytes.
        fn serving(buffer: &str) -> Self {
            Self {
                read: Some(Ok(buffer.to_string())),
                write: Some(Ok(())),
                buffer: std::sync::Mutex::new(Some(buffer.to_string())),
                delegated_writes: std::sync::Mutex::new(Vec::new()),
            }
        }

        /// A client that owns the path and was still holding the request when the user pressed
        /// stop, so no answer is coming. What `AcpFrontend::until_cancelled` reports.
        fn cancelled(buffer: Option<&str>) -> Self {
            Self {
                read: Some(Err(crate::frontend::FrontendError::cancelled(
                    "fs/read_text_file",
                ))),
                write: Some(Err(crate::frontend::FrontendError::cancelled(
                    "fs/write_text_file",
                ))),
                buffer: std::sync::Mutex::new(buffer.map(str::to_string)),
                delegated_writes: std::sync::Mutex::new(Vec::new()),
            }
        }

        /// Someone else changes the document: the user typing, or the editor reloading a file that
        /// was rewritten underneath it.
        fn set_buffer(&self, text: &str) {
            *self.buffer.lock().expect("lock") = Some(text.to_string());
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
            match self.buffer.lock().expect("lock").clone() {
                Some(text) => Some(Ok(text)),
                None => self.read.clone(),
            }
        }

        async fn delegate_fs_write(
            &self,
            path: &Path,
            content: &str,
        ) -> Option<std::result::Result<(), crate::frontend::FrontendError>> {
            self.delegated_writes
                .lock()
                .expect("lock")
                .push(path.to_path_buf());
            let outcome = self.write.clone();
            // An accepted write lands in the document, so later reads see it. Without this the
            // fixture would serve pre-edit text forever and consecutive edits could not be tested.
            if matches!(outcome, Some(Ok(())))
                && let Some(buffer) = self.buffer.lock().expect("lock").as_mut()
            {
                *buffer = content.to_string();
            }
            outcome
        }
    }

    /// Reports how many tool calls are inside the tool body at once. Every `edit_file` consults
    /// the delegate after it has taken the path lock, so a count taken here is a count taken
    /// where the lock is supposed to have serialised things. Answers `None`, leaving the edit to
    /// read locally exactly as `SilentFrontend` does.
    #[derive(Default)]
    struct ConcurrencyProbeFrontend {
        inside: std::sync::atomic::AtomicUsize,
        peak: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl crate::frontend::Frontend for ConcurrencyProbeFrontend {
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
            use std::sync::atomic::Ordering;
            let now = self.inside.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            // Long enough that the other call is certainly polled while this one is parked here,
            // so a missing lock shows up as a peak of two rather than as a lost race.
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            self.inside.fetch_sub(1, Ordering::SeqCst);
            None
        }
    }

    /// The lost-edit test above passes with the path lock removed, because the freshness check
    /// catches the interleaving it happens to produce: the first write lands before the second
    /// read, so the second edit is refused as stale. The freshness check cannot catch the other
    /// one, where both edits read and both pass the gate before either writes -- the recorded
    /// stamp is the pre-edit one for both, so both are judged fresh and the second write
    /// discards the first. Only the lock closes that, so pin the lock rather than the symptom:
    /// while one edit is inside the tool body, no second edit may enter it.
    #[tokio::test]
    async fn one_edit_holds_the_path_until_it_is_done_with_it() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("shared.txt");
        std::fs::write(&file_path, "alpha\nbeta\n").expect("write");

        let tracker = test_tracker();
        mark_read(&tracker, &file_path).await;
        let probe = Arc::new(ConcurrencyProbeFrontend::default());
        let make_tool = || EditFileTool {
            read_tracker: tracker.clone(),
            cwd: crate::workspace::test_cwd(),
            frontend: probe.clone(),
        };
        let path_arg = file_path.to_str().expect("path").to_string();

        let tool_one = make_tool();
        let tool_two = make_tool();
        let (first, second) = tokio::join!(
            tool_one.execute(
                serde_json::json!({"path": path_arg, "old_string": "alpha", "new_string": "ALPHA"}),
                CancellationToken::new(),
            ),
            tool_two.execute(
                serde_json::json!({"path": path_arg, "old_string": "beta", "new_string": "BETA"}),
                CancellationToken::new(),
            ),
        );
        first.expect("first edit");
        second.expect("second edit");

        assert_eq!(
            probe.peak.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "two edits on one path were inside the tool body at the same time",
        );
    }

    #[tokio::test]
    async fn test_read_file() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, "line1\nline2\nline3\n").expect("failed to write");

        let tool = ReadFileTool {
            read_tracker: test_tracker(),
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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

    /// The agent dispatches every tool call in one assistant message concurrently, so two
    /// `edit_file` calls on the same file used to both read the original, both pass the freshness
    /// gate against a stamp taken before either wrote, and the second write discarded the first --
    /// with both results reporting success. Serialising them means the loser sees the winner's
    /// content: either it applies on top, or its `old_string` no longer matches and it says so.
    /// Silently reporting two successes for one surviving edit is the outcome ruled out.
    #[tokio::test]
    async fn concurrent_edits_to_one_file_cannot_both_report_success_and_lose_one() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("shared.txt");
        std::fs::write(&file_path, "alpha\nbeta\n").expect("write");

        let tracker = test_tracker();
        mark_read(&tracker, &file_path).await;
        let make_tool = || EditFileTool {
            read_tracker: tracker.clone(),
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let path_arg = file_path.to_str().expect("path").to_string();

        let tool_one = make_tool();
        let tool_two = make_tool();
        let (first, second) = tokio::join!(
            tool_one.execute(
                serde_json::json!({"path": path_arg, "old_string": "alpha", "new_string": "ALPHA"}),
                CancellationToken::new(),
            ),
            tool_two.execute(
                serde_json::json!({"path": path_arg, "old_string": "beta", "new_string": "BETA"}),
                CancellationToken::new(),
            ),
        );

        let final_content = std::fs::read_to_string(&file_path).expect("read back");
        let succeeded = [&first, &second]
            .iter()
            .filter(|result| result.as_ref().is_ok_and(|output| !output.is_error))
            .count();
        let applied = usize::from(final_content.contains("ALPHA"))
            + usize::from(final_content.contains("BETA"));

        assert_eq!(
            succeeded, applied,
            "every reported success must be visible on disk; got {} success(es) but {} edit(s) \
             applied in {:?}",
            succeeded, applied, final_content
        );
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
        mark_read(&tracker, &file_path).await;
        let tool = EditFileTool {
            read_tracker: tracker,
            cwd: crate::workspace::test_cwd(),
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
        mark_read(&tracker, &file_path).await;
        let tool = EditFileTool {
            read_tracker: tracker,
            cwd: crate::workspace::test_cwd(),
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

    /// The Zed workflow: a file open with unsaved edits, read through the delegate, then saved by
    /// the user before the agent edits it. The save moves the disk stamp while changing nothing the
    /// agent had not already been shown, so a disk comparison would refuse the edit and blame a
    /// concurrent writer that does not exist.
    #[tokio::test]
    async fn test_a_delegated_read_is_not_invalidated_by_the_user_saving() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("open-in-editor.md");
        std::fs::write(&file_path, "saved bytes\n").expect("write");
        let canonical = std::fs::canonicalize(&file_path).expect("canonicalize");

        // Read through the editor, which is holding unsaved changes.
        let frontend = Arc::new(ScriptedDelegateFrontend::serving("unsaved buffer\n"));
        let tracker = test_tracker();
        record_delegated_read(&tracker, canonical.clone(), "unsaved buffer\n").await;

        // The user hits save: the disk now matches the buffer, and its stamp has moved.
        std::fs::write(&file_path, "unsaved buffer\n").expect("save");

        let tool = EditFileTool {
            read_tracker: tracker,
            cwd: crate::workspace::test_cwd(),
            frontend,
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "buffer",
                    "new_string": "BUFFER"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should return Ok");

        assert!(
            !result.is_error,
            "saving one's own unsaved edits is not a third party overwriting the file: {}",
            text_content(&result)
        );
    }

    /// The same false alarm one step later. The edit itself re-stamps the file, and stamping a
    /// delegated write from disk would put the tracker straight back into the state the test above
    /// exists to prevent: the write went to the editor's document, so the disk it left behind is
    /// still the user's to save whenever they like.
    #[tokio::test]
    async fn test_a_delegated_write_is_not_invalidated_by_the_user_saving() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("open-in-editor.md");
        std::fs::write(&file_path, "saved bytes\n").expect("write");
        let canonical = std::fs::canonicalize(&file_path).expect("canonicalize");

        let frontend = Arc::new(ScriptedDelegateFrontend::serving("alpha in the buffer\n"));
        let tracker = test_tracker();
        record_delegated_read(&tracker, canonical.clone(), "alpha in the buffer\n").await;
        let tool = EditFileTool {
            read_tracker: tracker,
            cwd: crate::workspace::test_cwd(),
            frontend,
        };
        let first = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "alpha",
                    "new_string": "beta"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should return Ok");
        assert!(!first.is_error, "{}", text_content(&first));

        // The user saves. The document the editor serves is unchanged -- it already held the
        // agent's edit -- so nothing the agent was shown has moved; only the bytes on disk.
        std::fs::write(&file_path, "something else entirely\n").expect("save");

        let second = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "beta",
                    "new_string": "gamma"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should return Ok");
        assert!(
            !second.is_error,
            "a delegated write must be stamped in the editor's terms, not the disk's: {}",
            text_content(&second)
        );
    }

    /// A regex read is a read, so it has to leave the tracker in the same terms as any other.
    ///
    /// It used to route locally on the grounds that searching has no `fs/*` call of its own, which
    /// left the find-then-edit path -- grep for the anchor, then edit it, the most ordinary thing
    /// the agent does -- searching the disk while the edit went to the buffer, and recording a
    /// stamp the freshness check could not compare against that buffer. The check does not fail
    /// loudly in that state; it declines to run.
    #[tokio::test]
    async fn test_a_delegated_regex_read_searches_and_stamps_the_editors_copy() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("open-in-editor.md");
        std::fs::write(&file_path, "on disk only\n").expect("write");

        let frontend = Arc::new(ScriptedDelegateFrontend::serving("needle in the buffer\n"));
        let tracker = test_tracker();
        let read = ReadFileTool {
            read_tracker: tracker.clone(),
            cwd: crate::workspace::test_cwd(),
            frontend: frontend.clone(),
        };
        let found = read
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "regex": "needle"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should return Ok");
        let text = text_content(&found);
        assert!(
            text.contains("needle in the buffer"),
            "a search must run over the document the editor holds: {text}"
        );

        // Someone rewrites the document between the search and the edit.
        frontend.set_buffer("needle, and a paragraph the agent has never seen\n");

        let edit = EditFileTool {
            read_tracker: tracker,
            cwd: crate::workspace::test_cwd(),
            frontend,
        };
        let result = edit
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "needle",
                    "new_string": "pin"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should return Ok");
        assert!(
            result.is_error,
            "a search read must arm the same freshness check every other read does"
        );
        assert!(text_content(&result).contains("changed in the editor"));
    }

    /// The other half, and the half whose absence let a broken check ship: an editor-hosted file
    /// whose *document* changes between the read and the edit must be refused.
    ///
    /// This is the ordinary case, not an exotic one. An editor serves its copy of every file it
    /// owns, saved or not, so under ACP essentially every project file is read through the
    /// delegate. Exempting that route from freshness checking altogether -- which is what comparing
    /// it against the disk and then giving up amounts to -- switches the protection off for the
    /// whole project, and nothing that only tests the no-false-alarm direction can see it.
    ///
    /// The replacement text is still present in the new document, so a `not found` rejection cannot
    /// be what fires here.
    #[tokio::test]
    async fn test_a_delegated_read_is_invalidated_by_the_document_changing() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("open-in-editor.md");
        std::fs::write(&file_path, "alpha\n").expect("write");
        let canonical = std::fs::canonicalize(&file_path).expect("canonicalize");

        let frontend = Arc::new(ScriptedDelegateFrontend::serving("alpha\n"));
        let tracker = test_tracker();
        record_delegated_read(&tracker, canonical.clone(), "alpha\n").await;

        // Someone rewrites the document: the user typing into the buffer, or the editor reloading
        // a file that a shell command or another agent rewrote underneath it.
        frontend.set_buffer("alpha, plus a paragraph the agent has never seen\n");

        let tool = EditFileTool {
            read_tracker: tracker,
            cwd: crate::workspace::test_cwd(),
            frontend,
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "alpha",
                    "new_string": "beta"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should return Ok");

        assert!(
            result.is_error,
            "a document that moved under the agent must be reported, not edited blind"
        );
        let text = text_content(&result);
        assert!(text.contains("changed in the editor"), "{text}");
        assert!(!text.contains("must be read before editing"), "{text}");
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
        record_read(&tracker, canonical.clone()).await;
        let tool = EditFileTool {
            read_tracker: tracker,
            cwd: crate::workspace::test_cwd(),
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
            buffer: std::sync::Mutex::new(Some("unsaved buffer\n".to_string())),
            delegated_writes: std::sync::Mutex::new(Vec::new()),
        });
        let tracker = test_tracker();
        record_read(&tracker, canonical).await;
        let tool = EditFileTool {
            read_tracker: tracker,
            cwd: crate::workspace::test_cwd(),
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
            buffer: std::sync::Mutex::new(None),
            delegated_writes: std::sync::Mutex::new(Vec::new()),
        });
        let tool = WriteFileTool {
            read_tracker: test_tracker(),
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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

    /// Stopping a turn while the editor holds an `fs/*` request must not turn into a local write.
    ///
    /// `delegate_fs_write` returns `Option<Result<..>>`, where `None` means "this frontend has no
    /// delegate, do it locally". `AcpFrontend` reported cancellation as exactly that `None`, so
    /// pressing stop mid-write did the write anyway, on disk, behind the back of an editor that
    /// may have been holding unsaved changes for the same file.
    #[tokio::test]
    async fn test_write_file_cancelled_delegate_does_not_fall_back_to_disk() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("owned.txt");
        std::fs::write(&file_path, "on disk\n").expect("seed");

        let tool = WriteFileTool {
            read_tracker: test_tracker(),
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(ScriptedDelegateFrontend::cancelled(Some("in buffer\n"))),
        };
        let error = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "content": "the model's version\n",
                    "force": true,
                }),
                CancellationToken::new(),
            )
            .await
            .expect_err("a cancelled write must not become a local write");
        assert!(
            matches!(error, MekaError::Interrupted),
            "a stopped turn reads as an interruption, not a client failure: {}",
            error,
        );
        assert_eq!(
            std::fs::read_to_string(&file_path).expect("read back"),
            "on disk\n",
            "the file must be untouched",
        );
    }

    /// The read half: `edit_file`'s pre-read picks the route *and* supplies the text the edit is
    /// computed from, so a cancelled read that fell through to disk would diff against bytes the
    /// editor had already moved past.
    #[tokio::test]
    async fn test_edit_file_cancelled_delegate_read_stops_the_edit() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("owned.txt");
        std::fs::write(&file_path, "on disk\n").expect("seed");

        let tracker = test_tracker();
        mark_read(&tracker, &file_path).await;
        let tool = EditFileTool {
            read_tracker: tracker,
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(ScriptedDelegateFrontend::cancelled(None)),
        };
        let error = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "on disk",
                    "new_string": "edited",
                }),
                CancellationToken::new(),
            )
            .await
            .expect_err("a cancelled pre-read must stop the edit");
        assert!(
            matches!(error, MekaError::Interrupted),
            "expected an interruption, got: {}",
            error,
        );
        assert_eq!(
            std::fs::read_to_string(&file_path).expect("read back"),
            "on disk\n",
            "the file must be untouched",
        );
    }

    /// `read_file` has the mildest consequence of the three -- a stale view rather than a lost
    /// edit -- and the same rule: a withdrawn question is not answered from somewhere else.
    #[tokio::test]
    async fn test_read_file_cancelled_delegate_does_not_read_disk() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("owned.txt");
        std::fs::write(&file_path, "on disk\n").expect("seed");

        let tool = ReadFileTool {
            read_tracker: test_tracker(),
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(ScriptedDelegateFrontend::cancelled(None)),
        };
        let error = tool
            .execute(
                serde_json::json!({"path": file_path.to_str().expect("path")}),
                CancellationToken::new(),
            )
            .await
            .expect_err("a cancelled read must not fall through to disk");
        assert!(
            matches!(error, MekaError::Interrupted),
            "expected an interruption, got: {}",
            error,
        );
    }

    #[tokio::test]
    async fn test_read_file_with_offset_and_limit() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, "line0\nline1\nline2\nline3\nline4\n").expect("failed to write");

        let tool = ReadFileTool {
            read_tracker: test_tracker(),
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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

    /// `write_file` through a symlink must name the same file `read_file` and `edit_file` name.
    ///
    /// All three canonicalize now, but `write_file` used to canonicalize only the parent and
    /// re-join the filename, so a symlinked final component gave it a path the other two never
    /// use. Three things followed from that one mismatch: the freshness check looked up a tracker
    /// key nothing writes and so never fired, `edit_file` and `write_file` on the one file took
    /// different per-path locks and stopped being serialised against each other, and the write
    /// itself landed on the link rather than through it -- replacing a dotfile-managed symlink
    /// with a regular file and leaving the file the model had just read untouched.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_write_file_follows_a_symlink_to_the_file_it_read() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let real = temp_dir.path().join("real.txt");
        std::fs::write(&real, "original").expect("seed real file");
        let link = temp_dir.path().join("link.txt");
        std::os::unix::fs::symlink(&real, &link).expect("symlink");

        let tracker = test_tracker();
        let write_tool = WriteFileTool {
            read_tracker: Arc::clone(&tracker),
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let output = write_tool
            .execute(
                serde_json::json!({
                    "path": link.to_str().expect("path"),
                    "content": "rewritten",
                    "force": true,
                }),
                CancellationToken::new(),
            )
            .await
            .expect("write should succeed");
        assert!(
            !output.is_error,
            "write_file through a symlink must succeed"
        );

        assert_eq!(
            std::fs::read_to_string(&real).expect("read real"),
            "rewritten",
            "the write must land on the file the link points at",
        );
        assert!(
            std::fs::symlink_metadata(&link)
                .expect("link metadata")
                .file_type()
                .is_symlink(),
            "the link itself must survive the write",
        );

        // The stamp lands under the canonical name, which is the name `read_file` and `edit_file`
        // look up. Recorded under the link's name it was invisible to both.
        let canonical = std::fs::canonicalize(&real).expect("canonicalize");
        assert!(
            tracker.read().await.contains_key(&canonical),
            "the write must stamp the tracker under the canonical path",
        );
    }

    /// The staleness guard has to fire for a symlinked path too, which is the concrete loss the
    /// key mismatch caused: `read_file` stamped the canonical name, `write_file` looked up the
    /// link's, missed, and clobbered whatever the user had saved in between.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_write_file_refuses_a_stale_write_through_a_symlink() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let real = temp_dir.path().join("real.txt");
        std::fs::write(&real, "original").expect("seed real file");
        let link = temp_dir.path().join("link.txt");
        std::os::unix::fs::symlink(&real, &link).expect("symlink");

        let tracker = test_tracker();
        mark_read(&tracker, &link).await;

        // The user saves over it between the read and the write.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        std::fs::write(&real, "the user's edit").expect("user edit");

        let write_tool = WriteFileTool {
            read_tracker: Arc::clone(&tracker),
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let output = write_tool
            .execute(
                serde_json::json!({
                    "path": link.to_str().expect("path"),
                    "content": "the model's stale copy",
                }),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(output.is_error, "a stale write through a link must refuse");
        assert_eq!(
            std::fs::read_to_string(&real).expect("read real"),
            "the user's edit",
            "the user's change must survive",
        );
    }

    /// The Windows half of `test_write_file_follows_a_symlink_to_the_file_it_read`.
    ///
    /// This used to assert the opposite -- that a symlinked target was refused -- which made
    /// Windows the one platform where `write_file` and `edit_file` disagreed about whether a link
    /// could be written at all. The reparse-point check in `open_write_nofollow` still stands; it
    /// guards the path *after* resolution, which is where a swap is a redirection rather than the
    /// user's own indirection.
    #[cfg(windows)]
    #[tokio::test]
    async fn test_write_file_follows_a_symlink_on_windows() {
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
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let output = write_tool
            .execute(
                serde_json::json!({
                    "path": link.to_str().expect("path"),
                    "content": "rewritten",
                    "force": true,
                }),
                CancellationToken::new(),
            )
            .await
            .expect("write should succeed");
        assert!(
            !output.is_error,
            "write_file through a symlink must succeed"
        );

        assert_eq!(
            std::fs::read_to_string(&real).expect("read real"),
            "rewritten",
            "the write must land on the file the link points at",
        );
        assert!(
            std::fs::symlink_metadata(&link)
                .expect("link metadata")
                .file_type()
                .is_symlink(),
            "the link itself must survive the write",
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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

    /// The silent-clobber shape: read, something else writes, edit lands against bytes that are no
    /// longer there. A shell `sed -i`, a concurrent agent, or the user's editor all produce it.
    #[tokio::test]
    async fn test_edit_refuses_a_file_that_changed_after_the_read() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("raced.txt");
        std::fs::write(&file_path, "hello world").expect("write");

        let tracker = test_tracker();
        mark_read(&tracker, &file_path).await;

        // Something else rewrites it. Length differs, so the stamp differs even where the
        // filesystem's mtime resolution is coarse.
        std::fs::write(&file_path, "hello world, and then some").expect("rewrite");

        let tool = EditFileTool {
            read_tracker: tracker,
            cwd: crate::workspace::test_cwd(),
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
            .expect("should return Ok");

        assert!(result.is_error);
        let text = text_content(&result);
        assert!(text.contains("changed on disk"), "{text}");
        // Must not be the never-read message: the agent's next move differs, and sending it to
        // `read_file` for the wrong reason hides that something else is writing here.
        assert!(!text.contains("must be read before editing"), "{text}");
        assert_eq!(
            std::fs::read_to_string(&file_path).expect("read"),
            "hello world, and then some",
            "the other writer's content must survive"
        );
    }

    /// The same race as above, routed through `write_file` instead of `edit_file`.
    ///
    /// A whole-file rewrite is the *more* destructive of the two, and it used to be the unguarded
    /// one: the model's stale copy replaced the user's saved edit with no error and no re-read
    /// prompt.
    #[tokio::test]
    async fn a_whole_file_rewrite_cannot_silently_clobber_a_concurrent_edit() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("raced.txt");
        std::fs::write(&file_path, "hello world").expect("write");

        let tracker = test_tracker();
        mark_read(&tracker, &file_path).await;

        std::fs::write(&file_path, "the user's saved edit").expect("rewrite");

        let tool = WriteFileTool {
            read_tracker: tracker.clone(),
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let arguments = serde_json::json!({
            "path": file_path.to_str().expect("path"),
            "content": "what the model still thinks is there",
        });
        let result = tool
            .execute(arguments.clone(), CancellationToken::new())
            .await
            .expect("should return Ok");

        assert!(result.is_error);
        assert!(text_content(&result).contains("changed on disk"));
        assert_eq!(
            std::fs::read_to_string(&file_path).expect("read"),
            "the user's saved edit",
            "the user's content must survive"
        );

        // `force` is the same escape hatch `edit_file` offers, and it must still write.
        let mut forced = arguments;
        forced["force"] = serde_json::Value::Bool(true);
        let result = tool
            .execute(forced, CancellationToken::new())
            .await
            .expect("should return Ok");
        assert!(!result.is_error, "{}", text_content(&result));
        assert_eq!(
            std::fs::read_to_string(&file_path).expect("read"),
            "what the model still thinks is there"
        );
    }

    /// The model edits what it was shown, so what it is shown has to be what is there. Windowing a
    /// whole-file read through `lines()` rewrote CRLF to LF, and the resulting `old_string` then
    /// matched nothing.
    #[tokio::test]
    async fn a_whole_file_read_preserves_crlf_so_a_later_edit_matches() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("dos.txt");
        std::fs::write(&file_path, "first\r\nsecond\r\nthird\r\n").expect("write");

        let tracker = test_tracker();
        let read = ReadFileTool {
            read_tracker: tracker.clone(),
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = read
            .execute(
                serde_json::json!({ "path": file_path.to_str().expect("path") }),
                CancellationToken::new(),
            )
            .await
            .expect("should return Ok");
        let shown = text_content(&result);
        assert_eq!(shown, "first\r\nsecond\r\nthird\r\n");

        // The round trip that matters: an `old_string` copied out of what was shown must apply.
        let edit = EditFileTool {
            read_tracker: tracker,
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = edit
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "first\r\nsecond",
                    "new_string": "first\r\nSECOND",
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should return Ok");
        assert!(!result.is_error, "{}", text_content(&result));
        assert_eq!(
            std::fs::read_to_string(&file_path).expect("read"),
            "first\r\nSECOND\r\nthird\r\n"
        );
    }

    /// A windowed read still normalises, so the failure it can cause has to be legible.
    #[tokio::test]
    async fn an_lf_old_string_against_a_crlf_file_names_the_line_endings() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("dos.txt");
        std::fs::write(&file_path, "first\r\nsecond\r\n").expect("write");

        let tracker = test_tracker();
        mark_read(&tracker, &file_path).await;

        let tool = EditFileTool {
            read_tracker: tracker,
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "first\nsecond",
                    "new_string": "first\nSECOND",
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should return Ok");

        assert!(result.is_error);
        assert!(
            text_content(&result).contains("CRLF"),
            "{}",
            text_content(&result)
        );
    }

    /// A windowed delegated read must disclose the cut and still stamp the whole document.
    ///
    /// Asking the editor for the window directly returned exactly `limit` lines, which is
    /// indistinguishable from a file that ends there, so the model got a silent truncation; and the
    /// stamp recorded that slice, so the next `edit_file` compared it against the whole buffer and
    /// refused a perfectly good edit with "changed in the editor".
    #[tokio::test]
    async fn a_windowed_delegated_read_discloses_the_cut_and_stamps_the_whole_document() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("buffered.txt");
        std::fs::write(&file_path, "on disk, not the buffer\n").expect("write");

        let buffer: String = (1..=100).map(|n| format!("line {n}\n")).collect();
        let frontend = Arc::new(ScriptedDelegateFrontend::serving(&buffer));
        let tracker = test_tracker();

        let read = ReadFileTool {
            read_tracker: tracker.clone(),
            cwd: crate::workspace::test_cwd(),
            frontend: frontend.clone(),
        };
        let result = read
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "offset": 9,
                    "limit": 50,
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should return Ok");
        let shown = text_content(&result);
        assert!(shown.starts_with("line 10\n"), "{shown}");
        assert!(shown.contains("showing lines 10-59 of 100"), "{shown}");

        // The window came from the buffer, not the disk.
        assert!(!shown.contains("on disk"), "{shown}");

        // The stamp has to describe the document, so an edit against a line the window never showed
        // is accepted rather than read as someone else's write.
        let edit = EditFileTool {
            read_tracker: tracker,
            cwd: crate::workspace::test_cwd(),
            frontend,
        };
        let result = edit
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "line 90",
                    "new_string": "line ninety",
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should return Ok");
        assert!(!result.is_error, "{}", text_content(&result));
    }

    /// The image path is bounded by the same ceiling the text path is.
    ///
    /// `read_file` picks the image branch on the *extension* alone, so a large file that merely
    /// ends `.tga` reached an unbounded `read_to_end`, went fully resident, and only then
    /// failed the byte sniff and fell through to the text read that would have refused it.
    /// `execute_command` spills output past 8 MiB to a capture file and tells the model the whole
    /// thing is "still reachable with `read_file`" -- and the shell docs promise the same. A
    /// residency ceiling applied to the file's *size* rather than to what a read *keeps* broke that
    /// promise for exactly the files it was written about: a runaway build log could be captured
    /// and then read by nothing.
    #[tokio::test]
    async fn a_capture_past_the_ceiling_is_still_readable_a_window_at_a_time() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let capture = temp_dir.path().join("capture.log");

        // Comfortably past the ceiling, with identifiable lines at a known offset.
        let filler = "x".repeat(1023);
        let mut body = String::with_capacity(MAX_READ_FILE_BYTES + 4096);
        let mut line = 0usize;
        while body.len() <= MAX_READ_FILE_BYTES + 2048 {
            body.push_str(&format!("line {} {}\n", line, filler));
            line += 1;
        }
        std::fs::write(&capture, &body).expect("write");

        let tool = ReadFileTool {
            read_tracker: test_tracker(),
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": capture.to_str().expect("path"),
                    "offset": 3,
                    "limit": 2,
                }),
                CancellationToken::new(),
            )
            .await
            .expect("a windowed read of an oversized capture must succeed");

        assert!(!result.is_error, "{}", text_content(&result));
        let shown = text_content(&result);
        assert!(
            shown.contains("line 3 ") && shown.contains("line 4 "),
            "the requested window must be returned: {}",
            shown.chars().take(120).collect::<String>()
        );
        assert!(
            !shown.contains("line 5 "),
            "and only the requested window: {}",
            shown.chars().take(120).collect::<String>()
        );
        assert!(
            shown.contains(&format!("of {}", line)),
            "with the file's real line count disclosed: {}",
            shown
        );
    }

    /// The other half of the same boundary: a read that asks for the whole of an oversized file
    /// still has to refuse, because there is no bounded way to return it.
    #[tokio::test]
    async fn an_unwindowed_read_of_an_oversized_file_still_refuses_and_says_how_to_proceed() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let huge = temp_dir.path().join("huge.log");
        std::fs::write(&huge, vec![b'x'; MAX_READ_FILE_BYTES + 1]).expect("write");

        let tool = ReadFileTool {
            read_tracker: test_tracker(),
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let error = tool
            .execute(
                serde_json::json!({"path": huge.to_str().expect("path")}),
                CancellationToken::new(),
            )
            .await
            .expect_err("a file too large to return whole must be refused");

        let message = error.to_string();
        assert!(
            message.contains("offset") && message.contains("limit"),
            "the refusal must name the route that does work: {}",
            message
        );
    }

    #[tokio::test]
    async fn read_file_bytes_refuses_a_file_past_the_ceiling() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("huge.tga");
        std::fs::write(&path, vec![0u8; MAX_READ_FILE_BYTES + 1]).expect("write");

        let error = read_file_bytes(&path)
            .await
            .expect_err("a file past the ceiling must be refused, not read whole");
        assert!(error.to_string().contains("ceiling"), "{error}");

        // And the boundary itself is still readable.
        let ok = dir.path().join("big.tga");
        std::fs::write(&ok, vec![0u8; MAX_READ_FILE_BYTES]).expect("write");
        assert_eq!(
            read_file_bytes(&ok).await.expect("at the ceiling").len(),
            MAX_READ_FILE_BYTES
        );
    }

    /// A target that exists but cannot be re-read is refused, not overwritten blind.
    ///
    /// `local_old_text` mapped every read failure except `NotFound` to `None`, and the staleness
    /// guard runs only on `Some` -- so a file holding invalid UTF-8, or one past the 16 MiB
    /// ceiling, skipped the check entirely. `edit_file` refuses when it cannot verify; this is
    /// the same posture, and `force` remains the way through.
    #[tokio::test]
    async fn write_file_refuses_a_target_it_cannot_re_read() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("binary.dat");
        std::fs::write(&path, [0xff, 0xfe, 0xff]).expect("write invalid utf-8");

        let tool = WriteFileTool {
            read_tracker: test_tracker(),
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let refused = tool
            .execute(
                serde_json::json!({
                    "path": path.to_str().expect("path"),
                    "content": "replacement\n",
                }),
                CancellationToken::new(),
            )
            .await
            .expect("the tool reports, it does not error");
        assert!(refused.is_error, "an unverifiable target must be refused");
        assert_eq!(
            std::fs::read(&path).expect("read back"),
            [0xff, 0xfe, 0xff],
            "the original bytes must survive the refusal"
        );

        // `force` is the documented way through, and still writes.
        let forced = tool
            .execute(
                serde_json::json!({
                    "path": path.to_str().expect("path"),
                    "content": "replacement\n",
                    "force": true,
                }),
                CancellationToken::new(),
            )
            .await
            .expect("forced write");
        assert!(!forced.is_error, "force must still write");
        assert_eq!(
            std::fs::read_to_string(&path).expect("read back"),
            "replacement\n"
        );
    }

    /// Carrying the mode across the rename fixes the *published* file, but the temp file it is
    /// renamed from sits in the target's own directory holding the entire plaintext, and is born
    /// at `0o666 & ~umask`. Applying the mode only after the write therefore leaves a 0600 secret
    /// world-readable on disk for the length of the write plus an fsync -- a window nothing else
    /// in this suite looks at, because every other assertion is about the file that survives.
    /// Watch the directory during the write instead: no temp file may hold bytes at a wider mode
    /// than the target it will replace.
    ///
    /// Under a umask tight enough that the temp file is born narrow (0077, say) there is no
    /// window to catch and this passes on any implementation, which is correct rather than
    /// vacuous: the bug is unreachable there.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_rewritten_secret_is_never_briefly_world_readable() {
        use std::{
            os::unix::fs::PermissionsExt,
            sync::atomic::{AtomicBool, Ordering},
        };

        let temp_dir = tempfile::tempdir().expect("tempdir");
        let secret = temp_dir.path().join("secret.txt");
        std::fs::write(&secret, "old\n").expect("write");
        std::fs::set_permissions(&secret, std::fs::Permissions::from_mode(0o600)).expect("chmod");

        // Large enough that the write is still running when the watcher first looks.
        let payload = vec![b'x'; 16 * 1024 * 1024];

        let done = Arc::new(AtomicBool::new(false));
        let watcher = tokio::task::spawn_blocking({
            let done = done.clone();
            let watch_dir = temp_dir.path().to_path_buf();
            move || {
                let mut saw_temp = false;
                let mut widest_holding_content = 0u32;
                while !done.load(Ordering::SeqCst) {
                    let Ok(entries) = std::fs::read_dir(&watch_dir) else {
                        continue;
                    };
                    for entry in entries.flatten() {
                        if !entry.file_name().to_string_lossy().contains(".meka-tmp-") {
                            continue;
                        }
                        saw_temp = true;
                        // A zero-length temp file discloses nothing, and there is always an
                        // instant between creating it and narrowing it. Only bytes matter.
                        if let Ok(metadata) = entry.metadata()
                            && metadata.len() > 0
                        {
                            widest_holding_content |= metadata.permissions().mode() & 0o777;
                        }
                    }
                }
                (saw_temp, widest_holding_content)
            }
        });

        write_file_bytes(&secret, &payload).await.expect("write");
        done.store(true, Ordering::SeqCst);
        let (saw_temp, widest_holding_content) = watcher.await.expect("watcher");

        assert!(
            saw_temp,
            "the watcher never caught the temp file, so it proved nothing; raise the payload size"
        );
        assert!(
            widest_holding_content & 0o077 == 0,
            "the temp file held the plaintext at mode {:o}, readable beyond the owner of the 0600 \
             file it replaces",
            widest_holding_content
        );
    }

    /// A write must not loosen a file's permissions. `rename(2)` replaces the inode, so the mode
    /// has to be carried across deliberately; before the atomic-write change it survived for free
    /// because the target was opened in place.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_write_preserves_the_modes_of_the_file_it_replaces() {
        use std::os::unix::fs::PermissionsExt;

        let temp_dir = tempfile::tempdir().expect("tempdir");

        // A secret at 0600 must not come back world-readable.
        let secret = temp_dir.path().join("credentials");
        std::fs::write(&secret, "token = old\n").expect("write");
        std::fs::set_permissions(&secret, std::fs::Permissions::from_mode(0o600)).expect("chmod");
        write_file_bytes(
            &secret,
            b"token = new
",
        )
        .await
        .expect("write succeeds");
        assert_eq!(
            std::fs::metadata(&secret)
                .expect("stat")
                .permissions()
                .mode()
                & 0o777,
            0o600,
            "a 0600 secret must not become world-readable"
        );
        assert_eq!(
            std::fs::read_to_string(&secret).expect("read"),
            "token = new
"
        );

        // And an executable script must stay executable.
        let script = temp_dir.path().join("run.sh");
        std::fs::write(&script, "#!/bin/sh\necho old\n").expect("write");
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        write_file_bytes(&script, b"#!/bin/sh\necho new\n")
            .await
            .expect("write succeeds");
        assert_eq!(
            std::fs::metadata(&script)
                .expect("stat")
                .permissions()
                .mode()
                & 0o777,
            0o755,
            "an executable script must stay executable"
        );
    }

    /// Creating a file has nothing to lose, so the guard must not demand a prior read.
    #[tokio::test]
    async fn writing_a_new_file_needs_no_prior_read() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("fresh.txt");

        let tool = WriteFileTool {
            read_tracker: test_tracker(),
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "content": "brand new",
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should return Ok");

        assert!(!result.is_error, "{}", text_content(&result));
        assert_eq!(
            std::fs::read_to_string(&file_path).expect("read"),
            "brand new"
        );
    }

    #[tokio::test]
    async fn test_edit_accepts_a_file_that_is_unchanged_since_the_read() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("quiet.txt");
        std::fs::write(&file_path, "hello world").expect("write");

        let tracker = test_tracker();
        mark_read(&tracker, &file_path).await;

        let tool = EditFileTool {
            read_tracker: tracker,
            cwd: crate::workspace::test_cwd(),
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
            .expect("should return Ok");

        assert!(!result.is_error, "{}", text_content(&result));
        assert_eq!(
            std::fs::read_to_string(&file_path).expect("read"),
            "hello rust"
        );
    }

    #[tokio::test]
    async fn test_force_bypasses_the_changed_on_disk_check_too() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("forced.txt");
        std::fs::write(&file_path, "hello world").expect("write");

        let tracker = test_tracker();
        mark_read(&tracker, &file_path).await;
        std::fs::write(&file_path, "hello world, changed").expect("rewrite");

        let tool = EditFileTool {
            read_tracker: tracker,
            cwd: crate::workspace::test_cwd(),
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
            .expect("should return Ok");

        assert!(!result.is_error, "{}", text_content(&result));
    }

    /// Editing twice in a row is the common case. Without re-stamping after a write, the second
    /// edit would report the agent's own change as somebody else's.
    #[tokio::test]
    async fn test_consecutive_edits_do_not_report_a_change() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("twice.txt");
        std::fs::write(&file_path, "alpha beta gamma").expect("write");

        let tracker = test_tracker();
        mark_read(&tracker, &file_path).await;
        let tool = EditFileTool {
            read_tracker: tracker,
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };

        let edit = |old: &'static str, new: &'static str| {
            let path = file_path.clone();
            let tool = &tool;
            async move {
                tool.execute(
                    serde_json::json!({
                        "path": path.to_str().expect("path"),
                        "old_string": old,
                        "new_string": new
                    }),
                    CancellationToken::new(),
                )
                .await
                .expect("should return Ok")
            }
        };

        // Length-changing on purpose: an equal-length replacement would leave the stamps equal on a
        // filesystem with coarse mtime resolution, and the test would pass whether or not the
        // re-stamp exists.
        let first = edit("alpha", "ALPHA_EXPANDED").await;
        assert!(!first.is_error, "{}", text_content(&first));
        let second = edit("gamma", "GAMMA").await;
        assert!(!second.is_error, "{}", text_content(&second));
        assert_eq!(
            std::fs::read_to_string(&file_path).expect("read"),
            "ALPHA_EXPANDED beta GAMMA"
        );
    }

    /// `write_file` records the file it just produced, so an immediately following `edit_file`
    /// neither demands a read nor reports a change.
    #[tokio::test]
    async fn test_write_then_edit_needs_no_read() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("fresh.txt");
        let tracker = test_tracker();

        let writer = WriteFileTool {
            read_tracker: tracker.clone(),
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        writer
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "content": "hello world"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("write succeeds");

        let editor = EditFileTool {
            read_tracker: tracker,
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = editor
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "world",
                    "new_string": "rust"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should return Ok");

        assert!(!result.is_error, "{}", text_content(&result));
    }

    #[tokio::test]
    async fn test_edit_with_force_bypasses_read_check() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "hello world").expect("failed to write");

        let tool = EditFileTool {
            read_tracker: test_tracker(),
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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

    /// `read_file` reads a file whole, so it needs a ceiling of its own.
    ///
    /// The cancellation race added alongside makes `read_file("/dev/zero")` *interruptible*; it
    /// does not make it *bounded*, and an unattended `serve` or ACP session has nobody to press
    /// stop. The cap turns "the process died" into a tool error the model can act on.
    #[tokio::test]
    async fn test_read_file_refuses_a_file_past_the_ceiling() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("huge.txt");
        // One byte over, so the boundary itself is what is being tested.
        std::fs::write(&file_path, vec![b'x'; MAX_READ_FILE_BYTES + 1]).expect("write");

        let tool = ReadFileTool {
            read_tracker: test_tracker(),
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let error = tool
            .execute(
                serde_json::json!({"path": file_path.to_str().expect("path")}),
                CancellationToken::new(),
            )
            .await
            .expect_err("a file past the ceiling must be refused");
        let text = error.to_string();
        assert!(
            text.contains("ceiling") && text.contains("execute_command"),
            "the refusal must name a way forward: {text}",
        );

        // And a file at the ceiling still reads, so the bound is not off by one.
        std::fs::write(&file_path, vec![b'x'; MAX_READ_FILE_BYTES]).expect("rewrite");
        let result = tool
            .execute(
                serde_json::json!({"path": file_path.to_str().expect("path")}),
                CancellationToken::new(),
            )
            .await
            .expect("a file exactly at the ceiling is readable");
        assert!(!result.is_error);
    }

    /// An offset past the end of the file returns nothing, and nothing is indistinguishable from
    /// an empty file -- a different fact, and the one the model goes on to act on.
    #[tokio::test]
    async fn test_read_file_past_the_end_says_so_rather_than_returning_nothing() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("short.txt");
        std::fs::write(&file_path, "one\ntwo\nthree\n").expect("write");

        let tool = ReadFileTool {
            read_tracker: test_tracker(),
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({"path": file_path.to_str().expect("path"), "offset": 100}),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        let text = text_content(&result);
        assert!(!result.is_error, "a past-the-end read is not an error");
        assert!(
            text.contains("past the end") && text.contains("3 lines"),
            "the notice must name the real length: {text}",
        );
    }

    /// The reported total has to be the real one.
    ///
    /// The old count resumed the scan by skipping `matches.len()` *lines* to account for
    /// `matches.len()` *matches*, so every hit past that index was counted twice. A file where
    /// every line matches cannot show it -- lines and matches coincide, so both forms agree -- and
    /// that is exactly the file the cap test above writes, which is why the bug survived a test
    /// named for the behaviour it broke. A sparse file separates them: 100 hits shown out of 100,
    /// reported as 150 under the old arithmetic.
    #[tokio::test]
    async fn test_read_file_regex_reports_the_real_total_on_a_sparse_file() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("sparse.txt");
        let mut body = String::new();
        for line in 0..200 {
            if line % 2 == 0 {
                body.push_str(&format!("match-{}\n", line));
            } else {
                body.push_str("quiet\n");
            }
        }
        std::fs::write(&file_path, &body).expect("write");

        let tool = ReadFileTool {
            read_tracker: test_tracker(),
            cwd: crate::workspace::test_cwd(),
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
            !text.contains("showing first"),
            "all 100 matches fit under the cap, so nothing was withheld; got: {}",
            text,
        );
        assert_eq!(
            text.lines().count(),
            100,
            "every match on an even line, and nothing else",
        );
    }

    #[tokio::test]
    async fn test_edit_file_insert_before() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "anchor line\n").expect("write");

        let tool = EditFileTool {
            read_tracker: test_tracker(),
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
            cwd: crate::workspace::test_cwd(),
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
