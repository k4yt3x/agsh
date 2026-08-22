//! Handlers for the `meka memory <subcommand>` CLI: list, get, show, add, edit, remove, export.
//! Mirrors [`crate::skills::cli`]: parseable data goes to stdout (the user ran the command to get
//! it), lifecycle and diagnostics go through `tracing`.

use std::path::{Path, PathBuf};

use crate::{
    error::{MekaError, Result},
    memory::{self, MemoryStore, store::WriteRequest},
};

const DESCRIPTION_TRUNCATE: usize = 50;

/// Argument bag for [`run_add`], borrowed so callers don't clone every field out of the
/// clap-derived `cli::MemoryAction::Add` variant.
pub struct AddArgs<'a> {
    pub name: &'a str,
    pub description: &'a str,
    pub priority: Option<u8>,
    pub tags: &'a [String],
    pub body: Option<&'a str>,
    pub from_file: Option<&'a Path>,
    pub force: bool,
}

/// Whether a listing ends with the priority histogram.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ListDetail {
    /// `meka memory list`: the table plus the distribution.
    WithDistribution,
    /// `/memory`: the table alone. A mid-session glance is asking "what do I have saved", and a
    /// histogram of a handful of entries is noise around the answer.
    TableOnly,
}

/// A table of every memory in index order, optionally followed by the priority distribution.
///
/// The distribution is half the point of `meka memory list`. The agent picks a priority at write
/// time and everything feels important then, so priorities drift downward (toward 0) over a
/// long-lived instance until the index stops ranking anything. Printing the histogram makes that
/// drift something you can see and rebalance rather than something you discover when the index
/// stops being useful. That is a deliberate inspection, though, not what `/memory` is for.
pub async fn run_list(store: &MemoryStore, detail: ListDetail) -> Result<()> {
    let index = store.index().await?;
    if index.is_empty() {
        eprintln!("No memories saved.");
        return Ok(());
    }

    let now = std::time::SystemTime::now();
    let rows: Vec<Vec<String>> = index
        .iter()
        .map(|entry| {
            vec![
                entry.name.clone(),
                entry.priority.to_string(),
                memory::render_age(entry.recorded_at, now),
                entry.tags.join(","),
                truncate(
                    &memory::render_description_for_model(&entry.description),
                    DESCRIPTION_TRUNCATE,
                ),
            ]
        })
        .collect();

    print!(
        "{}",
        crate::render::format_columns(
            &["Name", "Priority", "Recorded", "Tags", "Description"],
            &rows
        )
    );

    if detail == ListDetail::WithDistribution {
        println!();
        println!("{} memories. Priority distribution:", index.len());
        for priority in memory::MIN_PRIORITY..=memory::MAX_PRIORITY {
            let count = index
                .iter()
                .filter(|entry| entry.priority == priority)
                .count();
            if count > 0 {
                println!("  p{}: {}", priority, count);
            }
        }
    }

    Ok(())
}

/// `meka memory get <name>`: the stored fields as `key: value` lines.
pub async fn run_get(store: &MemoryStore, name: &str) -> Result<()> {
    let entry = require_memory(store, name).await?;
    let now = std::time::SystemTime::now();
    println!("name: {}", entry.name);
    println!(
        "description: {}",
        memory::render_description_for_model(&entry.description)
    );
    println!("priority: {}", entry.priority);
    // Two dates, because they answer different questions. "recorded" is when the note was made and
    // is stamped once; "updated" is when the row last changed, which a priority nudge moves without
    // the note saying anything new.
    println!("recorded: {}", memory::render_age(entry.recorded_at, now));
    println!("updated: {}", memory::render_age(entry.updated_at, now));
    println!("read count: {}", entry.read_count);
    if !entry.tags.is_empty() {
        println!("tags: {}", entry.tags.join(", "));
    }
    println!(
        "body: {} bytes",
        entry.body.as_deref().unwrap_or_default().len()
    );
    Ok(())
}

/// `meka memory show <name>`: print the body.
///
/// Sanitised, because this is a display path and a note may hold an escape sequence the terminal
/// would act on. `meka memory export` is the door that hands back exactly what is stored, and
/// `meka memory edit` is the one that round-trips it.
pub async fn run_show(store: &MemoryStore, name: &str) -> Result<()> {
    let entry = require_memory(store, name).await?;
    let body = memory::render_for_model(&entry.body.unwrap_or_default());
    print!("{}", body);
    if !body.ends_with('\n') {
        println!();
    }
    Ok(())
}

/// `meka memory add <name> --description <text> [flags]`: write a memory by hand.
pub async fn run_add(store: &MemoryStore, args: AddArgs<'_>) -> Result<()> {
    memory::validate_memory_name(args.name).map_err(MekaError::Config)?;
    if args.description.trim().is_empty() {
        return Err(MekaError::Config("description cannot be empty".to_string()));
    }
    if store.get(args.name).await?.is_some() && !args.force {
        return Err(MekaError::Config(format!(
            "memory '{}' already exists; pass --force to overwrite",
            args.name
        )));
    }

    let body = match (args.body, args.from_file) {
        (Some(_), Some(_)) => {
            return Err(MekaError::Config(
                "pass either --body or --from-file, not both".to_string(),
            ));
        }
        (Some(body), None) => Some(body.to_string()),
        (None, Some(file)) => Some(std::fs::read_to_string(file).map_err(|error| {
            MekaError::Config(format!("failed to read {}: {}", file.display(), error))
        })?),
        // Omitted entirely, which the upsert reads as "leave whatever is there". `--force` with no
        // `--body` is a metadata edit, and demoting the note's contents to nothing on a call that
        // never mentioned them is the failure the omit-to-keep rule exists to prevent.
        (None, None) => None,
    };

    // No range check here: `--priority` is `value_parser!(u8).range(0..=9)`, so clap refuses an
    // out-of-range value before this runs and re-checking would be unreachable code pretending to
    // be a guard.
    //
    // An empty `--tag` list means "none given", not "clear them", for the same reason `--body` is
    // omit-to-keep.
    let tags = if args.tags.is_empty() {
        None
    } else {
        Some(memory::normalize_tags(args.tags).map_err(MekaError::Config)?)
    };

    let written = store
        .write(WriteRequest {
            name: args.name.to_string(),
            // As `memory_write` and `PUT /v1/memory` do. A description is a one-line label at every
            // door, and `meka memory export` writes it normalised, so storing it raw here made the
            // CLI the one door whose descriptions changed on the way through a backup.
            description: memory::normalize_description(args.description),
            tags,
            body,
            priority: args.priority,
        })
        .await?;
    tracing::info!(
        "wrote memory '{}' (priority {})",
        written.name,
        written.priority
    );
    Ok(())
}

/// `meka memory edit <name>`: open the body in `$EDITOR` and write back what comes out.
///
/// The body only. Metadata already has a door -- `meka memory add <name> --force --description
/// ...`, which keeps what it does not mention -- and putting the frontmatter in front of an editor
/// would mean parsing YAML back out of it, which is the thing this store no longer does.
pub async fn run_edit(store: &MemoryStore, name: &str) -> Result<()> {
    // Validated before the name becomes a path, as every other door does. `run_export`'s comment
    // records that rows carrying names meka would not accept exist in the wild, and this function
    // joins the name onto a temp directory: `meka memory edit ../pwned` wrote the note's body to
    // `/tmp/pwned.md`, outside the directory the cleanup below removes, where it then stayed.
    memory::validate_memory_name(name).map_err(MekaError::Config)?;
    let entry = require_memory(store, name).await?;

    // `create_dir` rather than `create_dir_all`, on a name nothing else can be using: this fails
    // rather than reusing a directory somebody else owns, which in `/tmp` is the difference
    // between a scratch file and a symlink pointed at something of theirs.
    let directory = std::env::temp_dir().join(format!("meka-memory-edit-{}", uuid::Uuid::new_v4()));
    // 0700 straight from `mkdir(2)`, and the file below at 0600. `/tmp` is world-readable and a
    // memory body is somebody's private note; at the umask default both were readable by every
    // local user for as long as the editor stayed open.
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&directory)
            .map_err(|error| MekaError::Config(format!("failed to create a temp dir: {error}")))?;
    }
    #[cfg(not(unix))]
    std::fs::create_dir(&directory)
        .map_err(|error| MekaError::Config(format!("failed to create a temp dir: {error}")))?;
    // Named for the memory and suffixed `.md`, so the editor picks the right syntax mode and its
    // title bar says which note this is.
    let scratch = memory::memory_file_in(&directory, name);
    // Exactly what is stored, never the sanitised rendering. Whatever the editor does not touch
    // has to come back byte for byte, or editing one word silently strips every zero-width joiner
    // in the note -- which is what this door used to do.
    let original = entry.body.clone().unwrap_or_default();
    let edited = match edit_in(&scratch, &original) {
        Ok(edited) => edited,
        Err(error) => {
            return Err(unsaved_edit(
                &directory,
                &scratch,
                &original,
                reason_of(error),
            ));
        }
    };
    if edited == original {
        // The body is unchanged, which is not a failure -- but the *directory* may still hold work.
        // `:saveas other.md` then `:wq` writes the buffer beside the scratch file and leaves the
        // scratch file exactly as found, so this branch is reached with the user's writing sitting
        // right there. It discarded unconditionally, which deleted that writing and exited 0 with
        // nothing said. Measured through the real binary.
        match unstored_work(&directory, &scratch, &original) {
            Some(kept) => tracing::warn!(
                "memory '{}' is unchanged, but your editor left something else in {}; that file \
                 is not a memory and nothing has read it",
                name,
                kept.display()
            ),
            None => {
                discard_scratch(&directory);
                tracing::info!("memory '{}' unchanged", name);
            }
        }
        return Ok(());
    }

    // The body alone, through a door that names one column, and only if the stored text is still
    // the text the editor was handed. Going through `write` would send back the description read
    // *before* the editor opened, so a body edit reverted whatever the agent changed while the user
    // was typing; naming one column fixed that for every column except the one being written, and
    // the body stayed an unlocked read-modify-write across a window as long as the editing session.
    // Measured: the agent's write vanished under the human's, both reporting success.
    //
    // The scratch directory is removed *after* this, not before. Removing it first meant a refused
    // save destroyed the only copy of what the user had just typed: `edit_in` waits for the editor
    // to exit, so there is no buffer to recover from, and the refusal that was supposed to be the
    // safe answer lost ten minutes of someone's writing instead of one agent write.
    match store.write_body(&entry.name, &original, edited).await {
        Ok(memory::store::BodyWrite::Saved) => {
            discard_scratch(&directory);
            tracing::info!("updated the body of memory '{}'", name);
            Ok(())
        }
        Ok(memory::store::BodyWrite::Gone) => Err(unsaved_edit(
            &directory,
            &scratch,
            &original,
            format!("memory '{name}' was deleted while the editor was open"),
        )),
        // Refused rather than merged or overwritten: meka cannot know which version is right, so it
        // keeps both and says where each one is.
        Ok(memory::store::BodyWrite::ChangedUnderneath) => Err(unsaved_edit(
            &directory,
            &scratch,
            &original,
            format!(
                "memory '{name}' was rewritten while the editor was open, so saving would have \
                 discarded that. `meka memory show {name}` prints the current body"
            ),
        )),
        Err(error) => Err(unsaved_edit(
            &directory,
            &scratch,
            &original,
            reason_of(error),
        )),
    }
}

/// Remove the scratch directory, once what was in it is safely stored.
fn discard_scratch(directory: &Path) {
    if let Err(error) = std::fs::remove_dir_all(directory) {
        tracing::warn!("could not remove {}: {}", directory.display(), error);
    }
}

/// Report a failure that leaves edited text unsaved, naming where that text is.
///
/// The scratch directory is deliberately kept when it holds writing the store does not have. It was
/// removed on every failure, which was right for the privacy of a note nobody edited and wrong for
/// everything else: a save refused because the memory moved underneath destroyed the user's work
/// outright, and told them to "re-apply your change" to text that no longer existed anywhere.
/// Keeping a 0600 file in a 0700 directory is the same protection it had while the editor was open,
/// and naming it is what stops that from being a copy left in `/tmp` quietly.
///
/// The decision is made on the **directory's contents against `original`**, not on whether the
/// scratch file exists. Both simpler tests were wrong in opposite directions. `edit_in` writes the
/// scratch file *before* launching the editor, so "the file exists" is true even when `$EDITOR` was
/// a typo or the user quit with `:cq` -- which left a permanent plaintext copy of a private note in
/// `/tmp` and claimed an edit had been lost that was never made. And an editor told to `:saveas`
/// under another name leaves the user's hour of work in the directory with the scratch file gone,
/// which "the file exists" reads as nothing to keep and deletes.
fn unsaved_edit(directory: &Path, scratch: &Path, original: &str, reason: String) -> MekaError {
    let Some(kept) = unstored_work(directory, scratch, original) else {
        discard_scratch(directory);
        return MekaError::Config(reason);
    };
    MekaError::Config(format!(
        "{reason}. Your edit was not saved; it is in {}",
        kept.display()
    ))
}

/// What the scratch directory holds that the store does not, and where to point the user at it.
///
/// `None` means everything in there is already stored, so the directory can go. Used by both exits
/// from [`run_edit`] that do not write, because the question is the same at each and answering it
/// separately is how the unchanged-body branch came to delete a `:saveas` copy.
///
/// Three deliberate choices, each from a measured failure:
///
/// - **Anything that cannot be read counts as worth keeping**, including a directory this function
///   cannot enumerate. This is the last chance to preserve the text, and a wrong guess deletes
///   somebody's work. The enumeration failure used to fall through to "nothing to keep", which is
///   the one direction that cannot be undone.
/// - **Only regular files are read.** `read_to_string` on a FIFO blocks for ever -- `meka memory
///   edit` hung until it was killed -- and on a 2 GB file it reads the lot into memory, measured at
///   1.9 GB resident. Anything that is not a plain file is kept without being read.
/// - **The length is checked before the contents.** A file of a different length cannot equal the
///   stored body, which skips the read for every real edit.
fn unstored_work(directory: &Path, scratch: &Path, original: &str) -> Option<PathBuf> {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return Some(directory.to_path_buf());
    };
    let mut differing: Vec<PathBuf> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(metadata) = std::fs::symlink_metadata(&path) else {
            return Some(directory.to_path_buf());
        };
        if !metadata.is_file() || metadata.len() != original.len() as u64 {
            differing.push(path);
            continue;
        }
        // Unreadable falls through to "differs", which is the safe direction.
        if !std::fs::read_to_string(&path).is_ok_and(|text| text == original) {
            differing.push(path);
        }
    }
    match differing.as_slice() {
        [] => None,
        // The scratch file and nothing else: name the file. Anything else means the work is
        // somewhere the user chose, so name the directory -- naming the scratch file then sent them
        // to the one file that does *not* hold what they wrote.
        [only] if only == scratch => Some(scratch.to_path_buf()),
        _ => Some(directory.to_path_buf()),
    }
}

/// The message inside a `MekaError`, without the `configuration error:` prefix `Display` adds.
///
/// [`unsaved_edit`] composes a new error out of an existing one, and re-rendering it through
/// `to_string` produced `configuration error: configuration error: ...`.
fn reason_of(error: MekaError) -> String {
    match error {
        MekaError::Config(message) => message,
        other => other.to_string(),
    }
}

/// Write `original` to `scratch`, run the user's editor on it, and read back what came out.
///
/// Split out so [`run_edit`] decides what happens to the scratch directory, which depends on
/// whether the text in it made it to the store.
fn edit_in(scratch: &Path, original: &str) -> Result<String> {
    let mut command = crate::store::editor_command(scratch).ok_or_else(|| {
        MekaError::Config(
            "set $EDITOR or $VISUAL to edit a memory body, or use `meka memory add <name> --force \
             --body ...`"
                .to_string(),
        )
    })?;
    // 0600, and `create_new` so an existing path is never written through. See the directory mode
    // in `run_edit`: this is somebody's note landing in a world-readable `/tmp`.
    let mut options = std::fs::OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    {
        use std::io::Write as _;
        options
            .open(scratch)
            .and_then(|mut file| file.write_all(original.as_bytes()))
            .map_err(|error| MekaError::Config(format!("failed to write a temp file: {error}")))?;
    }
    let status = command
        .status()
        .map_err(|error| MekaError::Config(format!("failed to launch your editor: {error}")))?;
    if !status.success() {
        return Err(MekaError::Config(format!(
            "your editor exited with {status}; the memory is unchanged"
        )));
    }
    std::fs::read_to_string(scratch)
        .map_err(|error| MekaError::Config(format!("failed to read the edited body back: {error}")))
}

/// `meka memory remove <name>`: delete the memory.
pub async fn run_remove(store: &MemoryStore, name: &str) -> Result<()> {
    memory::validate_memory_name(name).map_err(MekaError::Config)?;
    if !store.delete(name).await? {
        return Err(MekaError::Config(format!("memory '{}' not found", name)));
    }
    tracing::info!("removed memory '{}'", name);
    Ok(())
}

/// `meka memory export --dir <path>`: one `<name>.md` per memory, frontmatter and body.
///
/// This is what replaced the store being a directory of files: `grep`, a git repository, a backup
/// that is not a database. The format is the one the file-backed store used, so
/// `import-memory-store.py` reads an export and a legacy store with one parser.
///
/// Refuses to write into a directory that already holds files, rather than merging into it. An
/// export is a snapshot, and silently leaving a stale `<name>.md` behind from a memory since
/// deleted would make it one that never quite matches the store.
pub async fn run_export(store: &MemoryStore, directory: &Path) -> Result<()> {
    let memories = store.export_all().await?;
    if memories.is_empty() {
        eprintln!("No memories saved; nothing to export.");
        return Ok(());
    }

    let mut missing_directory = false;
    match std::fs::read_dir(directory) {
        Ok(mut entries) => {
            if entries.next().is_some() {
                return Err(MekaError::Config(format!(
                    "{} is not empty; an export is a snapshot, so point --dir at a new or empty \
                     directory",
                    directory.display()
                )));
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => missing_directory = true,
        Err(error) => {
            return Err(MekaError::Config(format!(
                "failed to read {}: {}",
                directory.display(),
                error
            )));
        }
    }

    // Every memory checked before a single file is written, both for a name a filesystem cannot
    // take and for a description the file cannot carry. This is the one place a name becomes a
    // path, and a row can carry one meka would not have accepted -- an older importer was looser
    // than `validate_entry_name`, so a legacy `nul.md` or `-old.md` landed in the table. Validating
    // inside the loop half-wrote the export and then aborted, and the retry hit "directory is not
    // empty", which names neither the cause nor the remedy.
    let unusable: Vec<String> = memories
        .iter()
        .filter_map(|memory| {
            if let Err(reason) = memory::validate_memory_name(&memory.name) {
                return Some(format!("{} ({})", memory.name, reason));
            }
            // A description of nothing but control characters survives the write door -- they are
            // not whitespace, so `trim().is_empty()` says it is a description -- and leaves the
            // file with `description: ""`, which the importer skips as having none. That is a
            // memory lost through a backup, so the export refuses instead.
            if !memory::description_survives_export(&memory.description) {
                return Some(format!(
                    "{} (its description is made only of characters YAML cannot carry, so the \
                     file would have none and the importer would skip it)",
                    memory.name
                ));
            }
            None
        })
        .collect();
    if !unusable.is_empty() {
        return Err(MekaError::Config(format!(
            "{} memor{} cannot be written out, so nothing was exported: {}. Fix {} with `meka \
             memory add <name> --force --description ...`, then re-run.",
            unusable.len(),
            if unusable.len() == 1 { "y" } else { "ies" },
            unusable.join(", "),
            if unusable.len() == 1 { "it" } else { "them" },
        )));
    }

    // Created only now that every memory is known to be writable, so a refused export leaves no
    // directory behind either. It used to be created before the checks above, which meant the one
    // command that says "nothing was exported" still changed the filesystem.
    let created_directory = missing_directory;
    if created_directory {
        create_private_export_dir(directory)?;
    }

    // Through `write_file_atomic`, which is what the file-backed store wrote through before this
    // change, for the two properties `std::fs::write` does not have. It creates the file at 0600
    // inside a 0700 directory: an export is somebody's private notes, the database it came from is
    // 0600, and at the default umask this wrote the lot at 0644 for every local user to read. And
    // it `fsync`s before renaming into place, so "exported N memories" survives a power loss --
    // which for the one command whose whole job is to be a backup is not a detail.
    //
    // It also tightens an *existing* target directory to 0700. Deliberate, and documented in the
    // memory guide rather than left to be discovered: the directory has to be empty for the export
    // to start, so nothing else of the user's lives there, and the alternative is publishing the
    // contents of a 0600 database.
    let mut written: Vec<PathBuf> = Vec::with_capacity(memories.len());
    for memory in &memories {
        let path = memory::memory_file_in(directory, &memory.name);
        if let Err(error) = crate::config::write_file_atomic(&path, &memory::export_memory(memory))
        {
            // A truncated export is worse than none: it is a syntactically valid store that
            // restores a fraction of the memories and reports success, and the retry answers "is
            // not empty" rather than naming the cause. Measured at 3,719 of 20,001 files left
            // behind when the directory filled up.
            let remaining = remove_partial_export(&written, directory, created_directory);
            return Err(MekaError::Config(format!(
                "failed to write {}: {}. Nothing was exported{}.",
                path.display(),
                error,
                match (written.len(), remaining) {
                    (0, _) => String::new(),
                    (_, 0) => "; the files already written were removed".to_string(),
                    (total, remaining) => format!(
                        "; {remaining} of the {total} files already written could not be removed, \
                         so clear {} before retrying",
                        directory.display()
                    ),
                }
            )));
        }
        written.push(path);
    }
    tracing::info!(
        "exported {} memories to {}",
        memories.len(),
        directory.display()
    );
    Ok(())
}

/// Create the export directory born at 0700, rather than at whatever the umask allows.
///
/// `create_dir_all` followed by a `chmod` would leave the directory world-readable for the window
/// in between, which is the whole of the export for a fast enough reader.
///
/// **The leaf is created non-recursively**, so `AlreadyExists` still means something. Creating it
/// recursively made `create_dir_all` succeed on a directory another export had just made, so two
/// runs into the same new path both believed they owned it: they interleaved their files into one
/// directory holding a merge of two stores, and when either failed its cleanup unlinked the other's
/// files and reported "Nothing was exported" while the other reported success. Measured at 3 of 25
/// races. Parents are still created recursively -- the race that matters is over the leaf, which is
/// the directory this function's caller will later delete.
fn create_private_export_dir(directory: &Path) -> Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    if let Some(parent) = directory.parent()
        && !parent.as_os_str().is_empty()
        && !parent.exists()
    {
        let mut parents = std::fs::DirBuilder::new();
        parents.recursive(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            parents.mode(0o700);
        }
        parents.create(parent).map_err(|error| {
            MekaError::Config(format!("failed to create {}: {}", parent.display(), error))
        })?;
    }
    builder.create(directory).map_err(|error| {
        if error.kind() == std::io::ErrorKind::AlreadyExists {
            return MekaError::Config(format!(
                "{} exists after all; something created it while this export was starting, so \
                 point --dir somewhere else",
                directory.display()
            ));
        }
        MekaError::Config(format!(
            "failed to create {}: {}",
            directory.display(),
            error
        ))
    })
}

/// Undo a half-written export, returning how many files could *not* be removed.
///
/// Reported rather than assumed: telling the user "nothing was exported" while a partial snapshot
/// is still sitting there is the same class of lie the export refuses to commit in the first place.
fn remove_partial_export(written: &[PathBuf], directory: &Path, created_directory: bool) -> usize {
    let mut remaining = 0;
    for path in written {
        if let Err(error) = std::fs::remove_file(path) {
            tracing::warn!("could not remove {}: {}", path.display(), error);
            remaining += 1;
        }
    }
    // Only a directory this run created, and only once it is empty again: a directory the user
    // pointed at and pre-created is theirs, and `remove_dir` refusing a non-empty one is the guard
    // that keeps this from taking anything it did not put there.
    if created_directory
        && remaining == 0
        && let Err(error) = std::fs::remove_dir(directory)
    {
        tracing::debug!("could not remove {}: {}", directory.display(), error);
    }
    remaining
}

/// `meka memory verify [--rebuild]`: check the search index against the table, and repair it.
///
/// The index is derived and disposable, but until this existed nothing in meka could say whether
/// it was in step or put it back, and the docs handed the user a raw `sqlite3` incantation. A
/// desync is silent by nature -- searches simply stop finding things -- so the check has to be
/// reachable.
pub async fn run_verify(store: &MemoryStore, rebuild: bool) -> Result<()> {
    if rebuild {
        store.rebuild_index().await?;
        // After, not instead: a rebuild that still leaves the two disagreeing is worth hearing
        // about rather than reporting as a repair.
        store.integrity_check().await?;
        tracing::info!("rebuilt the memory search index");
        return Ok(());
    }
    match store.integrity_check().await {
        Ok(()) => {
            // Deliberately not "the index matches the store". FTS5 cannot tell you that for an
            // external-content table: a document changed under a trigger that stopped firing
            // leaves the structure sound and the counts equal. Claiming a guarantee the check
            // does not give is the failure this whole subsystem is written against.
            tracing::info!(
                "the memory search index is structurally sound and holds every stored memory; a \
                 document changed while its trigger was not firing is not detectable, so rebuild \
                 if search is missing something you know is there"
            );
            Ok(())
        }
        Err(error) => Err(MekaError::Config(format!(
            "{error}. Run `meka memory verify --rebuild` to regenerate it from the store; no \
             memory is lost either way, because the index is derived."
        ))),
    }
}

/// The directory `meka memory export` writes to when `--dir` is not given.
pub fn default_export_dir() -> PathBuf {
    PathBuf::from("meka-memory-export")
}

async fn require_memory(store: &MemoryStore, name: &str) -> Result<memory::Memory> {
    store
        .get(name)
        .await?
        .ok_or_else(|| MekaError::Config(format!("memory '{}' not found", name)))
}

fn truncate(text: &str, max: usize) -> String {
    let flattened = text.replace('\n', " ");
    if flattened.chars().count() <= max {
        return flattened;
    }
    let cut: String = flattened.chars().take(max.saturating_sub(1)).collect();
    format!("{}…", cut)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_leaves_short_text_alone() {
        assert_eq!(truncate("short", 10), "short");
    }

    #[test]
    fn test_truncate_flattens_newlines_and_elides() {
        assert_eq!(truncate("a\nb", 10), "a b");
        assert_eq!(truncate("abcdefghij", 5), "abcd…");
    }

    async fn store_with(entries: &[(&str, u8, &str, &str)]) -> std::sync::Arc<MemoryStore> {
        let store = MemoryStore::in_memory().await.expect("store");
        for (name, priority, description, body) in entries {
            store
                .write(WriteRequest {
                    name: name.to_string(),
                    description: description.to_string(),
                    tags: Some(vec!["infra".to_string()]),
                    body: Some(body.to_string()),
                    priority: Some(*priority),
                })
                .await
                .expect("write");
        }
        store
    }

    /// `add --force` without `--body` is a metadata edit. Reading the absent flag as "clear it"
    /// would make the CLI the one door that destroys a note's contents on a call about its
    /// description, which is exactly the defect the tool's omit-to-keep rule exists to prevent.
    #[tokio::test]
    async fn add_force_without_a_body_keeps_the_existing_one() {
        let store = store_with(&[("note", 5, "original", "contents worth keeping")]).await;
        run_add(&store, AddArgs {
            name: "note",
            description: "reworded",
            priority: None,
            tags: &[],
            body: None,
            from_file: None,
            force: true,
        })
        .await
        .expect("force add");

        let entry = store.get("note").await.expect("get").expect("present");
        assert_eq!(entry.description, "reworded");
        assert_eq!(entry.body.as_deref(), Some("contents worth keeping"));
        assert_eq!(entry.tags, ["infra"], "an unmentioned tag list is kept");
    }

    /// The CLI stores a one-line description, as the other two write doors do.
    ///
    /// `PUT /v1/memory` normalised and this did not, so a description written here kept its
    /// newlines -- and `meka memory export` normalises on the way out, which made the round trip
    /// change the stored text for the door most likely to be handed a multi-line shell string.
    #[tokio::test]
    async fn add_stores_a_one_line_description() {
        let store = MemoryStore::in_memory().await.expect("store");
        run_add(&store, AddArgs {
            name: "note",
            description: "first line\nsecond   line",
            priority: None,
            tags: &[],
            body: None,
            from_file: None,
            force: false,
        })
        .await
        .expect("add");
        assert_eq!(
            store
                .get("note")
                .await
                .expect("get")
                .expect("present")
                .description,
            "first line second line",
            "the description must be stored as the export would write it"
        );
    }

    /// Without `--force`, an existing name is refused rather than overwritten.
    #[tokio::test]
    async fn add_refuses_an_existing_name_without_force() {
        let store = store_with(&[("note", 5, "original", "body")]).await;
        let error = run_add(&store, AddArgs {
            name: "note",
            description: "different",
            priority: None,
            tags: &[],
            body: None,
            from_file: None,
            force: false,
        })
        .await
        .expect_err("must refuse");
        assert!(error.to_string().contains("--force"), "{error}");
        assert_eq!(
            store
                .get("note")
                .await
                .expect("get")
                .map(|entry| entry.description),
            Some("original".to_string())
        );
    }

    /// An edit must return everything the editor did not touch, byte for byte.
    ///
    /// The store used to sanitise on read, so `meka memory edit` read a stripped body, handed that
    /// to `$EDITOR`, and wrote the result back -- permanently destroying every format character in
    /// the note on an edit to one unrelated word. `meka memory show` then displayed the stripped
    /// text, so nothing revealed the loss. Measured: a Persian ZWNJ, the ZWJ holding an emoji
    /// sequence together, and a carriage return all vanished.
    #[tokio::test]
    async fn an_edit_does_not_strip_what_the_editor_left_alone() {
        let store = MemoryStore::in_memory().await.expect("store");
        let body = "\u{645}\u{6cc}\u{200c}\u{631}\u{648}\u{62f} and \u{1f469}\u{200d}\u{1f4bb} and a\rb keep";
        store
            .write(WriteRequest {
                name: "note".to_string(),
                description: "a note".to_string(),
                tags: None,
                body: Some(body.to_string()),
                priority: None,
            })
            .await
            .expect("write");

        // What `run_edit` hands to the editor is `entry.body` from this same read.
        let handed = require_memory(&store, "note")
            .await
            .expect("get")
            .body
            .expect("body");
        assert_eq!(
            handed, body,
            "the editor must be given the stored bytes, not a rendering"
        );
        for (name, needle) in [
            ("ZWNJ", "\u{200c}"),
            ("ZWJ", "\u{200d}"),
            ("carriage return", "\r"),
        ] {
            assert!(
                handed.contains(needle),
                "{name} was stripped before $EDITOR"
            );
        }

        // And writing back what the editor returned keeps them.
        store
            .write(WriteRequest {
                name: "note".to_string(),
                description: "a note".to_string(),
                tags: None,
                body: Some(handed.replace("keep", "kept")),
                priority: None,
            })
            .await
            .expect("rewrite");
        let after = store
            .get("note")
            .await
            .expect("get")
            .expect("present")
            .body
            .expect("body");
        assert!(after.contains("kept"), "the edit landed");
        assert!(
            after.contains("\u{200c}") && after.contains("\u{200d}") && after.contains('\r'),
            "an edit must not destroy characters it never touched: {after:?}"
        );
    }

    /// Points `$EDITOR` at `path` for the life of the value, and puts both editor variables back
    /// however the test ends.
    ///
    /// `$VISUAL` is cleared as well as set, because `crate::store::editor_command` prefers it: a
    /// developer with `VISUAL=nvim` exported ran these tests against their own editor, which failed
    /// the assertions or blocked on an interactive process. And the restore is in `Drop` rather
    /// than at the end of the test body, because `tokio::sync::Mutex` does not poison -- an
    /// assertion failure left `$EDITOR` pointing into a deleted temp directory for every test that
    /// ran afterwards.
    struct EditorEnv {
        editor: Option<std::ffi::OsString>,
        visual: Option<std::ffi::OsString>,
    }

    impl EditorEnv {
        fn pointing_at(path: &Path) -> Self {
            let saved = Self {
                editor: std::env::var_os("EDITOR"),
                visual: std::env::var_os("VISUAL"),
            };
            // SAFETY: both variables are process-global; `EDITOR_ENV_LOCK`, which every caller
            // holds for longer than this value lives, serialises the tests that touch them.
            unsafe {
                std::env::set_var("EDITOR", path);
                std::env::remove_var("VISUAL");
            }
            saved
        }
    }

    impl Drop for EditorEnv {
        fn drop(&mut self) {
            // SAFETY: as above.
            unsafe {
                match &self.editor {
                    Some(value) => std::env::set_var("EDITOR", value),
                    None => std::env::remove_var("EDITOR"),
                }
                match &self.visual {
                    Some(value) => std::env::set_var("VISUAL", value),
                    None => std::env::remove_var("VISUAL"),
                }
            }
        }
    }

    /// A refused save keeps what the user typed, and says where it is.
    ///
    /// The first test to drive `run_edit` itself. Until now nothing did -- the sibling above
    /// re-implements what it believes the function does -- and in that gap the scratch directory
    /// was removed *before* the write, so a save refused because the memory moved underneath
    /// destroyed the user's text and told them to "re-apply your change" to something that no
    /// longer existed. `edit_in` waits for the editor to exit, so there is no buffer to recover
    /// from either.
    // `multi_thread`, because `edit_in` blocks on `Command::status`. On the default single-threaded
    // runtime that call freezes the whole runtime, so the "concurrent" write below only landed
    // *after* the editor exited and the conflict this test exists to create never happened -- the
    // test passed a save it should have refused.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_edit_that_cannot_save_keeps_what_the_user_typed() {
        use std::os::unix::fs::PermissionsExt as _;

        let _guard = crate::store::EDITOR_ENV_LOCK.lock().await;
        let store = store_with(&[("race", 5, "d", "ORIGINAL")]).await;
        let temp = tempfile::tempdir().expect("tempdir");
        let editor = temp.path().join("editor.sh");
        // Appends, then dawdles, so the test can move the store underneath while it "edits".
        std::fs::write(
            &editor,
            "#!/bin/sh\nprintf ' plus the human' >> \"$1\"\nsleep 1\n",
        )
        .expect("write the editor");
        std::fs::set_permissions(&editor, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        let _editor_env = EditorEnv::pointing_at(&editor);

        let edit = tokio::spawn({
            let store = store.clone();
            async move { run_edit(&store, "race").await }
        });
        // The agent, while the editor is open.
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        store
            .write(WriteRequest {
                name: "race".to_string(),
                description: "d".to_string(),
                tags: None,
                body: Some("WHAT THE AGENT LEARNED".to_string()),
                priority: None,
            })
            .await
            .expect("the agent writes mid-edit");

        let error = edit
            .await
            .expect("join")
            .expect_err("saving over a moved body must be refused");
        let message = error.to_string();
        assert!(
            message.contains("was rewritten while the editor was open"),
            "{message}"
        );

        // The whole point: the text the user typed still exists, and the message says where.
        let kept = message
            .rsplit("it is in ")
            .next()
            .map(std::path::PathBuf::from)
            .expect("the message must name the file");
        assert!(kept.is_file(), "the user's text must survive: {message}");
        let text = std::fs::read_to_string(&kept).expect("read it back");
        assert_eq!(text, "ORIGINAL plus the human", "and be what they typed");
        assert_eq!(
            std::fs::metadata(&kept).expect("stat").permissions().mode() & 0o777,
            0o600,
            "kept as privately as it was written"
        );
        assert_eq!(
            store
                .get("race")
                .await
                .expect("get")
                .expect("row")
                .body
                .as_deref(),
            Some("WHAT THE AGENT LEARNED"),
            "and the write it would have discarded is still there"
        );
        if let Some(directory) = kept.parent() {
            std::fs::remove_dir_all(directory).expect("clean up the kept scratch");
        }
    }

    /// A successful edit leaves nothing behind, which is what makes keeping it on failure a signal.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_saved_edit_removes_its_scratch_directory() {
        use std::os::unix::fs::PermissionsExt as _;

        let _guard = crate::store::EDITOR_ENV_LOCK.lock().await;
        let store = store_with(&[("note", 5, "d", "ORIGINAL")]).await;
        let temp = tempfile::tempdir().expect("tempdir");
        let editor = temp.path().join("editor.sh");
        std::fs::write(&editor, "#!/bin/sh\nprintf ' edited' >> \"$1\"\n").expect("write");
        std::fs::set_permissions(&editor, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        let before = scratch_directory_count();
        let _editor_env = EditorEnv::pointing_at(&editor);

        run_edit(&store, "note").await.expect("the edit saves");
        assert_eq!(
            store
                .get("note")
                .await
                .expect("get")
                .expect("row")
                .body
                .as_deref(),
            Some("ORIGINAL edited")
        );
        assert_eq!(
            scratch_directory_count(),
            before,
            "a saved edit must not leave a memory body in the temp directory"
        );
    }

    /// How many `meka memory edit` scratch directories are sitting in the temp directory.
    fn scratch_directory_count() -> usize {
        let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
            return 0;
        };
        entries
            .flatten()
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("meka-memory-edit-")
            })
            .count()
    }

    /// An editor that changed nothing leaves no copy of the note in the temp directory.
    ///
    /// The other side of keeping a failed edit, and the one a naive "does the scratch file exist"
    /// test gets wrong: `edit_in` writes the file *before* launching the editor, so it exists even
    /// when `$EDITOR` was a typo or the user quit with `:cq`. Judging on existence left a permanent
    /// plaintext copy of a private note in `/tmp` on every such attempt, and claimed an edit had
    /// been lost that was never made.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_editor_that_changed_nothing_leaves_no_copy_behind() {
        let _guard = crate::store::EDITOR_ENV_LOCK.lock().await;
        let store = store_with(&[("private", 5, "d", "a private note")]).await;
        let before = scratch_directory_count();
        // Exits non-zero without touching the file, as `:cq` or a refused editor does.
        let _editor_env = EditorEnv::pointing_at(Path::new("/bin/false"));

        let error = run_edit(&store, "private")
            .await
            .expect_err("a failed editor is a failed edit");
        let message = error.to_string();
        assert!(
            !message.contains("it is in"),
            "nothing was typed, so there is nothing to point at: {message}"
        );
        assert_eq!(
            scratch_directory_count(),
            before,
            "and no copy of the note may be left in the temp directory"
        );
    }

    /// An editor that saved the work under a different name keeps the directory, and says so.
    ///
    /// `:saveas other.md` then `:q!` is an ordinary thing to do and leaves the scratch file gone
    /// with the user's writing beside it. Judging on the scratch file's existence read that as
    /// "nothing to keep" and deleted the lot.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_editor_that_renamed_the_file_keeps_what_it_left() {
        use std::os::unix::fs::PermissionsExt as _;

        let _guard = crate::store::EDITOR_ENV_LOCK.lock().await;
        let store = store_with(&[("note", 5, "d", "the stored body")]).await;
        let temp = tempfile::tempdir().expect("tempdir");
        let editor = temp.path().join("saveas.sh");
        std::fs::write(
            &editor,
            "#!/bin/sh\nprintf 'AN HOUR OF WRITING' > \"$(dirname \"$1\")/other.md\"\nrm -f \"$1\"\n",
        )
        .expect("write the editor");
        std::fs::set_permissions(&editor, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        let _editor_env = EditorEnv::pointing_at(&editor);

        let message = run_edit(&store, "note")
            .await
            .expect_err("reading the body back must fail")
            .to_string();
        let kept = message
            .rsplit("it is in ")
            .next()
            .map(std::path::PathBuf::from)
            .expect("the message must name where the work is");
        assert!(
            kept.is_dir(),
            "with the scratch file gone it must name the directory: {message}"
        );
        assert_eq!(
            std::fs::read_to_string(kept.join("other.md")).expect("the renamed file"),
            "AN HOUR OF WRITING",
            "and the user's work must still be there"
        );
        std::fs::remove_dir_all(&kept).expect("clean up");
    }

    /// An editor that saved elsewhere and left the body alone still keeps what it wrote.
    ///
    /// The `edited == original` branch, which is reached when the scratch file comes back
    /// byte-identical -- `:saveas other.md` then `:wq` is exactly that, since vim writes the buffer
    /// to the new name and never touches the old one. That branch discarded unconditionally while
    /// its sibling was being taught not to, so the fix covered one exit and not the other: exit 0,
    /// nothing printed, the user's writing deleted.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_editor_that_saved_elsewhere_without_touching_the_body_keeps_it() {
        use std::os::unix::fs::PermissionsExt as _;

        let _guard = crate::store::EDITOR_ENV_LOCK.lock().await;
        let store = store_with(&[("note", 5, "d", "the stored body")]).await;
        let temp = tempfile::tempdir().expect("tempdir");
        let editor = temp.path().join("saveas.sh");
        // Writes beside the scratch file and leaves it exactly as found, then exits 0.
        std::fs::write(
            &editor,
            "#!/bin/sh\nprintf 'AN HOUR OF WRITING' > \"$(dirname \"$1\")/other.md\"\n",
        )
        .expect("write the editor");
        std::fs::set_permissions(&editor, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        let _editor_env = EditorEnv::pointing_at(&editor);

        run_edit(&store, "note")
            .await
            .expect("the body is unchanged, which is not an error");

        let kept: Vec<_> = std::fs::read_dir(std::env::temp_dir())
            .expect("temp dir")
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with("meka-memory-edit-"))
            })
            .collect();
        let holding = kept
            .iter()
            .find(|directory| directory.join("other.md").is_file())
            .expect("the directory holding the user's writing must survive");
        assert_eq!(
            std::fs::read_to_string(holding.join("other.md")).expect("read"),
            "AN HOUR OF WRITING"
        );
        std::fs::remove_dir_all(holding).expect("clean up");
    }

    /// A name the store holds but a filesystem cannot take stops the export before it writes.
    ///
    /// An earlier importer was looser than `validate_entry_name`, so a legacy `nul.md` landed in
    /// the table; `meka memory export` then wrote some files, hit that row, and aborted -- and the
    /// retry answered "directory is not empty", naming neither the cause nor the remedy. Checked
    /// as a set, before anything is written.
    #[tokio::test]
    async fn an_unexportable_name_stops_the_export_before_it_writes_anything() {
        let store = MemoryStore::in_memory().await.expect("store");
        // Past the CLI door, as a hand-edited database or an older importer would.
        for name in ["aaa", "nul", "zzz"] {
            store
                .write(WriteRequest {
                    name: name.to_string(),
                    description: "a note".to_string(),
                    tags: None,
                    body: Some("body".to_string()),
                    priority: None,
                })
                .await
                .expect("write");
        }
        let temp = tempfile::tempdir().expect("tempdir");
        let directory = temp.path().join("export");
        let error = run_export(&store, &directory)
            .await
            .expect_err("must refuse");
        let message = error.to_string();
        assert!(
            message.contains("nul"),
            "it must name the offender: {message}"
        );
        assert!(
            message.contains("nothing was exported"),
            "and say the export did not happen: {message}"
        );
        assert!(
            std::fs::read_dir(&directory).is_err()
                || std::fs::read_dir(&directory)
                    .expect("read")
                    .next()
                    .is_none(),
            "no partial export may be left behind for the retry to trip over"
        );
    }

    /// An export is readable only by its owner, as the store it came from is.
    ///
    /// `std::fs::write` and `create_dir_all` take the umask, which at the default 022 published
    /// every memory body at 0644 inside a 0755 directory. The database is 0600, the file-backed
    /// store this replaced wrote 0600 files in a 0700 directory, and `run_edit` two functions above
    /// goes to the trouble of 0700/0600 for a scratch file that exists for the length of an editing
    /// session -- while the command whose whole job is to be a backup published the lot.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_export_is_readable_only_by_its_owner() {
        use std::os::unix::fs::PermissionsExt as _;

        let store = store_with(&[("alpha", 2, "first note", "private")]).await;
        let temp = tempfile::tempdir().expect("tempdir");
        let directory = temp.path().join("export");
        run_export(&store, &directory).await.expect("export");

        let directory_mode = std::fs::metadata(&directory)
            .expect("stat the directory")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(
            directory_mode, 0o700,
            "the export directory must be private"
        );
        let file_mode = std::fs::metadata(directory.join("alpha.md"))
            .expect("stat the file")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(file_mode, 0o600, "and so must every memory in it");
    }

    /// A description the file cannot carry stops the export, rather than writing a file the
    /// importer will skip.
    ///
    /// `require_str` only refuses a description that is blank after `trim`, and a control character
    /// is not whitespace -- so `\u{1}` is a description at every write door and nothing at all once
    /// `render_memory` has dropped what YAML cannot represent. The file then says
    /// `description: ""`, and the importer treats that as having none and skips the memory: a note
    /// lost through the one path that exists to preserve it.
    #[tokio::test]
    async fn a_description_the_file_cannot_carry_stops_the_export() {
        let store = MemoryStore::in_memory().await.expect("store");
        for (name, description) in [("fine", "an ordinary note"), ("blank", "\u{1}\u{2}")] {
            store
                .write(WriteRequest {
                    name: name.to_string(),
                    description: description.to_string(),
                    tags: None,
                    body: Some("body".to_string()),
                    priority: None,
                })
                .await
                .expect("write");
        }
        let temp = tempfile::tempdir().expect("tempdir");
        let directory = temp.path().join("export");
        let error = run_export(&store, &directory)
            .await
            .expect_err("must refuse");
        let message = error.to_string();
        assert!(
            message.contains("blank"),
            "it must name the offender: {message}"
        );
        assert!(
            message.contains("importer would skip it"),
            "and say what would go wrong: {message}"
        );
        assert!(
            std::fs::read_dir(&directory).is_err(),
            "and no file may be written, not even the exportable one"
        );
    }

    /// An export that cannot create its directory says so, and creates nothing.
    ///
    /// The reachable half of the failure path. A mid-loop write failure cannot be arranged
    /// deterministically -- the directory has to be empty for the export to start, so there is
    /// nothing to plant that would fail on the second file rather than the first -- which is why
    /// the cleanup itself is unit-tested in
    /// [`a_partial_export_is_removed_without_touching_what_it_did_not_write`] instead.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_export_that_cannot_create_its_directory_creates_nothing() {
        use std::os::unix::fs::PermissionsExt as _;

        let store = store_with(&[("alpha", 2, "first", "a"), ("beta", 2, "second", "b")]).await;
        let temp = tempfile::tempdir().expect("tempdir");
        let parent = temp.path().join("readonly");
        std::fs::create_dir(&parent).expect("create");
        std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o500))
            .expect("make read-only");

        // Running as root ignores the mode bits, so the failure this test needs cannot be
        // arranged. Probed rather than assumed, so the test never claims to have checked something
        // it did not.
        if std::fs::create_dir(parent.join(".probe")).is_ok() {
            std::fs::remove_dir(parent.join(".probe")).expect("remove the probe");
            std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o700))
                .expect("restore");
            return;
        }

        let directory = parent.join("export");
        let error = run_export(&store, &directory).await.expect_err("must fail");
        assert!(error.to_string().contains("failed to create"), "{error}");
        std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o700)).expect("restore");
        assert!(
            !directory.exists(),
            "a failed export must not leave a directory a restore could mistake for a snapshot"
        );
    }

    /// The cleanup removes what the export wrote and nothing else.
    ///
    /// Unit-tested directly because a mid-loop I/O failure cannot be arranged deterministically:
    /// the directory has to be empty for the export to start, so there is nothing to plant that
    /// would fail on the second file rather than the first.
    #[test]
    fn a_partial_export_is_removed_without_touching_what_it_did_not_write() {
        let temp = tempfile::tempdir().expect("tempdir");
        let directory = temp.path().join("export");
        std::fs::create_dir(&directory).expect("create");
        let ours = directory.join("alpha.md");
        let theirs = directory.join("not-ours.txt");
        std::fs::write(&ours, "written by this export").expect("write");
        std::fs::write(&theirs, "was here first").expect("write");

        // `created_directory` false: the user pointed at a directory that already existed, so it is
        // theirs to keep even once it is empty again.
        let remaining = remove_partial_export(std::slice::from_ref(&ours), &directory, false);
        assert_eq!(remaining, 0, "everything it wrote must come back out");
        assert!(!ours.exists(), "the export's own file is gone");
        assert!(theirs.exists(), "and a file it never wrote is untouched");
        assert!(directory.exists(), "as is a directory it did not create");

        // And one it did create is taken away with its contents.
        let owned = temp.path().join("owned");
        std::fs::create_dir(&owned).expect("create");
        let file = owned.join("beta.md");
        std::fs::write(&file, "x").expect("write");
        assert_eq!(remove_partial_export(&[file], &owned, true), 0);
        assert!(
            !owned.exists(),
            "a directory this run created is removed with it"
        );
    }

    /// The export is what replaced the store being a directory of files, so it has to round-trip:
    /// every memory becomes one parseable `<name>.md` carrying its metadata and body.
    #[tokio::test]
    async fn export_writes_one_parseable_file_per_memory() {
        let store = store_with(&[
            ("alpha", 2, "first note", "alpha body"),
            ("beta", 5, "second note", "beta body"),
        ])
        .await;
        let temp = tempfile::tempdir().expect("tempdir");
        let directory = temp.path().join("export");
        run_export(&store, &directory).await.expect("export");

        let alpha = std::fs::read_to_string(directory.join("alpha.md")).expect("alpha");
        let (frontmatter, body) =
            crate::store::split_frontmatter(&alpha).expect("must have frontmatter");
        let parsed: serde_norway::Value = serde_norway::from_str(frontmatter).expect("parses");
        assert_eq!(parsed["description"].as_str(), Some("first note"));
        assert_eq!(parsed["priority"].as_i64(), Some(2));
        assert_eq!(parsed["tags"][0].as_str(), Some("infra"));
        assert!(parsed["recorded"].as_str().is_some(), "{frontmatter}");
        assert_eq!(body.trim(), "alpha body");
        assert!(directory.join("beta.md").is_file());

        // A second export into the same directory is refused: a snapshot that merges would keep
        // a stale file for a memory since deleted, and never quite match the store.
        let error = run_export(&store, &directory)
            .await
            .expect_err("must refuse a non-empty directory");
        assert!(error.to_string().contains("not empty"), "{error}");
    }
}
