//! Handlers for the `meka memory <subcommand>` CLI: list, get, show, add, remove. Mirrors
//! [`crate::skills::cli`]: parseable data goes to stdout (the user ran the command to get it),
//! lifecycle and diagnostics go through `tracing`.

use std::path::Path;

use crate::{
    error::{MekaError, Result},
    memory,
};

const DESCRIPTION_TRUNCATE: usize = 50;

/// Argument bag for [`run_add`], borrowed so callers don't clone every field out of the
/// clap-derived `cli::MemoryAction::Add` variant.
pub struct AddArgs<'a> {
    pub name: &'a str,
    pub description: &'a str,
    pub priority: Option<u8>,
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
pub async fn run_list(detail: ListDetail) -> Result<()> {
    let memories = memory::discover_memories();
    if memories.is_empty() {
        println!("(no memories saved)");
        return Ok(());
    }

    let now = std::time::SystemTime::now();
    let rows: Vec<Vec<String>> = memories
        .iter()
        .map(|entry| {
            vec![
                entry.name.clone(),
                entry.priority.to_string(),
                memory::render_age(entry.mtime, now),
                truncate(&entry.description, DESCRIPTION_TRUNCATE),
            ]
        })
        .collect();

    print!(
        "{}",
        crate::render::format_columns(&["Name", "Priority", "Updated", "Description"], &rows)
    );

    if detail == ListDetail::WithDistribution {
        println!();
        println!("{} memories. Priority distribution:", memories.len());
        for priority in memory::MIN_PRIORITY..=memory::MAX_PRIORITY {
            let count = memories
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

/// `meka memory get <name>`: frontmatter and on-disk facts as `key: value` lines.
pub async fn run_get(name: &str) -> Result<()> {
    let entry = require_memory(name)?;
    let bytes = std::fs::metadata(&entry.path).map(|m| m.len()).unwrap_or(0);
    println!("name: {}", entry.name);
    println!("path: {}", entry.path.display());
    println!("description: {}", entry.description);
    println!("priority: {}", entry.priority);
    println!(
        "updated: {}",
        memory::render_age(entry.mtime, std::time::SystemTime::now())
    );
    println!("size: {} bytes", bytes);
    Ok(())
}

/// `meka memory show <name>`: print the body.
pub async fn run_show(name: &str) -> Result<()> {
    let entry = require_memory(name)?;
    let body = memory::load_memory_body(&entry)
        .await
        .map_err(|error| MekaError::Config(format!("failed to load memory body: {}", error)))?;
    print!("{}", body);
    Ok(())
}

/// `meka memory add <name> --description <text> [flags]`: write a memory by hand.
pub async fn run_add(args: AddArgs<'_>) -> Result<()> {
    memory::validate_memory_name(args.name).map_err(MekaError::Config)?;

    let root = memory::memory_dir()
        .ok_or_else(|| MekaError::Config("could not resolve meka config directory".to_string()))?;
    let path = memory::memory_file_in(&root, args.name);
    if path.exists() && !args.force {
        return Err(MekaError::Config(format!(
            "memory '{}' already exists at {}; pass --force to overwrite",
            args.name,
            path.display()
        )));
    }

    let body = match (args.body, args.from_file) {
        (Some(_), Some(_)) => {
            return Err(MekaError::Config(
                "pass either --body or --from-file, not both".to_string(),
            ));
        }
        (Some(body), None) => body.to_string(),
        (None, Some(file)) => std::fs::read_to_string(file).map_err(|error| {
            MekaError::Config(format!("failed to read {}: {}", file.display(), error))
        })?,
        (None, None) => String::new(),
    };

    let priority = args.priority.unwrap_or(memory::DEFAULT_PRIORITY);
    if priority > memory::MAX_PRIORITY {
        return Err(MekaError::Config(format!(
            "priority must be between {} and {}",
            memory::MIN_PRIORITY,
            memory::MAX_PRIORITY
        )));
    }

    let written = memory::write_memory(&root, args.name, args.description, priority, &body)
        .map_err(MekaError::Config)?;
    tracing::info!("wrote memory to {}", written.display());
    Ok(())
}

/// `meka memory remove <name>`: delete the file.
pub async fn run_remove(name: &str) -> Result<()> {
    let entry = require_memory(name)?;
    std::fs::remove_file(&entry.path).map_err(|error| {
        MekaError::Config(format!(
            "failed to remove {}: {}",
            entry.path.display(),
            error
        ))
    })?;
    tracing::info!("removed memory {}", entry.path.display());
    Ok(())
}

fn require_memory(name: &str) -> Result<memory::Memory> {
    memory::discover_memories()
        .into_iter()
        .find(|entry| entry.name == name)
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
}
