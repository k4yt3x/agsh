//! Interactive REPL: reedline-driven prompt loop, slash-command parsing, `!command` shell
//! pass-through, and the channels that exchange events between the REPL thread and the agent loop.

use std::{
    borrow::Cow,
    path::{Path, PathBuf},
    sync::Mutex,
};

use async_trait::async_trait;
use crossterm::style::{Color, Stylize};
use reedline::{
    ColumnarMenu, Completer, EditCommand, Emacs, ExternalPrinter, Highlighter, History, KeyCode,
    KeyModifiers, MenuBuilder, Prompt, PromptEditMode, PromptHistorySearch,
    PromptHistorySearchStatus, Reedline, ReedlineEvent, ReedlineMenu, Signal, Span, StyledText,
    Suggestion, default_emacs_keybindings,
};

use crate::{
    frontend::{Frontend, FrontendEvent, PermissionOutcome, PermissionRequest},
    permission::{EnabledPermissions, SharedPermission},
    relay::RELAY,
    render::{self, OutputSpacing, RenderMode, StreamingRenderer},
};

/// A top-level REPL slash command, used to drive both `print_help` and the Tab completer so the
/// command list lives in one place. The execution-side grammar (aliases, argument splitting,
/// `/mcp` and `/skill` subcommands) stays in `parse_slash_command`; this table only models the
/// names that are completed and documented.
struct CommandSpec {
    name: &'static str,
    /// Alternate spellings the parser also accepts. Honored by the highlighter but never offered
    /// as separate completions.
    aliases: &'static [&'static str],
    help: &'static str,
    /// Argument syntax shown after the name in help, empty for no-argument commands. A non-empty
    /// hint is the "takes an argument" predicate that drives completion's trailing space.
    arg_hint: &'static str,
}

const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        name: "help",
        aliases: &["?"],
        help: "Show this help message",
        arg_hint: "",
    },
    CommandSpec {
        name: "exit",
        aliases: &["quit"],
        help: "Exit the shell",
        arg_hint: "",
    },
    CommandSpec {
        name: "clear",
        aliases: &[],
        help: "Clear the terminal screen",
        arg_hint: "",
    },
    CommandSpec {
        name: "session",
        aliases: &[],
        help: "Show the current session ID",
        arg_hint: "",
    },
    CommandSpec {
        name: "permission",
        aliases: &[],
        help: "Show or set the permission level",
        arg_hint: "[none|read|ask|write]",
    },
    CommandSpec {
        name: "compact",
        aliases: &[],
        help: "Summarize and compact the session",
        arg_hint: "",
    },
    CommandSpec {
        name: "export",
        aliases: &[],
        help: "Export the current session as Markdown",
        arg_hint: "",
    },
    CommandSpec {
        name: "fork",
        aliases: &[],
        help: "Fork this session and continue in the copy",
        arg_hint: "",
    },
    CommandSpec {
        name: "cd",
        aliases: &[],
        help: "Change working directory",
        arg_hint: "<path>",
    },
    CommandSpec {
        name: "skill",
        aliases: &[],
        help: "List skills, or invoke one with extra context",
        arg_hint: "[name] [extra...]",
    },
    CommandSpec {
        name: "mcp",
        aliases: &[],
        help: "Manage MCP servers and prompts",
        arg_hint: "<subcommand>",
    },
    CommandSpec {
        name: "status",
        aliases: &[],
        help: "Show session stats (turns, tokens, cache, redactions)",
        arg_hint: "",
    },
    CommandSpec {
        name: "usage",
        aliases: &[],
        help: "Show account rate-limit usage (subscription providers)",
        arg_hint: "",
    },
    CommandSpec {
        name: "history",
        aliases: &[],
        help: "Reprint past conversation (bare = all, N = last N turns)",
        arg_hint: "[N]",
    },
];

/// Foreground applied to the leading token of a recognized slash command.
const KNOWN_COLOR: nu_ansi_term::Color = nu_ansi_term::Color::Green;
/// Foreground applied to the leading token when it starts with `/` but is not a known command.
const UNKNOWN_COLOR: nu_ansi_term::Color = nu_ansi_term::Color::Red;

/// Reedline highlighter for the input buffer. The leading `/command` token is recolored to signal
/// whether it is recognized; everything else keeps the base style. The final paint reedline emits
/// on submit is what lands in scrollback, so this is also what visually separates user prompts from
/// assistant output.
struct UserInputHighlighter {
    style: nu_ansi_term::Style,
}

impl Highlighter for UserInputHighlighter {
    fn highlight(&self, line: &str, _cursor: usize) -> StyledText {
        let mut text = StyledText::new();
        if let Some(after_slash) = line.strip_prefix('/') {
            let word_len = after_slash
                .find(char::is_whitespace)
                .unwrap_or(after_slash.len());
            let word = &after_slash[..word_len];
            let (token, remainder) = line.split_at(word_len + 1);
            let known = COMMANDS
                .iter()
                .any(|command| command.name == word || command.aliases.contains(&word));
            let token_color = if known { KNOWN_COLOR } else { UNKNOWN_COLOR };
            text.push((self.style.fg(token_color), token.to_string()));
            if !remainder.is_empty() {
                text.push((self.style, remainder.to_string()));
            }
        } else {
            text.push((self.style, line.to_string()));
        }
        text
    }
}

/// Tab completer for slash commands. The data needed to complete arguments (MCP server names,
/// skill names) is snapshotted at construction because reedline re-invokes `complete()` on every
/// keystroke while the menu is open, so a per-keystroke filesystem scan like `discover_skills`
/// (which reads every `SKILL.md`) must never live in the hot path.
struct SlashCompleter {
    mcp_server_names: Vec<String>,
    skill_names: Vec<String>,
    cwd: crate::agent::SharedCwd,
}

/// `/mcp` first-argument keywords, mirroring the grammar of `parse_mcp_slash`.
const MCP_SUBCOMMANDS: [&str; 4] = ["list", "reconnect", "login", "logout"];

/// Permission levels in canonical order, sourced through the `Display` impl so the completions
/// cannot drift from what the parser accepts.
const PERMISSION_LEVELS: [crate::permission::Permission; 4] = [
    crate::permission::Permission::None,
    crate::permission::Permission::Read,
    crate::permission::Permission::Ask,
    crate::permission::Permission::Write,
];

impl Completer for SlashCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<Suggestion> {
        let Some(after_slash) = line.strip_prefix('/') else {
            return Vec::new();
        };
        let before_cursor = line.get(..pos).unwrap_or(line);

        if !before_cursor.contains(char::is_whitespace) {
            // Cursor is still in the command word: complete command names. Aliases are
            // intentionally not prefix-matched, since offering both `/exit` and `/quit`
            // would just be noise.
            let typed = line.get(1..pos).unwrap_or("");
            return COMMANDS
                .iter()
                .filter(|command| command.name.starts_with(typed))
                .map(|command| Suggestion {
                    value: format!("/{}", command.name),
                    description: Some(command.help.to_string()),
                    append_whitespace: !command.arg_hint.is_empty(),
                    span: Span::new(0, pos),
                    ..Suggestion::default()
                })
                .collect();
        }

        let command = after_slash.split_whitespace().next().unwrap_or("");
        let token_start = before_cursor
            .char_indices()
            .rev()
            .find(|(_, character)| character.is_whitespace())
            .map_or(0, |(index, character)| index + character.len_utf8());
        let prefix = line.get(token_start..pos).unwrap_or("");
        // The command word is token 0, so the first argument is token 1.
        let argument_index = before_cursor
            .get(..token_start)
            .unwrap_or("")
            .split_whitespace()
            .count();

        match command {
            "permission" if argument_index == 1 => terminal_suggestions(
                PERMISSION_LEVELS.iter().map(|level| level.to_string()),
                prefix,
                token_start,
                pos,
            ),
            "skill" if argument_index == 1 => {
                terminal_suggestions(self.skill_names.iter().cloned(), prefix, token_start, pos)
            }
            "mcp" if argument_index == 1 => terminal_suggestions(
                MCP_SUBCOMMANDS.iter().map(|keyword| keyword.to_string()),
                prefix,
                token_start,
                pos,
            ),
            "mcp" if argument_index == 2 => {
                let subcommand = before_cursor
                    .get(..token_start)
                    .unwrap_or("")
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("");
                if matches!(subcommand, "reconnect" | "login" | "logout") {
                    terminal_suggestions(
                        self.mcp_server_names.iter().cloned(),
                        prefix,
                        token_start,
                        pos,
                    )
                } else {
                    Vec::new()
                }
            }
            "cd" => complete_cd_path(&self.cwd, prefix, token_start, pos),
            _ => Vec::new(),
        }
    }
}

/// Build suggestions for a terminal (single-token) argument, prefix-filtered. A trailing space is
/// appended so the user can move on once a value is chosen.
fn terminal_suggestions(
    candidates: impl IntoIterator<Item = String>,
    prefix: &str,
    token_start: usize,
    pos: usize,
) -> Vec<Suggestion> {
    candidates
        .into_iter()
        .filter(|candidate| candidate.starts_with(prefix))
        .map(|candidate| Suggestion {
            value: candidate,
            append_whitespace: true,
            span: Span::new(token_start, pos),
            ..Suggestion::default()
        })
        .collect()
}

/// Complete a `/cd` argument token to matching subdirectories. Only directories are offered (`/cd`
/// rejects files), and each value ends in `/` so Tab can keep drilling into nested directories.
fn complete_cd_path(
    cwd: &crate::agent::SharedCwd,
    token: &str,
    token_start: usize,
    pos: usize,
) -> Vec<Suggestion> {
    let (parent_portion, partial) = match token.rfind('/') {
        Some(index) => (&token[..=index], &token[index + 1..]),
        None => ("", token),
    };

    let scan_dir = if parent_portion.is_empty() {
        crate::agent::cwd_snapshot(cwd)
    } else {
        let Some(expanded) = expand_cd_target(parent_portion) else {
            return Vec::new();
        };
        crate::agent::resolve_against_cwd(cwd, expanded)
    };

    let entries = match std::fs::read_dir(&scan_dir) {
        Ok(entries) => entries,
        Err(_) => return Vec::new(),
    };

    let mut suggestions = Vec::new();
    for entry in entries.flatten() {
        if !entry.file_type().is_ok_and(|file_type| file_type.is_dir()) {
            continue;
        }
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        // Hide dotfiles unless the user has started typing a dot, mirroring shell completion.
        if name.starts_with('.') && !partial.starts_with('.') {
            continue;
        }
        if !name.starts_with(partial) {
            continue;
        }
        suggestions.push(Suggestion {
            value: format!("{parent_portion}{name}/"),
            append_whitespace: false,
            span: Span::new(token_start, pos),
            ..Suggestion::default()
        });
    }
    suggestions
}

const CYCLE_PERMISSION_SENTINEL: &str = "__cycle_permission__";
const COMPLETION_MENU: &str = "completion_menu";

struct MekaPrompt {
    shared_permission: SharedPermission,
    show_path: bool,
    /// Per-session working directory shared with the agent and the `/cd` slash command. Reading
    /// the lock per prompt render is cheap (microseconds) and bounded; `/cd` is the only
    /// writer.
    cwd: crate::agent::SharedCwd,
    /// Live context-window gauge, present only when `display.show_context_in_prompt` is set.
    context: Option<ContextIndicator>,
}

/// Shared handle to the live context-token counter plus the model window, for the optional prompt
/// gauge. The counter is the agent's `last_context_tokens` (updated after each turn / on compact).
struct ContextIndicator {
    tokens: std::sync::Arc<std::sync::atomic::AtomicU64>,
    window: u64,
}

impl ContextIndicator {
    /// Format as `used/window pct%`, or `None` before the first turn (no measurement yet) or when
    /// the window is unknown.
    fn render(&self) -> Option<String> {
        let tokens = self.tokens.load(std::sync::atomic::Ordering::Relaxed);
        if tokens == 0 || self.window == 0 {
            return None;
        }
        let pct = ((tokens as f64 / self.window as f64) * 100.0).round() as u64;
        Some(format!(
            "{}/{} {}%",
            crate::render::format_token_count(tokens),
            crate::render::format_token_count(self.window),
            pct
        ))
    }
}

impl Prompt for MekaPrompt {
    fn render_prompt_left(&self) -> Cow<'_, str> {
        let mut left = if self.show_path {
            let path = crate::agent::cwd_snapshot(&self.cwd);
            format!("meka {} ", shorten_path_with_tilde(&path))
        } else {
            "meka ".to_string()
        };
        if let Some(gauge) = self.context.as_ref().and_then(ContextIndicator::render) {
            left.push_str(&gauge);
            left.push(' ');
        }
        Cow::Owned(left)
    }

    fn render_prompt_right(&self) -> Cow<'_, str> {
        Cow::Borrowed("")
    }

    fn render_prompt_indicator(&self, _edit_mode: PromptEditMode) -> Cow<'_, str> {
        let permission = self.shared_permission.get();
        let colored_indicator =
            format!("[{}]", permission.indicator()).with(permission.indicator_color());
        Cow::Owned(format!("{} > ", colored_indicator))
    }

    fn get_prompt_color(&self) -> Color {
        Color::White
    }

    fn render_prompt_multiline_indicator(&self) -> Cow<'_, str> {
        Cow::Borrowed("::: ")
    }

    fn render_prompt_history_search_indicator(
        &self,
        history_search: PromptHistorySearch,
    ) -> Cow<'_, str> {
        let prefix = match history_search.status {
            PromptHistorySearchStatus::Passing => "",
            PromptHistorySearchStatus::Failing => "failing ",
        };
        Cow::Owned(format!(
            "({}reverse-i-search `{}')",
            prefix, history_search.term
        ))
    }

    fn get_indicator_color(&self) -> Color {
        Color::Reset
    }
}

fn build_reedline_editor(
    input_style: nu_ansi_term::Style,
    printer: ExternalPrinter<String>,
    history: Option<Box<dyn History>>,
    completer: SlashCompleter,
) -> Reedline {
    let mut keybindings = default_emacs_keybindings();

    keybindings.add_binding(
        KeyModifiers::SHIFT,
        KeyCode::BackTab,
        ReedlineEvent::ExecuteHostCommand(CYCLE_PERMISSION_SENTINEL.to_string()),
    );

    keybindings.add_binding(
        KeyModifiers::ALT,
        KeyCode::Enter,
        ReedlineEvent::Edit(vec![EditCommand::InsertNewline]),
    );

    keybindings.add_binding(
        KeyModifiers::SHIFT,
        KeyCode::Enter,
        ReedlineEvent::Edit(vec![EditCommand::InsertNewline]),
    );

    keybindings.add_binding(
        KeyModifiers::NONE,
        KeyCode::Tab,
        ReedlineEvent::UntilFound(vec![
            ReedlineEvent::Menu(COMPLETION_MENU.to_string()),
            ReedlineEvent::MenuNext,
        ]),
    );

    let emacs_mode = Emacs::new(keybindings);
    let mut editor = Reedline::create()
        .with_edit_mode(Box::new(emacs_mode))
        .with_highlighter(Box::new(UserInputHighlighter { style: input_style }))
        .with_completer(Box::new(completer))
        .with_menu(ReedlineMenu::EngineCompleter(Box::new(
            ColumnarMenu::default().with_name(COMPLETION_MENU),
        )))
        .use_bracketed_paste(true)
        .with_external_printer(printer);
    if let Some(history) = history {
        editor = editor.with_history(history);
    }
    editor
}

pub enum SlashCommand {
    Exit,
    Help,
    Clear,
    Session,
    Permission(Option<String>),
    Compact,
    Export,
    /// `/fork`: copy the current session and continue in the copy. The in-memory conversation is
    /// untouched, so the branch happens at the current head and the original freezes where it was.
    Fork,
    Cd(Option<String>),
    /// `/mcp <server>:<prompt> [args...]`: render an MCP prompt and send its messages as the next
    /// user turn.
    McpPrompt {
        server: String,
        prompt: String,
        args: Vec<String>,
    },
    /// `/mcp list`: display configured MCP servers.
    McpList,
    /// `/mcp reconnect <server>`: smoke-test connect for one server.
    McpReconnect {
        server: String,
    },
    /// `/mcp login <server>`: run the OAuth flow from the REPL.
    McpLogin {
        server: String,
    },
    /// `/mcp logout <server>`: clear stored credentials + revoke.
    McpLogout {
        server: String,
    },
    /// `/skill` (no argument): list installed skills.
    SkillList,
    /// `/skill <name> [extra...]`: invoke a user-invocable skill directly. Anything the user types
    /// after the skill name is captured verbatim in `extra` and prepended to the rendered skill
    /// body before the agent turn, so the model reads the user's directive first and the skill body
    /// as the method. Empty when the user just typed `/skill <name>`.
    SkillInvoke {
        name: String,
        extra: String,
    },
    /// `/status`: print cumulative session stats (turns, tokens, cache hit ratio, image
    /// redactions).
    Status,
    /// `/usage`: fetch and print the account's rate-limit usage from the active provider.
    Usage,
    /// `/history [N]`: reprint past conversation in REPL style. Bare `/history` dumps every
    /// materialised message; `/history N` shows the last `N` turns (turn = user prompt + the agent
    /// work it triggered). Any non-numeric argument (e.g. `all`) falls back to the dump-everything
    /// path.
    History(Option<usize>),
}

pub enum ReplEvent {
    UserInput(String),
    Command(SlashCommand),
    Exit,
}

/// Sent from the agent to the REPL when a tool call needs user approval in Ask mode.
pub struct ToolApprovalRequest {
    pub tool_name: String,
    /// Pre-computed summary (first required argument) to show next to the tool name in the
    /// approval prompt. Resolved agent-side because the REPL thread has no access to the tool
    /// registry needed for MCP schema lookups.
    pub primary_param: Option<String>,
    pub response_sender: tokio::sync::oneshot::Sender<bool>,
}

/// Messages sent from the agent to the REPL thread.
pub enum AgentToReplEvent {
    Done,
    ApprovalRequest(ToolApprovalRequest),
    /// Server-driven elicitation: the REPL prompts the user, then sends the response back via the
    /// embedded oneshot. `ReplFrontend::handle_elicitation` is the producer; the await on the
    /// matching receiver carries the response into the agent's task.
    McpElicitation {
        prompt: crate::mcp::elicitation::ElicitationPrompt,
        responder: tokio::sync::oneshot::Sender<crate::mcp::elicitation::ElicitationResponse>,
    },
    /// Incremental progress update for a running MCP tool.
    McpProgress(crate::mcp::progress::ProgressUpdate),
}

pub(crate) fn parse_slash_command(input: &str) -> Option<SlashCommand> {
    let input = input.strip_prefix('/')?;
    let mut parts = input.splitn(2, char::is_whitespace);
    let command = parts.next()?;
    let argument = parts.next().map(|s| s.trim().to_string());

    match command {
        "exit" | "quit" => Some(SlashCommand::Exit),
        "help" | "?" => Some(SlashCommand::Help),
        "clear" => Some(SlashCommand::Clear),
        "session" => Some(SlashCommand::Session),
        "permission" => Some(SlashCommand::Permission(argument)),
        "compact" => Some(SlashCommand::Compact),
        "export" => Some(SlashCommand::Export),
        "fork" => Some(SlashCommand::Fork),
        "cd" => Some(SlashCommand::Cd(argument)),
        "mcp" => parse_mcp_slash(argument.as_deref().unwrap_or("")),
        "skill" => Some(parse_skill_slash(argument.as_deref().unwrap_or(""))),
        "status" => Some(SlashCommand::Status),
        "usage" => Some(SlashCommand::Usage),
        "history" => Some(SlashCommand::History(
            argument
                .as_deref()
                .and_then(|s| s.trim().parse::<usize>().ok()),
        )),
        _ => None,
    }
}

/// Parse the argument to `/skill …`.
///
/// - Empty argument (bare `/skill`) → list installed skills. There is no `list` keyword: that token
///   would be treated as a skill name to invoke.
/// - Otherwise: first whitespace-separated token is the skill name; the remainder (if any) is
///   free-form extra context that gets prepended to the skill body before the agent turn. The
///   remainder is trimmed so trailing whitespace doesn't bloat the body.
fn parse_skill_slash(rest: &str) -> SlashCommand {
    let rest = rest.trim();
    if rest.is_empty() {
        return SlashCommand::SkillList;
    }
    let (name, extra) = match rest.split_once(char::is_whitespace) {
        Some((name, extra)) => (name.to_string(), extra.trim().to_string()),
        None => (rest.to_string(), String::new()),
    };
    SlashCommand::SkillInvoke { name, extra }
}

/// Parse the argument to `/mcp …`.
fn parse_mcp_slash(rest: &str) -> Option<SlashCommand> {
    let rest = rest.trim();
    if rest.is_empty() || rest == "list" {
        return Some(SlashCommand::McpList);
    }
    // `<subcommand> <server>` shapes. Reject bare `reconnect` / `login` / `logout` with no server
    // argument so users see the "Unknown command" error instead of silently firing against no
    // target.
    type McpServerCtor = fn(String) -> SlashCommand;
    fn mk_reconnect(s: String) -> SlashCommand {
        SlashCommand::McpReconnect { server: s }
    }
    fn mk_login(s: String) -> SlashCommand {
        SlashCommand::McpLogin { server: s }
    }
    fn mk_logout(s: String) -> SlashCommand {
        SlashCommand::McpLogout { server: s }
    }
    let subcommands: [(&str, McpServerCtor); 3] = [
        ("reconnect ", mk_reconnect),
        ("login ", mk_login),
        ("logout ", mk_logout),
    ];
    for (keyword, ctor) in subcommands {
        if let Some(server) = rest.strip_prefix(keyword) {
            let server = server.trim();
            if server.is_empty() {
                return None;
            }
            return Some(ctor(server.to_string()));
        }
    }
    // `<server>:<prompt> [args...]`: the first token is the prompt spec.
    let mut parts = rest.split_whitespace();
    let spec = parts.next()?;
    let (server, prompt) = spec.split_once(':')?;
    if server.is_empty() || prompt.is_empty() {
        return None;
    }
    let args = parts.map(str::to_string).collect();
    Some(SlashCommand::McpPrompt {
        server: server.to_string(),
        prompt: prompt.to_string(),
        args,
    })
}

fn format_enabled(enabled: EnabledPermissions) -> String {
    enabled
        .iter()
        .map(|mode| mode.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn print_help() {
    eprintln!("Commands:");
    for command in COMMANDS {
        let left = if command.arg_hint.is_empty() {
            format!("/{}", command.name)
        } else {
            format!("/{} {}", command.name, command.arg_hint)
        };
        eprintln!("  {left:<31}  {}", command.help);
        if command.name == "mcp" {
            // The /mcp subcommands are arguments, not top-level commands, so they are absent from
            // COMMANDS; list them here so help still documents the full grammar.
            eprintln!(
                "  {:<31}  Reconnect smoke-test for one server",
                "/mcp reconnect <server>"
            );
            eprintln!(
                "  {:<31}  Run the OAuth flow for a server",
                "/mcp login <server>"
            );
            eprintln!(
                "  {:<31}  Clear stored credentials for a server",
                "/mcp logout <server>"
            );
            eprintln!(
                "  {:<31}  Render an MCP prompt as the next turn",
                "/mcp <server>:<prompt> [args]"
            );
        }
    }
    eprintln!();
    eprintln!("Shortcuts:");
    eprintln!("  !<command>    Execute a shell command directly");
    eprintln!("  Shift+Tab     Cycle permission level");
    eprintln!("  Ctrl+D        Exit the shell");
}

#[allow(clippy::too_many_arguments)]
pub fn run_repl(
    shared_permission: SharedPermission,
    show_path_in_prompt: bool,
    context_indicator: Option<(std::sync::Arc<std::sync::atomic::AtomicU64>, u64)>,
    input_style: nu_ansi_term::Style,
    initial_turn_pending: bool,
    sandbox_state: crate::sandbox::SandboxState,
    input_sender: tokio::sync::mpsc::UnboundedSender<ReplEvent>,
    agent_event_receiver: std::sync::mpsc::Receiver<AgentToReplEvent>,
    cwd: crate::agent::SharedCwd,
    mcp_server_names: Vec<String>,
    history_db_path: Option<PathBuf>,
) {
    // Install reedline's `ExternalPrinter` on the process-global tracing writer BEFORE the first
    // `read_line()`. From this point on, log lines (including async MCP-connect warnings that fire
    // while the REPL is starting) print *above* the live prompt instead of being overwritten by
    // reedline's redraw.
    let printer = ExternalPrinter::default();
    RELAY.install(printer.clone());

    // Persistent, cross-session input history backed by the SQLite DB. On failure, degrade to
    // reedline's default in-memory history rather than taking down the REPL.
    const HISTORY_CAPACITY: usize = 5000;
    let history: Option<Box<dyn History>> = history_db_path.and_then(|path| {
        match crate::history::PromptHistory::open(&path, HISTORY_CAPACITY) {
            Ok(history) => Some(Box::new(history) as Box<dyn History>),
            Err(error) => {
                tracing::warn!("failed to open input history database: {}", error);
                None
            }
        }
    });

    // Snapshot skill names once. `discover_skills` reads every `SKILL.md`, so it must not run per
    // keystroke inside the completer. A skill added mid-session will not autocomplete until
    // restart, but `/skill` execution rediscovers live, so a stale snapshot never yields an
    // invalid command.
    let skill_names: Vec<String> = crate::skills::discover_skills()
        .into_iter()
        .map(|skill| skill.name)
        .collect();
    let completer = SlashCompleter {
        mcp_server_names,
        skill_names,
        cwd: cwd.clone(),
    };

    let mut editor = build_reedline_editor(input_style, printer, history, completer);
    let prompt = MekaPrompt {
        shared_permission: shared_permission.clone(),
        show_path: show_path_in_prompt,
        cwd: cwd.clone(),
        context: context_indicator.map(|(tokens, window)| ContextIndicator { tokens, window }),
    };

    // If the caller queued a synthetic first turn (e.g. `--skill` or a bare positional `[PROMPT]`
    // in interactive mode), drain agent events for that turn before drawing the first reedline
    // prompt. Otherwise the prompt indicator and the agent's stdout output collide on screen.
    if initial_turn_pending && !wait_for_agent(&agent_event_receiver) {
        return;
    }

    loop {
        // reedline drains the relay's `ExternalPrinter` only inside `read_line()`. Flag that window
        // so log lines route through the printer (cleanly above the live prompt) while it's active
        // and go straight to stderr otherwise (e.g. during a turn), surfacing immediately instead
        // of buffering until the turn ends and the next prompt is drawn.
        RELAY.set_at_prompt(true);
        let signal = editor.read_line(&prompt);
        RELAY.set_at_prompt(false);
        match signal {
            Ok(Signal::Success(buffer)) => {
                if buffer == CYCLE_PERMISSION_SENTINEL {
                    let new_permission = shared_permission.cycle();
                    tracing::debug!("permission cycled to {}", new_permission);
                    // Re-emit the "backend unavailable" warn at the moment the user enters read
                    // mode, so a misconfigured sandbox surfaces immediately instead of waiting for
                    // the first `execute_command` failure. The "stronger sandbox available" nudge
                    // (Warn 2) intentionally doesn't fire here: startup-only, to avoid nagging.
                    if new_permission == crate::permission::Permission::Read {
                        crate::sandbox::warn_if_sandbox_issues(
                            &sandbox_state,
                            crate::sandbox::WarnContext::ReadModeEntry,
                        );
                    }
                    continue;
                }

                let trimmed = buffer.trim();
                if trimmed.is_empty() {
                    continue;
                }

                if trimmed.starts_with('/') {
                    match parse_slash_command(trimmed) {
                        Some(SlashCommand::Exit) => {
                            if input_sender.send(ReplEvent::Exit).is_err() {
                                tracing::trace!("REPL event receiver already dropped");
                            }
                            break;
                        }
                        Some(SlashCommand::Help) => {
                            print_help();
                            continue;
                        }
                        Some(SlashCommand::Clear) => {
                            if crossterm::execute!(
                                std::io::stdout(),
                                crossterm::terminal::Clear(crossterm::terminal::ClearType::All),
                                crossterm::cursor::MoveTo(0, 0),
                            )
                            .is_err()
                            {
                                eprintln!("Failed to clear terminal");
                            }
                            continue;
                        }
                        Some(SlashCommand::Permission(argument)) => {
                            match argument {
                                None => {
                                    let current = shared_permission.get();
                                    eprintln!("Current permission level: {}", current);
                                }
                                Some(level) => {
                                    match level.parse::<crate::permission::Permission>() {
                                        Ok(permission) => {
                                            match shared_permission.try_set(permission) {
                                                Ok(()) => {
                                                    eprintln!(
                                                        "Permission level set to: {}",
                                                        permission
                                                    );
                                                }
                                                Err(_) => {
                                                    eprintln!(
                                                        "Error: '{}' is disabled in this config (enabled: {})",
                                                        permission,
                                                        format_enabled(shared_permission.enabled()),
                                                    );
                                                }
                                            }
                                        }
                                        Err(error) => {
                                            eprintln!("Error: {}", error);
                                        }
                                    }
                                }
                            }
                            continue;
                        }
                        Some(SlashCommand::Cd(argument)) => {
                            handle_cd(&cwd, argument.as_deref().unwrap_or(""));
                            continue;
                        }
                        Some(
                            command @ (SlashCommand::Session
                            | SlashCommand::Compact
                            | SlashCommand::Export
                            | SlashCommand::Fork
                            | SlashCommand::McpPrompt { .. }
                            | SlashCommand::McpList
                            | SlashCommand::McpReconnect { .. }
                            | SlashCommand::McpLogin { .. }
                            | SlashCommand::McpLogout { .. }
                            | SlashCommand::SkillList
                            | SlashCommand::SkillInvoke { .. }
                            | SlashCommand::Status
                            | SlashCommand::Usage
                            | SlashCommand::History(_)),
                        ) => {
                            if input_sender.send(ReplEvent::Command(command)).is_err() {
                                break;
                            }
                            if !wait_for_agent(&agent_event_receiver) {
                                break;
                            }
                            continue;
                        }
                        None => {
                            eprintln!(
                                "Unknown command: {}. Type /help for available commands.",
                                trimmed
                            );
                            continue;
                        }
                    }
                }

                if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
                    if input_sender.send(ReplEvent::Exit).is_err() {
                        tracing::trace!("REPL event receiver already dropped");
                    }
                    break;
                }

                if let Some(shell_command) = trimmed.strip_prefix('!') {
                    if shell_command.is_empty() {
                        continue;
                    }
                    // Run in the session's working directory so `!` commands track `/cd`. `/cd`
                    // updates the `SharedCwd` (not the process cwd), so without this `!pwd` would
                    // report the original launch directory.
                    let working_dir = crate::agent::cwd_snapshot(&cwd);
                    #[cfg(windows)]
                    let status = std::process::Command::new("powershell")
                        .arg("-Command")
                        .arg(shell_command)
                        .current_dir(&working_dir)
                        .status();

                    #[cfg(not(windows))]
                    let status = std::process::Command::new("sh")
                        .arg("-c")
                        .arg(shell_command)
                        .current_dir(&working_dir)
                        .status();
                    match status {
                        Ok(exit_status) => {
                            if !exit_status.success()
                                && let Some(code) = exit_status.code()
                            {
                                eprintln!("Command exited with status {}", code);
                            }
                        }
                        Err(error) => {
                            eprintln!("Failed to execute command: {}", error);
                        }
                    }
                    continue;
                }

                if input_sender
                    .send(ReplEvent::UserInput(trimmed.to_string()))
                    .is_err()
                {
                    break;
                }

                if !wait_for_agent(&agent_event_receiver) {
                    break;
                }
            }
            Ok(Signal::CtrlC) => {
                continue;
            }
            Ok(Signal::CtrlD) => {
                if input_sender.send(ReplEvent::Exit).is_err() {
                    tracing::trace!("REPL event receiver already dropped");
                }
                break;
            }
            // The pinned reedline fork (wtfbbqhax/reedline @ 3a457ff) has a slimmer `Signal` enum
            // than upstream, no `ExternalBreak` variant, so this catch-all is currently
            // unreachable. When we switch back to upstream after #1005 lands in a release, it'll
            // fire on the unhandled `ExternalBreak`.
            #[allow(unreachable_patterns)]
            Ok(other) => {
                tracing::warn!("unexpected reedline signal: {:?}", other);
                if input_sender.send(ReplEvent::Exit).is_err() {
                    tracing::trace!("REPL event receiver already dropped");
                }
                break;
            }
            Err(error) => {
                tracing::error!("readline error: {}", error);
                if input_sender.send(ReplEvent::Exit).is_err() {
                    tracing::trace!("REPL event receiver already dropped");
                }
                break;
            }
        }
    }
}

/// Wait for the agent to signal it is done, while also handling tool approval requests that arrive
/// in Ask mode.
fn wait_for_agent(agent_event_receiver: &std::sync::mpsc::Receiver<AgentToReplEvent>) -> bool {
    loop {
        match agent_event_receiver.recv() {
            Ok(AgentToReplEvent::Done) => return true,
            Ok(AgentToReplEvent::ApprovalRequest(request)) => {
                handle_approval_request(request);
            }
            Ok(AgentToReplEvent::McpElicitation { prompt, responder }) => {
                handle_elicitation_prompt(prompt, responder);
            }
            Ok(AgentToReplEvent::McpProgress(update)) => {
                render_progress_update(&update);
            }
            Err(_) => return false,
        }
    }
}

/// One-line status overwrite on stderr for a running MCP tool.
fn render_progress_update(update: &crate::mcp::progress::ProgressUpdate) {
    let line = format_progress_update(update);
    eprint!("{}", line);
    use std::io::Write;
    let _ = std::io::stderr().flush();
}

/// Format a progress line. Sanitises server-controlled strings so an MCP server can't inject ANSI
/// escapes to clear the screen or spoof prompts.
fn format_progress_update(update: &crate::mcp::progress::ProgressUpdate) -> String {
    let message = update
        .message
        .as_deref()
        .map(crate::mcp::sanitize::sanitize_text)
        .unwrap_or_default();
    let server = crate::mcp::sanitize::sanitize_text(&update.server_name);
    let tool = crate::mcp::sanitize::sanitize_text(&update.tool_name);
    let body = match update.total {
        Some(total) if total > 0.0 => format!(
            "\r[mcp:{}/{}] {:.0}/{:.0} {}",
            server, tool, update.progress, total, message
        ),
        _ => format!(
            "\r[mcp:{}/{}] {:.0} {}",
            server, tool, update.progress, message
        ),
    };
    // Pad with a few spaces so the next print clears trailing chars from any longer previous line.
    format!("{}     ", body)
}

/// Route a structured/url elicitation request to the user. For forms, walks the JSON Schema one
/// property at a time, collecting input. For URLs, opens the browser and waits for the user to
/// confirm. The response is sent back via the oneshot the agent's
/// `ReplFrontend::handle_elicitation` is awaiting.
fn handle_elicitation_prompt(
    prompt: crate::mcp::elicitation::ElicitationPrompt,
    responder: tokio::sync::oneshot::Sender<crate::mcp::elicitation::ElicitationResponse>,
) {
    use crate::mcp::{
        elicitation::{ElicitationKind, ElicitationResponse},
        sanitize::sanitize_text,
    };
    // Server-controlled strings get stripped of control/format codepoints before they reach the
    // terminal. Without this a malicious server could ship ANSI escapes to clear the screen or RTL
    // overrides to spoof the field the user thinks they're filling in.
    eprintln!(
        "[mcp elicit: {}] {}",
        sanitize_text(&prompt.server_name),
        sanitize_text(&prompt.message)
    );

    let response = match &prompt.kind {
        ElicitationKind::Url { url } => {
            eprint!(
                "Open {} in your browser? [Y/n/s=skip]: ",
                sanitize_text(url)
            );
            use std::io::Write;
            let _ = std::io::stderr().flush();
            let mut line = String::new();
            if std::io::stdin().read_line(&mut line).is_err() {
                ElicitationResponse::Decline
            } else {
                match line.trim().to_ascii_lowercase().as_str() {
                    "" | "y" | "yes" => {
                        if let Err(error) = open::that(url) {
                            // URL was printed right above; launch failure on headless hosts is
                            // expected noise, diagnostic only.
                            tracing::debug!(
                                "failed to open browser for URL elicitation: {}",
                                error
                            );
                        }
                        ElicitationResponse::Accept { content: None }
                    }
                    "s" | "skip" => ElicitationResponse::Cancel,
                    _ => ElicitationResponse::Decline,
                }
            }
        }
        ElicitationKind::Form { schema } => {
            let mut filled = serde_json::Map::new();
            if let Some(properties) = schema.get("properties").and_then(|v| v.as_object()) {
                for (field_name, field_schema) in properties {
                    let description = field_schema
                        .get("description")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let ty = field_schema
                        .get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("string");
                    let hint = if description.is_empty() {
                        ty
                    } else {
                        description
                    };
                    eprint!(
                        "  {} ({}): ",
                        sanitize_text(field_name),
                        sanitize_text(hint)
                    );
                    use std::io::Write;
                    let _ = std::io::stderr().flush();
                    let mut line = String::new();
                    if std::io::stdin().read_line(&mut line).is_err() {
                        break;
                    }
                    let value = line.trim().to_string();
                    if value.is_empty() {
                        continue;
                    }
                    let parsed = match ty {
                        "boolean" => match value.to_ascii_lowercase().as_str() {
                            "true" | "yes" | "y" => serde_json::Value::Bool(true),
                            "false" | "no" | "n" => serde_json::Value::Bool(false),
                            _ => serde_json::Value::String(value),
                        },
                        "integer" | "number" => value
                            .parse::<f64>()
                            .ok()
                            .and_then(serde_json::Number::from_f64)
                            .map(serde_json::Value::Number)
                            .unwrap_or(serde_json::Value::String(value)),
                        _ => serde_json::Value::String(value),
                    };
                    filled.insert(field_name.clone(), parsed);
                }
            }
            ElicitationResponse::Accept {
                content: Some(serde_json::Value::Object(filled)),
            }
        }
    };
    // Receiver-dropped means the agent's `handle_elicitation` future has been cancelled (turn
    // interrupt, session close, etc.). Nothing to recover; the agent already cleaned up.
    let _ = responder.send(response);
}

fn handle_approval_request(request: ToolApprovalRequest) {
    use crossterm::style::Stylize;

    let display_name = crate::render::tool_display_name_for_approval(&request.tool_name);
    let summary = request
        .primary_param
        .as_deref()
        .map(|s| s.replace('\n', " "))
        .unwrap_or_default();

    eprint!(
        "{} ",
        format!("[ask] {} {}", display_name, summary).with(crossterm::style::Color::Magenta)
    );
    eprint!("{}", "(Y/n) ".with(crossterm::style::Color::DarkGrey));

    if let Err(error) = std::io::Write::flush(&mut std::io::stderr()) {
        tracing::debug!("failed to flush stderr: {}", error);
    }

    let mut response = String::new();
    let allowed = match std::io::stdin().read_line(&mut response) {
        Ok(_) => {
            let trimmed = response.trim().to_lowercase();
            trimmed.is_empty() || trimmed == "y" || trimmed == "yes"
        }
        Err(_) => false,
    };

    if request.response_sender.send(allowed).is_err() {
        tracing::warn!("failed to send approval response (agent disconnected)");
    }
}

fn shorten_path_with_tilde(path: &Path) -> String {
    if let Some(home) = dirs::home_dir() {
        if path == home {
            return "~".to_string();
        }
        if let Ok(relative) = path.strip_prefix(&home) {
            // Normalize to forward slashes so the tilde form reads the same way on every platform
            // (Windows' native `\` looks jarring next to the `~/` prefix and breaks tests that
            // compare against a hard-coded literal).
            let relative_str = relative.display().to_string().replace('\\', "/");
            return format!("~/{}", relative_str);
        }
    }
    path.display().to_string()
}

/// Expand a `/cd` target's leading tilde to the home directory. Shared by `handle_cd` and the path
/// completer so both apply identical `~` / `~/` rules. Returns `None` only when a tilde needs the
/// home directory but it cannot be determined.
fn expand_cd_target(target: &str) -> Option<PathBuf> {
    if target.is_empty() || target == "~" {
        dirs::home_dir()
    } else if let Some(rest) = target.strip_prefix("~/") {
        dirs::home_dir().map(|home| home.join(rest))
    } else {
        Some(PathBuf::from(target))
    }
}

fn handle_cd(cwd: &crate::agent::SharedCwd, target: &str) {
    let raw = match expand_cd_target(target) {
        Some(raw) => raw,
        None => {
            eprintln!("cd: could not determine home directory");
            return;
        }
    };

    // Resolve relative inputs against the current per-session cwd so `/cd subdir` lands inside the
    // agent's current view, then canonicalize so the prompt and the tools see a clean path.
    let resolved = crate::agent::resolve_against_cwd(cwd, &raw);
    let canonical = match std::fs::canonicalize(&resolved) {
        Ok(canonical) => canonical,
        Err(error) => {
            eprintln!("cd: {}: {}", raw.display(), error);
            return;
        }
    };
    if !canonical.is_dir() {
        eprintln!("cd: {}: not a directory", canonical.display());
        return;
    }
    match cwd.write() {
        Ok(mut guard) => *guard = canonical,
        Err(poisoned) => *poisoned.into_inner() = canonical,
    }
}

/// Construction-time configuration for [`ReplFrontend`]. These fields used to live on
/// `AgentOptions`; they are UI concerns and now belong to the frontend impl.
pub struct ReplFrontendConfig {
    pub render_mode: RenderMode,
    pub newline_before_prompt: bool,
    pub newline_after_prompt: bool,
    pub show_session_id_on_create: bool,
    pub show_token_usage: bool,
    pub thinking_show_content: bool,
    /// Sender for the REPL's `AgentToReplEvent` channel, used to forward approval requests to the
    /// blocking REPL thread.
    pub agent_event_sender: std::sync::mpsc::Sender<AgentToReplEvent>,
}

/// REPL-side [`Frontend`] impl. Owns the [`StreamingRenderer`] and [`OutputSpacing`] state that
/// used to be threaded through `Agent::run_turn` / `run_streaming`, and forwards approval requests
/// over the existing mpsc to the blocking REPL thread.
///
/// Lives in `crate::repl` (alongside the REPL thread it talks to) rather than in `crate::frontend`,
/// so the trait module stays free of concrete UI types. See the module docs in `crate::frontend`.
pub struct ReplFrontend {
    config: ReplFrontendConfig,
    state: Mutex<ReplFrontendState>,
}

struct ReplFrontendState {
    spacing: OutputSpacing,
    /// Open across consecutive `AssistantTextDelta` events; closed by any non-text event (or
    /// `TurnFinished`).
    renderer: Option<StreamingRenderer>,
}

impl ReplFrontend {
    pub fn new(config: ReplFrontendConfig) -> Self {
        Self {
            config,
            state: Mutex::new(ReplFrontendState {
                spacing: OutputSpacing::new(),
                renderer: None,
            }),
        }
    }

    /// Flush and drop any open streaming renderer. Called before any non-text event so block types
    /// don't interleave on stderr.
    fn close_text_run(state: &mut ReplFrontendState) {
        if let Some(mut renderer) = state.renderer.take() {
            // Rendering errors here are typically a broken stderr pipe; log and move on rather than
            // panicking inside `emit`.
            if let Err(error) = renderer.finish() {
                tracing::debug!("frontend renderer finish failed: {}", error);
            }
        }
    }
}

#[async_trait]
impl Frontend for ReplFrontend {
    async fn emit(&self, event: FrontendEvent) {
        // Held briefly across synchronous render calls. The agent loop emits events serially per
        // turn, so contention is effectively zero; the lock is purely a `Send + Sync` discipline
        // check. `clippy::await_holding_lock` (deny-level, see Cargo.toml) enforces that no
        // `.await` appears between the lock acquisition and its drop.
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        match event {
            FrontendEvent::SessionStarted { id } => {
                if self.config.show_session_id_on_create {
                    render::render_session_id("Creating new session", &id.to_string());
                }
            }
            FrontendEvent::TurnStarted => {
                if self.config.newline_after_prompt {
                    eprintln!();
                    state.spacing.after_prompt();
                }
            }
            FrontendEvent::TurnFinished => {
                Self::close_text_run(&mut state);
                if self.config.newline_before_prompt {
                    eprintln!();
                }
            }
            FrontendEvent::AssistantTextDelta(text) => {
                if state.renderer.is_none() {
                    if state.spacing.before_text() {
                        eprintln!();
                    }
                    state.renderer = Some(StreamingRenderer::new(self.config.render_mode));
                }
                if let Some(renderer) = state.renderer.as_mut()
                    && let Err(error) = renderer.push_delta(&text)
                {
                    tracing::debug!("frontend renderer push_delta failed: {}", error);
                }
            }
            FrontendEvent::ThinkingBlock {
                content,
                signature: _,
            } => {
                Self::close_text_run(&mut state);
                if state.spacing.before_thinking() {
                    eprintln!();
                }
                render::render_thinking_block(&content, self.config.thinking_show_content);
            }
            FrontendEvent::ToolCallStarted {
                id: _,
                name,
                input,
                display_summary,
            } => {
                Self::close_text_run(&mut state);
                if state.spacing.before_tool_indicator() {
                    eprintln!();
                }
                render::render_tool_indicator(&name, &input, display_summary.as_deref());
            }
            // The REPL renders tool results inline through the agent's own message-history path
            // (the next assistant turn). No additional UI is needed at completion time; the
            // model's response that follows already summarizes what happened.
            FrontendEvent::ToolCallCompleted { .. } => {}
            FrontendEvent::TodoListUpdated { title, items } => {
                Self::close_text_run(&mut state);
                // Only advance spacing when the list actually rendered. An empty list prints
                // nothing; claiming a trailing blank would swallow the next text run's leading
                // blank after a tool indicator.
                if render::render_todo_list(title.as_deref(), &items) {
                    state.spacing.after_todo_list();
                }
            }
            FrontendEvent::TokenUsage(usage) => {
                Self::close_text_run(&mut state);
                if self.config.show_token_usage {
                    render::render_token_usage(&usage);
                }
            }
            FrontendEvent::Notice(notice) => {
                // Close any in-flight text run so the hint lands on its own line. Level is unused
                // by `render_hint` today (it always paints DarkGrey); future styling can branch
                // on `notice.level` when there's a need.
                Self::close_text_run(&mut state);
                render::render_hint(&notice.text);
            }
            // The REPL already prints the sub-agent's tool indicators as they happen, via the
            // parent's own renderer; a rolling rewrite of one tool call's content has no place in
            // a scrolling transcript.
            FrontendEvent::SubAgentActivity { .. } => {}
            FrontendEvent::McpProgress(update) => {
                // Forward through the existing REPL channel so the blocking REPL thread renders
                // the inline status line (carriage-return overwrite via `render_progress_update`).
                // If the REPL is gone the send is a no-op; we don't want to block the agent's
                // streaming loop on UI delivery.
                if self
                    .config
                    .agent_event_sender
                    .send(AgentToReplEvent::McpProgress(update))
                    .is_err()
                {
                    tracing::debug!("MCP progress dropped (REPL disconnected)");
                }
            }
        }
    }

    async fn request_permission(&self, request: PermissionRequest) -> PermissionOutcome {
        let (response_sender, response_receiver) = tokio::sync::oneshot::channel::<bool>();
        let approval = ToolApprovalRequest {
            tool_name: request.tool_name,
            primary_param: request.primary_param,
            response_sender,
        };
        if self
            .config
            .agent_event_sender
            .send(AgentToReplEvent::ApprovalRequest(approval))
            .is_err()
        {
            // REPL thread is gone; there is no human to ask. Treat as cancellation rather than
            // denial so the caller's ToolOutput message is honest about the cause.
            return PermissionOutcome::Cancelled;
        }
        match response_receiver.await {
            Ok(true) => PermissionOutcome::Allow,
            Ok(false) => PermissionOutcome::Deny,
            Err(_) => PermissionOutcome::Cancelled,
        }
    }

    async fn handle_elicitation(
        &self,
        prompt: crate::mcp::elicitation::ElicitationPrompt,
    ) -> crate::mcp::elicitation::ElicitationResponse {
        // Forward to the blocking REPL thread through the existing agent→shell channel. The thread
        // renders the prompt, collects user input, and pushes the response back via the oneshot
        // sender so this `.await` resolves.
        let (responder, receiver) =
            tokio::sync::oneshot::channel::<crate::mcp::elicitation::ElicitationResponse>();
        if self
            .config
            .agent_event_sender
            .send(AgentToReplEvent::McpElicitation { prompt, responder })
            .is_err()
        {
            // REPL thread is gone: no human to ask. Decline so the server learns the elicitation
            // wasn't answered. (Same posture as the agent-disconnected case in
            // `request_permission`.)
            tracing::debug!("MCP elicitation dropped (REPL disconnected); declining");
            return crate::mcp::elicitation::ElicitationResponse::Decline;
        }
        receiver
            .await
            .unwrap_or(crate::mcp::elicitation::ElicitationResponse::Decline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_cd_updates_shared_cwd_without_mutating_process_cwd() {
        // Working directory mutation is per-session now; verify `/cd` writes to the `SharedCwd` and
        // leaves `std::env::current_dir()` untouched. Use a tempdir + canonicalize so the assertion
        // is robust to platform-specific symlinks (e.g. `/tmp` → `/private/tmp` on macOS).
        let temp = tempfile::tempdir().expect("tempdir");
        let target = std::fs::canonicalize(temp.path()).expect("canonicalize tempdir");
        let process_cwd_before = std::env::current_dir().expect("read process cwd before /cd");

        let cwd: crate::agent::SharedCwd = std::sync::Arc::new(std::sync::RwLock::new(
            std::path::PathBuf::from("/this/path/does/not/exist"),
        ));
        handle_cd(&cwd, target.to_str().expect("utf-8 tempdir"));

        let stored = cwd.read().expect("cwd lock").clone();
        assert_eq!(stored, target, "shared cwd must point at the new directory");
        let process_cwd_after = std::env::current_dir().expect("read process cwd after /cd");
        assert_eq!(
            process_cwd_after, process_cwd_before,
            "process cwd must NOT be mutated by /cd",
        );
    }

    #[test]
    fn test_user_input_highlighter_default_preset_preserves_literal() {
        let highlighter = UserInputHighlighter {
            style: crate::config::default_input_style(),
        };
        let rendered = highlighter.highlight("hello world", 5).render_simple();
        assert!(
            rendered.contains("hello world"),
            "literal input must survive: {rendered:?}"
        );
        assert!(
            rendered.contains("\x1b[") && rendered.contains('m'),
            "at least one SGR escape must be emitted: {rendered:?}"
        );
    }

    #[test]
    fn test_user_input_highlighter_none_emits_no_escape() {
        let highlighter = UserInputHighlighter {
            style: nu_ansi_term::Style::default(),
        };
        let rendered = highlighter.highlight("hello", 0).render_simple();
        assert_eq!(rendered, "hello");
    }

    #[test]
    fn test_user_input_highlighter_known_command_distinct_from_unknown() {
        let highlighter = UserInputHighlighter {
            style: crate::config::default_input_style(),
        };
        let known = highlighter.highlight("/compact", 8).render_simple();
        let unknown = highlighter.highlight("/bogus", 6).render_simple();
        assert!(
            known.contains("/compact"),
            "known token survives: {known:?}"
        );
        assert!(
            unknown.contains("/bogus"),
            "unknown token survives: {unknown:?}"
        );
        assert_ne!(
            known, unknown,
            "known and unknown commands must render with different styles"
        );
    }

    #[test]
    fn test_user_input_highlighter_non_slash_single_style() {
        let highlighter = UserInputHighlighter {
            style: crate::config::default_input_style(),
        };
        let line = "hello world";
        let mut expected = StyledText::new();
        expected.push((highlighter.style, line.to_string()));
        assert_eq!(
            highlighter.highlight(line, 0).render_simple(),
            expected.render_simple()
        );
    }

    fn empty_completer() -> SlashCompleter {
        SlashCompleter {
            mcp_server_names: Vec::new(),
            skill_names: Vec::new(),
            cwd: crate::agent::test_cwd(),
        }
    }

    fn completer_at(cwd: crate::agent::SharedCwd) -> SlashCompleter {
        SlashCompleter {
            mcp_server_names: vec!["postgres".into(), "github".into()],
            skill_names: vec!["search".into(), "deep-research".into()],
            cwd,
        }
    }

    #[test]
    fn test_slash_completer_prefix_matches_expected() {
        let mut completer = empty_completer();
        let suggestions = completer.complete("/comp", 5);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].value, "/compact");
    }

    #[test]
    fn test_slash_completer_bare_slash_returns_all() {
        let mut completer = empty_completer();
        let suggestions = completer.complete("/", 1);
        assert_eq!(suggestions.len(), COMMANDS.len());
        assert!(suggestions.iter().all(|s| s.value.starts_with('/')));
    }

    #[test]
    fn test_slash_completer_non_slash_returns_empty() {
        let mut completer = empty_completer();
        assert!(completer.complete("hello", 5).is_empty());
        assert!(completer.complete("", 0).is_empty());
    }

    #[test]
    fn test_slash_completer_no_args_for_argless_commands() {
        let mut completer = empty_completer();
        // Commands without an argument completer return nothing once past the command word.
        assert!(completer.complete("/compact ", 9).is_empty());
        assert!(completer.complete("/status foo", 11).is_empty());
    }

    #[test]
    fn test_slash_completer_span_replaces_whole_token() {
        let mut completer = empty_completer();
        let suggestions = completer.complete("/comp", 5);
        assert_eq!(suggestions[0].span.start, 0);
        assert_eq!(suggestions[0].span.end, 5);
    }

    #[test]
    fn test_slash_completer_append_whitespace_tracks_arguments() {
        let mut completer = empty_completer();
        assert!(completer.complete("/permission", 11)[0].append_whitespace);
        assert!(completer.complete("/cd", 3)[0].append_whitespace);
        assert!(!completer.complete("/compact", 8)[0].append_whitespace);
        assert!(!completer.complete("/help", 5)[0].append_whitespace);
    }

    #[test]
    fn test_slash_completer_descriptions_present() {
        let mut completer = empty_completer();
        assert!(
            completer
                .complete("/", 1)
                .iter()
                .all(|s| s.description.as_deref().is_some_and(|d| !d.is_empty()))
        );
    }

    #[test]
    fn test_slash_completer_does_not_offer_aliases() {
        let mut completer = empty_completer();
        // `/q` matches the `quit` alias of `exit`, but aliases are never completed.
        assert!(completer.complete("/q", 2).is_empty());
    }

    #[test]
    fn test_slash_completer_permission_arg_prefix() {
        let mut completer = empty_completer();
        let one = completer.complete("/permission w", 13);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].value, "write");
        assert!(one[0].append_whitespace);
        let all: Vec<String> = completer
            .complete("/permission ", 12)
            .into_iter()
            .map(|suggestion| suggestion.value)
            .collect();
        assert_eq!(all, ["none", "read", "ask", "write"]);
    }

    #[test]
    fn test_slash_completer_permission_no_complete_second_arg() {
        let mut completer = empty_completer();
        assert!(completer.complete("/permission write extra", 23).is_empty());
    }

    #[test]
    fn test_slash_completer_skill_arg_prefix() {
        let mut completer = completer_at(crate::agent::test_cwd());
        let suggestions = completer.complete("/skill sea", 10);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].value, "search");
    }

    #[test]
    fn test_slash_completer_skill_no_complete_second_arg() {
        let mut completer = completer_at(crate::agent::test_cwd());
        assert!(completer.complete("/skill search foo", 17).is_empty());
    }

    #[test]
    fn test_slash_completer_mcp_arg1_keywords() {
        let mut completer = completer_at(crate::agent::test_cwd());
        let all: Vec<String> = completer
            .complete("/mcp ", 5)
            .into_iter()
            .map(|suggestion| suggestion.value)
            .collect();
        assert_eq!(all, ["list", "reconnect", "login", "logout"]);
        let rec = completer.complete("/mcp rec", 8);
        assert_eq!(rec.len(), 1);
        assert_eq!(rec[0].value, "reconnect");
    }

    #[test]
    fn test_slash_completer_mcp_arg2_server_after_subcommand() {
        let mut completer = completer_at(crate::agent::test_cwd());
        let servers: Vec<String> = completer
            .complete("/mcp reconnect ", 15)
            .into_iter()
            .map(|suggestion| suggestion.value)
            .collect();
        assert_eq!(servers, ["postgres", "github"]);
        assert_eq!(completer.complete("/mcp login git", 14)[0].value, "github");
        // `list` takes no server argument, so its second token completes nothing.
        assert!(completer.complete("/mcp list ", 10).is_empty());
    }

    #[test]
    fn test_slash_completer_cd_lists_directories() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = std::fs::canonicalize(temp.path()).expect("canonicalize");
        std::fs::create_dir(root.join("src")).expect("mkdir src");
        std::fs::create_dir(root.join("target")).expect("mkdir target");
        std::fs::create_dir(root.join(".git")).expect("mkdir .git");
        std::fs::write(root.join("README"), b"x").expect("write file");
        std::fs::create_dir_all(root.join("src/tools")).expect("mkdir src/tools");
        let cwd = std::sync::Arc::new(std::sync::RwLock::new(root));
        let mut completer = completer_at(cwd);

        let bare: Vec<String> = completer
            .complete("/cd ", 4)
            .into_iter()
            .map(|suggestion| suggestion.value)
            .collect();
        // Directories returned with a trailing slash; the file and dotdir are excluded.
        assert!(bare.contains(&"src/".to_string()));
        assert!(bare.contains(&"target/".to_string()));
        assert!(!bare.iter().any(|value| value.contains("README")));
        assert!(!bare.contains(&".git/".to_string()));

        // A leading dot in the partial opts dotdirs back in.
        let dot = completer.complete("/cd .gi", 7);
        assert_eq!(dot.len(), 1);
        assert_eq!(dot[0].value, ".git/");

        // Relative drill-down keeps the parent portion intact.
        let nested = completer.complete("/cd src/too", 11);
        assert_eq!(nested.len(), 1);
        assert_eq!(nested[0].value, "src/tools/");
        assert!(!nested[0].append_whitespace);
        assert_eq!(nested[0].span.start, 4);
        assert_eq!(nested[0].span.end, 11);
    }

    #[test]
    fn test_slash_completer_command_word_still_completes() {
        let mut completer = completer_at(crate::agent::test_cwd());
        let suggestions = completer.complete("/comp", 5);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].value, "/compact");
        assert_eq!(suggestions[0].span.start, 0);
        assert_eq!(suggestions[0].span.end, 5);
    }

    #[test]
    fn test_parse_slash_command_exit() {
        assert!(matches!(
            parse_slash_command("/exit"),
            Some(SlashCommand::Exit)
        ));
        assert!(matches!(
            parse_slash_command("/quit"),
            Some(SlashCommand::Exit)
        ));
    }

    #[test]
    fn test_parse_slash_command_help() {
        assert!(matches!(
            parse_slash_command("/help"),
            Some(SlashCommand::Help)
        ));
        assert!(matches!(
            parse_slash_command("/?"),
            Some(SlashCommand::Help)
        ));
    }

    #[test]
    fn test_parse_slash_command_clear() {
        assert!(matches!(
            parse_slash_command("/clear"),
            Some(SlashCommand::Clear)
        ));
    }

    #[test]
    fn test_parse_slash_command_session() {
        assert!(matches!(
            parse_slash_command("/session"),
            Some(SlashCommand::Session)
        ));
    }

    #[test]
    fn test_parse_slash_command_permission() {
        assert!(matches!(
            parse_slash_command("/permission"),
            Some(SlashCommand::Permission(None))
        ));
        match parse_slash_command("/permission write") {
            Some(SlashCommand::Permission(Some(arg))) => assert_eq!(arg, "write"),
            _ => panic!("expected Permission with argument"),
        }
    }

    #[test]
    fn test_parse_slash_command_compact() {
        assert!(matches!(
            parse_slash_command("/compact"),
            Some(SlashCommand::Compact)
        ));
    }

    #[test]
    fn test_parse_slash_command_unknown() {
        assert!(parse_slash_command("/unknown").is_none());
    }

    #[test]
    fn test_parse_slash_command_not_slash() {
        assert!(parse_slash_command("hello").is_none());
    }

    #[test]
    fn test_parse_slash_command_empty() {
        assert!(parse_slash_command("/").is_none());
    }

    #[test]
    fn test_parse_slash_command_cd_no_arg() {
        assert!(matches!(
            parse_slash_command("/cd"),
            Some(SlashCommand::Cd(None))
        ));
    }

    #[test]
    fn test_parse_slash_command_cd_with_path() {
        match parse_slash_command("/cd /tmp") {
            Some(SlashCommand::Cd(Some(arg))) => assert_eq!(arg, "/tmp"),
            _ => panic!("expected Cd with argument"),
        }
    }

    #[test]
    fn test_parse_slash_command_export() {
        assert!(matches!(
            parse_slash_command("/export"),
            Some(SlashCommand::Export)
        ));
    }

    #[test]
    fn test_parse_slash_command_fork() {
        assert!(matches!(
            parse_slash_command("/fork"),
            Some(SlashCommand::Fork)
        ));
    }

    #[test]
    fn test_parse_slash_command_history_no_args() {
        assert!(matches!(
            parse_slash_command("/history"),
            Some(SlashCommand::History(None))
        ));
    }

    #[test]
    fn test_parse_slash_command_history_with_n() {
        assert!(matches!(
            parse_slash_command("/history 5"),
            Some(SlashCommand::History(Some(5)))
        ));
        // Whitespace is tolerated.
        assert!(matches!(
            parse_slash_command("/history   12"),
            Some(SlashCommand::History(Some(12)))
        ));
    }

    #[test]
    fn test_parse_slash_command_history_garbage_falls_back_to_all() {
        // Non-numeric argument (including `all`) collapses to None so the
        // dispatch dumps the whole conversation. Documented behaviour:
        // graceful fallback, no error.
        assert!(matches!(
            parse_slash_command("/history all"),
            Some(SlashCommand::History(None))
        ));
        assert!(matches!(
            parse_slash_command("/history banana"),
            Some(SlashCommand::History(None))
        ));
    }

    #[test]
    fn test_shorten_path_with_tilde_home() {
        if let Some(home) = dirs::home_dir() {
            assert_eq!(shorten_path_with_tilde(&home), "~");
        }
    }

    #[test]
    fn test_shorten_path_with_tilde_subdir() {
        if let Some(home) = dirs::home_dir() {
            let subdir = home.join("projects").join("test");
            assert_eq!(shorten_path_with_tilde(&subdir), "~/projects/test");
        }
    }

    #[test]
    fn test_shorten_path_with_tilde_non_home() {
        let path = std::path::Path::new("/tmp/something");
        assert_eq!(shorten_path_with_tilde(path), "/tmp/something");
    }

    #[test]
    fn test_format_progress_update_strips_ansi_escapes() {
        let update = crate::mcp::progress::ProgressUpdate {
            server_name: "svr".to_string(),
            tool_name: "tool".to_string(),
            tool_use_id: None,
            message: Some("\x1b[2Jspoofed\x1b[H".to_string()),
            progress: 1.0,
            total: Some(4.0),
        };
        let line = format_progress_update(&update);
        assert!(
            !line.contains('\x1b'),
            "ANSI escape leaked into progress line: {:?}",
            line
        );
        assert!(line.contains("spoofed"));
        assert!(line.contains("[mcp:svr/tool]"));
    }

    #[test]
    fn test_parse_mcp_slash_empty_is_list() {
        assert!(matches!(
            parse_slash_command("/mcp"),
            Some(SlashCommand::McpList)
        ));
    }

    #[test]
    fn test_parse_mcp_slash_explicit_list() {
        assert!(matches!(
            parse_slash_command("/mcp list"),
            Some(SlashCommand::McpList)
        ));
    }

    #[test]
    fn test_parse_mcp_slash_reconnect_with_server() {
        match parse_slash_command("/mcp reconnect postgres") {
            Some(SlashCommand::McpReconnect { server }) => assert_eq!(server, "postgres"),
            other => panic!("expected McpReconnect, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn test_parse_mcp_slash_reconnect_without_server_is_none() {
        // Bare `reconnect` with no server name: neither the reconnect arm nor the
        // `<server>:<prompt>` arm matches, so the command is rejected rather than silently firing
        // against some default.
        assert!(parse_slash_command("/mcp reconnect").is_none());
    }

    #[test]
    fn test_parse_mcp_slash_login_with_server() {
        match parse_slash_command("/mcp login notion") {
            Some(SlashCommand::McpLogin { server }) => assert_eq!(server, "notion"),
            other => panic!("expected McpLogin, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn test_parse_mcp_slash_logout_with_server() {
        match parse_slash_command("/mcp logout notion") {
            Some(SlashCommand::McpLogout { server }) => assert_eq!(server, "notion"),
            other => panic!("expected McpLogout, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn test_parse_mcp_slash_login_without_server_is_none() {
        assert!(parse_slash_command("/mcp login").is_none());
    }

    #[test]
    fn test_parse_mcp_slash_logout_without_server_is_none() {
        assert!(parse_slash_command("/mcp logout").is_none());
    }

    #[test]
    fn test_parse_mcp_slash_login_trims_whitespace() {
        match parse_slash_command("/mcp login   notion  ") {
            Some(SlashCommand::McpLogin { server }) => assert_eq!(server, "notion"),
            other => panic!("expected McpLogin, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn test_parse_mcp_slash_prompt_no_args() {
        match parse_slash_command("/mcp postgres:schema") {
            Some(SlashCommand::McpPrompt {
                server,
                prompt,
                args,
            }) => {
                assert_eq!(server, "postgres");
                assert_eq!(prompt, "schema");
                assert!(args.is_empty());
            }
            other => panic!("expected McpPrompt, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn test_parse_mcp_slash_prompt_with_args() {
        match parse_slash_command("/mcp pg:query table=users limit=10") {
            Some(SlashCommand::McpPrompt {
                server,
                prompt,
                args,
            }) => {
                assert_eq!(server, "pg");
                assert_eq!(prompt, "query");
                assert_eq!(args, vec!["table=users", "limit=10"]);
            }
            other => panic!("expected McpPrompt, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn test_parse_mcp_slash_empty_server_rejected() {
        assert!(parse_slash_command("/mcp :prompt").is_none());
    }

    #[test]
    fn test_parse_mcp_slash_empty_prompt_rejected() {
        assert!(parse_slash_command("/mcp server:").is_none());
    }

    #[test]
    fn test_parse_mcp_slash_multiple_colons_splits_on_first() {
        // `split_once` returns the first colon, so prompt names can contain further colons.
        match parse_slash_command("/mcp srv:ns:prompt") {
            Some(SlashCommand::McpPrompt { server, prompt, .. }) => {
                assert_eq!(server, "srv");
                assert_eq!(prompt, "ns:prompt");
            }
            other => panic!("expected McpPrompt, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn test_parse_skill_slash_empty_is_list() {
        assert!(matches!(
            parse_slash_command("/skill"),
            Some(SlashCommand::SkillList)
        ));
        // Trailing whitespace is treated as no argument.
        assert!(matches!(
            parse_slash_command("/skill   "),
            Some(SlashCommand::SkillList)
        ));
    }

    #[test]
    fn test_parse_skill_slash_invokes_named_skill() {
        match parse_slash_command("/skill demo") {
            Some(SlashCommand::SkillInvoke { name, extra }) => {
                assert_eq!(name, "demo");
                assert!(extra.is_empty());
            }
            other => panic!("expected SkillInvoke, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn test_parse_skill_slash_captures_free_form_extra() {
        // The whole remainder after the skill name is captured verbatim (preserving inner
        // whitespace) and trimmed at the edges. This is free-form text the user wants prepended to
        // the skill body: no positional argument parsing.
        match parse_slash_command("/skill demo only fetch UK news") {
            Some(SlashCommand::SkillInvoke { name, extra }) => {
                assert_eq!(name, "demo");
                assert_eq!(extra, "only fetch UK news");
            }
            other => panic!("expected SkillInvoke, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn test_parse_skill_slash_trims_trailing_whitespace() {
        // Trailing whitespace after the skill name should produce an empty extra, not a
        // whitespace-padded one, equivalent to the bare-name invocation.
        match parse_slash_command("/skill demo   ") {
            Some(SlashCommand::SkillInvoke { name, extra }) => {
                assert_eq!(name, "demo");
                assert!(extra.is_empty());
            }
            other => panic!("expected SkillInvoke, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn test_parse_skill_slash_no_list_keyword() {
        // The token "list" is treated as a skill name, not a subcommand. (Bare `/skill` is the
        // listing form; `/skill list` would error at dispatch with "unknown skill 'list'" if no
        // such skill exists.)
        match parse_slash_command("/skill list") {
            Some(SlashCommand::SkillInvoke { name, extra }) => {
                assert_eq!(name, "list");
                assert!(extra.is_empty());
            }
            other => panic!("expected SkillInvoke, got {:?}", option_label(&other)),
        }
    }

    /// Short debug label: SlashCommand doesn't implement Debug so we map the few variants we care
    /// about manually to keep assertion messages readable.
    fn option_label(cmd: &Option<SlashCommand>) -> &'static str {
        match cmd {
            None => "None",
            Some(SlashCommand::Exit) => "Exit",
            Some(SlashCommand::Help) => "Help",
            Some(SlashCommand::Clear) => "Clear",
            Some(SlashCommand::Session) => "Session",
            Some(SlashCommand::Permission(_)) => "Permission",
            Some(SlashCommand::Compact) => "Compact",
            Some(SlashCommand::Export) => "Export",
            Some(SlashCommand::Fork) => "Fork",
            Some(SlashCommand::Cd(_)) => "Cd",
            Some(SlashCommand::McpList) => "McpList",
            Some(SlashCommand::McpReconnect { .. }) => "McpReconnect",
            Some(SlashCommand::McpLogin { .. }) => "McpLogin",
            Some(SlashCommand::McpLogout { .. }) => "McpLogout",
            Some(SlashCommand::McpPrompt { .. }) => "McpPrompt",
            Some(SlashCommand::SkillList) => "SkillList",
            Some(SlashCommand::SkillInvoke { .. }) => "SkillInvoke",
            Some(SlashCommand::Status) => "Status",
            Some(SlashCommand::Usage) => "Usage",
            Some(SlashCommand::History(_)) => "History",
        }
    }

    #[test]
    fn test_format_progress_update_strips_rtl_override_in_names() {
        // Defensive: even though server/tool names are normalised at registration time, this
        // confirms the renderer can't be tricked by a handler that someday forgets to normalise.
        let update = crate::mcp::progress::ProgressUpdate {
            server_name: "sv\u{202E}r".to_string(),
            tool_name: "t\u{200B}ool".to_string(),
            tool_use_id: None,
            message: None,
            progress: 0.5,
            total: None,
        };
        let line = format_progress_update(&update);
        assert!(!line.contains('\u{202E}'));
        assert!(!line.contains('\u{200B}'));
    }
}
