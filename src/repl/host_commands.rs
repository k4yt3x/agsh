//! Answering the slash commands the REPL thread cannot answer itself.
//!
//! `SlashCommand` is parsed on the REPL thread ([`crate::repl`]), and about half of its variants
//! need something only the host loop has: the live `Agent`, the conversation, the session id that
//! `/fork` moves. Those are forwarded and answered here.
//!
//! **One owner, checked by the compiler.** Split across a hand-written list of variants in
//! `repl.rs` and a `match` in `main.rs` ending in `_ => {}`, the two have to agree with nothing
//! making them. They agreed, but nothing made them: adding a variant to the forwarding list and
//! forgetting the match produced a command that was sent, silently did nothing, and still got its
//! episode brackets -- a blank-line sandwich around no output, which is exactly what
//! `Console::announce_foreign_output` warns against three lines above that match.
//! [`SlashCommand::answered_by`] is now exhaustive and so is [`answer`], so a new variant fails to
//! compile in both places.

use std::sync::Arc;

use super::SlashCommand;
use crate::{
    agent, cli, conversation, error, mcp, memory, provider, render, repl, session::SessionManager,
    skills, with_console,
};

/// What the host loop does once a command has been answered.
///
/// A return value rather than a `break` inside the arm, because the dispatcher sits outside the
/// loop. Only two of the arms need it: `/fork` and the rest leave the loop running, while a REPL
/// thread that has gone away ends it.
pub(crate) enum AfterCommand {
    Continue,
    Leave,
}

/// Answer one forwarded command.
///
/// Exhaustive over [`SlashCommand`] on purpose; see the module docs. The variants the REPL thread
/// answers itself are listed rather than swept into a wildcard, so that adding one is a decision
/// taken twice instead of a silent no-op.
pub(crate) async fn answer(command: SlashCommand, ctx: HostCommandContext<'_>) -> AfterCommand {
    // Destructured back into the names the arms already use, rather than `ctx.` at every site.
    // `agent` shadows nothing: a module path lives in the type namespace and this binding in the
    // value namespace, so `agent::PromptRetention` still resolves alongside it.
    let HostCommandContext {
        agent,
        messages,
        agent_event_sender,
        config,
        console,
        mcp_manager,
        providers,
        session_id,
        session_lock,
        session_manager,
        token_store,
    } = ctx;
    let session_id_cell = session_id;
    let mut session_id = *session_id_cell;
    // Every command answered here says something, even if only that a list is empty, and much of it
    // prints through the `cli` modules the console cannot see. One announcement covers all of them.
    //
    // There is deliberately no "does this one answer by running a turn" exception any more.
    // Announcing is idempotent within an episode -- the opening blank is spent once, by whichever
    // writer gets there first -- so a command that runs a turn is spaced identically whether the
    // turn happens or it bails first. A predicate making that distinction leaves `/skill
    // nosuchskill` printing its error flush against both the line above and the prompt below.
    //
    // "Answered here" is the qualification, and it is why this is gated rather than unconditional:
    // the arm below for the six the REPL owns prints nothing at all. In debug it trips an
    // assertion, but in release it is a silent no-op, and announcing first would give it exactly
    // the blank-line sandwich this call's own doc warns against -- in the builds where nothing is
    // watching.
    if command.answered_by() == repl::Answerer::Host {
        with_console(console, |console| console.announce_foreign_output());
    }
    match command {
        SlashCommand::Session => match &session_id {
            Some(id) => with_console(console, |console| {
                console.session_id("Current session", &id.to_string())
            }),
            None => eprintln!("No active session yet."),
        },
        SlashCommand::Compact(instructions) => {
            let request = crate::agent::CompactRequest {
                origin: crate::agent::CompactOrigin::Manual,
                instructions: instructions
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_string),
                keep_recent: None,
            };
            match crate::compact_interruptible(agent, &mut session_id, messages, request).await {
                Ok(outcome) => {
                    with_console(console, |console| {
                        console.hint(&render::compaction_summary(&outcome))
                    });
                }
                Err(error) => {
                    with_console(console, |console| console.error(&error));
                }
            }
        }
        SlashCommand::Rewind(turns) => {
            let turns = turns.unwrap_or(1);
            // `rewind(0)` returns `None` unconditionally, so without this the `None`
            // arm below would report "fewer than 0 turn(s)". The count is what's
            // wrong, not the conversation.
            let rewound = if turns == 0 {
                None
            } else {
                messages.rewind(turns)
            };
            match (session_id, rewound) {
                (Some(id), Some(event)) => {
                    if let Err(error) = session_manager.save_event(id, &event).await {
                        // Put the turns back rather than leave memory and disk
                        // disagreeing, which would resurrect them on the next resume
                        // and make the rewind look like it silently un-did itself.
                        messages.pop_repair();
                        with_console(console, |console| console.error(&error));
                    } else {
                        agent.reset_conversation_markers().await;
                        with_console(console, |console| {
                            console.hint(&format!(
                                "Rewound {} turn(s). The model no longer sees them; \
                             `meka session export` still does.",
                                turns,
                            ))
                        });
                    }
                }
                // No session means nothing was ever persisted, so the in-memory rewind
                // (which did happen) is the whole story.
                (None, Some(_)) => {
                    agent.reset_conversation_markers().await;
                    with_console(console, |console| {
                        console.hint(&format!("Rewound {} turn(s).", turns))
                    });
                }
                (_, None) if turns == 0 => {
                    eprintln!("Nothing to rewind: /rewind takes a turn count of 1 or more.");
                }
                (_, None) => {
                    eprintln!(
                        "Nothing to rewind: the conversation has fewer than {} turn(s).",
                        turns
                    );
                }
            }
        }
        SlashCommand::Export => match &session_id {
            Some(id) => {
                match crate::session::cli::export_session(
                    session_manager,
                    *id,
                    None,
                    cli::SessionExportFormat::Markdown,
                )
                .await
                {
                    // The name is generated from the session id and the file lands in
                    // the working directory, so a REPL user who is not told it has
                    // nowhere to look. `meka session export` stays quiet: there the
                    // shell is the one that knows.
                    Ok(Some(path)) => {
                        // Shown absolute: `/cd` moves the session's directory while
                        // the export lands in the process's, so a bare filename would
                        // point at the wrong one.
                        let shown = std::env::current_dir()
                            .map(|dir| dir.join(&path))
                            .unwrap_or(path);
                        eprintln!("Exported session to {}", shown.display());
                    }
                    Ok(None) => {}
                    Err(error) => with_console(console, |console| console.error(&error)),
                }
            }
            None => eprintln!("No active session to export."),
        },
        SlashCommand::Fork => match session_id {
            Some(id) => match crate::session::cli::fork_and_lock(session_manager, id).await {
                Ok(crate::session::cli::ForkHandoff::Switched { id, lock }) => {
                    // Replacing the slot's contents drops the original guard only now
                    // that the new one is held; see
                    // `crate::session::cli::fork_and_lock`.
                    crate::hold_session_lock(session_lock, Some(lock));
                    session_id = Some(id);
                    // `messages` is deliberately untouched, so the branch happens at
                    // the current head and the next turn continues in the copy.
                    with_console(console, |console| {
                        console.session_id("Forked session", &id.to_string())
                    });
                }
                Ok(crate::session::cli::ForkHandoff::LockFailed { id, error }) => {
                    with_console(console, |console| console.error(&error));
                    with_console(console, |console| {
                        console.hint(&format!("Staying in the original. The copy exists: {}", id))
                    });
                }
                Ok(crate::session::cli::ForkHandoff::SourceGone) => {
                    eprintln!("Session no longer exists: {}", id);
                }
                Err(error) => eprintln!("Failed to fork session: {}", error),
            },
            None => eprintln!("No active session to fork."),
        },
        SlashCommand::McpList => {
            if let Err(error) = mcp::cli::run_list(
                &config.mcp_servers,
                mcp_manager.as_ref(),
                &session_manager.token_store(),
            )
            .await
            {
                with_console(console, |console| console.error(&error));
            }
        }
        // These three report success at `info!` and print nothing, which is right for
        // the `meka mcp …` CLI (the exit code carries it) and wrong here: a REPL
        // command has no exit code, so silence is indistinguishable from the command
        // never having run, and it leaves the `[display]` blank lines wrapped around
        // an empty region. `/permission` sets the precedent for confirming a state
        // change the user asked for.
        SlashCommand::McpReconnect { server } => {
            match mcp::cli::run_reconnect(&config.mcp_servers, token_store, &server).await {
                // "Connected", not "Reconnected": this is a smoke test on a throwaway
                // client, and the session's own connection to that server is untouched.
                Ok(()) => eprintln!("Connected to '{}'.", server),
                Err(error) => with_console(console, |console| console.error(&error)),
            }
        }
        SlashCommand::McpLogin { server } => {
            match mcp::cli::run_login(&config.mcp_servers, token_store, &server).await {
                Ok(()) => eprintln!("Authorized '{}'.", server),
                Err(error) => with_console(console, |console| console.error(&error)),
            }
        }
        SlashCommand::McpLogout { server } => {
            match mcp::cli::run_logout(&config.mcp_servers, token_store, &server).await {
                Ok(()) => eprintln!("Cleared credentials for '{}'.", server),
                Err(error) => with_console(console, |console| console.error(&error)),
            }
        }
        SlashCommand::McpPrompt {
            server,
            prompt: prompt_name,
            args,
        } => 'prompt: {
            let Some(manager) = mcp_manager.as_ref() else {
                eprintln!("No MCP servers configured.");
                break 'prompt;
            };
            let entry = manager.server_entry(&server);
            let Some(entry) = entry else {
                // Labelled break, not `continue`: `continue` targets the agent
                // loop, skipping the `AgentToReplEvent::Done` send below and
                // leaving the REPL thread parked in `wait_for_agent` with no
                // prompt, for good. Same reason as `SkillInvoke`'s `'invoke`.
                eprintln!(
                    "Unknown MCP server '{}' (configured: {}).",
                    server,
                    manager.server_names().join(", ")
                );
                break 'prompt;
            };
            // Map positional args to declared prompt argument names (lookup via
            // prompts/list).
            let arg_names = match mcp::list_prompts(
                &entry,
                &tokio_util::sync::CancellationToken::new(),
            )
            .await
            {
                Ok(prompts) => prompts
                    .into_iter()
                    .find(|p| p.name == prompt_name)
                    .and_then(|p| p.arguments)
                    .map(|args| args.into_iter().map(|a| a.name).collect::<Vec<_>>())
                    .unwrap_or_default(),
                Err(error) => {
                    // The `McpConnection` error already names the server and the
                    // operation, so wrapping it here would say both twice.
                    with_console(console, |console| console.error(&error));
                    Vec::new()
                }
            };
            let mut arguments: Option<serde_json::Map<String, serde_json::Value>> = None;
            if !arg_names.is_empty() {
                let mut map = serde_json::Map::new();
                for (i, name) in arg_names.iter().enumerate() {
                    if let Some(value) = args.get(i) {
                        map.insert(name.clone(), serde_json::Value::String(value.clone()));
                    }
                }
                arguments = Some(map);
            }
            match mcp::get_prompt(
                &entry,
                prompt_name.clone(),
                arguments,
                &tokio_util::sync::CancellationToken::new(),
            )
            .await
            {
                Ok(result) => {
                    // Render the prompt messages as a single user turn, same shape
                    // as the `mcp_prompt_get` tool output.
                    let mut body = String::new();
                    for message in &result.messages {
                        let role = match message.role {
                            rmcp::model::Role::User => "user",
                            rmcp::model::Role::Assistant => "assistant",
                        };
                        if let rmcp::model::ContentBlock::Text(text) = &message.content {
                            body.push_str(&format!("{}: {}\n", role, text.text));
                        }
                    }
                    let user_input = body.trim().to_string();
                    if user_input.is_empty() {
                        // Every other exit from this arm prints; a server whose
                        // prompt renders to nothing would otherwise return the user
                        // straight to a fresh prompt, which reads as "the command
                        // did nothing" rather than "the prompt was empty".
                        eprintln!(
                            "'{}:{}' rendered an empty prompt; nothing to send.",
                            server, prompt_name
                        );
                    } else {
                        match crate::run_turn_interruptible(
                            agent,
                            &mut session_id,
                            messages,
                            user_input,
                            agent::PromptRetention::Keep,
                        )
                        .await
                        {
                            Ok(()) => {}
                            Err(error::MekaError::Interrupted) => {
                                with_console(console, |console| console.annotation("interrupted"));
                            }
                            Err(error) => with_console(console, |console| console.error(&error)),
                        }
                    }
                }
                Err(error) => {
                    with_console(console, |console| console.error(&error));
                }
            }
        }
        SlashCommand::MemoryList => {
            if let Err(error) = memory::cli::run_list(
                &session_manager.memory_store(true),
                memory::cli::ListDetail::TableOnly,
            )
            .await
            {
                with_console(console, |console| console.error(&error));
            }
        }
        SlashCommand::MemoryShow { name } => {
            if let Err(error) =
                memory::cli::run_show(&session_manager.memory_store(true), &name).await
            {
                with_console(console, |console| console.error(&error));
            }
        }
        // Scoped to the session in the REPL, unlike `meka schedule list`, which has no
        // conversation to be "this one" and so shows every session's jobs.
        SlashCommand::ScheduleList => match session_id {
            Some(id) => {
                if let Err(error) = crate::schedule::cli::run_list_for_session(
                    session_manager,
                    id,
                    &config.schedule,
                )
                .await
                {
                    with_console(console, |console| console.error(&error));
                }
            }
            None => eprintln!("No active session yet."),
        },
        SlashCommand::ScheduleCancel { id } => match session_id {
            Some(session) => {
                match session_manager
                    .schedule_store()
                    .cancel_scheduled_job(session, &id)
                    .await
                {
                    Ok(Some(cancelled)) => {
                        eprintln!("Cancelled job {}.", &cancelled[..8.min(cancelled.len())]);
                    }
                    Ok(None) => {
                        eprintln!("No scheduled job matching '{}'.", id);
                    }
                    Err(error) => with_console(console, |console| console.error(&error)),
                }
            }
            None => eprintln!("No active session yet."),
        },
        SlashCommand::TaskList => match session_id {
            Some(id) => {
                if let Err(error) =
                    crate::background::cli::run_list_for_session(session_manager, id).await
                {
                    with_console(console, |console| console.error(&error));
                }
            }
            None => eprintln!("No active session yet."),
        },
        SlashCommand::TaskCancel { id } => match session_id {
            Some(session) => {
                // Recorded first, then signalled: `finish_background_task` only
                // overwrites a `running` row, so a task finishing in the same instant
                // cannot report success after the user was told it stopped.
                match crate::background::cli::cancel(session_manager, session, id.as_deref()).await
                {
                    Ok(cancelled) if cancelled.is_empty() => {
                        eprintln!("No running background tasks.")
                    }
                    Ok(cancelled) => {
                        for task_id in &cancelled {
                            agent.background_tasks().cancel(task_id).await;
                        }
                        eprintln!("Cancelling {} background task(s).", cancelled.len());
                    }
                    Err(error) => with_console(console, |console| console.error(&error)),
                }
            }
            None => eprintln!("No active session yet."),
        },
        SlashCommand::SkillList => {
            if let Err(error) = skills::cli::run_list(&config.skill_roots(), false).await {
                with_console(console, |console| console.error(&error));
            }
        }
        SlashCommand::SkillInvoke { name, extra } => 'invoke: {
            // Labeled block so the early-exit error paths can `break 'invoke` out of
            // the arm body without skipping the `AgentToReplEvent::Done` send below;
            // `continue` would short-circuit the outer `while let`, leaving the REPL
            // stuck in `wait_for_agent` and never drawing the next prompt.
            let installed = agent.skills().current().await;
            let Some(skill) = installed.find(&name) else {
                // Prose, not `{:?}` on a `Vec<&str>`. This is user-facing output, and
                // the debug rendering printed `["a", "b"]` -- quotes, brackets and all
                // -- for a list a person is meant to read and pick from.
                let message = match installed.skip_reason(&name) {
                    // The same distinction `skill_read` draws for the model, in the
                    // same words: a file that is there and unreadable is not a missing
                    // skill.
                    Some(_) => installed.unavailable(&name),
                    None if installed.skills.is_empty() => {
                        format!("unknown skill '{}'; no skills are installed", name)
                    }
                    None => format!(
                        "unknown skill '{}'; available: {}",
                        name,
                        installed
                            .skills
                            .iter()
                            .map(|s| s.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                };
                with_console(console, |console| console.error(&message));
                break 'invoke;
            };
            let body = match skills::load_skill_body(skill).await {
                Ok(body) => body,
                Err(error) => {
                    with_console(console, |console| {
                        console.error(&format!("failed to load skill '{}': {}", name, error))
                    });
                    break 'invoke;
                }
            };
            // Prepend the user's free-form directive to the skill body when present.
            // The blank-line separator gives the model a visual cue that the first
            // paragraph is the user's "do this skill, but with this twist" and the rest
            // is the skill's static body.
            let body = if extra.is_empty() {
                body
            } else {
                format!("{}\n\n{}", extra, body)
            };
            match crate::run_turn_interruptible(
                agent,
                &mut session_id,
                messages,
                body,
                agent::PromptRetention::Keep,
            )
            .await
            {
                Ok(()) => {}
                Err(error::MekaError::Interrupted) => {
                    with_console(console, |console| console.annotation("interrupted"));
                }
                Err(error) => with_console(console, |console| console.error(&error)),
            }
        }
        SlashCommand::Status => {
            let snap = agent.session_stats_snapshot();
            let (context_tokens, context_window) = agent.context_usage();
            let effort = agent.resolved_effort();
            // This session's profile, not the process default's. Reading `config` here
            // reported the default profile's model and backend beside a window and an
            // effort that came from the session's, so `/status` and `/provider`
            // contradicted each other on any resume onto a non-default profile.
            let binding = agent.provider_binding().clone();
            let settings = providers.settings(&binding);
            render::render_session_status(
                &snap,
                &render::ModelStatus {
                    model: settings
                        .as_ref()
                        .ok()
                        .and_then(|settings| settings.model.as_deref()),
                    profile: Some(binding.as_str()),
                    backend: settings
                        .as_ref()
                        .ok()
                        .map(|settings| settings.backend.as_str()),
                    effort: effort.as_deref(),
                    thinking: settings
                        .as_ref()
                        .map(|settings| settings.thinking)
                        .unwrap_or_default(),
                },
                messages.len(),
                context_tokens,
                context_window,
            );
        }
        SlashCommand::Usage => match agent.fetch_usage().await {
            Ok(Some(usage)) => render::render_account_usage(&usage),
            Ok(None) => with_console(console, |console| {
                console.hint("Account usage isn't available for this provider.")
            }),
            Err(error) => with_console(console, |console| console.error(&error)),
        },
        SlashCommand::History(limit) => {
            let materialised = messages.as_slice();
            let slice = match limit {
                Some(n) => render::last_n_turns(materialised, n),
                None => materialised,
            };
            // Say so rather than printing nothing, like every other list command
            // (`/tasks`, `/memory`, `/skill`). Silence here would be ambiguous between
            // "no history" and "the command did not run", and it would leave the
            // `[display]` blank lines bracketing an empty region. `/history 0` asks
            // for nothing and gets the neutral wording: there may well be a
            // conversation, it just wasn't what was asked for.
            if !render::render_message_history(slice, &crate::history_render_options(config)) {
                if materialised.is_empty() {
                    eprintln!("No conversation history yet.");
                } else {
                    eprintln!("Nothing to show.");
                }
            }
        }
        // The REPL thread answers these itself, listed rather than swept into a wildcard. The
        // wildcard is what let a forwarded command match nothing and still get its episode
        // brackets; naming them means adding a variant fails here until someone has decided
        // which side owns it.
        SlashCommand::Cd(_)
        | SlashCommand::Clear
        | SlashCommand::Exit
        | SlashCommand::Help
        | SlashCommand::Permission(_)
        | SlashCommand::Provider(_) => {
            debug_assert!(
                false,
                "`answered_by` says the REPL answers this, so the forwarding arm should not have \
                 sent it here"
            );
        }
    }
    *session_id_cell = session_id;
    if agent_event_sender
        .send(repl::AgentToReplEvent::Done)
        .is_err()
    {
        return AfterCommand::Leave;
    }
    AfterCommand::Continue
}

/// What answering a command needs from the host loop.
///
/// Explicit rather than implicit in a closure over `run_interactive`'s locals, which is worth the
/// noise: this is the list of things a slash command can reach, and it was previously answerable
/// only by reading a 553-line match arm.
pub(crate) struct HostCommandContext<'a> {
    pub(crate) agent: &'a agent::Agent,
    pub(crate) agent_event_sender: &'a std::sync::mpsc::Sender<repl::AgentToReplEvent>,
    pub(crate) config: &'a crate::config::ResolvedConfig,
    pub(crate) console: &'a Arc<std::sync::Mutex<crate::console::Console>>,
    pub(crate) mcp_manager: &'a Option<Arc<mcp::McpClientManager>>,
    pub(crate) messages: &'a mut conversation::Conversation,
    pub(crate) providers: &'a Arc<provider::ProviderRegistry>,
    /// `/fork` moves the session the loop is serving, so this is the loop's own cell.
    pub(crate) session_id: &'a mut Option<uuid::Uuid>,
    pub(crate) session_lock: &'a crate::session::SessionLockSlot,
    pub(crate) session_manager: &'a SessionManager,
    pub(crate) token_store: &'a crate::session::TokenStore,
}
