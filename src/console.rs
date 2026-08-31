//! The terminal between two prompts.
//!
//! meka's `[display]` spacing is described in terms of prompts, not turns: `newline_after_prompt`
//! is a blank line after the line you typed and `newline_before_prompt` one before the next prompt,
//! and both apply to *anything* printed in between. The unit is therefore an **episode**, and this
//! module owns it.
//!
//! Two rules make the shapes that break unrepresentable.
//!
//! **One owner, and it is the one that always runs.** Split ownership -- the agent opening the
//! blanks on `FrontendEvent::TurnStarted` and closing them on success, the host closing them on
//! failure -- brackets a turn that failed before it started, or a slash command that answered
//! without running a turn, once or not at all. An episode is closed by whoever is about to draw the
//! next prompt, which happens exactly once per prompt whatever went on before it.
//!
//! **An episode either prints, and gets its brackets, or it leaves no trace.** The opening blank is
//! *armed* rather than printed, and fires just before the episode's first real output, so nothing
//! can slip above it. If nothing ever prints, [`Console::close_episode`] restores the row it opened
//! on, which is what stops a scheduler wake that finds nothing to run from leaving a second prompt
//! behind.
//!
//! [`RowState`] is the fact none of the previous code tracked: a blank line only makes a gap when
//! the cursor is at column zero. Two writers park it mid-row deliberately (the thinking indicator
//! and the MCP progress line), and reedline leaves a drawn prompt behind when it breaks out for a
//! wake. Before this existed each of those was compensated for by hand at the sites someone
//! remembered, and a blank printed anywhere else was silently spent terminating the row instead.

use std::io::Write;

use crate::render::{self, OutputSpacing, RenderMode, StreamingRenderer, ToolParams};

/// What occupies the row the cursor is sitting on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowState {
    /// Column zero of a row nothing has written to. Anything may print immediately.
    Empty,
    /// An in-place status line the writer expects to overwrite or erase: the thinking indicator, or
    /// an MCP progress line whose text comes from the server. Neither is content, so meka's own
    /// output replaces it rather than following it.
    Transient,
    /// reedline returned without the CRLF it normally writes on the way out of `read_line`, leaving
    /// the prompt it drew on this row and the cursor at the end of it. Real output has to move past
    /// it; an episode that produces none has to erase it, or reedline paints a second prompt on the
    /// row below.
    PromptParked,
}

/// The `[display]` blank-line settings.
///
/// They gate *printing* and nothing else. Every state transition below runs identically with them
/// off, which is the difference between "no blank line here" and "the spacing machine stops
/// advancing"; the latter leaks the previous turn's last block into the next one and prints the
/// blank the setting just disabled.
#[derive(Debug, Clone, Copy)]
pub struct Spacing {
    pub newline_before_prompt: bool,
    pub newline_after_prompt: bool,
}

/// A kind of block, in the sense that matters to spacing: what separates it from what came before.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    /// meka speaking for itself: an error, a hint, a session id, a status table. Deliberately does
    /// not advance the block machine, because these are punctuation between the model's blocks
    /// rather than blocks of their own.
    Chrome,
    Text,
    ToolIndicator(ToolParams),
    Thinking,
    /// Renders its own leading and trailing blank lines, so it asks for no separator and leaves the
    /// machine claiming a trailing blank that the next block must not double.
    TodoList,
}

/// Something that happens to the console, as a value, so the decision it forces can be tested
/// without a terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    OpenEpisode(RowState),
    CloseEpisode,
    /// Output that `Console` cannot see is about to print: a slash command answering through one of
    /// the `cli` modules, or a child process meka handed the terminal to.
    AnnounceForeign,
    Block(BlockKind),
    /// An in-place status line is about to be drawn.
    OpenTransient,
    /// Keep the transient line by writing the newline its writer withheld.
    CommitTransient,
    /// Discard the transient line, because something is about to render in its place.
    EraseTransient,
}

/// What has to happen to the current row before anything else prints on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Settle {
    Nothing,
    /// End the row, keeping what is on it.
    Terminate,
    /// Blank the row and return to column zero, discarding what is on it.
    Erase,
}

/// Everything an action writes before its own content, and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Emit {
    pub settle: Settle,
    /// The `newline_after_prompt` blank, fired late so it can never land below the first thing the
    /// episode prints.
    pub after_prompt_blank: bool,
    /// The separator between two blocks of different kinds, from [`OutputSpacing`].
    pub separator_blank: bool,
    /// The `newline_before_prompt` blank.
    pub before_prompt_blank: bool,
}

impl Emit {
    const NOTHING: Self = Self {
        settle: Settle::Nothing,
        after_prompt_blank: false,
        separator_blank: false,
        before_prompt_blank: false,
    };
}

/// The console's state, separated from the console so [`step`] can be a pure function of it.
#[derive(Clone, Copy)]
pub struct State {
    pub row: RowState,
    spacing: OutputSpacing,
    /// Whether the `newline_after_prompt` blank is still owed. Armed by `OpenEpisode`, spent by
    /// the first thing that prints.
    pending_after_blank: bool,
    /// Whether this episode has printed anything. Set when the opening blank is spent, *whether or
    /// not the flag let it print*, so it means "something happened" rather than "a blank was
    /// written".
    printed: bool,
}

impl State {
    fn new() -> Self {
        Self {
            row: RowState::Empty,
            spacing: OutputSpacing::new(),
            pending_after_blank: false,
            printed: false,
        }
    }

    #[cfg(test)]
    fn printed(&self) -> bool {
        self.printed
    }
}

/// The whole decision, as a pure function.
///
/// Every blank line meka prints between two prompts is one of the three in [`Emit`], and this is
/// the only place any of them is decided. Split out from the printing for the same reason
/// `repl::indicator_action` is: in a dispatch that mixes the two, the only way to see a wrong
/// answer is to run a terminal and look at it.
pub fn step(state: State, spacing: Spacing, action: Action) -> (Emit, State) {
    let mut next = state;
    match action {
        Action::OpenEpisode(row) => {
            next.row = row;
            next.pending_after_blank = true;
            next.printed = false;
            // Unconditional, and the fix for a bug that survived because it looked like a tidy
            // guard: this records "a prompt is what came last", which is true whether or not a
            // blank followed it. Gating it on `newline_after_prompt` left the next episode reading
            // the previous one's final block and printing the separator the setting had disabled.
            next.spacing.after_prompt();
            (Emit::NOTHING, next)
        }
        Action::CloseEpisode => {
            let settle = match (state.row, state.printed) {
                (RowState::Empty, _) => Settle::Nothing,
                // A leftover status line is not content and must not survive into the prompt.
                (RowState::Transient, _) => Settle::Erase,
                // Nothing printed, so the screen has to look exactly as it did: erasing the stale
                // prompt lets reedline repaint on this row instead of choosing the one below.
                (RowState::PromptParked, false) => Settle::Erase,
                (RowState::PromptParked, true) => Settle::Terminate,
            };
            next.row = RowState::Empty;
            next.pending_after_blank = false;
            // Closing twice is closing once. Without this a second call would emit the closing
            // blank again, which is reachable on the way out: the shutdown path closes after the
            // last prompt and then has more to say.
            next.printed = false;
            (
                Emit {
                    settle,
                    before_prompt_blank: state.printed && spacing.newline_before_prompt,
                    ..Emit::NOTHING
                },
                next,
            )
        }
        Action::AnnounceForeign => {
            let (settle, after_prompt_blank) = open_output(&mut next, spacing);
            (
                Emit {
                    settle,
                    after_prompt_blank,
                    ..Emit::NOTHING
                },
                next,
            )
        }
        Action::Block(kind) => {
            let (settle, after_prompt_blank) = open_output(&mut next, spacing);
            let separator_blank = match kind {
                BlockKind::Chrome | BlockKind::TodoList => false,
                BlockKind::Text => next.spacing.before_text(),
                BlockKind::ToolIndicator(params) => next.spacing.before_tool_indicator(params),
                BlockKind::Thinking => next.spacing.before_thinking(),
            };
            if kind == BlockKind::TodoList {
                next.spacing.after_todo_list();
            }
            (
                Emit {
                    settle,
                    after_prompt_blank,
                    separator_blank,
                    before_prompt_blank: false,
                },
                next,
            )
        }
        Action::OpenTransient => {
            // A redraw overwrites its own row, so only a parked prompt has to be moved past.
            let settle = match state.row {
                RowState::PromptParked => Settle::Terminate,
                RowState::Empty | RowState::Transient => Settle::Nothing,
            };
            let after_prompt_blank = spend_pending(&mut next, spacing);
            next.row = RowState::Transient;
            (
                Emit {
                    settle,
                    after_prompt_blank,
                    ..Emit::NOTHING
                },
                next,
            )
        }
        Action::CommitTransient => {
            let settle = if state.row == RowState::Transient {
                Settle::Terminate
            } else {
                Settle::Nothing
            };
            next.row = RowState::Empty;
            (
                Emit {
                    settle,
                    ..Emit::NOTHING
                },
                next,
            )
        }
        Action::EraseTransient => {
            let settle = if state.row == RowState::Transient {
                Settle::Erase
            } else {
                Settle::Nothing
            };
            next.row = RowState::Empty;
            (
                Emit {
                    settle,
                    ..Emit::NOTHING
                },
                next,
            )
        }
    }
}

/// Settle the row and spend the opening blank, which is what every writer of real output does
/// first.
fn open_output(next: &mut State, spacing: Spacing) -> (Settle, bool) {
    let settle = match next.row {
        RowState::Empty => Settle::Nothing,
        RowState::Transient => Settle::Erase,
        RowState::PromptParked => Settle::Terminate,
    };
    next.row = RowState::Empty;
    (settle, spend_pending(next, spacing))
}

fn spend_pending(next: &mut State, spacing: Spacing) -> bool {
    if !next.pending_after_blank {
        return false;
    }
    next.pending_after_blank = false;
    next.printed = true;
    spacing.newline_after_prompt
}

/// The single writer for everything that appears between two prompts.
///
/// Shared by the blocking REPL thread and the agent's frontend task, which also gives the two
/// threads that write to the terminal one lock to contend on rather than none.
pub struct Console {
    state: State,
    spacing: Spacing,
    render_mode: RenderMode,
    /// Open across consecutive text deltas; closed by any other block, and by the end of the
    /// episode. Episode-bounded rather than turn-bounded so a turn that dies mid-paragraph still
    /// shows what it streamed, in its own episode, instead of having it flushed under the next
    /// prompt.
    renderer: Option<StreamingRenderer>,
}

impl Console {
    pub fn new(spacing: Spacing, render_mode: RenderMode) -> Self {
        Self {
            state: State::new(),
            spacing,
            render_mode,
            renderer: None,
        }
    }

    fn act(&mut self, action: Action) {
        let (emit, next) = step(self.state, self.spacing, action);
        self.state = next;
        match emit.settle {
            Settle::Nothing => {}
            Settle::Terminate => eprintln!(),
            Settle::Erase => render::begin_own_line(),
        }
        if emit.after_prompt_blank {
            eprintln!();
        }
        if emit.separator_blank {
            eprintln!();
        }
        if emit.before_prompt_blank {
            eprintln!();
        }
    }

    /// Begin the episode that follows a prompt, given what reedline left on the row.
    pub fn open_episode(&mut self, row: RowState) {
        self.act(Action::OpenEpisode(row));
    }

    /// End the episode, immediately before the next prompt is drawn.
    pub fn close_episode(&mut self) {
        self.close_text();
        self.act(Action::CloseEpisode);
    }

    /// Declare that output this console cannot see is about to print.
    ///
    /// Needed only where meka hands the terminal to something else: a slash command that answers
    /// through one of the `cli` modules, or a child process under `!command`. Anything printed
    /// through the methods below announces itself.
    ///
    /// A command that prints *nothing* must not call this, or it gets blank lines wrapped around an
    /// empty region. Every REPL command says something, even if only that a list is empty; the
    /// exceptions are a successful `/cd` and `/clear`, where the prompt and the cleared screen are
    /// the confirmation.
    pub fn announce_foreign_output(&mut self) {
        self.act(Action::AnnounceForeign);
    }

    pub fn error(&mut self, error: &dyn std::fmt::Display) {
        self.close_text();
        self.act(Action::Block(BlockKind::Chrome));
        render::render_error(error);
    }

    pub fn hint(&mut self, message: &str) {
        self.close_text();
        self.act(Action::Block(BlockKind::Chrome));
        render::render_hint(message);
    }

    pub fn session_id(&mut self, label: &str, id: &str) {
        self.close_text();
        self.act(Action::Block(BlockKind::Chrome));
        render::render_session_id(label, id);
    }

    /// The heading above a block of command output. See [`render::render_heading`].
    pub fn heading(&mut self, heading: &str) {
        self.close_text();
        self.act(Action::Block(BlockKind::Chrome));
        render::render_heading(heading);
    }

    /// A stage direction about the output: `(interrupted)`. See [`render::render_annotation`].
    pub fn annotation(&mut self, note: &str) {
        self.close_text();
        self.act(Action::Block(BlockKind::Chrome));
        render::render_annotation(note);
    }

    pub fn line(&mut self, line: &str) {
        self.close_text();
        self.act(Action::Block(BlockKind::Chrome));
        eprintln!("{}", line);
    }

    /// Print through a closure, for the callers whose painter takes arguments this module has no
    /// reason to model (`render_session_status`, `render_account_usage`, the help text).
    pub fn chrome(&mut self, paint: impl FnOnce()) {
        self.close_text();
        self.act(Action::Block(BlockKind::Chrome));
        paint();
    }

    pub fn tool_indicator(
        &mut self,
        name: &str,
        input: &serde_json::Value,
        display_summary: Option<&str>,
        params: ToolParams,
    ) {
        self.close_text();
        self.act(Action::Block(BlockKind::ToolIndicator(params)));
        render::render_tool_indicator(name, input, display_summary, params);
    }

    pub fn thinking_block(&mut self, content: &str, show_full: bool) {
        self.close_text();
        self.act(Action::Block(BlockKind::Thinking));
        render::render_thinking_block(content, show_full);
    }

    /// An empty list prints nothing and must not claim the trailing blank line that
    /// [`render::render_todo_list`] would otherwise have left, so the block is opened only once the
    /// list is known to have content.
    pub fn todo_list(&mut self, title: Option<&str>, items: &[crate::tools::todo::TodoItem]) {
        if items.is_empty() {
            return;
        }
        self.close_text();
        self.act(Action::Block(BlockKind::TodoList));
        render::render_todo_list(title, items);
    }

    pub fn token_usage(&mut self, usage: &crate::provider::TokenUsage) {
        self.close_text();
        self.act(Action::Block(BlockKind::Chrome));
        render::render_token_usage(usage);
    }

    /// Draw an in-place status line, returning whether anything reached the terminal.
    ///
    /// `draw` returns false off a tty, where redrawing in place would instead accumulate one line
    /// per update. Nothing is spent in that case: an indicator that cannot be drawn must not take
    /// the episode's opening blank or move the row, or it shifts the layout of the output that
    /// *is* produced.
    pub fn transient(&mut self, draw: impl FnOnce() -> bool) -> bool {
        if !render::live_indicator_supported() {
            return false;
        }
        self.act(Action::OpenTransient);
        let drawn = draw();
        if !drawn {
            self.state.row = RowState::Empty;
        }
        drawn
    }

    /// Draw the thinking indicator, returning whether it reached the terminal.
    ///
    /// `opening` separates the first draw of a block like the thinking block it stands in for, so
    /// the line it eventually commits to sits apart from what preceded it. Redraws overwrite their
    /// own row and ask for nothing.
    pub fn thinking_indicator(&mut self, opening: bool, estimate: Option<u64>) -> bool {
        if !render::live_indicator_supported() {
            return false;
        }
        if opening {
            // The indicator draws from column zero, so an open text run would have its last line
            // overwritten -- and since the indicator is kept rather than erased, that damage would
            // stay on screen.
            self.close_text();
            self.act(Action::Block(BlockKind::Thinking));
        }
        self.act(Action::OpenTransient);
        let drawn = render::render_thinking_indicator(estimate);
        if !drawn {
            self.state.row = RowState::Empty;
        }
        drawn
    }

    /// Keep the transient line by writing the newline its writer withheld.
    pub fn commit_transient(&mut self) {
        self.act(Action::CommitTransient);
    }

    /// Discard the transient line, for the one case where something is about to render in its
    /// place.
    pub fn erase_transient(&mut self) {
        self.act(Action::EraseTransient);
    }

    /// Whether a text block is still open, for the frontend's tests: a run left open past the end
    /// of its episode is what flushes a failed turn's tail under the next prompt.
    #[cfg(test)]
    pub fn has_open_text(&self) -> bool {
        self.renderer.is_some()
    }

    /// Whether this episode has printed anything, for [`crate::relay`]'s tests.
    ///
    /// The one bit of console state a caller outside this module can observe without a terminal,
    /// and enough to answer the question the relay's test asks: did a log line reach the console at
    /// all, or go round it to stderr.
    #[cfg(test)]
    pub fn has_printed(&self) -> bool {
        self.state.printed()
    }

    /// Put the row into a state a test cannot reach through the drawing API.
    ///
    /// `transient` and `thinking_indicator` both return early unless
    /// `render::live_indicator_supported()`, which is false without a terminal, so the row a
    /// mid-turn log line actually collides with is otherwise unreachable under `cargo test`.
    #[cfg(test)]
    pub fn force_row(&mut self, row: RowState) {
        self.state.row = row;
    }

    /// What the cursor is sitting on, for the same tests.
    #[cfg(test)]
    pub fn row(&self) -> RowState {
        self.state.row
    }

    pub fn text_delta(&mut self, text: &str) {
        if self.renderer.is_none() {
            self.act(Action::Block(BlockKind::Text));
            self.renderer = Some(StreamingRenderer::new(self.render_mode));
        }
        if let Some(renderer) = self.renderer.as_mut()
            && let Err(error) = renderer.push_delta(text)
        {
            tracing::debug!("console renderer push_delta failed: {}", error);
        }
    }

    /// Flush and drop any open text block, so block types don't interleave.
    pub fn close_text(&mut self) {
        let Some(mut renderer) = self.renderer.take() else {
            return;
        };
        // A broken stderr pipe, typically. Log and move on rather than panicking inside a render
        // path.
        if let Err(error) = renderer.finish() {
            tracing::debug!("console renderer finish failed: {}", error);
        }
        // The assistant's text goes to stdout while every blank line here goes to stderr, and
        // `finish` does not flush the termimad path. Without this a final paragraph that did not
        // end in a newline would sit in the line buffer while the closing blank and the next prompt
        // were already on screen.
        if let Err(error) = std::io::stdout().flush() {
            tracing::debug!("console stdout flush failed: {}", error);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOTH: Spacing = Spacing {
        newline_before_prompt: true,
        newline_after_prompt: true,
    };
    const NEITHER: Spacing = Spacing {
        newline_before_prompt: false,
        newline_after_prompt: false,
    };
    const BEFORE_ONLY: Spacing = Spacing {
        newline_before_prompt: true,
        newline_after_prompt: false,
    };
    const AFTER_ONLY: Spacing = Spacing {
        newline_before_prompt: false,
        newline_after_prompt: true,
    };

    /// Replay a sequence from a fresh console, returning what each action emitted.
    fn run(spacing: Spacing, actions: &[Action]) -> Vec<Emit> {
        let mut state = State::new();
        actions
            .iter()
            .map(|action| {
                let (emit, next) = step(state, spacing, *action);
                state = next;
                emit
            })
            .collect()
    }

    fn blanks(emit: &Emit) -> usize {
        usize::from(emit.after_prompt_blank)
            + usize::from(emit.separator_blank)
            + usize::from(emit.before_prompt_blank)
    }

    fn total_blanks(emits: &[Emit]) -> usize {
        emits.iter().map(blanks).sum()
    }

    /// One typed line, answered by a turn that streams text: the shape every other case is a
    /// variation on.
    fn plain_turn() -> Vec<Action> {
        vec![
            Action::OpenEpisode(RowState::Empty),
            Action::Block(BlockKind::Text),
            Action::CloseEpisode,
        ]
    }

    #[test]
    fn a_turn_is_bracketed_once_on_each_side() {
        let emits = run(BOTH, &plain_turn());
        assert!(
            emits[1].after_prompt_blank,
            "the blank fires before the text"
        );
        assert!(
            !emits[1].separator_blank,
            "nothing precedes the text to separate it from"
        );
        assert!(emits[2].before_prompt_blank);
        assert_eq!(total_blanks(&emits), 2);
    }

    #[test]
    fn each_flag_removes_exactly_its_own_blank() {
        assert_eq!(total_blanks(&run(BOTH, &plain_turn())), 2);
        assert_eq!(total_blanks(&run(NEITHER, &plain_turn())), 0);

        let before = run(BEFORE_ONLY, &plain_turn());
        assert!(!before[1].after_prompt_blank);
        assert!(before[2].before_prompt_blank);

        let after = run(AFTER_ONLY, &plain_turn());
        assert!(after[1].after_prompt_blank);
        assert!(!after[2].before_prompt_blank);
    }

    /// The regression that made `newline_after_prompt = false` do nothing from the second turn on:
    /// the block machine stopped being reset when the blank was not printed, so the next episode's
    /// first tool indicator saw the previous episode's text and asked for a separator.
    #[test]
    fn a_disabled_opening_blank_still_resets_the_block_machine() {
        for spacing in [NEITHER, BEFORE_ONLY] {
            let emits = run(spacing, &[
                Action::OpenEpisode(RowState::Empty),
                Action::Block(BlockKind::Text),
                Action::CloseEpisode,
                Action::OpenEpisode(RowState::Empty),
                Action::Block(BlockKind::ToolIndicator(ToolParams::Summary)),
                Action::CloseEpisode,
            ]);
            assert!(
                !emits[4].separator_blank,
                "a new episode's first block has nothing above it to separate from",
            );
            assert!(!emits[4].after_prompt_blank);
        }
    }

    /// Every path that would otherwise be bracketed by whoever happened to notice: a turn that
    /// failed before it started, a slash command answering without a turn, a command that ran
    /// several turns. All of them are one episode with one bracket.
    #[test]
    fn every_episode_is_bracketed_the_same_however_it_answered() {
        let error_only = run(BOTH, &[
            Action::OpenEpisode(RowState::Empty),
            Action::Block(BlockKind::Chrome),
            Action::CloseEpisode,
        ]);
        assert_eq!(total_blanks(&error_only), 2);

        let foreign = run(BOTH, &[
            Action::OpenEpisode(RowState::Empty),
            Action::AnnounceForeign,
            Action::CloseEpisode,
        ]);
        assert_eq!(total_blanks(&foreign), 2);

        // Two turns fired by one wake. The gap between them is the blocks' own separator, not a
        // second pair of prompt brackets.
        let two_turns = run(BOTH, &[
            Action::OpenEpisode(RowState::PromptParked),
            Action::Block(BlockKind::Text),
            Action::Block(BlockKind::Text),
            Action::CloseEpisode,
        ]);
        assert!(two_turns[1].after_prompt_blank);
        assert!(two_turns[3].before_prompt_blank);
        assert_eq!(total_blanks(&two_turns), 2);
    }

    /// An episode that prints nothing gets no blank lines, and hands the row back as it found it.
    /// The second half is what stops a scheduler wake with nothing due from leaving a duplicate
    /// prompt on screen.
    #[test]
    fn an_episode_that_prints_nothing_leaves_no_trace() {
        for spacing in [BOTH, NEITHER] {
            let emits = run(spacing, &[
                Action::OpenEpisode(RowState::PromptParked),
                Action::CloseEpisode,
            ]);
            assert_eq!(total_blanks(&emits), 0);
            assert_eq!(
                emits[1].settle,
                Settle::Erase,
                "the stale prompt has to go, or reedline paints a second one below it",
            );
        }
    }

    /// The same wake, but something did run: the prompt row is real output's backdrop and must be
    /// kept.
    #[test]
    fn a_wake_that_runs_something_keeps_the_prompt_it_broke_out_of() {
        let emits = run(BOTH, &[
            Action::OpenEpisode(RowState::PromptParked),
            Action::Block(BlockKind::Text),
            Action::CloseEpisode,
        ]);
        assert_eq!(emits[1].settle, Settle::Terminate);
        assert_eq!(emits[2].settle, Settle::Nothing);
        assert_eq!(total_blanks(&emits), 2);
    }

    /// A blank line only makes a gap from column zero. An in-place status line is discarded rather
    /// than terminated, because its row is not content.
    #[test]
    fn a_transient_row_is_erased_by_whatever_prints_next() {
        let emits = run(BOTH, &[
            Action::OpenEpisode(RowState::Empty),
            Action::OpenTransient,
            Action::Block(BlockKind::Text),
            Action::CloseEpisode,
        ]);
        assert_eq!(emits[2].settle, Settle::Erase);
        assert!(
            emits[1].after_prompt_blank && !emits[2].after_prompt_blank,
            "the opening blank is spent once, by the first thing on screen",
        );

        // Left drawn at the end of an episode it is still not content, so it must not survive into
        // the prompt.
        let leftover = run(BOTH, &[
            Action::OpenEpisode(RowState::Empty),
            Action::Block(BlockKind::Text),
            Action::OpenTransient,
            Action::CloseEpisode,
        ]);
        assert_eq!(leftover[3].settle, Settle::Erase);
    }

    /// The shutdown path closes the last episode and then still has things to say, so closing an
    /// already-closed episode has to be free rather than a second closing blank.
    #[test]
    fn closing_an_episode_twice_closes_it_once() {
        let emits = run(BOTH, &[
            Action::OpenEpisode(RowState::Empty),
            Action::Block(BlockKind::Text),
            Action::CloseEpisode,
            Action::CloseEpisode,
        ]);
        assert!(emits[2].before_prompt_blank);
        assert!(!emits[3].before_prompt_blank);
        assert_eq!(emits[3].settle, Settle::Nothing);
    }

    /// Output the console cannot see still has to start on a row of its own. Enforced here rather
    /// than by hand at each prompt, which is how the MCP elicitation path goes without: a server's
    /// progress line parks the cursor mid-row, and meka's own chrome continuing that row is what
    /// makes a forged prompt possible.
    #[test]
    fn foreign_output_settles_the_row_it_lands_on() {
        let after_wake = run(BOTH, &[
            Action::OpenEpisode(RowState::PromptParked),
            Action::AnnounceForeign,
        ]);
        assert_eq!(
            after_wake[1].settle,
            Settle::Terminate,
            "the prompt row is content: move past it",
        );

        let after_status_line = run(BOTH, &[
            Action::OpenEpisode(RowState::Empty),
            Action::OpenTransient,
            Action::AnnounceForeign,
        ]);
        assert_eq!(
            after_status_line[2].settle,
            Settle::Erase,
            "a status line is not content: replace it",
        );
    }

    /// A status line drawn straight after a scheduler wake still has the drawn prompt beneath the
    /// cursor, and that row is content: it has to be moved past, not overwritten.
    #[test]
    fn a_transient_line_moves_past_a_parked_prompt_rather_than_over_it() {
        let emits = run(BOTH, &[
            Action::OpenEpisode(RowState::PromptParked),
            Action::OpenTransient,
            Action::CloseEpisode,
        ]);
        assert_eq!(emits[1].settle, Settle::Terminate);
        assert!(
            emits[1].after_prompt_blank,
            "an indicator is the episode's first output like anything else",
        );
        assert_eq!(
            emits[2].settle,
            Settle::Erase,
            "and the indicator itself is still not content",
        );
    }

    #[test]
    fn committing_a_transient_line_keeps_it_and_erasing_it_does_not() {
        let committed = run(BOTH, &[
            Action::OpenEpisode(RowState::Empty),
            Action::OpenTransient,
            Action::CommitTransient,
        ]);
        assert_eq!(committed[2].settle, Settle::Terminate);

        let erased = run(BOTH, &[
            Action::OpenEpisode(RowState::Empty),
            Action::OpenTransient,
            Action::EraseTransient,
        ]);
        assert_eq!(erased[2].settle, Settle::Erase);

        // Neither writes anything when no indicator is drawn, which is what lets the frontend call
        // them from a dispatch that cannot know.
        let absent = run(BOTH, &[
            Action::OpenEpisode(RowState::Empty),
            Action::CommitTransient,
            Action::EraseTransient,
        ]);
        assert_eq!(absent[1].settle, Settle::Nothing);
        assert_eq!(absent[2].settle, Settle::Nothing);
    }

    /// The opening blank fires before the episode's *first* output whatever that turns out to be,
    /// so a session-id notice can no longer print above it.
    #[test]
    fn the_opening_blank_precedes_whatever_prints_first() {
        let emits = run(BOTH, &[
            Action::OpenEpisode(RowState::Empty),
            Action::Block(BlockKind::Chrome),
            Action::Block(BlockKind::Text),
            Action::CloseEpisode,
        ]);
        assert!(emits[1].after_prompt_blank);
        assert!(!emits[2].after_prompt_blank);
    }

    /// Intra-episode separators are the block machine's, not the prompt brackets', so they are
    /// unaffected by either flag.
    #[test]
    fn block_separators_do_not_follow_the_prompt_flags() {
        for spacing in [BOTH, NEITHER, BEFORE_ONLY, AFTER_ONLY] {
            let emits = run(spacing, &[
                Action::OpenEpisode(RowState::Empty),
                Action::Block(BlockKind::ToolIndicator(ToolParams::Summary)),
                Action::Block(BlockKind::Text),
                Action::CloseEpisode,
            ]);
            assert!(
                emits[2].separator_blank,
                "text after a tool indicator is always separated from it",
            );
            let adjacent = run(spacing, &[
                Action::OpenEpisode(RowState::Empty),
                Action::Block(BlockKind::ToolIndicator(ToolParams::Summary)),
                Action::Block(BlockKind::ToolIndicator(ToolParams::Summary)),
                Action::CloseEpisode,
            ]);
            assert!(
                !adjacent[2].separator_blank,
                "a run of summary indicators reads as a list of steps",
            );
        }
    }

    /// The todo list paints its own surrounding blanks, so it asks for no separator and leaves
    /// nothing for the next block to double.
    #[test]
    fn a_todo_list_brings_its_own_separation() {
        let emits = run(BOTH, &[
            Action::OpenEpisode(RowState::Empty),
            Action::Block(BlockKind::ToolIndicator(ToolParams::Summary)),
            Action::Block(BlockKind::TodoList),
            Action::Block(BlockKind::Text),
            Action::CloseEpisode,
        ]);
        assert!(!emits[2].separator_blank);
        assert!(!emits[3].separator_blank);
    }

    /// `printed` means "the episode did something", not "a blank was written", or turning the
    /// spacing off would also turn off the closing bracket's condition.
    #[test]
    fn output_is_recorded_even_when_no_blank_is_printed() {
        let mut state = State::new();
        for action in [
            Action::OpenEpisode(RowState::Empty),
            Action::Block(BlockKind::Text),
        ] {
            let (_, next) = step(state, NEITHER, action);
            state = next;
        }
        assert!(state.printed());
    }
}
