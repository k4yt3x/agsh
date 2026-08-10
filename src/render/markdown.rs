//! CommonMark to minimad translation.
//!
//! termimad renders markdown with minimad, whose parser understands a small dialect rather than
//! CommonMark: only `*`/`**` for emphasis (never `_`/`__`), only `* ` for list items (never `-` or
//! `+`), no ordered lists, no links, no escapes. Feeding it markdown as models actually write it
//! meant either accepting mangled output or rewriting the text to fit the dialect, and rewriting is
//! guesswork that damages content: `snake_case` is not emphasis, and `    - x` may be an indented
//! code block rather than a list.
//!
//! So meka does the parsing. A real CommonMark parser produces the document structure, this module
//! translates it into the AST minimad would have produced, and termimad still owns everything it is
//! good at: wrapping, table layout, and the skin. Markers never survive into the rendered text,
//! because the text handed to termimad carries no markup at all.
//!
//! The AST borrows its strings ([`Compound::src`] is a `&str`), so translation happens in two
//! passes: [`MarkdownDoc::parse`] collects owned lines, and [`MarkdownDoc::to_minimad`] builds the
//! borrowed view. That keeps the whole thing in safe Rust.

use pulldown_cmark::{
    Alignment as CmarkAlignment, Event, HeadingLevel, Options, Parser, Tag, TagEnd,
};
use termimad::minimad::{
    Alignment, Composite, CompositeStyle, Compound, Line, TableRow, TableRule, Text,
};

/// A parsed markdown document, owning the text its minimad view borrows.
pub(super) struct MarkdownDoc {
    lines: Vec<OwnedLine>,
}

/// Owned mirror of minimad's [`Line`], holding `String`s where it holds `&str`.
enum OwnedLine {
    Composite {
        style: CompositeStyle,
        compounds: Vec<OwnedCompound>,
    },
    TableRow(Vec<Vec<OwnedCompound>>),
    TableRule(Vec<Alignment>),
    HorizontalRule,
}

/// Owned mirror of minimad's [`Compound`]: a run of text plus the styles covering it.
#[derive(Clone, Default)]
struct OwnedCompound {
    text: String,
    bold: bool,
    italic: bool,
    code: bool,
    strikeout: bool,
}

impl OwnedCompound {
    fn to_minimad(&self) -> Compound<'_> {
        let mut compound = Compound::raw_str(&self.text);
        compound.bold = self.bold;
        compound.italic = self.italic;
        compound.code = self.code;
        compound.strikeout = self.strikeout;
        compound
    }
}

fn to_composite(style: CompositeStyle, compounds: &[OwnedCompound]) -> Composite<'_> {
    Composite {
        style,
        compounds: compounds.iter().map(OwnedCompound::to_minimad).collect(),
    }
}

impl MarkdownDoc {
    pub(super) fn parse(markdown: &str) -> Self {
        let mut options = Options::empty();
        options.insert(Options::ENABLE_TABLES);
        options.insert(Options::ENABLE_STRIKETHROUGH);
        options.insert(Options::ENABLE_TASKLISTS);
        let mut builder = Builder::default();
        for event in Parser::new_ext(markdown, options) {
            builder.handle(event);
        }
        builder.finish();
        let mut lines = builder.lines;
        // The streaming flush hands over one settled block at a time, so whether this document is
        // followed by a blank line is knowable only from the source: a chunk cut at a blank line
        // ends with one and keeps its separator, while the tail drained at end-of-turn does not and
        // would otherwise render a stray empty line after the last block.
        if !markdown.ends_with("\n\n") {
            while matches!(lines.last(), Some(OwnedLine::Composite { compounds, .. }) if compounds.is_empty())
            {
                lines.pop();
            }
        }
        Self { lines }
    }

    /// Borrowed minimad view, ready for `FmtText::from_text`.
    pub(super) fn to_minimad(&self) -> Text<'_> {
        Text {
            lines: self
                .lines
                .iter()
                .map(|line| match line {
                    OwnedLine::Composite { style, compounds } => {
                        Line::Normal(to_composite(*style, compounds))
                    }
                    OwnedLine::TableRow(cells) => Line::TableRow(TableRow {
                        cells: cells
                            .iter()
                            .map(|cell| to_composite(CompositeStyle::Paragraph, cell))
                            .collect(),
                    }),
                    OwnedLine::TableRule(alignments) => Line::TableRule(TableRule {
                        cells: alignments.clone(),
                    }),
                    OwnedLine::HorizontalRule => Line::HorizontalRule,
                })
                .collect(),
        }
    }
}

/// Nesting counters rather than booleans: `**a *b* c**` closes the inner emphasis without ending
/// the outer one, so a boolean would drop the rest of the bold run.
#[derive(Default)]
struct InlineStyle {
    bold: usize,
    italic: usize,
    code: usize,
    strikeout: usize,
}

/// What kind of list a level is, and how far its ordered counter has advanced.
enum ListLevel {
    Bullet,
    /// Ordered lists have no minimad equivalent, so the number is rendered as text and this tracks
    /// the next one to emit.
    Ordered(u64),
}

struct Builder {
    lines: Vec<OwnedLine>,
    /// Inline runs accumulated for the composite currently being built.
    compounds: Vec<OwnedCompound>,
    style: CompositeStyle,
    inline: InlineStyle,
    lists: Vec<ListLevel>,
    /// Set while inside a blockquote, so paragraphs within it keep the quote styling that minimad
    /// expresses per line rather than as a container.
    quote_depth: usize,
    table: Option<TableState>,
    /// Destination of the link currently being built, appended after its text.
    link_target: Option<String>,
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            lines: Vec::new(),
            compounds: Vec::new(),
            style: CompositeStyle::Paragraph,
            inline: InlineStyle::default(),
            lists: Vec::new(),
            quote_depth: 0,
            table: None,
            link_target: None,
        }
    }
}

#[derive(Default)]
struct TableState {
    alignments: Vec<Alignment>,
    in_head: bool,
    row: Vec<Vec<OwnedCompound>>,
    cell: Vec<OwnedCompound>,
}

impl Builder {
    fn handle(&mut self, event: Event<'_>) {
        match event {
            Event::Start(tag) => self.start(tag),
            Event::End(tag) => self.end(tag),
            Event::Text(text) => self.push_text(&text),
            Event::Code(text) => {
                self.inline.code += 1;
                self.push_text(&text);
                self.inline.code -= 1;
            }
            // A soft break is a newline inside one paragraph. Joining with a space is what lets
            // termimad re-wrap the paragraph to the terminal width instead of preserving whatever
            // width the model happened to write at.
            Event::SoftBreak => self.push_text(" "),
            Event::HardBreak => self.flush_composite(),
            Event::Rule => {
                self.flush_composite();
                self.lines.push(OwnedLine::HorizontalRule);
                self.push_blank_line();
            }
            Event::TaskListMarker(checked) => {
                self.push_text(if checked { "[x] " } else { "[ ] " });
            }
            // No minimad equivalent, and dropping them would lose content the model wrote. Raw
            // text is the honest rendering.
            Event::Html(text) | Event::InlineHtml(text) => self.push_text(&text),
            Event::InlineMath(text) | Event::DisplayMath(text) => self.push_text(&text),
            Event::FootnoteReference(text) => self.push_text(&format!("[{}]", text)),
        }
    }

    fn start(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => self.style = self.block_style(),
            Tag::Heading { level, .. } => self.style = CompositeStyle::Header(header_depth(level)),
            Tag::BlockQuote(_) => {
                self.quote_depth += 1;
                self.style = CompositeStyle::Quote;
            }
            Tag::CodeBlock(_) => self.style = CompositeStyle::Code,
            Tag::List(first) => {
                // A nested list opens while the parent item's own text is still pending. Without
                // flushing, that text is swallowed into the first child item.
                self.flush_composite();
                self.lists.push(match first {
                    Some(start) => ListLevel::Ordered(start),
                    None => ListLevel::Bullet,
                });
            }
            Tag::Item => self.start_item(),
            Tag::Table(alignments) => {
                self.flush_composite();
                self.table = Some(TableState {
                    alignments: alignments.iter().map(convert_alignment).collect(),
                    ..TableState::default()
                });
            }
            Tag::TableHead => {
                if let Some(table) = &mut self.table {
                    table.in_head = true;
                }
            }
            Tag::TableRow | Tag::TableCell => {}
            Tag::Emphasis => self.inline.italic += 1,
            Tag::Strong => self.inline.bold += 1,
            Tag::Strikethrough => self.inline.strikeout += 1,
            // Terminals can't follow a link, so the destination is rendered alongside the text.
            Tag::Link { dest_url, .. } | Tag::Image { dest_url, .. } => {
                self.link_target = Some(dest_url.to_string());
            }
            _ => {}
        }
    }

    fn end(&mut self, tag: TagEnd) {
        match tag {
            // An item ends without a separator: consecutive items belong to one list.
            TagEnd::Item => self.flush_composite(),
            TagEnd::Paragraph | TagEnd::Heading(_) | TagEnd::CodeBlock => self.end_block(),
            TagEnd::BlockQuote(_) => {
                self.flush_composite();
                self.quote_depth = self.quote_depth.saturating_sub(1);
                self.style = self.block_style();
                self.end_block();
            }
            TagEnd::List(_) => {
                self.lists.pop();
                self.end_block();
            }
            TagEnd::Table => {
                self.table = None;
                self.end_block();
            }
            TagEnd::TableHead => {
                // pulldown-cmark delimits the header row with `TableHead` itself and emits no
                // `TableRow` around it, so it has to be closed here; otherwise its cells stay
                // pending and merge into the first body row, producing one row of doubled width.
                self.flush_table_row();
                if let Some(table) = &mut self.table {
                    table.in_head = false;
                    let alignments = table.alignments.clone();
                    self.lines.push(OwnedLine::TableRule(alignments));
                }
            }
            TagEnd::TableRow => self.flush_table_row(),
            TagEnd::TableCell => {
                if let Some(table) = &mut self.table {
                    let cell = std::mem::take(&mut table.cell);
                    table.row.push(cell);
                }
            }
            TagEnd::Emphasis => self.inline.italic = self.inline.italic.saturating_sub(1),
            TagEnd::Strong => self.inline.bold = self.inline.bold.saturating_sub(1),
            TagEnd::Strikethrough => {
                self.inline.strikeout = self.inline.strikeout.saturating_sub(1);
            }
            TagEnd::Link | TagEnd::Image => {
                if let Some(target) = self.link_target.take() {
                    let already_shown = self
                        .compounds
                        .last()
                        .is_some_and(|last| last.text.trim() == target);
                    if !already_shown && !target.is_empty() {
                        self.push_text(&format!(" ({})", target));
                    }
                }
            }
            _ => {}
        }
    }

    /// The style a fresh block takes: inside a blockquote everything stays quoted, because minimad
    /// marks quoting per line rather than wrapping a range of lines.
    fn block_style(&self) -> CompositeStyle {
        if self.quote_depth > 0 {
            CompositeStyle::Quote
        } else {
            CompositeStyle::Paragraph
        }
    }

    fn start_item(&mut self) {
        // Close the previous item before adopting this one's style, so sibling items don't merge.
        self.flush_composite();
        // minimad caps nesting at three levels; deeper items clamp rather than disappear.
        let depth = self.lists.len().saturating_sub(1).min(3) as u8;
        match self.lists.last_mut() {
            Some(ListLevel::Ordered(next)) => {
                let number = *next;
                *next += 1;
                // No ordered-list concept in minimad, so the marker is rendered as text. Using a
                // bullet composite as well would show "• 1." for every item, and dropping the
                // number would renumber the model's instructions.
                self.style = CompositeStyle::Paragraph;
                let indent = "  ".repeat(depth as usize);
                self.push_text(&format!("{}{}. ", indent, number));
            }
            _ => self.style = CompositeStyle::ListItem(depth),
        }
    }

    fn push_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        if let Some(table) = &mut self.table {
            push_run(&mut table.cell, text, &self.inline);
            return;
        }
        push_run(&mut self.compounds, text, &self.inline);
    }

    /// Close a top-level block and separate it from the next one.
    ///
    /// minimad has no notion of a block: it renders a list of lines, so the blank line between two
    /// paragraphs has to be an explicit empty line or the whole document renders as one dense
    /// slab. Only at the outermost level, since items within a list and rows within a table are
    /// meant to be adjacent.
    fn end_block(&mut self) {
        self.flush_composite();
        if self.lists.is_empty() && self.quote_depth == 0 && self.table.is_none() {
            self.push_blank_line();
        }
    }

    fn push_blank_line(&mut self) {
        // Never two in a row: markdown collapses runs of blank lines, and doubling them up here
        // would spread a document out further every time a block nested.
        if matches!(self.lines.last(), Some(OwnedLine::Composite { compounds, .. }) if compounds.is_empty())
        {
            return;
        }
        self.lines.push(OwnedLine::Composite {
            style: CompositeStyle::Paragraph,
            compounds: Vec::new(),
        });
    }

    fn flush_composite(&mut self) {
        if self.compounds.is_empty() {
            return;
        }
        let compounds = std::mem::take(&mut self.compounds);
        self.lines.push(OwnedLine::Composite {
            style: self.style,
            compounds,
        });
    }

    fn flush_table_row(&mut self) {
        let Some(table) = &mut self.table else {
            return;
        };
        if table.row.is_empty() {
            return;
        }
        let row = std::mem::take(&mut table.row);
        self.lines.push(OwnedLine::TableRow(row));
    }

    fn finish(&mut self) {
        self.flush_composite();
    }
}

/// Append `text` to `compounds`, merging into the previous run when the styles match so a
/// paragraph doesn't become one compound per event.
fn push_run(compounds: &mut Vec<OwnedCompound>, text: &str, inline: &InlineStyle) {
    let bold = inline.bold > 0;
    let italic = inline.italic > 0;
    let code = inline.code > 0;
    let strikeout = inline.strikeout > 0;
    if let Some(last) = compounds.last_mut()
        && last.bold == bold
        && last.italic == italic
        && last.code == code
        && last.strikeout == strikeout
    {
        last.text.push_str(text);
        return;
    }
    compounds.push(OwnedCompound {
        text: text.to_string(),
        bold,
        italic,
        code,
        strikeout,
    });
}

fn header_depth(level: HeadingLevel) -> u8 {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

fn convert_alignment(alignment: &CmarkAlignment) -> Alignment {
    match alignment {
        CmarkAlignment::None => Alignment::Unspecified,
        CmarkAlignment::Left => Alignment::Left,
        CmarkAlignment::Center => Alignment::Center,
        CmarkAlignment::Right => Alignment::Right,
    }
}
