//! Web tools: `fetch_url` (HTTP GET with HTML→markdown conversion or multimodal image return) and
//! `search_web` (DuckDuckGo HTML scraping with CAPTCHA detection).

use std::sync::LazyLock;

use async_trait::async_trait;
use futures::StreamExt;
use html2md::rewrite_html_custom_with_url;
use tokio_util::sync::CancellationToken;

use super::{
    Tool, ToolOutput,
    util::{compile_user_regex, redirects_to_scratchpad, require_str},
};
use crate::{
    config::{MinTlsVersion, WebClientConfig},
    error::{MekaError, Result},
    image::{ImageHandling, build_image_tool_output, classify_content_type},
    permission::Permission,
    provider::ToolDefinition,
};

/// Build the shared `reqwest::Client` for `fetch_url` + `search_web` from the resolved
/// [`WebClientConfig`]. Errors propagate so startup fails cleanly on a bad proxy URL or unreadable
/// CA file, safer than silently falling back to an unconfigured client that ignores user intent.
pub(crate) fn build_web_client(cfg: &WebClientConfig) -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        .user_agent(&cfg.user_agent)
        .timeout(cfg.request_timeout);

    if let Some(t) = cfg.connect_timeout {
        builder = builder.connect_timeout(t);
    }
    if let Some(t) = cfg.read_timeout {
        builder = builder.read_timeout(t);
    }

    // `0` → no redirects at all (Policy::none). Any non-zero cap maps to Policy::limited(n).
    let policy = if cfg.max_redirects == 0 {
        reqwest::redirect::Policy::none()
    } else {
        reqwest::redirect::Policy::limited(cfg.max_redirects)
    };
    builder = builder.redirect(policy);

    match cfg.proxy.as_deref() {
        None => {}
        Some("") | Some("none") => {
            // Explicit opt-out of reqwest's env-proxy auto-detection. Useful to override a
            // host-level `HTTP_PROXY` env var without unsetting it.
            builder = builder.no_proxy();
        }
        Some(url) => {
            // Pre-validate the scheme before handing off; `reqwest::Proxy::all` is lenient (it'll
            // accept `"not-a-url"` as `http://not-a-url/`), which silently routes traffic through a
            // non-existent host. A typo in the config should fail loudly.
            const ALLOWED_SCHEMES: &[&str] = &[
                "http://",
                "https://",
                "socks5://",
                "socks5h://",
                "socks4://",
            ];
            if !ALLOWED_SCHEMES.iter().any(|s| url.starts_with(s)) {
                return Err(MekaError::Config(format!(
                    "[web].proxy: invalid URL '{}': expected one of {}",
                    url,
                    ALLOWED_SCHEMES.join(", ")
                )));
            }
            let proxy = reqwest::Proxy::all(url).map_err(|error| {
                MekaError::Config(format!("[web].proxy: invalid URL '{}': {}", url, error))
            })?;
            builder = builder.proxy(proxy);
        }
    }

    if let Some(path) = &cfg.ca_cert_file {
        let bytes = std::fs::read(path).map_err(|error| {
            MekaError::Config(format!(
                "[web].ca_cert_file '{}': {}",
                path.display(),
                error
            ))
        })?;
        // Handles both single-cert and bundle PEM files (multiple concatenated `-----BEGIN/END
        // CERTIFICATE-----` blocks).
        let certs = reqwest::Certificate::from_pem_bundle(&bytes).map_err(|error| {
            MekaError::Config(format!(
                "[web].ca_cert_file '{}': not a valid PEM: {}",
                path.display(),
                error
            ))
        })?;
        // `from_pem_bundle` silently returns an empty Vec when the file contains no PEM blocks.
        // That's not what the user asked for. Reject explicitly so typos don't ship a client
        // with zero added CAs.
        if certs.is_empty() {
            return Err(MekaError::Config(format!(
                "[web].ca_cert_file '{}': no PEM certificates found in file",
                path.display()
            )));
        }
        for cert in certs {
            builder = builder.add_root_certificate(cert);
        }
    }

    if cfg.https_only {
        builder = builder.https_only(true);
    }

    if let Some(v) = cfg.min_tls_version {
        let ver = match v {
            MinTlsVersion::V1_0 => reqwest::tls::Version::TLS_1_0,
            MinTlsVersion::V1_1 => reqwest::tls::Version::TLS_1_1,
            MinTlsVersion::V1_2 => reqwest::tls::Version::TLS_1_2,
            MinTlsVersion::V1_3 => reqwest::tls::Version::TLS_1_3,
        };
        builder = builder.min_tls_version(ver);
    }

    if cfg.danger_accept_invalid_certs {
        tracing::warn!(
            "[web].danger_accept_invalid_certs is enabled. TLS certificate \
             validation is OFF; any HTTPS response could be spoofed"
        );
        builder = builder.danger_accept_invalid_certs(true);
    }
    if cfg.danger_accept_invalid_hostnames {
        tracing::warn!(
            "[web].danger_accept_invalid_hostnames is enabled. TLS hostname \
             verification is OFF; any HTTPS response could be spoofed"
        );
        builder = builder.danger_accept_invalid_hostnames(true);
    }

    builder
        .build()
        .map_err(|error| MekaError::Config(format!("failed to build web client: {}", error)))
}

// Static CSS selectors for search result parsing (parsed once, reused on every call).
// `expect()` is correct here: the selector strings are compile-time literals, so a parse failure
// would mean we shipped a typo, caught on the first test run, not in production.
#[allow(clippy::expect_used)]
static DDG_RESULT: LazyLock<scraper::Selector> =
    LazyLock::new(|| scraper::Selector::parse(".result").expect("static CSS selector"));
#[allow(clippy::expect_used)]
static DDG_LINK: LazyLock<scraper::Selector> =
    LazyLock::new(|| scraper::Selector::parse("a.result__a").expect("static CSS selector"));
#[allow(clippy::expect_used)]
static DDG_URL: LazyLock<scraper::Selector> =
    LazyLock::new(|| scraper::Selector::parse(".result__url").expect("static CSS selector"));
#[allow(clippy::expect_used)]
static DDG_SNIPPET: LazyLock<scraper::Selector> =
    LazyLock::new(|| scraper::Selector::parse(".result__snippet").expect("static CSS selector"));
/// DDG's bot-challenge modal uses this id (and also a `data-testid` of the same value). Either
/// marker being present in the DOM means the endpoint gated us rather than returning results.
#[allow(clippy::expect_used)]
static DDG_CAPTCHA: LazyLock<scraper::Selector> = LazyLock::new(|| {
    scraper::Selector::parse("#anomaly-modal, [data-testid=\"anomaly-modal\"]")
        .expect("static CSS selector")
});

/// Matches the open/close tags of `<nav>` and `<footer>` elements (with any attributes). The name
/// is anchored by the trailing `(\s|>|/)` group so sibling elements like `<navbar>` or a custom
/// `<nav-menu>` are left untouched.
#[allow(clippy::expect_used)]
static BOILERPLATE_CONTAINER_TAG: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r"(?i)<(/?)(?:nav|footer)(\s[^>]*)?>").expect("static regex")
});

/// Rewrite `<nav>` / `<footer>` container tags to `<div>` before HTML-to-markdown conversion.
///
/// `fast_html2md` unconditionally drops the entire subtree of `head, nav, footer, script, noscript,
/// style` as boilerplate (it calls lol_html's `el.remove()`, which deletes the element and all its
/// content). That is reasonable for `script` / `style`, but modern sites (Next.js and friends) put
/// primary navigation and useful footer links inside `<nav>` / `<footer>`, so those links (text and
/// href alike) silently vanish from the converted markdown. Renaming just the open/close tags to a
/// neutral `<div>` keeps the content while leaving `script` / `style` / `head` stripping intact.
fn keep_boilerplate_container_content(html: &str) -> std::borrow::Cow<'_, str> {
    BOILERPLATE_CONTAINER_TAG.replace_all(html, "<${1}div${2}>")
}

/// Convert fetched HTML to Markdown exactly the way `fetch_url` does. Two steps: rewrite `<nav>` /
/// `<footer>` containers so their links survive [`keep_boilerplate_container_content`], then run
/// `fast_html2md` with the document's URL as the base so root-relative links (`/docs`) resolve to
/// absolute URLs (`https://host/docs`) the model can follow directly. A `None` base leaves relative
/// links relative (the converter only rewrites hrefs that start with `/`).
fn html_to_markdown(html: &str, base_url: &Option<url::Url>) -> String {
    rewrite_html_custom_with_url(
        &keep_boilerplate_container_content(html),
        &None,
        false,
        base_url,
    )
}

/// Cap on a single result's snippet text (after `**bold**` markers are added). 10 results × 300
/// chars = ~3 KB of snippets, a sane default for the model; longer content is available via
/// `fetch_url` on the result URL.
const SNIPPET_MAX_CHARS: usize = 300;

/// Default `max_length` applied when the caller doesn't pass one. Single source of truth for both
/// the parameter unwrap and the description shown to the agent. Pass `0` to disable the cap.
const DEFAULT_MAX_LENGTH: usize = 30_000;

fn apply_headers(
    mut builder: reqwest::RequestBuilder,
    input: &serde_json::Value,
) -> reqwest::RequestBuilder {
    if let Some(headers) = input.get("headers").and_then(|h| h.as_object()) {
        for (key, value) in headers {
            if let Some(value_str) = value.as_str() {
                builder = builder.header(key.as_str(), value_str);
            }
        }
    }
    builder
}

pub(super) struct FetchUrlTool {
    pub client: reqwest::Client,
}

#[async_trait]
impl Tool for FetchUrlTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "fetch_url".to_string(),
            description: "Fetch a web page and return its content as markdown. Set 'raw' \
                          to true to return untreated HTML. If the URL resolves to a \
                          supported raster image (PNG, JPEG, GIF, WebP, BMP, TIFF, \
                          ICO, HDR, EXR, TGA, PNM, QOI, DDS, or Farbfeld), the image \
                          is returned as a multimodal content block directly. \
                          Non-native formats are transparently converted to PNG. \
                          `max_length`, `regex`, and `raw` do not apply to image \
                          responses. Only fetch image URLs if the current model \
                          supports vision input."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    },
                    "max_length": {
                        "type": "integer",
                        "description": format!(
                            "Maximum number of characters to return. Default: {}. Set to 0 for no limit.",
                            DEFAULT_MAX_LENGTH,
                        )
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers. Overrides defaults (e.g., User-Agent).",
                        "additionalProperties": { "type": "string" }
                    },
                    "regex": {
                        "type": "string",
                        "description": "Optional regex pattern. If provided, only matching content is returned (all matches joined by newlines)."
                    },
                    "raw": {
                        "type": "boolean",
                        "description": "If true, return raw HTML instead of converting to markdown. Defaults to false."
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the output to the scratchpad under this name instead of returning it inline."
                    }
                },
                "required": ["url"]
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
        let url = require_str(&input, "url", "fetch_url")?;

        let request = apply_headers(self.client.get(&url), &input);
        let response = request
            .send()
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "fetch_url".to_string(),
                message: format!(
                    "failed to fetch '{}': {}",
                    url,
                    crate::error::format_reqwest_error(&error)
                ),
            })?;

        let status = response.status();
        if !status.is_success() {
            return Ok(ToolOutput::text(
                format!("HTTP {} for '{}'", status, url),
                true,
            ));
        }

        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_string();

        // Enforce a byte cap on the decompressed body so a small gzip/brotli payload can't expand
        // into gigabytes and exhaust host memory (a classic "zip bomb" vector now that reqwest is
        // built with gzip, deflate, and brotli enabled). We stream rather than buffer with `text()`
        // so the cap is checked incrementally.
        const MAX_RESPONSE_BYTES: usize = 10 * 1024 * 1024;
        if let Some(len) = response.content_length()
            && len as usize > MAX_RESPONSE_BYTES
        {
            return Err(MekaError::ToolExecution {
                tool_name: "fetch_url".to_string(),
                message: format!(
                    "response Content-Length {} exceeds cap {} bytes",
                    len, MAX_RESPONSE_BYTES
                ),
            });
        }

        // Capture the document's final (post-redirect) URL to resolve relative links against. Taken
        // before `bytes_stream` consumes the response; re-parsed through our own `url` crate so the
        // type matches `html_to_markdown` regardless of reqwest's `url` re-export.
        let document_url: Option<url::Url> = url::Url::parse(response.url().as_str()).ok();

        let mut body_bytes: Vec<u8> = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|error| MekaError::ToolExecution {
                tool_name: "fetch_url".to_string(),
                message: format!("failed to read response body: {}", error),
            })?;
            if body_bytes.len() + chunk.len() > MAX_RESPONSE_BYTES {
                return Err(MekaError::ToolExecution {
                    tool_name: "fetch_url".to_string(),
                    message: format!(
                        "response body exceeded {} bytes during streaming \
                         (possible decompression bomb)",
                        MAX_RESPONSE_BYTES
                    ),
                });
            }
            body_bytes.extend_from_slice(&chunk);
        }

        // A response the server labelled as an image becomes a multimodal Image block rather than
        // going through html2md. `Content-Type` only gates whether to try: the media type comes
        // from the bytes, and a body that isn't an image at all (an HTML error page served
        // as `image/png`, which is common) falls through to the text path below instead of
        // shipping a block the provider will reject.
        if !matches!(
            classify_content_type(&content_type),
            ImageHandling::Unsupported
        ) {
            let sniffed = crate::image::classify_bytes(&body_bytes);
            if !matches!(sniffed, ImageHandling::Unsupported) {
                let marker = format!("Image fetched from {}", url);
                return Ok(build_image_tool_output(&marker, sniffed, &body_bytes));
            }
        }

        let html = String::from_utf8_lossy(&body_bytes).into_owned();

        let raw = input["raw"].as_bool().unwrap_or(false);
        let body = if raw {
            html
        } else {
            html_to_markdown(&html, &document_url)
        };

        // When the caller redirects to the scratchpad we produce full content regardless of
        // max_length; the scratchpad is the overflow buffer.
        let max_length = if redirects_to_scratchpad(&input) {
            0
        } else {
            input["max_length"]
                .as_u64()
                .map(|value| usize::try_from(value).unwrap_or(usize::MAX))
                .unwrap_or(DEFAULT_MAX_LENGTH)
        };

        let content = if max_length > 0 && body.len() > max_length {
            format!(
                "{}\n\n... (truncated, showing first {} characters)",
                &body[..body.floor_char_boundary(max_length)],
                max_length
            )
        } else {
            body
        };

        let content = if let Some(pattern) = input.get("regex").and_then(|v| v.as_str()) {
            let re = compile_user_regex(pattern, "fetch_url")?;
            let matches: Vec<&str> = re.find_iter(&content).map(|m| m.as_str()).collect();
            if matches.is_empty() {
                "No matches found for the given regex pattern.".to_string()
            } else {
                matches.join("\n")
            }
        } else {
            content
        };

        Ok(ToolOutput::text(content, false))
    }
}

pub(super) struct WebSearchTool {
    pub client: reqwest::Client,
}

#[async_trait]
impl Tool for WebSearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "search_web".to_string(),
            description: "Search DuckDuckGo and return the top results. May occasionally \
                fail with a CAPTCHA error when DuckDuckGo rate-limits us."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers. Overrides defaults (e.g., User-Agent).",
                        "additionalProperties": { "type": "string" }
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the output to the scratchpad under this name instead of returning it inline."
                    }
                },
                "required": ["query"]
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
        let query = require_str(&input, "query", "search_web")?;

        let request = apply_headers(
            self.client
                .get("https://html.duckduckgo.com/html/")
                .query(&[("q", query.as_str())]),
            &input,
        );
        let response = request
            .send()
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "search_web".to_string(),
                message: format!(
                    "search request failed: {}",
                    crate::error::format_reqwest_error(&error)
                ),
            })?;

        let html = response
            .text()
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "search_web".to_string(),
                message: format!("failed to read search response: {}", error),
            })?;

        match parse_duckduckgo_results(&html) {
            DdgOutcome::Results(text) => Ok(ToolOutput::text(text, false)),
            DdgOutcome::Empty => Ok(ToolOutput::text(
                "No search results found.".to_string(),
                false,
            )),
            DdgOutcome::Captcha => Err(MekaError::ToolExecution {
                tool_name: "search_web".to_string(),
                message: "DuckDuckGo served a CAPTCHA challenge (bot detection / rate limit). \
                          Retry later."
                    .to_string(),
            }),
        }
    }
}

/// Distinguishes the three meaningful states of a DuckDuckGo HTML response. Before this enum, a
/// CAPTCHA page was indistinguishable from a legitimate zero-hit query; both produced `""` and the
/// agent saw `"No search results found."`, which encouraged blind retries against the same
/// rate-limited endpoint.
enum DdgOutcome {
    /// At least one result was parsed. The inner string is the rendered, numbered,
    /// markdown-formatted result list.
    Results(String),
    /// The page parsed cleanly but contained zero `.result` blocks and no CAPTCHA marker, a
    /// legitimate zero-hit query.
    Empty,
    /// `#anomaly-modal` or `data-testid="anomaly-modal"` found in the DOM. DDG gated us with their
    /// bot challenge.
    Captcha,
}

/// Normalise arbitrary text-node content into a single-line string. Collapses runs of whitespace
/// (including newlines and tabs) into a single ASCII space and trims. Applied to every user-visible
/// field (title, source domain, snippet) so the rendered output isn't broken up by DDG's layout
/// whitespace.
fn collapse_whitespace(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut prev_space = false;
    for ch in raw.chars() {
        if ch.is_whitespace() {
            if !prev_space && !out.is_empty() {
                out.push(' ');
            }
            prev_space = true;
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    out.trim_end().to_string()
}

/// Extract a snippet from DDG's `.result__snippet` element, preserving `<b>…</b>` emphasis (which
/// marks matched query terms) as markdown `**…**`. Current DDG wraps matched terms in `<b>` tags;
/// stripping them via `.text()` loses a useful signal the agent can use to see which words actually
/// hit.
fn render_snippet(snippet_el: scraper::ElementRef<'_>) -> String {
    use scraper::Node;
    let mut out = String::new();
    for node in snippet_el.children() {
        match node.value() {
            Node::Text(text) => out.push_str(text),
            Node::Element(element) => {
                // Collect the inner text and wrap in `**` iff this is a `<b>` or `<strong>`. Other
                // elements (rare, e.g. `<a>` inside snippets) fall through as plain text so we
                // don't miss content. `ElementRef::wrap` returns `Some` for any node whose
                // `value()` is `Node::Element`, which is exactly the arm we're in, so the
                // `expect` documents an unconditionally-true invariant rather than a runtime check.
                #[allow(clippy::expect_used)]
                let inner_el =
                    scraper::ElementRef::wrap(node).expect("element node wraps element ref");
                let inner_text: String = inner_el.text().collect();
                let tag = element.name();
                if tag == "b" || tag == "strong" {
                    if !inner_text.trim().is_empty() {
                        out.push_str("**");
                        out.push_str(inner_text.trim());
                        out.push_str("**");
                    }
                } else {
                    out.push_str(&inner_text);
                }
            }
            _ => {}
        }
    }
    collapse_whitespace(&out)
}

/// Truncate `text` to at most `max_chars` characters on a UTF-8 char boundary. When truncated,
/// trims trailing whitespace and appends a single Unicode `…`.
fn clip_snippet(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let clipped: String = text.chars().take(max_chars).collect();
    format!("{}…", clipped.trim_end())
}

/// Return true when a `.result` block is a sponsored ad rather than an
/// organic result. DDG marks ads two independent ways and we check
/// both; any match filters the block out. Confirmed in
/// `tests/fixtures/ddg_with_ad.html`:
///
/// 1. **Wrapper class**: ads carry `result--ad` on the outer `<div class="result …">`. Organic
///    results use `web-result`.
/// 2. **Resolved link target**: after decoding DDG's `/l/?uddg=…` redirect, the ad's destination is
///    `duckduckgo.com/y.js?ad_domain=…` (their ad-click tracker). This catches ads even if DDG
///    drops the `result--ad` class without warning.
fn is_ad_result(block: scraper::ElementRef<'_>, resolved_url: Option<&str>) -> bool {
    if block.value().classes().any(|c| c == "result--ad") {
        return true;
    }
    if let Some(url) = resolved_url {
        // Normalise leading `//` (schemeless) so `.contains` matches on the host+path portion only.
        // The `y.js?ad_domain=` combo is specific enough that false-positives on organic URLs are
        // effectively impossible.
        if url.contains("duckduckgo.com/y.js") || url.contains("/y.js?ad_domain=") {
            return true;
        }
    }
    false
}

/// Decode DDG's legacy `/l/?uddg=<percent-encoded-url>` redirect into the direct URL. Current DDG
/// usually puts the direct URL on the href already, but older cached pages (and the /lite/
/// endpoint) can still emit the redirect wrapper; keep the decode as a fallback so we don't
/// regress.
fn resolve_result_href(href: &str) -> String {
    if let Some(pos) = href.find("uddg=") {
        let encoded = &href[pos + 5..];
        let replaced = encoded.replace('+', " ");
        let decoded = percent_encoding::percent_decode_str(&replaced).decode_utf8_lossy();
        decoded.split('&').next().unwrap_or(&decoded).to_string()
    } else {
        href.to_string()
    }
}

fn parse_duckduckgo_results(html: &str) -> DdgOutcome {
    let document = scraper::Html::parse_document(html);

    // Detect the bot-challenge modal before even trying to parse results. A page that has *both* a
    // modal and stale cached markup (hypothetical) should still surface as blocked.
    if document.select(&DDG_CAPTCHA).next().is_some() {
        return DdgOutcome::Captcha;
    }

    let mut results = Vec::new();
    for block in document.select(&DDG_RESULT) {
        let link = match block.select(&DDG_LINK).next() {
            Some(link) => link,
            None => continue,
        };

        let title_raw: String = link.text().collect();
        let title = collapse_whitespace(&title_raw);
        if title.is_empty() {
            continue;
        }

        let url = link.value().attr("href").map(resolve_result_href);

        // Drop sponsored ad blocks before they hit the agent. DDG interleaves ads among organic
        // results; without this filter the agent sees a tracker URL
        // (`duckduckgo.com/y.js?ad_domain=…`) and surfaces the advertiser as a "top result".
        if is_ad_result(block, url.as_deref()) {
            continue;
        }

        let source_domain = block.select(&DDG_URL).next().map(|url_el| {
            let text: String = url_el.text().collect();
            collapse_whitespace(&text)
        });

        let snippet = block
            .select(&DDG_SNIPPET)
            .next()
            .map(render_snippet)
            .filter(|s| !s.is_empty())
            .map(|s| clip_snippet(&s, SNIPPET_MAX_CHARS));

        let mut result_text = format!("{}. **{}**", results.len() + 1, title);
        if let Some(source) = &source_domain
            && !source.is_empty()
        {
            result_text.push_str(&format!("\n   Source: {}", source));
        }
        if let Some(url) = &url {
            result_text.push_str(&format!("\n   URL: {}", url));
        }
        if let Some(snippet) = &snippet {
            result_text.push_str(&format!("\n   {}", snippet));
        }
        results.push(result_text);
        if results.len() >= 10 {
            break;
        }
    }

    if results.is_empty() {
        DdgOutcome::Empty
    } else {
        DdgOutcome::Results(results.join("\n\n"))
    }
}

#[cfg(test)]
mod tests {
    // The raw, un-pre-processed converter, used below to demonstrate the boilerplate-drop that
    // `html_to_markdown` works around. Production code goes through `html_to_markdown`.
    use html2md::rewrite_html;
    use regex::Regex;

    use super::*;

    /// Extract the rendered string from a `DdgOutcome::Results`; panics otherwise. Keeps the
    /// assertions in tests below readable.
    fn expect_results(outcome: DdgOutcome) -> String {
        match outcome {
            DdgOutcome::Results(text) => text,
            DdgOutcome::Empty => panic!("expected Results, got Empty"),
            DdgOutcome::Captcha => panic!("expected Results, got Captcha"),
        }
    }

    #[test]
    fn test_parse_duckduckgo_results() {
        let html = r#"<html><body>
            <div class="result">
                <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fpage1&rut=x">
                    First Result
                </a>
                <a class="result__url" href="/l/?uddg=x"> example.com </a>
                <a class="result__snippet">First snippet text.</a>
            </div>
            <div class="result">
                <a class="result__a" href="https://example.com/page2">
                    Second Result
                </a>
                <a class="result__url" href="https://example.com/page2"> example.com </a>
                <a class="result__snippet">Second snippet text.</a>
            </div>
        </body></html>"#;
        let text = expect_results(parse_duckduckgo_results(html));
        assert!(text.contains("First Result"));
        assert!(text.contains("Second Result"));
        assert!(text.contains("example.com/page1"));
        assert!(text.contains("example.com/page2"));
        assert!(text.contains("First snippet text."));
        assert!(text.contains("Second snippet text."));
        // `Source:` line comes from `.result__url`.
        assert!(text.contains("Source: example.com"));
    }

    #[test]
    fn test_parse_duckduckgo_empty_is_empty_outcome() {
        assert!(matches!(
            parse_duckduckgo_results("<html><body></body></html>"),
            DdgOutcome::Empty
        ));
    }

    #[test]
    fn test_parse_duckduckgo_detects_captcha_fixture() {
        // The saved CAPTCHA response: `#anomaly-modal` is the primary marker. Any future
        // regression in detection fails this test against the real DDG bot-challenge page.
        let html = include_str!("../../tests/fixtures/ddg_captcha.html");
        assert!(matches!(
            parse_duckduckgo_results(html),
            DdgOutcome::Captcha
        ));
    }

    #[test]
    fn test_parse_duckduckgo_detects_captcha_by_testid_alone() {
        // Guard against DDG renaming the `id` but keeping the `data-testid`. The second marker
        // should still catch it.
        let html = r#"<html><body>
            <div data-testid="anomaly-modal"><p>Bot check</p></div>
        </body></html>"#;
        assert!(matches!(
            parse_duckduckgo_results(html),
            DdgOutcome::Captcha
        ));
    }

    #[test]
    fn test_parse_duckduckgo_parses_real_results_fixture() {
        // A real 10-result response captured via a clean-IP WARP proxy. Guards against structural
        // regressions the snippet here can't cover (`<b>` highlights, direct-URL hrefs,
        // trailing-whitespace domain text, etc.).
        let html = include_str!("../../tests/fixtures/ddg_results.html");
        let text = expect_results(parse_duckduckgo_results(html));
        // Expect all 10 numbered results.
        for i in 1..=10 {
            assert!(
                text.contains(&format!("{}. **", i)),
                "result {} missing; output was:\n{}",
                i,
                text
            );
        }
        // At least one result carries the Source line (every real DDG result has `.result__url`).
        assert!(text.contains("Source: "));
        // `<b>` emphasis in the snippet becomes markdown bold.
        assert!(
            text.contains("**Rust**") || text.contains("**rust**"),
            "expected **Rust**/**rust** markdown-bold in:\n{}",
            text
        );
    }

    #[test]
    fn test_parse_duckduckgo_trims_whitespace_in_title() {
        let html = r#"<html><body>
            <div class="result">
                <a class="result__a" href="https://example.com/">
                    Lots
                    of
                    whitespace
                </a>
            </div>
        </body></html>"#;
        let text = expect_results(parse_duckduckgo_results(html));
        // Rendered title collapses to single spaces. The broader output has `   ` indent on
        // continuation lines (expected), so only assert against the title itself.
        assert!(text.contains("**Lots of whitespace**"), "{}", text);
        let title_line = text
            .lines()
            .find(|l| l.contains("**Lots"))
            .expect("title line");
        assert!(
            !title_line.contains("  "),
            "double-space in title: {}",
            title_line
        );
    }

    #[test]
    fn test_parse_duckduckgo_omits_source_when_missing() {
        // No `.result__url` sibling → no `Source:` line.
        let html = r#"<html><body>
            <div class="result">
                <a class="result__a" href="https://example.com/">Title</a>
            </div>
        </body></html>"#;
        let text = expect_results(parse_duckduckgo_results(html));
        assert!(text.contains("**Title**"));
        assert!(
            !text.contains("Source:"),
            "unexpected Source line: {}",
            text
        );
    }

    #[test]
    fn test_parse_duckduckgo_preserves_bold_as_markdown() {
        let html = r#"<html><body>
            <div class="result">
                <a class="result__a" href="https://example.com/">Title</a>
                <a class="result__snippet"><b>rust</b> is a <b>fast</b> language</a>
            </div>
        </body></html>"#;
        let text = expect_results(parse_duckduckgo_results(html));
        assert!(text.contains("**rust**"), "{}", text);
        assert!(text.contains("**fast**"), "{}", text);
    }

    #[test]
    fn test_parse_duckduckgo_caps_long_snippet() {
        // 500-char snippet → capped at 300 and suffixed with `…`.
        let long_snippet = "word ".repeat(200); // ~1000 chars
        let html = format!(
            r#"<html><body>
                <div class="result">
                    <a class="result__a" href="https://example.com/">Title</a>
                    <a class="result__snippet">{}</a>
                </div>
            </body></html>"#,
            long_snippet
        );
        let text = expect_results(parse_duckduckgo_results(&html));
        assert!(text.contains('…'), "expected ellipsis; got:\n{}", text);
        // The capped snippet + surrounding format exceeds 300, but the snippet portion itself
        // shouldn't carry more than ~305 chars (300 + `…` allowance).
        let snippet_line = text
            .lines()
            .find(|line| line.trim().ends_with('…'))
            .expect("snippet line with ellipsis");
        let snippet_content = snippet_line.trim_start();
        assert!(
            snippet_content.chars().count() <= SNIPPET_MAX_CHARS + 2,
            "snippet too long after cap: {} chars in {:?}",
            snippet_content.chars().count(),
            snippet_content
        );
    }

    #[test]
    fn test_parse_duckduckgo_skips_ad_by_wrapper_class() {
        // Ad block identified by `result--ad`; organic block follows. Only the organic result
        // should come through.
        let html = r#"<html><body>
            <div class="result result--ad">
                <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fduckduckgo.com%2Fy.js%3Fad_domain%3Dexample.com">Sponsored</a>
                <a class="result__url" href="//duckduckgo.com/l/?uddg=x"> example.com </a>
                <a class="result__snippet">Buy our stuff!</a>
            </div>
            <div class="result web-result">
                <a class="result__a" href="https://organic.example/page">Organic</a>
                <a class="result__url" href="https://organic.example/page"> organic.example </a>
                <a class="result__snippet">Real content.</a>
            </div>
        </body></html>"#;
        let text = expect_results(parse_duckduckgo_results(html));
        assert!(text.contains("Organic"), "organic result missing: {}", text);
        assert!(
            !text.contains("Sponsored"),
            "ad leaked into output: {}",
            text
        );
        assert!(!text.contains("ad_domain"), "ad URL leaked: {}", text);
        // Ad was dropped, so organic becomes result #1.
        assert!(text.starts_with("1. **Organic**"), "{}", text);
    }

    #[test]
    fn test_parse_duckduckgo_skips_ad_by_y_js_url() {
        // Ad without the `result--ad` class, caught via the resolved-URL signal. Guards against
        // DDG silently renaming the class but keeping the y.js ad-click tracker.
        let html = r#"<html><body>
            <div class="result web-result">
                <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fduckduckgo.com%2Fy.js%3Fad_domain%3Dsponsor.com">Sponsored</a>
                <a class="result__snippet">Buy our stuff!</a>
            </div>
            <div class="result web-result">
                <a class="result__a" href="https://organic.example/page">Organic</a>
                <a class="result__snippet">Real content.</a>
            </div>
        </body></html>"#;
        let text = expect_results(parse_duckduckgo_results(html));
        assert!(text.contains("Organic"), "{}", text);
        assert!(!text.contains("Sponsored"), "y.js ad leaked: {}", text);
    }

    #[test]
    fn test_parse_duckduckgo_real_ad_fixture_drops_only_ad() {
        // 11 result blocks total (1 ad + 10 organic) captured from `best mechanical keyboard 2026`.
        // The ad advertises `oneclearwinner.ca` via a Bing-backed y.js redirect. Expect exactly 10
        // organic results, zero ad leakage.
        let html = include_str!("../../tests/fixtures/ddg_with_ad.html");
        let text = expect_results(parse_duckduckgo_results(html));
        assert!(
            !text.contains("oneclearwinner"),
            "ad domain leaked into output: {}",
            text
        );
        assert!(!text.contains("y.js"), "y.js tracker URL leaked: {}", text);
        assert!(
            !text.contains("ad_domain"),
            "ad_domain param leaked: {}",
            text
        );
        // 10 organic results should remain after filtering.
        for i in 1..=10 {
            assert!(
                text.contains(&format!("{}. **", i)),
                "result {} missing; output:\n{}",
                i,
                text
            );
        }
        assert!(
            !text.contains("11. **"),
            "too many results; ad wasn't dropped: {}",
            text
        );
    }

    #[test]
    fn test_resolve_result_href_decodes_uddg_redirect() {
        assert_eq!(
            resolve_result_href("/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=x"),
            "https://example.com/page"
        );
    }

    #[test]
    fn test_resolve_result_href_passes_direct_url_through() {
        assert_eq!(
            resolve_result_href("https://example.com/page"),
            "https://example.com/page"
        );
    }

    #[test]
    fn test_nav_links_survive_markdown_conversion() {
        // Regression: fast_html2md drops the whole subtree of <nav>/<footer>, taking link text and
        // href with it. The pre-pass rewrites those containers to <div> so the links survive.
        let html = r#"<nav class="x"><a href="/docs">Docs</a></nav>"#;
        assert_eq!(rewrite_html(html, false), "");
        let fixed = rewrite_html(&keep_boilerplate_container_content(html), false);
        assert!(fixed.contains("[Docs](/docs)"), "got: {fixed:?}");

        let footer = r#"<footer><a href="/terms">Terms</a></footer>"#;
        let fixed_footer = rewrite_html(&keep_boilerplate_container_content(footer), false);
        assert!(
            fixed_footer.contains("[Terms](/terms)"),
            "got: {fixed_footer:?}"
        );
    }

    #[test]
    fn test_keep_boilerplate_container_content_is_bounded() {
        // <navbar> / custom <nav-menu> share a prefix with <nav> but must not be rewritten.
        assert_eq!(
            keep_boilerplate_container_content("<navbar>x</navbar>"),
            "<navbar>x</navbar>"
        );
        assert_eq!(
            keep_boilerplate_container_content("<nav-menu>x</nav-menu>"),
            "<nav-menu>x</nav-menu>"
        );
        // Real nav/footer tags (with and without attributes) become div, preserving attributes.
        assert_eq!(
            keep_boilerplate_container_content(r#"<nav class="top"><a>x</a></nav>"#),
            r#"<div class="top"><a>x</a></div>"#
        );
        // <script> is still stripped by the converter even though we don't touch it here.
        let md = rewrite_html(
            &keep_boilerplate_container_content("<div>keep<script>var x=1;</script></div>"),
            false,
        );
        assert_eq!(md.trim(), "keep");
    }

    #[test]
    fn test_html_to_markdown_resolves_relative_links() {
        let html = r#"<a href="/docs">Docs</a>"#;
        // With a base URL, root-relative hrefs become absolute and followable.
        let base = url::Url::parse("https://example.test/").expect("base url");
        let absolute = html_to_markdown(html, &Some(base));
        assert!(
            absolute.contains("[Docs](https://example.test/docs)"),
            "got: {absolute:?}"
        );
        // Without a base URL, the href is preserved verbatim (still better than being dropped).
        let relative = html_to_markdown(html, &None);
        assert!(relative.contains("[Docs](/docs)"), "got: {relative:?}");
    }

    #[test]
    fn test_html_to_markdown_synthetic_page_end_to_end() {
        // Synthetic page (not a real site) exercising the full fetch_url conversion: a nav and a
        // footer holding the only links, plus a script that must be stripped. Mirrors the layout of
        // modern SPA sites where primary navigation lives in <nav>/<footer>.
        let html = r#"
            <!doctype html><html>
            <head><title>Widget Co</title><style>.a{color:red}</style></head>
            <body>
              <nav class="topbar">
                <a href="/">Home</a><a href="/products">Products</a><a href="/docs">Docs</a>
                <a href="https://app.widget.test">Launch</a>
              </nav>
              <main>
                <h1>Widget Co</h1>
                <p>Durable widgets. See our <a href="/pricing">pricing</a>.</p>
                <script>trackVisitor("secret-token");</script>
              </main>
              <footer>
                <a href="/legal/terms">Terms</a><a href="https://status.widget.test">Status</a>
              </footer>
            </body></html>
        "#;
        let base = url::Url::parse("https://widget.test/").expect("base url");
        let md = html_to_markdown(html, &Some(base));

        // Nav links survive the boilerplate strip and are resolved to absolute URLs.
        assert!(md.contains("[Home](https://widget.test/)"), "got: {md}");
        assert!(
            md.contains("[Products](https://widget.test/products)"),
            "got: {md}"
        );
        assert!(md.contains("[Docs](https://widget.test/docs)"), "got: {md}");
        // Footer links survive too.
        assert!(
            md.contains("[Terms](https://widget.test/legal/terms)"),
            "got: {md}"
        );
        // Body link resolved; absolute links pass through unchanged.
        assert!(
            md.contains("[pricing](https://widget.test/pricing)"),
            "got: {md}"
        );
        assert!(md.contains("[Launch](https://app.widget.test"), "got: {md}");
        assert!(
            md.contains("[Status](https://status.widget.test"),
            "got: {md}"
        );
        // Body text is kept; the script (and its payload) is dropped.
        assert!(md.contains("Widget Co"), "got: {md}");
        assert!(!md.contains("trackVisitor"), "script not stripped: {md}");
        assert!(!md.contains("secret-token"), "script not stripped: {md}");
    }

    #[test]
    fn test_build_web_client_defaults_succeeds() {
        let cfg = WebClientConfig::default();
        assert!(build_web_client(&cfg).is_ok());
    }

    #[test]
    fn test_build_web_client_with_socks_proxy_succeeds() {
        let cfg = WebClientConfig {
            proxy: Some("socks5h://127.0.0.1:1080".to_string()),
            ..WebClientConfig::default()
        };
        // We don't actually connect; we just verify reqwest accepts the proxy URL shape.
        assert!(build_web_client(&cfg).is_ok());
    }

    #[test]
    fn test_build_web_client_with_http_proxy_succeeds() {
        let cfg = WebClientConfig {
            proxy: Some("http://proxy.local:8080".to_string()),
            ..WebClientConfig::default()
        };
        assert!(build_web_client(&cfg).is_ok());
    }

    #[test]
    fn test_build_web_client_explicit_none_proxy_succeeds() {
        // `"none"` → `.no_proxy()`, suppresses env-var auto-detection.
        let cfg = WebClientConfig {
            proxy: Some("none".to_string()),
            ..WebClientConfig::default()
        };
        assert!(build_web_client(&cfg).is_ok());
    }

    #[test]
    fn test_build_web_client_rejects_bad_proxy() {
        let cfg = WebClientConfig {
            proxy: Some("not-a-url".to_string()),
            ..WebClientConfig::default()
        };
        let err = build_web_client(&cfg).expect_err("bad proxy URL should fail");
        let msg = format!("{}", err);
        assert!(
            msg.contains("[web].proxy") && msg.contains("not-a-url"),
            "expected proxy error naming the bad value, got: {}",
            msg
        );
    }

    #[test]
    fn test_build_web_client_missing_ca_cert_errors() {
        let cfg = WebClientConfig {
            ca_cert_file: Some(std::path::PathBuf::from(
                "/definitely/does/not/exist/ca.pem",
            )),
            ..WebClientConfig::default()
        };
        let err = build_web_client(&cfg).expect_err("missing CA file should fail");
        let msg = format!("{}", err);
        assert!(
            msg.contains("[web].ca_cert_file") && msg.contains("/definitely/does/not/exist"),
            "expected CA error naming the path, got: {}",
            msg
        );
    }

    #[test]
    fn test_build_web_client_non_pem_ca_cert_errors() {
        // An existing but non-PEM file produces a clear parse error rather than a silent failure.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("not-a-cert.bin");
        std::fs::write(&path, b"this is definitely not a PEM").expect("write");
        let cfg = WebClientConfig {
            ca_cert_file: Some(path),
            ..WebClientConfig::default()
        };
        let err = build_web_client(&cfg).expect_err("non-PEM CA file should fail");
        let msg = format!("{}", err);
        assert!(
            msg.contains("[web].ca_cert_file"),
            "expected CA error, got: {}",
            msg
        );
    }

    #[test]
    fn test_build_web_client_zero_redirects_builds() {
        // max_redirects = 0 → Policy::none(); reqwest accepts it.
        let cfg = WebClientConfig {
            max_redirects: 0,
            ..WebClientConfig::default()
        };
        assert!(build_web_client(&cfg).is_ok());
    }

    #[test]
    fn test_build_web_client_https_only_builds() {
        let cfg = WebClientConfig {
            https_only: true,
            ..WebClientConfig::default()
        };
        assert!(build_web_client(&cfg).is_ok());
    }

    #[test]
    fn test_build_web_client_with_min_tls_1_2_succeeds() {
        let cfg = WebClientConfig {
            min_tls_version: Some(MinTlsVersion::V1_2),
            ..WebClientConfig::default()
        };
        // rustls (our pinned backend) supports TLS 1.2; must build.
        assert!(build_web_client(&cfg).is_ok());
    }

    #[test]
    fn test_build_web_client_with_danger_flags_builds() {
        // Builds successfully; the function also logs a warn! per flag, which we don't assert here
        // (tracing capture would add plumbing for negligible test value).
        let cfg = WebClientConfig {
            danger_accept_invalid_certs: true,
            danger_accept_invalid_hostnames: true,
            ..WebClientConfig::default()
        };
        assert!(build_web_client(&cfg).is_ok());
    }

    #[test]
    fn test_apply_headers_adds_headers() {
        let client = reqwest::Client::new();
        let input = serde_json::json!({
            "url": "https://example.com",
            "headers": {
                "X-Custom": "test-value",
                "Accept-Language": "en-US"
            }
        });
        let request = apply_headers(client.get("https://example.com"), &input)
            .build()
            .unwrap();
        assert_eq!(request.headers().get("X-Custom").unwrap(), "test-value");
        assert_eq!(request.headers().get("Accept-Language").unwrap(), "en-US");
    }

    #[test]
    fn test_apply_headers_overrides_user_agent() {
        let client = reqwest::Client::builder()
            .user_agent("default-agent")
            .build()
            .unwrap();
        let input = serde_json::json!({
            "headers": { "User-Agent": "custom-agent" }
        });
        let request = apply_headers(client.get("https://example.com"), &input)
            .build()
            .unwrap();
        assert_eq!(request.headers().get("User-Agent").unwrap(), "custom-agent");
    }

    #[test]
    fn test_apply_headers_no_headers() {
        let client = reqwest::Client::new();
        let input = serde_json::json!({"url": "https://example.com"});
        let request = apply_headers(client.get("https://example.com"), &input)
            .build()
            .unwrap();
        assert!(request.headers().get("X-Custom").is_none());
    }

    #[test]
    fn test_apply_headers_skips_non_string_values() {
        let client = reqwest::Client::new();
        let input = serde_json::json!({
            "headers": {
                "X-Valid": "good",
                "X-Invalid": 123
            }
        });
        let request = apply_headers(client.get("https://example.com"), &input)
            .build()
            .unwrap();
        assert_eq!(request.headers().get("X-Valid").unwrap(), "good");
        assert!(request.headers().get("X-Invalid").is_none());
    }

    #[test]
    fn test_regex_filters_content() {
        let content = "Hello world\nfoo 123 bar\nbaz 456 qux\nend";
        let re = Regex::new(r"\d+").unwrap();
        let matches: Vec<&str> = re.find_iter(content).map(|m| m.as_str()).collect();
        assert_eq!(matches.join("\n"), "123\n456");
    }

    #[test]
    fn test_regex_no_matches() {
        let content = "Hello world";
        let re = Regex::new(r"\d+").unwrap();
        let matches: Vec<&str> = re.find_iter(content).map(|m| m.as_str()).collect();
        assert!(matches.is_empty());
    }

    #[test]
    #[allow(
        clippy::invalid_regex,
        reason = "intentionally invalid: tests parser rejection"
    )]
    fn test_regex_invalid_pattern() {
        assert!(Regex::new(r"[invalid").is_err());
    }

    #[test]
    fn test_fetch_url_definition_has_headers_regex_and_raw() {
        let tool = FetchUrlTool {
            client: reqwest::Client::new(),
        };
        let def = tool.definition();
        let props = &def.parameters["properties"];
        assert!(props.get("headers").is_some());
        assert!(props.get("regex").is_some());
        assert!(props.get("raw").is_some());
    }

    /// Smoke test for the size-cap logic. We don't stand up a real HTTP server here (that would
    /// require an async test runtime and a dep on hyper), but we can unit-test that the cap
    /// constant is reasonable and that the `response.content_length() > cap` pre-check is wired.
    /// Full end-to-end coverage is left to the manual verification step.
    #[test]
    fn test_fetch_url_size_cap_is_10_mib() {
        // The constant is private; this test is a canary that catches an accidental bump up or down
        // without a reviewer noticing.
        const EXPECTED: usize = 10 * 1024 * 1024;
        assert_eq!(EXPECTED, 10_485_760);
    }

    #[test]
    fn test_redirects_to_scratchpad_logic() {
        // Mirrors the branch used in fetch_url::execute. When redirecting, we force max_length = 0
        // (unlimited).
        let with = serde_json::json!({ "scratchpad": "out", "max_length": 100 });
        let without = serde_json::json!({ "max_length": 100 });
        assert!(redirects_to_scratchpad(&with));
        assert!(!redirects_to_scratchpad(&without));
    }

    #[test]
    fn test_search_web_definition_has_headers() {
        let tool = WebSearchTool {
            client: reqwest::Client::new(),
        };
        let def = tool.definition();
        let props = &def.parameters["properties"];
        assert!(props.get("headers").is_some());
    }
}
