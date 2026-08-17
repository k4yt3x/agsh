//! `meka provider` subcommand suite and the provider OAuth login flows.
//!
//! Provider profiles live in `[providers.<name>]` in config.toml (non-secret settings only); the
//! credential — an API key or OAuth bundle — is stored in the database keyed by profile name and
//! acquired here via `add` / `login`. This replaces the old one-shot `meka setup` wizard.

use std::io::{self, IsTerminal, Read, Write};

use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use rand::RngExt;
use sha2::{Digest, Sha256};

use super::{
    AuthCredential, DEFAULT_CLAUDE_CLIENT_ID, DEFAULT_OPENAI_CODEX_CLIENT_ID, SUPPORTED_PROVIDERS,
};
use crate::{cli::ProviderAction, config, session::TokenStore};

const REDIRECT_URI: &str = "https://platform.claude.com/oauth/code/callback";
const AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
const TOKEN_URL: &str = "https://api.anthropic.com/v1/oauth/token";
const SCOPES: &str =
    "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers";

/// `openai-codex` OAuth flow constants. Mirror Codex's first-party CLI: the authorization server
/// lives at `auth.openai.com`, the redirect listener binds on `localhost:1455`.
const CODEX_AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
const CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CODEX_REDIRECT_PORT: u16 = 1455;
const CODEX_SCOPES: &str =
    "openid profile email offline_access api.connectors.read api.connectors.invoke";
const CODEX_CALLBACK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(180);

/// Dispatch a `meka provider` subcommand.
pub async fn run(action: &ProviderAction, token_store: &TokenStore) -> anyhow::Result<()> {
    match action {
        ProviderAction::Add {
            name,
            r#type,
            model,
            base_url,
            api_key_stdin,
        } => {
            run_add(
                name,
                r#type.as_deref(),
                model.clone(),
                base_url.clone(),
                *api_key_stdin,
                token_store,
            )
            .await
        }
        ProviderAction::List => run_list(token_store).await,
        ProviderAction::Use { name } => run_use(name),
        ProviderAction::Remove { name } => run_remove(name, token_store).await,
        ProviderAction::Login { name } => run_login(name, token_store).await,
    }
}

async fn run_add(
    name: &str,
    type_flag: Option<&str>,
    model_flag: Option<String>,
    base_url_flag: Option<String>,
    api_key_stdin: bool,
    token_store: &TokenStore,
) -> anyhow::Result<()> {
    if name.trim().is_empty() {
        anyhow::bail!("profile name cannot be empty");
    }
    // Hard-fails on an unparseable config rather than warning: this guard is the only thing
    // standing between `provider add <existing>` and `upsert_profile_document` replacing the
    // profile's table, and an empty parsed map defeats it.
    if config::load_config_file_or_err()?
        .providers
        .contains_key(name)
    {
        anyhow::bail!(
            "a profile named '{}' already exists. Use `meka provider login {}` to re-authenticate, \
             or `meka provider remove {}` first.",
            name,
            name,
            name
        );
    }

    let backend = match type_flag {
        Some(value) => validate_backend(value)?.to_string(),
        None => prompt_backend()?,
    };

    let model = match model_flag {
        Some(model) => model,
        None => {
            let default_model = default_model_for(&backend);
            let prompt = match default_model {
                Some(default) => format!("\nModel name [{}]: ", default),
                None => "\nModel name: ".to_string(),
            };
            let input = prompt_line(&prompt)?;
            match (input.is_empty(), default_model) {
                // Empty entry accepts the backend's default.
                (true, Some(default)) => default.to_string(),
                (true, None) => anyhow::bail!("model name cannot be empty"),
                (false, _) => input,
            }
        }
    };

    let base_url = match base_url_flag {
        Some(url) => Some(url),
        None => {
            let input = prompt_line("API base URL (leave empty for default): ")?;
            (!input.is_empty()).then_some(input)
        }
    };

    // Acquire the credential last: the Codex OAuth login races a pasted-callback-URL reader against
    // the loopback callback, and if the callback wins it can leave a stdin read parked. Keeping the
    // interactive prompts above (which read stdin) before this ensures nothing reads stdin after.
    let credential = acquire_credential(&backend, api_key_stdin, None).await?;

    token_store
        .save_provider_credential(name, &credential)
        .await?;
    write_profile(name, &backend, model.as_str(), base_url.as_deref())?;

    tracing::info!("added provider profile '{}'", name);
    Ok(())
}

async fn run_login(name: &str, token_store: &TokenStore) -> anyhow::Result<()> {
    let config_file = config::load_config_file_or_err()?;
    let Some(profile) = config_file.providers.get(name) else {
        anyhow::bail!(
            "no provider profile named '{}'. Run `meka provider add {}` to create it.",
            name,
            name
        );
    };
    let credential =
        acquire_credential(&profile.backend, false, profile.client_id.as_deref()).await?;
    token_store
        .save_provider_credential(name, &credential)
        .await?;
    tracing::info!("re-authenticated provider profile '{}'", name);
    Ok(())
}

async fn run_remove(name: &str, token_store: &TokenStore) -> anyhow::Result<()> {
    // Deliberately does not require a configured profile: this is the only path that deletes a
    // credential, so it has to work on one whose `[providers.<name>]` block was deleted by hand.
    // Both sides are read before either is touched so the confirmation can say which of them
    // actually existed, and so a typo'd name fails instead of reporting a removal that removed
    // nothing. `open_document` rather than the parsed config on purpose: `remove` must still run on
    // a config.toml that meka can't deserialize, since it is one of the ways such a file gets
    // repaired.
    let has_credential = token_store.load_provider_credential(name).await?.is_some();

    // Probed under its own short-lived guard, and the guard dropped before the `await` below.
    //
    // `ConfigFileLock` tracks reentrancy in a *thread*-local depth counter, so holding one across
    // an await is unsound on a multi-threaded runtime: the task can resume on another worker, where
    // the depth reads zero, and a nested acquisition then tries to take a file lock this process
    // already holds. That is a self-deadlock, and the counter it leaves behind is under-balanced on
    // one thread and over-balanced on the other. Two short critical sections cost a TOCTOU window
    // no CLI invocation can lose anything to; one long one costs correctness.
    let has_profile = {
        let (_lock, _path, document) = open_document()?;
        document
            .get("providers")
            .and_then(|item| item.as_table())
            .is_some_and(|providers| providers.contains_key(name))
    };

    if !has_profile && !has_credential {
        anyhow::bail!("no provider profile or stored credential named '{}'", name);
    }

    // Delete the credential first; the config write can still fail, but the secret should go
    // regardless so a `remove` that gets this far always logs you out. A config meka cannot *read*
    // stops the command above instead, with nothing done: an error that leaves the secret deleted
    // reads to the user as "nothing happened", which is the one thing it must not mean.
    token_store.delete_provider_credential(name).await?;
    // Re-read under a fresh guard so the edit below is applied to the current file, and so the
    // read-modify-write is one critical section with no await inside it.
    let (_lock, path, mut document) = open_document()?;
    // Written even with no profile to remove: `default_provider` can still point at the name, and
    // dropping that dangling pointer is exactly the cleanup this case is for.
    remove_profile_document(&mut document, name);
    config::write_file_atomic(&path, &document.to_string())?;

    if has_profile {
        tracing::info!("removed provider profile '{}'", name);
    } else {
        tracing::info!(
            "cleared the stored credential for '{}'; no profile was configured",
            name
        );
    }
    Ok(())
}

fn run_use(name: &str) -> anyhow::Result<()> {
    let config_file = config::load_config_file_or_err()?;
    if !config_file.providers.contains_key(name) {
        anyhow::bail!(
            "no provider profile named '{}' (configured: {})",
            name,
            join_profile_names(&config_file)
        );
    }
    set_default_provider(name)?;
    tracing::info!("default provider set to '{}'", name);
    Ok(())
}

async fn run_list(token_store: &TokenStore) -> anyhow::Result<()> {
    let config_file = config::load_config_file_or_err()?;
    // Computed before the early return below: "every profile is gone but the secrets are still
    // here" is precisely the state worth reporting, and it is the one the old early return hid.
    let orphans = orphaned_profiles(token_store, &config_file).await?;

    if config_file.providers.is_empty() {
        eprintln!("No provider profiles configured.");
        report_orphaned_profiles(&orphans);
        return Ok(());
    }
    let default = config_file.default_provider.as_deref();
    let mut rows: Vec<Vec<String>> = Vec::with_capacity(config_file.providers.len());
    for (name, profile) in &config_file.providers {
        let authed = match token_store.load_provider_credential(name).await {
            Ok(Some(_)) => "yes",
            _ => "no",
        };
        let default_marker = if Some(name.as_str()) == default {
            "*"
        } else {
            ""
        };
        rows.push(vec![
            name.clone(),
            profile.backend.clone(),
            profile.model.clone().unwrap_or_else(|| "-".to_string()),
            authed.to_string(),
            default_marker.to_string(),
        ]);
    }
    // Requested data goes to stdout via the shared column formatter, matching `meka mcp list`.
    print!(
        "{}",
        crate::render::format_columns(
            &["Name", "Type", "Model", "Authenticated", "Default"],
            &rows
        )
    );
    report_orphaned_profiles(&orphans);
    Ok(())
}

/// Profile names holding a stored credential that no configured profile claims.
///
/// A credential is keyed by profile name and nothing deletes it when the `[providers.<name>]` block
/// goes away by hand, so an API key or OAuth refresh token can outlive its profile indefinitely.
/// This diff is the only thing that can name one, and `provider list` is where a name is worth
/// something: `meka provider remove <name>` then clears it.
///
/// Safe to compute here only because the caller already failed on an unreadable config. An empty
/// profile map that came from a config meka could not parse would report every credential in the
/// database as an orphan.
async fn orphaned_profiles(
    token_store: &TokenStore,
    config_file: &config::ConfigFile,
) -> anyhow::Result<Vec<String>> {
    Ok(token_store
        .list_credential_profiles()
        .await?
        .into_iter()
        .filter(|profile| !config_file.providers.contains_key(profile))
        .collect())
}

/// Print the orphan block, if there is one. The names go to stdout with the rest of the answer --
/// hiding them behind a stderr-only note is how they stayed invisible in the first place -- and the
/// instruction for acting on them is a stderr hint.
fn report_orphaned_profiles(orphans: &[String]) {
    if orphans.is_empty() {
        return;
    }
    println!();
    println!("Stored credentials with no profile: {}", orphans.join(", "));
    // Says only what the diff proves. Deleting the block by hand is the usual cause, but an `add`
    // that stored the secret and then failed to write the profile leaves the same trace, and a hint
    // that names one cause would send that user looking for an edit they never made.
    crate::render::render_hint(
        "left over from a profile that is no longer configured; \
         delete one with `meka provider remove <name>`",
    );
}

fn validate_backend(value: &str) -> anyhow::Result<&str> {
    if SUPPORTED_PROVIDERS.contains(&value) {
        Ok(value)
    } else {
        anyhow::bail!(
            "'{}' is not a valid provider type. Supported: {}",
            value,
            SUPPORTED_PROVIDERS.join(", ")
        )
    }
}

/// Default model offered at the `provider add` prompt for a given backend. The user can override it
/// by typing a different name; an empty entry accepts the default. `None` for backends without a
/// sensible default (none currently), where the prompt then requires an explicit answer.
fn default_model_for(backend: &str) -> Option<&'static str> {
    match backend {
        "claude-api" | "claude-oauth" => Some("claude-opus-5"),
        "openai-api" | "openai-codex" => Some("gpt-5.6-sol"),
        _ => None,
    }
}

fn join_profile_names(config_file: &config::ConfigFile) -> String {
    config_file
        .providers
        .keys()
        .cloned()
        .collect::<Vec<_>>()
        .join(", ")
}

/// Acquire a credential for `backend`: run the OAuth flow for OAuth backends, or read an API key
/// (from stdin when `api_key_stdin`, else an interactive prompt) for key backends.
async fn acquire_credential(
    backend: &str,
    api_key_stdin: bool,
    client_id: Option<&str>,
) -> anyhow::Result<AuthCredential> {
    match backend {
        "claude-oauth" => claude_login(client_id).await,
        "openai-codex" => codex_login(client_id).await,
        "claude-api" | "openai-api" => {
            let key = if api_key_stdin {
                let mut buffer = String::new();
                io::stdin().read_to_string(&mut buffer)?;
                buffer.trim().to_string()
            } else {
                prompt_secret("Enter your API key: ")?
            };
            if key.is_empty() {
                anyhow::bail!("API key cannot be empty");
            }
            Ok(AuthCredential::ApiKey(key))
        }
        other => validate_backend(other).map(|_| unreachable!()),
    }
}

// ----- Config file editing (toml_edit, comment-preserving) ---------------------------------------

/// Returns the lock alongside the document so a caller cannot read, mutate and write without
/// holding it: the whole point is that the read and the write are one critical section, and a
/// separate `lock_config_file()` call would be a step someone eventually forgets.
fn open_document() -> anyhow::Result<(
    config::ConfigFileLock,
    std::path::PathBuf,
    toml_edit::DocumentMut,
)> {
    let lock = config::lock_config_file()?;
    let path = config::config_file_path()
        .ok_or_else(|| anyhow::anyhow!("could not determine config directory"))?;
    // Only a genuinely absent file starts from empty. Treating *any* read failure as "" turns
    // "I couldn't read your config" into "your config is blank", and the caller writes that blank
    // document straight back over the real file: one non-UTF-8 byte or a mode-000 file, and
    // `provider remove` truncates config.toml to nothing, profiles and MCP servers included. The
    // `meka mcp` editors already tolerate `NotFound` only; this matches them.
    let contents = match std::fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => {
            anyhow::bail!("failed to read config at {}: {}", path.display(), error);
        }
    };
    let document = contents.parse::<toml_edit::DocumentMut>()?;
    Ok((lock, path, document))
}

/// Borrow the `[providers]` table as a real (header) table, creating it implicit if absent. Without
/// this, auto-vivifying `document["providers"][name]` produces an *inline* table, which both
/// renders the whole block on one line and makes `as_table_mut()` return `None` (so removals
/// silently fail).
fn ensure_providers_table(
    document: &mut toml_edit::DocumentMut,
) -> anyhow::Result<&mut toml_edit::Table> {
    if document
        .get("providers")
        .map(|item| !item.is_table())
        .unwrap_or(true)
    {
        let mut table = toml_edit::Table::new();
        // Implicit so the parent emits `[providers.<name>]` headers rather than a bare
        // `[providers]`.
        table.set_implicit(true);
        document["providers"] = toml_edit::Item::Table(table);
    }
    document
        .get_mut("providers")
        .and_then(|item| item.as_table_mut())
        .ok_or_else(|| anyhow::anyhow!("config 'providers' is not a table"))
}

/// Insert or replace `[providers.<name>]` in `document`, defaulting to it if no `default_provider`
/// is set yet. Pure mutation so it can be unit-tested without touching the filesystem.
fn upsert_profile_document(
    document: &mut toml_edit::DocumentMut,
    name: &str,
    backend: &str,
    model: &str,
    base_url: Option<&str>,
) -> anyhow::Result<()> {
    let mut profile = toml_edit::Table::new();
    profile.insert("type", toml_edit::value(backend));
    profile.insert("model", toml_edit::value(model));
    if let Some(url) = base_url {
        profile.insert("base_url", toml_edit::value(url));
    }
    ensure_providers_table(document)?.insert(name, toml_edit::Item::Table(profile));

    // Make the first profile the default so a single-profile setup needs no extra step.
    if document.get("default_provider").is_none() {
        document["default_provider"] = toml_edit::value(name);
    }
    Ok(())
}

/// Remove `[providers.<name>]` from `document`, clearing `default_provider` if it pointed at the
/// removed profile. Pure mutation, unit-testable.
fn remove_profile_document(document: &mut toml_edit::DocumentMut, name: &str) {
    if let Some(providers) = document
        .get_mut("providers")
        .and_then(|item| item.as_table_mut())
    {
        providers.remove(name);
    }
    // If this profile was the default, drop the dangling pointer.
    if document
        .get("default_provider")
        .and_then(|item| item.as_str())
        == Some(name)
    {
        document.as_table_mut().remove("default_provider");
    }
}

fn write_profile(
    name: &str,
    backend: &str,
    model: &str,
    base_url: Option<&str>,
) -> anyhow::Result<()> {
    // `_lock` is held to the end of the function, so the read above and the write below are
    // one critical section.
    let (_lock, path, mut document) = open_document()?;
    upsert_profile_document(&mut document, name, backend, model, base_url)?;
    config::write_file_atomic(&path, &document.to_string())?;
    Ok(())
}

fn set_default_provider(name: &str) -> anyhow::Result<()> {
    // `_lock` is held to the end of the function, so the read above and the write below are
    // one critical section.
    let (_lock, path, mut document) = open_document()?;
    document["default_provider"] = toml_edit::value(name);
    config::write_file_atomic(&path, &document.to_string())?;
    Ok(())
}

// ----- Interactive prompts -----------------------------------------------------------------------

/// Read a line without echoing it, so a pasted API key is not left on screen, in a scrollback
/// buffer, or in a screen recording.
///
/// Falls back to a visible prompt where echo cannot be suppressed (not a tty, or a platform without
/// termios), and says so, because silently echoing a secret the caller asked to hide is worse than
/// the visible prompt they can decide about.
fn prompt_secret(prompt: &str) -> io::Result<String> {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;

        let fd = io::stdin().as_raw_fd();
        // SAFETY: `fd` is stdin's descriptor and `termios` is a valid out-parameter for the
        // duration of the call. `tcgetattr` only writes through it.
        let mut original: libc::termios = unsafe { std::mem::zeroed() };
        if unsafe { libc::tcgetattr(fd, &mut original) } == 0 {
            let mut quiet = original;
            quiet.c_lflag &= !libc::ECHO;
            // SAFETY: `quiet` is a termios obtained from this same descriptor with one flag
            // cleared.
            if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &quiet) } == 0 {
                // Restored on the way out of *every* exit, including the one the ordinary code
                // path cannot see. Ctrl-C at this prompt is a normal thing to do -- wrong profile,
                // wrong account, changed your mind -- and SIGINT's default disposition kills the
                // process where it stands, so nothing below runs and the user is left in a shell
                // that shows nothing they type until they find `stty sane`. That is the exact
                // outcome the doc above says must not happen.
                let _echo = EchoGuard::install(fd, original);
                let result = prompt_line(prompt);
                // The Enter the user pressed was not echoed either, so the cursor is still on the
                // prompt line.
                drop(_echo);
                eprintln!();
                return result;
            }
        }
        // `warn!`, not `debug!`: this is a secret about to appear on screen, and the doc above
        // promises meka says so rather than quietly echoing it.
        tracing::warn!("could not disable terminal echo; the API key will be visible as typed");
    }
    prompt_line(prompt)
}

/// Restores terminal echo when dropped *and* when the process is interrupted.
///
/// The handler is what makes this more than a `Drop` impl. `Drop` covers a return and a panic;
/// SIGINT bypasses both. It runs only `tcsetattr` and `_exit`, which are async-signal-safe, and
/// reads the saved settings through an `AtomicPtr` because a lock is not.
#[cfg(unix)]
struct EchoGuard {
    fd: std::os::unix::io::RawFd,
    original: libc::termios,
    previous_handler: libc::sighandler_t,
}

#[cfg(unix)]
static SAVED_TERMIOS: std::sync::atomic::AtomicPtr<libc::termios> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());
#[cfg(unix)]
static SAVED_FD: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

#[cfg(unix)]
extern "C" fn restore_echo_on_interrupt(_signal: libc::c_int) {
    let saved = SAVED_TERMIOS.load(std::sync::atomic::Ordering::Acquire);
    let fd = SAVED_FD.load(std::sync::atomic::Ordering::Acquire);
    if !saved.is_null() && fd >= 0 {
        // SAFETY: `saved` was published by `EchoGuard::install` from a live `Box` that outlives the
        // guard, and `fd` is the descriptor those settings came from. `tcsetattr` is
        // async-signal-safe.
        unsafe { libc::tcsetattr(fd, libc::TCSANOW, saved) };
    }
    // 128 + SIGINT, the conventional status, and `_exit` rather than `exit` because only the
    // former is async-signal-safe.
    unsafe { libc::_exit(130) };
}

#[cfg(unix)]
impl EchoGuard {
    fn install(fd: std::os::unix::io::RawFd, original: libc::termios) -> Self {
        // Leaked deliberately: the handler may read it at any point until the guard is dropped, and
        // freeing it on the drop path would race a signal arriving in the same instant. One
        // termios per prompt is a rounding error against a process that is about to hold a
        // conversation in memory.
        let saved = Box::into_raw(Box::new(original));
        SAVED_TERMIOS.store(saved, std::sync::atomic::Ordering::Release);
        SAVED_FD.store(fd, std::sync::atomic::Ordering::Release);
        // SAFETY: installing a handler for SIGINT; the function pointer is a valid `extern "C"`
        // handler and the returned value is the previous disposition, restored on drop.
        let previous_handler = unsafe {
            libc::signal(
                libc::SIGINT,
                restore_echo_on_interrupt as *const () as libc::sighandler_t,
            )
        };
        Self {
            fd,
            original,
            previous_handler,
        }
    }
}

#[cfg(unix)]
impl Drop for EchoGuard {
    fn drop(&mut self) {
        // SAFETY: `self.original` came from this descriptor, and the handler is being put back to
        // whatever it was before `install`.
        unsafe {
            libc::tcsetattr(self.fd, libc::TCSANOW, &self.original);
            libc::signal(libc::SIGINT, self.previous_handler);
        }
        SAVED_TERMIOS.store(std::ptr::null_mut(), std::sync::atomic::Ordering::Release);
        SAVED_FD.store(-1, std::sync::atomic::Ordering::Release);
    }
}

fn prompt_line(prompt: &str) -> io::Result<String> {
    eprint!("{}", prompt);
    // The prompt went to stderr, so that is what has to be flushed; flushing stdout left the
    // prompt sitting in stderr's buffer and the user staring at a blank line.
    io::stderr().flush()?;
    let mut input = String::new();
    // `read_line` reports end of input as `Ok(0)` with an empty buffer, which is indistinguishable
    // from a bare Enter unless the count is checked. A caller that re-prompts on a bad answer would
    // otherwise spin forever against a closed stdin, which `prompt_backend` did.
    if io::stdin().read_line(&mut input)? == 0 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "no more input to read",
        ));
    }
    Ok(input.trim().to_string())
}

fn prompt_backend() -> anyhow::Result<String> {
    let options = [
        ("claude-oauth", "Claude Code OAuth login"),
        ("claude-api", "Claude API key"),
        ("openai-codex", "ChatGPT subscription login"),
        ("openai-api", "OpenAI API key"),
    ];
    eprintln!("Select a provider type:");
    for (index, (id, label)) in options.iter().enumerate() {
        eprintln!("  {}. {} ({})", index + 1, id, label);
    }
    loop {
        let input = prompt_line("> ")?;
        if let Ok(choice) = input.parse::<usize>()
            && (1..=options.len()).contains(&choice)
        {
            return Ok(options[choice - 1].0.to_string());
        }
        eprintln!("Please enter a number between 1 and {}.", options.len());
    }
}

// ----- Claude OAuth (paste-back) -----------------------------------------------------------------

fn generate_pkce_pair() -> (String, String) {
    let mut bytes = [0u8; 32];
    rand::rng().fill(&mut bytes);
    let code_verifier = URL_SAFE_NO_PAD.encode(bytes);
    let digest = Sha256::digest(code_verifier.as_bytes());
    let code_challenge = URL_SAFE_NO_PAD.encode(digest);
    (code_verifier, code_challenge)
}

fn generate_state() -> String {
    let mut bytes = [0u8; 32];
    rand::rng().fill(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

fn build_authorize_url(
    client_id: &str,
    code_challenge: &str,
    state: &str,
) -> anyhow::Result<String> {
    let mut url = reqwest::Url::parse(AUTHORIZE_URL)?;
    url.query_pairs_mut()
        .append_pair("code", "true")
        .append_pair("client_id", client_id)
        .append_pair("response_type", "code")
        .append_pair("redirect_uri", REDIRECT_URI)
        .append_pair("scope", SCOPES)
        .append_pair("code_challenge", code_challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("state", state);
    Ok(url.to_string())
}

async fn claude_login(client_id: Option<&str>) -> anyhow::Result<AuthCredential> {
    let client_id = client_id.unwrap_or(DEFAULT_CLAUDE_CLIENT_ID);
    let (code_verifier, code_challenge) = generate_pkce_pair();
    let state = generate_state();
    let url = build_authorize_url(client_id, &code_challenge, &state)?;

    if let Err(error) = open::that(&url) {
        tracing::debug!("failed to open browser: {}", error);
    }
    eprintln!("\nTo authorize, open this URL in your browser:");
    eprintln!("    {}\n", url);

    let code_input = prompt_line("After authorizing, paste the authorization code here:\n> ")?;
    if code_input.is_empty() {
        anyhow::bail!("authorization code cannot be empty");
    }
    // The pasted value may include the state after a '#' delimiter (e.g. "code#state").
    let code = code_input.split('#').next().unwrap_or(&code_input);

    exchange_claude_code(code, &code_verifier, client_id, &state).await
}

#[derive(serde::Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<i64>,
    /// Anthropic's OAuth token response carries the subscriber's account here; its `uuid` is what
    /// Claude Code sends as `metadata.user_id.account_uuid` on every request.
    account: Option<OAuthAccount>,
}

#[derive(serde::Deserialize)]
struct OAuthAccount {
    uuid: String,
}

async fn exchange_claude_code(
    code: &str,
    code_verifier: &str,
    client_id: &str,
    state: &str,
) -> anyhow::Result<AuthCredential> {
    let client = reqwest::Client::new();
    let response = client
        .post(TOKEN_URL)
        .json(&serde_json::json!({
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": code_verifier,
            "redirect_uri": REDIRECT_URI,
            "client_id": client_id,
            "state": state,
        }))
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("token exchange failed ({}): {}", status, body);
    }

    let token: TokenResponse = response.json().await?;
    let expires_at = token.expires_in.map(|seconds| {
        let now_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_millis() as i64)
            .unwrap_or(0);
        // Saturating: a nonsense `expires_in` should read as "far future" and let the 401 correct
        // it, not overflow to a past instant and refresh on every request.
        seconds
            .checked_mul(1000)
            .map_or(i64::MAX, |millis| now_millis.saturating_add(millis))
    });

    Ok(AuthCredential::OAuthToken {
        access_token: token.access_token,
        refresh_token: token.refresh_token,
        expires_at,
        account_id: token.account.map(|account| account.uuid),
    })
}

// ----- OpenAI Codex OAuth (localhost callback) ---------------------------------------------------

async fn codex_login(client_id: Option<&str>) -> anyhow::Result<AuthCredential> {
    let client_id = client_id.unwrap_or(DEFAULT_OPENAI_CODEX_CLIENT_ID);
    let (code_verifier, code_challenge) = generate_pkce_pair();
    let state = generate_state();
    let redirect_uri = format!("http://localhost:{}/auth/callback", CODEX_REDIRECT_PORT);
    let url = build_codex_authorize_url(client_id, &code_challenge, &state, &redirect_uri)?;

    // The loopback callback only works when the browser runs on this machine. On a remote/headless
    // box the redirect lands on the user's laptop instead, so a TTY session can still finish by
    // pasting the callback URL; a bind failure there degrades to paste-only rather than aborting.
    let paste_enabled = io::stdin().is_terminal();
    let listener =
        match tokio::net::TcpListener::bind(format!("127.0.0.1:{}", CODEX_REDIRECT_PORT)).await {
            Ok(listener) => Some(listener),
            Err(error) if paste_enabled => {
                tracing::warn!(
                    "failed to bind callback listener on 127.0.0.1:{}: {}; \
                     falling back to pasting the callback URL",
                    CODEX_REDIRECT_PORT,
                    error
                );
                None
            }
            Err(error) => {
                anyhow::bail!(
                    "failed to bind callback listener on 127.0.0.1:{}: {}. \
                     Is another login already running?",
                    CODEX_REDIRECT_PORT,
                    error
                );
            }
        };

    if let Err(error) = open::that(&url) {
        tracing::debug!("failed to open browser for Codex login: {}", error);
    }
    eprintln!("\nTo authorize, open this URL in your browser:");
    eprintln!("    {}\n", url);
    if listener.is_some() {
        eprintln!(
            "Waiting up to {}s for the callback on 127.0.0.1:{}...",
            CODEX_CALLBACK_TIMEOUT.as_secs(),
            CODEX_REDIRECT_PORT
        );
    }
    if paste_enabled {
        eprintln!(
            "If your browser is on another machine, paste the full callback URL here and press Enter."
        );
    }

    // Race the loopback callback against a pasted-URL reader when both are viable. The accept
    // future carries the timeout that bounds the whole wait; the paste reader parks on EOF so a
    // non-interactive stdin can never win the race against a real callback.
    let (received_code, received_state) = match (listener, paste_enabled) {
        (Some(listener), true) => tokio::select! {
            result = accept_codex_callback(listener, CODEX_CALLBACK_TIMEOUT) => result?,
            result = read_pasted_codex_callback() => result?,
        },
        (Some(listener), false) => accept_codex_callback(listener, CODEX_CALLBACK_TIMEOUT).await?,
        (None, _) => read_pasted_codex_callback().await?,
    };
    if received_state != state {
        anyhow::bail!("OAuth state mismatch, possible CSRF; aborting");
    }
    exchange_codex_code(&received_code, &code_verifier, client_id, &redirect_uri).await
}

/// Read a manually pasted callback URL from stdin, the fallback for when the loopback callback
/// can't reach this machine. Blank lines re-prompt; an unparseable line prints a hint and
/// re-prompts (a mis-paste must not abort while a real callback may still arrive); EOF parks the
/// future forever so a non-interactive stdin can't win the `select!` against the callback branch.
async fn read_pasted_codex_callback() -> anyhow::Result<(String, String)> {
    use tokio::io::{AsyncBufReadExt, BufReader};

    let mut reader = BufReader::new(tokio::io::stdin());
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line).await {
            Err(error) => anyhow::bail!("failed to read pasted callback URL: {}", error),
            Ok(0) => std::future::pending::<()>().await,
            Ok(_) => {
                if line.trim().is_empty() {
                    continue;
                }
                match extract_codex_paste(&line) {
                    CodexCallback::Match { code, state } => return Ok((code, state)),
                    CodexCallback::AuthError(message) => {
                        anyhow::bail!("authorization server returned error: {}", message)
                    }
                    CodexCallback::NotCallback | CodexCallback::Malformed(_) => {
                        eprintln!(
                            "Could not find 'code' and 'state' in that input; paste the full \
                             callback URL and press Enter."
                        );
                        continue;
                    }
                }
            }
        }
    }
}

fn build_codex_authorize_url(
    client_id: &str,
    code_challenge: &str,
    state: &str,
    redirect_uri: &str,
) -> anyhow::Result<String> {
    let mut url = reqwest::Url::parse(CODEX_AUTHORIZE_URL)?;
    url.query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", client_id)
        .append_pair("redirect_uri", redirect_uri)
        .append_pair("scope", CODEX_SCOPES)
        .append_pair("code_challenge", code_challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("id_token_add_organizations", "true")
        .append_pair("codex_cli_simplified_flow", "true")
        .append_pair("state", state)
        .append_pair("originator", "meka_cli");
    Ok(url.to_string())
}

async fn accept_codex_callback(
    listener: tokio::net::TcpListener,
    timeout: std::time::Duration,
) -> anyhow::Result<(String, String)> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            anyhow::bail!("authorization timed out after {}s", timeout.as_secs());
        }
        let (mut stream, _) = match tokio::time::timeout(remaining, listener.accept()).await {
            Ok(Ok(pair)) => pair,
            Ok(Err(error)) => {
                anyhow::bail!("failed to accept the OAuth callback connection: {}", error)
            }
            Err(_) => anyhow::bail!("authorization timed out after {}s", timeout.as_secs()),
        };

        const MAX_BYTES: usize = 64 * 1024;
        let mut buffer = Vec::with_capacity(4096);
        let mut temp = [0u8; 4096];
        let headers_complete = loop {
            if buffer.windows(4).any(|window| window == b"\r\n\r\n") {
                break true;
            }
            if buffer.len() >= MAX_BYTES {
                break false;
            }
            let read_remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if read_remaining.is_zero() {
                anyhow::bail!("authorization timed out after {}s", timeout.as_secs());
            }
            match tokio::time::timeout(read_remaining, stream.read(&mut temp)).await {
                Ok(Ok(0)) => break buffer.windows(4).any(|window| window == b"\r\n\r\n"),
                Ok(Ok(n)) => buffer.extend_from_slice(&temp[..n]),
                Ok(Err(error)) => {
                    anyhow::bail!("failed to read the OAuth callback request: {}", error)
                }
                Err(_) => anyhow::bail!("authorization timed out after {}s", timeout.as_secs()),
            }
        };

        if !headers_complete {
            let _ = stream
                .write_all(b"HTTP/1.1 431 Request Header Fields Too Large\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .await;
            continue;
        }

        let request = String::from_utf8_lossy(&buffer);
        match parse_codex_callback_query(&request) {
            CodexCallback::Match { code, state } => {
                let body = b"<!DOCTYPE html><html><body>\
                    <h1>Codex authorization successful</h1>\
                    <p>You can close this tab and return to meka.</p>\
                    </body></html>";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.write_all(body).await;
                return Ok((code, state));
            }
            CodexCallback::NotCallback => {
                let _ = stream
                    .write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    )
                    .await;
                continue;
            }
            CodexCallback::Malformed(message) => {
                let _ = stream
                    .write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                    .await;
                anyhow::bail!(message);
            }
            CodexCallback::AuthError(message) => {
                let _ = stream
                    .write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                    .await;
                anyhow::bail!("authorization server returned error: {}", message);
            }
        }
    }
}

enum CodexCallback {
    Match { code: String, state: String },
    NotCallback,
    Malformed(String),
    AuthError(String),
}

fn parse_codex_callback_query(request: &str) -> CodexCallback {
    let Some(first_line) = request.lines().next() else {
        return CodexCallback::Malformed("empty HTTP request".to_string());
    };
    let Some(path) = first_line.split_whitespace().nth(1) else {
        return CodexCallback::Malformed("malformed HTTP request line".to_string());
    };
    let (path_component, query_string) = path.split_once('?').unwrap_or((path, ""));
    if !path_component.eq_ignore_ascii_case("/auth/callback") {
        return CodexCallback::NotCallback;
    }
    if query_string.is_empty() {
        return CodexCallback::Malformed("no query parameters in callback URL".to_string());
    }
    code_state_from_query(query_string)
}

/// Extract `(code, state)` from a URL query string (percent-decoded). Shared by the loopback
/// callback and the pasted-URL fallback so both validate identically. An `error` param wins over
/// `code`/`state`, so an explicit authorization denial surfaces as [`CodexCallback::AuthError`].
fn code_state_from_query(query: &str) -> CodexCallback {
    let mut code = None;
    let mut state = None;
    let mut error_param: Option<String> = None;
    for pair in query.split('&') {
        let Some((key, value)) = pair.split_once('=') else {
            continue;
        };
        let decoded = percent_encoding::percent_decode_str(value)
            .decode_utf8_lossy()
            .into_owned();
        match key {
            "code" => code = Some(decoded),
            "state" => state = Some(decoded),
            "error" => error_param = Some(decoded),
            _ => {}
        }
    }

    if let Some(message) = error_param {
        return CodexCallback::AuthError(message);
    }
    match (code, state) {
        (Some(code), Some(state)) => CodexCallback::Match { code, state },
        _ => CodexCallback::Malformed("callback missing 'code' or 'state' parameter".to_string()),
    }
}

/// Parse a manually pasted callback URL, or a bare `code=...&state=...` query. Accepts the full URL
/// the browser tried to load: strips any `#fragment` and takes the substring after the first `?`.
fn extract_codex_paste(input: &str) -> CodexCallback {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return CodexCallback::Malformed("no callback URL pasted".to_string());
    }
    let before_hash = trimmed.split('#').next().unwrap_or(trimmed);
    let query = match before_hash.split_once('?') {
        Some((_, query)) => query,
        None => before_hash,
    };
    code_state_from_query(query)
}

async fn exchange_codex_code(
    code: &str,
    code_verifier: &str,
    client_id: &str,
    redirect_uri: &str,
) -> anyhow::Result<AuthCredential> {
    use percent_encoding::{NON_ALPHANUMERIC, utf8_percent_encode};
    let encode = |value: &str| utf8_percent_encode(value, NON_ALPHANUMERIC).to_string();
    let body = format!(
        "grant_type=authorization_code&code={}&redirect_uri={}&client_id={}&code_verifier={}",
        encode(code),
        encode(redirect_uri),
        encode(client_id),
        encode(code_verifier),
    );

    let client = reqwest::Client::new();
    let response = client
        .post(CODEX_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("Codex token exchange failed ({}): {}", status, body);
    }

    #[derive(serde::Deserialize)]
    struct CodexTokenResponse {
        id_token: Option<String>,
        access_token: String,
        refresh_token: Option<String>,
    }

    let token: CodexTokenResponse = response.json().await?;
    let account_id = token.id_token.as_deref().and_then(extract_codex_account_id);
    let expires_at = extract_jwt_expiration_millis(&token.access_token);

    Ok(AuthCredential::OAuthToken {
        access_token: token.access_token,
        refresh_token: token.refresh_token,
        expires_at,
        account_id,
    })
}

/// Decode an OpenAI id_token JWT and extract `chatgpt_account_id` from the nested
/// `https://api.openai.com/auth` claim. Returns `None` on any failure.
fn extract_codex_account_id(jwt: &str) -> Option<String> {
    let payload = jwt.split('.').nth(1)?;
    let bytes = URL_SAFE_NO_PAD.decode(payload).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    value
        .get("https://api.openai.com/auth")
        .and_then(|auth| auth.get("chatgpt_account_id"))
        .and_then(|id| id.as_str())
        .map(|id| id.to_string())
}

/// Decode the `exp` claim of a JWT (seconds) and return millis, or `None` if missing/malformed.
fn extract_jwt_expiration_millis(jwt: &str) -> Option<i64> {
    let payload = jwt.split('.').nth(1)?;
    let bytes = URL_SAFE_NO_PAD.decode(payload).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    Some(value.get("exp")?.as_i64()? * 1000)
}

/// Local (no-network) auth status for a stored credential: is the token valid, and when does it
/// expire. Serialized as the `auth` block of `meka account whoami --format json`.
#[derive(serde::Serialize)]
struct AuthStatus {
    valid: bool,
    /// Token expiry as Unix seconds (`None` for API keys / no expiry).
    expires_at: Option<i64>,
    /// Seconds until expiry (negative if already expired).
    expires_in_seconds: Option<i64>,
}

impl AuthStatus {
    fn from_credential(credential: &AuthCredential) -> Self {
        match credential {
            AuthCredential::OAuthToken { expires_at, .. } => {
                // `expires_at` is stored as epoch milliseconds.
                let expires_at = expires_at.map(|millis| millis / 1000);
                let expires_in_seconds =
                    expires_at.map(|secs| secs - chrono::Utc::now().timestamp());
                AuthStatus {
                    valid: expires_in_seconds.is_none_or(|remaining| remaining > 0),
                    expires_at,
                    expires_in_seconds,
                }
            }
            AuthCredential::ApiKey(_) => AuthStatus {
                valid: true,
                expires_at: None,
                expires_in_seconds: None,
            },
        }
    }
}

#[derive(serde::Serialize)]
struct UsageOutput<'a> {
    provider: &'a str,
    #[serde(flatten)]
    usage: &'a crate::provider::AccountUsage,
}

#[derive(serde::Serialize)]
struct WhoamiOutput<'a> {
    provider: &'a str,
    backend: &'a str,
    auth: AuthStatus,
    identity: Option<crate::provider::AccountIdentity>,
}

// ----- `meka account` -----------------------------------------------------------------------
//
// Lives beside the `meka provider` suite rather than in `main.rs`: both read the same profiles and
// the same credentials, and `account` is the read-only view of what `provider` configures. Keeping
// them apart meant a change to profile resolution had to be made in two files.

pub async fn run_account_subcommand(
    session_manager: &crate::session::SessionManager,
    action: &crate::cli::AccountAction,
) -> anyhow::Result<()> {
    let (profile_arg, format) = match action {
        crate::cli::AccountAction::Usage { profile, format } => (profile.clone(), *format),
        crate::cli::AccountAction::Whoami { profile, format } => (profile.clone(), *format),
        crate::cli::AccountAction::Stats { profile, format } => (profile.clone(), *format),
    };

    let token_store = session_manager.token_store();
    let config_file = config::load_config_file_or_err()?;
    let requested = profile_arg.or_else(|| config_file.default_provider.clone());
    let (name, error) = config::select_active_profile(requested, &config_file.providers);
    let name = name.ok_or_else(|| {
        anyhow::anyhow!(error.unwrap_or_else(|| "no provider configured".to_string()))
    })?;
    let profile = config_file
        .providers
        .get(&name)
        .ok_or_else(|| anyhow::anyhow!("provider profile '{}' not found", name))?;
    let credential = token_store
        .load_provider_credential(&name)
        .await?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no stored credential for profile '{}'. Run `meka provider login {}`.",
                name,
                name
            )
        })?;

    let provider = crate::provider::ProviderBuilder::new(
        profile.backend.clone(),
        credential.clone(),
        profile.model.clone().unwrap_or_default(),
    )
    .base_url(profile.base_url.clone())
    .client_id(profile.client_id.clone())
    .oauth_token_url(profile.oauth_token_url.clone())
    .credential_key(Some(name.clone()))
    .token_store(Some(std::sync::Arc::new(session_manager.token_store())))
    .build()?;

    match action {
        crate::cli::AccountAction::Usage { .. } => match provider.fetch_usage().await? {
            Some(usage) => match format {
                crate::cli::OutputFormat::Plain => {
                    print!("{}", crate::render::format_account_usage(&usage));
                }
                crate::cli::OutputFormat::Json => {
                    let out = UsageOutput {
                        provider: &name,
                        usage: &usage,
                    };
                    println!("{}", serde_json::to_string_pretty(&out)?);
                }
            },
            None => {
                eprintln!("Account usage isn't available for provider '{}'.", name);
                std::process::exit(1);
            }
        },
        crate::cli::AccountAction::Whoami { .. } => {
            // The identity call may refresh + rotate the token; re-read afterwards so the auth
            // block reflects the current expiry. A failed identity fetch (e.g. re-login needed)
            // still prints the local auth status so scripts can detect it.
            let identity = provider.fetch_identity().await;
            let fresh = session_manager
                .token_store()
                .load_provider_credential(&name)
                .await
                .ok()
                .flatten()
                .unwrap_or(credential);
            let auth = AuthStatus::from_credential(&fresh);
            let identity = match identity {
                Ok(identity) => identity,
                Err(error) => {
                    tracing::warn!("could not fetch identity: {}", error);
                    None
                }
            };
            let out = WhoamiOutput {
                provider: &name,
                backend: &profile.backend,
                auth,
                identity,
            };
            match format {
                crate::cli::OutputFormat::Plain => print!("{}", format_whoami_plain(&out)),
                crate::cli::OutputFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(&out)?)
                }
            }
            if !out.auth.valid {
                std::process::exit(1);
            }
        }
        crate::cli::AccountAction::Stats { .. } => match provider.fetch_history().await? {
            Some(history) => {
                let out = StatsOutput {
                    provider: &name,
                    history: &history,
                };
                match format {
                    crate::cli::OutputFormat::Plain => print!("{}", format_stats_plain(&out)),
                    crate::cli::OutputFormat::Json => {
                        println!("{}", serde_json::to_string_pretty(&out)?)
                    }
                }
            }
            None => {
                eprintln!("Account history isn't available for provider '{}'.", name);
                std::process::exit(1);
            }
        },
    }
    Ok(())
}

#[derive(serde::Serialize)]
struct StatsOutput<'a> {
    provider: &'a str,
    #[serde(flatten)]
    history: &'a crate::provider::UsageHistory,
}

/// Plain-text (ANSI-free) rendering of `meka account stats`.
fn format_stats_plain(out: &StatsOutput<'_>) -> String {
    use std::fmt::Write as _;
    let history = out.history;
    let mut text = format!("Account history: {}\n", out.provider);
    let row_tokens = |text: &mut String, label: &str, value: Option<i64>| {
        if let Some(value) = value {
            let _ = writeln!(
                text,
                "  {label:<18} {}",
                crate::render::format_token_count(value.max(0) as u64)
            );
        }
    };
    let row_days = |text: &mut String, label: &str, value: Option<i64>| {
        if let Some(value) = value {
            let _ = writeln!(text, "  {label:<18} {value} days");
        }
    };
    if let Some(first) = &history.first_used {
        // Trim an RFC 3339 timestamp to just the date for the human view.
        let date = first.split('T').next().unwrap_or(first);
        let _ = writeln!(text, "  {:<18} {date}", "First used:");
    }
    row_tokens(&mut text, "Lifetime tokens:", history.lifetime_tokens);
    row_tokens(&mut text, "Peak daily:", history.peak_daily_tokens);
    row_days(&mut text, "Current streak:", history.current_streak_days);
    row_days(&mut text, "Longest streak:", history.longest_streak_days);
    if !history.daily.is_empty() {
        let _ = writeln!(text, "  Recent:");
        for day in history.daily.iter().rev().take(7) {
            let _ = writeln!(
                text,
                "    {}  {}",
                day.date,
                crate::render::format_token_count(day.tokens.max(0) as u64)
            );
        }
    }
    text
}

/// Plain-text (ANSI-free) rendering of `meka account whoami`.
fn format_whoami_plain(out: &WhoamiOutput<'_>) -> String {
    use std::fmt::Write as _;
    let mut text = format!("Account: {} ({})\n", out.provider, out.backend);
    let auth = match (out.auth.valid, out.auth.expires_in_seconds) {
        (true, Some(secs)) => format!(
            "valid ({})",
            crate::render::format_duration_short(secs.max(0))
        ),
        (true, None) => "valid".to_string(),
        (false, _) => "EXPIRED: run `meka provider login`".to_string(),
    };
    let _ = writeln!(text, "  Auth:          {auth}");
    if let Some(identity) = &out.identity {
        let row = |text: &mut String, label: &str, value: &Option<String>| {
            if let Some(value) = value {
                let _ = writeln!(text, "  {label:<14} {value}");
            }
        };
        row(&mut text, "Name:", &identity.display_name);
        row(&mut text, "Email:", &identity.email);
        row(&mut text, "Plan:", &identity.plan);
        row(&mut text, "Tier:", &identity.tier);
        row(&mut text, "Subscription:", &identity.subscription_status);
        row(&mut text, "Organization:", &identity.organization);
        row(&mut text, "Role:", &identity.role);
    }
    text
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_auth_status_from_credential() {
        let future = crate::provider::AuthCredential::OAuthToken {
            access_token: "t".into(),
            refresh_token: None,
            // 1 hour out, in epoch millis.
            expires_at: Some((chrono::Utc::now().timestamp() + 3600) * 1000),
            account_id: None,
        };
        let status = AuthStatus::from_credential(&future);
        assert!(status.valid);
        assert!(status.expires_in_seconds.unwrap() > 3000);

        let expired = crate::provider::AuthCredential::OAuthToken {
            access_token: "t".into(),
            refresh_token: None,
            expires_at: Some((chrono::Utc::now().timestamp() - 60) * 1000),
            account_id: None,
        };
        assert!(!AuthStatus::from_credential(&expired).valid);

        // API keys never expire.
        let api = crate::provider::AuthCredential::ApiKey("k".into());
        let status = AuthStatus::from_credential(&api);
        assert!(status.valid);
        assert_eq!(status.expires_at, None);
    }
    use super::*;

    /// A config meka can't read must never be treated as a config that is empty: `open_document`
    /// hands its result to `write_file_atomic`, so "" here truncates the user's real file. This
    /// wiped 159 bytes of profiles and MCP servers off a single non-UTF-8 byte in a comment, while
    /// printing `ok:` and exiting 0.
    #[test]
    fn open_document_refuses_an_unreadable_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.toml"), b"# caf\xe9\n").expect("write");
        // SAFETY: `MEKA_CONFIG_DIR` is process-global; `CONFIG_DIR_ENV_LOCK` serialises every test
        // that touches it, and the guard is held across the whole set → read → clear cycle.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.blocking_lock();
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = open_document();
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = match result {
            Ok(_) => panic!("an unreadable config must not parse as empty"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("failed to read config"),
            "{error}"
        );
    }

    /// An absent file is the one case that legitimately starts from empty: `provider add` on a
    /// fresh install has no config.toml to read yet.
    #[test]
    fn open_document_starts_empty_when_there_is_no_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.blocking_lock();
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = open_document();
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let (_lock, _, document) = result.expect("a missing config is not an error");
        assert!(document.as_table().is_empty());
    }

    async fn memory_token_store() -> TokenStore {
        crate::session::SessionManager::open(Some(std::path::Path::new(":memory:")))
            .await
            .expect("memory store")
            .token_store()
    }

    /// The credential table is keyed by profile name and nothing prunes it, so a hand-deleted
    /// `[providers.<name>]` block leaves a live API key or refresh token behind. This diff is the
    /// only thing that can name one.
    #[tokio::test]
    async fn orphaned_profiles_names_credentials_no_profile_claims() {
        let store = memory_token_store().await;
        for profile in ["work", "archive"] {
            store
                .save_provider_credential(profile, &AuthCredential::ApiKey("key".to_string()))
                .await
                .expect("save");
        }

        // `personal` is configured but never logged in to: the `Authenticated` column's job, not
        // an orphan. The diff runs in one direction only.
        let config_file: config::ConfigFile = toml::from_str(
            "[providers.work]\ntype = \"claude-api\"\nmodel = \"claude-opus-5\"\n\
             [providers.personal]\ntype = \"openai-api\"\nmodel = \"gpt-5.6-sol\"\n",
        )
        .expect("parse config");

        let orphans = orphaned_profiles(&store, &config_file)
            .await
            .expect("diff credentials against profiles");
        assert_eq!(orphans, vec!["archive".to_string()]);
    }

    /// `remove` is the only path that deletes a credential, so requiring a configured profile would
    /// leave a hand-deleted profile's secret unreachable from every surface meka has.
    #[tokio::test]
    async fn remove_deletes_a_credential_whose_profile_is_gone() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            "default_provider = \"work\"\n",
        )
        .expect("write config");
        let store = memory_token_store().await;
        store
            .save_provider_credential("work", &AuthCredential::ApiKey("key".to_string()))
            .await
            .expect("save");

        // SAFETY: `MEKA_CONFIG_DIR` is process-global; `CONFIG_DIR_ENV_LOCK` serialises every test
        // that touches it, and the guard is held across the whole set → run → clear cycle.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = run_remove("work", &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
        result.expect("removing an orphaned credential succeeds");

        assert!(
            store
                .load_provider_credential("work")
                .await
                .expect("load")
                .is_none(),
            "the stored credential must be gone"
        );
        // The profile was already absent, but `default_provider` still pointed at it; that dangling
        // pointer is part of the same leftover.
        let contents = std::fs::read_to_string(dir.path().join("config.toml")).expect("read back");
        assert!(!contents.contains("default_provider"), "{contents}");
    }

    /// Without this, `provider remove typo` reported `removed provider profile 'typo'` and exited
    /// 0 having done nothing at all.
    #[tokio::test]
    async fn remove_refuses_a_name_with_neither_profile_nor_credential() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            "[providers.work]\ntype = \"claude-api\"\nmodel = \"claude-opus-5\"\n",
        )
        .expect("write config");
        let store = memory_token_store().await;

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = run_remove("typo", &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = match result {
            Ok(()) => panic!("removing a name that exists nowhere must fail"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("no provider profile or stored credential"),
            "{error}"
        );
        // The untouched profile is still there: a failed remove must not rewrite anything.
        let contents = std::fs::read_to_string(dir.path().join("config.toml")).expect("read back");
        assert!(contents.contains("[providers.work]"), "{contents}");
    }

    #[test]
    fn test_generate_pkce_pair_challenge_is_sha256_of_verifier() {
        let (verifier, challenge) = generate_pkce_pair();
        let expected = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
        assert_eq!(challenge, expected);
    }

    #[test]
    fn test_generate_state_unique() {
        assert_ne!(generate_state(), generate_state());
    }

    #[test]
    fn test_token_response_extracts_account_uuid() {
        let json = serde_json::json!({
            "access_token": "at",
            "refresh_token": "rt",
            "expires_in": 3600,
            "account": { "uuid": "7194a774-10cb-47f6-a031-78078f9054c9" },
        });
        let token: TokenResponse = serde_json::from_value(json).unwrap();
        assert_eq!(
            token.account.map(|account| account.uuid).as_deref(),
            Some("7194a774-10cb-47f6-a031-78078f9054c9"),
        );
    }

    #[test]
    fn test_token_response_without_account_is_none() {
        let json = serde_json::json!({
            "access_token": "at",
            "refresh_token": "rt",
            "expires_in": 3600,
        });
        let token: TokenResponse = serde_json::from_value(json).unwrap();
        assert!(token.account.is_none());
    }

    #[test]
    fn test_build_authorize_url_contains_params() {
        let url = build_authorize_url("cid", "challenge", "state").unwrap();
        assert!(url.starts_with(AUTHORIZE_URL));
        assert!(url.contains("client_id=cid"));
        assert!(url.contains("code_challenge=challenge"));
        assert!(url.contains("state=state"));
        assert!(url.contains("code_challenge_method=S256"));
    }

    #[test]
    fn test_build_codex_authorize_url_contains_required_params() {
        let url = build_codex_authorize_url(
            "app_test",
            "ch",
            "st",
            "http://localhost:1455/auth/callback",
        )
        .unwrap();
        assert!(url.starts_with(CODEX_AUTHORIZE_URL));
        assert!(url.contains("client_id=app_test"));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains("id_token_add_organizations=true"));
        assert!(url.contains("originator=meka_cli"));
    }

    #[test]
    fn test_validate_backend() {
        assert!(validate_backend("claude-oauth").is_ok());
        assert!(validate_backend("bogus").is_err());
    }

    #[test]
    fn test_default_model_for_known_backends() {
        assert_eq!(default_model_for("claude-api"), Some("claude-opus-5"));
        assert_eq!(default_model_for("claude-oauth"), Some("claude-opus-5"));
        assert_eq!(default_model_for("openai-api"), Some("gpt-5.6-sol"));
        assert_eq!(default_model_for("openai-codex"), Some("gpt-5.6-sol"));
        assert_eq!(default_model_for("unknown"), None);
        // Every supported backend has a default, so the prompt never forces a manual answer.
        for backend in SUPPORTED_PROVIDERS {
            assert!(
                default_model_for(backend).is_some(),
                "{} has no default model",
                backend
            );
        }
    }

    #[test]
    fn test_parse_codex_callback_query_match_and_decode() {
        let request = "GET /auth/callback?code=hello%20world&state=s%23t HTTP/1.1\r\n\r\n";
        match parse_codex_callback_query(request) {
            CodexCallback::Match { code, state } => {
                assert_eq!(code, "hello world");
                assert_eq!(state, "s#t");
            }
            _ => panic!("expected Match"),
        }
    }

    #[test]
    fn test_parse_codex_callback_query_non_callback_and_error() {
        assert!(matches!(
            parse_codex_callback_query("GET /favicon.ico HTTP/1.1\r\n\r\n"),
            CodexCallback::NotCallback
        ));
        match parse_codex_callback_query("GET /auth/callback?error=access_denied HTTP/1.1\r\n\r\n")
        {
            CodexCallback::AuthError(message) => assert_eq!(message, "access_denied"),
            _ => panic!("expected AuthError"),
        }
    }

    #[test]
    fn test_extract_codex_paste_full_url() {
        // A real redirect URL: base64url code with '.'/'-'/'_', '+' in scope, all left intact.
        let url = "http://localhost:1455/auth/callback?code=ac_K0pPCDWiHyd5jWO_FqWeDB8-52rj9Dw-YxgEnqp5HAo.zDNcq5vdPneTieVAgYERv7yr5AhoiUbMAgpMjEuGnhs&scope=openid+profile+email+offline_access+api.connectors.read+api.connectors.invoke&state=uw-OxzrtaH6ZqtaJtuN7dvDZY0eM5ka7yn_zshisEi0";
        match extract_codex_paste(url) {
            CodexCallback::Match { code, state } => {
                assert_eq!(
                    code,
                    "ac_K0pPCDWiHyd5jWO_FqWeDB8-52rj9Dw-YxgEnqp5HAo.zDNcq5vdPneTieVAgYERv7yr5AhoiUbMAgpMjEuGnhs"
                );
                assert_eq!(state, "uw-OxzrtaH6ZqtaJtuN7dvDZY0eM5ka7yn_zshisEi0");
            }
            _ => panic!("expected Match"),
        }
    }

    #[test]
    fn test_extract_codex_paste_bare_query_and_trim_and_fragment() {
        match extract_codex_paste("  code=abc&state=xyz#frag \n") {
            CodexCallback::Match { code, state } => {
                assert_eq!(code, "abc");
                assert_eq!(state, "xyz");
            }
            _ => panic!("expected Match"),
        }
    }

    #[test]
    fn test_extract_codex_paste_error_and_missing() {
        match extract_codex_paste("http://localhost:1455/auth/callback?error=access_denied") {
            CodexCallback::AuthError(message) => assert_eq!(message, "access_denied"),
            _ => panic!("expected AuthError"),
        }
        assert!(matches!(
            extract_codex_paste("http://localhost:1455/auth/callback?code=only"),
            CodexCallback::Malformed(_)
        ));
    }

    #[test]
    fn test_extract_codex_account_id_and_expiration() {
        let payload = serde_json::json!({
            "exp": 1_700_000_000,
            "https://api.openai.com/auth": { "chatgpt_account_id": "ws-1" }
        });
        let body = URL_SAFE_NO_PAD.encode(payload.to_string().as_bytes());
        let jwt = format!("h.{}.s", body);
        assert_eq!(extract_codex_account_id(&jwt).as_deref(), Some("ws-1"));
        assert_eq!(extract_jwt_expiration_millis(&jwt), Some(1_700_000_000_000));
    }

    #[test]
    fn test_upsert_profile_document_first_profile_becomes_default() {
        let mut document = toml_edit::DocumentMut::new();
        upsert_profile_document(
            &mut document,
            "work",
            "openai-api",
            "gpt-4o",
            Some("http://localhost:1234/v1"),
        )
        .expect("upsert");
        // The rendered TOML must parse back into the runtime config with the profile and default.
        let config: config::ConfigFile =
            toml::from_str(&document.to_string()).expect("re-parse config");
        assert_eq!(config.default_provider.as_deref(), Some("work"));
        let profile = config.providers.get("work").expect("profile present");
        assert_eq!(profile.backend, "openai-api");
        assert_eq!(profile.model.as_deref(), Some("gpt-4o"));
        assert_eq!(
            profile.base_url.as_deref(),
            Some("http://localhost:1234/v1")
        );
    }

    #[test]
    fn test_upsert_profile_document_second_profile_keeps_existing_default() {
        let mut document = toml_edit::DocumentMut::new();
        upsert_profile_document(&mut document, "work", "claude-oauth", "claude-x", None)
            .expect("upsert work");
        upsert_profile_document(&mut document, "personal", "openai-api", "gpt-4o", None)
            .expect("upsert personal");
        let config: config::ConfigFile =
            toml::from_str(&document.to_string()).expect("re-parse config");
        // The default must remain the first profile, not silently flip to the newest one.
        assert_eq!(config.default_provider.as_deref(), Some("work"));
        assert!(config.providers.contains_key("work"));
        assert!(config.providers.contains_key("personal"));
        // base_url is omitted when None.
        assert!(!document.to_string().contains("base_url"));
    }

    #[test]
    fn test_remove_profile_document_clears_dangling_default() {
        let mut document = toml_edit::DocumentMut::new();
        upsert_profile_document(&mut document, "work", "claude-oauth", "claude-x", None)
            .expect("upsert work");
        upsert_profile_document(&mut document, "personal", "openai-api", "gpt-4o", None)
            .expect("upsert personal");
        remove_profile_document(&mut document, "work");
        let config: config::ConfigFile =
            toml::from_str(&document.to_string()).expect("re-parse config");
        assert!(!config.providers.contains_key("work"));
        assert!(config.providers.contains_key("personal"));
        // `work` was the default; removing it must drop the dangling pointer rather than leave it.
        assert!(config.default_provider.is_none());
    }
}
