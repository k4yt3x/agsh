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
    AuthCredential, DEFAULT_CHATGPT_SUBSCRIPTION_CLIENT_ID, DEFAULT_CLAUDE_SUBSCRIPTION_CLIENT_ID,
    SUPPORTED_PROVIDERS,
};
use crate::{cli::ProviderAction, config, session::TokenStore};

const REDIRECT_URI: &str = "https://platform.claude.com/oauth/code/callback";
const AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
const TOKEN_URL: &str = "https://api.anthropic.com/v1/oauth/token";
const SCOPES: &str =
    "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers";

/// `chatgpt-subscription` OAuth flow constants. Mirror Codex's first-party CLI: the authorization
/// server lives at `auth.openai.com`, the redirect listener binds on `localhost:1455`.
const CODEX_AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
const CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CODEX_REDIRECT_PORT: u16 = 1455;
const CODEX_SCOPES: &str =
    "openid profile email offline_access api.connectors.read api.connectors.invoke";
const CODEX_CALLBACK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(180);

/// Dispatch a `meka provider` subcommand.
pub async fn run(
    action: &ProviderAction,
    // Taken whole rather than as a bare `TokenStore` because `remove` has to say how many sessions
    // it is about to strand, and only the session table can answer that.
    session_manager: &crate::session::SessionManager,
) -> anyhow::Result<()> {
    let token_store = &session_manager.token_store();
    match action {
        ProviderAction::Add {
            name,
            r#type,
            model,
            base_url,
            thinking,
            context_window,
            effort,
            thinking_budget,
            vision,
            max_output_tokens,
            redact_thinking,
            oauth_token_url,
            client_id,
            api_key_stdin,
        } => {
            run_add(
                name,
                r#type.as_deref(),
                model.clone(),
                base_url.clone(),
                ProfileTuning {
                    thinking: *thinking,
                    context_window: *context_window,
                    effort: effort.clone(),
                    thinking_budget: *thinking_budget,
                    vision: *vision,
                    max_output_tokens: *max_output_tokens,
                    redact_thinking: *redact_thinking,
                    oauth_token_url: oauth_token_url.clone(),
                    client_id: client_id.clone(),
                },
                *api_key_stdin,
                token_store,
            )
            .await
        }
        ProviderAction::List => run_list(token_store).await,
        ProviderAction::Set {
            name,
            key,
            value,
            unset,
        } => run_set(name, key, value.as_deref(), *unset),
        ProviderAction::Use { name } => run_use(name),
        ProviderAction::Remove { name } => run_remove(name, token_store, session_manager).await,
        ProviderAction::Login {
            name,
            api_key_stdin,
        } => run_login(name, *api_key_stdin, token_store).await,
    }
}

/// The settings `provider add` can write beyond the four it always asks for. Each is `None` when
/// the flag was absent and the user declined (or skipped) the advanced prompt, which leaves the key
/// out of the profile entirely so the documented default applies.
///
/// Only the first three are ever *prompted* for. The rest are flag-only, so a profile of any shape
/// can be created in one non-interactive command without making the interactive path a nine-step
/// wizard for settings most users never state. [`resolve_tuning`] says the same thing from the
/// other side: its short-circuit tests those three and no others, because a rare flag must not
/// silently skip the prompt for the common ones.
#[derive(Default)]
struct ProfileTuning {
    thinking: Option<crate::provider::ThinkingMode>,
    context_window: Option<u64>,
    effort: Option<String>,
    thinking_budget: Option<u64>,
    vision: Option<bool>,
    max_output_tokens: Option<u64>,
    redact_thinking: Option<bool>,
    oauth_token_url: Option<String>,
    client_id: Option<String>,
}

async fn run_add(
    name: &str,
    type_flag: Option<&str>,
    model_flag: Option<String>,
    base_url_flag: Option<String>,
    tuning_flags: ProfileTuning,
    api_key_stdin: bool,
    token_store: &TokenStore,
) -> anyhow::Result<()> {
    if name.trim().is_empty() {
        anyhow::bail!("profile name cannot be empty");
    }
    // Every prompt below reads the same stdin the key is piped on, so one that fires under
    // `--api-key-stdin` eats the secret and then fails with "API key cannot be empty", naming the
    // wrong field entirely; with two lines piped, line one would be written to `config.toml` as a
    // `base_url`. `resolve_tuning` already suppressed the advanced prompt for this reason; the
    // three above it -- backend, model and base URL -- did not. The two required ones are refused
    // up front rather than prompted into a pipe, and the optional base URL takes the backend
    // default.
    if api_key_stdin {
        // `--type` first and alone, because neither check below can be asked without knowing the
        // backend.
        let Some(backend) = type_flag else {
            anyhow::bail!(
                "`--api-key-stdin` reads the key from stdin, so it cannot prompt for --type. Pass \
                 it as a flag."
            );
        };
        let backend = validate_backend(backend)?;
        // Ahead of the missing-`--model` check so a subscription profile hears the real objection
        // rather than being asked for a flag that would not have saved it. The same refusal
        // `meka provider login` makes: `acquire_credential` ignores this flag for a backend that
        // opens a browser, so a script piping a key would hang on a flow it cannot see while the
        // key it piped went unread.
        if !matches!(credential_kind(backend), Some(CredentialKind::ApiKey)) {
            anyhow::bail!(
                "'{}' authenticates through the browser and has no API key to read from stdin. \
                 Run `meka provider add {} --type {}` without `--api-key-stdin`.",
                backend,
                name,
                backend
            );
        }
        if model_flag.is_none() {
            anyhow::bail!(
                "`--api-key-stdin` reads the key from stdin, so it cannot prompt for --model. \
                 Pass it as a flag. `--base-url` is optional; omitting it accepts the backend \
                 default."
            );
        }
    }
    // Hard-fails on an unparseable config rather than warning: this guard is the only thing
    // standing between `provider add <existing>` and `upsert_profile_document` replacing the
    // profile's table, and an empty parsed map defeats it.
    let existing = config::load_config_file_or_err()?;
    if existing.providers.contains_key(name) {
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
        // Under `--api-key-stdin` the guard above has already established that the two required
        // flags were given; this one is optional, so an absent value takes the backend default
        // rather than prompting into the pipe the key is on.
        None if api_key_stdin => None,
        None => {
            // Shows the endpoint an empty answer accepts, matching the model prompt above. Unlike
            // the model, an empty answer writes nothing: pinning the default into the profile would
            // freeze it, and this is the one setting where meka's own value is the right one for
            // almost everybody.
            let prompt = match crate::provider::default_base_url(&backend) {
                Some(default) => format!("API base URL [{}]: ", default),
                None => "API base URL (leave empty for default): ".to_string(),
            };
            let input = prompt_line(&prompt)?;
            (!input.is_empty()).then_some(input)
        }
    };

    let tuning = resolve_tuning(
        tuning_flags,
        &backend,
        name,
        existing
            .session
            .as_ref()
            .and_then(|session| session.context_window),
        existing
            .thinking
            .as_ref()
            .and_then(|thinking| thinking.budget_tokens),
        api_key_stdin,
    )?;

    // Acquire the credential last: the Codex OAuth login races a pasted-callback-URL reader against
    // the loopback callback, and if the callback wins it can leave a stdin read parked. Keeping the
    // interactive prompts above (which read stdin) before this ensures nothing reads stdin after.
    let credential = acquire_credential(&backend, api_key_stdin, None).await?;

    // The profile before the secret, so the half that lands first is the visible half. A config
    // write can fail for ordinary reasons -- a read-only directory, a full disk -- and doing it
    // second left a credential in the database that no profile named: `provider list` reports it as
    // an orphan, but only if the user thinks to look. This way round, the failure leaves a profile
    // with no credential, which the next run refuses by name and tells you to `meka provider
    // login`.
    write_profile(name, &backend, model.as_str(), base_url.as_deref(), &tuning)?;
    token_store
        .save_provider_credential(name, &credential)
        .await?;

    tracing::info!("added provider profile '{}'", name);
    Ok(())
}

async fn run_login(
    name: &str,
    api_key_stdin: bool,
    token_store: &TokenStore,
) -> anyhow::Result<()> {
    let config_file = config::load_config_file_or_err()?;
    let Some(profile) = config_file.providers.get(name) else {
        anyhow::bail!(
            "no provider profile named '{}'. Run `meka provider add {}` to create it.",
            name,
            name
        );
    };
    // Before the guard below, which asks `credential_kind` a question it answers `None` to for a
    // backend it does not recognise. Without this, a typo'd `type` was diagnosed as a browser login
    // and the user was sent to run the command again without the flag, only to meet the real error
    // then. `run_add` has always validated first; this is the same order.
    validate_backend(&profile.backend)?;
    // `acquire_credential` ignores the flag for a backend that logs in through the browser, and a
    // script piping a key to one would hang on an OAuth flow it cannot see. The profile names its
    // backend, so this is answerable before anything opens.
    if api_key_stdin
        && !matches!(
            credential_kind(&profile.backend),
            Some(CredentialKind::ApiKey)
        )
    {
        anyhow::bail!(
            "profile '{}' is a '{}' profile, which authenticates through the browser and has no \
             API key to read from stdin. Run `meka provider login {}` without `--api-key-stdin`.",
            name,
            profile.backend,
            name
        );
    }
    let credential = acquire_credential(
        &profile.backend,
        api_key_stdin,
        profile.client_id.as_deref(),
    )
    .await?;
    token_store
        .save_provider_credential(name, &credential)
        .await?;
    tracing::info!("re-authenticated provider profile '{}'", name);
    Ok(())
}

async fn run_remove(
    name: &str,
    token_store: &TokenStore,
    session_manager: &crate::session::SessionManager,
) -> anyhow::Result<()> {
    // Deliberately does not require a configured profile: this is the only path that deletes a
    // credential, so it has to work on one whose `[providers.<name>]` block was deleted by hand.
    // Both sides are read before either is touched so the confirmation can say which of them
    // actually existed, and so a typo'd name fails instead of reporting a removal that removed
    // nothing. `open_document` rather than the parsed config on purpose: `remove` must still run on
    // a config.toml that meka can't deserialize, since it is one of the ways such a file gets
    // repaired.
    //
    // Asks whether the row is *there*, not whether it parses. `load_provider_credential` fails on a
    // credential it cannot deserialize, which stopped the command before the delete -- so the one
    // surface that removes a corrupt row refused to, on the grounds that it was corrupt.
    let has_credential = token_store
        .list_credential_profiles()
        .await?
        .iter()
        .any(|profile| profile == name);

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
        profile_names(&document)
            .iter()
            .any(|profile| profile == name)
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
    let was_default = document
        .get("default_provider")
        .and_then(|item| item.as_str())
        == Some(name);
    let removed_profile = remove_profile_document(&mut document, name);
    config::write_file_atomic(&path, &document.to_string())?;

    // Losing the default is not a detail: with two or more profiles left, nothing picks one, and
    // the next `meka` with no `--provider` stops with an error about a setting the user did not
    // knowingly change. Said at `warn!` rather than `info!` because it needs a follow-up action
    // and is visible at the default verbosity.
    if was_default {
        let remaining = profile_names(&document);
        if remaining.len() > 1 {
            tracing::warn!(
                "'{}' was the default provider, so `default_provider` is now unset and no profile \
                 is picked for a new session. Run `meka provider use <name>` to choose one of: {}",
                name,
                remaining.join(", ")
            );
        }
    }

    // Sessions pinned to the profile are what makes a removal consequential, and they are silent
    // otherwise: the refusal arrives whenever the user next resumes one, which may be days later
    // and in another directory. The store is already open, so this costs one count.
    match session_manager.count_sessions_on_provider(name).await {
        Ok(0) => {}
        Ok(pinned) => tracing::warn!(
            "{} session(s) run on '{}' and will refuse to resume until it is configured again. \
             Move one with `meka -r <id> --provider <name>`",
            pinned,
            name
        ),
        // Not worth failing the removal over: the profile and its credential are already gone, and
        // this is advisory.
        Err(error) => tracing::debug!("could not count sessions on '{}': {}", name, error),
    }

    // What the write actually did, not what a probe predicted: the two disagreed whenever
    // `providers` was spelled in a way `as_table` could not see, and the message that came out was
    // the opposite of the truth.
    if removed_profile {
        tracing::info!("removed provider profile '{}'", name);
    } else {
        tracing::info!(
            "cleared the stored credential for '{}'; no profile was configured",
            name
        );
    }
    Ok(())
}

/// The settings `meka provider set` will write, in the order `provider add` prompts for them and
/// `config.toml` documents them.
///
/// `type` is deliberately absent. The stored credential was acquired *for* the current backend and
/// differs in kind between them (an API key against an OAuth bundle), so rewriting it in place
/// leaves a profile whose credential cannot serve it: a state no other door can produce, and the
/// one thing this command must not be able to do. `device_id` is absent for the opposite reason -
/// meka resolves and persists it itself ([`crate::config::resolve_device_id`]), so offering it
/// invites a user to overwrite bookkeeping they do not own.
const SETTABLE_PROFILE_KEYS: &[&str] = &[
    "model",
    "base_url",
    "context_window",
    "vision",
    "max_output_tokens",
    "effort",
    "thinking",
    "thinking_budget",
    "redact_thinking",
    "oauth_token_url",
    "client_id",
];

/// Parse one `key`/`value` pair into the TOML value the profile should carry.
///
/// Split from the write so the whole vocabulary is testable without a filesystem, and so a value
/// that cannot be parsed is refused before the config lock is taken rather than after.
///
/// Each key parses the way the matching `provider add` flag does, `thinking` through the same
/// `ValueEnum` for the reason [`resolve_tuning`]'s prompt gives: a second hand-written match would
/// be a second thing to keep in step with the enum.
fn parse_profile_value(key: &str, value: &str) -> anyhow::Result<toml_edit::Value> {
    let integer = |what: &str| -> anyhow::Result<toml_edit::Value> {
        let parsed: u64 = value
            .parse()
            .map_err(|_| anyhow::anyhow!("{} must be a whole number, got '{}'", what, value))?;
        Ok(toml_edit::Value::from(toml_integer(what, parsed)?))
    };
    let boolean = |what: &str| -> anyhow::Result<toml_edit::Value> {
        match value {
            "true" => Ok(toml_edit::Value::from(true)),
            "false" => Ok(toml_edit::Value::from(false)),
            other => anyhow::bail!("{} must be true or false, got '{}'", what, other),
        }
    };
    match key {
        "model" | "base_url" | "effort" | "oauth_token_url" | "client_id" => {
            Ok(toml_edit::Value::from(value))
        }
        "context_window" => integer("context_window"),
        "max_output_tokens" => integer("max_output_tokens"),
        "thinking_budget" => integer("thinking_budget"),
        "vision" => boolean("vision"),
        "redact_thinking" => boolean("redact_thinking"),
        "thinking" => {
            let mode = <crate::provider::ThinkingMode as clap::ValueEnum>::from_str(value, true)
                .map_err(|_| {
                    anyhow::anyhow!(
                        "'{}' is not a thinking mode. Expected adaptive, budgeted, or off.",
                        value
                    )
                })?;
            Ok(toml_edit::Value::from(mode.as_str()))
        }
        other => anyhow::bail!(
            "'{}' is not a profile setting. Settable: {}.{}",
            other,
            SETTABLE_PROFILE_KEYS.join(", "),
            unsettable_key_hint(other)
        ),
    }
}

/// Why a key that exists on the profile still cannot be written here.
///
/// An empty string for anything else, so the refusal above reads the same either way. Worth saying
/// rather than leaving to the list: a user who typed `type` did not typo, and "settable: model,
/// base_url, ..." alone would read as though meka had simply forgotten it.
fn unsettable_key_hint(key: &str) -> String {
    match key {
        "type" => {
            " Changing `type` would leave the profile's stored credential, which was acquired \
                   for the current backend, unable to serve it; use `meka provider remove` and \
                   `meka provider add` instead."
                .to_string()
        }
        "device_id" => {
            " `device_id` is meka's own, resolved and persisted per profile.".to_string()
        }
        _ => String::new(),
    }
}

/// Write one key into `[providers.<name>]`, or remove it when `value` is `None`.
///
/// Answers whether the profile was there to change, which the caller turns into the refusal: a
/// silent no-op on a mistyped profile name is exactly the failure `meka provider remove` was
/// carrying before it started reporting the same thing.
///
/// A field-level edit rather than [`upsert_profile_document`]'s whole-table replace, so `toml_edit`
/// keeps every other key, its ordering, and any comment the user wrote beside it. Replacing the
/// table would silently eat all three on every `set`.
fn set_profile_field(
    document: &mut toml_edit::DocumentMut,
    name: &str,
    key: &str,
    value: Option<toml_edit::Value>,
) -> bool {
    let Some(profile) = document
        .get_mut("providers")
        .and_then(|item| item.as_table_like_mut())
        .and_then(|providers| providers.get_mut(name))
        .and_then(|item| item.as_table_like_mut())
    else {
        return false;
    };
    match value {
        Some(mut value) => {
            // A key's surroundings live in two places, and `insert` clears both. The *value*'s
            // decor holds what follows it on the line, so losing it deleted the trailing `# note`
            // beside the model being changed. The *key*'s leaf decor holds everything before it,
            // which is where `toml_edit` puts whole-line comments and blank lines above the key --
            // so losing that deleted the paragraph a user had written to explain the setting, and
            // the blank line separating it from the one above. Both are captured before the insert
            // and put back after, because `insert` is what clears them.
            if let Some(existing) = profile.get(key).and_then(|item| item.as_value()) {
                *value.decor_mut() = existing.decor().clone();
            }
            let key_decor = profile.key(key).map(|key| key.leaf_decor().clone());
            profile.insert(key, toml_edit::Item::Value(value));
            if let (Some(decor), Some(mut written)) = (key_decor, profile.key_mut(key)) {
                *written.leaf_decor_mut() = decor;
            }
        }
        None => {
            profile.remove(key);
        }
    }
    true
}

/// Every profile key that only means something to the Anthropic Messages request shape.
const THINKING_ONLY_PROFILE_KEYS: &[&str] = &["thinking", "thinking_budget", "redact_thinking"];

/// Refuse one of those keys on a profile whose backend will never send it.
///
/// [`resolve_tuning`] drops such a flag on `provider add` with a warning, on the grounds that
/// writing it "would produce a setting that reads plausibly and does nothing". `set` reaches the
/// same outcome by refusing rather than dropping, and the difference is deliberate: `add` is
/// building a bundle out of many flags, where ignoring one inapplicable knob and saying so is
/// proportionate, whereas `set` exists to write exactly one key, so dropping it would mean
/// reporting success for a command that did nothing at all.
///
/// Read from the document being written rather than from a parameter, because `set type` is refused
/// and so the backend cannot be the thing this command is changing. Asked only of a write:
/// `run_set` skips it for `--unset`, which can only be removing such a key.
fn refuse_an_inert_thinking_key(
    document: &toml_edit::DocumentMut,
    name: &str,
    key: &str,
) -> anyhow::Result<()> {
    if !THINKING_ONLY_PROFILE_KEYS.contains(&key) {
        return Ok(());
    }
    let backend = document
        .get("providers")
        .and_then(|item| item.as_table_like())
        .and_then(|providers| providers.get(name))
        .and_then(|item| item.as_table_like())
        .and_then(|profile| profile.get("type"))
        .and_then(|item| item.as_str())
        .unwrap_or_default();
    // An unrecognised backend is left alone: `validate_backend` is the door that reports that, and
    // guessing here would refuse a key over a typo in a different one.
    if backend.is_empty()
        || !crate::provider::SUPPORTED_PROVIDERS.contains(&backend)
        || crate::provider::backend_takes_thinking(backend)
    {
        return Ok(());
    }
    anyhow::bail!(
        "'{}' is an Anthropic Messages request field, and profile '{}' has backend '{}', which \
         never sends one. Nothing was written.",
        key,
        name,
        backend
    )
}

/// Refuse a key this command will not write, naming the ones it will.
///
/// Asked before the value is looked at, so `--unset modle` is refused for the same reason and in
/// the same words as `set modle x`. Checking it inside the value branch instead left `--unset` to
/// accept any spelling and report success, having changed nothing.
fn ensure_settable_key(key: &str) -> anyhow::Result<()> {
    if SETTABLE_PROFILE_KEYS.contains(&key) {
        return Ok(());
    }
    anyhow::bail!(
        "'{}' is not a profile setting. Settable: {}.{}",
        key,
        SETTABLE_PROFILE_KEYS.join(", "),
        unsettable_key_hint(key)
    )
}

/// `meka provider set <name> <key> <value>`, or `--unset` to drop the key.
fn run_set(name: &str, key: &str, value: Option<&str>, unset: bool) -> anyhow::Result<()> {
    ensure_settable_key(key)?;
    // Clap's `conflicts_with` rules out "both"; this is the other half, which it cannot express.
    let parsed = match (value, unset) {
        (Some(value), _) => Some(parse_profile_value(key, value)?),
        (None, true) => None,
        (None, false) => anyhow::bail!(
            "`meka provider set {} {}` needs a value, or `--unset` to remove the setting",
            name,
            key
        ),
    };

    // `_lock` is held to the end of the function, so the read, the validation and the write are one
    // critical section.
    let (_lock, path, mut document) = open_document()?;
    let before = document.to_string();
    if !set_profile_field(&mut document, name, key, parsed) {
        let config_file = config::load_config_file_or_err()?;
        anyhow::bail!(
            "no provider profile named '{}' (configured: {})",
            name,
            join_profile_names(&config_file)
        );
    }
    // Only a write is refused. `--unset` on such a key moves the profile *towards* the state this
    // guards, and a hand-edited file is the one place an inert key can already be sitting, so
    // refusing to remove it would leave no door that could.
    if value.is_some() {
        refuse_an_inert_thinking_key(&document, name, key)?;
    }

    // Asked of the document this command is about to write, not of the one on disk, so neither
    // write door can leave behind a profile the other would have refused. Checked here rather than
    // at the next run because a config that fails to start is a worse answer than a command that
    // declines, and the value that broke it is in front of the user right now.
    let after = document.to_string();
    refuse_a_profile_that_cannot_run(&before, &after, name)?;

    config::write_file_atomic(&path, &after)?;
    match value {
        Some(value) => tracing::info!("set {} = {} on profile '{}'", key, value, name),
        None => tracing::info!("cleared {} on profile '{}'", key, name),
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
        // The error arm is its own answer, not a "no". `load_provider_credential` fails on a row it
        // cannot deserialize, and reporting that as "never logged in" sends the user to
        // `meka provider login`, which writes a new credential over a row they were never told was
        // corrupt.
        let authed = match token_store.load_provider_credential(name).await {
            Ok(Some(_)) => "yes",
            Ok(None) => "no",
            Err(error) => {
                tracing::warn!(
                    "could not read the stored credential for '{}': {}",
                    name,
                    error
                );
                "unreadable"
            }
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
    // A `default_provider` naming nothing renders as a table with no `*`, which is exactly what "no
    // default set" looks like, and the next `meka` run then fails on a setting the user believes is
    // fine. This listing is where they come to check, so it is where the discrepancy belongs.
    if let Some(default) = default
        && !config_file.providers.contains_key(default)
    {
        eprintln!(
            "`default_provider` names '{}', which is not configured. Run `meka provider use \
             <name>` to point it at one of the profiles above.",
            crate::render::sanitize_for_display(default)
        );
    }
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
        "anthropic-messages" | "claude-subscription" => Some("claude-opus-5"),
        "openai-chat-completions" | "openai-responses" | "chatgpt-subscription" => {
            Some("gpt-5.6-sol")
        }
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

/// How a backend proves who it is: an interactive OAuth flow, or a key the user pastes.
///
/// Split out from [`acquire_credential`] so the "every supported backend is accounted for" property
/// can be tested without running a login. It used to be a bare match arm ending in
/// `unreachable!()`, which was reachable: [`validate_backend`] accepts anything in
/// [`crate::provider::SUPPORTED_PROVIDERS`], so adding a backend there and forgetting this match
/// panicked at the credential step rather than failing to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CredentialKind {
    /// A Claude subscription login, against Anthropic's authorization server.
    ClaudeLogin,
    /// A ChatGPT subscription login, against OpenAI's.
    ChatGptLogin,
    /// A key the user supplies, for any endpoint the backend's protocol reaches.
    ApiKey,
}

/// Which flow acquires this backend's credential, naming the *specific* login rather than "OAuth".
///
/// The distinction is load-bearing: the two subscription flows use different client IDs, scopes and
/// callback handling, and they are not interchangeable. An earlier shape had one `OAuth` variant
/// with the vendor picked by a string test inside the dispatch arm, which meant a third OAuth
/// backend would silently receive OpenAI's login while still satisfying a test that only asserted
/// *some* kind existed. Making the variant carry the vendor moves that into the match.
fn credential_kind(backend: &str) -> Option<CredentialKind> {
    match backend {
        "claude-subscription" => Some(CredentialKind::ClaudeLogin),
        "chatgpt-subscription" => Some(CredentialKind::ChatGptLogin),
        "anthropic-messages" | "openai-chat-completions" | "openai-responses" => {
            Some(CredentialKind::ApiKey)
        }
        _ => None,
    }
}

/// Acquire a credential for `backend`: run the OAuth flow for OAuth backends, or read an API key
/// (from stdin when `api_key_stdin`, else an interactive prompt) for key backends.
async fn acquire_credential(
    backend: &str,
    api_key_stdin: bool,
    client_id: Option<&str>,
) -> anyhow::Result<AuthCredential> {
    match credential_kind(backend) {
        Some(CredentialKind::ClaudeLogin) => claude_login(client_id).await,
        Some(CredentialKind::ChatGptLogin) => codex_login(client_id).await,
        Some(CredentialKind::ApiKey) => {
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
        // `validate_backend` rejects with the supported list; the `?` is what actually returns.
        None => {
            validate_backend(backend)?;
            anyhow::bail!("no credential flow is defined for backend '{}'", backend)
        }
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
/// this, auto-vivifying `document["providers"][name]` produces an *inline* table, which renders the
/// whole block on one line.
///
/// A `providers` that is present but written some other way is refused, not replaced. Both
/// spellings deserialize, so `provider list` and the duplicate guard see the same profiles either
/// way; treating "not a header table" as "absent" overwrote the lot. `providers = { work = … }`
/// plus one `meka provider add home` left a file naming only `home`, with `work`'s credential
/// orphaned in the database and `default_provider` pointing at a profile that was no longer there,
/// silently and with exit 0.
fn ensure_providers_table(
    document: &mut toml_edit::DocumentMut,
) -> anyhow::Result<&mut toml_edit::Table> {
    if document.get("providers").is_none() {
        let mut table = toml_edit::Table::new();
        // Implicit so the parent emits `[providers.<name>]` headers rather than a bare
        // `[providers]`.
        table.set_implicit(true);
        document["providers"] = toml_edit::Item::Table(table);
    }
    document
        .get_mut("providers")
        .and_then(|item| item.as_table_mut())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "`providers` in config.toml is not a section, so meka cannot add a profile to it \
                 without rewriting the rest. Spell each profile as its own `[providers.<name>]` \
                 section and run this again"
            )
        })
}

/// The configured profile names, whichever way `providers` is spelled.
///
/// `as_table_like` rather than `as_table` because an inline `providers = { … }` deserializes and so
/// is a config meka runs on; a probe that could not see it reported profiles as absent while they
/// were in the file.
/// Narrow a profile's `u64` setting to the `i64` a TOML integer actually is.
///
/// TOML has one integer type and it is signed 64-bit, so a `u64` past `i64::MAX` has no
/// representation at all. `as i64` wrapped it silently: `provider add x --context-window
/// 18446744073709551615` wrote `context_window = -1` and exited 0, after which every meka command
/// refused the file with `invalid value: integer -1, expected u64` -- including the `provider set`
/// that would have repaired it, leaving `provider remove` or a hand-edit as the only way out.
/// Refused here, before anything is written.
fn toml_integer(field: &str, value: u64) -> anyhow::Result<i64> {
    i64::try_from(value).map_err(|_| {
        anyhow::anyhow!(
            "{} must be at most {}, the largest integer TOML can represent; got {}.",
            field,
            i64::MAX,
            value
        )
    })
}

fn profile_names(document: &toml_edit::DocumentMut) -> Vec<String> {
    document
        .get("providers")
        .and_then(|item| item.as_table_like())
        .map(|providers| providers.iter().map(|(name, _)| name.to_string()).collect())
        .unwrap_or_default()
}

/// Insert or replace `[providers.<name>]` in `document`, defaulting to it if no `default_provider`
/// is set yet. Pure mutation so it can be unit-tested without touching the filesystem.
fn upsert_profile_document(
    document: &mut toml_edit::DocumentMut,
    name: &str,
    backend: &str,
    model: &str,
    base_url: Option<&str>,
    tuning: &ProfileTuning,
) -> anyhow::Result<()> {
    let mut profile = toml_edit::Table::new();
    profile.insert("type", toml_edit::value(backend));
    profile.insert("model", toml_edit::value(model));
    if let Some(url) = base_url {
        profile.insert("base_url", toml_edit::value(url));
    }
    // An unset knob is left out rather than written at its default, so the profile records only
    // what the user actually chose and a later change to a default reaches existing profiles.
    if let Some(mode) = tuning.thinking {
        profile.insert("thinking", toml_edit::value(mode.as_str()));
    }
    if let Some(window) = tuning.context_window {
        profile.insert(
            "context_window",
            toml_edit::value(toml_integer("context_window", window)?),
        );
    }
    if let Some(effort) = tuning.effort.as_deref() {
        profile.insert("effort", toml_edit::value(effort));
    }
    if let Some(budget) = tuning.thinking_budget {
        profile.insert(
            "thinking_budget",
            toml_edit::value(toml_integer("thinking_budget", budget)?),
        );
    }
    if let Some(vision) = tuning.vision {
        profile.insert("vision", toml_edit::value(vision));
    }
    if let Some(cap) = tuning.max_output_tokens {
        profile.insert(
            "max_output_tokens",
            toml_edit::value(toml_integer("max_output_tokens", cap)?),
        );
    }
    if let Some(redact) = tuning.redact_thinking {
        profile.insert("redact_thinking", toml_edit::value(redact));
    }
    if let Some(url) = tuning.oauth_token_url.as_deref() {
        profile.insert("oauth_token_url", toml_edit::value(url));
    }
    if let Some(client_id) = tuning.client_id.as_deref() {
        profile.insert("client_id", toml_edit::value(client_id));
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
///
/// Answers whether a profile was actually removed, because the caller cannot tell from anywhere
/// else. Inferring it from a separate `as_table` probe made `remove` report the opposite of what it
/// did on an inline `providers`: the profile probed as absent, so the command deleted the
/// credential, dropped `default_provider`, left `[providers.<name>]` untouched in the file, and
/// said "no profile was configured" about one `meka provider list` still shows.
fn remove_profile_document(document: &mut toml_edit::DocumentMut, name: &str) -> bool {
    let removed = document
        .get_mut("providers")
        .and_then(|item| item.as_table_like_mut())
        .and_then(|providers| providers.remove(name))
        .is_some();
    // If this profile was the default, drop the dangling pointer.
    if document
        .get("default_provider")
        .and_then(|item| item.as_str())
        == Some(name)
    {
        document.as_table_mut().remove("default_provider");
    }
    removed
}

fn write_profile(
    name: &str,
    backend: &str,
    model: &str,
    base_url: Option<&str>,
    tuning: &ProfileTuning,
) -> anyhow::Result<()> {
    // `_lock` is held to the end of the function, so the read above and the write below are
    // one critical section.
    let (_lock, path, mut document) = open_document()?;
    let before = document.to_string();
    upsert_profile_document(&mut document, name, backend, model, base_url, tuning)?;
    let after = document.to_string();
    refuse_a_profile_that_cannot_run(&before, &after, name)?;
    config::write_file_atomic(&path, &after)?;
    Ok(())
}

/// Refuse a write that would leave `name` unusable, before the file is touched.
///
/// Both write doors ask this, and that is the point of it being one function. `set` asked it and
/// `add` did not, so `provider add work --thinking budgeted --thinking-budget 32000
/// --max-output-tokens 8000` exited 0 and wrote a profile that failed at *startup* on every later
/// run -- and was then a trap, because the `provider set` sent to repair it re-derived the same
/// refusal from the two keys already on disk and declined an unrelated edit.
///
/// `before` is what the file said when the lock was taken. The reparse below covers the whole
/// document, so an unrelated defect anywhere in it would otherwise be reported as something this
/// command did; asking whether it already failed is the difference between naming the line the user
/// must fix and blaming an edit that had nothing to do with it. It stays a refusal either way,
/// because writing on top of a file meka cannot read would strand the rest of its contents.
fn refuse_a_profile_that_cannot_run(before: &str, after: &str, name: &str) -> anyhow::Result<()> {
    let candidate: config::ConfigFile = toml::from_str(after).map_err(|error| {
        if toml::from_str::<config::ConfigFile>(before).is_err() {
            anyhow::anyhow!(
                "config.toml could not be read before this change either, so this is not what \
                 broke it. Fix the file first: {}",
                error
            )
        } else {
            anyhow::anyhow!("that change makes config.toml unreadable: {}", error)
        }
    })?;
    if let Some(profile) = candidate.providers.get(name) {
        crate::config::validate_max_output_tokens(
            name,
            Some(&profile.backend),
            profile.max_output_tokens,
            profile.thinking.unwrap_or_default(),
            profile
                .thinking_budget
                .or(candidate.thinking.as_ref().and_then(|it| it.budget_tokens))
                .unwrap_or(crate::config::DEFAULT_THINKING_BUDGET_TOKENS),
        )?;
    }
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

/// What declining the advanced step leaves in force, or `None` when the flags already pinned every
/// setting so there is no default left to report.
///
/// Only the still-unset ones are named: stating a default for something the flags set would
/// contradict the file this same command is about to write. Split out from the prompt so the
/// composition is testable without stdin - the prompt is the one part of `resolve_tuning` a test
/// cannot drive.
fn unset_defaults_summary(
    tuning: &ProfileTuning,
    takes_thinking: bool,
    effective_window: u64,
) -> Option<String> {
    let mut defaults: Vec<String> = Vec::new();
    if takes_thinking && tuning.thinking.is_none() {
        defaults.push(format!(
            "thinking {}",
            crate::provider::ThinkingMode::default().as_str()
        ));
    }
    if tuning.context_window.is_none() {
        defaults.push(format!("context window {}", effective_window));
    }
    if tuning.effort.is_none() {
        defaults.push("the provider's own reasoning effort".to_string());
    }
    (!defaults.is_empty()).then(|| defaults.join(", "))
}

/// Whether the advanced step should ask for a thinking budget.
///
/// The one flag-only setting that is also prompted for, and only in the case that creates it. A
/// budget means nothing under `adaptive` (the default) or `off`, which send no `budget_tokens` at
/// all, so asking unconditionally would put a fourth question in front of every user to serve the
/// one who just answered "budgeted" -- and that user needs it now, because this is where the
/// `max_output_tokens` pairing starts to matter.
///
/// Split out from the prompt for the reason [`unset_defaults_summary`] is: the prompt itself is the
/// one part of [`resolve_tuning`] a test cannot drive, so the condition deciding whether it fires
/// has to be reachable on its own or it is guarded by nothing.
fn budget_is_worth_asking_about(thinking: Option<crate::provider::ThinkingMode>) -> bool {
    thinking == Some(crate::provider::ThinkingMode::Budgeted)
}

/// Fill in whatever the flags didn't set, offering one opt-in prompt rather than three
/// unconditional ones. Declining leaves every unset knob out of the profile, and says which
/// defaults that implies, because these are the settings meka stopped inferring: nothing else will
/// tell the user they exist.
///
/// `stdin_is_the_key` suppresses the prompt entirely. Under `--api-key-stdin` the piped line is the
/// credential, and a prompt here would read it as the answer and then leave the key read at end of
/// input. The flags stay available for setting these non-interactively.
fn resolve_tuning(
    flags: ProfileTuning,
    backend: &str,
    profile_name: &str,
    session_window: Option<u64>,
    global_budget: Option<u64>,
    stdin_is_the_key: bool,
) -> anyhow::Result<ProfileTuning> {
    // What an unset profile would actually budget against: `[session].context_window` if the user
    // already set one, else the built-in default. Showing the constant unconditionally would state
    // a window the run is not going to use.
    let effective_window = session_window.unwrap_or(crate::provider::DEFAULT_CONTEXT_WINDOW);
    // The same question one field over, and it was got wrong once: the budget prompt showed the
    // built-in constant while the profile would actually fall back to `[thinking].budget_tokens`,
    // so pressing Enter to "accept the default" produced a different number from typing the one on
    // screen.
    let effective_budget = global_budget.unwrap_or(crate::config::DEFAULT_THINKING_BUDGET_TOKENS);
    // `thinking` is an Anthropic Messages request field, so an OpenAI profile is neither asked
    // about it nor told what it defaults to: writing the key there would produce a setting that
    // reads plausibly and does nothing.
    let takes_thinking = crate::provider::backend_takes_thinking(backend);
    let mut flags = flags;
    // The flags are dropped, not just left unprompted. Guarding only the prompt produced a run that
    // printed "using defaults:" without thinking *and* wrote `thinking` into the profile. All three
    // of them, because they are one request field between them: guarding `thinking` alone let
    // `--thinking-budget 2048` write `thinking_budget` into an OpenAI profile, which is the same
    // inert key one field over.
    if !takes_thinking {
        let dropped: Vec<&str> = [
            flags.thinking.take().map(|_| "--thinking"),
            flags.thinking_budget.take().map(|_| "--thinking-budget"),
            flags.redact_thinking.take().map(|_| "--redact-thinking"),
        ]
        .into_iter()
        .flatten()
        .collect();
        if !dropped.is_empty() {
            tracing::warn!(
                "ignoring {} for a '{}' profile: thinking is an Anthropic Messages request field",
                dropped.join(", "),
                backend
            );
        }
    }
    let complete = flags.context_window.is_some()
        && flags.effort.is_some()
        && (!takes_thinking || flags.thinking.is_some());
    if complete || stdin_is_the_key {
        return Ok(flags);
    }
    if !prompt_yes_no("Configure advanced settings? [y/N]: ")? {
        if let Some(defaults) = unset_defaults_summary(&flags, takes_thinking, effective_window) {
            eprintln!(
                "using defaults: {}. Change these under [providers.{}] in config.toml.",
                defaults, profile_name,
            );
        }
        return Ok(flags);
    }

    let thinking = match flags.thinking {
        Some(mode) => Some(mode),
        None if !takes_thinking => None,
        None => {
            let input = prompt_line("  Thinking mode (adaptive, budgeted, off) [adaptive]: ")?;
            match input.as_str() {
                "" => None,
                // Parsed through clap's `ValueEnum` rather than a fourth hand-written match, so
                // this prompt accepts exactly what `--thinking` accepts.
                other => Some(
                    <crate::provider::ThinkingMode as clap::ValueEnum>::from_str(other, true)
                        .map_err(|_| {
                            anyhow::anyhow!(
                                "'{}' is not a thinking mode. Expected adaptive, budgeted, or off.",
                                other
                            )
                        })?,
                ),
            }
        }
    };

    let context_window = match flags.context_window {
        Some(window) => Some(window),
        None => {
            let input = prompt_line(&format!(
                "  Context window in tokens [{}]: ",
                effective_window
            ))?;
            match input.as_str() {
                "" => None,
                other => Some(other.parse::<u64>().map_err(|_| {
                    anyhow::anyhow!("'{}' is not a whole number of tokens.", other)
                })?),
            }
        }
    };

    let effort = match flags.effort {
        Some(effort) => Some(effort),
        None => {
            let input = prompt_line("  Reasoning effort (empty for the provider's default): ")?;
            (!input.is_empty()).then_some(input)
        }
    };

    let thinking_budget = match flags.thinking_budget {
        Some(budget) => Some(budget),
        None if budget_is_worth_asking_about(thinking) => {
            let input = prompt_line(&format!(
                "  Thinking budget in tokens [{}]: ",
                effective_budget
            ))?;
            match input.as_str() {
                "" => None,
                other => Some(other.parse::<u64>().map_err(|_| {
                    anyhow::anyhow!("'{}' is not a whole number of tokens.", other)
                })?),
            }
        }
        None => None,
    };

    Ok(ProfileTuning {
        thinking,
        context_window,
        effort,
        thinking_budget,
        // Flag-only, so they pass through whatever the caller set. Prompting for them would make
        // the advanced step twice as long for settings that are stated far less often.
        vision: flags.vision,
        max_output_tokens: flags.max_output_tokens,
        redact_thinking: flags.redact_thinking,
        oauth_token_url: flags.oauth_token_url,
        client_id: flags.client_id,
    })
}

/// A `[y/N]` prompt: anything but `y` / `yes` (case-insensitively) is a no, **including end of
/// input**. An optional prompt must not be able to fail a run that would otherwise succeed: with
/// stdin closed or redirected, `provider add` takes the defaults and carries on to the credential
/// step, which is what it did before this prompt existed.
fn prompt_yes_no(prompt: &str) -> io::Result<bool> {
    match prompt_line(prompt) {
        Ok(answer) => Ok(matches!(answer.to_ascii_lowercase().as_str(), "y" | "yes")),
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => Ok(false),
        Err(error) => Err(error),
    }
}

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

/// The `provider add` menu.
///
/// Ordered subscription-then-key within each vendor, and `openai-responses` directly under
/// `openai-chat-completions` because the two are a protocol choice against the same key: Responses
/// is what new work should use, Chat Completions is what a server that doesn't serve Responses
/// still speaks.
/// The menu entries, in the order they are offered. Separate from [`prompt_backend`] so a test can
/// check it against [`crate::provider::SUPPORTED_PROVIDERS`]: this is the last hand-written backend
/// list, and a backend missing from it is simply never offered interactively, with nothing failing.
fn backend_menu() -> [(&'static str, &'static str); 5] {
    [
        ("claude-subscription", "Claude subscription login"),
        ("anthropic-messages", "Anthropic Messages API key"),
        ("chatgpt-subscription", "ChatGPT subscription login"),
        (
            "openai-chat-completions",
            "OpenAI-compatible Chat Completions API key",
        ),
        ("openai-responses", "OpenAI-compatible Responses API key"),
    ]
}

fn prompt_backend() -> anyhow::Result<String> {
    let options = backend_menu();
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
    let client_id = client_id.unwrap_or(DEFAULT_CLAUDE_SUBSCRIPTION_CLIENT_ID);
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
        anyhow::bail!(
            "token exchange failed ({}): {}",
            status,
            crate::error::render_error_body(&body)
        );
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
    let client_id = client_id.unwrap_or(DEFAULT_CHATGPT_SUBSCRIPTION_CLIENT_ID);
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
    // `form_urlencoded` rather than a hand-rolled split, for the reason given at the matching site
    // in `mcp::auth`: a redirect query is form-encoded, so `+` is a space and decoding it as a
    // literal `+` corrupts any value that contains one.
    for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
        let decoded = value.into_owned();
        match key.as_ref() {
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
        anyhow::bail!(
            "Codex token exchange failed ({}): {}",
            status,
            crate::error::render_error_body(&body)
        );
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
    let (source, requested) = match profile_arg {
        Some(name) => (config::ProfileRequest::Flag, Some(name)),
        None => (
            config::ProfileRequest::DefaultProvider,
            config_file.default_provider.clone(),
        ),
    };
    let (name, error) = config::select_active_profile(requested, source, &config_file.providers);
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
        crate::session::SessionManager::open(
            Some(std::path::Path::new(":memory:")),
            &Default::default(),
        )
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
            "[providers.work]\ntype = \"anthropic-messages\"\nmodel = \"claude-opus-5\"\n\
             [providers.personal]\ntype = \"openai-chat-completions\"\nmodel = \"gpt-5.6-sol\"\n",
        )
        .expect("parse config");

        let orphans = orphaned_profiles(&store, &config_file)
            .await
            .expect("diff credentials against profiles");
        assert_eq!(orphans, vec!["archive".to_string()]);
    }

    /// The sibling of `login_refuses_a_piped_key_for_a_browser_backend`, and asserted together
    /// because the two commands taking the same flag must answer it the same way. `add` knows the
    /// backend too: `--api-key-stdin` requires `--type`, so the refusal can come before the browser
    /// opens rather than after a script has hung on it.
    #[tokio::test]
    async fn add_refuses_a_piped_key_for_a_browser_backend() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.toml"), "").expect("write config");
        let store = memory_token_store().await;

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = run_add(
            "sub",
            Some("claude-subscription"),
            Some("claude-opus-5".to_string()),
            None,
            ProfileTuning::default(),
            true,
            &store,
        )
        .await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = match result {
            Ok(()) => panic!("a piped key for an OAuth backend must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("claude-subscription"), "{error}");
        assert!(error.contains("browser"), "{error}");
        // Nothing was written: the refusal lands before the config document is even opened.
        let contents = std::fs::read_to_string(dir.path().join("config.toml")).expect("read back");
        assert!(contents.is_empty(), "{contents}");
    }

    /// `acquire_credential` ignores `--api-key-stdin` for a backend that logs in through the
    /// browser, so without this a script piping a key to a subscription profile would sit on an
    /// OAuth flow it cannot see while its key went unread. `login` knows the backend from the
    /// profile, so it can say so before anything opens.
    #[tokio::test]
    async fn login_refuses_a_piped_key_for_a_browser_backend() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            "[providers.sub]\ntype = \"claude-subscription\"\nmodel = \"claude-opus-5\"\n",
        )
        .expect("write config");
        let store = memory_token_store().await;

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = run_login("sub", true, &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = match result {
            Ok(()) => panic!("a piped key for an OAuth backend must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("claude-subscription"), "{error}");
        assert!(error.contains("browser"), "{error}");
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
        let manager = crate::session::SessionManager::open(
            Some(std::path::Path::new(":memory:")),
            &Default::default(),
        )
        .await
        .expect("memory store");
        let store = manager.token_store();
        store
            .save_provider_credential("work", &AuthCredential::ApiKey("key".to_string()))
            .await
            .expect("save");

        // SAFETY: `MEKA_CONFIG_DIR` is process-global; `CONFIG_DIR_ENV_LOCK` serialises every test
        // that touches it, and the guard is held across the whole set → run → clear cycle.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = run_remove("work", &store, &manager).await;
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
            "[providers.work]\ntype = \"anthropic-messages\"\nmodel = \"claude-opus-5\"\n",
        )
        .expect("write config");
        let manager = crate::session::SessionManager::open(
            Some(std::path::Path::new(":memory:")),
            &Default::default(),
        )
        .await
        .expect("memory store");
        let store = manager.token_store();

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = run_remove("typo", &store, &manager).await;
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
        assert!(validate_backend("claude-subscription").is_ok());
        assert!(validate_backend("bogus").is_err());
    }

    #[test]
    fn test_default_model_for_known_backends() {
        assert_eq!(
            default_model_for("anthropic-messages"),
            Some("claude-opus-5")
        );
        assert_eq!(
            default_model_for("claude-subscription"),
            Some("claude-opus-5")
        );
        assert_eq!(
            default_model_for("openai-chat-completions"),
            Some("gpt-5.6-sol")
        );
        assert_eq!(
            default_model_for("chatgpt-subscription"),
            Some("gpt-5.6-sol")
        );
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

    /// Every supported backend must be offered by the interactive menu.
    ///
    /// `meka provider add` with no `--type` is how most people meet the backend list, and a name
    /// missing from the menu is unreachable that way while failing nothing: no error, no warning,
    /// just a backend the wizard never mentions.
    #[test]
    fn every_supported_backend_is_offered_by_the_menu() {
        let offered: Vec<&str> = backend_menu().iter().map(|(id, _)| *id).collect();
        for backend in crate::provider::SUPPORTED_PROVIDERS {
            assert!(
                offered.contains(backend),
                "{backend} is supported but is not in the `provider add` menu"
            );
        }
        assert_eq!(
            offered.len(),
            crate::provider::SUPPORTED_PROVIDERS.len(),
            "the menu offers something that is not a supported backend: {offered:?}"
        );
    }

    /// Every supported backend must have a credential flow.
    ///
    /// This is the invariant that broke when `openai-responses` was added: the match ended in
    /// `unreachable!()`, `validate_backend` waved the new name through as supported, and
    /// `meka provider add --type openai-responses` panicked at the credential step. Nothing caught
    /// it -- the backend built, resolved, and had a default model and endpoint. Asserting over
    /// `SUPPORTED_PROVIDERS` rather than a hand-written list is the point: a backend added there
    /// and forgotten here fails this test instead of a user's first `provider add`.
    #[test]
    fn every_supported_backend_has_a_credential_flow() {
        for backend in crate::provider::SUPPORTED_PROVIDERS {
            assert!(
                credential_kind(backend).is_some(),
                "{backend} is supported but has no credential flow"
            );
        }
        // Each subscription backend gets its *own* vendor's login, not merely "an OAuth flow":
        // the client IDs, scopes and callbacks differ and are not interchangeable.
        assert_eq!(
            credential_kind("claude-subscription"),
            Some(CredentialKind::ClaudeLogin)
        );
        assert_eq!(
            credential_kind("chatgpt-subscription"),
            Some(CredentialKind::ChatGptLogin)
        );
        assert_eq!(
            credential_kind("openai-responses"),
            Some(CredentialKind::ApiKey)
        );
        assert_eq!(credential_kind("not-a-backend"), None);
    }

    /// Every thinking flag must be *dropped* on a backend whose requests have no thinking field,
    /// not merely left unprompted.
    ///
    /// Guarding only the prompt is the bug this closes: the run then printed a "using defaults:"
    /// line with no thinking in it *and* wrote `thinking` into the profile anyway, so the file
    /// disagreed with what the user had just been told, and the key sat there reading plausibly
    /// while doing nothing. Both halves are asserted, because either one alone passes the wrong
    /// implementation.
    ///
    /// All three flags, because the fix was originally applied to `thinking` alone and
    /// `--thinking-budget` then walked straight through the hole it left: the three are one request
    /// field between them, so a guard that names one of them is a guard for none of them.
    #[test]
    fn a_thinking_flag_aimed_at_a_backend_without_thinking_is_dropped() {
        let flags = || ProfileTuning {
            thinking: Some(crate::provider::ThinkingMode::Budgeted),
            thinking_budget: Some(2_048),
            redact_thinking: Some(true),
            context_window: Some(1_024),
            effort: Some("low".to_string()),
            ..Default::default()
        };
        // Every setting is pinned, so this returns before any prompt: no stdin involved.
        let openai = resolve_tuning(flags(), "openai-chat-completions", "oai", None, None, false)
            .expect("resolve");
        assert_eq!(openai.thinking, None, "written into an OpenAI profile");
        assert_eq!(
            openai.thinking_budget, None,
            "the budget is the same request field, one key over"
        );
        assert_eq!(
            openai.redact_thinking, None,
            "and so is the redaction of what it produces"
        );
        assert_eq!(openai.context_window, Some(1_024));
        assert_eq!(openai.effort.as_deref(), Some("low"));

        let claude = resolve_tuning(flags(), "anthropic-messages", "work", None, None, false)
            .expect("resolve");
        assert_eq!(
            claude.thinking,
            Some(crate::provider::ThinkingMode::Budgeted),
            "the same flag must still reach a Claude profile"
        );
        assert_eq!(
            claude.thinking_budget,
            Some(2_048),
            "and so must the budget"
        );
        assert_eq!(claude.redact_thinking, Some(true), "and the redaction flag");
    }

    /// The defaults line names what is still unset, and nothing else.
    ///
    /// It is the only thing that tells a user these settings exist - meka stopped inferring them,
    /// so nothing else in the run mentions them. Naming one the flags already pinned would state a
    /// default that contradicts the file being written in the same breath.
    #[test]
    fn the_defaults_line_reports_only_the_settings_still_unset() {
        let bare = ProfileTuning::default();
        let claude = unset_defaults_summary(&bare, true, 1_000_000).expect("something is unset");
        assert!(claude.contains("thinking adaptive"), "{claude}");
        assert!(claude.contains("context window 1000000"), "{claude}");
        assert!(claude.contains("reasoning effort"), "{claude}");

        // No thinking on a backend that has no such field, so it is not reported either.
        let openai = unset_defaults_summary(&bare, false, 1_000_000).expect("something is unset");
        assert!(!openai.contains("thinking"), "{openai}");

        // The window reported is the one an unset profile would actually budget against, which is
        // `[session].context_window` when the user already set one.
        let session_window = unset_defaults_summary(&bare, false, 262_144).expect("unset");
        assert!(
            session_window.contains("context window 262144"),
            "{session_window}"
        );

        // A pinned setting is not reported as a default.
        let pinned = ProfileTuning {
            thinking: Some(crate::provider::ThinkingMode::Off),
            context_window: Some(8_192),
            effort: None,
            ..Default::default()
        };
        let partial = unset_defaults_summary(&pinned, true, 1_000_000).expect("effort is unset");
        assert!(!partial.contains("thinking"), "{partial}");
        assert!(!partial.contains("context window"), "{partial}");
        assert!(partial.contains("reasoning effort"), "{partial}");

        // Nothing unset, nothing to say.
        assert_eq!(
            unset_defaults_summary(
                &ProfileTuning {
                    thinking: Some(crate::provider::ThinkingMode::Off),
                    context_window: Some(8_192),
                    effort: Some("low".to_string()),
                    ..Default::default()
                },
                true,
                1_000_000,
            ),
            None
        );
    }

    /// `set` edits one key and leaves the rest of the file exactly as the user wrote it.
    ///
    /// The whole reason this is not [`upsert_profile_document`]: replacing the table would rewrite
    /// every key in meka's order and drop the comments beside them, so a user who ran `set` once
    /// would find their annotated config quietly reformatted and their notes gone. Nothing else
    /// would fail, which is why the guard is here.
    ///
    /// A key's surroundings live in two decors and `insert` clears both, so this asserts on both.
    /// The first version of the fix carried the *value*'s decor only, which saved the trailing
    /// `# note` and still deleted the whole-line comment and blank line above the key, since those
    /// belong to the key's leaf decor. Half a fix passes an assertion on either half alone.
    #[test]
    fn set_edits_one_key_and_preserves_everything_around_it() {
        let mut document = r#"default_provider = "work"

# Why this profile exists.
[providers.work]
type = "anthropic-messages"

# The 1M-window model; keep context_window in step with it.
model = "old-model"       # the model, annotated
context_window = 200000

[providers.other]
type = "openai-responses"
model = "untouched"
"#
        .parse::<toml_edit::DocumentMut>()
        .expect("parse");

        assert!(set_profile_field(
            &mut document,
            "work",
            "model",
            Some(toml_edit::Value::from("new-model"))
        ));

        let rendered = document.to_string();
        assert!(rendered.contains("model = \"new-model\""), "{rendered}");
        assert!(
            rendered.contains("# Why this profile exists."),
            "the comment above the table survives: {rendered}"
        );
        assert!(
            rendered.contains("# the model, annotated"),
            "the comment beside the edited key survives: {rendered}"
        );
        assert!(
            rendered.contains("# The 1M-window model; keep context_window in step with it."),
            "and so does the one above it, which lives in the key's decor rather than the \
             value's: {rendered}"
        );
        assert!(
            rendered.contains("\n\n# The 1M-window model"),
            "including the blank line that separated it from the key before: {rendered}"
        );
        assert!(
            rendered.contains("context_window = 200000"),
            "the profile's other keys survive: {rendered}"
        );
        assert!(
            rendered.contains("model = \"untouched\""),
            "another profile is not touched: {rendered}"
        );
        assert!(
            rendered.find("type").unwrap() < rendered.find("context_window").unwrap(),
            "key order is the user's, not meka's: {rendered}"
        );
    }

    /// A profile setting past `i64::MAX` is refused rather than wrapped into the file.
    ///
    /// TOML has one integer type and it is signed, so `as i64` turned `u64::MAX` into `-1` and
    /// wrote it. `provider add brick --context-window 18446744073709551615` then exited 0 having
    /// bricked the config: every later command, including the `provider set` that would have
    /// repaired it, refused the file with `invalid value: integer -1, expected u64`.
    #[test]
    fn an_integer_too_large_for_toml_is_refused_rather_than_wrapped() {
        for field in ["context_window", "thinking_budget", "max_output_tokens"] {
            let error = toml_integer(field, u64::MAX).expect_err("u64::MAX has no TOML form");
            let message = error.to_string();
            assert!(
                message.contains(field) && message.contains(&i64::MAX.to_string()),
                "the refusal names the setting and the ceiling: {message}"
            );
        }
        assert_eq!(
            toml_integer("context_window", 1_000_000).expect("an ordinary window fits"),
            1_000_000
        );
        // The boundary itself, so the refusal is `>` and not `>=`.
        assert_eq!(
            toml_integer("context_window", i64::MAX as u64).expect("the largest legal value fits"),
            i64::MAX
        );

        // And through the door `set` actually uses, since that is where a user meets it.
        let error = parse_profile_value("context_window", &u64::MAX.to_string())
            .expect_err("`set` refuses it too");
        assert!(
            error.to_string().contains("context_window"),
            "{}",
            error.to_string()
        );
    }

    /// The shared refusal both write doors ask, checked directly.
    ///
    /// Reached only through `write_profile` and `run_set`, which both touch the real config path,
    /// so nothing exercised it: `cargo mutants` replaced the whole function with `Ok(())` and the
    /// suite stayed green. It is a pure function of the two documents and a name, so it does not
    /// need the filesystem to be tested -- only to be called.
    ///
    /// The pairing is what `provider add` used to let through. `run_set`'s comment claimed parity
    /// with `add` while `add` refused nothing, so the flags below wrote a profile that exited 0 and
    /// then failed at startup on every later run.
    #[test]
    fn a_profile_that_could_not_start_is_refused_by_both_write_doors() {
        let good = r#"[providers.work]
type = "anthropic-messages"
thinking = "budgeted"
thinking_budget = 8000
max_output_tokens = 32000
"#;
        let cap_below_budget = r#"[providers.work]
type = "anthropic-messages"
thinking = "budgeted"
thinking_budget = 32000
max_output_tokens = 8000
"#;
        refuse_a_profile_that_cannot_run(good, good, "work").expect("a workable pairing passes");
        let error = refuse_a_profile_that_cannot_run(good, cap_below_budget, "work")
            .expect_err("a cap at or below the budget cannot produce a valid request");
        let message = error.to_string();
        assert!(
            message.contains("work") && message.contains("32000"),
            "the refusal names the profile and the budget it must exceed: {message}"
        );

        // The budget falls back to the global when the profile states none, so the same pairing is
        // refused even though neither number is written beside the other.
        let global_budget = r#"[thinking]
budget_tokens = 64000

[providers.work]
type = "anthropic-messages"
thinking = "budgeted"
max_output_tokens = 16000
"#;
        refuse_a_profile_that_cannot_run(good, global_budget, "work")
            .expect_err("the fallback budget is checked too, not skipped for being absent");

        // A profile the edit did not touch is not this call's business.
        refuse_a_profile_that_cannot_run(good, cap_below_budget, "other")
            .expect("only the named profile is judged");
    }

    /// A file that was already unreadable is not blamed on the edit that found it.
    ///
    /// The reparse covers the whole document, so any unrelated defect anywhere in it blocks the
    /// write. Reporting that as "that change makes config.toml unreadable" sent the user looking at
    /// the key they had just set, which was not the problem. Still a refusal either way; only the
    /// sentence changes, and the sentence is the whole value.
    #[test]
    fn a_pre_existing_config_error_is_not_blamed_on_this_edit() {
        let already_broken = "[typo_section]\nfoo = 1\n";
        let error = refuse_a_profile_that_cannot_run(already_broken, already_broken, "work")
            .expect_err("an unreadable file is refused");
        assert!(
            error.to_string().contains("before this change either"),
            "the message must say the edit is not what broke it: {error}"
        );

        let fine = "[providers.work]\ntype = \"anthropic-messages\"\n";
        let error = refuse_a_profile_that_cannot_run(fine, already_broken, "work")
            .expect_err("an edit that breaks parsing is refused");
        assert!(
            error.to_string().contains("makes config.toml unreadable"),
            "and when the edit *is* what broke it, it says so: {error}"
        );
    }

    /// A thinking key is refused on a profile whose backend never sends one.
    ///
    /// `provider add` drops such a flag with a warning; `set` refuses, because a command whose
    /// entire job is one key cannot report success having written nothing. Both doors reach the
    /// same place: the key does not end up in a profile that ignores it.
    #[test]
    fn setting_a_thinking_key_on_a_backend_without_thinking_is_refused() {
        let document = r#"[providers.oai]
type = "openai-responses"
model = "gpt-5.6"

[providers.work]
type = "anthropic-messages"
model = "claude-opus-5"

[providers.typo]
type = "not-a-backend"
model = "m"
"#
        .parse::<toml_edit::DocumentMut>()
        .expect("parse");

        for key in THINKING_ONLY_PROFILE_KEYS {
            let error = refuse_an_inert_thinking_key(&document, "oai", key)
                .expect_err("inert on a Responses profile");
            let message = error.to_string();
            assert!(
                message.contains(key) && message.contains("openai-responses"),
                "the refusal names the key and the backend: {message}"
            );
            refuse_an_inert_thinking_key(&document, "work", key)
                .expect("the same key is live on a Messages profile");
            // An unrecognised `type` is `validate_backend`'s to report. Refusing here would
            // answer a typo in one key with a complaint about a different one.
            refuse_an_inert_thinking_key(&document, "typo", key)
                .expect("an unknown backend is not this function's to judge");
        }
        refuse_an_inert_thinking_key(&document, "oai", "model")
            .expect("a key that is not thinking-only passes on any backend");
    }

    /// `--unset` removes the key rather than writing an empty value.
    ///
    /// The two are not the same: an absent key follows whatever meka's default becomes, which is
    /// the documented meaning of an unstated setting, while `model = ""` is a model named the empty
    /// string and would be sent as one.
    #[test]
    fn unsetting_removes_the_key_rather_than_emptying_it() {
        let mut document = r#"[providers.work]
type = "anthropic-messages"
effort = "high"
"#
        .parse::<toml_edit::DocumentMut>()
        .expect("parse");

        assert!(set_profile_field(&mut document, "work", "effort", None));
        let rendered = document.to_string();
        assert!(!rendered.contains("effort"), "the key is gone: {rendered}");
        assert!(rendered.contains("type ="), "the rest stays: {rendered}");
    }

    /// A profile that is not there is reported, never silently created.
    ///
    /// Answering `true` here would have `set` write a `[providers.<typo>]` table with one key in
    /// it, report success, and leave the user's real profile unchanged with no indication why.
    #[test]
    fn setting_a_key_on_an_absent_profile_says_so() {
        let mut document = r#"[providers.work]
type = "anthropic-messages"
"#
        .parse::<toml_edit::DocumentMut>()
        .expect("parse");

        assert!(!set_profile_field(
            &mut document,
            "ghost",
            "model",
            Some(toml_edit::Value::from("x"))
        ));
        assert!(
            !document.to_string().contains("ghost"),
            "a refused set writes nothing at all"
        );
    }

    /// Every settable key parses its own value type, and refuses what it cannot mean.
    #[test]
    fn each_profile_key_parses_the_way_its_add_flag_does() {
        assert!(parse_profile_value("model", "claude-opus-5").is_ok());
        assert!(parse_profile_value("context_window", "200000").is_ok());
        assert!(parse_profile_value("context_window", "lots").is_err());
        assert!(parse_profile_value("vision", "false").is_ok());
        assert!(parse_profile_value("vision", "no").is_err());
        assert!(parse_profile_value("thinking", "budgeted").is_ok());
        assert!(parse_profile_value("thinking", "sideways").is_err());
        assert!(parse_profile_value("thinking_budget", "2048").is_ok());

        // Every key in the advertised list has an arm. A key listed but unhandled would fall to the
        // catch-all and be refused as unknown, which reads as meka having forgotten its own field.
        for key in SETTABLE_PROFILE_KEYS {
            let sample = match *key {
                "context_window" | "max_output_tokens" | "thinking_budget" => "1000",
                "vision" | "redact_thinking" => "true",
                "thinking" => "adaptive",
                _ => "value",
            };
            assert!(
                parse_profile_value(key, sample).is_ok(),
                "'{key}' is advertised as settable but does not parse"
            );
        }
    }

    /// An unknown key is refused whichever way it arrived, including behind `--unset`.
    ///
    /// The refusal used to live inside the value branch, so `--unset modle` skipped it entirely:
    /// the command exited 0, removed nothing, and told the user nothing. Nothing else would notice,
    /// because the write it did not do is indistinguishable from a key that was already absent.
    #[test]
    fn an_unknown_key_is_refused_whether_or_not_a_value_came_with_it() {
        let error = ensure_settable_key("modle").expect_err("a typo is not a setting");
        assert!(
            error.to_string().contains("is not a profile setting"),
            "{}",
            error
        );
        for key in SETTABLE_PROFILE_KEYS {
            assert!(
                ensure_settable_key(key).is_ok(),
                "'{key}' is advertised as settable but the door refuses it"
            );
        }
        assert!(
            ensure_settable_key("type").is_err() && ensure_settable_key("device_id").is_err(),
            "the two deliberate exclusions are refused by the same door"
        );
    }

    /// The advanced step asks for a budget exactly when one will be sent, and never otherwise.
    ///
    /// `adaptive` and `off` send no `budget_tokens` at all, so a budget collected under them is a
    /// question asked for nothing and a key written into the profile that does nothing. Inverting
    /// this, or pinning it to `true`, is invisible to every other test: the prompt is the one part
    /// of `resolve_tuning` a test cannot drive.
    #[test]
    fn the_budget_is_asked_for_under_budgeted_and_nothing_else() {
        use crate::provider::ThinkingMode;
        assert!(budget_is_worth_asking_about(Some(ThinkingMode::Budgeted)));
        assert!(!budget_is_worth_asking_about(Some(ThinkingMode::Adaptive)));
        assert!(!budget_is_worth_asking_about(Some(ThinkingMode::Off)));
        assert!(
            !budget_is_worth_asking_about(None),
            "an unstated mode takes the default, which is adaptive and sends no budget"
        );
    }

    /// `type` is refused with the reason, not merely omitted from the list.
    ///
    /// A user who typed it did not typo, so "settable: model, base_url, …" on its own would read as
    /// an oversight rather than a decision, and the obvious next move would be to hand-edit the key
    /// that meka is declining to change for them.
    #[test]
    fn changing_a_profiles_backend_is_refused_with_its_reason() {
        let error = parse_profile_value("type", "openai-responses")
            .expect_err("the backend is not settable in place");
        let message = error.to_string();
        assert!(
            message.contains("credential") && message.contains("meka provider add"),
            "the refusal must say why and where to go: {message}"
        );

        let device = parse_profile_value("device_id", "abc")
            .expect_err("meka's own bookkeeping is not settable");
        assert!(
            device.to_string().contains("meka's own"),
            "{}",
            device.to_string()
        );
    }

    /// Every flag-only setting reaches the profile, and none is written when its flag was absent.
    ///
    /// These six never pass through a prompt, so the writer is the only thing standing between the
    /// flag and the file: a missing `insert` would make `--vision false` exit 0 and change nothing,
    /// and the profile would keep advertising images the model cannot take. The absent half matters
    /// for the reason the sibling test gives -- an unstated key follows the documented default, and
    /// writing it eagerly would freeze today's default into every profile.
    #[test]
    fn the_flag_only_settings_reach_the_profile_and_only_when_given() {
        let mut bare = toml_edit::DocumentMut::new();
        upsert_profile_document(
            &mut bare,
            "local",
            "anthropic-messages",
            "some-local-model",
            None,
            &ProfileTuning::default(),
        )
        .expect("upsert");
        let rendered = bare.to_string();
        for key in [
            "thinking_budget",
            "vision",
            "max_output_tokens",
            "redact_thinking",
            "oauth_token_url",
            "client_id",
        ] {
            assert!(!rendered.contains(key), "{key} written unasked: {rendered}");
        }

        let mut tuned = toml_edit::DocumentMut::new();
        upsert_profile_document(
            &mut tuned,
            "local",
            "anthropic-messages",
            "some-local-model",
            None,
            &ProfileTuning {
                thinking_budget: Some(4_096),
                vision: Some(false),
                max_output_tokens: Some(32_000),
                redact_thinking: Some(false),
                oauth_token_url: Some("https://auth.invalid/token".to_string()),
                client_id: Some("my-client".to_string()),
                ..Default::default()
            },
        )
        .expect("upsert");

        // Read back through the real parser, so a key written under the wrong name or type is
        // caught here rather than by `deny_unknown_fields` on the user's next run.
        let parsed: config::ConfigFile = toml::from_str(&tuned.to_string()).expect("parses");
        let profile = parsed.providers.get("local").expect("the profile");
        assert_eq!(profile.thinking_budget, Some(4_096));
        assert_eq!(profile.vision, Some(false));
        assert_eq!(profile.max_output_tokens, Some(32_000));
        assert_eq!(profile.redact_thinking, Some(false));
        assert_eq!(
            profile.oauth_token_url.as_deref(),
            Some("https://auth.invalid/token")
        );
        assert_eq!(profile.client_id.as_deref(), Some("my-client"));
    }

    /// The advanced prompt covers exactly the three settings it has always covered.
    ///
    /// `complete` is what decides whether the prompt fires at all. Widening it to the flag-only
    /// settings would mean `--client-id x` silently skips the thinking / window / effort questions,
    /// so a user who set one advanced thing would never be asked about the three that matter most,
    /// and would get their defaults without being told. Nothing else would fail.
    #[test]
    fn a_flag_only_setting_does_not_suppress_the_advanced_prompt() {
        let all_three = ProfileTuning {
            thinking: Some(crate::provider::ThinkingMode::Off),
            context_window: Some(1_000),
            effort: Some("low".to_string()),
            ..Default::default()
        };
        assert!(
            unset_defaults_summary(&all_three, true, 1_000).is_none(),
            "with the prompted three set there is no default left to report"
        );

        let only_flag_only = ProfileTuning {
            client_id: Some("my-client".to_string()),
            vision: Some(false),
            ..Default::default()
        };
        let summary = unset_defaults_summary(&only_flag_only, true, 1_000)
            .expect("the prompted three are still unset, so their defaults are still worth naming");
        for expected in ["thinking", "context window", "reasoning effort"] {
            assert!(
                summary.contains(expected),
                "'{expected}' missing from: {summary}"
            );
        }
    }

    #[test]
    fn the_advanced_settings_are_written_only_when_chosen() {
        // Unset knobs stay out of the profile rather than being written at their defaults, so a
        // profile records the user's choices and a later change to a default still reaches it.
        let mut bare = toml_edit::DocumentMut::new();
        upsert_profile_document(
            &mut bare,
            "local",
            "anthropic-messages",
            "some-local-model",
            None,
            &ProfileTuning::default(),
        )
        .expect("upsert");
        let rendered = bare.to_string();
        for key in ["thinking", "context_window", "effort"] {
            assert!(!rendered.contains(key), "{key} in: {rendered}");
        }

        // Chosen ones round-trip through the runtime config, which is what actually reads them.
        let mut tuned = toml_edit::DocumentMut::new();
        upsert_profile_document(
            &mut tuned,
            "local",
            "anthropic-messages",
            "some-local-model",
            None,
            &ProfileTuning {
                thinking: Some(crate::provider::ThinkingMode::Budgeted),
                context_window: Some(262_144),
                effort: Some("medium".to_string()),
                ..Default::default()
            },
        )
        .expect("upsert");
        let config: config::ConfigFile =
            toml::from_str(&tuned.to_string()).expect("re-parse config");
        let profile = config.providers.get("local").expect("profile present");
        assert_eq!(
            profile.thinking,
            Some(crate::provider::ThinkingMode::Budgeted)
        );
        assert_eq!(profile.context_window, Some(262_144));
        assert_eq!(profile.effort.as_deref(), Some("medium"));
    }

    #[test]
    fn test_upsert_profile_document_first_profile_becomes_default() {
        let mut document = toml_edit::DocumentMut::new();
        upsert_profile_document(
            &mut document,
            "work",
            "openai-chat-completions",
            "gpt-4o",
            Some("http://localhost:1234/v1"),
            &ProfileTuning::default(),
        )
        .expect("upsert");
        // The rendered TOML must parse back into the runtime config with the profile and default.
        let config: config::ConfigFile =
            toml::from_str(&document.to_string()).expect("re-parse config");
        assert_eq!(config.default_provider.as_deref(), Some("work"));
        let profile = config.providers.get("work").expect("profile present");
        assert_eq!(profile.backend, "openai-chat-completions");
        assert_eq!(profile.model.as_deref(), Some("gpt-4o"));
        assert_eq!(
            profile.base_url.as_deref(),
            Some("http://localhost:1234/v1")
        );
    }

    #[test]
    fn test_upsert_profile_document_second_profile_keeps_existing_default() {
        let mut document = toml_edit::DocumentMut::new();
        upsert_profile_document(
            &mut document,
            "work",
            "claude-subscription",
            "claude-x",
            None,
            &ProfileTuning::default(),
        )
        .expect("upsert work");
        upsert_profile_document(
            &mut document,
            "personal",
            "openai-chat-completions",
            "gpt-4o",
            None,
            &ProfileTuning::default(),
        )
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

    /// `providers = { work = { … } }` is valid TOML that serde reads, so `provider list` and the
    /// duplicate guard both see `work`. Treating "not a header table" as "absent" and overwriting
    /// destroyed it: one `provider add home` left a config naming only `home`, `work`'s credential
    /// orphaned in the database, and `default_provider` pointing at a profile that was gone --
    /// silently, with exit 0.
    #[test]
    fn adding_a_profile_refuses_an_inline_providers_table_rather_than_replacing_it() {
        let mut document = "default_provider = \"work\"\n\
             providers = { work = { type = \"anthropic-messages\", model = \"m1\" } }\n"
            .parse::<toml_edit::DocumentMut>()
            .expect("parse");
        let error = upsert_profile_document(
            &mut document,
            "home",
            "openai-responses",
            "m2",
            None,
            &ProfileTuning::default(),
        )
        .expect_err("an inline `providers` must be refused, not overwritten");
        assert!(
            error.to_string().contains("is not a section"),
            "the refusal must say what to do about it: {error}"
        );
        let config: config::ConfigFile =
            toml::from_str(&document.to_string()).expect("re-parse config");
        assert!(
            config.providers.contains_key("work"),
            "the existing profile must survive a refused add"
        );
        assert_eq!(config.default_provider.as_deref(), Some("work"));
    }

    /// The other half of the same bug. Removal *is* safe on either spelling, and the caller cannot
    /// infer whether it happened from a separate `as_table` probe: that probe said "absent" while
    /// the profile was in the file, so `remove` deleted the credential, dropped
    /// `default_provider`, left `[providers.work]` in place, and reported "no profile was
    /// configured" about one `meka provider list` still showed.
    #[test]
    fn removing_a_profile_reaches_an_inline_providers_table_and_says_it_did() {
        let mut document = "default_provider = \"work\"\n\
             providers = { work = { type = \"anthropic-messages\", model = \"m1\" }, \
             side = { type = \"openai-responses\", model = \"m2\" } }\n"
            .parse::<toml_edit::DocumentMut>()
            .expect("parse");
        assert!(
            remove_profile_document(&mut document, "work"),
            "a profile that was there must be reported as removed"
        );
        let config: config::ConfigFile =
            toml::from_str(&document.to_string()).expect("re-parse config");
        assert!(!config.providers.contains_key("work"));
        assert!(config.providers.contains_key("side"));
        assert_eq!(config.default_provider, None);

        assert!(
            !remove_profile_document(&mut document, "work"),
            "a second removal has nothing to remove and must say so"
        );
    }

    #[test]
    fn test_remove_profile_document_clears_dangling_default() {
        let mut document = toml_edit::DocumentMut::new();
        upsert_profile_document(
            &mut document,
            "work",
            "claude-subscription",
            "claude-x",
            None,
            &ProfileTuning::default(),
        )
        .expect("upsert work");
        upsert_profile_document(
            &mut document,
            "personal",
            "openai-chat-completions",
            "gpt-4o",
            None,
            &ProfileTuning::default(),
        )
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
