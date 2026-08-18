//! The on-disk Markdown stores meka owns: skills and memories.
//!
//! Both are process-wide rather than per-session, which is why they carry their own scopes
//! (`skills:r` / `skills:w`, `memory:r` / `memory:w`) instead of riding on `sessions:*`. A bridge
//! token that should be able to run turns has no business emptying the memory directory.
//!
//! These endpoints are *not* gated by `[skills] agent_managed` or `[memory] access`. Those flags
//! describe what the model may do on its own initiative; a token presented to this API is the
//! operator acting remotely, and is the wire equivalent of running `meka skill add` in a shell. The
//! same reasoning is why the `source_url` refusal that `skill_write` and `skill_delete` apply is
//! absent here: an *agent* edit to an upstream-managed skill is futile because the next
//! `meka skill update` reverts it, but a person deciding to edit one anyway is making a choice the
//! CLI has always let them make. The field is returned so a client can warn its own user.

use axum::{
    Extension, Json,
    body::Bytes,
    extract::{Path as AxumPath, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::server::{
    auth::Principal,
    errors::{ErrorKind, ProblemDetail},
    scope::{self, ANY_READ_SCOPES},
    state::ServerState,
};

/// Turn a store-layer `Err(String)` into a 422. Every one of them is a statement about the caller's
/// input (bad name, empty description, a symlink in the way), not a server fault.
#[allow(clippy::result_large_err)]
fn store_error(message: String) -> ProblemDetail {
    ProblemDetail::new(
        ErrorKind::InvalidBody,
        StatusCode::UNPROCESSABLE_ENTITY,
        message,
    )
}

#[allow(clippy::result_large_err)]
fn not_found(noun: &str, name: &str) -> ProblemDetail {
    ProblemDetail::new(
        ErrorKind::NotFound,
        StatusCode::NOT_FOUND,
        format!("no {} named '{}'", noun, name),
    )
}

/// The store is switched off, or meka could not resolve its config directory.
///
/// `not-found` rather than `invalid-body`: nothing is wrong with the request, there is simply
/// nowhere to write. A client switching on `type` would read `invalid-body` as "fix your JSON" and
/// retry forever against a store that does not exist.
#[allow(clippy::result_large_err)]
fn store_disabled(noun: &str) -> ProblemDetail {
    ProblemDetail::new(
        ErrorKind::NotFound,
        StatusCode::NOT_FOUND,
        format!(
            "{} is disabled, or the meka config directory could not be resolved",
            noun
        ),
    )
}

// ---------------------------------------------------------------------------
// Skills
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, ToSchema)]
pub struct SkillDetail {
    pub name: String,
    pub description: String,
    /// Listing rank, 0..=9, lower first. Orders the `[Skills]` index the model sees and decides
    /// which entries that index's cap drops.
    pub priority: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    /// When set, `meka skill update` re-fetches this skill and overwrites local edits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_url: Option<String>,
    /// The `SKILL.md` body. Present on `GET /v1/skills/{name}`, absent from the collection listing
    /// so the palette stays small.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
}

impl SkillDetail {
    fn from_skill(skill: &crate::skills::Skill, body: Option<String>) -> Self {
        Self {
            name: skill.name.clone(),
            description: skill.description.clone(),
            priority: skill.priority,
            version: skill.version.clone(),
            author: skill.author.clone(),
            source_url: skill.source_url.clone(),
            body,
        }
    }
}

/// `GET /v1/skills/{name}`: one skill, body included.
#[utoipa::path(
    get,
    path = "/v1/skills/{name}",
    tag = "skills",
    params(("name" = String, Path, description = "Skill name")),
    responses(
        (status = 200, description = "Skill with its body", body = SkillDetail),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such skill", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn get_skill(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<SkillDetail>, ProblemDetail> {
    // `skills:r`, not any read scope: the palette at `GET /v1/skills` is a listing, but this
    // returns the body, which is the instruction text itself.
    scope::require(&principal, "skills:r")?;
    let installed = state.shared.skills.current().await;
    let skill = installed
        .iter()
        .find(|skill| skill.name == name)
        .ok_or_else(|| not_found("skill", &name))?;
    // `load_skill_source`, not `load_skill_body`: the latter is what the *agent* is handed, with a
    // base-directory header prepended. Returning that here would make `GET` and `PUT` disagree, and
    // an editing client's read-modify-write would bake the header into the file, once per cycle.
    let body = crate::skills::load_skill_source(skill)
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to read skill body", error))?;
    Ok(Json(SkillDetail::from_skill(skill, Some(body))))
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct WriteSkillRequest {
    /// One line saying when to reach for this skill. Required; an empty description produces a
    /// skill that cannot be loaded again.
    pub description: String,
    /// 0..=9, lower listed first. Omit to keep an existing skill's priority; defaults to 5 on
    /// creation.
    #[serde(default)]
    pub priority: Option<u8>,
    /// The `SKILL.md` body. Omit to keep the existing body when updating; send `""` to clear it.
    /// The same semantics as `memory_write` and `skill_write`, which exist so a caller correcting
    /// a description does not have to resend a body it never wanted to touch.
    #[serde(default)]
    pub body: Option<String>,
    /// Attribution. Recorded on creation only; an existing skill keeps whatever author it already
    /// had, so editing a hand-written skill never reassigns it.
    #[serde(default)]
    pub author: Option<String>,
}

/// `PUT /v1/skills/{name}`: create or update a skill.
#[utoipa::path(
    put,
    path = "/v1/skills/{name}",
    tag = "skills",
    params(("name" = String, Path, description = "Skill name, [A-Za-z0-9_-]")),
    request_body = WriteSkillRequest,
    responses(
        (status = 200, description = "Skill written", body = SkillDetail),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "The skill store is disabled", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid name, empty description, or an unwritable path", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn put_skill(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    AxumPath(name): AxumPath<String>,
    raw_body: Bytes,
) -> Result<Json<SkillDetail>, ProblemDetail> {
    scope::require(&principal, "skills:w")?;
    let body: WriteSkillRequest = serde_json::from_slice(&raw_body)
        .map_err(|error| store_error(format!("invalid skill request body: {}", error)))?;
    let root = state
        .shared
        .skills
        .root()
        .ok_or_else(|| store_disabled("skills"))?
        .to_path_buf();

    if body
        .priority
        .is_some_and(|priority| priority > crate::store::MAX_PRIORITY)
    {
        return Err(store_error(format!(
            "priority must be between {} and {}",
            crate::store::MIN_PRIORITY,
            crate::store::MAX_PRIORITY
        )));
    }
    // Omitted means "leave it alone", the way `body` and `author` already do. Resetting to the
    // default instead would make the obvious edit -- `GET`, change the text, `PUT` it back --
    // silently demote a prioritised skill, and priority both orders the `[Skills]` index the model
    // reads and decides which entries the index cap drops. Defaults only on creation.
    let priority = match body.priority {
        Some(priority) => priority,
        None => state
            .shared
            .skills
            .current()
            .await
            .iter()
            .find(|skill| skill.name == name)
            .map_or(crate::store::DEFAULT_PRIORITY, |skill| skill.priority),
    };
    let description = crate::store::normalize_description(&body.description);
    crate::skills::write_skill(
        &root,
        &name,
        &description,
        priority,
        body.author.as_deref(),
        body.body.as_deref(),
    )
    .map_err(store_error)?;
    // Before the read-back below, which goes through the same cache: without this a second write
    // inside one mtime tick that renders to the same length is invisible, and this response would
    // echo the *previous* values as though they had been applied.
    state.shared.skills.invalidate().await;
    tracing::info!("wrote skill '{}' via HTTP", name);

    // Read back through the cache rather than echoing the request: the response then reports what
    // is actually on disk, including the `author` and `source_url` a pre-existing skill kept.
    let installed = state.shared.skills.current().await;
    let skill = installed
        .iter()
        .find(|skill| skill.name == name)
        .ok_or_else(|| {
            ProblemDetail::internal_sanitized(
                "skill vanished between write and read-back",
                format!(
                    "skill '{}' is not in the cache after a successful write",
                    name
                ),
            )
        })?;
    Ok(Json(SkillDetail::from_skill(skill, None)))
}

/// `DELETE /v1/skills/{name}`: remove a skill and everything bundled with it.
#[utoipa::path(
    delete,
    path = "/v1/skills/{name}",
    tag = "skills",
    params(("name" = String, Path, description = "Skill name")),
    responses(
        (status = 204, description = "Skill removed"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such skill", body = ProblemDetail),
        (status = 422, description = "Invalid name, or a symlinked skill directory", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn delete_skill(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    AxumPath(name): AxumPath<String>,
) -> Result<StatusCode, ProblemDetail> {
    scope::require(&principal, "skills:w")?;
    let root = state
        .shared
        .skills
        .root()
        .ok_or_else(|| store_disabled("skills"))?
        .to_path_buf();
    // Validated before the probe, the way `delete_memory` does. Probing first answered "does this
    // directory exist" for any string a caller sent, including one the name rules would have
    // refused, which is a filesystem oracle reachable with only `skills:w`.
    crate::skills::validate_skill_name(&name).map_err(store_error)?;
    // Classified on the path rather than on the error text, the way `delete_memory` does. The
    // store layer races its own existence check against `remove_dir_all`, whose ENOENT renders as
    // "No such file or directory" and would fall through a substring match for "not found" into a
    // 422 for what is plainly a 404.
    let missing = !root.join(&name).is_dir();
    crate::skills::delete_skill(&root, &name).map_err(|message| {
        if missing {
            not_found("skill", &name)
        } else {
            store_error(message)
        }
    })?;
    state.shared.skills.invalidate().await;
    tracing::info!("deleted skill '{}' via HTTP", name);
    Ok(StatusCode::NO_CONTENT)
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, ToSchema)]
pub struct MemoryDetail {
    pub name: String,
    pub description: String,
    /// 0..=9, lower first. Unlike a skill's, a memory's priority *is* shown to the model, because
    /// it says how heavily to weigh a note the model is already reasoning from.
    pub priority: u8,
    /// Last-modified time, RFC 3339. Breaks priority ties (newest first) and is rendered to the
    /// model as a human-readable age.
    pub updated_at: String,
    /// Present on `GET /v1/memory/{name}`, absent from the collection listing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct MemoryListResponse {
    pub memories: Vec<MemoryDetail>,
    /// Files in the memory root that could not be parsed into a memory, with the reason.
    ///
    /// Reported rather than omitted because the failure mode is silence: from inside a session a
    /// skipped file is indistinguishable from a memory nobody wrote, so someone can drop in a
    /// standing rule and believe it is in force.
    pub skipped: Vec<SkippedMemoryView>,
    /// Valid memories the discovery cap dropped. Distinct from `skipped`: these parsed fine.
    pub ignored_over_cap: usize,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct SkippedMemoryView {
    pub file: String,
    pub reason: String,
}

fn memory_detail(memory: &crate::memory::Memory, body: Option<String>) -> MemoryDetail {
    MemoryDetail {
        name: memory.name.clone(),
        description: memory.description.clone(),
        priority: memory.priority,
        updated_at: chrono::DateTime::<chrono::Utc>::from(memory.mtime).to_rfc3339(),
        body,
    }
}

/// `GET /v1/memory`: the memory index, most important first.
#[utoipa::path(
    get,
    path = "/v1/memory",
    tag = "memory",
    responses(
        (status = 200, description = "Memory index", body = MemoryListResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn list_memories(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
) -> Result<Json<MemoryListResponse>, ProblemDetail> {
    scope::require(&principal, "memory:r")?;
    let index = state.shared.memories.current().await;
    Ok(Json(MemoryListResponse {
        memories: index
            .memories
            .iter()
            .map(|memory| memory_detail(memory, None))
            .collect(),
        skipped: index
            .skipped
            .iter()
            .map(|skipped| SkippedMemoryView {
                file: skipped.file.clone(),
                reason: skipped.reason.clone(),
            })
            .collect(),
        ignored_over_cap: index.ignored_over_cap,
    }))
}

/// `GET /v1/memory/{name}`: one memory, body included.
#[utoipa::path(
    get,
    path = "/v1/memory/{name}",
    tag = "memory",
    params(("name" = String, Path, description = "Memory name")),
    responses(
        (status = 200, description = "Memory with its body", body = MemoryDetail),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such memory", body = ProblemDetail),
        (status = 422, description = "The file exists but could not be parsed as a memory", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn get_memory(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<MemoryDetail>, ProblemDetail> {
    scope::require(&principal, "memory:r")?;
    let index = state.shared.memories.current().await;
    let memory = index
        .memories
        .iter()
        .find(|memory| memory.name == name)
        .ok_or_else(|| {
            // A file that failed to parse is a different answer from one that never existed, and
            // the reason is what the operator needs to fix it.
            match index.skip_reason(&name) {
                Some(reason) => store_error(format!(
                    "memory '{}' exists on disk but could not be parsed: {}",
                    name, reason
                )),
                None => not_found("memory", &name),
            }
        })?;
    let body = crate::memory::load_memory_body(memory)
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to read memory body", error))?;
    Ok(Json(memory_detail(memory, Some(body))))
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct WriteMemoryRequest {
    /// One line saying what this memory is for.
    pub description: String,
    /// 0..=9, lower first. Omit to keep an existing memory's priority; defaults to 5 on creation.
    #[serde(default)]
    pub priority: Option<u8>,
    /// Omit to keep the existing body when updating; send `""` to clear it.
    #[serde(default)]
    pub body: Option<String>,
}

/// `PUT /v1/memory/{name}`: create or update a memory.
#[utoipa::path(
    put,
    path = "/v1/memory/{name}",
    tag = "memory",
    params(("name" = String, Path, description = "Memory name, [A-Za-z0-9_-]")),
    request_body = WriteMemoryRequest,
    responses(
        (status = 200, description = "Memory written", body = MemoryDetail),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "The memory store is disabled", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid name, empty description, or an unwritable path", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn put_memory(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    AxumPath(name): AxumPath<String>,
    raw_body: Bytes,
) -> Result<Json<MemoryDetail>, ProblemDetail> {
    scope::require(&principal, "memory:w")?;
    let body: WriteMemoryRequest = serde_json::from_slice(&raw_body)
        .map_err(|error| store_error(format!("invalid memory request body: {}", error)))?;
    let root = state
        .shared
        .memories
        .root()
        .ok_or_else(|| store_disabled("memory"))?
        .to_path_buf();

    if body
        .priority
        .is_some_and(|priority| priority > crate::store::MAX_PRIORITY)
    {
        return Err(store_error(format!(
            "priority must be between {} and {}",
            crate::store::MIN_PRIORITY,
            crate::store::MAX_PRIORITY
        )));
    }
    // Preserved on omission, as in `put_skill`; a memory's priority is rendered to the model, so
    // resetting it silently changes how heavily a standing note is weighed.
    let priority = match body.priority {
        Some(priority) => priority,
        None => state
            .shared
            .memories
            .current()
            .await
            .memories
            .iter()
            .find(|memory| memory.name == name)
            .map_or(crate::store::DEFAULT_PRIORITY, |memory| memory.priority),
    };
    let description = crate::store::normalize_description(&body.description);
    crate::memory::write_memory(&root, &name, &description, priority, body.body.as_deref())
        .map_err(store_error)?;
    // See the note on `put_skill`: the read-back below must not be served a stale snapshot.
    state.shared.memories.invalidate().await;
    tracing::info!("wrote memory '{}' via HTTP", name);

    let index = state.shared.memories.current().await;
    let memory = index
        .memories
        .iter()
        .find(|memory| memory.name == name)
        .ok_or_else(|| {
            // The one way a successful write can be missing from the index: discovery caps the
            // store and this entry sorted below the cut. The file is on disk and the write did
            // happen, so reporting a 500 would be both the wrong class and the wrong story.
            if index.ignored_over_cap > 0 {
                store_error(format!(
                    "memory '{}' was written, but the store is over its discovery cap ({} entries \
                     ignored) and this one sorted below the cut, so no session will load it. \
                     Delete or re-prioritise some memories.",
                    name, index.ignored_over_cap
                ))
            } else {
                ProblemDetail::internal_sanitized(
                    "memory vanished between write and read-back",
                    format!(
                        "memory '{}' is not in the index after a successful write",
                        name
                    ),
                )
            }
        })?;
    Ok(Json(memory_detail(memory, None)))
}

/// `DELETE /v1/memory/{name}`: remove a memory.
#[utoipa::path(
    delete,
    path = "/v1/memory/{name}",
    tag = "memory",
    params(("name" = String, Path, description = "Memory name")),
    responses(
        (status = 204, description = "Memory removed"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such memory", body = ProblemDetail),
        (status = 422, description = "Invalid name, or a symlinked path", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn delete_memory(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    AxumPath(name): AxumPath<String>,
) -> Result<StatusCode, ProblemDetail> {
    scope::require(&principal, "memory:w")?;
    let root = state
        .shared
        .memories
        .root()
        .ok_or_else(|| store_disabled("memory"))?
        .to_path_buf();
    crate::memory::validate_memory_name(&name).map_err(store_error)?;
    let path = crate::memory::memory_file_in(&root, &name);
    // Before the `is_file` check, which follows symlinks: a link pointing at a real file would pass
    // it, and `remove_file` would then take the link and leave the target. Same order as
    // `memory_delete`; see `crate::store::reject_symlinked_path`.
    crate::store::reject_symlinked_path(&path, "memory").map_err(store_error)?;
    if !path.is_file() {
        return Err(not_found("memory", &name));
    }
    tokio::fs::remove_file(&path)
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to delete memory", error))?;
    state.shared.memories.invalidate().await;
    tracing::info!("deleted memory '{}' via HTTP", name);
    Ok(StatusCode::NO_CONTENT)
}

// ---------------------------------------------------------------------------
// Tools, instructions, providers
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, ToSchema)]
pub struct ToolView {
    pub name: String,
    pub description: String,
    /// Permission level this tool needs: `none`, `read`, `ask`, or `write`.
    pub required_permission: String,
    /// Whether the tool is deferred: present in the catalogue by name but with its schema withheld
    /// until the model calls `load_tool`.
    pub deferred: bool,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ToolsResponse {
    pub tools: Vec<ToolView>,
}

/// `GET /v1/sessions/{id}/tools`: what this session's agent can call.
///
/// Per-session rather than server-global because the catalogue includes MCP tools and reflects that
/// registry's deferred / loaded state, neither of which is a process-wide fact.
#[utoipa::path(
    get,
    path = "/v1/sessions/{id}/tools",
    tag = "discovery",
    params(("id" = uuid::Uuid, Path, description = "Session UUID")),
    responses(
        (status = 200, description = "Tool catalogue", body = ToolsResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 409, description = "Session is not loaded; run a turn first", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn list_tools(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    AxumPath(id): AxumPath<uuid::Uuid>,
) -> Result<Json<ToolsResponse>, ProblemDetail> {
    scope::require(&principal, "sessions:r")?;
    crate::server::reattach::require_session_exists(&state, id).await?;
    // Deliberately does NOT revive: re-attaching takes the session's cross-process file lock for
    // up to `idle_timeout`, and a read scope must not be able to seize a write-exclusive resource.
    // See the same note on `GET /context`. A catalogue needs a live registry, so an evicted session
    // gets a 409 rather than a guess assembled from process-wide defaults.
    let entry = state
        .sessions
        .read()
        .await
        .get(&id)
        .cloned()
        .ok_or_else(|| {
            ProblemDetail::new(
                ErrorKind::SessionNotLoaded,
                StatusCode::CONFLICT,
                "session is not loaded; submit a turn to load it before reading its tool catalogue",
            )
            .with("session_id", id.to_string())
        })?;
    // The hoisted clone, not `runtime.tool_registry`: a streaming turn holds the runtime mutex from
    // admission to completion, so reading through it would make this hang for minutes on exactly
    // the sessions a client is most likely to ask about. The registry is `Arc`-backed, so this
    // observes the same one the agent dispatches through, live MCP updates included.
    let catalogue = entry.tool_registry.tool_catalogue();
    Ok(Json(ToolsResponse {
        tools: catalogue
            .into_iter()
            .map(|(name, description, permission, deferred)| ToolView {
                name,
                description,
                required_permission: permission.to_string(),
                deferred,
            })
            .collect(),
    }))
}

#[derive(Debug, Serialize, ToSchema)]
pub struct InstructionsResponse {
    /// Where the instructions came from, for an operator diagnosing "why is it behaving like
    /// that". Absent when none are configured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// The resolved instruction text the agent runs under, or `null` when there is none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// `GET /v1/instructions`: the system instructions this server's agents run under.
#[utoipa::path(
    get,
    path = "/v1/instructions",
    tag = "discovery",
    responses(
        (status = 200, description = "Resolved instructions", body = InstructionsResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn instructions(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
) -> Result<Json<InstructionsResponse>, ProblemDetail> {
    // `sessions:r` rather than any read scope: this returns the *whole* system-instruction text,
    // which is the same class of content as a skill body and is gated for the same reason (see
    // `get_skill`). A `schedule:r` token has no business reading the persona the agent runs under.
    // `sessions:r` is the right pairing because anyone who can read a session's messages can
    // already infer much of it.
    scope::require(&principal, "sessions:r")?;
    let config = &state.shared.config;
    Ok(Json(InstructionsResponse {
        source: config.user_instructions_source.clone(),
        content: config.user_instructions.clone(),
    }))
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ProviderView {
    pub name: String,
    /// Backend kind: the wire protocol this profile speaks, e.g. `anthropic-messages`,
    /// `openai-responses`, or `chatgpt-subscription`.
    #[serde(rename = "type")]
    pub backend: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Whether this is the profile the server is running.
    pub active: bool,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ProvidersResponse {
    pub providers: Vec<ProviderView>,
}

/// `GET /v1/providers`: configured provider profiles, names and models only.
///
/// Read-only by design. Provider selection is config-only with no environment tier, precisely so an
/// ambient value can never silently rebind which account a named profile bills; a per-session or
/// per-request override would reintroduce exactly that. No credentials are returned: they live in
/// the database, keyed by profile name, and never transit this API.
#[utoipa::path(
    get,
    path = "/v1/providers",
    tag = "discovery",
    responses(
        (status = 200, description = "Configured provider profiles", body = ProvidersResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn providers(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
) -> Result<Json<ProvidersResponse>, ProblemDetail> {
    scope::require_any(&principal, ANY_READ_SCOPES)?;
    let config = &state.shared.config;
    // `active_profile`, not `provider_name`: the former is the profile key, which is what these
    // rows are named by. They coincide today, but the credential store is keyed on the profile.
    let active = config.active_profile.as_deref();
    let providers: Vec<ProviderView> = config
        .provider_profiles
        .iter()
        .map(|(name, backend, model)| ProviderView {
            name: name.clone(),
            backend: backend.clone(),
            model: model.clone(),
            active: Some(name.as_str()) == active,
        })
        .collect();
    Ok(Json(ProvidersResponse { providers }))
}
