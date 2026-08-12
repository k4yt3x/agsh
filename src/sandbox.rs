//! Filesystem sandboxing for read-only command execution.
//!
//! On Linux, uses Landlock LSM (kernel 5.13+) to restrict child processes to read-only filesystem
//! access. On macOS, uses `sandbox-exec`. On Windows, spawns the child with a duplicated primary
//! token dropped to Low integrity via `SetTokenInformation(TokenIntegrityLevel, …)`; this blocks
//! writes to anything outside the documented Low-integrity surface (the user's home directory,
//! `%APPDATA%`, Program Files, system dirs, etc.). On unsupported platforms, sandboxing is
//! unavailable and shell execution is gated by the permission level alone.

/// What kind of read-mode sandbox is available on this platform. Resolved once at config time and
/// threaded into `tools::shell::ExecuteCommandTool` so the spawn path knows which argv shape and
/// `pre_exec` hook to use.
#[derive(Debug, Clone)]
pub enum SandboxCapability {
    /// Linux: filesystem-write restriction via Landlock LSM (kernel 5.13+). Below ABI v9 the kernel
    /// has no right governing `connect(2)` on a *pathname* Unix socket, so dbus and systemd-user
    /// stay reachable and a confined process can have them write on its behalf; from v9 that right
    /// is handled and granted nowhere, which also costs socket-based clients like `docker` and
    /// `psql`. Prefer Bubblewrap when available: its tmpfs masks remove the sockets outright, on
    /// every kernel.
    #[cfg(target_os = "linux")]
    Landlock { abi_version: i32 },
    /// Linux: read-only root bind via `bwrap --ro-bind /` plus tmpfs masks over `/tmp`, `/run`,
    /// `/var/tmp`, and `$XDG_RUNTIME_DIR`. Blocks both filesystem writes and IPC-socket mutation;
    /// network is unrestricted.
    #[cfg(target_os = "linux")]
    Bubblewrap { bwrap_path: std::path::PathBuf },
    /// macOS: `sandbox-exec` with the hardened SBPL profile defined in
    /// [`SANDBOX_PROFILE_READONLY`]. Blocks filesystem writes and IPC mutation (no launchd,
    /// pasteboard, LaunchServices, etc.); network is unrestricted.
    #[cfg(target_os = "macos")]
    SandboxExec,
    /// Windows: child runs with a duplicated primary token dropped to Low integrity. Blocks writes
    /// outside the Low-integrity surface (user home, AppData, Program Files); IPC mutation is
    /// constrained but not as tightly as Linux/macOS.
    #[cfg(target_os = "windows")]
    LowIntegrity,
    /// No sandbox available on this platform / configuration. Read-mode shell commands hard-error
    /// rather than silently bypass the sandbox.
    Unavailable,
}

/// Result of probing a specific sandbox backend at config-resolution time. The probe is run once
/// per meka launch (twice when the resolver needs to consider both Landlock and Bubblewrap for
/// auto-pick) and cached on `ResolvedConfig.backend_probe`.
#[derive(Debug, Clone)]
pub enum BackendProbe {
    Ok(SandboxCapability),
    /// The backend's prerequisite is missing: `bwrap` isn't on `$PATH`, the Landlock kernel ABI
    /// isn't supported, etc. The `reason` is plain text and is plumbed into user-facing
    /// warnings/errors verbatim.
    Missing {
        reason: String,
    },
    /// Linux + bubblewrap only: the user-namespace smoke test failed with stderr that matched the
    /// documented denial fingerprints. Stored stderr is truncated to a few KiB. The only
    /// constructor (`smoke_test_bwrap`) is Linux-only, so the variant is dead on other platforms;
    /// the explicit allow lets non-Linux clippy stay clean without hiding regressions on Linux.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    UserNamespaceDenied {
        stderr: String,
    },
    /// The asked-for backend doesn't apply on this platform. No production code currently
    /// constructs this variant (the legacy non-Linux `probe_*` wrappers were folded into
    /// `resolve_sandbox_backend`), so the explicit allow is for the test-only constructor in
    /// `tests::test_backend_unavailable_reason_maps_each_variant`.
    #[allow(dead_code)]
    UnsupportedPlatform,
}

/// Snapshot of the sandbox-relevant config slice. Carried by components that need to emit the
/// sandbox warns (`warn_if_sandbox_issues`) without depending on the whole `ResolvedConfig`.
///
/// All fields are functionally only read on Linux (`warn_if_sandbox_issues` early-returns on other
/// platforms because the warnings reference Linux-only config keys), but the struct is constructed
/// unconditionally so the call sites in `src/main.rs` and `src/repl.rs` don't need a platform
/// branch. The `cfg_attr(not(target_os = "linux"), allow(dead_code))]` silences the "field never
/// read" warning on non-Linux without hiding real regressions on Linux where the lint stays loud.
#[derive(Clone)]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub struct SandboxState {
    pub enabled: bool,
    pub backend: crate::config::SandboxBackend,
    pub auto_resolved: bool,
    pub probe: BackendProbe,
}

impl SandboxState {
    pub fn from_config(config: &crate::config::ResolvedConfig) -> Self {
        Self {
            enabled: config.sandbox,
            backend: config.sandbox_backend,
            auto_resolved: config.sandbox_auto_resolved,
            probe: config.backend_probe.clone(),
        }
    }
}

/// Where in the meka lifecycle the sandbox-state check is happening. The "stronger sandbox
/// available" nudge (Warn 2) only fires at startup; "backend unavailable" (Warn 1) fires at every
/// relevant boundary because the user needs to know read-mode shell is broken right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarnContext {
    /// Once-per-launch warn during `ResolvedConfig` construction or agent setup. Both Warn 1 and
    /// Warn 2 fire here.
    Startup,
    /// Initial permission mode was `Read` at `meka --permission read` launch. Only Warn 1 fires.
    InitialReadMode,
    /// User pressed Shift+Tab and cycled into `Read`. Only Warn 1 fires.
    ReadModeEntry,
}

/// Emit any relevant sandbox warnings for the configured backend state.
///
/// * **Warn 1** (backend unavailable): probe failed and `sandbox = true`. Read-mode shell commands
///   will hard-error at use time, so we tell the user up front. Re-emitted at every lifecycle
///   boundary.
/// * **Warn 2** (could be stronger): the user has not pinned a backend and we auto-resolved to
///   landlock because bubblewrap wasn't usable. Nudges them once toward installing bwrap, with an
///   explicit escape hatch (pin landlock to suppress). Startup only.
pub fn warn_if_sandbox_issues(state: &SandboxState, context: WarnContext) {
    if !state.enabled {
        return;
    }

    // `sandbox_backend` is a Linux-only config knob; the warnings below name it directly and would
    // be misleading on macOS / Windows where the platform has a single fixed backend. On those
    // hosts an unusable platform sandbox is a near-impossible configuration and surfaces at use
    // time via the hard-error path in `src/tools/shell.rs` anyway.
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (state, context);
        return;
    }

    #[cfg(target_os = "linux")]
    {
        if let Some(reason) = backend_unavailable_reason(&state.probe) {
            // We deliberately don't suggest a specific alternative backend here; the "other"
            // backend might also be unavailable on this host (kernel without Landlock, bwrap not
            // installed, etc.).
            tracing::warn!(
                "read-mode sandbox: {} (configured: {}). Read-mode shell commands will fail \
                 until this is fixed. Set [shell].sandbox_backend in config.toml (or \
                 MEKA_SANDBOX_BACKEND) to a usable backend, or disable sandboxing.",
                reason,
                state.backend,
            );
            return;
        }

        if context == WarnContext::Startup
            && state.auto_resolved
            && matches!(state.backend, crate::config::SandboxBackend::Landlock)
        {
            tracing::warn!(
                "using Landlock for sandbox; install Bubblewrap for stronger \
                 protection, or pin `sandbox_backend = \"landlock\"` to suppress this warning."
            );
        }
    }
}

/// Human-readable reason a backend probe failed, or `None` when the probe is `Ok`. Used by both the
/// startup `warn!` path ([`warn_if_sandbox_issues`]) and the lazy hard-error path in
/// `src/tools/shell.rs` so the two surfaces stay in sync.
pub(crate) fn backend_unavailable_reason(probe: &BackendProbe) -> Option<String> {
    match probe {
        BackendProbe::Ok(_) => None,
        BackendProbe::Missing { reason } => Some(reason.clone()),
        BackendProbe::UserNamespaceDenied { stderr } => {
            let first_line = stderr.lines().next().unwrap_or("").trim();
            if first_line.is_empty() {
                Some("user namespaces are denied on this host".to_string())
            } else {
                Some(format!(
                    "user namespaces are denied on this host ({})",
                    first_line
                ))
            }
        }
        BackendProbe::UnsupportedPlatform => {
            Some("backend is not supported on this platform".to_string())
        }
    }
}

/// Probe a specific sandbox backend. Linux-only: the `SandboxBackend` enum represents
/// Linux-specific backends, and non-Linux platforms route through `detect()` in
/// `src/config.rs::resolve_sandbox_backend` instead.
#[cfg(target_os = "linux")]
pub fn probe_backend(backend: crate::config::SandboxBackend) -> BackendProbe {
    match backend {
        crate::config::SandboxBackend::Landlock => probe_landlock(),
        crate::config::SandboxBackend::Bubblewrap => probe_bubblewrap(),
    }
}

#[cfg(target_os = "linux")]
fn probe_landlock() -> BackendProbe {
    match detect_landlock() {
        Some(abi_version) => BackendProbe::Ok(SandboxCapability::Landlock { abi_version }),
        None => BackendProbe::Missing {
            reason: "Landlock LSM not supported by this kernel (needs Linux 5.13+)".to_string(),
        },
    }
}

#[cfg(target_os = "linux")]
fn probe_bubblewrap() -> BackendProbe {
    let Some(bwrap_path) = bwrap_on_path() else {
        return BackendProbe::Missing {
            reason: "bwrap not found on PATH".to_string(),
        };
    };

    match smoke_test_bwrap(&bwrap_path, BWRAP_PROBE_TIMEOUT) {
        SmokeResult::Success => BackendProbe::Ok(SandboxCapability::Bubblewrap { bwrap_path }),
        SmokeResult::UserNamespaceDenied { stderr } => BackendProbe::UserNamespaceDenied { stderr },
        SmokeResult::OtherFailure { reason } => BackendProbe::Missing { reason },
    }
}

#[cfg(target_os = "linux")]
const BWRAP_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(500);
#[cfg(target_os = "linux")]
const BWRAP_PROBE_POLL: std::time::Duration = std::time::Duration::from_millis(50);
#[cfg(target_os = "linux")]
const BWRAP_STDERR_LIMIT: usize = 64 * 1024;

/// Stderr substrings that indicate the kernel refused the user namespace request rather than some
/// other transient failure. Mirrors the fingerprint list Codex uses
/// (`temp/codex/codex-rs/sandboxing/src/bwrap.rs:30-35`).
#[cfg(target_os = "linux")]
const USER_NAMESPACE_FAILURE_MARKERS: &[&str] = &[
    "loopback: Failed RTM_NEWADDR",
    "loopback: Failed RTM_NEWLINK",
    "setting up uid map: Permission denied",
    "No permissions to create a new namespace",
];

#[cfg(target_os = "linux")]
enum SmokeResult {
    Success,
    UserNamespaceDenied { stderr: String },
    OtherFailure { reason: String },
}

/// Look up `bwrap` on `$PATH`. Returns the first entry whose target metadata is a regular
/// executable file. Manual implementation to avoid pulling in the `which` crate just for one
/// lookup.
#[cfg(target_os = "linux")]
fn bwrap_on_path() -> Option<std::path::PathBuf> {
    use std::os::unix::fs::PermissionsExt;

    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join("bwrap");
        let Ok(metadata) = std::fs::metadata(&candidate) else {
            continue;
        };
        if metadata.is_file() && metadata.permissions().mode() & 0o111 != 0 {
            return Some(candidate);
        }
    }
    None
}

/// Run a `bwrap … /bin/true` smoke test with a short timeout.
///
/// The flag set mirrors the production-path argv in `src/tools/shell.rs` so a host that succeeds
/// here also succeeds at runtime; without it, a kernel that quietly rejects (say)
/// `--unshare-cgroup-try` or `--die-with-parent` would pass the probe and blow past the lazy
/// hard-error gate the first time `execute_command` ran. `--unshare-net` is added on top so the
/// probe stays self-contained (no outbound DNS / network calls), even though production keeps the
/// host network namespace.
#[cfg(target_os = "linux")]
fn smoke_test_bwrap(bwrap_path: &std::path::Path, timeout: std::time::Duration) -> SmokeResult {
    use std::{io::Read, os::fd::AsRawFd};

    let mut child = match std::process::Command::new(bwrap_path)
        .args([
            "--new-session",
            "--die-with-parent",
            "--ro-bind",
            "/",
            "/",
            "--proc",
            "/proc",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-uts",
            "--unshare-ipc",
            "--unshare-cgroup-try",
            "--unshare-net",
            "/bin/true",
        ])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            return SmokeResult::OtherFailure {
                reason: format!("failed to spawn bwrap for smoke test: {}", error),
            };
        }
    };

    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                // Drain stderr non-blocking. The grandchild is gone so nothing else will write; any
                // data already buffered is all we'll get.
                let stderr = match child.stderr.take() {
                    Some(mut handle) => {
                        let fd = handle.as_raw_fd();
                        // SAFETY: fcntl with F_GETFL/F_SETFL on a valid open file descriptor;
                        // failure is fine and just means we'll attempt a regular read that may
                        // block briefly on a closed pipe.
                        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
                        if flags >= 0 {
                            unsafe {
                                libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
                            }
                        }
                        let mut bytes = Vec::new();
                        let mut take = handle.by_ref().take(BWRAP_STDERR_LIMIT as u64);
                        if let Err(error) = take.read_to_end(&mut bytes)
                            && error.kind() != std::io::ErrorKind::WouldBlock
                        {
                            tracing::debug!("smoke test stderr read: {}", error);
                        }
                        String::from_utf8_lossy(&bytes).into_owned()
                    }
                    None => String::new(),
                };
                if status.success() {
                    return SmokeResult::Success;
                }
                if USER_NAMESPACE_FAILURE_MARKERS
                    .iter()
                    .any(|marker| stderr.contains(marker))
                {
                    return SmokeResult::UserNamespaceDenied { stderr };
                }
                let truncated_stderr = stderr.lines().next().unwrap_or("").trim().to_string();
                let reason = if truncated_stderr.is_empty() {
                    format!("bwrap smoke test failed (exit {:?})", status.code())
                } else {
                    format!(
                        "bwrap smoke test failed (exit {:?}): {}",
                        status.code(),
                        truncated_stderr
                    )
                };
                return SmokeResult::OtherFailure { reason };
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    reap_smoke_test_child(&mut child);
                    return SmokeResult::OtherFailure {
                        reason: format!(
                            "bwrap smoke test exceeded {}ms timeout",
                            timeout.as_millis()
                        ),
                    };
                }
                std::thread::sleep(BWRAP_PROBE_POLL);
            }
            Err(error) => {
                reap_smoke_test_child(&mut child);
                return SmokeResult::OtherFailure {
                    reason: format!("bwrap smoke test wait failed: {}", error),
                };
            }
        }
    }
}

/// Best-effort cleanup of a stuck smoke-test child: kill it and reap its status so we don't leave a
/// zombie. Errors are logged at debug level only; by this point the smoke test has already failed
/// and the caller is about to return a higher-priority error reason.
#[cfg(target_os = "linux")]
fn reap_smoke_test_child(child: &mut std::process::Child) {
    if let Err(error) = child.kill() {
        tracing::debug!("bwrap smoke test: failed to kill stuck child: {}", error);
    }
    if let Err(error) = child.wait() {
        tracing::debug!("bwrap smoke test: failed to reap child: {}", error);
    }
}

/// Test-only "what's the strongest sandbox available right now?" entry point. Production code
/// consults [`crate::config::ResolvedConfig::backend_probe`] instead; tests reach for whatever
/// capability the host happens to support.
#[cfg(any(test, not(target_os = "linux")))]
pub fn detect() -> SandboxCapability {
    #[cfg(target_os = "linux")]
    {
        if let Some(version) = detect_landlock() {
            return SandboxCapability::Landlock {
                abi_version: version,
            };
        }
    }

    #[cfg(target_os = "macos")]
    {
        if std::path::Path::new("/usr/bin/sandbox-exec").exists() {
            return SandboxCapability::SandboxExec;
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Token-integrity APIs are available on every supported Windows version (7+). No runtime
        // probe is needed.
        return SandboxCapability::LowIntegrity;
    }

    #[allow(unreachable_code)]
    SandboxCapability::Unavailable
}

#[cfg(target_os = "linux")]
fn detect_landlock() -> Option<i32> {
    let version = unsafe {
        libc::syscall(
            libc::SYS_landlock_create_ruleset,
            std::ptr::null::<u8>(),
            0usize,
            LANDLOCK_CREATE_RULESET_VERSION,
        )
    };
    if version >= 1 {
        Some(version as i32)
    } else {
        None
    }
}

/// Apply Landlock read-only restrictions to the current process.
///
/// # Safety
///
/// This function uses raw syscalls and must only be called in a `pre_exec` context (after fork,
/// before exec) where the process is single-threaded. All operations are async-signal-safe
/// (syscalls only, no heap allocation).
#[cfg(target_os = "linux")]
pub unsafe fn apply_landlock_readonly(abi_version: i32) -> Result<(), i32> {
    unsafe {
        // PR_SET_NO_NEW_PRIVS is required for unprivileged Landlock usage
        if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 {
            return Err(*libc::__errno_location());
        }

        let attr = LandlockRulesetAttr {
            handled_access_fs: handled_access_for_abi(abi_version),
            handled_access_net: 0,
            scoped: scoped_for_abi(abi_version),
        };

        let ruleset_fd = libc::syscall(
            libc::SYS_landlock_create_ruleset,
            &attr as *const LandlockRulesetAttr,
            std::mem::size_of::<LandlockRulesetAttr>(),
            0u32,
        ) as i32;
        if ruleset_fd < 0 {
            return Err(*libc::__errno_location());
        }

        // Allow read + execute for the entire filesystem
        let root_fd = libc::open(c"/".as_ptr(), libc::O_PATH | libc::O_CLOEXEC);
        if root_fd < 0 {
            // `close(2)` is permitted to set `errno` even on success, so read the failure reason
            // before releasing the ruleset.
            let error = *libc::__errno_location();
            libc::close(ruleset_fd);
            return Err(error);
        }

        let path_beneath = LandlockPathBeneathAttr {
            allowed_access: LANDLOCK_ACCESS_FS_EXECUTE
                | LANDLOCK_ACCESS_FS_READ_FILE
                | LANDLOCK_ACCESS_FS_READ_DIR,
            parent_fd: root_fd,
        };

        let ret = libc::syscall(
            libc::SYS_landlock_add_rule,
            ruleset_fd,
            LANDLOCK_RULE_PATH_BENEATH,
            &path_beneath as *const LandlockPathBeneathAttr,
            0u32,
        );
        if ret < 0 {
            let error = *libc::__errno_location();
            libc::close(root_fd);
            libc::close(ruleset_fd);
            return Err(error);
        }
        libc::close(root_fd);

        let ret = libc::syscall(libc::SYS_landlock_restrict_self, ruleset_fd, 0u32);
        if ret < 0 {
            let error = *libc::__errno_location();
            libc::close(ruleset_fd);
            return Err(error);
        }
        libc::close(ruleset_fd);

        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn handled_access_for_abi(abi_version: i32) -> u64 {
    let mut access = LANDLOCK_ACCESS_FS_EXECUTE
        | LANDLOCK_ACCESS_FS_WRITE_FILE
        | LANDLOCK_ACCESS_FS_READ_FILE
        | LANDLOCK_ACCESS_FS_READ_DIR
        | LANDLOCK_ACCESS_FS_REMOVE_DIR
        | LANDLOCK_ACCESS_FS_REMOVE_FILE
        | LANDLOCK_ACCESS_FS_MAKE_CHAR
        | LANDLOCK_ACCESS_FS_MAKE_DIR
        | LANDLOCK_ACCESS_FS_MAKE_REG
        | LANDLOCK_ACCESS_FS_MAKE_SOCK
        | LANDLOCK_ACCESS_FS_MAKE_FIFO
        | LANDLOCK_ACCESS_FS_MAKE_BLOCK
        | LANDLOCK_ACCESS_FS_MAKE_SYM;
    if abi_version >= 2 {
        access |= LANDLOCK_ACCESS_FS_REFER;
    }
    if abi_version >= 3 {
        access |= LANDLOCK_ACCESS_FS_TRUNCATE;
    }
    // ABI v4 added network access flags (BIND_TCP, CONNECT_TCP), not filesystem flags
    if abi_version >= 5 {
        access |= LANDLOCK_ACCESS_FS_IOCTL_DEV;
    }
    if abi_version >= 9 {
        access |= LANDLOCK_ACCESS_FS_RESOLVE_UNIX;
    }
    access
}

/// IPC scoping flags for the ruleset. ABI v6 (kernel 6.12) added scoping; restricting it blocks the
/// sandboxed child from reaching abstract Unix sockets (D-Bus and similar) and from signalling
/// processes outside its own Landlock domain. Setting an unknown `scoped` bit makes
/// `landlock_create_ruleset` fail with `EINVAL`, so this stays zero below v6.
#[cfg(target_os = "linux")]
fn scoped_for_abi(abi_version: i32) -> u64 {
    if abi_version >= 6 {
        LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET | LANDLOCK_SCOPE_SIGNAL
    } else {
        0
    }
}

// Landlock constants
#[cfg(target_os = "linux")]
const LANDLOCK_CREATE_RULESET_VERSION: u32 = 1 << 0;
#[cfg(target_os = "linux")]
const LANDLOCK_RULE_PATH_BENEATH: u32 = 1;

#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_EXECUTE: u64 = 1 << 0;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_WRITE_FILE: u64 = 1 << 1;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_READ_FILE: u64 = 1 << 2;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_READ_DIR: u64 = 1 << 3;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_REMOVE_DIR: u64 = 1 << 4;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_REMOVE_FILE: u64 = 1 << 5;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_MAKE_CHAR: u64 = 1 << 6;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_MAKE_DIR: u64 = 1 << 7;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_MAKE_REG: u64 = 1 << 8;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_MAKE_SOCK: u64 = 1 << 9;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_MAKE_FIFO: u64 = 1 << 10;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_MAKE_BLOCK: u64 = 1 << 11;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_MAKE_SYM: u64 = 1 << 12;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_REFER: u64 = 1 << 13;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_TRUNCATE: u64 = 1 << 14;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_IOCTL_DEV: u64 = 1 << 15;
/// ABI v9 (kernel 7.1). Mediates `connect(2)` and addressed `sendmsg(2)` on *pathname* Unix
/// sockets, the class Landlock left entirely unmediated before it: the D-Bus system and session
/// buses, and `/run/systemd/private`. Without this bit in `handled_access_fs` a confined process
/// can hand work to a privileged daemon and have it done on its behalf, which is a complete bypass
/// of the filesystem boundary rather than a gap in it. `scoped` covers only *abstract* sockets, so
/// it does not reach these.
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_RESOLVE_UNIX: u64 = 1 << 16;

#[cfg(target_os = "linux")]
const LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET: u64 = 1 << 0;
#[cfg(target_os = "linux")]
const LANDLOCK_SCOPE_SIGNAL: u64 = 1 << 1;

// Landlock kernel structs (stack-allocated, no heap)
#[cfg(target_os = "linux")]
#[repr(C)]
struct LandlockRulesetAttr {
    handled_access_fs: u64,
    handled_access_net: u64,
    scoped: u64,
}

#[cfg(target_os = "linux")]
#[repr(C)]
struct LandlockPathBeneathAttr {
    allowed_access: u64,
    parent_fd: i32,
}

/// Path of the macOS `sandbox-exec` binary. Hardcoded (not PATH-searched) so a hostile `PATH` entry
/// can't shadow it with a wrapper that drops the sandbox.
#[cfg(target_os = "macos")]
pub const SANDBOX_EXEC_PATH: &str = "/usr/bin/sandbox-exec";

/// Read-mode SBPL profile for the macOS sandbox. Modeled after Codex's hardened Seatbelt profile
/// (Apache 2.0; see attribution inside the policy), which is itself inspired by Chrome's renderer
/// sandbox.
///
/// Threat-model parity with Linux Bubblewrap:
/// - Filesystem read-only: `(deny default)` denies writes; only `/dev/null` and PTY device nodes
///   get write access for legitimate shell behavior.
/// - IPC mutation blocked: `mach-lookup` is denied by default; only a curated allow-list of safe
///   Mach services is whitelisted. Mutation services (`com.apple.launchd`,
///   `com.apple.pasteboard.1`, `com.apple.launchservicesd`, the cfprefsd *write* path via
///   `user-preference-write`) are NOT in the allow-list.
/// - Network allowed: outbound BSD sockets, DNS resolution, TLS trust evaluation, and proxy/network
///   configuration reads are explicitly permitted.
#[cfg(target_os = "macos")]
pub const SANDBOX_PROFILE_READONLY: &str = r#"
; Vendored from Codex (Apache 2.0 License):
;   github.com/openai/codex/blob/main/codex-rs/sandboxing/src/seatbelt_base_policy.sbpl
;   github.com/openai/codex/blob/main/codex-rs/sandboxing/src/seatbelt_network_policy.sbpl
; The base policy is itself inspired by Chrome's renderer sandbox:
;   https://source.chromium.org/chromium/chromium/src/+/main:sandbox/policy/mac/common.sb

(version 1)

; start with closed-by-default
(deny default)

; broad filesystem read: agent needs to read arbitrary files in read-mode
(allow file-read*)
(allow file-test-existence)
(allow file-ioctl)
(allow file-map-executable)
(allow file-read-metadata)

; child processes inherit the policy of their parent
(allow process-exec)
(allow process-fork)
(allow signal (target same-sandbox))

; process-info
(allow process-info* (target same-sandbox))

; /dev/null writes are universally legitimate for shell redirects
(allow file-write-data
  (require-all
    (path "/dev/null")
    (vnode-type CHARACTER-DEVICE)))

; sysctls permitted (CPU / kernel info reads)
(allow sysctl-read
  (sysctl-name "hw.activecpu")
  (sysctl-name "hw.busfrequency_compat")
  (sysctl-name "hw.byteorder")
  (sysctl-name "hw.cacheconfig")
  (sysctl-name "hw.cachelinesize_compat")
  (sysctl-name "hw.cpufamily")
  (sysctl-name "hw.cpufrequency_compat")
  (sysctl-name "hw.cputype")
  (sysctl-name "hw.l1dcachesize_compat")
  (sysctl-name "hw.l1icachesize_compat")
  (sysctl-name "hw.l2cachesize_compat")
  (sysctl-name "hw.l3cachesize_compat")
  (sysctl-name "hw.logicalcpu_max")
  (sysctl-name "hw.machine")
  (sysctl-name "hw.model")
  (sysctl-name "hw.memsize")
  (sysctl-name "hw.ncpu")
  (sysctl-name "hw.nperflevels")
  (sysctl-name-prefix "hw.optional.arm.")
  (sysctl-name-prefix "hw.optional.armv8_")
  (sysctl-name "hw.packages")
  (sysctl-name "hw.pagesize_compat")
  (sysctl-name "hw.pagesize")
  (sysctl-name "hw.physicalcpu")
  (sysctl-name "hw.physicalcpu_max")
  (sysctl-name "hw.logicalcpu")
  (sysctl-name "hw.cpufrequency")
  (sysctl-name "hw.tbfrequency_compat")
  (sysctl-name "hw.vectorunit")
  (sysctl-name "machdep.cpu.brand_string")
  (sysctl-name "kern.argmax")
  (sysctl-name "kern.hostname")
  (sysctl-name "kern.maxfilesperproc")
  (sysctl-name "kern.maxproc")
  (sysctl-name "kern.osproductversion")
  (sysctl-name "kern.osrelease")
  (sysctl-name "kern.ostype")
  (sysctl-name "kern.osvariant_status")
  (sysctl-name "kern.osversion")
  (sysctl-name "kern.secure_kernel")
  (sysctl-name "kern.usrstack64")
  (sysctl-name "kern.version")
  (sysctl-name "sysctl.proc_cputype")
  (sysctl-name "vm.loadavg")
  (sysctl-name-prefix "hw.perflevel")
  (sysctl-name-prefix "kern.proc.pgrp.")
  (sysctl-name-prefix "kern.proc.pid.")
  (sysctl-name-prefix "net.routetable.")
)

; Java reads some CPU info via a misclassified "sysctl-write"
(allow sysctl-write
  (sysctl-name "kern.grade_cputype"))

; IOKit
(allow iokit-open
  (iokit-registry-entry-class "RootDomainUserClient"))

; Python multiprocessing
(allow ipc-posix-sem)

; PyTorch/libomp register OpenMP runtimes
(allow ipc-posix-shm-read-data
  ipc-posix-shm-write-create
  ipc-posix-shm-write-unlink
  (ipc-posix-name-regex #"^/__KMP_REGISTERED_LIB_[0-9]+$"))

; power management queries
(allow mach-lookup
  (global-name "com.apple.PowerManagement.control"))

; PTYs (interactive shell behavior)
(allow pseudo-tty)
(allow file-read* file-write* file-ioctl (literal "/dev/ptmx"))
(allow file-read* file-write*
  (require-all
    (regex #"^/dev/ttys[0-9]+")
    (extension "com.apple.sandbox.pty")))
(allow file-ioctl (regex #"^/dev/ttys[0-9]+"))

; read-only user preferences (writes are blocked by deny default since
; we do NOT allow `user-preference-write`)
(allow ipc-posix-shm-read* (ipc-posix-name-prefix "apple.cfprefs."))
(allow mach-lookup
  (global-name "com.apple.cfprefsd.daemon")
  (global-name "com.apple.cfprefsd.agent")
  (local-name "com.apple.cfprefsd.agent"))
(allow user-preference-read)

; ====== network rules ======
; AF_SYSTEM control sockets used by some platform helpers.
(allow system-socket
  (require-all
    (socket-domain AF_SYSTEM)
    (socket-protocol 2)))

; Outbound BSD sockets (curl, http clients, etc.)
(allow network-outbound)
(allow network-bind (local ip "*:0"))
(allow network-inbound (local ip "*:0"))

; Services needed for hostname lookup, TLS trust evaluation, proxy config.
(allow mach-lookup
  (global-name "com.apple.bsd.dirhelper")
  (global-name "com.apple.system.opendirectoryd.membership")
  (global-name "com.apple.SecurityServer")
  (global-name "com.apple.networkd")
  (global-name "com.apple.ocspd")
  (global-name "com.apple.trustd.agent")
  (global-name "com.apple.SystemConfiguration.DNSConfiguration")
  (global-name "com.apple.SystemConfiguration.configd")
  (global-name "com.apple.mDNSResponder"))

(allow sysctl-read
  (sysctl-name-regex #"^net.routetable"))
"#;

/// PowerShell prelude that switches `$OutputEncoding` and `[Console]::OutputEncoding` to UTF-8. See
/// [`wrap_command_with_utf8_output`] for why this is necessary.
#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
const POWERSHELL_UTF8_PRELUDE: &str = "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;\
     $OutputEncoding=[System.Text.Encoding]::UTF8;";

/// Prepend the UTF-8 encoding prelude to a PowerShell command. Used by both the sandboxed and
/// non-sandboxed Windows `execute_command` paths so pipe output is always decoded as UTF-8 on the
/// Rust side regardless of the console's legacy code page.
#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
pub fn wrap_command_with_utf8_output(command: &str) -> String {
    let mut wrapped = String::with_capacity(POWERSHELL_UTF8_PRELUDE.len() + command.len() + 1);
    wrapped.push_str(POWERSHELL_UTF8_PRELUDE);
    wrapped.push(' ');
    wrapped.push_str(command);
    wrapped
}

/// Quote a single command-line argument per Windows `CommandLineToArgvW` rules. Mirrors the
/// algorithm used by `std::process::Command` on Windows.
///
/// This is the correct encoding for any program that parses its command line with
/// `CommandLineToArgvW`, including `powershell.exe`, which is what the Low-integrity sandbox
/// invokes. It is **not** the correct encoding for `cmd.exe /C` (cmd treats `\` literally); don't
/// apply this to cmd command bodies.
///
/// Compiled on every platform even though the rules are Windows-specific:
/// the implementation is pure string manipulation, so unit tests run on
/// Linux/macOS without an `#[cfg(target_os = "windows")]` gate (the
/// `cfg_attr` below just silences the dead-code warning off-Windows).
#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
pub fn quote_command_arg(arg: &str) -> String {
    if !arg.is_empty()
        && !arg
            .chars()
            .any(|c| matches!(c, ' ' | '\t' | '\n' | '\u{000B}' | '"'))
    {
        return arg.to_string();
    }

    let mut quoted = String::from("\"");
    let mut pending_backslashes: usize = 0;
    for c in arg.chars() {
        match c {
            '\\' => {
                pending_backslashes += 1;
            }
            '"' => {
                // Double the run of backslashes, then emit an escaped quote.
                for _ in 0..(pending_backslashes * 2 + 1) {
                    quoted.push('\\');
                }
                pending_backslashes = 0;
                quoted.push('"');
            }
            _ => {
                for _ in 0..pending_backslashes {
                    quoted.push('\\');
                }
                pending_backslashes = 0;
                quoted.push(c);
            }
        }
    }
    // Any trailing backslashes must be doubled so the closing quote is not escaped by them.
    for _ in 0..(pending_backslashes * 2) {
        quoted.push('\\');
    }
    quoted.push('"');
    quoted
}

/// Curated env-var set for a sandboxed shell child. Applied via `Command::env_clear()` +
/// `Command::envs(...)` before spawn so it covers Bubblewrap, Landlock, Seatbelt, and the Windows
/// Low-integrity path uniformly without per-backend flag plumbing.
///
/// Read-mode sandboxes still allow outbound network (curl, dns, etc.), so a leaked secret in env
/// (`ANTHROPIC_API_KEY`, `AWS_SECRET_ACCESS_KEY`, `GITHUB_TOKEN`, …) is a live exfiltration vector
/// under prompt injection. Stripping the env at spawn time closes that gap without touching what
/// the sandbox itself enforces.
///
/// **Unix** uses an explicit allow-list (small, curated). Unknown vars are dropped; `EDITOR`,
/// `PAGER`, `BAT_THEME`, etc. don't survive into read-mode shells. Users who need a specific var
/// should switch to write mode (trusted-operation path; no scrubbing applies).
///
/// **Windows** uses a heuristic deny-list ([`is_sensitive_env_name`]) because PowerShell pulls in a
/// long tail of system vars (`PSModulePath`, `APPDATA`, `ProgramFiles`, etc.) that don't fit a tidy
/// allow-list; an allow-list version was tried first and broke core cmdlets.
pub fn sandbox_child_env() -> Vec<(std::ffi::OsString, std::ffi::OsString)> {
    std::env::vars_os()
        .filter(|(name_os, _)| match name_os.to_str() {
            Some(name) => keep_sandbox_env_var(name),
            None => false,
        })
        .collect()
}

#[cfg(unix)]
fn keep_sandbox_env_var(name: &str) -> bool {
    // Exact-match allow-list. Names that an empty-env `sh -c …` typically needs to function: `PATH`
    // so commands resolve, `HOME` for tools that read `~/.config`, locale so `grep`/`sort` don't
    // mangle non-ASCII, etc.
    const ALLOW_EXACT: &[&str] = &[
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "PWD",
        "TERM",
        "COLORTERM",
        "LANG",
        "TMPDIR",
        "TMP",
        "TEMP",
    ];
    // Prefix allow-list keeps the LC_* / XDG_* families future-proof without enumerating each var.
    // Locale (`LC_ALL`, `LC_CTYPE`, `LC_MESSAGES`, …) and XDG paths (`XDG_RUNTIME_DIR`,
    // `XDG_CONFIG_HOME`, …) are both legitimately broad.
    const ALLOW_PREFIX: &[&str] = &["LC_", "XDG_"];

    if ALLOW_EXACT.contains(&name) {
        return true;
    }
    if ALLOW_PREFIX.iter().any(|prefix| name.starts_with(prefix)) {
        return true;
    }
    // Apple frameworks (CFString, foundation, etc.) read this to pick a text encoding; dropping it
    // makes some CLIs misbehave with no useful error.
    #[cfg(target_os = "macos")]
    if name == "__CF_USER_TEXT_ENCODING" {
        return true;
    }
    false
}

#[cfg(windows)]
fn keep_sandbox_env_var(name: &str) -> bool {
    !is_sensitive_env_name(name)
}

#[cfg(not(any(unix, windows)))]
fn keep_sandbox_env_var(_name: &str) -> bool {
    // No sandbox is reachable on other platforms (SandboxCapability::Unavailable hard-errors at use
    // time), so this filter is never exercised. Pass through for completeness.
    true
}

/// Heuristic match for variable names that commonly carry credentials or point to
/// credential-bearing resources (SSH agent socket, kubeconfig, `.netrc`, GPG home, etc.).
/// Case-insensitive substring match on a list of credential-shaped markers plus prefix match on
/// known provider / service / database namespaces.
///
/// Tuned to be **aggressive on false positives** (a legitimate `GITHUB_ACTOR` is dropped alongside
/// `GITHUB_TOKEN`, `SLACK_CHANNEL` alongside `SLACK_WEBHOOK_URL`) because the downside of a
/// missing env var is a confusing tool error the user can recover from, while the downside of a
/// leaked secret is a live exfiltration channel.
///
/// Used by the Windows arm of [`sandbox_child_env`]; not consulted on Unix, where the curated
/// allow-list already drops every var by default. Lives at module scope (not inside `windows_impl`)
/// so its tests exercise both platforms in CI; the function is pure string manipulation with no
/// Windows-specific dependency.
#[cfg_attr(unix, allow(dead_code))]
pub(crate) fn is_sensitive_env_name(name: &str) -> bool {
    const SENSITIVE_SUBSTRINGS: &[&str] = &[
        // Credential-shaped name fragments.
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "PASSWD",
        "PASSPHRASE",
        "API_KEY",
        "APIKEY",
        "PRIVATE_KEY",
        "BEARER",
        "CREDENTIAL",
        "SESSION_KEY",
        "ACCESS_KEY",
        // Broader `_KEY` catches `SIGNING_KEY`, `ENCRYPTION_KEY`, `DEPLOY_KEY`, `MASTER_KEY`, etc.
        // without enumerating each.
        "_KEY",
        // Specific names that don't share a credential-shaped fragment but point to credentials,
        // sockets, or other exfil-relevant resources. Substring (not exact) match so derivatives
        // like `WSL_SSH_AUTH_SOCK` are also caught.
        "SSH_AUTH_SOCK",
        "SSH_ASKPASS",
        "GIT_ASKPASS",
        "GIT_SSH_COMMAND",
        "KUBECONFIG",
        "GNUPGHOME",
        "NETRC",
    ];
    const SENSITIVE_PREFIXES: &[&str] = &[
        // Agent / first-party.
        "ANTHROPIC_",
        "OPENAI_",
        "CLAUDE_",
        "MEKA_",
        // Major clouds.
        "AWS_",
        "GCP_",
        "GOOGLE_",
        "AZURE_",
        // Source control / CI.
        "GITHUB_",
        "GITLAB_",
        // Model hubs / AI APIs.
        "HF_",
        "HUGGINGFACE_",
        "OPENROUTER_",
        "GROQ_",
        "MISTRAL_",
        "COHERE_",
        "REPLICATE_",
        "TOGETHER_",
        "FIREWORKS_",
        // Package registries.
        "NPM_",
        "PYPI_",
        "CARGO_REGISTRY_",
        "DOCKER_",
        // Database connection strings often embed credentials.
        "DATABASE_",
        "POSTGRES_",
        "MYSQL_",
        "MONGO_",
        "REDIS_",
        // PaaS / hosting providers with API tokens.
        "STRIPE_",
        "CLOUDFLARE_",
        "HEROKU_",
        "VERCEL_",
        "NETLIFY_",
        "SUPABASE_",
        "RAILWAY_",
        // Identity / secret managers.
        "OKTA_",
        "AUTH0_",
        "VAULT_",
        "JWT_",
        "OAUTH_",
        // Observability tools with ingest keys.
        "SENTRY_",
        "DATADOG_",
        // Communication APIs with bot tokens / webhooks.
        "SLACK_",
        "DISCORD_",
    ];

    let upper = name.to_ascii_uppercase();
    SENSITIVE_SUBSTRINGS
        .iter()
        .any(|needle| upper.contains(needle))
        || SENSITIVE_PREFIXES
            .iter()
            .any(|prefix| upper.starts_with(prefix))
}

#[cfg(target_os = "windows")]
pub use windows_impl::spawn_low_integrity_command;

#[cfg(target_os = "windows")]
mod windows_impl {
    use std::{
        fs::File,
        mem,
        os::windows::{io::FromRawHandle, process::ExitStatusExt},
        process::ExitStatus,
        ptr,
    };

    use windows_sys::Win32::{
        Foundation::{
            CloseHandle, ERROR_PRIVILEGE_NOT_HELD, GENERIC_READ, HANDLE, HANDLE_FLAG_INHERIT,
            INVALID_HANDLE_VALUE, LocalFree, SetHandleInformation, TRUE, WAIT_OBJECT_0,
        },
        Security::{
            AdjustTokenPrivileges, Authorization::ConvertStringSidToSidW, DuplicateTokenEx,
            SECURITY_ATTRIBUTES, SID_AND_ATTRIBUTES, SecurityAnonymous, SetTokenInformation,
            TOKEN_ADJUST_DEFAULT, TOKEN_ADJUST_PRIVILEGES, TOKEN_ASSIGN_PRIMARY, TOKEN_DUPLICATE,
            TOKEN_MANDATORY_LABEL, TOKEN_QUERY, TokenIntegrityLevel, TokenPrimary,
        },
        Storage::FileSystem::{CreateFileW, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING},
        System::{
            JobObjects::{
                AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
                JOBOBJECT_BASIC_LIMIT_INFORMATION, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
                JobObjectExtendedLimitInformation, SetInformationJobObject, TerminateJobObject,
            },
            Pipes::CreatePipe,
            Threading::{
                CREATE_NO_WINDOW, CREATE_SUSPENDED, CREATE_UNICODE_ENVIRONMENT,
                CreateProcessAsUserW, CreateProcessWithTokenW, DeleteProcThreadAttributeList,
                EXTENDED_STARTUPINFO_PRESENT, GetCurrentProcess, GetExitCodeProcess, INFINITE,
                InitializeProcThreadAttributeList, LPPROC_THREAD_ATTRIBUTE_LIST, OpenProcessToken,
                PROC_THREAD_ATTRIBUTE_HANDLE_LIST, PROCESS_INFORMATION, ResumeThread,
                STARTF_USESTDHANDLES, STARTUPINFOEXW, STARTUPINFOW, TerminateProcess,
                UpdateProcThreadAttribute, WaitForSingleObject,
            },
        },
    };

    // SE_GROUP_INTEGRITY isn't exported by the `Win32_Security` feature in windows-sys 0.59; define
    // it locally. See
    // <https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-sid_and_attributes>.
    const SE_GROUP_INTEGRITY: u32 = 0x0000_0020;

    /// RAII wrapper for a Win32 `HANDLE`. Closes the handle on drop, unless ownership is
    /// transferred out via [`OwnedHandle::into_raw`], which invalidates the wrapper.
    /// `!Send`/`!Sync` for raw pointers is overridden here because the underlying kernel object is
    /// process-wide and thread-safe to close from any thread; we serialize usage through the owning
    /// struct.
    struct OwnedHandle(HANDLE);

    unsafe impl Send for OwnedHandle {}
    unsafe impl Sync for OwnedHandle {}

    impl OwnedHandle {
        fn as_raw(&self) -> HANDLE {
            self.0
        }

        /// Consume the wrapper and return the raw handle, suppressing the Drop-time `CloseHandle`.
        /// Use when the handle is being transferred into another owner (e.g.
        /// `File::from_raw_handle`, or into the `SandboxedChild` long-lived handles).
        fn into_raw(mut self) -> HANDLE {
            let h = self.0;
            self.0 = INVALID_HANDLE_VALUE;
            h
        }
    }

    impl Drop for OwnedHandle {
        fn drop(&mut self) {
            if !self.0.is_null() && self.0 != INVALID_HANDLE_VALUE {
                // SAFETY: We own this handle and haven't already closed it. After Drop the struct
                // is gone so no double-close is possible.
                unsafe {
                    CloseHandle(self.0);
                }
            }
        }
    }

    /// Child process spawned under a Low-integrity token. `stdout`/`stderr` are anonymous pipes
    /// wrapped in [`File`] (convertible to tokio async readers via `tokio::fs::File::from_std`).
    /// `wait_blocking` / `kill` run synchronous Win32 calls; call them from
    /// `tokio::task::spawn_blocking`.
    ///
    /// The child is wrapped in a Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, so any
    /// grandchildren spawned by the user command are atomically killed when the job handle drops
    /// (matching Unix's `setsid()` + `kill(-pgid, …)` semantics). `kill()` terminates the entire
    /// job, not just the direct child.
    pub struct SandboxedChild {
        process: OwnedHandle,
        job: OwnedHandle,
        stdout: Option<File>,
        stderr: Option<File>,
    }

    impl SandboxedChild {
        pub fn take_stdout(&mut self) -> Option<File> {
            self.stdout.take()
        }

        pub fn take_stderr(&mut self) -> Option<File> {
            self.stderr.take()
        }

        /// Block the current thread until the child exits. Must be called from a blocking context
        /// (e.g. `tokio::task::spawn_blocking`).
        pub fn wait_blocking(&self) -> std::io::Result<ExitStatus> {
            // SAFETY: `process` is a valid open process HANDLE until Drop.
            unsafe {
                let rc = WaitForSingleObject(self.process.as_raw(), INFINITE);
                if rc != WAIT_OBJECT_0 {
                    return Err(std::io::Error::last_os_error());
                }
                let mut exit_code: u32 = 0;
                if GetExitCodeProcess(self.process.as_raw(), &mut exit_code) == 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(ExitStatus::from_raw(exit_code))
            }
        }

        /// Terminate the child process and every grandchild via the Job Object. Returns success
        /// even if the job was already empty; Win32 distinguishes these but the shell tool treats
        /// both as "gone".
        pub fn kill(&self) -> std::io::Result<()> {
            // SAFETY: `job` is a valid open Job HANDLE until Drop. Terminating the job cascades to
            // every process assigned to it, including any grandchildren the user command spawned.
            unsafe {
                if TerminateJobObject(self.job.as_raw(), 1) == 0 {
                    let err = std::io::Error::last_os_error();
                    // ERROR_ACCESS_DENIED (5) is returned when the job is already gone; treat as
                    // success.
                    if err.raw_os_error() == Some(5) {
                        return Ok(());
                    }
                    return Err(err);
                }
                Ok(())
            }
        }
    }

    /// Spawn `powershell.exe -NoProfile -NonInteractive -Command <command>` under a Low-integrity
    /// token. Stdout and stderr are captured via anonymous pipes; stdin is not connected.
    ///
    /// PowerShell parses its command line per `CommandLineToArgvW` rules, so the user command is
    /// encoded with the standard argv-escape helper; embedded `"`, `\`, spaces, and shell
    /// metacharacters all pass through unmangled. `-NoProfile` skips user profile scripts (fast
    /// startup, no unrelated side effects); `-NonInteractive` makes the child fail fast on any
    /// prompt instead of hanging on stdin.
    ///
    /// Returns [`std::io::Error`] mirroring the underlying Win32 call so the shell tool can surface
    /// a standard error message.
    pub fn spawn_low_integrity_command(command: &str) -> std::io::Result<SandboxedChild> {
        // Embedded NULs would silently truncate the CreateProcess command line (Win32 treats the
        // UTF-16 command-line buffer as a C string). Agent-driven commands shouldn't contain these,
        // but fail loudly rather than silently execute a truncated prefix.
        if command.contains('\0') {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "command contains embedded NUL byte",
            ));
        }

        // Force UTF-8 output before running the user's command. PowerShell 5.1 (the inbox version
        // we invoke as `powershell.exe`) defaults `[Console]::OutputEncoding` to the system's
        // legacy OEM code page, CP 437 / 1252 on most English installs, which mangles non-ASCII
        // output (日本語 → `???`) when the process writes to a redirected pipe like ours. Prefixing
        // every script with a UTF-8 encoding switch makes output round-trip losslessly regardless
        // of the host's console configuration.
        let wrapped_command = super::wrap_command_with_utf8_output(command);

        let mut cmd_line = String::from(r#""powershell.exe" -NoProfile -NonInteractive -Command "#);
        cmd_line.push_str(&super::quote_command_arg(&wrapped_command));

        // SAFETY: All Win32 calls below are documented and we check return values. Handles are
        // wrapped in `OwnedHandle` to close on drop. Pipe handles transfer ownership into the
        // spawned child (for the write ends) or into the returned `File` (for the read ends).
        unsafe {
            // 1. Open our own process token and duplicate it as a primary token we can modify. The
            //    duplicate is what we'll drop to Low integrity; we must NOT mutate our own token.
            let mut self_token: HANDLE = ptr::null_mut();
            if OpenProcessToken(
                GetCurrentProcess(),
                TOKEN_DUPLICATE | TOKEN_QUERY | TOKEN_ASSIGN_PRIMARY | TOKEN_ADJUST_DEFAULT,
                &mut self_token,
            ) == 0
            {
                return Err(std::io::Error::last_os_error());
            }
            let self_token = OwnedHandle(self_token);

            let mut low_token: HANDLE = ptr::null_mut();
            // `SecurityAnonymous` is the least-capable impersonation level and is the correct
            // "don't care" value when the target is a primary token; per Win32 docs the parameter
            // is only consulted for impersonation tokens, but some kernel versions have
            // historically honored it, so pick the safest constant.
            if DuplicateTokenEx(
                self_token.as_raw(),
                TOKEN_ASSIGN_PRIMARY
                    | TOKEN_DUPLICATE
                    | TOKEN_QUERY
                    | TOKEN_ADJUST_DEFAULT
                    | TOKEN_ADJUST_PRIVILEGES,
                ptr::null(),
                SecurityAnonymous,
                TokenPrimary,
                &mut low_token,
            ) == 0
            {
                return Err(std::io::Error::last_os_error());
            }
            let low_token = OwnedHandle(low_token);

            // 2. Strip all privileges from the duplicate before anything else touches it.
            //    Integrity-level enforcement already makes most privileges inert against Medium+
            //    resources, but defense-in-depth: a Low-integrity token that still claims (say)
            //    `SeShutdownPrivilege` is a sharper edge than one that has none at all. Passing
            //    DisableAllPrivileges=TRUE with a NULL NewState disables every privilege on the
            //    token.
            if AdjustTokenPrivileges(
                low_token.as_raw(),
                TRUE,
                ptr::null(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            ) == 0
            {
                return Err(std::io::Error::last_os_error());
            }

            // 3. Build the Low-integrity SID via ConvertStringSidToSidW and point a
            //    TOKEN_MANDATORY_LABEL at it. The SID buffer is allocated by the OS and must be
            //    released via LocalFree.
            let sid_str: Vec<u16> = "S-1-16-4096\0".encode_utf16().collect();
            let mut low_sid: *mut core::ffi::c_void = ptr::null_mut();
            if ConvertStringSidToSidW(sid_str.as_ptr(), &mut low_sid) == 0 {
                return Err(std::io::Error::last_os_error());
            }
            let _sid_guard = LocalFreeGuard(low_sid);

            let label = TOKEN_MANDATORY_LABEL {
                Label: SID_AND_ATTRIBUTES {
                    Sid: low_sid,
                    Attributes: SE_GROUP_INTEGRITY,
                },
            };

            if SetTokenInformation(
                low_token.as_raw(),
                TokenIntegrityLevel,
                &label as *const _ as *const core::ffi::c_void,
                mem::size_of::<TOKEN_MANDATORY_LABEL>() as u32,
            ) == 0
            {
                return Err(std::io::Error::last_os_error());
            }

            // 4. Create two anonymous pipes with **non-inheritable** handles. We use
            //    `PROC_THREAD_ATTRIBUTE_HANDLE_LIST` (step 6) to narrow inheritance to exactly the
            //    three handles our child needs; the inherit flag is only flipped to TRUE briefly on
            //    those three handles, not the read ends, which eliminates the classic
            //    CreatePipe→SetHandleInformation→CreateProcess race where a concurrent
            //    CreateProcess in the same process could leak the read ends to an unrelated child.
            let sa_noninherit = SECURITY_ATTRIBUTES {
                nLength: mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
                lpSecurityDescriptor: ptr::null_mut(),
                bInheritHandle: 0,
            };

            let (stdout_read, stdout_write) = create_pipe(&sa_noninherit)?;
            let (stderr_read, stderr_write) = create_pipe(&sa_noninherit)?;

            // 5. Open NUL as the child's stdin. Non-inheritable; inherit flag flipped on just
            //    before CreateProcess.
            let nul_stdin = open_nul_read(&sa_noninherit)?;

            // 6. Promote the three child-bound handles to inheritable. The
            //    PROC_THREAD_ATTRIBUTE_HANDLE_LIST filter (step 7) requires each listed handle to
            //    have HANDLE_FLAG_INHERIT set.
            if SetHandleInformation(
                stdout_write.as_raw(),
                HANDLE_FLAG_INHERIT,
                HANDLE_FLAG_INHERIT,
            ) == 0
                || SetHandleInformation(
                    stderr_write.as_raw(),
                    HANDLE_FLAG_INHERIT,
                    HANDLE_FLAG_INHERIT,
                ) == 0
                || SetHandleInformation(
                    nul_stdin.as_raw(),
                    HANDLE_FLAG_INHERIT,
                    HANDLE_FLAG_INHERIT,
                ) == 0
            {
                return Err(std::io::Error::last_os_error());
            }

            // 7. Build a STARTUPINFOEXW with PROC_THREAD_ATTRIBUTE_HANDLE_LIST naming exactly the
            //    three handles we want the child to see. With bInheritHandles=TRUE and
            //    EXTENDED_STARTUPINFO_PRESENT, the child inherits *only* the listed handles even if
            //    other inheritable handles exist in this process.
            let child_handles: [HANDLE; 3] = [
                nul_stdin.as_raw(),
                stdout_write.as_raw(),
                stderr_write.as_raw(),
            ];
            let attr_list = ProcThreadAttributeList::new_with_handle_list(&child_handles)?;

            let mut startup: STARTUPINFOEXW = mem::zeroed();
            startup.StartupInfo.cb = mem::size_of::<STARTUPINFOEXW>() as u32;
            startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
            startup.StartupInfo.hStdInput = nul_stdin.as_raw();
            startup.StartupInfo.hStdOutput = stdout_write.as_raw();
            startup.StartupInfo.hStdError = stderr_write.as_raw();
            startup.lpAttributeList = attr_list.as_raw();

            let mut proc_info: PROCESS_INFORMATION = mem::zeroed();

            // 8. Create a Job Object with `KILL_ON_JOB_CLOSE` BEFORE spawning, so the child can be
            //    assigned to it while still suspended. When `SandboxedChild` drops, the job handle
            //    drops too; that automatic close cascades to every assigned process, eliminating
            //    grandchild leaks on normal exit, kill, or panic.
            let job = create_kill_on_close_job()?;

            // 9. Spawn SUSPENDED. We assign the child to the job before any of its code runs;
            //    otherwise the child could spawn a grandchild outside the job in the gap between
            //    create and assign. With `CREATE_SUSPENDED` set, the main thread is created
            //    suspended and we manually resume it after assignment.
            let spawn_result = create_process_low_integrity(
                low_token.as_raw(),
                &cmd_line,
                &startup,
                &mut proc_info,
            );

            // Parent no longer needs the child-side write ends or the NUL stdin handle regardless
            // of success/failure. Dropping the OwnedHandle wrappers closes them. On success,
            // closing the write ends ensures the parent's read end sees EOF when the child exits.
            // Drop before any early-return so the handles aren't leaked if later steps fail.
            drop(stdout_write);
            drop(stderr_write);
            drop(nul_stdin);
            drop(attr_list);

            spawn_result?;

            // 10. Assign the suspended child to the job, then resume.
            if AssignProcessToJobObject(job.as_raw(), proc_info.hProcess) == 0 {
                let err = std::io::Error::last_os_error();
                // Best-effort kill of the suspended child before bailing, so the orphan doesn't sit
                // around if AssignProcess failed.
                TerminateProcess(proc_info.hProcess, 1);
                CloseHandle(proc_info.hProcess);
                if !proc_info.hThread.is_null() {
                    CloseHandle(proc_info.hThread);
                }
                return Err(err);
            }

            // Resume the main thread. ResumeThread returns the previous suspend count, or u32::MAX
            // on failure.
            if ResumeThread(proc_info.hThread) == u32::MAX {
                let err = std::io::Error::last_os_error();
                // Kill via job since assignment already succeeded.
                TerminateJobObject(job.as_raw(), 1);
                CloseHandle(proc_info.hProcess);
                if !proc_info.hThread.is_null() {
                    CloseHandle(proc_info.hThread);
                }
                return Err(err);
            }

            // We don't need the main thread handle; close it immediately.
            if !proc_info.hThread.is_null() {
                CloseHandle(proc_info.hThread);
            }

            // Transfer pipe read ends into owned `File`s. `File::from_raw_handle` takes ownership
            // of the HANDLE; `OwnedHandle::into_raw` suppresses the wrapper's Drop.
            let stdout_handle = stdout_read.into_raw();
            let stderr_handle = stderr_read.into_raw();

            Ok(SandboxedChild {
                process: OwnedHandle(proc_info.hProcess),
                job,
                stdout: Some(File::from_raw_handle(stdout_handle as _)),
                stderr: Some(File::from_raw_handle(stderr_handle as _)),
            })
        }
    }

    /// Create an empty Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` set. Any process later
    /// assigned to the job is killed when the job's last handle closes, the Windows analogue to
    /// Unix process groups teardown via `kill(-pgid, SIGKILL)`. Grandchildren inherit job
    /// membership automatically.
    unsafe fn create_kill_on_close_job() -> std::io::Result<OwnedHandle> {
        // SAFETY: CreateJobObjectW with null name and null SECURITY_ATTRIBUTES returns an unnamed
        // Job HANDLE the current process owns.
        let job = unsafe { CreateJobObjectW(ptr::null(), ptr::null()) };
        if job.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        let job = OwnedHandle(job);

        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { mem::zeroed() };
        info.BasicLimitInformation = JOBOBJECT_BASIC_LIMIT_INFORMATION {
            LimitFlags: JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            ..unsafe { mem::zeroed() }
        };

        if unsafe {
            SetInformationJobObject(
                job.as_raw(),
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const core::ffi::c_void,
                mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error());
        }

        Ok(job)
    }

    /// Create an anonymous pipe using the supplied SECURITY_ATTRIBUTES.
    ///
    /// A 1 MiB buffer hint is passed to `CreatePipe`. This is belt-and-
    /// braces with the concurrent draining in `run_windows_low_integrity`:
    /// even if the drain task is momentarily starved, the child has a MiB
    /// of slack before it blocks in `WriteFile`.
    unsafe fn create_pipe(sa: &SECURITY_ATTRIBUTES) -> std::io::Result<(OwnedHandle, OwnedHandle)> {
        const PIPE_BUFFER_SIZE: u32 = 1 << 20;
        let mut read: HANDLE = ptr::null_mut();
        let mut write: HANDLE = ptr::null_mut();
        // SAFETY: CreatePipe writes two HANDLEs through the provided pointers on success.
        // SECURITY_ATTRIBUTES is a valid initialized struct.
        if unsafe { CreatePipe(&mut read, &mut write, sa, PIPE_BUFFER_SIZE) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok((OwnedHandle(read), OwnedHandle(write)))
    }

    /// Open the `NUL` device for read. Inherit flag is left unset by the caller's
    /// `SECURITY_ATTRIBUTES`; promote via `SetHandleInformation` right before the handle is passed
    /// to `CreateProcess`. The child sees immediate EOF on any read, the correct "no stdin"
    /// primitive on Windows, equivalent to `/dev/null` on Unix.
    unsafe fn open_nul_read(sa: &SECURITY_ATTRIBUTES) -> std::io::Result<OwnedHandle> {
        let path: Vec<u16> = "NUL\0".encode_utf16().collect();
        // SAFETY: `path` is NUL-terminated; `sa` is a valid initialized SECURITY_ATTRIBUTES owned
        // by the caller for the duration of the call.
        let h = unsafe {
            CreateFileW(
                path.as_ptr(),
                GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                sa as *const SECURITY_ATTRIBUTES,
                OPEN_EXISTING,
                0,
                ptr::null_mut(),
            )
        };
        if h == INVALID_HANDLE_VALUE {
            return Err(std::io::Error::last_os_error());
        }
        Ok(OwnedHandle(h))
    }

    /// Create a process under the Low-integrity token. Tries `CreateProcessAsUserW` first (the
    /// usual path); on `ERROR_PRIVILEGE_NOT_HELD`, which happens when the current user lacks
    /// `SE_INCREASE_QUOTA_NAME` (common on locked-down corp-managed accounts), falls back to
    /// `CreateProcessWithTokenW`, which requires the more broadly-granted `SE_IMPERSONATE_NAME`
    /// instead.
    ///
    /// The command line is re-encoded to UTF-16 for *each* attempt: Win32 documents `lpCommandLine`
    /// as in/out, and the first call may mutate the buffer (typically inserting a NUL to split
    /// argv[0]) before failing, so re-using the same buffer between attempts could hand the
    /// fallback a corrupted string.
    ///
    /// Both invocations pass `EXTENDED_STARTUPINFO_PRESENT` together with `STARTUPINFOEXW`, so the
    /// handle-list filter in the attribute list applies uniformly across both paths.
    unsafe fn create_process_low_integrity(
        token: HANDLE,
        cmd_line_utf8: &str,
        startup: &STARTUPINFOEXW,
        proc_info: &mut PROCESS_INFORMATION,
    ) -> std::io::Result<()> {
        // CREATE_SUSPENDED so the child sits at its entry point until we've assigned it to the Job
        // Object; otherwise the child could spawn a grandchild before assignment, and that
        // grandchild would never be bound to the job.
        let creation_flags = CREATE_NO_WINDOW
            | EXTENDED_STARTUPINFO_PRESENT
            | CREATE_UNICODE_ENVIRONMENT
            | CREATE_SUSPENDED;
        let startup_ptr = startup as *const STARTUPINFOEXW as *const STARTUPINFOW;

        // Build a scrubbed UTF-16 environment block once. Passing this for both
        // `CreateProcessAsUserW` and the `CreateProcessWithTokenW` fallback ensures the sandboxed
        // child never sees the agent's `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or any `*_TOKEN` /
        // `*_SECRET` variable; a Low-integrity child can still open outbound sockets, so a leaked
        // key in env is a live exfil vector.
        let mut env_block = build_scrubbed_env_block_utf16();
        let env_ptr = env_block.as_mut_ptr() as *const core::ffi::c_void;

        let mut cmd_line_utf16: Vec<u16> = cmd_line_utf8
            .encode_utf16()
            .chain(std::iter::once(0u16))
            .collect();

        // SAFETY: All pointers are valid for the duration of the call per the caller's obligations.
        // Win32 writes to `proc_info` on success.
        let ok = unsafe {
            CreateProcessAsUserW(
                token,
                ptr::null(),
                cmd_line_utf16.as_mut_ptr(),
                ptr::null(),
                ptr::null(),
                TRUE,
                creation_flags,
                env_ptr,
                ptr::null(),
                startup_ptr,
                proc_info,
            )
        };
        if ok != 0 {
            return Ok(());
        }

        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != Some(ERROR_PRIVILEGE_NOT_HELD as i32) {
            return Err(err);
        }

        tracing::warn!(
            "CreateProcessAsUserW denied (SE_INCREASE_QUOTA_NAME not held); \
             falling back to CreateProcessWithTokenW. The child is still spawned \
             under the Low-integrity token with the same scrubbed environment \
             block; semantics differ slightly (no custom process/thread \
             SECURITY_ATTRIBUTES)."
        );

        // Rebuild the command-line buffer; the previous call may have mutated it before failing
        // (Win32 documents lpCommandLine as in/out).
        let mut cmd_line_utf16_retry: Vec<u16> = cmd_line_utf8
            .encode_utf16()
            .chain(std::iter::once(0u16))
            .collect();

        // SAFETY: same contract as CreateProcessAsUserW; the two APIs only differ in their
        // parameter list (no process/thread security attrs, no bInheritHandles; inheritance is
        // driven by the per-handle `HANDLE_FLAG_INHERIT` flag plus the attribute-list filter). We
        // re-use the scrubbed environment block so the fallback path doesn't accidentally regress
        // to inheriting the parent's env.
        let ok = unsafe {
            CreateProcessWithTokenW(
                token,
                0, // dwLogonFlags: 0 means "use the token as-is"
                ptr::null(),
                cmd_line_utf16_retry.as_mut_ptr(),
                creation_flags,
                env_ptr,
                ptr::null(),
                startup_ptr,
                proc_info,
            )
        };
        // Keep `env_block` alive until after both calls complete; Win32 copies the contents but
        // documents `lpEnvironment` as a pointer that must be valid through the call.
        drop(env_block);
        if ok == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    /// Build a UTF-16 `NAME=VALUE\0NAME=VALUE\0\0` environment block for the sandboxed child.
    /// Delegates to [`super::sandbox_child_env`] for the filter (Windows uses the deny-list arm) so
    /// the Low-integrity spawn path stays in sync with the Unix sandbox paths.
    fn build_scrubbed_env_block_utf16() -> Vec<u16> {
        let mut block: Vec<u16> = Vec::new();
        for (name_os, value_os) in super::sandbox_child_env() {
            let Some(name) = name_os.to_str() else {
                continue;
            };
            let Some(value) = value_os.to_str() else {
                continue;
            };
            append_env_entry(&mut block, name, value);
        }
        // Double-NUL terminator (each entry already ends with one NUL; we need another to close the
        // block).
        block.push(0);
        block
    }

    fn append_env_entry(block: &mut Vec<u16>, name: &str, value: &str) {
        block.extend(name.encode_utf16());
        block.push(u16::from(b'='));
        block.extend(value.encode_utf16());
        block.push(0);
    }

    /// RAII wrapper around `PROC_THREAD_ATTRIBUTE_LIST`. Owns both the attribute-list backing
    /// buffer and the HANDLE array it points into; Win32 stores the handle-list address (not a
    /// copy), so the array must outlive any `CreateProcess*` call that consumes the attribute list.
    struct ProcThreadAttributeList {
        // Both fields are kept alive for Drop. The Vec's heap buffer is the attribute-list
        // storage; `list_ptr` caches a stable mutable pointer to it. The boxed handle
        // slice is referenced (by pointer) from inside the attribute-list buffer, so it
        // must not move or drop while the list is alive.
        _buffer: Vec<u8>,
        _handles: Box<[HANDLE]>,
        list_ptr: LPPROC_THREAD_ATTRIBUTE_LIST,
    }

    impl ProcThreadAttributeList {
        /// Build a one-attribute list containing a `HANDLE_LIST` attribute referencing the supplied
        /// handles. `UpdateProcThreadAttribute` stores the pointer to the handle array, not a copy;
        /// the array is boxed into the returned wrapper so it stays at a fixed address for the
        /// wrapper's lifetime.
        unsafe fn new_with_handle_list(handles: &[HANDLE]) -> std::io::Result<Self> {
            // First call: buffer=NULL, size=0 → fails with ERROR_INSUFFICIENT_BUFFER but writes the
            // required size.
            let mut size: usize = 0;
            unsafe {
                InitializeProcThreadAttributeList(ptr::null_mut(), 1, 0, &mut size);
            }
            if size == 0 {
                return Err(std::io::Error::last_os_error());
            }

            let mut buffer: Vec<u8> = vec![0; size];
            let list_ptr = buffer.as_mut_ptr() as LPPROC_THREAD_ATTRIBUTE_LIST;

            // SAFETY: `list_ptr` points to a correctly-sized buffer from the previous size query,
            // and `size` is that queried value.
            if unsafe { InitializeProcThreadAttributeList(list_ptr, 1, 0, &mut size) } == 0 {
                return Err(std::io::Error::last_os_error());
            }

            let boxed_handles: Box<[HANDLE]> = handles.to_vec().into_boxed_slice();
            let handles_bytes = std::mem::size_of_val(&*boxed_handles);

            // SAFETY: `list_ptr` was just initialized; `boxed_handles` lives for 'self because it's
            // stored in the returned wrapper; the byte size passed matches the boxed array.
            if unsafe {
                UpdateProcThreadAttribute(
                    list_ptr,
                    0,
                    PROC_THREAD_ATTRIBUTE_HANDLE_LIST as usize,
                    boxed_handles.as_ptr() as *const core::ffi::c_void,
                    handles_bytes,
                    ptr::null_mut(),
                    ptr::null(),
                )
            } == 0
            {
                let err = std::io::Error::last_os_error();
                // SAFETY: Initialize succeeded; must be paired with Delete regardless of subsequent
                // failures.
                unsafe { DeleteProcThreadAttributeList(list_ptr) };
                return Err(err);
            }

            Ok(Self {
                _buffer: buffer,
                _handles: boxed_handles,
                list_ptr,
            })
        }

        fn as_raw(&self) -> LPPROC_THREAD_ATTRIBUTE_LIST {
            self.list_ptr
        }
    }

    impl Drop for ProcThreadAttributeList {
        fn drop(&mut self) {
            // SAFETY: constructor either fully initialized the list (and stored its pointer in
            // `list_ptr`) or returned Err (in which case this Drop doesn't run).
            unsafe {
                DeleteProcThreadAttributeList(self.list_ptr);
            }
        }
    }

    /// RAII guard for a pointer allocated by the OS and freed via `LocalFree`.
    struct LocalFreeGuard(*mut core::ffi::c_void);

    impl Drop for LocalFreeGuard {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: pointer was returned by a Win32 API that documents `LocalFree` as the
                // correct release call.
                unsafe {
                    LocalFree(self.0);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_sensitive_env_name_matches_known_secret_patterns() {
        // Provider API keys.
        assert!(is_sensitive_env_name("ANTHROPIC_API_KEY"));
        assert!(is_sensitive_env_name("OPENAI_API_KEY"));
        assert!(is_sensitive_env_name("anthropic_api_key"));
        // VCS / CI tokens.
        assert!(is_sensitive_env_name("GITHUB_TOKEN"));
        assert!(is_sensitive_env_name("GITLAB_PRIVATE_TOKEN"));
        // Cloud secrets.
        assert!(is_sensitive_env_name("AWS_SECRET_ACCESS_KEY"));
        assert!(is_sensitive_env_name("AWS_SESSION_TOKEN"));
        // Credential-shaped fragments.
        assert!(is_sensitive_env_name("my_bearer_auth"));
        assert!(is_sensitive_env_name("DATABASE_PASSWORD"));
        assert!(is_sensitive_env_name("GPG_PASSPHRASE"));
        // `_KEY` catches non-API-key creds that don't match the older
        // `API_KEY`/`PRIVATE_KEY`/`SESSION_KEY`/`ACCESS_KEY` patterns.
        assert!(is_sensitive_env_name("SIGNING_KEY"));
        assert!(is_sensitive_env_name("ENCRYPTION_KEY"));
        assert!(is_sensitive_env_name("DEPLOY_KEY"));
    }

    #[test]
    fn test_is_sensitive_env_name_catches_pointer_vars() {
        // Specific named variables that point to credentials, agent sockets, or other
        // exfil-relevant resources, caught by substring even when wrapped in a longer name.
        assert!(is_sensitive_env_name("SSH_AUTH_SOCK"));
        assert!(is_sensitive_env_name("WSL_SSH_AUTH_SOCK"));
        assert!(is_sensitive_env_name("KUBECONFIG"));
        assert!(is_sensitive_env_name("GNUPGHOME"));
        assert!(is_sensitive_env_name("NETRC"));
        assert!(is_sensitive_env_name("CURLOPT_NETRC"));
        assert!(is_sensitive_env_name("GIT_SSH_COMMAND"));
        assert!(is_sensitive_env_name("GIT_ASKPASS"));
        assert!(is_sensitive_env_name("SSH_ASKPASS"));
    }

    #[test]
    fn test_is_sensitive_env_name_catches_service_prefixes() {
        // AI provider namespaces beyond the original list.
        assert!(is_sensitive_env_name("OPENROUTER_API_KEY"));
        assert!(is_sensitive_env_name("GROQ_API_KEY"));
        assert!(is_sensitive_env_name("MISTRAL_API_KEY"));
        assert!(is_sensitive_env_name("COHERE_API_KEY"));
        // Database connection strings: DATABASE_URL embeds the password.
        assert!(is_sensitive_env_name("DATABASE_URL"));
        assert!(is_sensitive_env_name("POSTGRES_HOST"));
        assert!(is_sensitive_env_name("MONGO_URI"));
        assert!(is_sensitive_env_name("REDIS_PASSWORD"));
        // PaaS / hosting providers.
        assert!(is_sensitive_env_name("STRIPE_SECRET_KEY"));
        assert!(is_sensitive_env_name("CLOUDFLARE_API_TOKEN"));
        assert!(is_sensitive_env_name("VERCEL_TOKEN"));
        assert!(is_sensitive_env_name("SUPABASE_KEY"));
        // Identity / secret managers.
        assert!(is_sensitive_env_name("VAULT_TOKEN"));
        assert!(is_sensitive_env_name("OKTA_CLIENT_SECRET"));
        assert!(is_sensitive_env_name("AUTH0_CLIENT_ID"));
        // Generic auth tokens.
        assert!(is_sensitive_env_name("JWT_SECRET"));
        assert!(is_sensitive_env_name("OAUTH_CLIENT_SECRET"));
        // Observability and communications.
        assert!(is_sensitive_env_name("SENTRY_DSN"));
        assert!(is_sensitive_env_name("DATADOG_API_KEY"));
        assert!(is_sensitive_env_name("SLACK_WEBHOOK_URL"));
        assert!(is_sensitive_env_name("DISCORD_BOT_TOKEN"));
    }

    #[test]
    fn test_is_sensitive_env_name_allows_system_vars() {
        // Windows system vars PowerShell needs at startup must NOT be flagged sensitive; that's
        // the whole reason Windows uses deny-list instead of allow-list.
        assert!(!is_sensitive_env_name("SystemRoot"));
        assert!(!is_sensitive_env_name("PATH"));
        assert!(!is_sensitive_env_name("PSModulePath"));
        assert!(!is_sensitive_env_name("APPDATA"));
        assert!(!is_sensitive_env_name("LOCALAPPDATA"));
        assert!(!is_sensitive_env_name("ProgramFiles"));
        assert!(!is_sensitive_env_name("USERPROFILE"));
        assert!(!is_sensitive_env_name("TEMP"));
        // Unix basics also shouldn't flag (the function is used on Windows but compiles
        // cross-platform for testability).
        assert!(!is_sensitive_env_name("HOME"));
        assert!(!is_sensitive_env_name("USER"));
        assert!(!is_sensitive_env_name("LANG"));
        assert!(!is_sensitive_env_name("TERM"));
        // `KEYBOARD_LAYOUT` doesn't have `_KEY` as a substring (the pattern requires an underscore
        // before KEY), so it survives.
        assert!(!is_sensitive_env_name("KEYBOARD_LAYOUT"));
    }

    /// `cargo test` always runs with `PATH` set (the test binary needs it to invoke itself), so
    /// this is a no-mutation sanity check that the filter doesn't accidentally strip it. Windows
    /// env-var names are case-insensitive and typically stored as `Path`, so the match is
    /// case-insensitive.
    #[test]
    fn test_sandbox_child_env_keeps_path() {
        let env = sandbox_child_env();
        assert!(
            env.iter()
                .any(|(name, _)| name.to_string_lossy().eq_ignore_ascii_case("PATH")),
            "expected PATH to survive the sandbox env filter"
        );
    }

    /// Token-shaped sentinel: dropped by the Unix allow-list (not in the curated list) AND by the
    /// Windows deny-list (`TOKEN` substring match in `is_sensitive_env_name`). Verifies both arms
    /// strip it.
    #[test]
    fn test_sandbox_child_env_drops_token_sentinel() {
        const NAME: &str = "MEKA_TEST_SCRUB_TOKEN_PROBE";
        // SAFETY: `set_var`/`remove_var` are process-global and `cargo test` runs in-process tests
        // in parallel. The variable name is long and test-specific so it can't collide with another
        // test or the real environment.
        unsafe {
            std::env::set_var(NAME, "sentinel-should-be-dropped");
        }
        let env = sandbox_child_env();
        let leaked = env.iter().any(|(name, _)| name.to_string_lossy() == NAME);
        unsafe {
            std::env::remove_var(NAME);
        }
        assert!(
            !leaked,
            "token-shaped sentinel leaked through the sandbox env filter"
        );
    }

    /// Unix: any var not in the curated allow-list (and not matching `LC_*`/`XDG_*`) is dropped.
    /// The sentinel name has no special shape: pure "unknown var" test.
    #[cfg(unix)]
    #[test]
    fn test_sandbox_child_env_drops_unknown_var() {
        const NAME: &str = "MEKA_TEST_SCRUB_UNKNOWN_PROBE";
        unsafe {
            std::env::set_var(NAME, "should-be-dropped");
        }
        let env = sandbox_child_env();
        let leaked = env.iter().any(|(name, _)| name.to_string_lossy() == NAME);
        unsafe {
            std::env::remove_var(NAME);
        }
        assert!(!leaked, "unknown var leaked through the Unix allow-list");
    }

    /// Unix: `LC_*` prefix match keeps the locale family without enumerating each variant.
    #[cfg(unix)]
    #[test]
    fn test_sandbox_child_env_keeps_lc_prefix() {
        const NAME: &str = "LC_MEKA_TEST_PROBE";
        unsafe {
            std::env::set_var(NAME, "en_US.UTF-8");
        }
        let env = sandbox_child_env();
        let kept = env.iter().any(|(name, _)| name.to_string_lossy() == NAME);
        unsafe {
            std::env::remove_var(NAME);
        }
        assert!(kept, "LC_* prefix var was dropped from sandbox env");
    }

    /// Unix: `XDG_*` prefix match keeps the XDG basedir family without enumerating each variant.
    #[cfg(unix)]
    #[test]
    fn test_sandbox_child_env_keeps_xdg_prefix() {
        const NAME: &str = "XDG_MEKA_TEST_PROBE";
        unsafe {
            std::env::set_var(NAME, "/tmp/meka-probe");
        }
        let env = sandbox_child_env();
        let kept = env.iter().any(|(name, _)| name.to_string_lossy() == NAME);
        unsafe {
            std::env::remove_var(NAME);
        }
        assert!(kept, "XDG_* prefix var was dropped from sandbox env");
    }

    #[test]
    fn test_detect_sandbox_capability() {
        let capability = detect();
        // Should detect something on Linux/macOS/Windows, Unavailable on others
        match capability {
            #[cfg(target_os = "linux")]
            SandboxCapability::Landlock { abi_version } => {
                assert!(abi_version >= 1);
            }
            #[cfg(target_os = "macos")]
            SandboxCapability::SandboxExec => {}
            #[cfg(target_os = "windows")]
            SandboxCapability::LowIntegrity => {}
            SandboxCapability::Unavailable => {}
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }

    #[test]
    fn test_wrap_command_with_utf8_output_prepends_prelude() {
        let wrapped = wrap_command_with_utf8_output("Write-Output '日本語'");
        assert!(wrapped.starts_with("[Console]::OutputEncoding="));
        assert!(wrapped.contains("$OutputEncoding=[System.Text.Encoding]::UTF8"));
        assert!(wrapped.ends_with("Write-Output '日本語'"));
        // A space must separate the prelude from the user command so PowerShell doesn't glue them
        // into one malformed statement.
        assert!(wrapped.contains("UTF8; Write-Output"));
    }

    /// Reference table covering the corners of the `CommandLineToArgvW` encoding. Cross-platform:
    /// `quote_command_arg` is pure string manipulation and has no Windows-specific runtime
    /// dependency.
    #[test]
    fn test_quote_command_arg_reference_table() {
        let cases: &[(&str, &str)] = &[
            ("cmd.exe", "cmd.exe"),
            ("", r#""""#),
            ("with space", r#""with space""#),
            ("with\ttab", "\"with\ttab\""),
            (r#"say "hi""#, r#""say \"hi\"""#),
            (r#"a\"b"#, r#""a\\\"b""#),
            (r"path with space\", r#""path with space\\""#),
            // A quote preceded by a single backslash: the backslash is doubled and the quote is
            // escaped.
            (r#"\""#, r#""\\\"""#),
            // Backslashes not adjacent to a quote pass through literally (no escaping needed, no
            // quoting needed, no special chars).
            (r"a\\b", r"a\\b"),
            // Unicode and newlines pass through. Newline counts as whitespace so the argument gets
            // quoted.
            ("日本語", "日本語"),
            ("hello world\n", "\"hello world\n\""),
        ];
        for (input, expected) in cases {
            assert_eq!(
                &quote_command_arg(input),
                expected,
                "input {:?} produced wrong quoting",
                input
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_handled_access_abi_v1() {
        let access = handled_access_for_abi(1);
        assert!(access & LANDLOCK_ACCESS_FS_WRITE_FILE != 0);
        assert!(access & LANDLOCK_ACCESS_FS_READ_FILE != 0);
        assert!(access & LANDLOCK_ACCESS_FS_REFER == 0);
        assert!(access & LANDLOCK_ACCESS_FS_TRUNCATE == 0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_handled_access_abi_v3() {
        let access = handled_access_for_abi(3);
        assert!(access & LANDLOCK_ACCESS_FS_REFER != 0);
        assert!(access & LANDLOCK_ACCESS_FS_TRUNCATE != 0);
        assert!(access & LANDLOCK_ACCESS_FS_IOCTL_DEV == 0);
    }

    /// The root rule grants only execute, read-file and read-dir, so handling `RESOLVE_UNIX` is
    /// what denies it: a sandboxed command cannot ask a daemon over the D-Bus or systemd socket to
    /// write on its behalf. Taking the bit below v9 would make `landlock_create_ruleset` fail with
    /// `EINVAL` and leave the process unconfined.
    #[cfg(target_os = "linux")]
    #[test]
    fn test_handled_access_takes_resolve_unix_only_from_abi_v9() {
        assert!(handled_access_for_abi(8) & LANDLOCK_ACCESS_FS_RESOLVE_UNIX == 0);
        assert!(handled_access_for_abi(9) & LANDLOCK_ACCESS_FS_RESOLVE_UNIX != 0);
    }

    #[test]
    fn test_backend_unavailable_reason_maps_each_variant() {
        assert!(
            backend_unavailable_reason(&BackendProbe::Ok(SandboxCapability::Unavailable)).is_none()
        );
        let reason = backend_unavailable_reason(&BackendProbe::Missing {
            reason: "bwrap not found on PATH".to_string(),
        });
        assert_eq!(reason.as_deref(), Some("bwrap not found on PATH"));
        let reason = backend_unavailable_reason(&BackendProbe::UserNamespaceDenied {
            stderr: "bwrap: setting up uid map: Permission denied\n".to_string(),
        });
        assert!(
            reason
                .as_deref()
                .unwrap_or("")
                .contains("user namespaces are denied")
        );
        assert!(
            backend_unavailable_reason(&BackendProbe::UnsupportedPlatform)
                .as_deref()
                .unwrap_or("")
                .contains("not supported")
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_probe_backend_landlock_returns_known_variant() {
        // Smoke test: confirms the probe runs without panicking on whatever kernel this build host
        // has. We can't assert which specific variant comes back because CI may have an older
        // kernel where Landlock is unavailable.
        let probe = probe_backend(crate::config::SandboxBackend::Landlock);
        assert!(matches!(
            probe,
            BackendProbe::Ok(_) | BackendProbe::Missing { .. }
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore]
    fn test_probe_backend_bubblewrap_via_path_when_available() {
        // Opt-in: only runs when explicitly requested via `--ignored`. Skipped if `bwrap` isn't on
        // `$PATH` since the probe will report `Missing { reason: "bwrap not found on PATH" }` which
        // would fail the assertion below.
        if bwrap_on_path().is_none() {
            eprintln!("skipping: bwrap not on PATH");
            return;
        }
        let probe = probe_backend(crate::config::SandboxBackend::Bubblewrap);
        match probe {
            BackendProbe::Ok(SandboxCapability::Bubblewrap { bwrap_path }) => {
                assert!(bwrap_path.is_absolute());
            }
            BackendProbe::UserNamespaceDenied { .. } => {
                eprintln!("skipping: host doesn't support user namespaces");
            }
            other => panic!("unexpected probe result: {:?}", other),
        }
    }
}
