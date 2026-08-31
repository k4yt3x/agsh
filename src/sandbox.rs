//! Filesystem sandboxing for read-only command execution.
//!
//! On Linux there are two backends. Bubblewrap (`bwrap --ro-bind /` plus tmpfs masks) is preferred
//! whenever it is installed and its user-namespace smoke test passes; Landlock LSM is the fallback,
//! and requires ABI v3 (kernel 6.2+) because `truncate(2)` is unmediated below it (see
//! `MIN_LANDLOCK_ABI`, which is Linux-only and so deliberately not an intra-doc link: the link
//! would be unresolvable on the two targets where the item does not exist, and CI gates rustdoc on
//! all three). On macOS, uses `sandbox-exec`. On Windows there are two mechanisms rather than one:
//! `read` spawns the child with a duplicated primary token dropped to Low integrity via
//! `SetTokenInformation(TokenIntegrityLevel, …)`, which blocks writes to anything outside the
//! documented Low-integrity surface, while `workspace` uses a `WRITE_RESTRICTED` token plus a
//! per-root capability ACE and deliberately leaves the integrity label alone. Where no backend is
//! usable, sandboxing is unavailable and read-mode shell execution hard-errors rather than running
//! unconfined.
//!
//! **What every backend does not restrict**: reads. A sandboxed child can read any file the user
//! can, including credential files, and the network is deliberately left open on all of them. The
//! boundary these enforce is "this command cannot change the machine", not "this command cannot see
//! or send anything".

/// What a single `execute_command` call runs under.
///
/// Three states rather than the boolean this replaced, because `workspace` is neither of the two
/// the boolean could express: it is not unconfined, and it is not read-only. Folding it into either
/// one fails in a direction that matters. Treated as unconfined, the shell ignores the boundary the
/// file tools enforce; treated as read-only, the mode promises writes it never delivers.
///
/// A boolean also hid the asymmetry that made this dangerous: the *absence* of confinement is the
/// permissive state, so any code path that forgets a case fails open. An enum makes each case one
/// the compiler asks about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Confinement {
    /// No sandbox. `ask` or `unrestricted`, or sandboxing turned off in config.
    Unconfined,
    /// Reads everywhere, writes nowhere. `none` and `read`.
    ReadOnly,
    /// Reads everywhere, writes only beneath these canonical roots. `workspace`.
    ///
    /// An empty list is meaningful and behaves as [`Self::ReadOnly`]: it means no root resolved,
    /// which happens when the working directory was deleted under a running session.
    Workspace(Vec<std::path::PathBuf>),
}

impl Confinement {
    /// What this call runs under, given the config switch, the level and the live scope.
    ///
    /// `permission` is passed in rather than read back off `scope`, even though the scope holds a
    /// handle that would answer. The caller has already read the level once at its enforcement
    /// site, and re-reading it here would make the decision depend on two handles that are only
    /// the same one by convention: a tool wired with a scope built from a different handle would
    /// sandbox against one level while denying against another, and nothing would say so.
    pub fn resolve(
        sandbox_enabled: bool,
        permission: crate::permission::Permission,
        scope: &crate::workspace::WriteScope,
        cwd: &crate::workspace::SharedCwd,
    ) -> Self {
        if !sandbox_enabled {
            return Self::Unconfined;
        }
        match permission {
            // `ask` sits with `unrestricted` rather than below it, because at `ask` the gate is the
            // prompt and the prompt has already shown the user the command. Confining an approved
            // command read-only meant approving `foo > bar` and watching it fail for a reason the
            // user did not choose, while `write_file` approved in the same breath wrote anywhere.
            // It also inverted the ladder it sits in: `ask` outranks `workspace` on the grounds
            // that an approved call reaches further, yet its shell reached less far than
            // `workspace`'s. Both are now true at once.
            crate::permission::Permission::Ask | crate::permission::Permission::Unrestricted => {
                Self::Unconfined
            }
            crate::permission::Permission::Workspace => {
                Self::Workspace(scope.confined_to(cwd).unwrap_or_default())
            }
            _ => Self::ReadOnly,
        }
    }

    /// Whether a sandbox is applied at all, i.e. anything other than [`Self::Unconfined`].
    pub fn is_sandboxed(&self) -> bool {
        !matches!(self, Self::Unconfined)
    }

    /// The roots this call may write beneath. Empty for every state but [`Self::Workspace`].
    pub fn writable(&self) -> &[std::path::PathBuf] {
        match self {
            Self::Workspace(roots) => roots,
            _ => &[],
        }
    }
}

/// Hand back any standing OS-level grant this process placed, before an exit that will not unwind.
///
/// A no-op on every platform but Windows. Landlock, bubblewrap and seatbelt confine a child process
/// and die with it, leaving nothing behind to clean up. The Windows boundary is instead an ACE
/// written onto the user's own directories, which outlives the process unless something takes it
/// back, so it needs a counterpart the others do not.
///
/// Call this immediately before each [`std::process::exit`], which skips destructors: see the call
/// sites in `main.rs` and `server.rs`. A crash or `SIGKILL` still leaves the ACE standing, which
/// `windows_impl::WindowsGrants` documents. Unlinked deliberately: that module is
/// `#[cfg(windows)]`, so an intra-doc link to it fails the rustdoc gate everywhere else.
pub fn release_process_grants() {
    #[cfg(windows)]
    windows_impl::process_grants().revoke_all();
}

/// End-to-end proof that the kernel honours the workspace boundary, not just that meka computed it.
///
/// Spawns a real `sh` under a real Landlock ruleset and checks what actually lands on disk. Skips
/// itself when the host has no usable Landlock, since CI runs this matrix on macOS and Windows too.
#[cfg(all(test, target_os = "linux"))]
mod landlock_boundary {
    use std::{ffi::CString, os::unix::process::CommandExt};

    #[test]
    fn a_confined_shell_writes_inside_the_root_and_is_refused_outside() {
        let Some(abi) = super::landlock_abi().filter(|abi| *abi >= super::MIN_LANDLOCK_ABI) else {
            eprintln!("skipping: no usable Landlock on this host");
            return;
        };

        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let work = base.join("work");
        let outside = base.join("outside");
        std::fs::create_dir(&work).expect("work");
        std::fs::create_dir(&outside).expect("outside");

        // `OsStrExt::as_bytes`, matching the production path in `shell.rs`. `as_encoded_bytes` is
        // documented as an unspecified encoding and explicitly not for FFI, so a test using it
        // would be exercising a different conversion than the one that ships.
        let writable = vec![
            CString::new(std::os::unix::ffi::OsStrExt::as_bytes(work.as_os_str()))
                .expect("cstring"),
        ];
        // The exit code carries the result, so `status.success()` is an assertion rather than a
        // formality. The script must not end in `; true`, which makes "the confined shell itself
        // must run" pass whatever happened inside it: a ruleset denying every write and one
        // allowing every write produce the same success. Only the two `exists()` checks below were
        // doing any work.
        let script = format!(
            "echo in > {}/inside.txt 2>/dev/null || exit 3\n\
             if echo out > {}/escaped.txt 2>/dev/null; then exit 4; fi\n\
             exit 0",
            work.display(),
            outside.display()
        );

        let mut command = std::process::Command::new("/bin/sh");
        command.arg("-c").arg(&script);
        unsafe {
            command.pre_exec(move || {
                super::apply_landlock(abi, &writable).map_err(std::io::Error::from_raw_os_error)
            });
        }
        let status = command.status().expect("spawn");
        match status.code() {
            Some(0) => {}
            Some(3) => panic!("the write inside the workspace root was refused"),
            Some(4) => panic!("the write outside every root was permitted"),
            other => panic!("the confined shell did not run: exit {other:?}"),
        }

        assert!(
            work.join("inside.txt").exists(),
            "a write inside the workspace root must land"
        );
        assert!(
            !outside.join("escaped.txt").exists(),
            "a write outside every root must be refused by the kernel, not merely by meka"
        );
    }

    /// With no writable root, the same ruleset refuses both. This is the read-only case, and it
    /// proves the grant above came from the root rather than from Landlock not being applied.
    #[test]
    fn an_unwritable_confinement_refuses_even_the_workspace() {
        let Some(abi) = super::landlock_abi().filter(|abi| *abi >= super::MIN_LANDLOCK_ABI) else {
            eprintln!("skipping: no usable Landlock on this host");
            return;
        };

        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let script = format!("echo x > {}/nope.txt 2>/dev/null; true", base.display());

        let mut command = std::process::Command::new("/bin/sh");
        command.arg("-c").arg(&script);
        unsafe {
            command.pre_exec(move || {
                super::apply_landlock(abi, &[]).map_err(std::io::Error::from_raw_os_error)
            });
        }
        command.status().expect("spawn");

        assert!(
            !base.join("nope.txt").exists(),
            "read-only means read-only: granting nothing must write nothing"
        );
    }

    /// `2>/dev/null` works under Landlock, as it already did under Bubblewrap and Seatbelt.
    ///
    /// It did not before: Landlock granted only read and execute on `/`, so the redirect failed
    /// with a bare "Permission denied" and every `cmd 2>/dev/null` in an agent's shell broke on
    /// this backend alone. Discarding output is not a write to the machine.
    #[test]
    fn discarding_output_to_dev_null_is_permitted_in_every_confinement() {
        let Some(abi) = super::landlock_abi().filter(|abi| *abi >= super::MIN_LANDLOCK_ABI) else {
            eprintln!("skipping: no usable Landlock on this host");
            return;
        };

        // Both root lists, since the name says "every confinement" and the body tested one.
        // `&[]` is read mode; a real root is `workspace`. `/dev/null` is granted by its own rule
        // rather than by the roots, so it has to hold under both -- and a rule that only worked
        // when the root list happened to be empty would have passed the old single case.
        let temp = tempfile::tempdir().expect("tempdir");
        let root = crate::workspace::canonical_for_test(temp.path());
        let workspace_root = [
            std::ffi::CString::new(std::os::unix::ffi::OsStrExt::as_bytes(root.as_os_str()))
                .expect("cstring"),
        ];

        for (label, writable) in [
            ("read mode", &[] as &[std::ffi::CString]),
            ("workspace", &workspace_root[..]),
        ] {
            let writable = writable.to_vec();
            let mut command = std::process::Command::new("/bin/sh");
            command.arg("-c").arg("echo discarded > /dev/null");
            unsafe {
                command.pre_exec(move || {
                    super::apply_landlock(abi, &writable).map_err(std::io::Error::from_raw_os_error)
                });
            }
            let status = command.status().expect("spawn");
            assert!(
                status.success(),
                "a redirect to /dev/null must succeed under {label}, where nothing else outside \
                 the roots is writable"
            );
        }
    }
}

#[cfg(test)]
mod confinement_tests {
    use super::Confinement;
    use crate::{permission::Permission, workspace};

    /// Every level maps to exactly one confinement, and only `unrestricted` maps to none.
    ///
    /// The table is spelled out rather than derived because this is the one decision in the change
    /// whose mistakes fail *open*: a level that should confine but resolves to `Unconfined` runs
    /// the shell with no sandbox at all, and nothing downstream would report it.
    #[test]
    fn every_level_resolves_to_exactly_one_confinement() {
        let temp = tempfile::tempdir().expect("tempdir");
        // Through `strip_verbatim`, because that is what `writable_roots` does and this asserts
        // equality against its output. Bare `canonicalize` returns a `\\?\`-prefixed path on
        // Windows, so the expectation was the one spelling production never produces and this test
        // failed on Windows alone -- the same prefix that reached the prompt, the model and the
        // database before `/cd` was taught to strip it.
        let base = crate::workspace::canonical_for_test(temp.path());
        let cwd: workspace::SharedCwd = std::sync::Arc::new(std::sync::RwLock::new(base.clone()));
        let scope = workspace::WriteScope::confined(vec![base.clone()]);

        for level in [Permission::None, Permission::Read] {
            assert_eq!(
                Confinement::resolve(true, level, &scope, &cwd),
                Confinement::ReadOnly,
                "{level} must confine the shell read-only"
            );
        }
        assert_eq!(
            Confinement::resolve(true, Permission::Workspace, &scope, &cwd),
            Confinement::Workspace(vec![base]),
            "workspace must hand the shell the same roots the file tools fence against"
        );
        for level in [Permission::Ask, Permission::Unrestricted] {
            assert_eq!(
                Confinement::resolve(true, level, &scope, &cwd),
                Confinement::Unconfined,
                "{level} runs the shell unsandboxed: at `ask` the approval prompt is the gate, and \
                 an approved command must reach as far as an approved `write_file`"
            );
        }
    }

    /// `[shell].sandbox = false` disables confinement at every level, as it always has.
    #[test]
    fn disabling_the_sandbox_unconfines_every_level() {
        let cwd = workspace::test_cwd();
        let scope = workspace::WriteScope::confined(vec![]);
        for level in [
            Permission::None,
            Permission::Read,
            Permission::Workspace,
            Permission::Ask,
            Permission::Unrestricted,
        ] {
            assert_eq!(
                Confinement::resolve(false, level, &scope, &cwd),
                Confinement::Unconfined
            );
        }
    }

    /// A `workspace` whose roots all failed to resolve grants nothing, rather than everything.
    #[test]
    fn a_workspace_with_no_resolvable_root_writes_nowhere() {
        let temp = tempfile::tempdir().expect("tempdir");
        let missing = temp.path().join("deleted-under-us");
        let cwd: workspace::SharedCwd = std::sync::Arc::new(std::sync::RwLock::new(missing));
        let scope = workspace::WriteScope::confined(vec![]);

        let confinement = Confinement::resolve(true, Permission::Workspace, &scope, &cwd);
        assert!(confinement.is_sandboxed(), "it must still be sandboxed");
        assert!(
            // Not `is_sandboxed()`, which holds by construction for any `Workspace(_)` and so
            // said nothing. What matters is that the root list came back empty, because that is
            // what every dialect turns into "no write may land anywhere".
            confinement.writable().is_empty(),
            "and must grant no writable root, which is read-only in effect"
        );
    }
}

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
    }

    #[cfg(target_os = "linux")]
    {
        if let Some(reason) = backend_unavailable_reason(&state.probe) {
            // We deliberately don't suggest a specific alternative backend here; the "other"
            // backend might also be unavailable on this host (kernel without Landlock, bwrap not
            // installed, etc.).
            tracing::warn!(
                "read-mode sandbox unavailable: {} (configured: {}); read-mode shell commands \
                 will fail until [shell].sandbox_backend names a usable backend, or sandboxing \
                 is disabled",
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
                "using Landlock for sandbox; install Bubblewrap for stronger protection, or pin \
                 [shell].sandbox_backend = \"landlock\" to suppress this warning"
            );
        }

        // Warn 3: the ABI clears `MIN_LANDLOCK_ABI` so the filesystem is genuinely write-protected,
        // but the mitigations added after v3 are absent. Each is a real hole a read-mode command
        // can walk through (a D-Bus or `systemd-run --user` call reaches a privileged
        // daemon that will happily write on its behalf), and none of them is visible to the
        // user otherwise, so the gap is named rather than left to the kernel version.
        if context == WarnContext::Startup
            && let BackendProbe::Ok(SandboxCapability::Landlock { abi_version }) = &state.probe
            && *abi_version < 9
        {
            let mut missing: Vec<&str> = Vec::new();
            if *abi_version < 5 {
                missing.push("device ioctls (v5)");
            }
            if *abi_version < 6 {
                missing.push("abstract Unix sockets and cross-domain signals (v6)");
            }
            missing.push("pathname Unix sockets (v9)");
            // Deliberately does not name Bubblewrap as the remedy. Measured: bwrap masks four
            // directories and unmounts nothing else, and it never unshares the network namespace,
            // so a socket in the abstract namespace or under `$HOME` stays reachable from inside
            // it. On this axis Landlock at v9 is the *stronger* backend, and sending a user to
            // install bwrap to close these channels sent them the wrong way.
            tracing::warn!(
                "Landlock ABI v{} write-protects the filesystem but does not restrict {}; a \
                 read-mode command can still reach a local service over those channels. A newer \
                 kernel closes them, and Bubblewrap does not",
                abi_version,
                missing.join(", "),
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
    landlock_probe_from_abi(landlock_abi())
}

/// The [`MIN_LANDLOCK_ABI`] policy, split from the syscall so it can be exercised at ABI values
/// this host does not have. Kernels below v3 are the ones that matter and are exactly the ones a
/// developer machine running a current kernel cannot reproduce.
#[cfg(target_os = "linux")]
fn landlock_probe_from_abi(abi: Option<i32>) -> BackendProbe {
    match abi {
        Some(abi_version) if abi_version >= MIN_LANDLOCK_ABI => {
            BackendProbe::Ok(SandboxCapability::Landlock { abi_version })
        }
        Some(abi_version) => BackendProbe::Missing {
            reason: format!(
                "Landlock ABI v{} is too old to write-protect the filesystem: truncate(2) is \
                 unmediated below v{} (needs Linux 6.2+), so a read-mode command could still empty \
                 an existing file",
                abi_version, MIN_LANDLOCK_ABI,
            ),
        },
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
/// other transient failure. Mirrors the fingerprint list Codex uses in
/// `codex-rs/sandboxing/src/bwrap.rs`.
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

/// Whether `path` is a directory only root can write to.
///
/// The test that decides whether a `bwrap` found there can be trusted. Group- and other-writable
/// are both disqualifying: a directory writable by any group the user is in is writable by the
/// user.
#[cfg(target_os = "linux")]
fn only_root_can_write(path: &std::path::Path) -> bool {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    let Ok(metadata) = std::fs::metadata(path) else {
        return false;
    };
    metadata.uid() == 0 && metadata.permissions().mode() & 0o022 == 0
}

/// A regular file with at least one execute bit set.
///
/// Named so the rule can be exercised on a file a test can actually create. Inline in
/// [`bwrap_on_path`] it was reachable only through `$PATH`, and both halves of the conjunction were
/// mutable without any test noticing.
#[cfg(target_os = "linux")]
fn is_executable_file(metadata: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt;

    metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
}

/// Whether a `bwrap` at `candidate`, found in `directory`, can be trusted to confine anything.
///
/// **Both** paths have to be root-only-writable. A root-owned binary in a directory the user can
/// write is one `mv` away from being the user's, so checking the binary alone is no check at all.
///
/// Split out from [`bwrap_on_path`] because the conjunction *is* the security property and cannot
/// be exercised through that function: the interesting inputs are mixed, one trusted path and one
/// not, and a test cannot create a root-owned file to supply them. As a named predicate it takes
/// two paths that exist on any host, `/usr/bin` and a temp dir, so the mixed cases become ordinary
/// arguments.
#[cfg(target_os = "linux")]
fn trusted_to_confine(candidate: &std::path::Path, directory: &std::path::Path) -> bool {
    only_root_can_write(candidate) && only_root_can_write(directory)
}

/// Look up `bwrap` on `$PATH`, accepting only a binary that the user cannot replace.
///
/// The plain `$PATH` walk was an unconfinement primitive. `$PATH` on an ordinary desktop holds
/// several directories the user can write -- `~/.local/bin`, a cargo or go bin dir, a toolchain
/// shim dir -- and every one of them precedes `/usr/bin`. A `bwrap` planted in any of them is
/// executed verbatim by the spawn path, so a six-line shell script that `exec`s its final argument
/// turns every `read` and `workspace` shell command into an unconfined one. Nothing noticed:
/// [`smoke_test_bwrap`] runs `bwrap <flags> /bin/true`, which a shim satisfies by construction, so
/// the probe reported `Ok` and both the startup warning and the lazy hard-error gate stayed quiet.
/// Demonstrated end to end -- a confined command wrote outside every root and the file landed on
/// the host, where real `bwrap` refuses the same argv.
///
/// It is a persistence primitive rather than a one-shot: one turn at `unrestricted`, one approved
/// `ask` command, or any post-install hook buys unconfined shells in every later session,
/// including the sessions a user opens at `read` precisely because they do not trust the turn.
///
/// `macos_impl` has hardcoded `/usr/bin/sandbox-exec` since it was written, with a comment naming
/// this exact attack. Linux is the backend meka auto-prefers, and it was the one searching `$PATH`.
///
/// Checking ownership rather than hardcoding a list keeps the distributions that put it elsewhere
/// working -- NixOS serves it out of a root-owned `/nix/store` path, which passes -- while refusing
/// anything under a directory the user can write. A rejected candidate is `warn!`ed rather than
/// skipped silently, because "bubblewrap is installed but meka fell back to Landlock" is otherwise
/// indistinguishable from "bubblewrap is not installed".
#[cfg(target_os = "linux")]
fn bwrap_on_path() -> Option<std::path::PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join("bwrap");
        let Ok(metadata) = std::fs::metadata(&candidate) else {
            continue;
        };
        if !is_executable_file(&metadata) {
            continue;
        }
        if trusted_to_confine(&candidate, &dir) {
            return Some(candidate);
        }
        tracing::warn!(
            "ignoring {} for sandboxing: it or its directory is writable by someone other than \
             root, so it cannot be trusted to confine anything",
            candidate.display()
        );
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

    let mut command = std::process::Command::new(bwrap_path);
    command
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
        .stderr(std::process::Stdio::piped());

    // Arm the parent-death signal here rather than leaving it to `--die-with-parent`.
    //
    // Measured, because the obvious reading is wrong in both directions. bwrap's own flag *does*
    // handle a `meka` that is SIGKILLed outright: kill the parent a second after spawning and the
    // sandbox goes with it. What it cannot cover is the window before bwrap reaches its own
    // `prctl`, several syscalls into a startup that includes a user-namespace handshake with its
    // child -- and a `bwrap` found parked for eleven days on this machine was blocked in exactly
    // that handshake, `read()`ing its sync eventfd, reparented to init with the flag in its argv.
    // Arming in `pre_exec` covers the child from before `execve`, which is the earliest point that
    // exists.
    //
    // The `getppid` re-read is what makes it a guarantee rather than a smaller window: the signal
    // only fires on a death that happens *after* the `prctl`, so a parent that died between fork
    // and this line would never deliver it. Reading the parent back afterwards catches that and
    // exits instead.
    //
    // `PR_SET_PDEATHSIG` tracks the parent *thread*, not the process, which is safe here only
    // because the caller blocks in the poll loop below for the child's whole life: the thread that
    // spawned it cannot retire while the child still matters. Do not lift this onto a spawn whose
    // child outlives the call.
    let parent = std::process::id() as libc::pid_t;
    // SAFETY: the closure runs after `fork` in the child and calls only async-signal-safe
    // functions (`prctl`, `getppid`, `_exit`), allocating nothing.
    unsafe {
        std::os::unix::process::CommandExt::pre_exec(&mut command, move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::getppid() != parent {
                libc::_exit(0);
            }
            Ok(())
        });
    }

    let mut child = match command.spawn() {
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
                            tracing::debug!("bwrap smoke test: stderr read failed: {}", error);
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
        // Routed through the probe rather than the raw syscall so the `MIN_LANDLOCK_ABI` policy is
        // applied in exactly one place: a test asking "what sandbox does this host have?" must get
        // the same answer production would act on, or it would happily exercise an ABI meka
        // refuses.
        if let BackendProbe::Ok(capability) = probe_landlock() {
            return capability;
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

/// Lowest Landlock ABI meka will sandbox with.
///
/// v3 (Linux 6.2) is where `LANDLOCK_ACCESS_FS_TRUNCATE` arrives. Below it `truncate(2)` is
/// unmediated, so a "read-only" child can still open an existing file for truncation and empty it:
/// `os.truncate(path, 0)` succeeds at v1 even though `open(O_WRONLY)` is denied. That is a write,
/// and meka documents read mode as write-protecting the filesystem, so accepting v1/v2 would be
/// promising a boundary the kernel is not enforcing.
///
/// Failing closed costs read-mode shell on kernels 5.13-6.1 (Ubuntu 22.04, Debian 12) that lack
/// Bubblewrap. That is the intended trade: Bubblewrap is auto-preferred whenever `bwrap` is on
/// `PATH` and is unaffected, and a refusal the user can act on beats a guarantee that quietly does
/// not hold.
#[cfg(target_os = "linux")]
const MIN_LANDLOCK_ABI: i32 = 3;

/// Raw kernel ABI probe. Reports what the kernel supports, not what meka will accept: the
/// [`MIN_LANDLOCK_ABI`] policy lives in [`probe_landlock`] so the "too old" case can be reported
/// differently from "no Landlock at all".
#[cfg(target_os = "linux")]
fn landlock_abi() -> Option<i32> {
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

/// Apply Landlock restrictions to the current process: read and execute everywhere, plus full
/// access beneath each path in `writable`.
///
/// `writable` is empty for a read-only confinement and holds the canonical workspace roots
/// otherwise. The paths arrive as [`std::ffi::CString`] because this runs after `fork`: building
/// them here would allocate, which the safety contract below forbids, so the caller prepares them
/// in the parent.
///
/// Landlock rules are additive grants with no deny form, so a writable root cannot have a subtree
/// carved back out of it. Nothing in meka asks for that today; if something ever does, this backend
/// cannot express it and the caller must say so rather than silently granting the whole root.
///
/// # Safety
///
/// This function uses raw syscalls and must only be called in a `pre_exec` context (after fork,
/// before exec) where the process is single-threaded. All operations are async-signal-safe
/// (syscalls only, no heap allocation).
#[cfg(target_os = "linux")]
pub unsafe fn apply_landlock(abi_version: i32, writable: &[std::ffi::CString]) -> Result<(), i32> {
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

        // `/dev/null` is writable in every confinement, including read-only.
        //
        // The other two Unix backends already do this and say why: the macOS profile calls
        // `/dev/null` writes "universally legitimate for shell redirects", and Bubblewrap's
        // `--dev /dev` supplies a writable one. Landlock granted neither, so `cmd 2>/dev/null`
        // failed with a bare "Permission denied" under this backend alone. That is a redirect
        // discards output; it is not a write to the machine, and refusing it confines nothing.
        let dev_null_fd = libc::open(c"/dev/null".as_ptr(), libc::O_PATH | libc::O_CLOEXEC);
        if dev_null_fd >= 0 {
            let path_beneath = LandlockPathBeneathAttr {
                allowed_access: LANDLOCK_ACCESS_FS_WRITE_FILE | LANDLOCK_ACCESS_FS_READ_FILE,
                parent_fd: dev_null_fd,
            };
            let ret = libc::syscall(
                libc::SYS_landlock_add_rule,
                ruleset_fd,
                LANDLOCK_RULE_PATH_BENEATH,
                &path_beneath as *const LandlockPathBeneathAttr,
                0u32,
            );
            libc::close(dev_null_fd);
            if ret < 0 {
                let error = *libc::__errno_location();
                libc::close(ruleset_fd);
                return Err(error);
            }
        }

        // Grant every right the ruleset handles beneath each workspace root. "Writable" here means
        // the full set rather than just `WRITE_FILE`: creating, removing, renaming and truncating
        // are all separate Landlock rights, and a shell that can write bytes but not create a file
        // would fail on the first `>` redirect.
        for path in writable {
            let root_fd = libc::open(path.as_ptr(), libc::O_PATH | libc::O_CLOEXEC);
            if root_fd < 0 {
                // A root that cannot be opened grants nothing. Skipping rather than failing keeps
                // a deleted directory from turning every command into a spawn error, and the
                // effect is restrictive: the confinement stays as tight as it was.
                continue;
            }
            let path_beneath = LandlockPathBeneathAttr {
                allowed_access: handled_access_for_abi(abi_version),
                parent_fd: root_fd,
            };
            let ret = libc::syscall(
                libc::SYS_landlock_add_rule,
                ruleset_fd,
                LANDLOCK_RULE_PATH_BENEATH,
                &path_beneath as *const LandlockPathBeneathAttr,
                0u32,
            );
            libc::close(root_fd);
            if ret < 0 {
                let error = *libc::__errno_location();
                libc::close(ruleset_fd);
                return Err(error);
            }
        }

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
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
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
;
; Outbound only. This carried `(allow network-bind (local ip "*:0"))` and the matching
; `network-inbound`, vendored from Codex, and a current macOS rejects `"*:0"` outright:
; `sandbox-exec: invalid port in network address`. That is a *parse* failure, so the whole profile
; is refused and every read-mode command exits 65 rather than running confined -- the mode was
; entirely broken on macOS, not merely narrowed. Dropped rather than respelled because the right
; spelling cannot be confirmed without a macOS host, and because read mode's stated network need is
; outbound (`curl http://x | pdftotext`); nothing in it binds a listening socket.
(allow network-outbound)

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

/// Build the SBPL profile and the `-D` parameter arguments for one confinement.
///
/// The read-only profile is used verbatim when `writable` is empty. Otherwise one
/// `(allow file-write* (subpath (param "MEKA_WRITABLE_n")))` clause is appended per root.
///
/// Roots travel as **parameters** rather than interpolated into the profile text, which is what
/// Codex does and for the same reason: SBPL string literals need escaping, and a path containing a
/// quote or a backslash would otherwise either break the profile or, worse, change what it matches.
/// A parameter is passed out of band and needs no quoting at all.
///
/// Compiled on every platform, and gated only at the call site in `tools::shell`. Nothing in here
/// touches a macOS API: it builds a string and a list of `OsString`s. Gating the function meant no
/// local target compiled it and no test ran it, so the one piece of macOS logic meka could check
/// anywhere was the one piece nothing checked. `sandbox-exec` itself stays macOS-only and stays
/// genuinely unverified; this at least shrinks that to the exec call. Same `cfg_attr` shape as
/// `POWERSHELL_UTF8_PRELUDE` above.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub fn sandbox_profile_for(writable: &[std::path::PathBuf]) -> (String, Vec<std::ffi::OsString>) {
    let mut profile = String::from(SANDBOX_PROFILE_READONLY);
    let mut params = Vec::new();
    for (index, root) in writable.iter().enumerate() {
        let key = format!("MEKA_WRITABLE_{index}");
        profile.push_str(&format!(
            "\n(allow file-write* (subpath (param \"{key}\")))\n"
        ));
        // The value is assembled as an `OsString` and the path pushed in whole, so a root that is
        // not valid UTF-8 is passed through byte-for-byte rather than being dropped. `Command::arg`
        // takes an `OsStr`, so nothing downstream needs it to be text. Only the key, which meka
        // authors itself, is built as a string.
        let mut param = std::ffi::OsString::from(&key);
        param.push("=");
        param.push(root.as_os_str());
        params.push(std::ffi::OsString::from("-D"));
        params.push(param);
    }
    (profile, params)
}

/// PowerShell prelude that switches `$OutputEncoding` and `[Console]::OutputEncoding` to UTF-8. See
/// [`wrap_command_with_utf8_output`] for why this is necessary.
///
/// Guarded on the language mode, and belt-and-braces wrapped in `try`/`catch`, because a
/// `WRITE_RESTRICTED` token puts PowerShell into **ConstrainedLanguage** mode, where setting a
/// property on a non-core type is refused: "Property setting is supported only on core types in
/// this language mode." Unguarded, that error was printed to stderr ahead of *every* shell command
/// at `workspace` permission on Windows, which the model reads as the command having failed.
/// Measured on Windows 11 10.0.26200: `unrestricted` and `read` both report `FullLanguage`, and
/// only the restricted token constrains it, so this is a cost of the workspace boundary rather than
/// a property of the host.
///
/// Skipping it means output at `workspace` is decoded with the host's legacy code page, so
/// non-ASCII may be mangled there. A wrong character beats an error on every line, and there is no
/// other way to reach the encoding from inside ConstrainedLanguage.
#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
const POWERSHELL_UTF8_PRELUDE: &str = "if($ExecutionContext.SessionState.LanguageMode -eq 'FullLanguage'){try{\
     [Console]::OutputEncoding=[System.Text.Encoding]::UTF8;\
     $OutputEncoding=[System.Text.Encoding]::UTF8}catch{}};";

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
/// should switch to `unrestricted` (trusted-operation path; no scrubbing applies).
///
/// **Windows** uses a heuristic deny-list ([`is_sensitive_env_name`]) because PowerShell pulls in a
/// long tail of system vars (`PSModulePath`, `APPDATA`, `ProgramFiles`, etc.) that don't fit a tidy
/// allow-list; an allow-list version was tried first and broke core cmdlets.
pub fn sandbox_child_env() -> Vec<(std::ffi::OsString, std::ffi::OsString)> {
    std::env::vars_os()
        .filter(|(name_os, _)| match name_os.to_str() {
            Some(name) => keep_sandbox_env_var(name),
            // A name that is not UTF-8 is dropped on purpose, and only the name. Both filters below
            // decide by matching text, so a name they cannot read is one they cannot rule out, and
            // Windows' arm is a deny-list: passing an unexaminable name through would be the one
            // direction that fails open. Values are a different question and go through
            // `encode_wide` untouched, since the destination block is UTF-16 natively.
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
    // How the machine reaches the network at all, on a host that does not route directly. A child
    // that cannot see these connects to nothing and reports a TLS or DNS failure that names none of
    // the real cause, which for an MCP server means it starts, registers its tools, and then fails
    // every call. None of them grants authority: they say where to go and whom to trust, and the
    // child was going to make the request either way.
    //
    // Deliberately not extended to `SSH_AUTH_SOCK` (a live credential agent), `NODE_OPTIONS` (which
    // takes `--require`, i.e. arbitrary code), or the import-path family `PYTHONPATH` / `NODE_PATH`
    // / `VIRTUAL_ENV`, which redirect what a program loads. A server that needs one of those takes
    // it explicitly through `${VAR}` in its own `[[mcp.servers]] env` table, where the user has
    // said so.
    const ALLOW_NETWORK: &[&str] = &[
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "ALL_PROXY",
        "all_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
    ];
    // Prefix allow-list keeps the LC_* / XDG_* families future-proof without enumerating each var.
    // Locale (`LC_ALL`, `LC_CTYPE`, `LC_MESSAGES`, …) and XDG paths (`XDG_RUNTIME_DIR`,
    // `XDG_CONFIG_HOME`, …) are both legitimately broad.
    const ALLOW_PREFIX: &[&str] = &["LC_", "XDG_"];

    if ALLOW_EXACT.contains(&name) || ALLOW_NETWORK.contains(&name) {
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
        // Code-execution vectors, which the Unix allow-list refuses by name and this arm did not.
        // The two arms are supposed to implement one policy, and a test on the Unix side names
        // exactly these while the Windows side let all of them through.
        //
        // They are not credentials; they are ways to make an ordinary command run something else.
        // `NODE_OPTIONS` takes `--require`, `PYTHONPATH` / `NODE_PATH` prepend an import path, and
        // `PIP_INDEX_URL` redirects where a package is fetched from. The values come from meka's
        // own parent environment, so this is not an escalation the agent can drive -- but the
        // sandboxed child is exactly the place where "the same command, quietly doing something
        // else" is worth refusing.
        "NODE_OPTIONS",
        "NODE_PATH",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "VIRTUAL_ENV",
        "PIP_INDEX_URL",
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
pub mod windows_impl {
    use std::{
        fs::File,
        mem,
        os::windows::{ffi::OsStrExt, io::FromRawHandle, process::ExitStatusExt},
        process::ExitStatus,
        ptr,
    };

    use windows_sys::Win32::{
        Foundation::{
            CloseHandle, ERROR_PRIVILEGE_NOT_HELD, GENERIC_READ, GENERIC_WRITE, HANDLE,
            HANDLE_FLAG_INHERIT, INVALID_HANDLE_VALUE, LocalFree, SetHandleInformation, TRUE,
            WAIT_OBJECT_0,
        },
        Security::{
            ACCESS_ALLOWED_ACE, ACE_HEADER, ACL, AdjustTokenPrivileges,
            Authorization::{
                ConvertStringSidToSidW, EXPLICIT_ACCESS_W, GRANT_ACCESS, GetNamedSecurityInfoW,
                NO_MULTIPLE_TRUSTEE, REVOKE_ACCESS, SE_FILE_OBJECT, SetEntriesInAclW,
                SetNamedSecurityInfoW, TRUSTEE_IS_SID, TRUSTEE_IS_UNKNOWN, TRUSTEE_W,
            },
            CreateRestrictedToken, DACL_SECURITY_INFORMATION, DuplicateTokenEx, EqualSid, GetAce,
            GetLengthSid, GetTokenInformation, SECURITY_ATTRIBUTES, SID_AND_ATTRIBUTES,
            SecurityAnonymous, SetTokenInformation, TOKEN_ADJUST_DEFAULT, TOKEN_ADJUST_PRIVILEGES,
            TOKEN_ASSIGN_PRIMARY, TOKEN_DUPLICATE, TOKEN_GROUPS, TOKEN_MANDATORY_LABEL,
            TOKEN_QUERY, TokenGroups, TokenIntegrityLevel, TokenPrimary,
        },
        Storage::FileSystem::{CreateFileW, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING},
        System::{
            Console::{
                AllocConsole, GetConsoleWindow, GetStdHandle, STD_ERROR_HANDLE, STD_INPUT_HANDLE,
                STD_OUTPUT_HANDLE, SetStdHandle,
            },
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
        UI::WindowsAndMessaging::{SW_HIDE, ShowWindow},
    };

    // SE_GROUP_INTEGRITY isn't exported by the `Win32_Security` feature in windows-sys 0.61; define
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
    /// Which token a sandboxed Windows child should run under.
    ///
    /// The two are different mechanisms, not two settings of one. `LowIntegrity` drops the token's
    /// integrity label so it cannot touch anything above Low; `WriteRestricted` leaves integrity
    /// alone and instead intersects every *write* access against a restricting-SID list, granting
    /// back exactly the workspace roots. Neither can express the other's boundary.
    pub enum WindowsConfinement {
        /// Reads everywhere, writes nowhere outside the Low-integrity surface.
        LowIntegrity,
        /// Reads everywhere, writes only beneath these canonical roots.
        WriteRestricted(Vec<std::path::PathBuf>),
    }

    /// `WRITE_RESTRICTED`: the token's restricting SIDs are intersected for write accesses only.
    const WRITE_RESTRICTED: u32 = 0x8;
    /// `DISABLE_MAX_PRIVILEGE`: strip every privilege from the restricted token except
    /// `SeChangeNotifyPrivilege`, which has to stay or traverse checks fail on every path.
    const DISABLE_MAX_PRIVILEGE: u32 = 0x1;
    // Not re-exported by the `Win32_Foundation` feature in windows-sys 0.61, and defined here for
    // the same reason as the constants around it. `DELETE` is a standard access right, fixed at
    // this value since NT and documented under ACCESS_MASK.
    const DELETE: u32 = 0x0001_0000;
    /// Inheritable by both files and subdirectories, so one ACE on the root covers the tree.
    ///
    /// Belongs on this constant, not on `DELETE` above.
    const SUB_CONTAINERS_AND_OBJECTS_INHERIT: u32 = 0x3;
    const SE_GROUP_LOGON_ID: u32 = 0xC000_0000;
    const ACCESS_ALLOWED_ACE_TYPE: u8 = 0x0;

    /// The per-workspace write identity: a deterministic SID in the non-unique-authority space
    /// (`S-1-4-x-y`), derived from the canonical root path.
    ///
    /// `S-1-4` is the point of the design. It is a SID space that corresponds to no real principal,
    /// so meka can mint one per workspace without creating an account and without any elevation.
    /// Its power is defined entirely by the ACEs that name it, which exist only on that workspace's
    /// own tree; the string itself is not a secret.
    ///
    /// Deterministic so the same workspace derives the same identity across sessions, which is what
    /// lets a grant be recognised and revoked rather than accumulating a fresh one per run.
    pub fn workspace_write_sid(root: &std::path::Path) -> String {
        // FNV-1a over the path's own UTF-16 units, which is how Windows stores it, rather than over
        // a `to_string_lossy` rendering. Lossy conversion turns every unpaired surrogate into
        // U+FFFD, so two directories that Windows considers distinct could hash alike and share one
        // capability. Reading the units the OS actually holds removes the question.
        //
        // A cryptographic digest would buy nothing: the input is not secret, and a collision
        // between two of a user's own workspaces costs a shared grant rather than an escape.
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let mut absorb = |byte: u8| {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        };
        for unit in root.as_os_str().encode_wide() {
            // Windows matches paths case-insensitively, so `C:\Work` and `c:\work` name one
            // directory and must reach one capability. Folded over ASCII only: that covers drive
            // letters and nearly every real path, and a non-ASCII case difference yields a second
            // SID, which is a second grant rather than a wrong one.
            let folded = match u8::try_from(unit) {
                Ok(ascii) => u16::from(ascii.to_ascii_lowercase()),
                Err(_) => unit,
            };
            absorb((folded & 0xff) as u8);
            absorb((folded >> 8) as u8);
        }
        let first = ((hash & 0x3fff_ffff) as u32).max(1);
        let second = (((hash >> 32) & 0x3fff_ffff) as u32).max(1);
        format!("S-1-4-{first}-{second}")
    }

    unsafe fn has_workspace_ace(acl: *const ACL, sid: *mut core::ffi::c_void) -> bool {
        if acl.is_null() {
            return false;
        }
        let count = unsafe { (*acl).AceCount };
        for index in 0..u32::from(count) {
            let mut ace: *mut core::ffi::c_void = ptr::null_mut();
            if unsafe { GetAce(acl, index, &mut ace) } == 0 {
                continue;
            }
            let header = ace as *const ACE_HEADER;
            if unsafe { (*header).AceType } != ACCESS_ALLOWED_ACE_TYPE {
                continue;
            }
            // An ACE that does not carry both inherit flags covers the root only, so the tree below
            // it is unwritable and the grant has to be re-placed. Measured on a real ACL, the flags
            // meka's own ACE comes back with are `0xb`: both inherit bits plus `INHERIT_ONLY_ACE`,
            // which `SetEntriesInAclW` adds because a mask holding generic bits means nothing when
            // applied to the object itself. So this tests for the two bits it needs rather than for
            // equality, which would never match.
            let flags = u32::from(unsafe { (*header).AceFlags });
            if flags & SUB_CONTAINERS_AND_OBJECTS_INHERIT != SUB_CONTAINERS_AND_OBJECTS_INHERIT {
                continue;
            }
            let allowed = ace as *const ACCESS_ALLOWED_ACE;
            // The generic bits survive the round trip: measured on a real ACL, the mask comes back
            // as `0x40010000`, exactly the `GENERIC_WRITE | DELETE` that was written. Nothing maps
            // it to `FILE_GENERIC_WRITE` on the way in, so reading it back is a plain comparison.
            let mask = unsafe { (*allowed).Mask };
            if mask & (GENERIC_WRITE | DELETE) != GENERIC_WRITE | DELETE {
                continue;
            }
            let ace_sid = unsafe { &raw const (*allowed).SidStart } as *const core::ffi::c_void;
            if unsafe { EqualSid(ace_sid as *mut core::ffi::c_void, sid) } != 0 {
                return true;
            }
        }
        false
    }

    /// Add or remove the workspace capability's inheritable write ACE on `root`.
    ///
    /// Granting requires only that the caller *own* the directory, which supplies `WRITE_DAC`
    /// implicitly. No elevation, and no account creation.
    ///
    /// The ACE is real, standing state on the user's filesystem, visible to `icacls`. meka takes it
    /// back when the process exits; see [`WindowsGrants`] for what that does and does not cover.
    /// Whether `acl` already carries meka's inheritable write ACE for `sid`.
    ///
    /// Conservative in the safe direction: anything it cannot positively identify reads as absent,
    /// which re-places an ACE that was already correct. That costs time, never reach.
    ///
    /// Identifying by SID alone is sound here because `workspace_write_sid` mints an `S-1-4`
    /// identity that corresponds to no real principal and is derived from the root path, so nothing
    /// but meka ever names it in an ACE.
    unsafe fn set_workspace_ace(root: &std::path::Path, grant: bool) -> std::io::Result<()> {
        let sid_text: Vec<u16> = workspace_write_sid(root)
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();
        let mut sid: *mut core::ffi::c_void = ptr::null_mut();
        if unsafe { ConvertStringSidToSidW(sid_text.as_ptr(), &mut sid) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        let _sid_guard = LocalFreeGuard(sid);

        let target: Vec<u16> = root
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();

        let mut existing: *mut ACL = ptr::null_mut();
        let mut descriptor: *mut core::ffi::c_void = ptr::null_mut();
        let status = unsafe {
            GetNamedSecurityInfoW(
                target.as_ptr(),
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION,
                ptr::null_mut(),
                ptr::null_mut(),
                &mut existing,
                ptr::null_mut(),
                &mut descriptor,
            )
        };
        if status != 0 {
            return Err(std::io::Error::from_raw_os_error(status as i32));
        }
        let _descriptor_guard = LocalFreeGuard(descriptor);

        // Placing the ACE re-propagates it over the whole tree, and `ensure` runs before *every*
        // `execute_command`. Measured on the test box against a 5000-file workspace: 282ms per
        // command, and every file's USN bumped each time.
        //
        // So skip the write when the ACE is already there. The read above has happened either way,
        // and this only reads its result, so the check costs nothing and does not walk the tree.
        // Same workspace with the skip in place: 282ms for the first command, 21us for each one
        // after it.
        //
        // This is a short-circuit, not a cache: it asks the filesystem, not a ledger. If another
        // meka's `revoke_all` took the ACE off the root while this one was running, it reads as
        // absent here and gets re-placed.
        //
        // What it does *not* preserve is repair below the root. Re-propagating on every command
        // also fixed any object beneath the root that had lost the inherited ACE -- a directory
        // with inheritance disabled, a tree restored with `robocopy /COPYALL`, a folder moved in
        // from another volume. Those now stay unwritable, with a bare access-denied and nothing
        // explaining it, until something takes the root's ACE off and back on. Judged worth it
        // against 282ms on every single command, but it is a real narrowing rather than a pure
        // optimisation.
        if grant && unsafe { has_workspace_ace(existing, sid) } {
            return Ok(());
        }

        // `GENERIC_WRITE | DELETE`, not `GENERIC_ALL`. The capability is granted so a confined
        // child can *write* inside the workspace, and nothing in that job needs the rest of full
        // control. Granting it anyway meant `icacls` showed `(F)` while the code and the docs both
        // called this a write ACE, so a reader auditing the ACL and a reader auditing the source
        // came away with different answers.
        //
        // `GENERIC_WRITE` covers what a workspace write actually needs: on a file, write and append
        // data plus attributes; on a directory, the same two bits mean add-file and
        // add-subdirectory. `DELETE` is separate and is required for replacing a file, which is how
        // every atomic write lands -- `write_file` renames a temp file over the target, and the
        // rename needs delete rights on what it displaces.
        let access = EXPLICIT_ACCESS_W {
            grfAccessPermissions: GENERIC_WRITE | DELETE,
            grfAccessMode: if grant { GRANT_ACCESS } else { REVOKE_ACCESS },
            grfInheritance: SUB_CONTAINERS_AND_OBJECTS_INHERIT,
            Trustee: TRUSTEE_W {
                pMultipleTrustee: ptr::null_mut(),
                MultipleTrusteeOperation: NO_MULTIPLE_TRUSTEE,
                TrusteeForm: TRUSTEE_IS_SID,
                TrusteeType: TRUSTEE_IS_UNKNOWN,
                ptstrName: sid as *mut u16,
            },
        };

        let mut updated: *mut ACL = ptr::null_mut();
        let status = unsafe { SetEntriesInAclW(1, &access, existing, &mut updated) };
        if status != 0 {
            return Err(std::io::Error::from_raw_os_error(status as i32));
        }
        let _updated_guard = LocalFreeGuard(updated as *mut core::ffi::c_void);

        let status = unsafe {
            SetNamedSecurityInfoW(
                target.as_ptr() as *mut u16,
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION,
                ptr::null_mut(),
                ptr::null_mut(),
                updated,
                ptr::null_mut(),
            )
        };
        if status != 0 {
            return Err(std::io::Error::from_raw_os_error(status as i32));
        }
        Ok(())
    }

    /// The workspace ACEs this process has placed, so they can be taken back.
    ///
    /// The grant is standing state on the user's own directories, and meka is the only thing that
    /// knows it put it there. Revoking is what keeps `workspace` from leaving a permission trail
    /// behind on every folder the agent was ever run in. The cost is that the next run
    /// re-propagates the ACE, which is one pass over the tree.
    ///
    /// Reach the process's ledger through [`process_grants`] rather than constructing one, for the
    /// reason given there. [`Drop`] revokes as a backstop for the instances tests build; the shared
    /// one is a `static` and never drops, so [`crate::sandbox::release_process_grants`] is what
    /// actually closes an ordinary run.
    ///
    /// A `SIGKILL` or a hard crash still strands the ACE. That is untidy rather than unsafe: the
    /// capability SID names no real principal, so the residue grants nothing to anyone, and the
    /// next run in the same directory recognises and reuses it instead of adding a second.
    #[derive(Default)]
    pub struct WindowsGrants {
        granted: std::sync::Mutex<std::collections::BTreeSet<std::path::PathBuf>>,
    }

    /// The one ledger for this process.
    ///
    /// An ACE is filesystem state, not session state. Two sessions confining the same root place a
    /// single ACE between them, so a per-session ledger has whichever session ends first revoke the
    /// grant out from under the other. A sub-agent made that concrete rather than theoretical: its
    /// `ToolRegistry` is dropped the moment its task finishes, which under a per-registry ledger
    /// took the parent's still-live grants with it.
    pub fn process_grants() -> &'static std::sync::Arc<WindowsGrants> {
        static GRANTS: std::sync::OnceLock<std::sync::Arc<WindowsGrants>> =
            std::sync::OnceLock::new();
        GRANTS.get_or_init(|| std::sync::Arc::new(WindowsGrants::default()))
    }

    impl WindowsGrants {
        /// Ensure `root` carries the capability's write ACE.
        ///
        /// Placed every time rather than skipped when the ledger already lists it. The ACE is
        /// machine state, not process state, and `process_grants` reasoned only about the two
        /// registries inside one process. Two mekas at `workspace` in the same directory both grant
        /// -- `SetEntriesInAcl` merges, so there is one ACE -- and whichever exits first revokes
        /// it. The survivor's ledger still says "granted", so a cached `ensure`
        /// short-circuits and never re-places it, and every later shell write in that
        /// session fails with a bare access-denied that nothing explains. Re-placing is
        /// idempotent and costs one pass over the tree, which is why the caller runs this
        /// off the async executor.
        pub fn ensure(&self, root: &std::path::Path) -> std::io::Result<()> {
            unsafe { set_workspace_ace(root, true)? };
            let mut granted = self
                .granted
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            granted.insert(root.to_path_buf());
            Ok(())
        }

        /// The roots this ledger currently believes carry an ACE.
        ///
        /// Test-only. Production never reads the set, it only adds to it and drains it at exit;
        /// this exists so a test can ask which ledger a tool built by the production path is
        /// actually writing into.
        #[cfg(test)]
        pub fn granted_roots(&self) -> std::collections::BTreeSet<std::path::PathBuf> {
            self.granted
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()
        }

        /// Take every ACE back. Best-effort per root: one failure must not strand the others.
        ///
        /// A root whose revoke failed stays in the ledger, so a later attempt (or the next process
        /// running against the same root) still knows the ACE is out there. Clearing the whole set
        /// unconditionally meant the one case worth remembering -- the ACE meka placed and could
        /// not take back -- was the case it forgot.
        pub fn revoke_all(&self) {
            let mut granted = self
                .granted
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut stranded = std::collections::BTreeSet::new();
            for root in granted.iter() {
                if let Err(error) = unsafe { set_workspace_ace(root, false) } {
                    stranded.insert(root.clone());
                    tracing::warn!(
                        "could not revoke the workspace write ACE on {}: {}. Remove it with \
                         `icacls \"{}\" /remove:g *{}` if it is unwanted",
                        root.display(),
                        error,
                        root.display(),
                        workspace_write_sid(root)
                    );
                }
            }
            *granted = stranded;
        }
    }

    impl Drop for WindowsGrants {
        fn drop(&mut self) {
            self.revoke_all();
        }
    }

    /// The token's logon-session SID (`S-1-5-5-x-y`), or `None` if it carries none.
    ///
    /// Needed in the restricting list so the child keeps reach to the per-logon objects a shell
    /// expects: the window station, the desktop, and the named pipes under them. Leaving it out
    /// does not tighten the filesystem boundary, it just breaks the process.
    ///
    /// Returns an owned copy of the SID's bytes. The caller must keep it alive across the
    /// `CreateRestrictedToken` call, which reads through the pointer into it; see the note in the
    /// body for why this copies rather than leaking the whole `TOKEN_GROUPS` buffer.
    unsafe fn logon_session_sid(token: HANDLE) -> Option<Vec<u8>> {
        let mut needed = 0u32;
        unsafe { GetTokenInformation(token, TokenGroups, ptr::null_mut(), 0, &mut needed) };
        if needed == 0 {
            return None;
        }
        // `u64` elements for alignment; the length is rounded up to whole units.
        let mut buffer: Vec<u64> = vec![0u64; needed.div_ceil(8) as usize];
        if unsafe {
            GetTokenInformation(
                token,
                TokenGroups,
                buffer.as_mut_ptr() as *mut core::ffi::c_void,
                needed,
                &mut needed,
            )
        } == 0
        {
            return None;
        }
        // Aligned storage, because `TOKEN_GROUPS` needs 8-byte alignment and `Vec<u8>` guarantees
        // only 1. Windows' allocator happens to satisfy it, so the `u8` version worked in practice
        // and was UB by the letter.
        //
        // Read through raw pointers rather than through a `&TOKEN_GROUPS`. Forming the reference
        // asserted that a whole `TOKEN_GROUPS` (24 bytes) is dereferenceable, which a token
        // reporting zero groups would not satisfy, and `Groups` is a flexible array member whose
        // declared length is 1 -- so building a longer slice from a reference to it is the C idiom
        // that Rust's aliasing model rejects. Neither is reachable with a real token, and the
        // comment above claimed the letter-of-the-law problem had been dealt with, so this makes
        // that true instead of nearly true.
        let groups = buffer.as_ptr() as *const TOKEN_GROUPS;
        let group_count = unsafe { (*groups).GroupCount } as usize;
        let entries = unsafe {
            std::slice::from_raw_parts(
                (&raw const (*groups).Groups) as *const SID_AND_ATTRIBUTES,
                group_count,
            )
        };
        let found = entries
            .iter()
            .find(|entry| entry.Attributes & SE_GROUP_LOGON_ID != 0)?;

        // Copied out rather than leaked. The SID has to outlive `CreateRestrictedToken`, and the
        // previous version bought that with `Box::leak` on the whole `TOKEN_GROUPS` buffer -- one
        // permanent leak per spawn, 1-4 KB on a domain-joined account. Invisible for a one-shot CLI
        // run and unbounded for `meka serve`. A SID is self-describing and fixed-size, so the
        // caller can own just those bytes.
        let length = unsafe { GetLengthSid(found.Sid) } as usize;
        if length == 0 {
            return None;
        }
        let mut sid = vec![0u8; length];
        unsafe {
            std::ptr::copy_nonoverlapping(found.Sid as *const u8, sid.as_mut_ptr(), length);
        }
        Some(sid)
    }

    /// Make sure this process has a console, allocating a hidden one if it does not.
    ///
    /// A `WRITE_RESTRICTED` child cannot *create* a console, only inherit one. That is the measured
    /// rule, not the documented one: `CREATE_NO_WINDOW` and `CREATE_NEW_CONSOLE` both ask for a new
    /// console and both die with `STATUS_DLL_INIT_FAILED` (0xC0000142) before `main` runs, and so
    /// does a child of a parent that has no console to pass on. Allocating one here and hiding its
    /// window makes the headless cases (ACP under an editor, `meka serve` as a service) behave like
    /// the terminal case.
    ///
    /// Runs at most once, and only when a restricted spawn is actually about to happen, so a meka
    /// that never reaches `workspace` never allocates anything.
    fn ensure_console() {
        // Not a `Once`: a failed allocation must be retried.
        //
        // `Once` is consumed whether the closure succeeded or not, so a single transient
        // `AllocConsole` failure disabled `workspace`'s shell for the life of the process -- every
        // later command dying with `STATUS_DLL_INIT_FAILED` before `main`, with nothing to retry
        // it. The `GetConsoleWindow` check is itself the idempotence guard: once a console exists
        // this returns immediately, so the only repeated work is on the path that has not
        // succeeded yet.
        if unsafe { !GetConsoleWindow().is_null() } {
            return;
        }
        unsafe { allocate_hidden_console() };
    }

    /// The body of [`ensure_console`], minus the `Once` and the already-have-one check.
    ///
    /// Split out so a test can reach it: with a console present -- which is every `cargo test` run
    /// -- `ensure_console` returns at its first line, so a test calling it exercises nothing. The
    /// one interaction that needed proving was the only one it could not reach.
    unsafe fn allocate_hidden_console() {
        // Snapshot the three standard handles across `AllocConsole`.
        //
        // Windows documents it as *initialising* the process's standard handles to the new
        // console's buffers. That is exactly wrong for the cases this function exists for: under
        // ACP meka speaks JSON-RPC over stdin/stdout, and `meka serve` may be piped. Rebinding them
        // mid-session on the first `workspace` shell command would break the transport silently,
        // with no error anywhere. Restoring unconditionally costs nothing when Windows leaves them
        // alone.
        let saved: [(u32, HANDLE); 3] = [
            (STD_INPUT_HANDLE, unsafe { GetStdHandle(STD_INPUT_HANDLE) }),
            (STD_OUTPUT_HANDLE, unsafe {
                GetStdHandle(STD_OUTPUT_HANDLE)
            }),
            (STD_ERROR_HANDLE, unsafe { GetStdHandle(STD_ERROR_HANDLE) }),
        ];

        if unsafe { AllocConsole() } == 0 {
            // `warn!`, because the consequence is total rather than cosmetic: a restricted child
            // cannot *create* a console, only inherit one, so without this every subsequent
            // `workspace` shell command dies with `STATUS_DLL_INIT_FAILED` before reaching `main`.
            // At `debug!` the user saw nothing at all at default verbosity and a shell that simply
            // did not work.
            tracing::warn!(
                "could not allocate a console ({}); shell commands at `workspace` will fail to \
                 start. Running meka from a terminal avoids this",
                std::io::Error::last_os_error()
            );
            return;
        }

        for (which, handle) in saved {
            if !handle.is_null() && handle != INVALID_HANDLE_VALUE {
                unsafe { SetStdHandle(which, handle) };
            }
        }

        let window = unsafe { GetConsoleWindow() };
        if !window.is_null() {
            unsafe { ShowWindow(window, SW_HIDE) };
        }
    }

    /// Spawn one sandboxed child under `confinement`.
    ///
    /// Both variants share every step after the token: the pipe setup with its handle-inheritance
    /// narrowing, the job object, the suspended spawn. Only the token differs, and only the
    /// restricted path needs a console.
    pub fn spawn_sandboxed_command(
        command: &str,
        confinement: &WindowsConfinement,
        cwd: &std::path::Path,
    ) -> std::io::Result<SandboxedChild> {
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
        // Before anything is spawned, and only for the path that needs it.
        if matches!(confinement, WindowsConfinement::WriteRestricted(_)) {
            ensure_console();
        }

        let wrapped_command = super::wrap_command_with_utf8_output(command);

        // The session working directory, NUL-terminated for `lpCurrentDirectory`.
        //
        // Both `CreateProcess*` calls passed null here, which means "inherit the *process* cwd" --
        // and meka deliberately never mutates that (`main.rs` says so outright), so the sandboxed
        // child ran somewhere else entirely from every other path. Under ACP or `meka serve` the
        // session cwd is client-supplied and the process cwd is wherever the editor or unit
        // started, so at `workspace` the ACE was granted on one directory and the command ran in
        // another: relative writes hit a directory with no capability and were denied, and relative
        // reads silently read the wrong tree. Invisible in a plain terminal REPL, where the two are
        // the same path.
        let cwd_utf16: Vec<u16> = cwd
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();

        let mut cmd_line = String::from(r#""powershell.exe" -NoProfile -NonInteractive -Command "#);
        cmd_line.push_str(&super::quote_command_arg(&wrapped_command));

        // SAFETY: All Win32 calls below are documented and we check return values. Handles are
        // wrapped in `OwnedHandle` to close on drop. Pipe handles transfer ownership into the
        // spawned child (for the write ends) or into the returned `File` (for the read ends).
        unsafe {
            // 1. Open our own process token and duplicate it as a primary token we can modify. The
            //    duplicate is what we'll confine; we must NOT mutate our own token.
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

            // Applied only for the Low-integrity confinement. The restricted path deliberately
            // leaves the integrity label alone: dropping to Low there would confine the child to
            // the Low surface *as well*, which would take away the workspace the ACEs just granted.
            if matches!(confinement, WindowsConfinement::LowIntegrity)
                && SetTokenInformation(
                    low_token.as_raw(),
                    TokenIntegrityLevel,
                    &label as *const _ as *const core::ffi::c_void,
                    mem::size_of::<TOKEN_MANDATORY_LABEL>() as u32,
                ) == 0
            {
                return Err(std::io::Error::last_os_error());
            }

            // 3b. For a workspace confinement, replace the token with a WRITE_RESTRICTED one whose
            //     restricting SIDs are the logon session, Everyone, and one capability per root.
            //
            //     `WRITE_RESTRICTED` intersects the restricting list against the DACL for *write*
            //     accesses only, so reads and execution are untouched and a write succeeds exactly
            //     where one of those SIDs has a write ACE. The capability SIDs have one only on
            //     their own workspace root, which is what makes the boundary.
            let low_token = match confinement {
                WindowsConfinement::LowIntegrity => low_token,
                WindowsConfinement::WriteRestricted(roots) => {
                    let mut restricting: Vec<SID_AND_ATTRIBUTES> = Vec::new();
                    let mut sid_guards: Vec<LocalFreeGuard> = Vec::new();

                    // Held in this scope so the pointer handed to `CreateRestrictedToken` below
                    // stays valid; it is dropped with the rest of the arm, after the call.
                    let logon_sid = logon_session_sid(low_token.as_raw());
                    if let Some(bytes) = logon_sid.as_ref() {
                        restricting.push(SID_AND_ATTRIBUTES {
                            Sid: bytes.as_ptr() as *mut core::ffi::c_void,
                            Attributes: 0,
                        });
                    }
                    // Everyone, plus one capability per root.
                    //
                    // Everyone is load-bearing, and not for the reason a filesystem test suggests.
                    // Dropping it still let a child create a file under a granted root, overwrite
                    // one already there, and write to a capture pipe: by that measure it looks
                    // free, and removing it looks like a pure tightening. What it actually costs is
                    // the shell itself. meka spawns `powershell.exe`, and with the restricting list
                    // cut to the logon SID and the capability, PowerShell dies before evaluating
                    // anything with `Starting the CLR failed with HRESULT 80070005`
                    // (E_ACCESSDENIED). Not one command in five ran. Measured both ways on Windows
                    // 11 10.0.26200; the .NET runtime reaches something on startup that only
                    // Everyone admits.
                    //
                    // The price is the one hole the docs name: a file carrying an explicit
                    // `Everyone: Write` ACE stays writable from outside the workspace. That is the
                    // only case the two configurations differed on, apart from the shell not
                    // starting. Paying it buys a workspace mode that can run commands at all.
                    //
                    // Probe the shell, not just the filesystem, before touching this list. A
                    // `cmd.exe` probe passes every case here and tells you nothing about the
                    // process meka actually spawns.
                    let mut sid_texts: Vec<String> = vec!["S-1-1-0".to_string()];
                    sid_texts.extend(roots.iter().map(|root| workspace_write_sid(root)));
                    for text in &sid_texts {
                        let wide: Vec<u16> =
                            text.encode_utf16().chain(std::iter::once(0)).collect();
                        let mut sid: *mut core::ffi::c_void = ptr::null_mut();
                        if ConvertStringSidToSidW(wide.as_ptr(), &mut sid) == 0 {
                            return Err(std::io::Error::last_os_error());
                        }
                        sid_guards.push(LocalFreeGuard(sid));
                        restricting.push(SID_AND_ATTRIBUTES {
                            Sid: sid,
                            Attributes: 0,
                        });
                    }

                    let mut restricted: HANDLE = ptr::null_mut();
                    // Derived from our *own* token, not from the privilege-stripped duplicate.
                    //
                    // `CreateProcessAsUserW` normally needs `SE_ASSIGNPRIMARYTOKEN` and
                    // `SE_INCREASE_QUOTA`, and waives them only when the token is recognisably a
                    // restricted version of the caller's. Deriving from the duplicate that step 2
                    // had already emptied of privileges broke that lineage: the kernel refused with
                    // "SE_INCREASE_QUOTA_NAME not held", fell through to
                    // `CreateProcessWithTokenW`, and that rejected the restricted token outright
                    // with ERROR_INVALID_PARAMETER. Both commands then failed to spawn at all.
                    // `DISABLE_MAX_PRIVILEGE` alongside `WRITE_RESTRICTED`.
                    //
                    // The restricted token is derived from `self_token`, which never went through
                    // the privilege strip that step 2 applies to the Low-integrity duplicate -- so
                    // without this the `read` child ran with zero privileges while the `workspace`
                    // child kept meka's entire set, and the stronger-sounding mode was the less
                    // hardened one. Restricting SIDs bound *write access checks*; they do nothing
                    // about privileges, which are checked separately and several of which exist
                    // precisely to bypass a DACL. Launched from an elevated shell the child kept
                    // `SeImpersonatePrivilege`, enabled by default, which is the whole "Potato"
                    // family and takes the boundary with it; unelevated it still kept
                    // `SeShutdownPrivilege`, so `shutdown /r` worked from inside a sandbox that
                    // promises the command cannot change the machine.
                    //
                    // This does not touch the "restricted version of the caller's token" lineage
                    // that `CreateProcessAsUserW`'s privilege waiver depends on -- the derivation
                    // is still from `self_token` -- so the fallback bug that cost a live debugging
                    // round should not return. It is verified on the box regardless.
                    if CreateRestrictedToken(
                        self_token.as_raw(),
                        WRITE_RESTRICTED | DISABLE_MAX_PRIVILEGE,
                        0,
                        ptr::null(),
                        0,
                        ptr::null(),
                        restricting.len() as u32,
                        restricting.as_ptr(),
                        &mut restricted,
                    ) == 0
                    {
                        return Err(std::io::Error::last_os_error());
                    }
                    OwnedHandle(restricted)
                }
            };

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
            let spawn_result = create_process_confined(
                low_token.as_raw(),
                &cwd_utf16,
                &cmd_line,
                &startup,
                &mut proc_info,
                // The restricted token cannot make a console, so it must inherit the one
                // `ensure_console` guaranteed rather than ask for a fresh one.
                !matches!(confinement, WindowsConfinement::WriteRestricted(_)),
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
    /// braces with the concurrent draining in the Windows spawn path:
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
    /// `argv[0]`) before failing, so re-using the same buffer between attempts could hand the
    /// fallback a corrupted string.
    ///
    /// Both invocations pass `EXTENDED_STARTUPINFO_PRESENT` together with `STARTUPINFOEXW`, so the
    /// handle-list filter in the attribute list applies uniformly across both paths.
    unsafe fn create_process_confined(
        token: HANDLE,
        cwd_utf16: &[u16],
        cmd_line_utf8: &str,
        startup: &STARTUPINFOEXW,
        proc_info: &mut PROCESS_INFORMATION,
        // `low_integrity`: true for the Low-integrity path, false for `WRITE_RESTRICTED`. It
        // governs two things that happen to share an answer, both measured rather than documented:
        // whether `CREATE_NO_WINDOW` may be asked for (a restricted child cannot *create* a
        // console, only inherit one), and whether the `CreateProcessWithTokenW` fallback below is
        // worth attempting at all.
        low_integrity: bool,
    ) -> std::io::Result<()> {
        // CREATE_SUSPENDED so the child sits at its entry point until we've assigned it to the Job
        // Object; otherwise the child could spawn a grandchild before assignment, and that
        // grandchild would never be bound to the job.
        // `CREATE_NO_WINDOW` asks for a *new* console, which a WRITE_RESTRICTED child cannot
        // create: it dies with STATUS_DLL_INIT_FAILED before `main` runs. Measured, not documented.
        // The restricted path inherits the console `ensure_console` guaranteed instead.
        let no_window = if low_integrity { CREATE_NO_WINDOW } else { 0 };
        let creation_flags = no_window
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
                cwd_utf16.as_ptr(),
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

        // The fallback is Low-integrity only, and saying so is the point.
        //
        // `CreateProcessWithTokenW` rejects a restricted token outright with
        // `ERROR_INVALID_PARAMETER` -- measured during the feasibility probe, and recorded at the
        // token-construction site. Running it anyway on the `workspace` path replaced the real
        // diagnosis (`ERROR_PRIVILEGE_NOT_HELD`: this account does not hold
        // `SE_INCREASE_QUOTA_NAME`) with "The parameter is incorrect. (os error 87)", and logged a
        // warning describing a fallback that could not have happened. A corp-managed account
        // switching to `workspace` got error 87 on every command and nothing pointing at why.
        if !low_integrity {
            return Err(std::io::Error::new(
                err.kind(),
                format!(
                    "{err}. `workspace` spawns through CreateProcessAsUserW, which needs \
                     SE_INCREASE_QUOTA_NAME; this account does not hold it. `unrestricted` \
                     avoids that path, or run meka from an account that holds it."
                ),
            ));
        }

        tracing::warn!(
            "CreateProcessAsUserW denied (SE_INCREASE_QUOTA_NAME not held); falling back to \
             CreateProcessWithTokenW, which still spawns the child at Low integrity with the \
             same scrubbed environment"
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
                cwd_utf16.as_ptr(),
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
            append_env_entry(&mut block, &name_os, &value_os);
        }
        // Double-NUL terminator (each entry already ends with one NUL; we need another to close the
        // block).
        block.push(0);
        block
    }

    /// Append one `NAME=VALUE\0` entry, in the environment block's own encoding.
    ///
    /// Takes `OsStr` and walks it with `encode_wide` rather than going via `&str`. The block is
    /// UTF-16 either way, so the old `to_str()` round-trip existed only to *discard* values the
    /// destination represents natively: a `PATH` or `TMP` holding an unpaired surrogate was
    /// dropped from the child's environment with no diagnostic, and a shell with no `PATH`
    /// resolves no commands.
    fn append_env_entry(block: &mut Vec<u16>, name: &std::ffi::OsStr, value: &std::ffi::OsStr) {
        block.extend(name.encode_wide());
        block.push(u16::from(b'='));
        block.extend(value.encode_wide());
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

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Read a directory's DACL and ask whether meka's write ACE is on it.
        ///
        /// Test-only, because production only ever asks this from inside `set_workspace_ace`, which
        /// already holds the descriptor it read.
        fn carries_workspace_ace(root: &std::path::Path) -> bool {
            let sid_text: Vec<u16> = workspace_write_sid(root)
                .encode_utf16()
                .chain(std::iter::once(0))
                .collect();
            let mut sid: *mut core::ffi::c_void = ptr::null_mut();
            assert_ne!(
                unsafe { ConvertStringSidToSidW(sid_text.as_ptr(), &mut sid) },
                0,
                "the capability SID must parse"
            );
            let _sid_guard = LocalFreeGuard(sid);
            let target: Vec<u16> = root
                .as_os_str()
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();
            let mut acl: *mut ACL = ptr::null_mut();
            let mut descriptor: *mut core::ffi::c_void = ptr::null_mut();
            let status = unsafe {
                GetNamedSecurityInfoW(
                    target.as_ptr(),
                    SE_FILE_OBJECT,
                    DACL_SECURITY_INFORMATION,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    &mut acl,
                    ptr::null_mut(),
                    &mut descriptor,
                )
            };
            assert_eq!(status, 0, "reading the DACL must succeed");
            let _descriptor_guard = LocalFreeGuard(descriptor);
            unsafe { has_workspace_ace(acl, sid) }
        }

        /// The short-circuit that keeps `ensure` from re-propagating the DACL on every command must
        /// answer both ways against a real ACL.
        ///
        /// Both failure directions are silent, which is why this is asserted rather than reasoned
        /// about. Always-false costs only time, so nothing would ever surface it: the measured
        /// 280ms-per-command would simply stay. Always-true is worse -- `ensure` would stop placing
        /// the ACE at all and every `workspace` write would be denied on a tree meka believed it
        /// had granted.
        ///
        /// The specific traps are the mask and the flags, and both were settled by dumping a real
        /// ACL rather than by reasoning: the mask comes back as the `GENERIC_WRITE | DELETE` that
        /// was written and is *not* mapped to `FILE_GENERIC_WRITE`, while the flags come back as
        /// `0xb`, carrying an `INHERIT_ONLY_ACE` bit nothing in meka asked for. A check written
        /// from the documentation alone would have tested for flag equality and never matched.
        #[test]
        fn the_workspace_ace_is_recognised_once_placed_and_not_before() {
            let workspace = tempfile::tempdir().expect("temp dir");
            let root = workspace.path();

            assert!(
                !carries_workspace_ace(root),
                "a fresh directory carries no workspace ACE"
            );

            unsafe { set_workspace_ace(root, true) }.expect("granting must succeed");
            assert!(
                carries_workspace_ace(root),
                "the ACE just placed must be recognised, or `ensure` re-propagates the whole tree \
                 on every command"
            );

            unsafe { set_workspace_ace(root, false) }.expect("revoking must succeed");
            assert!(
                !carries_workspace_ace(root),
                "a revoked ACE must read as absent, or `ensure` would skip re-placing a grant that \
                 another meka took back"
            );
        }

        /// The capability SID is stable for a path and distinct between paths.
        ///
        /// Stability is what lets a grant be recognised and revoked instead of accumulating a new
        /// ACE per run; distinctness is what stops two of the user's workspaces sharing one.
        #[test]
        fn the_workspace_sid_is_deterministic_and_per_path() {
            let a = std::path::Path::new(r"C:\Users\x\project");
            let b = std::path::Path::new(r"C:\Users\x\other");
            assert_eq!(workspace_write_sid(a), workspace_write_sid(a));
            assert_ne!(workspace_write_sid(a), workspace_write_sid(b));
            assert!(workspace_write_sid(a).starts_with("S-1-4-"));
            // Windows paths are case-insensitive, so two spellings of one directory must not mint
            // two identities and leave half the ACEs unrevoked.
            assert_eq!(
                workspace_write_sid(a),
                workspace_write_sid(std::path::Path::new(r"C:\USERS\X\PROJECT"))
            );
        }

        /// Every caller shares one ledger, so no teardown can revoke another's grant.
        ///
        /// The failure this guards is quiet: with a ledger per `ToolRegistry`, a sub-agent
        /// finishing its task dropped its registry and took the parent's ACEs with it,
        /// leaving a parent still at `workspace` unable to write to its own root.
        #[test]
        fn the_process_shares_one_grant_ledger() {
            let temp = tempfile::tempdir().expect("tempdir");
            let root = crate::workspace::canonical_for_test(temp.path());

            // A sub-agent's handle, taken and dropped the way its `ToolRegistry` is dropped the
            // moment its task finishes. Identity alone (`Arc::ptr_eq` on two calls) was the whole
            // test and could not fail: a `OnceLock` returns itself by construction, so it held
            // equally well for the per-registry ledger this exists to rule out.
            {
                let borrowed = std::sync::Arc::clone(process_grants());
                borrowed.ensure(&root).expect("grant");
            }
            assert!(
                icacls(&root).contains(&workspace_write_sid(&root)),
                "one holder going away must not take the grant with it"
            );

            // And the grant placed through that handle is the one the process-wide release takes
            // back, which is the other half of "one ledger".
            process_grants().revoke_all();
            assert!(
                !icacls(&root).contains(&workspace_write_sid(&root)),
                "the shared ledger must still own what was granted through a clone of it"
            );
        }

        /// A granted ACE is visible and a revoked one leaves nothing behind.
        #[test]
        fn a_workspace_grant_round_trips() {
            let temp = tempfile::tempdir().expect("tempdir");
            let root = crate::workspace::canonical_for_test(temp.path());
            let grants = WindowsGrants::default();

            grants.ensure(&root).expect("grant");
            let after_grant = icacls(&root);
            assert!(
                after_grant.contains(&workspace_write_sid(&root)),
                "the capability must appear in the DACL: {after_grant}"
            );

            grants.revoke_all();
            let after_revoke = icacls(&root);
            assert!(
                !after_revoke.contains(&workspace_write_sid(&root)),
                "revoking must leave no residue: {after_revoke}"
            );
        }

        fn icacls(path: &std::path::Path) -> String {
            let output = std::process::Command::new("icacls")
                .arg(path)
                .output()
                .expect("icacls");
            String::from_utf8_lossy(&output.stdout).into_owned()
        }

        /// Allocating a console must not disturb stdin/stdout.
        ///
        /// Under ACP meka speaks JSON-RPC over those handles, so a console allocation that
        /// redirected them would break the session silently, with no error anywhere.
        ///
        /// **What this covers and what it cannot.** It calls `allocate_hidden_console` rather than
        /// `ensure_console`, because a `cargo test` process always has a console and
        /// `ensure_console` returns early: the previous version of this test never reached
        /// the code it was named after. Calling the inner function does run the
        /// snapshot-and-restore, but `AllocConsole` itself fails with `ERROR_ACCESS_DENIED`
        /// in a process that already has one, so the branch where the allocation *succeeds*
        /// and rebinds the handles is still not reachable from a test binary. That branch
        /// is verified by running meka headless on Windows hardware, which is where the
        /// feasibility probe measured it in the first place.
        #[test]
        fn allocating_a_console_leaves_the_standard_handles_alone() {
            use windows_sys::Win32::System::Console::{
                GetStdHandle, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
            };
            let (before_in, before_out) = unsafe {
                (
                    GetStdHandle(STD_INPUT_HANDLE),
                    GetStdHandle(STD_OUTPUT_HANDLE),
                )
            };
            // Safe: the body is a snapshot-and-restore around one `AllocConsole`, and a test
            // process already holds a console, so that call is a no-op here.
            unsafe { allocate_hidden_console() };
            let (after_in, after_out) = unsafe {
                (
                    GetStdHandle(STD_INPUT_HANDLE),
                    GetStdHandle(STD_OUTPUT_HANDLE),
                )
            };
            assert_eq!(before_in, after_in, "stdin was rebound");
            assert_eq!(before_out, after_out, "stdout was rebound");
        }
    }
}

#[cfg(test)]
mod tests {

    /// The Landlock masks are pinned per ABI, bit for bit.
    ///
    /// These two functions are the whole of what meka asks the kernel to police, and they were the
    /// least guarded code in the module: a mutation sweep flipped 34 operators across them without
    /// a single test noticing. The end-to-end boundary test only catches the subset that stops a
    /// *write*, and it cannot catch the rest, because `&` binds tighter than `|` in Rust: flipping
    /// a middle operator drops only the two adjacent bits and leaves the write flags standing.
    /// Losing `MAKE_SOCK` and `MAKE_FIFO` that way would let a confined shell create a socket or a
    /// fifo outside its roots while every existing assertion still passed.
    ///
    /// Spelled as literal sums rather than by rebuilding the expression, so the test states the
    /// intended mask instead of restating the code. The bit values are Landlock's, fixed by its
    /// ABI, and the per-version gating is the fact worth pinning: v2 adds `REFER`, v3 `TRUNCATE`,
    /// v5 `IOCTL_DEV`, v9 `RESOLVE_UNIX`, and v4 adds only network flags, which is why nothing
    /// changes there. Scoping arrives whole at v6 and must stay zero below it, because an unknown
    /// `scoped` bit makes `landlock_create_ruleset` fail with `EINVAL` and takes the sandbox down
    /// with it.
    #[test]
    #[cfg(target_os = "linux")]
    fn the_landlock_masks_are_exactly_what_each_abi_supports() {
        // The thirteen filesystem rights every supported ABI handles: bits 0..=12.
        const BASE: u64 = (1 << 13) - 1;
        const REFER: u64 = 1 << 13;
        const TRUNCATE: u64 = 1 << 14;
        const IOCTL_DEV: u64 = 1 << 15;
        const RESOLVE_UNIX: u64 = 1 << 16;

        for (abi, expected) in [
            (1, BASE),
            (2, BASE | REFER),
            (3, BASE | REFER | TRUNCATE),
            // v4 is network-only, so the filesystem mask is unchanged from v3.
            (4, BASE | REFER | TRUNCATE),
            (5, BASE | REFER | TRUNCATE | IOCTL_DEV),
            (8, BASE | REFER | TRUNCATE | IOCTL_DEV),
            (9, BASE | REFER | TRUNCATE | IOCTL_DEV | RESOLVE_UNIX),
        ] {
            assert_eq!(
                handled_access_for_abi(abi),
                expected,
                "ABI {abi} handled-access mask: got {:#018b}, want {expected:#018b}",
                handled_access_for_abi(abi)
            );
        }

        const ABSTRACT_UNIX_SOCKET: u64 = 1 << 0;
        const SIGNAL: u64 = 1 << 1;
        for abi in 1..=5 {
            assert_eq!(
                scoped_for_abi(abi),
                0,
                "scoping arrived in v6; setting a bit below it fails ruleset creation outright"
            );
        }
        for abi in 6..=9 {
            assert_eq!(
                scoped_for_abi(abi),
                ABSTRACT_UNIX_SOCKET | SIGNAL,
                "v6 and up must scope both the abstract socket namespace and signals"
            );
        }
    }

    /// A `bwrap` the user can replace is not a sandbox, and must be refused.
    ///
    /// `bwrap_on_path` walked `$PATH` and took the first executable named `bwrap`. Every ordinary
    /// desktop has several user-writable directories ahead of `/usr/bin` -- `~/.local/bin`, a cargo
    /// or go bin dir, a toolchain shim dir -- so a six-line script that `exec`s its final argument
    /// turned every `read` and `workspace` shell command into an unconfined one. The smoke test
    /// could not catch it, because `bwrap <flags> /bin/true` is exactly what such a shim satisfies.
    ///
    /// Asserted on the predicate rather than by planting a real shim, because the interesting half
    /// (a root-owned binary in a root-owned directory) cannot be constructed in a test without
    /// root. `/usr/bin` stands in for it, and the temp dir stands in for `~/.local/bin`.
    #[test]
    #[cfg(target_os = "linux")]
    fn a_bwrap_in_a_user_writable_directory_is_not_trusted() {
        let temp = tempfile::tempdir().expect("tempdir");
        assert!(
            !super::only_root_can_write(temp.path()),
            "a directory this process just created is writable by this user, so anything in it \
             could be swapped for a shim"
        );

        // The control, and the reason this is a predicate rather than a hardcoded path list: a
        // distribution that ships `bwrap` somewhere unusual still works as long as root owns it.
        let system = std::path::Path::new("/usr/bin");
        if system.is_dir() {
            assert!(
                super::only_root_can_write(system),
                "/usr/bin must be trusted, or bubblewrap is unreachable on an ordinary host"
            );
        }
    }

    /// Trust needs *both* the binary and its directory, and the mixed cases are the whole point.
    ///
    /// Asserting the two ends only (both trusted, neither trusted) leaves the conjunction free:
    /// `&&` and `||` agree whenever their operands agree, so a mutation to `||` survived every test
    /// in the suite. `||` is precisely the bug this check was added to fix, since it re-admits a
    /// root-owned binary sitting in a directory the user can write.
    ///
    /// The mixed pairs are built from paths that exist on any Linux host rather than by planting
    /// files, because the trusted half cannot be created without root.
    #[test]
    #[cfg(target_os = "linux")]
    fn trust_requires_the_binary_and_its_directory_together() {
        let temp = tempfile::tempdir().expect("tempdir");
        let mine = temp.path();
        let system = std::path::Path::new("/usr/bin");
        if !system.is_dir() {
            return;
        }

        assert!(
            super::trusted_to_confine(system, system),
            "a root-owned binary in a root-owned directory is the case that must be admitted"
        );
        assert!(
            !super::trusted_to_confine(mine, mine),
            "neither half trusted must be refused"
        );
        assert!(
            !super::trusted_to_confine(system, mine),
            "a root-owned binary in a user-writable directory is one `mv` from being the user's"
        );
        assert!(
            !super::trusted_to_confine(mine, system),
            "a user-writable binary is not redeemed by the directory around it"
        );
    }

    /// Only an executable regular file is a candidate.
    ///
    /// Both halves were mutable: dropping the `is_file` conjunct admits a directory named `bwrap`,
    /// and flipping the mode test to `== 0` admits only *non*-executable files, which quietly makes
    /// bubblewrap undiscoverable on every host and silently downgrades the backend.
    #[test]
    #[cfg(target_os = "linux")]
    fn only_an_executable_regular_file_is_a_bwrap_candidate() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().expect("tempdir");
        let plain = temp.path().join("plain");
        std::fs::write(&plain, "not executable").expect("write");
        let runnable = temp.path().join("runnable");
        std::fs::write(&runnable, "#!/bin/sh\n").expect("write");
        std::fs::set_permissions(&runnable, std::fs::Permissions::from_mode(0o755)).expect("chmod");

        let meta = |path: &std::path::Path| std::fs::metadata(path).expect("metadata");
        assert!(super::is_executable_file(&meta(&runnable)));
        assert!(
            !super::is_executable_file(&meta(&plain)),
            "a file with no execute bit cannot be the sandbox helper"
        );
        assert!(
            !super::is_executable_file(&meta(temp.path())),
            "a directory named `bwrap` is not a binary"
        );
    }
    /// Between ABI 3 and 9 the filesystem is genuinely write-protected but the later mitigations
    /// are absent, and the warning naming them is the only way a user learns which.
    ///
    /// Nothing asserted it: delete the whole block and every suite stayed green, leaving a host
    /// that believes read mode restricts more than the running kernel actually does. Driven through
    /// a subscriber pinned to `WARN` because that is the default floor, so this also fails if the
    /// level is dropped to `info` where `-v` would be needed to see it.
    ///
    /// Linux-gated because [`super::SandboxCapability::Landlock`] is: without this the test breaks
    /// the macOS and Windows halves of CI's lint and test matrix, which is exactly the
    /// platform-only compile error that matrix exists to catch and that cannot be reproduced on a
    /// Linux workstation.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_landlock_abi_below_9_names_the_mitigations_it_does_not_provide() {
        use std::sync::{Arc, Mutex};

        #[derive(Clone)]
        struct Capture(Arc<Mutex<Vec<u8>>>);

        impl std::io::Write for Capture {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .extend_from_slice(buf);
                Ok(buf.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Capture {
            type Writer = Self;

            fn make_writer(&'a self) -> Self::Writer {
                self.clone()
            }
        }

        // v3 clears the floor, so this is the "protected, but not fully" band the warning owns.
        for (abi, expected) in [
            (3, vec![
                "device ioctls",
                "abstract Unix sockets",
                "pathname Unix sockets",
            ]),
            (6, vec!["pathname Unix sockets"]),
        ] {
            let capture = Capture(Arc::new(Mutex::new(Vec::new())));
            let buffer = Arc::clone(&capture.0);
            let subscriber = tracing_subscriber::fmt()
                .with_writer(capture)
                .with_max_level(tracing::Level::WARN)
                .finish();

            let state = super::SandboxState {
                enabled: true,
                backend: crate::config::SandboxBackend::Landlock,
                auto_resolved: false,
                probe: super::BackendProbe::Ok(super::SandboxCapability::Landlock {
                    abi_version: abi,
                }),
            };
            tracing::subscriber::with_default(subscriber, || {
                super::warn_if_sandbox_issues(&state, super::WarnContext::Startup);
            });

            let logged = String::from_utf8(
                buffer
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone(),
            )
            .expect("log output is utf-8");
            for gap in expected {
                assert!(
                    logged.contains(gap),
                    "ABI v{abi} must name '{gap}' as unrestricted: {logged:?}"
                );
            }
        }

        // v9 has them all, so there is nothing to warn about and a warning would be noise.
        let capture = Capture(Arc::new(Mutex::new(Vec::new())));
        let buffer = Arc::clone(&capture.0);
        let subscriber = tracing_subscriber::fmt()
            .with_writer(capture)
            .with_max_level(tracing::Level::WARN)
            .finish();
        let state = super::SandboxState {
            enabled: true,
            backend: crate::config::SandboxBackend::Landlock,
            auto_resolved: false,
            probe: super::BackendProbe::Ok(super::SandboxCapability::Landlock { abi_version: 9 }),
        };
        tracing::subscriber::with_default(subscriber, || {
            super::warn_if_sandbox_issues(&state, super::WarnContext::Startup);
        });
        let logged = String::from_utf8(
            buffer
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone(),
        )
        .expect("log output is utf-8");
        assert!(
            !logged.contains("does not restrict"),
            "v9 restricts all of them; warning anyway trains the user to ignore it: {logged:?}"
        );
    }

    /// A child that cannot see the machine's proxy or CA configuration reaches nothing, and says so
    /// in terms that name none of the cause. For an MCP server that means it connects, registers
    /// its tools, and then fails every call. Neither kind of variable grants authority: they say
    /// where to go and whom to trust, and the child was making the request either way.
    ///
    /// The three families below are refused on purpose, so a widening of the list has to argue with
    /// this test rather than slip past it.
    #[cfg(unix)]
    #[test]
    fn network_configuration_reaches_a_sandboxed_child_but_credentials_do_not() {
        for allowed in [
            "HTTPS_PROXY",
            "https_proxy",
            "NO_PROXY",
            "SSL_CERT_FILE",
            "NODE_EXTRA_CA_CERTS",
        ] {
            assert!(
                super::keep_sandbox_env_var(allowed),
                "{allowed} is how the child reaches the network",
            );
        }
        for refused in [
            "SSH_AUTH_SOCK",
            "NODE_OPTIONS",
            "PYTHONPATH",
            "NODE_PATH",
            "VIRTUAL_ENV",
            "AWS_SECRET_ACCESS_KEY",
        ] {
            assert!(
                !super::keep_sandbox_env_var(refused),
                "{refused} carries either a credential or a way to run code",
            );
        }
    }

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

    /// The seatbelt profile names each root as a `-D` parameter, never inside the profile text.
    ///
    /// This is the whole reason roots travel out of band: a directory called `it's "here"` or one
    /// ending in a backslash would otherwise have to survive SBPL string quoting, and getting that
    /// wrong does not fail loudly -- it changes which subpath the rule matches. The assertion that
    /// no root's text appears in the profile body is the one that would catch a future rewrite
    /// deciding interpolation is simpler.
    #[test]
    fn the_seatbelt_profile_passes_roots_as_parameters_not_profile_text() {
        let roots = vec![
            std::path::PathBuf::from(r#"/tmp/it's "quoted""#),
            std::path::PathBuf::from(r"/tmp/trailing\"),
        ];
        let (profile, params) = sandbox_profile_for(&roots);

        assert_eq!(params, vec![
            std::ffi::OsString::from("-D"),
            std::ffi::OsString::from(r#"MEKA_WRITABLE_0=/tmp/it's "quoted""#),
            std::ffi::OsString::from("-D"),
            std::ffi::OsString::from(r"MEKA_WRITABLE_1=/tmp/trailing\"),
        ]);
        for (index, _) in roots.iter().enumerate() {
            assert!(
                profile.contains(&format!(r#"(subpath (param "MEKA_WRITABLE_{index}"))"#)),
                "root {index} must be referenced by parameter name: {profile}"
            );
        }
        for root in &roots {
            assert!(
                !profile.contains(&root.display().to_string()),
                "no root's text may appear in the profile body: {profile}"
            );
        }
        assert!(
            profile.starts_with(SANDBOX_PROFILE_READONLY),
            "the writable rules must be appended to the read-only base, not replace it"
        );
    }

    /// A root that is not valid UTF-8 is still made writable.
    ///
    /// The parameter is built as an `OsString` and the path pushed whole, so there is no UTF-8
    /// requirement to fail. A `to_str()` with a `continue` drops such a root silently from the
    /// allow-list and runs the command with it still read-only, a boundary quietly narrower than
    /// the one meka reported.
    #[test]
    #[cfg(unix)]
    fn a_non_utf8_root_still_becomes_a_seatbelt_parameter() {
        use std::{ffi::OsStr, os::unix::ffi::OsStrExt};

        let root = std::path::PathBuf::from(OsStr::from_bytes(b"/tmp/work\xff"));
        assert!(root.to_str().is_none(), "precondition: not valid UTF-8");

        let (profile, params) = sandbox_profile_for(std::slice::from_ref(&root));
        assert_eq!(params.len(), 2, "the root must not be dropped: {params:?}");

        let mut expected = std::ffi::OsString::from("MEKA_WRITABLE_0=");
        expected.push(root.as_os_str());
        assert_eq!(params[1], expected, "the path must survive byte-for-byte");
        assert!(profile.contains(r#"(param "MEKA_WRITABLE_0")"#));
    }

    #[test]
    fn test_wrap_command_with_utf8_output_prepends_prelude() {
        let wrapped = wrap_command_with_utf8_output("Write-Output '日本語'");
        assert!(wrapped.contains("[Console]::OutputEncoding="));
        assert!(wrapped.contains("$OutputEncoding=[System.Text.Encoding]::UTF8"));
        assert!(wrapped.ends_with("Write-Output '日本語'"));
        // A space must separate the prelude from the user command so PowerShell doesn't glue them
        // into one malformed statement.
        assert!(wrapped.contains("}; Write-Output"));
    }

    /// The prelude may not throw under ConstrainedLanguage, which is what a `WRITE_RESTRICTED`
    /// token puts PowerShell into.
    ///
    /// Both guards are asserted because either alone is thin: the language-mode test is what
    /// normally skips the encoding switch, and the `try`/`catch` is what stops a host that
    /// constrains something else from printing an error ahead of every command the agent runs. The
    /// symptom is not a failed command but a successful one that looks failed, which the model then
    /// reports to the user as an error.
    #[test]
    fn the_utf8_prelude_cannot_throw_under_constrained_language() {
        let wrapped = wrap_command_with_utf8_output("Get-Date");
        assert!(
            wrapped.contains("LanguageMode -eq 'FullLanguage'"),
            "the encoding switch must be skipped outright when the language mode forbids it: \
             {wrapped}"
        );
        assert!(
            wrapped.contains("try{") && wrapped.contains("}catch{}"),
            "and must still be caught if it runs and fails anyway: {wrapped}"
        );
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

    /// Below ABI v3 the ruleset does not handle `LANDLOCK_ACCESS_FS_TRUNCATE`, so a sandboxed child
    /// can still empty an existing file even though every open-for-write is denied. meka documents
    /// read mode as write-protecting the filesystem, so the only honest answer on such a kernel is
    /// to report the backend unusable and let the shell tool hard-error, rather than to sandbox
    /// with a ruleset that does not enforce what was promised.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_landlock_abi_below_the_truncate_floor_is_reported_unusable() {
        for abi in [1, 2] {
            let probe = landlock_probe_from_abi(Some(abi));
            let reason = backend_unavailable_reason(&probe)
                .unwrap_or_else(|| panic!("ABI v{} must not be accepted", abi));
            assert!(
                reason.contains("truncate(2)"),
                "the reason must name what is unenforced, got: {}",
                reason
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_landlock_abi_at_or_above_the_floor_is_accepted() {
        for abi in [MIN_LANDLOCK_ABI, 6, 9] {
            assert!(
                matches!(
                    landlock_probe_from_abi(Some(abi)),
                    BackendProbe::Ok(SandboxCapability::Landlock { abi_version }) if abi_version == abi
                ),
                "ABI v{} should be accepted",
                abi
            );
        }
    }

    /// A kernel with no Landlock at all and one whose Landlock is too old are both unusable, but a
    /// user can only act on the difference: the first needs a newer kernel or Bubblewrap, the
    /// second is specifically about `truncate(2)`. Keep the two messages distinct.
    #[cfg(target_os = "linux")]
    #[test]
    fn absent_landlock_and_too_old_landlock_report_different_reasons() {
        let absent = backend_unavailable_reason(&landlock_probe_from_abi(None))
            .expect("no Landlock must be unusable");
        let too_old = backend_unavailable_reason(&landlock_probe_from_abi(Some(1)))
            .expect("ABI v1 must be unusable");
        assert_ne!(absent, too_old);
        assert!(absent.contains("5.13"), "got: {}", absent);
        assert!(too_old.contains("6.2"), "got: {}", too_old);
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
