# Environment Variables

The [config file](./config-file.md) is the recommended way to configure meka. Environment variables are useful for operational overrides; for example, in CI pipelines, containers, or to isolate a per-project config and data directory.

These operational variables override config file values but are overridden by CLI flags.

> **Provider configuration is not configurable via the environment.** Provider selection, model, and base URL come from the [config file](./config-file.md) (with per-run `--provider` / `--model` / `--base-url` flags); secrets come from the database via [`meka provider`](./config-file.md#meka-provider-cli). There are no provider env vars. This is deliberate: an ambient `OPENAI_API_KEY` or `MEKA_PROVIDER` left in the environment must never silently rebind which account a named profile bills.

## meka-Specific Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MEKA_PERMISSION` | Default permission mode | `none`, `read`, `write` |
| `MEKA_INSTRUCTIONS` | Standing instructions as a string, overriding the `instructions.md` file. Equivalent to `--instructions`. Used by the `mekabox` container wrapper, which mounts the config directory read-only and so cannot supply a file. | `Be terse.` |
| `MEKA_INSTRUCTIONS_FILE` | Standing instructions read from this path (a file, or a directory of `*.md`). For a file you did not choose the location of, such as a Kubernetes ConfigMap. Conflicts with `MEKA_INSTRUCTIONS`. | `/run/secrets/meka-instructions` |
| `MEKA_CONFIG_DIR` | Override the default config directory. Points at the `meka` directory itself (contains `config.toml` and `skills/`). The only isolation knob that works on every platform: `dirs::config_dir()` ignores `$XDG_CONFIG_HOME` on macOS/Windows. | `/tmp/meka-test/meka` |
| `MEKA_DATA_DIR` | Override the default data directory (where `meka.db` lives). Same cross-platform escape hatch: `dirs::data_dir()` ignores `$XDG_DATA_HOME` on macOS/Windows. Useful for tests, portable installs, and per-project session isolation. | `/tmp/meka-test/data/meka` |
| `MEKA_SANDBOX_BACKEND` | Override `[shell].sandbox_backend` (Linux only). Pinning a value also suppresses the "install Bubblewrap" auto-resolve warning. Used by the `mekabox` wrapper to pin Landlock in the container without editing the read-only host config. | `landlock`, `bubblewrap` |
| `MEKA_RENDER_MODE` | Override `[display].render_mode`. Handy for CI / non-TTY runs that want plain or no output. | `syntect`, `termimad`, `raw`, `silent` |

## MCP Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MEKA_MCP_TOOL_TIMEOUT` | Per-call timeout for MCP tools, in milliseconds. Applies to every remote tool invocation; on timeout meka cancels the request and returns an error to the model. | `600000` (600s) |

## Logging

meka uses the `tracing` framework. The log level can be controlled with:

| Variable | Description | Example |
|----------|-------------|---------|
| `RUST_LOG` | Standard Rust log filter | `meka=debug`, `meka=trace` |

If `RUST_LOG` is not set, the verbosity flag (`-v`, `-vv`, `-vvv`) controls the level:

| Flag | Level |
|------|-------|
| (none) | `warn` |
| `-v` | `info` |
| `-vv` | `debug` |
| `-vvv` | `trace` |

Logs are written to stderr so they do not interfere with agent output.
