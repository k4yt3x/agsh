# Installation

meka is written in Rust and builds as a single binary.

## Pre-Built Binaries

Download the latest release for your platform from the [GitHub Releases](https://github.com/k4yt3x/meka/releases/latest) page.

| Platform | Archive |
|----------|---------|
| Linux (x86_64) | `meka-linux-amd64.tar.gz` |
| macOS (Apple Silicon) | `meka-macos-arm64.tar.gz` |
| Windows (x86_64) | `meka-windows-amd64.zip` |

Extract the binary and place it somewhere on your `$PATH`:

```bash
# Linux/macOS
tar -xzf meka-*.tar.gz
cp meka ~/.local/bin/
```

## Container

Every tagged release publishes an image to the GitHub Container Registry:

```bash
docker run --rm -it ghcr.io/k4yt3x/meka:latest --help
```

The image carries the binary and nothing else, so a session inside it starts with no config and no
database. Mount both to reach an existing setup:

```bash
docker run --rm -it \
    -v ~/.config/meka:/root/.config/meka:ro \
    -v ~/.local/share/meka:/root/.local/share/meka:rw \
    ghcr.io/k4yt3x/meka:latest
```

The data directory has to be writable: it holds `meka.db`, which is where sessions and every
credential live.

### `mekabox`

[`contrib/container/mekabox`](https://github.com/k4yt3x/meka/blob/master/contrib/container/mekabox)
does that mounting for you, against a stock `archlinux:latest` with your *host* binary bind-mounted
in, and starts the agent at `unrestricted` with instructions saying it may install whatever the task
needs. It is the answer to "let it do anything, just not to my machine": the container is disposable
and the host config is mounted read-only. It picks podman over docker when both are present.

## Cargo Install

If you have [Rust](https://www.rust-lang.org/tools/install) installed, you can install meka directly from the Git repository:

```bash
cargo install --locked --git https://github.com/k4yt3x/meka.git
```

This builds the latest version from source and installs it to `~/.cargo/bin/`.

## Building from Source

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) 1.95 or newer, the version `Cargo.toml` declares
- A C compiler (for the bundled SQLite)

### Build

```bash
git clone https://github.com/k4yt3x/meka.git
cd meka
cargo build --release
```

The binary will be at `target/release/meka`. Copy it somewhere on your `$PATH`:

```bash
cp target/release/meka ~/.local/bin/
```

## Verify

```bash
meka --version
meka --help
```
