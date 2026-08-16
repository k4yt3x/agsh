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
