/// Claude Code version string. Single source of truth; also emitted as
/// `CC_VERSION` for use via `env!("CC_VERSION")` in the main crate.
const CC_VERSION: &str = "2.1.241";

/// Returns rather than `expect`s, because `[lints.clippy] expect_used` reaches the build script
/// too: the Windows lint job compiles this file, and the panicking call was a hard error there
/// while every other platform cfg'd the block away and never saw it.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rustc-env=CC_VERSION={}", CC_VERSION);
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(windows)]
    {
        let mut resource = winres::WindowsResource::new();
        resource.set_icon("assets/meka.ico");
        resource.compile()?;
    }
    Ok(())
}
