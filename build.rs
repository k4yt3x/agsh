/// Claude Code version string. Single source of truth; also emitted as
/// `CC_VERSION` for use via `env!("CC_VERSION")` in the main crate.
const CC_VERSION: &str = "2.1.241";

fn main() {
    println!("cargo:rustc-env=CC_VERSION={}", CC_VERSION);
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(windows)]
    {
        let mut resource = winres::WindowsResource::new();
        resource.set_icon("assets/meka.ico");
        resource.compile().expect("failed to compile resources");
    }
}
