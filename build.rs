// Link macOS Accelerate framework when the accelerate feature is enabled.
// Required for CBLAS symbols (e.g. cblas_sdot, cblas_sgemv) used by ndarray with blas-src/accelerate.
fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let has_accelerate = std::env::var("CARGO_CFG_FEATURE_ACCELERATE").is_ok();

    if target_os == "macos" && has_accelerate {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
