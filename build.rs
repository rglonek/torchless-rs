// Link macOS Accelerate framework when the accelerate feature is enabled.
// Required for CBLAS symbols (e.g. cblas_sdot, cblas_sgemv) used by ndarray with blas-src/accelerate.
fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let has_accelerate = std::env::var("CARGO_CFG_FEATURE_ACCELERATE").is_ok();

    if target_os == "macos" && has_accelerate {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // Help find OpenCL on Windows when OPENCL_PATH is set (e.g. via vcpkg).
    // The ocl crate's cl-sys dependency may not find OpenCL.lib if it's installed
    // via vcpkg rather than the OCL SDK Light.
    if target_os == "windows" {
        println!("cargo:rerun-if-env-changed=OPENCL_PATH");
        if let Ok(opencl_path) = std::env::var("OPENCL_PATH") {
            println!("cargo:rustc-link-search=native={}/lib", opencl_path);
            println!("cargo:rustc-link-search=native={}\\lib", opencl_path);
        }
    }
}
