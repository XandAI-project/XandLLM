/*!
 * build.rs — Build script for xandllm-core
 *
 * When the `cuda_engine` feature is enabled this script:
 *  1. Compiles the C sources (gguf.c, engine.c) into a static library
 *     `libxandengine_c.a` via the `cc` crate.
 *  2. Compiles the CUDA sources (quant.cu, gemma3.cu) into a static library
 *     `libxandengine_cuda.a` by invoking nvcc directly.
 *  3. Emits the necessary `cargo:rustc-link-lib` directives for cublas and
 *     cudart so that the final Rust binary links against them.
 *
 * When `cuda_engine` is NOT active the script does nothing — the candle-based
 * QuantizedModel path remains the only backend.
 *
 * Environment variables used:
 *  CUDA_PATH   — root of the CUDA toolkit (e.g. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4)
 *                Defaults to /usr/local/cuda on Linux/macOS.
 *  NVCC_ARCH   — GPU compute capability passed to nvcc (default: sm_75 which
 *                covers Turing and later — RTX 20xx/30xx/40xx).
 */

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Only build the native engine when the feature is active.
    if env::var("CARGO_FEATURE_CUDA_ENGINE").is_err() {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let csrc = manifest_dir.join("csrc");

    // ── CUDA toolkit paths ──────────────────────────────────────────────────
    let cuda_path = cuda_toolkit_path();
    let cuda_include = cuda_path.join("include");
    let cuda_lib = if cfg!(target_os = "windows") {
        cuda_path.join("lib").join("x64")
    } else {
        cuda_path.join("lib64")
    };

    let nvcc = nvcc_path(&cuda_path);
    let sm_arch = env::var("NVCC_ARCH").unwrap_or_else(|_| "sm_75".to_string());

    // ── Step 1: compile C files ─────────────────────────────────────────────
    cc::Build::new()
        .files([
            csrc.join("gguf.c"),
            csrc.join("engine.c"),
        ])
        .include(&csrc)
        .include(&cuda_include)  // for engine.c which includes gemma3.h → cuda_runtime.h
        .opt_level(3)
        .warnings(false)
        .compile("xandengine_c");

    // ── Step 2: compile CUDA files via nvcc ─────────────────────────────────
    let cuda_objs = compile_cuda_sources(
        &nvcc,
        &csrc,
        &cuda_include,
        &sm_arch,
        &out_dir,
    );

    // Archive CUDA objects into a static lib.
    // On Windows (MSVC) the convention is .lib; on Unix it is .a.
    let cuda_lib_name = if cfg!(target_os = "windows") {
        "xandengine_cuda.lib"
    } else {
        "libxandengine_cuda.a"
    };
    let cuda_lib_path = out_dir.join(cuda_lib_name);
    create_static_lib(&out_dir, &cuda_objs, &cuda_lib_path);

    // ── Step 3: link directives ─────────────────────────────────────────────
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=xandengine_cuda");
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cudart");
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // Re-run if any C/CUDA source changes
    println!("cargo:rerun-if-changed=csrc/gguf.h");
    println!("cargo:rerun-if-changed=csrc/gguf.c");
    println!("cargo:rerun-if-changed=csrc/quant.h");
    println!("cargo:rerun-if-changed=csrc/quant.cu");
    println!("cargo:rerun-if-changed=csrc/gemma3.h");
    println!("cargo:rerun-if-changed=csrc/gemma3.cu");
    println!("cargo:rerun-if-changed=csrc/engine.h");
    println!("cargo:rerun-if-changed=csrc/engine.c");
    println!("cargo:rerun-if-changed=build.rs");
}

/// Locate the CUDA toolkit root directory.
fn cuda_toolkit_path() -> PathBuf {
    if let Ok(p) = env::var("CUDA_PATH") {
        return PathBuf::from(p);
    }
    // Common Windows install path — scan from newest to oldest
    if cfg!(target_os = "windows") {
        for ver in ["v12.6", "v12.5", "v12.4", "v12.3", "v12.2", "v12.1", "v12.0", "v11.8"] {
            let p = PathBuf::from(format!(
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{ver}"
            ));
            if p.exists() { return p; }
        }
    }
    // Linux / macOS default
    PathBuf::from("/usr/local/cuda")
}

/// Find the nvcc binary inside a CUDA toolkit root.
fn nvcc_path(cuda_root: &Path) -> PathBuf {
    let bin = if cfg!(target_os = "windows") {
        cuda_root.join("bin").join("nvcc.exe")
    } else {
        cuda_root.join("bin").join("nvcc")
    };
    if bin.exists() {
        return bin;
    }
    // Fall back to PATH lookup
    PathBuf::from("nvcc")
}

/// Find the directory containing cl.exe using the `cc` crate's compiler
/// detection, which already knows how to locate MSVC via vswhere / VS env vars.
/// Returns None on non-Windows or when not found.
fn find_cl_dir() -> Option<PathBuf> {
    if !cfg!(target_os = "windows") {
        return None;
    }
    let compiler = cc::Build::new()
        .opt_level(0)
        .host("x86_64-pc-windows-msvc")
        .target("x86_64-pc-windows-msvc")
        .get_compiler();
    // compiler.path() is the full path to cl.exe; take its parent directory.
    compiler.path().parent().map(|p| p.to_path_buf())
}

/// Compile each .cu file to a .o object and return the list of object paths.
fn compile_cuda_sources(
    nvcc: &Path,
    csrc: &Path,
    cuda_include: &Path,
    sm_arch: &str,
    out_dir: &Path,
) -> Vec<PathBuf> {
    let sources = ["quant.cu", "gemma3.cu"];
    let mut objs = Vec::new();

    // On Windows, tell nvcc exactly where cl.exe lives via -ccbin so that the
    // build succeeds even when cargo is not launched from a VS Developer Prompt.
    let ccbin_arg: Option<String> = find_cl_dir().map(|dir| dir.display().to_string());

    for src in &sources {
        let src_path = csrc.join(src);
        let stem = Path::new(src).file_stem().unwrap().to_str().unwrap();
        let obj_path = out_dir.join(format!("{stem}.obj"));

        let mut cmd = Command::new(nvcc);

        // Point nvcc at the MSVC host compiler when running on Windows.
        if let Some(ref ccbin) = ccbin_arg {
            cmd.args(["-ccbin", ccbin.as_str()]);
        }

        cmd.args([
            "-O3",
            "--use_fast_math",
            &format!("-arch={sm_arch}"),
            // Generate relocatable device code so we can link multiple .cu files.
            "--device-c",
            "-I",
        ]);
        cmd.arg(csrc);
        cmd.arg("-I");
        cmd.arg(cuda_include);
        cmd.args(["-c", "-o"]);
        cmd.arg(&obj_path);
        cmd.arg(&src_path);

        let status = cmd
            .status()
            .unwrap_or_else(|e| panic!("Failed to run nvcc for {src}: {e}"));

        if !status.success() {
            panic!("nvcc compilation of {src} failed with exit code {status}");
        }

        objs.push(obj_path);
    }

    // Device-link step: combine relocatable device code objects.
    let linked_obj = out_dir.join("xandengine_cuda_dlink.obj");
    let mut dlink_cmd = Command::new(nvcc);

    if let Some(ref ccbin) = ccbin_arg {
        dlink_cmd.args(["-ccbin", ccbin.as_str()]);
    }

    dlink_cmd.arg(&format!("-arch={sm_arch}"));
    dlink_cmd.arg("--device-link");
    dlink_cmd.arg("-o");
    dlink_cmd.arg(&linked_obj);
    for obj in &objs {
        dlink_cmd.arg(obj);
    }
    let status = dlink_cmd
        .status()
        .unwrap_or_else(|e| panic!("nvcc device-link failed: {e}"));
    if !status.success() {
        panic!("nvcc device-link failed with exit code {status}");
    }
    objs.push(linked_obj);
    objs
}

/// Package a list of .o/.obj files into a static library using `ar`
/// (Linux/macOS) or `lib.exe` (Windows).
fn create_static_lib(out_dir: &Path, objs: &[PathBuf], lib_path: &Path) {
    if cfg!(target_os = "windows") {
        // Locate lib.exe next to cl.exe via the cc crate's MSVC detection.
        // Fall back to searching PATH if detection fails.
        let lib_exe = find_cl_dir()
            .map(|dir| dir.join("lib.exe"))
            .filter(|p| p.exists())
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "lib.exe".to_string());

        let mut cmd = Command::new(&lib_exe);
        cmd.arg(format!("/OUT:{}", lib_path.display()));
        for obj in objs {
            cmd.arg(obj);
        }
        let status = cmd
            .status()
            .unwrap_or_else(|e| panic!("lib.exe failed: {e}"));
        if !status.success() {
            panic!("lib.exe failed with exit code {status}");
        }
    } else {
        let mut cmd = Command::new("ar");
        cmd.arg("crs");
        cmd.arg(lib_path);
        for obj in objs {
            cmd.arg(obj);
        }
        let _ = out_dir; // silence unused warning
        let status = cmd.status().unwrap_or_else(|e| panic!("ar failed: {e}"));
        if !status.success() {
            panic!("ar failed with exit code {status}");
        }
    }
}
