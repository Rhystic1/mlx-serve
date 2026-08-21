//! Which engines and subsystems this build contains.
//!
//! Two independent reductions of the full macOS build exist, and they cut in
//! opposite directions, so neither flag can stand in for the other:
//!
//!   `ios`       — the on-device iPhone engine. MLX safetensors ONLY; both
//!                 embedded GGUF engines (ds4, llama.cpp) are stubbed out.
//!   `gguf_only` — the Windows/Linux port. The exact inverse: llama.cpp is the
//!                 whole inference floor, and MLX, ds4, ANE and media
//!                 generation are all absent.
//!
//! Naming the CAPABILITY rather than testing the flag at each site is what
//! keeps that straight: `if (opts.ios)` at a ds4 import silently reads as "ds4
//! is off" even in the build where ds4 is the only thing that is on.
//!
//! See build.zig's `-Dgguf-only` for why the port drops MLX rather than porting
//! it: MLX has no Windows build, its CUDA backend is Linux-only, and ~600 call
//! sites go through `mlx_fast_metal_kernel`, which is Metal by construction.

const opts = @import("build_options");

/// ds4 (DeepSeek-V4-Flash GGUF). Its entire GPU backend is `lib/ds4/ds4_metal.m`
/// — Objective-C plus Metal shaders — so it cannot follow off Apple. There is no
/// CUDA path for it upstream.
pub const ds4_enabled = !opts.ios and !opts.gguf_only;

/// llama.cpp (generic GGUF). Deliberately NOT gated on `gguf_only`: that build
/// is defined by this engine being present, not absent.
pub const llama_enabled = !opts.ios;

/// MLX safetensors inference, plus everything built on it: the native
/// transformer, speculative decoding, the prefix cache, and all media
/// generation backends.
pub const mlx_enabled = !opts.gguf_only;

/// Apple Neural Engine prefill offload (`lib/ane/`, private framework).
pub const ane_enabled = !opts.ios and !opts.gguf_only;

/// Image / audio / video / 3D generation. Every backend is written against
/// MLX, so it stands or falls with it.
pub const media_gen_enabled = mlx_enabled;
