//! DeepSeek-V4-Flash native arch for builds without MLX (src/build_cfg.zig).
//!
//! `deepseek_v4.zig` is the largest MLX surface in the tree (124 distinct MLX
//! symbols, ~137 `metal_kernel` sites) and owns its decode state outside the
//! KVCache. The safetensors path is gone here; the GGUF DSV4-Flash path went
//! with the ds4 engine, whose only backend is Metal.
//!
//! Only one symbol crosses into the shared plumbing: the prefill sub-block
//! width, which the memory guard bills with (`server.dsv4PrefillMemoryNeeded`
//! reads the LIVE value rather than a constant, so it must exist).

/// Prefill sub-block width. Zero because no dsv4 prefill can run; the guard
/// that reads it is only reached for a dsv4 model, which cannot load here.
pub fn prefillSub() u32 {
    return 0;
}
