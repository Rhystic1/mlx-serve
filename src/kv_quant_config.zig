//! KV-cache quantization CONFIG — the portable half of kv_quant.zig.
//!
//! Split out for the Windows/Linux port. `--kv-quant` is parsed into a
//! `KVQuantConfig` by the CLI before any backend exists, and `server.zig`
//! switches on `Scheme` when reporting configuration; neither needs the packed
//! read/write paths, which are 313 MLX references of quantize/dequantize and
//! fused Metal kernels.
//!
//! A GGUF-only build always leaves this `.dense`: llama.cpp has its own KV
//! cache types, selected through `--llama-kv-type-{k,v}` instead.

const std = @import("std");

/// KV-cache storage scheme.
///   * `off`      — dense bf16 (legacy).
///   * `affine`   — group-wise affine quant via `mlx_quantize`/`mlx_dequantize`.
///   * `turboquant_2` — Hadamard-rotated 2-bit affine quant. Each cache
///                     carries one `[head_dim, head_dim]` rotation matrix per
///                     K and V per layer (see `TurboState`). On write we
///                     compute `q = quantizeAffine(x @ H, group, 2)`; on read
///                     we recover `x ≈ dequantizeAffine(q, …) @ H` (Hadamard
///                     matrices are symmetric and self-inverse modulo a
///                     scalar, so `H = H^T = H^{-1}` after normalization).
///   * `turboquant_4` — same rotation idea at 4-bit. Useful when the bit
///                     budget can spare a couple of bits in exchange for
///                     reduced rotation overhead at the cost ceiling.
///                     (Compared to plain `affine` at 4-bit, TurboQuant 4
///                     spends a `[D,D]` matmul per layer per token; the
///                     rotation breaks the worst-case correlation patterns
///                     that hurt straight affine at long context.)
///
/// 1-bit TurboQuant from the Path B roadmap requires a custom 1-bit
/// pack/unpack — `mlx_quantize`/`mlx_dequantize` only support 2/4/8 bits in
/// mlx 0.31.2. Deferred to a follow-up that pairs with the fused-kernel work.
pub const Scheme = enum { off, affine, turboquant_2, turboquant_4 };

/// Configuration for the cache's storage backend. Stored on `KVCache.config`
/// and switched on at every read/write boundary.
pub const KVQuantConfig = struct {
    scheme: Scheme,
    /// Affine: 4 or 8. TurboQuant: 2 or 4. Ignored when `scheme == .off`.
    bits: u8,
    /// Affine group size — number of consecutive elements that share one
    /// scale+bias pair along the last axis. mlx-c convention is 64 for
    /// 4-bit and 8-bit weights; we match that.
    group_size: u32,

    pub const dense: KVQuantConfig = .{ .scheme = .off, .bits = 0, .group_size = 0 };

    pub fn affine(bits: u8) KVQuantConfig {
        std.debug.assert(bits == 4 or bits == 8);
        return .{ .scheme = .affine, .bits = bits, .group_size = 64 };
    }

    pub fn turboquant(bits: u8) KVQuantConfig {
        // Bits 2 and 4 ride mlx-c's native packing. We intentionally do NOT
        // accept 1 here — adding 1-bit requires a custom pack/unpack on top
        // of the rotation; see Scheme docstring.
        std.debug.assert(bits == 2 or bits == 4);
        return .{
            .scheme = if (bits == 2) .turboquant_2 else .turboquant_4,
            .bits = bits,
            .group_size = 64,
        };
    }

    pub fn isQuant(self: KVQuantConfig) bool {
        return self.scheme != .off;
    }
};
