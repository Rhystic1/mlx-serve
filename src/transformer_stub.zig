//! Transformer surface for builds without MLX (see src/build_cfg.zig).
//!
//! The real `transformer.zig` is 32k lines of MLX forward passes and ~376
//! `mlx_fast_metal_kernel` call sites. None of it can exist off Apple, and a
//! GGUF-only build never needs it: llama.cpp runs its own graph internally, and
//! `scheduler.zig` already skips every MLX per-slot structure for an embedded
//! engine (`is_embedded` -> zero KV layers, no SSM entries, no `legacy_gen`).
//!
//! What remains here is the shape the shared plumbing names: `LoadedModel`
//! holds a `?*Transformer` (always null in this build), the tuning globals are
//! read when the server prints its configuration, and the capability predicates
//! answer honestly so nothing believes a kernel is available.
//!
//! `Transformer.init` returns an error rather than a dummy instance: a build
//! that reaches it has routed a safetensors model into an engine that does not
//! exist, and that must surface as a named failure, not a null forward pass.

const std = @import("std");

pub const Error = error{MlxUnavailable};

/// Opaque: nothing in this build constructs one, and every field access is
/// behind a null check on `LoadedModel.transformer`.
pub const Transformer = struct {
    pub const MOE_EVAL_EVERY_N_LAYERS: usize = 4;

    /// Names of per-arch module-owned decode-state fields. Empty because no
    /// native architecture is compiled in; the spec-decode exclusion scan reads
    /// this list and correctly finds nothing to exclude.
    pub const module_owned_state_fields = [_][]const u8{};

    // Fields the shared plumbing names directly. All inert: `init` never
    // returns, so no instance exists to read them from. They are declared
    // rather than removed so `scheduler.zig` / `main.zig` compile unchanged --
    // the alternative is comptime-gating dozens of individual statements, which
    // is exactly the per-site branching build_cfg exists to avoid.
    s: @import("mlx_stub.zig").mlx_stream = .{},
    cache: KVCache = .{},
    /// The real ModelConfig BY VALUE (as in transformer.zig -- the Transformer
    /// holds a copy made at build time, which is why the prefill-chunk pin has
    /// to ride InitOptions rather than being set on the shared config).
    /// model.zig's config half is portable, so this is the genuine type.
    config: @import("model.zig").ModelConfig = .{},
    ssm_entries: []SSMCacheEntry = &.{},
    vision_embeddings: ?*anyopaque = null,
    /// Optional in the real Transformer (null = not a MoE checkpoint), and
    /// compared against null at its use sites.
    moe_layers: ?[]const u8 = null,
    dsv: ?*anyopaque = null,
    /// DeepSeek-V4-Flash module state (its decode state lives outside the
    /// KVCache). Always null -- no native arch is compiled in -- but it is
    /// field-accessed behind that null check, so it needs a named type.
    dsv4: ?*Dsv4State = null,
    /// ANE prefill offload state. The ANE is Apple hardware; `/props` reads
    /// this to decide whether to publish its `"ane"` object at all.
    ane_prefill: ?*@import("ane_stub.zig").AneEngine = null,

    pub fn init(_: std.Io, _: std.mem.Allocator, _: anytype, _: anytype) anyerror!Transformer {
        return Error.MlxUnavailable;
    }

    pub fn deinit(_: *Transformer) void {}

    pub fn nativeMoeMtpHeadMeasured(_: anytype) bool {
        return false;
    }

    /// No native arch is compiled in, so no arch owns decode state outside the
    /// KV cache and none can earn spec-decode rollback. Both answers are the
    /// conservative ones.
    pub fn ownsModuleDecodeState(_: *const Transformer) bool {
        return false;
    }

    pub fn moduleStateSpecRollback(_: *const Transformer) bool {
        return false;
    }

    pub fn resetCache(_: *Transformer) anyerror!void {
        return Error.MlxUnavailable;
    }
    pub fn warmup(_: *Transformer) anyerror!void {
        return Error.MlxUnavailable;
    }
    pub fn defaultCtx(_: *Transformer) ForwardCtx {
        return .{};
    }

    // Kernel pre-compilation: no kernels exist.
    pub fn compileForward(_: *Transformer) void {}
    pub fn compileGdnGate(_: *Transformer) void {}
    pub fn compileGeglu(_: *Transformer) void {}
    pub fn compileGelu(_: *Transformer) void {}
    pub fn compileMoeRouting(_: *Transformer) void {}
    pub fn compileSoftcap(_: *Transformer) void {}
    pub fn diagProjBench(_: *Transformer, _: usize, _: anytype) void {}
    pub fn buildAnePrefill(_: *Transformer, _: anytype, _: anytype, _: anytype, _: anytype) void {}

    pub fn supportsBatchedGdnDecode(_: *const Transformer) bool {
        return false;
    }
    pub fn batchedGdnReady(_: *const Transformer, _: anytype) bool {
        return false;
    }

    pub fn forwardWith(_: *Transformer, _: anytype, _: anytype) anyerror!@import("mlx_stub.zig").mlx_array {
        return Error.MlxUnavailable;
    }
    /// Returns ONE logits array per slot in the batch (not a single stacked
    /// array) -- the caller frees them individually.
    pub fn forwardBatchedDecode(_: *Transformer, _: anytype, _: anytype, _: anytype) anyerror![]@import("mlx_stub.zig").mlx_array {
        return Error.MlxUnavailable;
    }
    /// Returns ONE logits array per slot in the batch (not a single stacked
    /// array) -- the caller frees them individually.
    pub fn forwardMoeBatchedDecode(_: *Transformer, _: anytype, _: anytype, _: anytype) anyerror![]@import("mlx_stub.zig").mlx_array {
        return Error.MlxUnavailable;
    }
};

/// Sized (not opaque): `scheduler.Slot` embeds one by value. It is inert --
/// an embedded-engine slot already allocates zero KV layers (`is_embedded` in
/// scheduler.zig), so nothing ever grows or reads it.
pub const KVCache = struct {
    step: usize = 0,

    pub fn deinit(_: *KVCache) void {}

    /// Real signature is fallible (see the "fallible re-init behind a deinit"
    /// rule in CLAUDE.md -- it builds first, then swaps). Kept fallible so the
    /// call sites' `try` stays meaningful. Same zero-layer reasoning as
    /// `initWithConfigAndHeadDim`.
    pub fn reinit(self: *KVCache, layers: anytype, _: anytype, _: anytype) anyerror!void {
        if (@as(usize, @intCast(layers)) != 0) return Error.MlxUnavailable;
        self.* = .{};
    }

    /// The scheduler sizes a slot's cache from the arch's own K/V widths, and
    /// builds one for EVERY slot -- an embedded-engine slot (ds4 / llama.cpp)
    /// asks for zero layers because the engine owns its own KV.
    ///
    /// So this SUCCEEDS: a zero-layer cache is a shell that allocates nothing,
    /// and there is no MLX for it to be missing. Refusing here made every GGUF
    /// chat request fail with `MlxUnavailable` after tokenizing fine, which is
    /// the one path this build exists to serve. A non-zero request cannot
    /// happen: no MLX model can load here.
    pub fn initWithConfigAndHeadDim(_: std.mem.Allocator, layers: anytype, _: anytype, _: anytype) anyerror!KVCache {
        if (@as(usize, @intCast(layers)) != 0) return Error.MlxUnavailable;
        return .{};
    }
};
/// The REAL config type, from src/kv_quant_config.zig — it is a plain settings
/// struct with no MLX in it, so there is no reason for the stub to carry a
/// hand-copied twin that could drift from what the CLI parses.
pub const KVQuantScheme = @import("kv_quant_config.zig").Scheme;
pub const KVQuantConfig = @import("kv_quant_config.zig").KVQuantConfig;
/// Per-forward context. Field NAMES mirror the real one because the scheduler
/// and the profiler set them directly before a forward; nothing reads them here
/// (no forward ever runs), so the types are the loosest that still compile.
pub const ForwardCtx = struct {
    cache: ?*KVCache = null,
    ssm_entries: ?[]SSMCacheEntry = null,
    vision_embeddings: ?@import("mlx_stub.zig").mlx_array = null,
    /// Optional in the real ctx (null = do not capture); spec decode sets it
    /// to a destination slot.
    capture_hidden: ?*anyopaque = null,
    skip_lm_head: bool = false,
    kv_attn_fused: bool = false,
    moe_seq_offset: ?*usize = null,
    mrope_pos: ?[]const i32 = null,
    mrope_delta: i32 = 0,
    mrope_total: usize = 0,
    decode_ns: u64 = 0,
};
/// Mirrors the real entry's field NAMES because `scheduler.Slot.deinit` frees
/// them unconditionally. They are `mlx_array` handles there; here they are the
/// stub's null handle, and freeing a null handle is a no-op.
pub const SSMCacheEntry = struct {
    conv_state: @import("mlx_stub.zig").mlx_array = .{},
    ssm_state: @import("mlx_stub.zig").mlx_array = .{},
    initialized: bool = false,
};
/// A recorded SSM recurrence state. Ownership transfers to the caller via
/// `Generator.takeSsmCheckpoints`, so it carries its own deinit.
pub const SSMCheckpoint = struct {
    pub fn deinit(_: *SSMCheckpoint, _: std.mem.Allocator) void {}
};

/// See `Transformer.dsv4`. Uninhabited.
pub const Dsv4State = struct {
    n_mtp: u32 = 0,
};

/// Minimum M at which the dequantized-GEMM prefill path engages. Retained as a
/// constant because the memory-billing helpers compare against it.
pub const PREFILL_DQ_GEMM_MIN_M: usize = 0;

// Tuning overrides. Settable (the CLI writes them) but never read by anything
// in this build.
pub var decode_attn_quant_flag: bool = false;
pub var fused256_override: ?bool = null;
pub var prefill_dq_gemm_override: ?bool = null;

pub fn prefillDqGemmEnabled() bool {
    return false;
}

/// No fused head-dim kernels exist here, so the prefill guard bills the
/// unfused path — the conservative direction.
pub fn prefillHeadDimFused(_: u32) bool {
    return false;
}

pub fn verifyQmmNaxAvailable() bool {
    return false;
}

/// The NAX (Neural Accelerator) status string surfaced by `--version` and
/// `/props`. A string, matching transformer.zig, so version_mod renders it
/// unchanged. NAX is an Apple-GPU feature; naming CUDA here would be a lie
/// about which accelerator is doing the work.
pub fn naxStatus() []const u8 {
    return "unavailable (built without MLX)";
}

const testing = std.testing;

test "KVCache init returns a zero-layer SHELL, it does not refuse" {
    // Regression: `Slot.init` builds a cache for EVERY slot, and for an
    // embedded-engine slot (ds4 / llama.cpp) it asks for zero layers because
    // the engine owns its own KV. Refusing here made every GGUF chat request
    // fail with `MlxUnavailable` AFTER tokenizing successfully -- the one path
    // this whole build exists to serve.
    //
    // Zero layers allocate nothing, so there is no MLX to be unavailable. A
    // non-zero request cannot occur: no MLX model can load in this build.
    var cache = try KVCache.initWithConfigAndHeadDim(testing.allocator, 0, KVQuantConfig.dense, 128);
    defer cache.deinit();
    try testing.expectEqual(@as(usize, 0), cache.step);
}
