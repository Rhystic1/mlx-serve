//! Generation surface for builds without MLX (see src/build_cfg.zig).
//!
//! `generate.zig` is the MLX autoregressive loop (1147 MLX references):
//! sampling, PLD/drafter/MTP orchestration, prefill chunking, the stall clock.
//! A GGUF-only build reaches none of it — `scheduler.zig` drives a llama.cpp
//! slot through the engine's own per-token loop, and `generate.zig`'s only
//! mentions of llama today are comments and one test.
//!
//! The types below exist because the shared plumbing names them in signatures
//! and struct fields (`SamplingParams` travels from the HTTP layer regardless of
//! backend; `TokenLogprob` is the logprobs wire shape). They are the real
//! shapes, not placeholders — only the MLX-driven FUNCTIONS are refused.

const std = @import("std");

pub const Error = error{MlxUnavailable};

/// Sampling knobs parsed from the request. Backend-independent by design: the
/// llama.cpp path reads the same struct.
/// Mirrors generate.zig's field set. This one is genuinely backend-independent
/// — the HTTP layer parses a request into it regardless of which engine serves
/// the request, and the llama.cpp path reads the same values.
pub const SamplingParams = struct {
    temperature: f32 = 1.0,
    top_p: f32 = 1.0,
    top_k: u32 = 0, // 0 = disabled
    repeat_penalty: f32 = 1.0,
    presence_penalty: f32 = 0.0, // 0.0 = disabled
    seed: ?u64 = null,
    constraint: ?*Constraint = null,
    suppress_mask: ?@import("mlx_stub.zig").mlx_array = null,
};

pub const Constraint = struct {};

pub const TokenLogprob = struct {
    token_id: u32 = 0,
    logprob: f32 = 0,
    bytes: []const u8 = "",
};

pub const LogprobResult = struct {
    token_logprob: f32 = 0, // logprob of the chosen token
    top_logprobs: []TokenLogprob = &.{}, // top N alternatives (caller must free)
};

/// Mirrors generate.zig's field set — `server.zig` reads every one of these
/// when building a completion response, and the usage/timings block is part of
/// the wire contract regardless of backend.
pub const GenerationResult = struct {
    text: []u8 = &.{},
    token_ids: []u32 = &.{},
    prompt_tokens: u32 = 0,
    completion_tokens: u32 = 0,
    finish_reason: []const u8 = "stop",
    prefill_tps: f64 = 0,
    decode_tps: f64 = 0,
    prefill_ns: u64 = 0,
    decode_ns: u64 = 0,
    /// Prompt tokens served from a KV-cache prefix. `prompt_tokens -
    /// cached_tokens` is what actually ran this turn -- the llama.cpp path
    /// reports its persistent-session prefix reuse through this same field, so
    /// it stays meaningful in a GGUF-only build.
    cached_tokens: u32 = 0,
    logprobs: ?[]LogprobResult = null,
    finish_details: ?[]const u8 = null,
};

// Empty structs rather than `opaque`: `LoadedModel` stores several of these
// as `?T` by value, and an opaque type has no size so it cannot be optional.
// Nothing constructs one, so an empty struct is equally uninhabited in
// practice while still being a complete type.
/// Owns the parsed schema + grammar in the real module; here it is inert but
/// still needs `init`/`deinit` because the JSON-mode path constructs one before
/// any backend is involved.
pub const SchemaConstraint = struct {
    /// The handle `SamplingParams.constraint` points at once built.
    constraint: Constraint = .{},
    /// `[vocab]` bool scratch the mask is written into.
    mask_buf: []bool = &.{},

    pub fn initFromValue(_: *SchemaConstraint, _: std.mem.Allocator, _: anytype, _: anytype) anyerror!void {
        return Error.MlxUnavailable;
    }
    pub fn deinit(_: *SchemaConstraint) void {}
};
/// A tagged union in the real module (one arm per MTP-capable arch) and
/// switched on at its use sites, so it must stay a union here. The single arm
/// is uninhabited: nothing constructs a KVCache in this build.
pub const MtpCacheRef = union(enum) {
    qwen: @import("transformer_stub.zig").KVCache,

    pub fn step(_: *const MtpCacheRef) usize {
        return 0;
    }

    /// Trims the committed history to `len` before a prefix-cache commit --
    /// the cache at rest can hold a stale draft tail.
    pub fn truncate(_: *MtpCacheRef, _: usize, _: anytype) anyerror!void {
        return Error.MlxUnavailable;
    }

    /// The underlying KV handle the commit path stores.
    pub fn kv(_: *MtpCacheRef) *anyopaque {
        return undefined;
    }

    pub fn deinit(_: *MtpCacheRef) void {}
};
/// A tagged union in the real module, switched on at its use sites. The one
/// arm is uninhabited here.
/// The arm's payload is the SAME type `spec_stub.loadMtp` produces -- the
/// scheduler assigns a loaded head straight into this union, so two separate
/// uninhabited types would not interconvert.
pub const MtpHead = @import("spec_stub.zig").MtpModel;

pub const MtpHeadRef = union(enum) {
    qwen: *MtpHead,

    pub fn makeCache(_: MtpHeadRef, _: std.mem.Allocator) anyerror!MtpCacheRef {
        return Error.MlxUnavailable;
    }
};
pub const MtpRestored = struct { cache: MtpCacheRef, base: usize };
/// Mirrors generate.zig. The loop-stop guard is what turns a degenerate tail
/// into a `finish_reason: "length"` truncation, and `scheduler.zig` names the
/// tier when reporting it.
// The REAL detector, not a stub: loop detection is pure token analysis and a
// llama.cpp generation loops exactly as readily as an MLX one. Stubbing it out
// would ship a server with no loop-stop guard.
const loop_detect = @import("loop_detect.zig");
pub const DegenerateTail = loop_detect.DegenerateTail;
pub const degenerateTail = loop_detect.degenerateTail;
pub const isNearRepeatTailLoop = loop_detect.isNearRepeatTailLoop;

/// Smallest prefill chunk the ladder will select. Kept so the memory-billing
/// helpers keep their floor.
pub const PREFILL_CHUNK_FLOOR: usize = 512;

pub var prefill_chunk_override: usize = 8192;
pub var prefill_chunk_explicit: bool = false;
pub var prefill_trace_force: bool = false;
pub var mtp_history_window_override: ?usize = null;
pub const degenerate_loop_reps = loop_detect.degenerate_loop_reps;
pub const near_repeat_window = loop_detect.near_repeat_window;

// ── MLX-driven entry points: refused by name ──

pub fn generate(_: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) anyerror!GenerationResult {
    return Error.MlxUnavailable;
}

pub fn generateMtp(_: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) anyerror!GenerationResult {
    return Error.MlxUnavailable;
}

pub fn computeEmbeddingsBatch(_: std.mem.Allocator, _: anytype, _: anytype) anyerror![][]f32 {
    return Error.MlxUnavailable;
}

pub fn visionPrefill(_: anytype) anyerror!void {
    return Error.MlxUnavailable;
}

/// Returns a LAZY array (the sampling graph), not a realized id -- the
/// caller evals it. Infallible in the real module.
pub fn sampleTokenLazy(_: anytype, _: SamplingParams, _: anytype) @import("mlx_stub.zig").mlx_array {
    return .{};
}

pub fn installSuppressMask(_: anytype, _: anytype, _: []const u8, _: []const u32) void {}

// ── Pure helpers: real behaviour, no backend needed ──

pub fn isEosId(id: u32, eos: []const u32) bool {
    for (eos) |e| {
        if (e == id) return true;
    }
    return false;
}

pub fn tokensPerSec(tokens: u64, elapsed_ns: u64) f64 {
    if (elapsed_ns == 0) return 0;
    return @as(f64, @floatFromInt(tokens)) * 1_000_000_000.0 / @as(f64, @floatFromInt(elapsed_ns));
}

pub fn prefillTokensPerSec(prompt_tokens: u32, cached_tokens: u32, prefill_ns: u64) f64 {
    return tokensPerSec(prompt_tokens -| cached_tokens, prefill_ns);
}

/// The chunk ladder is an MLX memory-billing decision; without it the only
/// honest answer is the floor.
pub fn effectivePrefillChunk(_: u32, _: u32, _: usize, _: bool, _: bool, pinned_chunk: usize) usize {
    return if (pinned_chunk != 0) pinned_chunk else prefill_chunk_override;
}

/// Vision prefill is never chunked here because it never runs.
pub fn visionPrefillUnchunked(_: bool) bool {
    return true;
}

/// No MTP head exists, so no depth is available to cap.
pub const Generator = struct {
    /// DFlash's runtime yield gate: if realized acceptance per round falls
    /// below these within ~4 rounds, speculation disables itself. Real values
    /// kept so the gate's reported thresholds do not change; unreachable
    /// because no drafter loads. The lower bar keys on RESOLVED thinking
    /// (sidecars are trained on reasoning-mode output), never on has_tools.
    pub const DFLASH_GATE_MIN_ACCEPTED_PER_ROUND: f32 = 2.0;
    pub const DFLASH_THINKING_GATE_MIN_ACCEPTED_PER_ROUND: f32 = 1.0;
    /// MoE targets verify at a lower per-round bar: their trunk forward is
    /// cheaper relative to the drafter, so a block pays for itself sooner.
    pub const DFLASH_MOE_GATE_MIN_ACCEPTED_PER_ROUND: f32 = 1.8;

    // Fields the scheduler and the HTTP layer read directly off a live
    // generator. Inert: `init` never returns one. Named and typed to match
    // generate.zig so the shared decode-loop plumbing compiles unchanged.
    done: bool = true,
    finish_reason: []const u8 = "stop",
    prompt_tokens: u32 = 0,
    completion_tokens: u32 = 0,
    generated_ids: std.ArrayListUnmanaged(u32) = .empty,
    next_token_id: u32 = 0,
    has_pending_token: bool = false,
    has_pending_logits: bool = false,
    consecutive_pad: u32 = 0,
    last_logprob: ?LogprobResult = null,
    logprobs_n: u32 = 0,
    sampling: SamplingParams = .{},
    timeout_ns: u64 = 0,
    ctx: @import("transformer_stub.zig").ForwardCtx = .{},
    /// Allocator the SSM checkpoints were made with -- distinct from the
    /// generator's own, because `takeSsmCheckpoints` transfers ownership.
    ssm_checkpoint_alloc: ?std.mem.Allocator = null,

    // Speculation state. `pld_enabled` stays TRUE through a runtime kill --
    // `spec_disabled_runtime` is the clause that records the kill -- because
    // the batched-decode gate reads dispatch, not the armed flags.
    pld_enabled: bool = false,
    spec_disabled_runtime: bool = false,
    dspark_enabled: bool = false,
    drafter: ?*anyopaque = null,
    dflash: ?*anyopaque = null,
    /// The dflash assistant's trunk-hidden context, snapshotted into the
    /// prefix cache so a reused prefix does not draft blind.
    dflash_ctx: ?@import("spec_stub.zig").DflashCtx = null,
    /// The MTP head's committed decode history, likewise snapshotted into
    /// the prefix cache (a restore forwards no trunk layers, so without it
    /// every warm hit would draft blind).
    mtp_cache: ?MtpCacheRef = null,
    /// Absolute trunk position the MTP history's index 0 represents.
    mtp_position_base: usize = 0,
    mtp: ?MtpHeadRef = null,

    /// What a speculative step hands back: the tokens to emit plus how many
    /// DRAFTED ones were accepted. The callers destructure it, so the shape
    /// has to match even though no step ever runs.
    pub const PldStepResult = struct {
        tokens: []const u32 = &.{},
        accepted_tokens: u32 = 0,
    };

    pub fn init(_: std.Io, _: std.mem.Allocator, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) anyerror!Generator {
        return Error.MlxUnavailable;
    }

    /// The chokepoint every scheduler-driven generator goes through -- it is
    /// what wires the model's reserved-token suppression mask into
    /// `gen.sampling`, which is why the batched path reads that copy rather
    /// than the slot's raw request params.
    pub fn initWithOptions(_: std.Io, _: std.mem.Allocator, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) anyerror!Generator {
        return Error.MlxUnavailable;
    }

    pub fn deinit(_: *Generator, _: std.mem.Allocator) void {}

    /// The EV controller's depth caps. Zero width = serial.
    pub fn resolveMtpDepthCap(_: u32, _: bool) u32 {
        return 0;
    }

    pub fn resolveMtpDepthCapForProfile(_: u32, _: anytype) u32 {
        return 0;
    }

    /// One decode step, plus the speculative variants the scheduler dispatches
    /// to via `specTickMode`. All unreachable: `init` never returns a
    /// Generator, so the GGUF path drives llama.cpp's own per-token loop.
    pub fn next(_: *Generator, _: std.mem.Allocator) anyerror!?u32 {
        return Error.MlxUnavailable;
    }
    pub fn nextPld(_: *Generator, _: std.mem.Allocator, _: u32, _: u32) anyerror!?PldStepResult {
        return Error.MlxUnavailable;
    }
    pub fn nextDrafter(_: *Generator, _: std.mem.Allocator) anyerror!?PldStepResult {
        return Error.MlxUnavailable;
    }
    pub fn nextDflash(_: *Generator, _: std.mem.Allocator) anyerror!?PldStepResult {
        return Error.MlxUnavailable;
    }
    pub fn nextMtp(_: *Generator, _: std.mem.Allocator) anyerror!?PldStepResult {
        return Error.MlxUnavailable;
    }
    pub fn nextDspark(_: *Generator, _: std.mem.Allocator) anyerror!?PldStepResult {
        return Error.MlxUnavailable;
    }

    /// How much of the MTP history is safely committed. The cache at rest can
    /// hold a stale draft tail, so a commit trims to this first.
    pub fn mtpCommittedHistoryLen(_: *const Generator) usize {
        return 0;
    }

    /// Hands ownership of the SSM checkpoints recorded during prefill to the
    /// caller (which is why they are freed with the caller's allocator, not
    /// the generator's). None exist.
    pub fn takeSsmCheckpoints(_: *Generator) []@import("transformer_stub.zig").SSMCheckpoint {
        return &.{};
    }

    /// Emits the `[spec-stats] mode=...` line. Nothing speculative ran.
    pub fn logSpecStats(_: *const Generator) void {}

    /// The ONE place the decode step counter moves (scan-pinned in the real
    /// module, so it is mirrored rather than dropped).
    pub fn advanceStep(_: *Generator, _: anytype) void {}

    /// Drains lazy pipeline state before a slot joins a batched decode group.
    /// Optional: null = nothing was pending.
    pub fn drainPipelineForBatch(_: *Generator, _: std.mem.Allocator) anyerror!?u32 {
        return Error.MlxUnavailable;
    }
};
