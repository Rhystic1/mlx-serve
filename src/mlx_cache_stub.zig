//! Prefix cache + SSD KV tier for builds without MLX (see src/build_cfg.zig).
//!
//! Both cache an MLX `KVCache` — the hot tier holds live `mlx_array` handles,
//! the disk tier serializes them to safetensors — so neither can exist here.
//!
//! This is NOT a lost feature on the GGUF path: llama.cpp keeps its own
//! persistent per-model sessions and does its own prefix reuse
//! (`--llama-cache-entries`, `LoadedModel.llama_sessions`), which is what
//! `GenerationResult.cached_tokens` reports for a llama slot. The MLX hot/disk
//! tiers simply never applied to it.
//!
//! `lookupAndRestore` returning a zero-length match is the "cold prompt"
//! answer every caller already handles, so the scheduler needs no new branch.

const std = @import("std");

pub const Error = error{MlxUnavailable};

// ── Hot prefix cache ──

/// The dflash assistant context committed alongside a cached prefix. It is
/// DRAFT-side state: a restore that fails simply starts blind rather than
/// wrong, which is why it can be absent here without correctness risk.
pub const DflashCommit = struct {
    cache: *anyopaque = undefined,
    base_pos: usize = 0,
};

pub const LookupResult = struct {
    /// Prompt tokens served from cache. Always 0: nothing is ever stored, so
    /// every prompt is a cold prefill -- the branch callers already handle.
    matched: usize = 0,
    /// Did the restore span the FULL new prompt (identical re-issue)?
    full_match: bool = false,
    /// Absolute trunk position a restored DFlash context starts at. Null means
    /// the assistant starts blind, which is the safe direction: DRAFT-side
    /// state, so blind costs acceptance, never a wrong token.
    dflash_base: ?usize = null,
    mtp_base: ?usize = null,
};

pub const HotPrefixCache = struct {
    /// The SSD tier, attached after construction. Never reached: `shouldUse`
    /// is false, so no HotPrefixCache is ever built to attach one to.
    disk: ?DiskTier = null,

    pub fn deinit(_: *HotPrefixCache) void {}

    /// Whether this arch benefits from the hot tier at all. False here means
    /// the scheduler never constructs one -- which is why `initWithMem` is
    /// unreachable rather than merely inert.
    pub fn shouldUse(_: anytype, _: bool) bool {
        return false;
    }

    /// Optional (not fallible) in the real module: null means "no hot tier",
    /// which is what `shouldUse` already guarantees here.
    pub fn initWithMem(_: std.mem.Allocator, _: anytype, _: anytype) ?HotPrefixCache {
        return null;
    }

    /// Fallible in the real module (it restores KV state). A zero-length match
    /// is the "cold prompt" answer every caller already handles.
    pub fn lookupAndRestore(_: *HotPrefixCache, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) anyerror!LookupResult {
        return LookupResult{};
    }

    pub fn commitWithState(_: *HotPrefixCache, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) anyerror!void {
        return Error.MlxUnavailable;
    }

    pub fn flushPendingDisk(_: *HotPrefixCache, _: anytype) void {}
};

// ── SSD tier ──

/// Matches the real default so `--prefix-cache-disk` still parses and reports
/// the same number; the tier itself just never stores anything.
pub const DEFAULT_CHUNK_TOKENS: usize = 256;

pub const DiskTier = struct {
    pub fn deinit(_: *DiskTier) void {}

    pub fn init(_: std.mem.Allocator, _: std.Io, _: []const u8, _: anytype, _: u64, _: usize) anyerror!DiskTier {
        return Error.MlxUnavailable;
    }
};

/// Where the SSD tier would live. Refuses, so the scheduler's `break
/// :attach` path runs and no disk tier is attached.
pub fn defaultBaseDir(_: std.mem.Allocator) anyerror![]u8 {
    return Error.MlxUnavailable;
}

/// Identifies which model a cache entry belongs to. Nothing is written, so
/// nothing needs identifying.
/// Fallible in the real module (it reads the model dir). Nothing is ever
/// cached here, so it succeeds with no fingerprint.
/// Fallible in the real module (it reads the model dir). Refusing here means
/// the scheduler takes its `break :attach` path and no tier is attached.
pub fn modelFingerprint(_: std.mem.Allocator, _: std.Io, _: []const u8) anyerror![]u8 {
    return Error.MlxUnavailable;
}
