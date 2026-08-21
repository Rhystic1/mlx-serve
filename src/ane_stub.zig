//! Apple Neural Engine prefill offload for builds without it (src/build_cfg.zig).
//!
//! The ANE is Apple hardware reached through a private framework
//! (`lib/ane/ane_bridge.m`, Objective-C). It has no equivalent anywhere else,
//! and its whole purpose — offloading MLP/GDN prefill from the GPU — is moot
//! without an MLX forward pass to offload from.
//!
//! `anePrefillAllowed` returning false is the same answer the real module gives
//! on a NAX-class Mac (where the GPU already outruns the seam), so `/props`
//! simply omits its `"ane"` object exactly as it does on any off boot.

const std = @import("std");

/// Bytes of headroom the admission gate reserves before billing an ANE engine.
/// Unused: nothing is ever built.
pub const GATE_BASELINE_BYTES: u64 = 0;
/// The real cap is 2 (dual-die Ultra). Kept non-zero so the fixed-size stats
/// array `server.zig` declares from it is still a legal type; it is never
/// populated because no engine exists.
pub const MAX_UNITS: usize = 2;
pub const MIN_CONTEXT_TOKENS: usize = 0;

/// Published gauges (`mlx_serve:ane_*`). Always zero.
pub var live_int8_bytes: std.atomic.Value(u64) = .init(0);
pub var live_layers: std.atomic.Value(u64) = .init(0);

/// Refused BY NAME on non-Apple silicon, same as on a NAX-class Mac.
pub fn anePrefillAllowed(_: anytype, _: anytype) bool {
    return false;
}

/// Keys the per-silicon tuning tables (draft block cap, verify-lane parity
/// slack, ANE split share). Off Apple there is no chip brand to key on, and a
/// row must never be interpolated — so every table takes its default arm.
pub fn chipBrand() []const u8 {
    return "";
}

pub fn splitShare() f32 {
    return 0;
}

/// A built ANE engine. Uninhabited -- `Transformer.ane_prefill` is always null
/// -- but `/props` reads these fields behind that null check.
pub const AneUnit = struct {
    instance: u32 = 0,
    evals_ok: std.atomic.Value(u64) = .init(0),
    evals_failed: std.atomic.Value(u64) = .init(0),
};

pub const AneEngine = struct {
    units: []AneUnit = &.{},
    mode: enum { channel, row } = .channel,
    rows: u32 = 0,
    chunk_rows: u32 = 0,
    share: f32 = 0,
    int8_bytes: u64 = 0,
    evals: u64 = 0,
    eval_failures: u64 = 0,

    pub fn coveredLayers(_: *const AneEngine) usize {
        return 0;
    }

    /// GDN input-projection coverage is counted separately from MLP coverage
    /// (a MoE checkpoint gets GDN-only offload), so `/props` reports both.
    pub fn coveredGdnLayers(_: *const AneEngine) usize {
        return 0;
    }
};
