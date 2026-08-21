//! DiffusionGemma block-diffusion runner for builds without MLX.
//!
//! `diffusion_gemma` is a Gemma 4 26B-A4B trunk driven by a block-diffusion
//! canvas loop (<=48-step denoise) — an MLX forward pass per step. See
//! src/build_cfg.zig.

const std = @import("std");

pub const Error = error{MlxUnavailable};

/// What one canvas denoise step commits.
pub const CanvasResult = struct {
    tokens: []u32 = &.{},
    steps: u32 = 0,
};

pub const Runner = struct {
    /// Polled between denoise steps so a long canvas can be aborted.
    cancel_flag: ?*std.atomic.Value(bool) = null,

    pub fn init(_: std.mem.Allocator, _: anytype, _: anytype, _: anytype, _: anytype) anyerror!Runner {
        return Error.MlxUnavailable;
    }
    pub fn deinit(_: *Runner) void {}

    /// Runs the prompt through the trunk before the first canvas. Unreachable:
    /// `init` never returns a Runner.
    pub fn prefill(_: *Runner, _: []const u32) anyerror!void {
        return Error.MlxUnavailable;
    }

    /// One canvas denoise step. Unreachable: `init` never returns a Runner.
    pub fn nextCanvas(_: *Runner, _: std.mem.Allocator) anyerror!?CanvasResult {
        return Error.MlxUnavailable;
    }
};
