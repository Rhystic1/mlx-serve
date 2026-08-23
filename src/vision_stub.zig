//! Vision encoder for builds without MLX (see src/build_cfg.zig).
//!
//! The SigLIP / Qwen3-VL / Muse / LFM2-VL towers are MLX graphs, so image
//! understanding is absent here. Note this stubs only the ENCODER: the
//! preprocessing modules (`qwen_vision.zig`, `muse_vision.zig`,
//! `lfm2_vision.zig`) are pure buffer math on f32 pixels and compile fine —
//! Zig never analyses their MLX halves because nothing calls them. Only
//! `vision.zig`'s `VisionEncoder` had to be swapped.
//!
//! `init` refuses BY NAME, so `scheduler.zig` leaves `vision_encoder` null and
//! the existing `VisionEncoderNotLoaded` path answers image requests — the same
//! path a text-only checkpoint already takes on macOS. Nothing new is invented
//! for the GGUF build; it simply always takes that branch.
//!
//! When vision returns here it will be through llama.cpp's `mtmd` (the
//! prebuilt `mtmd.dll` already ships beside `llama.dll`), which does its own
//! preprocessing and projection — not through this type.

const std = @import("std");
const mlx = @import("mlx_stub.zig");

pub const Error = error{MlxUnavailable};

pub const VisionEncoder = struct {
    s: mlx.mlx_stream = .{},

    pub fn init(_: std.mem.Allocator, _: anytype, _: anytype) anyerror!VisionEncoder {
        return Error.MlxUnavailable;
    }

    pub fn deinit(_: *VisionEncoder) void {}

    /// Gemma 4's tower can also encode raw audio. No tower exists here.
    pub fn supportsAudio(_: *const VisionEncoder) bool {
        return false;
    }

    pub fn forward(_: *VisionEncoder, _: anytype) anyerror!mlx.mlx_array {
        return Error.MlxUnavailable;
    }

    pub fn forwardPatches(_: *VisionEncoder, _: anytype, _: u32, _: u32) anyerror!mlx.mlx_array {
        return Error.MlxUnavailable;
    }

    pub fn forwardVideoPatches(_: *VisionEncoder, _: anytype, _: u32, _: u32, _: u32) anyerror!mlx.mlx_array {
        return Error.MlxUnavailable;
    }

    pub fn forwardAudio(_: *VisionEncoder, _: anytype) anyerror!mlx.mlx_array {
        return Error.MlxUnavailable;
    }
};
