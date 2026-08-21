//! Weight loading for builds without MLX (see src/build_cfg.zig).
//!
//! A `-Dgguf-only` build serves GGUF through llama.cpp, which loads its own
//! weights inside the engine; safetensors never enter this process. So the
//! whole map is absent rather than empty, and every loader refuses BY NAME.
//!
//! `error.MlxUnavailable` rather than a silent empty map: an empty `Weights`
//! would sail through `resolveWeightPrefix`, reach the transformer builder, and
//! die on the first missing tensor -- the same "incomplete pack" failure mode
//! the media-marker rule exists to prevent. A named error at the load call is
//! what lets the scheduler turn it into an honest 4xx.
//!
//! Callers reach these through `model.zig`'s re-export, never directly.

const std = @import("std");

pub const Error = error{
    /// This build contains no MLX backend, so safetensors cannot be loaded.
    /// Serve a GGUF model instead.
    MlxUnavailable,
};

/// Shape-compatible with the real `Weights` for the few places that hold one
/// by value, but it can never contain anything: nothing in this build produces
/// an `mlx_array`, so there is no element type to store.
pub const Weights = struct {
    /// Same shape as the real map, and always EMPTY. Carrying the real field
    /// (rather than omitting it) is what lets the MLX-side architecture
    /// modules still type-check in a GGUF-only test build -- Zig analyses a
    /// module's tests even when nothing calls its loaders. They are never
    /// linked, because nothing reaches them.
    map: std.StringHashMap(@import("mlx_stub.zig").mlx_array),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Weights {
        return .{
            .map = std.StringHashMap(@import("mlx_stub.zig").mlx_array).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Weights) void {
        self.map.deinit();
        self.* = undefined;
    }

    pub fn count(_: *const Weights) usize {
        return 0;
    }

    /// Always null: the map is empty by construction. Present because the
    /// MLX-side architecture modules still TYPE-CHECK against `Weights` in a
    /// GGUF-only test build (Zig analyses a module's tests even when nothing
    /// calls its loaders); they are never linked, since nothing reaches them.
    pub fn get(_: *const Weights, _: []const u8) ?@import("mlx_stub.zig").mlx_array {
        return null;
    }
};

/// No-op: the prefix is inferred from which tensors a checkpoint actually
/// carries, and this build never has any.
pub fn resolveWeightPrefix(_: anytype, _: *const Weights) void {}

pub fn loadWeights(_: std.Io, _: std.mem.Allocator, _: []const u8) anyerror!Weights {
    return Error.MlxUnavailable;
}

pub fn loadWeightsSingleFile(_: std.mem.Allocator, _: []const u8) anyerror!Weights {
    return Error.MlxUnavailable;
}

pub fn loadWeightsWithVision(_: std.Io, _: std.mem.Allocator, _: []const u8) anyerror!Weights {
    return Error.MlxUnavailable;
}

/// The f16-narrowing report describes tensors read at load time; there are none.
pub fn reportF16Narrowing() void {}

pub fn loadSafetensorsFile(_: anytype, _: anytype, _: anytype, _: anytype, _: anytype) anyerror!void {
    return Error.MlxUnavailable;
}

/// Whether a loaded f16 tensor is a per-channel table that must be narrowed.
/// Nothing is loaded, so nothing narrows.
pub fn narrowsLoadedF16(_: []const u8, _: usize, _: anytype) bool {
    return false;
}

const testing = std.testing;

test "every safetensors loader refuses by name rather than returning an empty map" {
    // The failure this guards is not "loading fails" -- it is loading appearing
    // to SUCCEED with nothing in it, which surfaces much later as a missing
    // tensor deep in a forward pass.
    var buf: [256]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const alloc = fba.allocator();
    try testing.expectError(Error.MlxUnavailable, loadWeightsSingleFile(alloc, "/x.safetensors"));
}
