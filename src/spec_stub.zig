//! Speculative-decode modules for builds without MLX (see src/build_cfg.zig).
//!
//! Covers the drafter (Gemma cross-attention), DFlash block-drafter, and the
//! native MTP head. All three are MLX: each is a second model whose forward
//! pass runs on the same trunk graph, and each is verified by an MLX trunk
//! forward. None can exist here.
//!
//! This is not a permanent gap in the way media generation is. llama.cpp has
//! its own draft-model speculative decode, and wiring it through the shim is
//! planned; it just does not route through these types.
//!
//! One module serves all three because the shared plumbing only ever names
//! their loader entry points and a couple of resolver helpers -- the real
//! modules' internals never surface outside the MLX build.

const std = @import("std");

pub const Error = error{MlxUnavailable};

/// A loaded speculative sidecar. Uninhabited: every loader below refuses, so
/// `LoadedModel.drafter` / `.dflash` / `.mtp` stay null and the four per-surface
/// `use_*` gates all resolve to serial decode.
/// The sidecar's own config, read to resolve the draft block width.
/// DFlash's config CONTRACT is what identifies a sidecar as a block-drafter
/// (`block_size` + `mask_token_id` + `target_layer_ids` -- never `model_type`),
/// and the scheduler reads those fields when binding one. Mirrored so that
/// binding code compiles; no sidecar is ever loaded here.
pub const SidecarConfig = struct {
    block_size: u32 = 0,
    mask_token_id: u32 = 0,
    target_layer_ids: []const u32 = &.{},
};

pub const Sidecar = struct {
    config: SidecarConfig = .{},

    pub fn deinit(_: *Sidecar) void {}

    /// Binds a loaded sidecar to its target trunk. Unreachable — every loader
    /// refuses before one exists — but named so the load sequence compiles.
    pub fn bind(_: *Sidecar, _: anytype) anyerror!void {
        return Error.MlxUnavailable;
    }

    /// Which measured round-cost surface the MTP EV controller should plan
    /// under. `.generic` is the unmeasured default and the only honest answer
    /// without a trunk to fingerprint.
    pub fn m5NaxCostProfile(_: *const Sidecar, _: anytype) MtpCostProfile {
        return .generic;
    }
};

pub const Drafter = Sidecar;
pub const DFlash = Sidecar;
pub const MtpModel = Sidecar;

pub fn load(_: std.Io, _: std.mem.Allocator, _: []const u8, _: anytype) anyerror!*Sidecar {
    return Error.MlxUnavailable;
}

/// A checkpoint may ship a sidecar in `<model_dir>/drafter`; without a backend
/// to run it, the honest answer is that none is usable.
pub fn resolveInDirDrafter(_: std.Io, _: std.mem.Allocator, _: []const u8) ?[]u8 {
    return null;
}

pub fn resolveMtpSource(_: std.Io, _: std.mem.Allocator, _: []const u8) ?[]u8 {
    return null;
}

/// Per-silicon caps on the draft block / MTP depth. Zero means "no speculative
/// width", which is what the schedulers read as serial.
/// The real helper returns a labelled row (the cap PLUS which silicon row it
/// came from, so a fence is debuggable). Mirrored, with the "no wide lane"
/// answer.
pub const BlockCap = struct { cap: u32 = 0, label: []const u8 = "no-mlx" };

pub fn blockCapForMachine(_: anytype) BlockCap {
    return .{};
}

pub fn adaptiveDepthCapForMachine() u32 {
    return 0;
}

// ── Names the shared plumbing reads directly ───────────────────────────────
//
// Constants keep their real values so the CLI still parses and reports the
// same defaults; the LOADERS are what refuse. A user who passes `--drafter`
// or `--mtp` to this build gets a named load failure, not a silently ignored
// flag (the "silent flag eater" class in CLAUDE.md).

pub const DrafterModel = Sidecar;
pub const DflashModel = Sidecar;
pub const DflashCtx = struct {
    /// Mirrors the fields the prefix-cache commit path reads. `cache.step`
    /// is compared against the restored base to decide adoption
    /// (`base + step == matched` exactly).
    cache: struct { step: usize = 0 } = .{},
    base_pos: usize = 0,

    pub fn init(_: std.mem.Allocator, _: anytype, _: usize) anyerror!DflashCtx {
        return Error.MlxUnavailable;
    }

    pub fn deinit(_: *DflashCtx) void {}

    /// Absolute length of the captured trunk-hidden context. Zero, so the
    /// "context covers the prefix" test never passes and nothing is committed.
    pub fn absLen(_: *const DflashCtx) usize {
        return 0;
    }
};

pub const DEFAULT_BLOCK_SIZE: u32 = 4;
pub const DEFAULT_DEPTH: u32 = 3;
pub const MAX_DEPTH: u32 = 8;

/// Which measured round-cost surface the MTP EV controller plans under. There
/// is no trunk to measure, so `.generic` — the unmeasured default — is the only
/// honest arm.
pub const MtpCostProfile = enum { generic };

pub fn loadDrafter(_: std.Io, _: std.mem.Allocator, _: anytype, _: []const u8) anyerror!Sidecar {
    return Error.MlxUnavailable;
}

pub fn loadDflash(_: std.Io, _: std.mem.Allocator, _: anytype, _: []const u8) anyerror!Sidecar {
    return Error.MlxUnavailable;
}

/// Returns by VALUE (the real `loadMtp` does), unlike the drafter loaders
/// which hand back a pointer. Mirrored so the call sites' optionals match.
pub fn loadMtp(_: std.Io, _: std.mem.Allocator, _: anytype, _: []const u8) anyerror!MtpModel {
    return Error.MlxUnavailable;
}

/// Whether a checkpoint ships an in-checkpoint MTP head. Answering false is
/// what keeps `defaultEnableMtp` off rather than arming a head that cannot run.
pub fn hasMtpHead(_: std.Io, _: std.mem.Allocator, _: []const u8) bool {
    return false;
}

pub fn probeIsDflash(_: std.Io, _: std.mem.Allocator, _: []const u8) bool {
    return false;
}

pub fn recommendedBlockSize(_: anytype) u32 {
    return 0;
}

/// The wide-verify qmm lane is a Metal kernel; there is none.
pub fn wideVerifyLaneAvailable() bool {
    return false;
}

pub fn resolveBlockSize(_: u32, _: u32, _: bool, _: bool, _: u32) u32 {
    return 0;
}
