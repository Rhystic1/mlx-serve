//! The DFlash config CONTRACT — pure, and the single source of truth for it.
//!
//! DFlash is a METHOD, keyed on the contract triple (`block_size` +
//! `mask_token_id` + `target_layer_ids`), never on `model_type`: DFlash2
//! checkpoints ship a bare "qwen3" model_type, and the `_assistant` suffix is
//! only the discovery-level "this is a drafter" signal.
//!
//! Split out of dflash.zig for the Windows/Linux port. `model_discovery.zig`
//! classifies drafters with this predicate and must work in every build, but
//! dflash.zig itself pulls transformer.zig and deepseek_v4.zig behind it. The
//! contract is load-bearing, so it is SHARED rather than duplicated into a
//! stub — two copies would be two contracts.

const std = @import("std");

/// Does this config.json declare the DFlash contract? ALL THREE fields must
/// be present — `model_type` (`*_assistant`) is only the discovery-level
/// "this is a drafter" signal, never the DFlash detection.
pub fn isDflashConfigJson(root: std.json.ObjectMap) bool {
    return dflashContractObject(root) != null;
}

/// The contract fields, looked up NESTED-FIRST then at the root. DFlash2
/// checkpoints nest the triple under `dflash_config`, v1 assistants declare
/// it at the root, and DSpark SPLITS it across both (`block_size` at the
/// root, the rest nested) — so neither object alone answers the question and
/// a nested-only reader silently classifies the sidecar as "not a drafter".
pub const Contract = struct {
    nested: ?std.json.ObjectMap,
    root: std.json.ObjectMap,

    pub fn get(self: Contract, key: []const u8) ?std.json.Value {
        if (self.nested) |n| {
            if (n.get(key)) |v| return v;
        }
        return self.root.get(key);
    }
};

/// The contract view over this config.json, or null when the triple is not
/// declared by the nested object and the root taken together.
pub fn dflashContractObject(root: std.json.ObjectMap) ?Contract {
    const nested: ?std.json.ObjectMap = blk: {
        if (root.get("dflash_config")) |dc| {
            if (dc == .object) break :blk dc.object;
        }
        break :blk null;
    };
    const c = Contract{ .nested = nested, .root = root };
    if (c.get("block_size") == null) return null;
    if (c.get("mask_token_id") == null) return null;
    if (c.get("target_layer_ids") == null) return null;
    return c;
}
