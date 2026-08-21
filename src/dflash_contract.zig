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

/// The object holding the DFlash contract triple: DFlash2 checkpoints nest it
/// under `dflash_config`, v1 assistants declare it at the root. Null when
/// neither shape declares all three fields.
pub fn dflashContractObject(root: std.json.ObjectMap) ?std.json.ObjectMap {
    if (root.get("dflash_config")) |dc| {
        if (dc == .object and hasContractTriple(dc.object)) return dc.object;
    }
    if (hasContractTriple(root)) return root;
    return null;
}

fn hasContractTriple(o: std.json.ObjectMap) bool {
    return o.get("block_size") != null and
        o.get("mask_token_id") != null and
        o.get("target_layer_ids") != null;
}
