//! Live tower-parity test for the Qwen3-VL vision encoder.
//!
//! Own file, imported only by the MLX suite (src/tests_mlx.zig): it is the one
//! test in qwen_vision.zig that touches the MLX TOWER, while
//! `server.zig` imports that module for its pure preprocessing. Zig compiles
//! the tests of every file reachable from the test root's import graph, so
//! leaving this beside them pulled the tower -- and MLX -- into a -Dgguf-only
//! test binary.
//!
//! Env-gated and skipped without a fixture, exactly as before.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const log = @import("log.zig");
const qwen_vision = @import("qwen_vision.zig");
const QwenVision = qwen_vision.QwenVision;
const QBlock = qwen_vision.QBlock;
const Weights = model_mod.Weights;
const resolveVisionPrefix = qwen_vision.resolveVisionPrefix;
const readBinF32 = qwen_vision.readBinF32;
const PatchProjLayout = qwen_vision.PatchProjLayout;
const patchProjLayout = qwen_vision.patchProjLayout;
const fmtLayer = qwen_vision.fmtLayer;

test "qwen vision tower prefix and patch_embed layout are PROBED, not hardcoded" {
    const allocator = std.testing.allocator;
    const put = struct {
        fn add(w: *Weights, alloc: std.mem.Allocator, key: []const u8) !void {
            const k = try alloc.dupe(u8, key);
            try w.map.put(k, mlx.mlx_array_new());
        }
    }.add;

    // Qwen3-VL spelling.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "vision_tower.patch_embed.proj.weight");
        try std.testing.expectEqualStrings("vision_tower.", resolveVisionPrefix(&w).?);
    }
    // avlp12 Alis spelling (live: vision silently disabled, 0.92 GB dead weight).
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.visual.patch_embed.proj.weight");
        try std.testing.expectEqualStrings("model.visual.", resolveVisionPrefix(&w).?);
    }
    // No tower (text-only pack, or --no-vision dropped it) → vision disabled.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.layers.0.self_attn.q_proj.weight");
        try std.testing.expect(resolveVisionPrefix(&w) == null);
    }
    // patch_embed.proj axis order is derived from the shape, not assumed.
    {
        // mlx-community/Qwen3.5-0.8B-MLX-4bit
        try std.testing.expectEqual(PatchProjLayout.channels_last, patchProjLayout(&.{ 768, 2, 16, 16, 3 }, 2, 16).?);
        // avlp12/Qwen3.8-27B-Alis-MLX-4bit
        try std.testing.expectEqual(PatchProjLayout.channels_first, patchProjLayout(&.{ 1152, 3, 2, 16, 16 }, 2, 16).?);
        try std.testing.expect(patchProjLayout(&.{ 1152, 3, 2, 16 }, 2, 16) == null);
        try std.testing.expect(patchProjLayout(&.{ 1152, 5, 5, 5, 5 }, 2, 16) == null);
    }
    // Layer keys are built from the RESOLVED prefix.
    {
        var buf: [256]u8 = undefined;
        try std.testing.expectEqualStrings(
            "model.visual.blocks.7.attn.qkv.weight",
            fmtLayer(&buf, "model.visual.", 7, "attn.qkv.weight"),
        );
    }
}

// Live parity vs the mlx-vlm reference vision tower. Feeds the REFERENCE's own
// pixel_values straight into QwenVision (isolating the ViT math from the
// preprocessing), then compares post-merger embeddings. Build the fixture first:
//   python3 tests/build_qwen_vision_fixture.py --model <dir> --image <img> --out <fix>
// then run via tests/test_qwen_vision_parity.sh (sets the env vars below).
test "qwen vision parity vs reference embeddings (QWEN_VISION_TEST_MODEL)" {
    const model_raw = std.c.getenv("QWEN_VISION_TEST_MODEL") orelse return error.SkipZigTest;
    const fix_raw = std.c.getenv("QWEN_VISION_FIXTURE") orelse return error.SkipZigTest;
    const gh_raw = std.c.getenv("QV_GH") orelse return error.SkipZigTest;
    const gw_raw = std.c.getenv("QV_GW") orelse return error.SkipZigTest;
    const dir = std.mem.sliceTo(model_raw, 0);
    const fix = std.mem.sliceTo(fix_raw, 0);
    if (dir.len == 0 or fix.len == 0) return error.SkipZigTest;
    const grid_h = try std.fmt.parseInt(u32, std.mem.sliceTo(gh_raw, 0), 10);
    const grid_w = try std.fmt.parseInt(u32, std.mem.sliceTo(gw_raw, 0), 10);

    const alloc = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    const config = try model_mod.parseConfig(io, alloc, dir);
    try std.testing.expect(config.qwen_vision);
    var weights = try model_mod.loadWeightsWithVision(io, alloc, dir);
    defer weights.deinit();
    var qv = try QwenVision.init(alloc, config, &weights);
    defer qv.deinit();

    // Reference pixel_values [N, C*tps*ps*ps] and post-merger embeddings.
    const px_path = try std.fmt.allocPrint(alloc, "{s}/pixel_values.bin", .{fix});
    defer alloc.free(px_path);
    const ref_path = try std.fmt.allocPrint(alloc, "{s}/ref_embeds.bin", .{fix});
    defer alloc.free(ref_path);
    const px = try readBinF32(io, alloc, px_path);
    defer alloc.free(px);
    const ref = try readBinF32(io, alloc, ref_path);
    defer alloc.free(ref);

    const n = grid_h * grid_w;
    const feat = px.len / n;
    const px_shape = [_]c_int{ @intCast(n), @intCast(feat) };
    const px_arr = mlx.mlx_array_new_data(px.ptr, &px_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(px_arr);

    const out = try qv.forward(px_arr, grid_h, grid_w);
    defer _ = mlx.mlx_array_free(out);
    var out_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out_f32);
    try mlx.check(mlx.mlx_astype(&out_f32, out, .float32, qv.s));
    try mlx.check(mlx.mlx_array_eval(out_f32));
    const od = mlx.mlx_array_data_float32(out_f32) orelse return error.TestUnexpectedNullData;

    try std.testing.expectEqual(ref.len, n / 4 * config.qv_out_hidden);
    const dim = config.qv_out_hidden;
    var max_abs: f32 = 0;
    var max_idx: usize = 0;
    var sum_abs: f64 = 0;
    var gt_010: usize = 0;
    var gt_030: usize = 0;
    // Per-token max diff to see clustering (a structural bug clusters by token).
    var worst_tok_diff: f64 = 0;
    var worst_tok: usize = 0;
    var tok_sum: f64 = 0;
    var cur_tok: usize = 0;
    for (0..ref.len) |i| {
        const d = @abs(od[i] - ref[i]);
        if (d > max_abs) {
            max_abs = d;
            max_idx = i;
        }
        if (d > 0.10) gt_010 += 1;
        if (d > 0.30) gt_030 += 1;
        sum_abs += d;
        const tok = i / dim;
        if (tok != cur_tok) {
            if (tok_sum > worst_tok_diff) {
                worst_tok_diff = tok_sum;
                worst_tok = cur_tok;
            }
            tok_sum = 0;
            cur_tok = tok;
        }
        tok_sum += d;
    }
    const mean_abs = sum_abs / @as(f64, @floatFromInt(ref.len));
    std.debug.print("\nqwen vision parity: max_abs={d:.5} mean_abs={d:.6} (N_merged={d}, dim={d})\n", .{ max_abs, mean_abs, n / 4, dim });
    std.debug.print("  max@ token={d} chan={d} ours={d:.4} ref={d:.4}\n", .{ max_idx / dim, max_idx % dim, od[max_idx], ref[max_idx] });
    std.debug.print("  diffs>0.10: {d}/{d}  diffs>0.30: {d}  worst-token meanrow-diff token={d} sum={d:.3}\n", .{ gt_010, ref.len, gt_030, worst_tok, worst_tok_diff / @as(f64, @floatFromInt(dim)) });
    // bf16 ViT through 12 blocks with different reduction orders than the
    // reference: tolerate accumulated rounding + a handful of outliers, catch
    // real bugs (a structural bug blows up mean_abs by 10-100x).
    try std.testing.expect(mean_abs < 0.02);
    try std.testing.expect(gt_030 < ref.len / 1000); // <0.1% of elements may exceed 0.30
}

test "qwen forwardVideo == per-group forward()+concat (self-consistency)" {
    // No hermetic reference exists for ViT-forward CORRECTNESS (that needs a
    // trained checkpoint — see the QWEN_VISION_TEST_MODEL-gated parity test
    // above). What IS hermetically checkable, and what's new in this task, is
    // the forwardVideo WRAPPER: does it slice `grid_t` temporal-patch groups
    // out of the concatenated patches array and route each through the
    // existing, unmodified single-group `forward` correctly? This builds a
    // tiny synthetic tower (weight VALUES are arbitrary) and asserts
    // forwardVideo(3 groups) is bit-identical to manually slicing+forward()ing
    // each group and concatenating — the exact operation forwardVideo performs
    // internally, so any slicing/indexing bug in the wrapper shows up here.
    const a = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();

    const mkArr = struct {
        fn f(shape: []const c_int, val_base: f32) mlx.mlx_array {
            var buf: [64]f32 = undefined;
            var total: usize = 1;
            for (shape) |d| total *= @intCast(d);
            for (0..total) |i| buf[i] = val_base + @as(f32, @floatFromInt(i)) * 0.01;
            return mlx.mlx_array_new_data(&buf, shape.ptr, @intCast(shape.len), .float32);
        }
    }.f;

    // hidden=4, 1 head of head_dim=4, depth=1, merge=1 (no reduction — keeps
    // token counts trivial), grid 2x2 (n=4 patches/group) matching
    // num_grid_per_side=2 exactly (identity pos-embed interpolation).
    const hidden = [_]c_int{4};
    const hidden2 = [_]c_int{ 4, 4 };
    const qkv_w_shape = [_]c_int{ 12, 4 };
    const qkv_b_shape = [_]c_int{12};
    const patch_w_shape = [_]c_int{ 4, 1 };
    const pos_shape = [_]c_int{ 4, 4 };

    var qv = QwenVision{
        .s = s,
        .allocator = a,
        .depth = 1,
        .hidden = 4,
        .heads = 1,
        .head_dim = 4,
        .merge = 1,
        .num_grid_per_side = 2,
        .out_hidden = 4,
        .patch_w = mkArr(&patch_w_shape, 0.1),
        .patch_b = mkArr(&hidden, 0.0),
        .pos_embed = mkArr(&pos_shape, 0.2),
        .blocks = try a.alloc(QBlock, 1),
        .merger_norm_w = mkArr(&hidden, 1.0),
        .merger_norm_b = mkArr(&hidden, 0.0),
        .merger_fc1_w = mkArr(&hidden2, 0.05),
        .merger_fc1_b = mkArr(&hidden, 0.0),
        .merger_fc2_w = mkArr(&hidden2, 0.05),
        .merger_fc2_b = mkArr(&hidden, 0.0),
    };
    qv.blocks[0] = .{
        .norm1_w = mkArr(&hidden, 1.0),
        .norm1_b = mkArr(&hidden, 0.0),
        .norm2_w = mkArr(&hidden, 1.0),
        .norm2_b = mkArr(&hidden, 0.0),
        .qkv_w = mkArr(&qkv_w_shape, 0.03),
        .qkv_b = mkArr(&qkv_b_shape, 0.0),
        .proj_w = mkArr(&hidden2, 0.04),
        .proj_b = mkArr(&hidden, 0.0),
        .fc1_w = mkArr(&hidden2, 0.02),
        .fc1_b = mkArr(&hidden, 0.0),
        .fc2_w = mkArr(&hidden2, 0.02),
        .fc2_b = mkArr(&hidden, 0.0),
    };
    defer {
        _ = mlx.mlx_array_free(qv.patch_w);
        _ = mlx.mlx_array_free(qv.patch_b);
        _ = mlx.mlx_array_free(qv.pos_embed);
        _ = mlx.mlx_array_free(qv.merger_norm_w);
        _ = mlx.mlx_array_free(qv.merger_norm_b);
        _ = mlx.mlx_array_free(qv.merger_fc1_w);
        _ = mlx.mlx_array_free(qv.merger_fc1_b);
        _ = mlx.mlx_array_free(qv.merger_fc2_w);
        _ = mlx.mlx_array_free(qv.merger_fc2_b);
        for (qv.blocks) |b| {
            _ = mlx.mlx_array_free(b.norm1_w);
            _ = mlx.mlx_array_free(b.norm1_b);
            _ = mlx.mlx_array_free(b.norm2_w);
            _ = mlx.mlx_array_free(b.norm2_b);
            _ = mlx.mlx_array_free(b.qkv_w);
            _ = mlx.mlx_array_free(b.qkv_b);
            _ = mlx.mlx_array_free(b.proj_w);
            _ = mlx.mlx_array_free(b.proj_b);
            _ = mlx.mlx_array_free(b.fc1_w);
            _ = mlx.mlx_array_free(b.fc1_b);
            _ = mlx.mlx_array_free(b.fc2_w);
            _ = mlx.mlx_array_free(b.fc2_b);
        }
        a.free(qv.blocks);
    }

    // grid_t=3 groups of a 2x2 grid (n=4 patches/group, feat=1) — 12 rows total.
    const grid_t: u32 = 3;
    const n_per_group: usize = 4;
    var patches_buf: [12]f32 = undefined;
    for (0..12) |i| patches_buf[i] = @as(f32, @floatFromInt(i)) * 0.1;
    const all_shape = [_]c_int{ 12, 1 };
    const patches_all = mlx.mlx_array_new_data(&patches_buf, &all_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(patches_all);

    const out_video = try qv.forwardVideo(patches_all, grid_t, 2, 2);
    defer _ = mlx.mlx_array_free(out_video);

    // Manual reference: slice each group from the SAME source buffer, call
    // forward() directly, concatenate along the token axis.
    var manual_parts: [3]mlx.mlx_array = undefined;
    for (0..grid_t) |g| {
        const group_shape = [_]c_int{ @intCast(n_per_group), 1 };
        const group_arr = mlx.mlx_array_new_data(patches_buf[g * n_per_group ..].ptr, &group_shape, 2, .float32);
        defer _ = mlx.mlx_array_free(group_arr);
        manual_parts[g] = try qv.forward(group_arr, 2, 2);
    }
    defer for (manual_parts) |p| { _ = mlx.mlx_array_free(p); };
    const cat_vec = mlx.mlx_vector_array_new_data(&manual_parts, manual_parts.len);
    defer _ = mlx.mlx_vector_array_free(cat_vec);
    var out_manual = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out_manual);
    try mlx.check(mlx.mlx_concatenate_axis(&out_manual, cat_vec, 1, s));

    var v_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_f32);
    try mlx.check(mlx.mlx_astype(&v_f32, out_video, .float32, s));
    try mlx.check(mlx.mlx_array_eval(v_f32));
    var m_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m_f32);
    try mlx.check(mlx.mlx_astype(&m_f32, out_manual, .float32, s));
    try mlx.check(mlx.mlx_array_eval(m_f32));

    const v_shape = mlx.getShape(v_f32);
    const m_shape = mlx.getShape(m_f32);
    try std.testing.expectEqualSlices(c_int, m_shape, v_shape);
    try std.testing.expectEqual(@as(c_int, 1), v_shape[0]);
    try std.testing.expectEqual(@as(c_int, 12), v_shape[1]); // 3 groups × 4 merged tokens (merge=1)

    const v_data = mlx.mlx_array_data_float32(v_f32) orelse return error.TestUnexpectedNullData;
    const m_data = mlx.mlx_array_data_float32(m_f32) orelse return error.TestUnexpectedNullData;
    const total: usize = 12 * 4; // tokens × out_hidden
    try std.testing.expectEqualSlices(f32, m_data[0..total], v_data[0..total]);
}
