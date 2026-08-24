//! Live tower-parity test for the LFM2-VL vision encoder.
//!
//! Lives in its own file, imported only by the MLX test suite
//! (src/tests_mlx.zig), because it is the one test in lfm2_vision.zig that
//! touches the MLX TOWER. `server.zig` imports lfm2_vision.zig for its pure
//! preprocessing (smartResize / gridLayout / buildPixelValues), and Zig
//! compiles the tests of every file reachable from the test root's import
//! graph -- so leaving this test beside them pulled the whole tower, and MLX
//! with it, into a -Dgguf-only test binary.
//!
//! Env-gated and skipped without a fixture, exactly as before.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const lfm2_vision = @import("lfm2_vision.zig");
const qwen_vision = @import("qwen_vision.zig");
const Lfm2Vision = lfm2_vision.Lfm2Vision;
const log = @import("log.zig");

// Vision-tower parity vs the EXECUTED reference
// (tests/dump_lfm2_vision_fixtures.py runs transformers' own Siglip2VisionModel
// plus LFM2-VL's projector on OUR pack's weights — which ship dense bf16 in
// every quant width — so a diff is a layout/math bug, never quantization error).
//
//   LFM2_VISION_MODEL="/Volumes/G Drive SSD/models-dl/LiquidAI/LFM2.5-VL-3B-MLX-4bit" \
//   LFM2_VISION_FIXTURE=~/claude-tmp/lfm2-vision/lfm2_vision_fixture.safetensors \
//   zig build test -Doptimize=ReleaseFast -Dtest-filter="lfm2 vision live"
test "lfm2 vision live: tower parity vs the executed reference" {
    const raw_model = std.c.getenv("LFM2_VISION_MODEL") orelse return error.SkipZigTest;
    const raw_fix = std.c.getenv("LFM2_VISION_FIXTURE") orelse return error.SkipZigTest;
    const model_dir = std.mem.sliceTo(raw_model, 0);
    const fix_path = std.mem.sliceTo(raw_fix, 0);
    if (model_dir.len == 0 or fix_path.len == 0) return error.SkipZigTest;
    const a = testing.allocator;

    const config = try model_mod.parseConfig(std.testing.io, a, model_dir);
    var weights = try model_mod.loadWeightsWithVision(std.testing.io, a, model_dir);
    defer weights.deinit();
    var fx = try model_mod.loadWeightsSingleFile(a, fix_path);
    defer fx.deinit();

    var lv = try Lfm2Vision.init(a, config, &weights);
    defer lv.deinit();
    const s = lv.s;

    // The position resample runs before every block, so a diff here makes
    // nothing downstream attributable. Both directions are covered: 32x32 only
    // upsamples the stored 16x16 table, 32x8 and 8x32 DOWNSAMPLE one axis,
    // which is the only place the anti-aliasing footprint changes the answer
    // (mlx-vlm's bicubic scores cos 0.99 / rms 1.13 against this).
    for ([_][2]u32{ .{ 14, 20 }, .{ 32, 32 }, .{ 32, 8 }, .{ 8, 32 }, .{ 26, 36 }, .{ 16, 64 } }) |g| {
        var name_buf: [32]u8 = undefined;
        const key = try std.fmt.bufPrint(&name_buf, "pos_{d}x{d}", .{ g[0], g[1] });
        const want = fx.get(key) orelse continue;
        const pos = try lv.posEmbed(g[0], g[1]);
        defer _ = mlx.mlx_array_free(pos);
        const c = try cosineSim(pos, want, s);
        const r = try rmsRatio(pos, want, s);
        std.debug.print("[lfm2-vit] {s} cos={d:.6} rms_ratio={d:.4}\n", .{ key, c, r });
        try testing.expect(c > 0.9999 and r > 0.995 and r < 1.005);
    }

    for ([_][]const u8{ "a", "b" }) |case| {
        var kb: [32]u8 = undefined;
        const grid_arr = fx.get(try std.fmt.bufPrint(&kb, "{s}_grid", .{case})) orelse return error.MissingFixtureTensor;
        try mlx.check(mlx.mlx_array_eval(grid_arr));
        const g: [*]const i32 = @ptrCast(@alignCast(mlx.mlx_array_data_int32(grid_arr)));
        const gh: u32 = @intCast(g[0]);
        const gw: u32 = @intCast(g[1]);
        const pv = fx.get(try std.fmt.bufPrint(&kb, "{s}_pixel_values", .{case})) orelse return error.MissingFixtureTensor;

        const hidden = try lv.towerHidden(pv, gh, gw);
        defer _ = mlx.mlx_array_free(hidden);
        {
            const want = fx.get(try std.fmt.bufPrint(&kb, "{s}_hidden", .{case})) orelse return error.MissingFixtureTensor;
            const c = try cosineSim(hidden, want, s);
            const r = try rmsRatio(hidden, want, s);
            std.debug.print("[lfm2-vit] {s} grid {d}x{d} hidden cos={d:.6} rms_ratio={d:.4}\n", .{ case, gh, gw, c, r });
            try testing.expect(c > 0.99 and r > 0.98 and r < 1.02);
        }

        const feats = try lv.project(hidden, gh, gw);
        defer _ = mlx.mlx_array_free(feats);
        const want = fx.get(try std.fmt.bufPrint(&kb, "{s}_features", .{case})) orelse return error.MissingFixtureTensor;
        const c = try cosineSim(feats, want, s);
        // MAGNITUDE too: these rows are spliced into the token stream, which is
        // exactly where a scale error hides from a cosine.
        const r = try rmsRatio(feats, want, s);
        std.debug.print("[lfm2-vit] {s} features cos={d:.6} rms_ratio={d:.4}\n", .{ case, c, r });
        try testing.expect(r > 0.98 and r < 1.02);
        // Ours serves bf16 against an fp32 reference through 27 blocks, so the
        // bar is "same features", not bit equality — a layout bug lands far below.
        try testing.expect(c > 0.99);
    }
}

fn cosineSim(a_arr: mlx.mlx_array, b_arr: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    const dot = try sumSq(a_arr, b_arr, s);
    const na = try sumSq(a_arr, a_arr, s);
    const nb = try sumSq(b_arr, b_arr, s);
    if (!std.math.isFinite(dot) or na <= 0 or nb <= 0) return std.math.nan(f32);
    return dot / (@sqrt(na) * @sqrt(nb));
}

fn rmsRatio(a_arr: mlx.mlx_array, b_arr: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    const na = try sumSq(a_arr, a_arr, s);
    const nb = try sumSq(b_arr, b_arr, s);
    if (!std.math.isFinite(na) or nb <= 0) return std.math.nan(f32);
    return @sqrt(na) / @sqrt(nb);
}

/// sum(a*b) in fp32 over flattened inputs. NaN propagates: `NaN > threshold` is
/// false, so an all-NaN candidate can never pass the comparisons above.
fn sumSq(a_arr: mlx.mlx_array, b_arr: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var af = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(af);
    try mlx.check(mlx.mlx_astype(&af, a_arr, .float32, s));
    var bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bf);
    try mlx.check(mlx.mlx_astype(&bf, b_arr, .float32, s));
    const n = [_]c_int{@intCast(mlx.mlx_array_size(af))};
    var a1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a1);
    try mlx.check(mlx.mlx_reshape(&a1, af, &n, 1, s));
    var b1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b1);
    try mlx.check(mlx.mlx_reshape(&b1, bf, &n, 1, s));
    var prod = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prod);
    try mlx.check(mlx.mlx_multiply(&prod, a1, b1, s));
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_sum(&o, prod, false, s));
    try mlx.check(mlx.mlx_array_eval(o));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, o));
    return v;
}
