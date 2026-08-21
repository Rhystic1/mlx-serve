//! Live tower-parity test for the Muse-Glimmer vision encoder.
//!
//! Own file, imported only by the MLX suite (src/tests_mlx.zig): it is the one
//! test in muse_vision.zig that touches the MLX TOWER, while
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
const muse_vision = @import("muse_vision.zig");
const qwen_vision = @import("qwen_vision.zig");

// Vision-tower parity vs the EXECUTED reference
// (tests/dump_muse_vision_fixture.py runs transformers' own muse_glimmer vision
// half on OUR dequantized weights, so a diff is a layout/math bug, never
// quantization error).
//
//   MUSE_VISION_MODEL=~/.mlx-serve/models/ddalcu/Muse-Glimmer-30B-MLX-Serve-8bit \
//   MUSE_VISION_FIXTURE=~/claude-tmp/muse-vision/muse_vision_fixture.safetensors \
//   zig build test -Doptimize=ReleaseFast -Dtest-filter="muse vision parity"
test "muse vision live: tower parity vs the executed reference" {
    const raw_model = std.c.getenv("MUSE_VISION_MODEL") orelse return error.SkipZigTest;
    const raw_fix = std.c.getenv("MUSE_VISION_FIXTURE") orelse return error.SkipZigTest;
    const model_dir = std.mem.sliceTo(raw_model, 0);
    const fix_path = std.mem.sliceTo(raw_fix, 0);
    if (model_dir.len == 0 or fix_path.len == 0) return error.SkipZigTest;
    const a = testing.allocator;

    const config = try model_mod.parseConfig(std.testing.io, a, model_dir);
    var weights = try model_mod.loadWeightsWithVision(std.testing.io, a, model_dir);
    defer weights.deinit();
    var fx = try model_mod.loadWeightsSingleFile(a, fix_path);
    defer fx.deinit();

    var mv = try MuseVision.init(a, config, &weights);
    defer mv.deinit();
    const s = mv.s;

    const grid = fx.get("grid_thw") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(grid));
    const g: [*]const i32 = @ptrCast(@alignCast(mlx.mlx_array_data_int32(grid)));
    const gh: u32 = @intCast(g[1]);
    const gw: u32 = @intCast(g[2]);
    const pv = fx.get("pixel_values") orelse return error.MissingFixtureTensor;

    // Position interpolation first: it runs before every block, so if it
    // differs, nothing downstream can be attributed to the tower.
    {
        const pos = try mv.posEmbed(gh, gw);
        defer _ = mlx.mlx_array_free(pos);
        const want = fx.get("pos_embeds") orelse return error.MissingFixtureTensor;
        const c = try cosineSim(pos, want, s);
        const r = try rmsRatio(pos, want, s);
        std.debug.print("[muse-vit] pos_embeds cos={d:.6} rms_ratio={d:.4}\n", .{ c, r });
        try testing.expect(c > 0.999 and r > 0.99 and r < 1.01);
    }

    const out = try mv.forward(pv, gh, gw);
    defer _ = mlx.mlx_array_free(out);
    const want = fx.get("features") orelse return error.MissingFixtureTensor;
    const c = try cosineSim(out, want, s);
    // MAGNITUDE too: these rows are concatenated into the token stream, where a
    // scale error is exactly the bug a cosine cannot see. (The perception norm
    // pins the RMS at 1, so the ratio also catches a missing norm.)
    const r = try rmsRatio(out, want, s);
    std.debug.print("[muse-vit] features cos={d:.6} rms_ratio={d:.4}\n", .{ c, r });
    try testing.expect(r > 0.99 and r < 1.01);
    // Ours serves bf16 against an fp32 reference through 50 blocks, so the bar
    // is "same features", not bit equality — a layout bug lands far below this.
    try testing.expect(c > 0.99);
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

// ── Tests ──

const testing = std.testing;
