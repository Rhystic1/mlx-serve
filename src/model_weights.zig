//! Safetensors weight loading — the MLX-backed half of model.zig.
//!
//! Split out for the Windows/Linux port. `model.zig` above this is pure config
//! parsing (ModelConfig and everything the memory-billing helpers need), which
//! every build requires; a `Weights` map is `std.StringHashMap(mlx_array)` and
//! cannot exist without MLX. `src/model_weights_stub.zig` is the twin that a
//! `-Dgguf-only` build gets instead, and `model.zig` re-exports whichever is
//! active so callers are unchanged.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const ModelConfig = @import("model.zig").ModelConfig;
const shouldKeepWeightKey = @import("model.zig").shouldKeepWeightKey;

pub const Weights = struct {
    map: std.StringHashMap(mlx.mlx_array),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Weights {
        return .{
            .map = std.StringHashMap(mlx.mlx_array).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Weights) void {
        var it = self.map.iterator();
        while (it.next()) |entry| {
            _ = mlx.mlx_array_free(entry.value_ptr.*);
            self.allocator.free(entry.key_ptr.*);
        }
        self.map.deinit();
    }

    pub fn get(self: *const Weights, name: []const u8) ?mlx.mlx_array {
        return self.map.get(name);
    }

    pub fn count(self: *const Weights) u32 {
        return @intCast(self.map.count());
    }
};

/// The generic nestings a text trunk ships under: flat, mlx-community's
/// re-nest, and meta's VL original (Muse-Glimmer). `parseConfigFromJson`
/// picks from config KEYS; this probe corrects it from the checkpoint.
const FLAT_PREFIX = "model";
const NESTED_PREFIX = "language_model.model";
const VL_NESTED_PREFIX = "model.language_model";

fn hasWeightsUnder(weights: *const Weights, prefix: []const u8) bool {
    var it = weights.map.keyIterator();
    while (it.next()) |k| {
        const key = k.*;
        if (key.len > prefix.len and key[prefix.len] == '.' and std.mem.startsWith(u8, key, prefix)) return true;
    }
    return false;
}

/// Re-point `config.weight_prefix` at the nesting the CHECKPOINT actually uses.
///
/// Which of the two a converter emits is not reliably declared in config.json,
/// so `parseConfigFromJson` guesses from `text_config` presence — wrong for any
/// checkpoint that nests without declaring one (mlx-community LFM2.5-2.6B:
/// `Lfm2ForCausalLM`, an EMPTY `vision_config`, every weight under
/// `language_model.model.*`; the guess picked `model` and the load died on
/// `MISSING WEIGHT: model.embed_tokens.weight`). The class has now shipped in
/// both directions, so the weights get the last word.
///
/// Conservative by construction: only the generic spellings participate
/// (never an arch with its own — `backbone`, `model.llm`, `""`), and a swap
/// happens only when the configured one holds NOTHING, so every checkpoint
/// that already loaded binds byte-identically. Scan order puts the most
/// specific spelling first: a `model.language_model.*` checkpoint also
/// satisfies the bare "model" probe.
pub fn resolveWeightPrefix(config: *ModelConfig, weights: *const Weights) void {
    const candidates = [_][]const u8{ NESTED_PREFIX, VL_NESTED_PREFIX, FLAT_PREFIX };
    var known = false;
    for (candidates) |p| {
        if (std.mem.eql(u8, config.weight_prefix, p)) known = true;
    }
    if (!known) return;

    if (hasWeightsUnder(weights, config.weight_prefix)) return;
    for (candidates) |p| {
        if (std.mem.eql(u8, config.weight_prefix, p)) continue;
        if (!hasWeightsUnder(weights, p)) continue;
        log.info("weight prefix: config implies \"{s}\", checkpoint uses \"{s}\" — using the checkpoint's\n", .{ config.weight_prefix, p });
        config.weight_prefix = p;
        return;
    }
}

/// Load all safetensors files from model_dir.
/// When `load_vision` is true, vision_tower and multi_modal_projector weights are included.
pub fn loadWeights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Weights {
    return loadWeightsOpt(io, allocator, model_dir, false);
}

/// Load ONE safetensors file (absolute path) into a Weights map — for
/// sidecar files that live beside the trunk shards (e.g. a root-level
/// `mtp.safetensors`), where a directory scan would sweep in the trunk.
pub fn loadWeightsSingleFile(allocator: std.mem.Allocator, abs_path: []const u8) !Weights {
    var weights = Weights.init(allocator);
    errdefer weights.deinit();

    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    const pathz = try allocator.dupeSentinel(u8, abs_path, 0);
    defer allocator.free(pathz);
    try loadSafetensorsFile(allocator, &weights, pathz, s, false);

    if (weights.count() == 0) {
        log.err("no usable weights loaded from {s} — corrupt or empty safetensors file?\n", .{abs_path});
        return error.NoWeightFiles;
    }
    return weights;
}

pub fn loadWeightsWithVision(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Weights {
    return loadWeightsOpt(io, allocator, model_dir, true);
}

fn loadWeightsOpt(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, load_vision: bool) !Weights {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true });
    defer dir.close(io);
    return loadWeightsFromOpenDir(io, allocator, dir, model_dir, load_vision);
}

/// Load every `*.safetensors` in an already-open `dir` into a Weights map.
/// `model_dir` is the on-disk path string, used both to build the per-file
/// absolute path for `mlx_load_safetensors` and to phrase the error message.
/// Split out of `loadWeightsOpt` so the incomplete-checkpoint guard below is
/// unit-testable against a `tmpDir` (mirrors `model_discovery.discoverModelsInDir`).
fn loadWeightsFromOpenDir(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir, model_dir: []const u8, load_vision: bool) !Weights {
    var weights = Weights.init(allocator);
    errdefer weights.deinit();

    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    var file_count: u32 = 0;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        // Accept regular files AND symlinks: HuggingFace cache snapshots store
        // every weight file as a symlink into ../../blobs/<hash>. mlx_load_safetensors
        // resolves the link at the OS level, so a symlinked *.safetensors loads fine.
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;

        const path_slice = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_dir, entry.name });
        defer allocator.free(path_slice);
        const path = try allocator.dupeSentinel(u8, path_slice, 0);
        defer allocator.free(path);

        log.info("Loading {s}...\n", .{entry.name});
        try loadSafetensorsFile(allocator, &weights, path, s, load_vision);
        file_count += 1;
    }

    // Incomplete-checkpoint guard. A dir with config/tokenizer but no (or no
    // usable) *.safetensors is the classic interrupted-download shape: the
    // small files land first, the multi-GB weight shards never finalize. Before
    // this guard the loader returned an empty map and the caller crashed with a
    // misleading `MISSING WEIGHT: <prefix>.embed_tokens.weight` (the first
    // weight looked up) + `unreachable`, pointing at the model arch instead of
    // the download. Fail here with an actionable message, mirroring the
    // tokenizer path's "incomplete download?" hint (see main.zig).
    if (weights.count() == 0) {
        log.err("no usable weights loaded from {s} ({d} *.safetensors file(s) found) — the checkpoint looks like an incomplete download (config/tokenizer present, weight shards missing). Re-download the model (e.g. `mlx-serve pull <model>`) or delete the dir and re-fetch.\n", .{ model_dir, file_count });
        return error.NoWeightFiles;
    }

    log.info("Loaded {d} weights from {d} file(s)\n", .{ weights.count(), file_count });
    reportF16Narrowing();
    return weights;
}

/// Whether a just-loaded f16 tensor must be narrowed to the engine's bf16
/// activation dtype.
///
/// Two shapes qualify, for the same underlying reason — an f16 value that
/// meets a bf16 activation promotes the RESULT to f32:
///
///   - Quant SIDE tensors (scales/biases), which can be 2-D so they are keyed
///     on the suffix. f16 side tensors force gather_qmm/qmatmul onto a ~4x
///     slower mixed-dtype path (hy_v3 2-bit live, 2026-07-14: 0.70 vs 0.18 ms
///     per 8-expert gather — 1.2 tok/s on the 295B instead of ~15+).
///   - ANY 1-D f16 tensor: a per-channel table (norm weight, bias, A_log,
///     dt_bias) that is multiplied or added straight into the residual. Leave
///     one f16 and the residual turns f32 at the first layer and STAYS f32,
///     so every later weight read is upcast — the Laguna YaRN-mscale class,
///     one level up. Measured on prism-ml/Ternary-Bonsai-27B-mlx-2bit (the
///     only f16 checkpoint on hand, qwen3_5 GDN hybrid): 27.99 -> 23.88
///     ms/forward, 14.7%, three paired boots with cooldown.
///
/// Plain multi-dimensional WEIGHTS keep their dtype. They are matmul
/// OPERANDS, and MLX selects its kernel off that dtype, so narrowing one is a
/// kernel-selection change rather than a promotion fix — measured as a wash
/// here (23.15 vs 23.47 ms, inside boot-to-boot drift), so the minimal rule
/// is the one that ships.
///
/// The cast node stays lazy, so the load-time batch eval materializes bf16
/// directly. Delta from the 3 dropped mantissa bits: cos 0.99999994 — far
/// below any quant noise floor.
pub fn narrowsLoadedF16(key: []const u8, ndim: usize, dtype: mlx.mlx_dtype) bool {
    if (dtype != .float16) return false;
    if (std.mem.endsWith(u8, key, ".scales") or std.mem.endsWith(u8, key, ".biases")) return true;
    return ndim == 1;
}

/// Kill switch for the 1-D arm (`MLX_SERVE_F16_NARROW_1D=0`). A load-time
/// dtype normalization is invisible once the model is up, so a one-boot A/B
/// switch is the only way to attribute a future f16-checkpoint regression to
/// it. The side-tensor arm predates this and is not switchable.
var narrow_1d_env: ?bool = null;
fn narrow1dEnabled() bool {
    if (narrow_1d_env) |v| return v;
    const on = blk: {
        const raw = std.c.getenv("MLX_SERVE_F16_NARROW_1D") orelse break :blk true;
        break :blk !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
    };
    narrow_1d_env = on;
    return on;
}

/// Count of 1-D f16 tables narrowed this load — reported once per model so a
/// declined normalization is nameable from the log instead of silently
/// reading as "this checkpoint just isn't f16".
var narrowed_1d: usize = 0;

pub fn reportF16Narrowing() void {
    if (narrowed_1d == 0) return;
    log.info("[dtype] narrowed {d} 1-D f16 tables to bf16 (MLX_SERVE_F16_NARROW_1D=0 disables)\n", .{narrowed_1d});
    narrowed_1d = 0;
}

pub fn loadSafetensorsFile(
    allocator: std.mem.Allocator,
    weights: *Weights,
    path: [*:0]const u8,
    s: mlx.mlx_stream,
    load_vision: bool,
) !void {
    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);

    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);

    try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, path, s));

    const iter = mlx.mlx_map_string_to_array_iterator_new(tensor_map);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(iter);

    while (true) {
        var key: ?[*:0]const u8 = null;
        var value = mlx.mlx_array_new();

        const ret = mlx.mlx_map_string_to_array_iterator_next(&key, &value, iter);
        if (ret != 0 or key == null) {
            _ = mlx.mlx_array_free(value);
            break;
        }

        const key_str = std.mem.span(key.?);

        if (!shouldKeepWeightKey(key_str, load_vision)) {
            _ = mlx.mlx_array_free(value);
            continue;
        }

        // Read the shape BEFORE the cast frees `value` — a freed handle's
        // ndim is a use-after-free, not a zero.
        const ndim = mlx.mlx_array_ndim(value);
        var final_value = value;
        if (narrowsLoadedF16(key_str, ndim, mlx.mlx_array_dtype(value)) and
            (ndim != 1 or narrow1dEnabled()))
        {
            var cast = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&cast, value, .bfloat16, s));
            _ = mlx.mlx_array_free(value);
            final_value = cast;
            if (ndim == 1) narrowed_1d += 1;
        }

        const owned_key = try allocator.dupe(u8, key_str);
        try weights.map.put(owned_key, final_value);
    }
}

const testing = std.testing;

test "loadWeights casts f16 quant scales/biases to bf16 (mixed-dtype qmm slow-path class)" {
    // hy_v3 2-bit (ox-ox) ships F16 scales/biases beside bf16 activations —
    // MLX's gather_qmm/qmatmul take a ~4x slower mixed-dtype path (measured
    // 2026-07-14: 0.70 vs 0.18 ms per 8-expert gather; 1.2 tok/s on the 295B
    // instead of ~15+). The loader must cast quant SIDE tensors to bf16 once;
    // weights and non-quant tensors keep their dtype. Dequant delta from the
    // 3 dropped mantissa bits: cos 0.99999994 — under the 2-bit noise floor.
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp_dir.dir.realPath(io, &path_buf);
    const dir_path = path_buf[0..root_len];
    const st_path = try std.fmt.allocPrintSentinel(allocator, "{s}/model.safetensors", .{dir_path}, 0);
    defer allocator.free(st_path);

    // Build a tiny map: an f16 "scales", an f16 "biases", an f16 plain weight
    // (must NOT be cast), and a bf16 scales (no-op).
    {
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);

        const shape = [_]c_int{ 4, 4 };
        const data: [16]f32 = @splat(0.5);
        const f32_arr = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(f32_arr);
        var f16_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f16_arr);
        try mlx.check(mlx.mlx_astype(&f16_arr, f32_arr, .float16, s));
        var bf16_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bf16_arr);
        try mlx.check(mlx.mlx_astype(&bf16_arr, f32_arr, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(f16_arr));
        try mlx.check(mlx.mlx_array_eval(bf16_arr));

        _ = mlx.mlx_map_string_to_array_insert(map, "model.layers.0.mlp.gate_proj.scales", f16_arr);
        _ = mlx.mlx_map_string_to_array_insert(map, "model.layers.0.mlp.gate_proj.biases", f16_arr);
        _ = mlx.mlx_map_string_to_array_insert(map, "model.layers.0.mlp.up_proj.weight", f16_arr);
        _ = mlx.mlx_map_string_to_array_insert(map, "model.layers.0.mlp.down_proj.scales", bf16_arr);
        try mlx.check(mlx.mlx_save_safetensors(st_path.ptr, map, meta));
    }

    var weights = try loadWeights(io, allocator, dir_path);
    defer weights.deinit();

    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(weights.get("model.layers.0.mlp.gate_proj.scales").?));
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(weights.get("model.layers.0.mlp.gate_proj.biases").?));
    // A plain WEIGHT stays f16 (dense-f16 tables are legitimate — only the
    // quant side tensors force the mixed-dtype qmm path).
    try testing.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(weights.get("model.layers.0.mlp.up_proj.weight").?));
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(weights.get("model.layers.0.mlp.down_proj.scales").?));
}

test "loadWeights on a weightless dir (incomplete download) errors clearly, not empty map" {
    // Reproduces the live misdiagnosis: an interrupted `hf download`/`mlx-serve
    // pull` lands config + tokenizer but never finalizes the *.safetensors
    // weight shards. Before the guard, loadWeights returned an empty map and
    // the caller crashed with a misleading "MISSING WEIGHT:
    // model.embed_tokens.weight" (the first weight looked up) + `unreachable`,
    // pointing at the model arch instead of the incomplete checkpoint.
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"mistral\"}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = "{}" });
    // The index file names the shards but is NOT itself a weight file — it must
    // not be mistaken for one (it ends in .json, not .safetensors).
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{}" });

    try std.testing.expectError(
        error.NoWeightFiles,
        loadWeightsFromOpenDir(io, allocator, tmp.dir, "/incomplete-model", false),
    );
}

test "resolveWeightPrefix: the CHECKPOINT decides the nesting, not the config keys" {
    // mlx-community/LFM2.5-2.6B-{8bit,nvfp4} declare `Lfm2ForCausalLM` with NO
    // text_config (just an empty `vision_config`), yet ship every weight under
    // `language_model.model.*`. The config-key guess picked "model" and the
    // load died on `MISSING WEIGHT: model.embed_tokens.weight` (live
    // 2026-08-04). The same class shipped in the opposite direction before, so
    // the probe corrects either way.
    const allocator = testing.allocator;
    const put = struct {
        fn add(w: *Weights, alloc: std.mem.Allocator, key: []const u8) !void {
            const k = try alloc.dupe(u8, key);
            try w.map.put(k, mlx.mlx_array_new());
        }
    }.add;

    // Nested checkpoint, flat guess → re-pointed (the LFM2.5 crash).
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "language_model.model.embed_tokens.weight");
        try put(&w, allocator, "language_model.model.layers.0.self_attn.q_proj.weight");
        var config = ModelConfig{ .model_type = "lfm2", .weight_prefix = "model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    }
    // Flat checkpoint, nested guess → re-pointed the other way.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "lfm2", .weight_prefix = "language_model.model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("model", config.weight_prefix);
    }
    // Both present (a real VL checkpoint) → the configured prefix stands, so
    // nothing that loads today can be re-pointed by this probe.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.embed_tokens.weight");
        try put(&w, allocator, "language_model.model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "lfm2", .weight_prefix = "language_model.model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    }
    // An arch with its OWN prefix is never touched, even when it holds nothing
    // (a genuinely broken checkpoint must stay a clear MISSING WEIGHT).
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "language_model.model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "nemotron_h", .weight_prefix = "backbone" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("backbone", config.weight_prefix);
    }
    // A prefix that is a strict PREFIX of the key's first segment must not
    // count as a hit ("model" vs "model_extra.*").
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model_extra.embed_tokens.weight");
        try put(&w, allocator, "language_model.model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "lfm2", .weight_prefix = "model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    }
    // mlx-community/Muse-Glimmer-30B-4bit (live 2026-08-11): meta's config
    // keeps text_config, so the guess is the VL-original "model.language_model"
    // — but mlx_lm convert re-nests every text weight under
    // "language_model.model.*". The third spelling joins the probe.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "language_model.model.embed_tokens.weight");
        try put(&w, allocator, "language_model.lm_head.weight");
        try put(&w, allocator, "vision_tower.layers.0.norm1.weight");
        var config = ModelConfig{ .model_type = "muse_glimmer", .weight_prefix = "model.language_model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    }
    // Our own mirror layout (meta-original nesting) stays put.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.language_model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "muse_glimmer", .weight_prefix = "model.language_model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("model.language_model", config.weight_prefix);
    }
    // Ordering: a "model.language_model.*" checkpoint ALSO matches the bare
    // "model" probe (the '.' check passes at "model.language_model"), so the
    // most specific spelling must win the scan.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.language_model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "muse_glimmer", .weight_prefix = "language_model.model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("model.language_model", config.weight_prefix);
    }
}

test "narrowsLoadedF16 catches per-channel tables, not matmul operands" {
    // Quant side tensors: the pre-existing rule, keyed on the suffix because
    // they can be 2-D.
    try testing.expect(narrowsLoadedF16("model.layers.0.mlp.down_proj.scales", 2, .float16));
    try testing.expect(narrowsLoadedF16("model.layers.0.mlp.down_proj.biases", 2, .float16));

    // Any 1-D f16 tensor is a PER-CHANNEL table — a norm weight, a bias, a
    // gate table. It gets multiplied or added straight into the activation
    // stream, so leaving it f16 beside a bf16 residual promotes the residual
    // (and therefore every later weight read) to f32.
    try testing.expect(narrowsLoadedF16("language_model.model.layers.0.input_layernorm.weight", 1, .float16));
    try testing.expect(narrowsLoadedF16("language_model.model.layers.0.linear_attn.A_log", 1, .float16));
    try testing.expect(narrowsLoadedF16("language_model.model.layers.0.linear_attn.dt_bias", 1, .float16));
    try testing.expect(narrowsLoadedF16("language_model.model.norm.weight", 1, .float16));

    // A 2-D dense f16 weight is a MATMUL OPERAND, not a table. MLX picks its
    // kernel off that dtype, so narrowing it is a kernel-selection change and
    // not this rule's business — it stays per-site.
    try testing.expect(!narrowsLoadedF16("vision_tower.blocks.0.attn.qkv.weight", 2, .float16));
    try testing.expect(!narrowsLoadedF16("language_model.model.layers.0.linear_attn.conv1d.weight", 3, .float16));

    // Everything already in the engine's dtype, and packed weights, are left
    // alone.
    try testing.expect(!narrowsLoadedF16("model.layers.0.input_layernorm.weight", 1, .bfloat16));
    try testing.expect(!narrowsLoadedF16("model.layers.0.mlp.down_proj.weight", 2, .uint32));
    try testing.expect(!narrowsLoadedF16("model.layers.0.mlp.down_proj.scales", 2, .bfloat16));
}
