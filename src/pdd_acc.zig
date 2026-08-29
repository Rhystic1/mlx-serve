//! MiniMax-H3 Parallel Decoding Distillation (PDD) Acc — host math.
//!
//! An Acc file is NOT a style LoRA. It carries a rank-64 trunk adapter PLUS a
//! 32-interval PDD output-head bank. Copying the packed bank onto the native
//! `[96, 5376]` / `[32, 5376]` heads is the Comfy crash; fusion + per-step
//! indexing is the distill. Math matches `pdd_acc_core.py` in
//! Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc (and alibaba-pai's
//! `reference_minimax_h3_pdd.py`): do not invent fusion weights.
//!
//! This file is CPU-only. mlx load / File construction lives in `lora.zig`.

const std = @import("std");

pub const FINE_STEPS: u32 = 32;
pub const TRAINED_BLOCK: u32 = 4;
pub const KNOT_TOLERANCE: f64 = 1e-6;
pub const VIDEO_SHIFT: f64 = 12.0;
pub const AUDIO_SHIFT: f64 = 3.0;
pub const NATIVE_VIDEO_OUT: u32 = 96;
pub const NATIVE_AUDIO_OUT: u32 = 32;
pub const HIDDEN: u32 = 5376;
pub const CONVERTED_FORMAT = "minimax_h3_pdd_acc_comfyui_v1";
pub const HF_REPO_URL = "https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs";
pub const HF_REPO_ID = "alibaba-pai/MiniMax-H3-Acc-LoRAs";
pub const FL2VA_FILENAME = "MiniMax-H3-FL2VA-Acc-8Step.safetensors";
pub const REF2VA_FILENAME = "MiniMax-H3-Ref2VA-Acc-8Step.safetensors";
/// Acc convert (Comfy / alibaba-pai) emits 50×5 trunk + 2×4 refiner = 258.
/// Turbo's 259th is `final_layer.adaln_proj.linear`, which Acc files omit.
pub const ACC_TRUNK_TARGETS: u32 = 50 * 5 + 2 * 4;
/// ~1.4 GB bf16 on disk, billed on the DiT term when Acc is engaged.
pub const ACC_LORA_BYTES: u64 = 1500 * 1024 * 1024;

pub const HEAD_KEYS = [_][]const u8{
    "proj_out.weight",
    "proj_out.bias",
    "audio_proj_out.weight",
    "audio_proj_out.bias",
};

pub const Partition = enum { fl2va, ref2va };
pub const OffGrid = enum { err, clamp };
pub const SourceFormat = enum { original, converted };

pub const AccError = error{
    AccTurboExclusive,
    AccPartitionMismatch,
    AccBadPartition,
    AccOffGrid,
    AccMissingHeads,
    AccBadHeadShape,
    AccNativeHeadOverwrite,
    AccLeftoverKeys,
    AccFileMissing,
    AccIncomplete,
};

pub fn shiftedSigma(shift: f64, t: f64) f64 {
    return shift * t / (1.0 + (shift - 1.0) * t);
}

/// `linspace(1, 0, num_steps+1)` mapped through `shift`. Matches
/// `pdd_acc_core.fine_sigmas` and mlx-serve `sigmaSchedule(num_steps, shift)`.
pub fn fineSigmas(shift: f64, num_steps: u32, out: []f64) void {
    std.debug.assert(out.len == num_steps + 1);
    const n: f64 = @floatFromInt(num_steps);
    for (0..num_steps + 1) |i| {
        const t = @as(f64, @floatFromInt(num_steps - i)) / n;
        out[i] = shiftedSigma(shift, t);
    }
}

pub fn isHeadKey(key: []const u8) bool {
    inline for (HEAD_KEYS) |h| {
        if (std.mem.eql(u8, key, h)) return true;
    }

    return false;
}

pub fn isLoraRoleKey(key: []const u8) bool {
    const suffixes = .{
        ".lora_A.weight",
        ".lora_B.weight",
        ".lora_down.weight",
        ".lora_up.weight",
        ".lora_down",
        ".lora_up",
        ".lora_A",
        ".lora_B",
        ".alpha",
    };
    inline for (suffixes) |sf| {
        if (std.mem.endsWith(u8, key, sf)) return true;
    }

    return false;
}

pub const KeyClass = enum { head, lora, leftover };

pub fn classifyKey(key: []const u8) KeyClass {
    if (isHeadKey(key)) return .head;
    if (isLoraRoleKey(key)) return .lora;

    return .leftover;
}

pub const SplitSummary = struct {
    n_heads: u32 = 0,
    n_lora: u32 = 0,
    n_leftover: u32 = 0,
    all_heads: bool = false,
    source: ?SourceFormat = null,
};

pub fn splitKeySet(keys: []const []const u8) SplitSummary {
    var s: SplitSummary = .{};
    var seen_heads: u32 = 0;
    var converted = false;
    var original = false;
    for (keys) |k| {
        switch (classifyKey(k)) {
            .head => {
                s.n_heads += 1;
                for (HEAD_KEYS, 0..) |h, i| {
                    if (std.mem.eql(u8, k, h)) seen_heads |= @as(u32, 1) << @intCast(i);
                }
            },
            .lora => {
                s.n_lora += 1;
                if (std.mem.startsWith(u8, k, "diffusion_model.")) converted = true;
                if (std.mem.indexOf(u8, k, "transformer_blocks.") != null or
                    std.mem.indexOf(u8, k, ".lora_down") != null) original = true;
            },
            .leftover => s.n_leftover += 1,
        }
    }
    s.all_heads = seen_heads == 0b1111;
    if (converted) s.source = .converted else if (original) s.source = .original;

    return s;
}

pub fn detectFromFilename(name: []const u8) bool {
    if (containsIgnoreCase(name, "Acc-8Step")) return true;
    if (containsIgnoreCase(name, "pdd_acc")) return true;

    return false;
}

pub fn detectFromMetadataFormat(format: []const u8) bool {
    return std.mem.eql(u8, format, CONVERTED_FORMAT);
}

/// Acc if any head key exists, the converted-format tag is set, or the
/// filename matches the Acc artifacts.
pub fn detectPdd(filename: []const u8, format: ?[]const u8, keys: []const []const u8) bool {
    if (detectFromFilename(filename)) return true;
    if (format) |f| {
        if (detectFromMetadataFormat(f)) return true;
    }
    for (keys) |k| {
        if (isHeadKey(k)) return true;
    }

    return false;
}

fn containsIgnoreCase(hay: []const u8, needle: []const u8) bool {
    if (needle.len > hay.len) return false;
    var i: usize = 0;
    while (i + needle.len <= hay.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(hay[i .. i + needle.len], needle)) return true;
    }

    return false;
}

/// Unique `fl2va` / `ref2va` hit. Metadata wins over the filename. Ambiguous
/// or absent → null (do not guess; cross-trunk apply renders silently wrong).
pub fn partitionFromName(meta_partition: ?[]const u8, filename: []const u8) ?Partition {
    if (hitPartition(meta_partition)) |p| return p;
    return hitPartition(filename);
}

fn hitPartition(s: ?[]const u8) ?Partition {
    const t = s orelse return null;
    const fl = containsIgnoreCase(t, "fl2va");
    const rf = containsIgnoreCase(t, "ref2va");
    if (fl and !rf) return .fl2va;
    if (rf and !fl) return .ref2va;

    return null;
}

pub fn refuseTurbo(acc: bool, turbo: bool) AccError!void {
    if (acc and turbo) return error.AccTurboExclusive;
}

pub fn checkPartitionPairing(file: ?Partition, model: Partition) AccError!void {
    const f = file orelse return;
    if (f != model) return error.AccPartitionMismatch;
}

pub fn isDistillFilename(name: []const u8) bool {
    if (detectFromFilename(name)) return true;
    if (containsIgnoreCase(name, "turbo_lora")) return true;
    if (containsIgnoreCase(name, "Turbo-Lora")) return true;
    if (containsIgnoreCase(name, "turbo_4step")) return true;

    return false;
}

/// Packed 2-D `[N*out, in]` vs native `[out, in]`. Copying a packed bank onto
/// the native head is the Comfy crash.
pub fn bankWouldOverwriteNative(shape: []const usize, native_out: usize) bool {
    if (shape.len == 2 and shape[0] != native_out) return true;

    return false;
}

/// Legal Acc head geometries: `[N, out, in]` or packed `[N*out, in]`.
pub fn parseHeadWeightShape(shape: []const usize, native_out: usize) AccError!struct { n: usize, out: usize, in: usize } {
    if (shape.len == 3) {
        if (shape[1] != native_out) return error.AccBadHeadShape;

        return .{ .n = shape[0], .out = shape[1], .in = shape[2] };
    }
    if (shape.len == 2) {
        if (shape[0] == native_out) return error.AccNativeHeadOverwrite;
        if (shape[0] % native_out != 0) return error.AccBadHeadShape;

        return .{ .n = shape[0] / native_out, .out = native_out, .in = shape[1] };
    }

    return error.AccBadHeadShape;
}

pub fn parseHeadBiasShape(shape: []const usize, native_out: usize) AccError!struct { n: usize, out: usize } {
    if (shape.len == 2) {
        if (shape[1] != native_out) return error.AccBadHeadShape;

        return .{ .n = shape[0], .out = shape[1] };
    }
    if (shape.len == 1) {
        if (shape[0] == native_out) return error.AccNativeHeadOverwrite;
        if (shape[0] % native_out != 0) return error.AccBadHeadShape;

        return .{ .n = shape[0] / native_out, .out = native_out };
    }

    return error.AccBadHeadShape;
}

/// Allowed block sizes are 4 or 8 only; they must sum to `n_fine` (32).
pub fn resolvePartition(n_fine: u32, nfe: u32, text: ?[]const u8, out: []u32) AccError![]const u32 {
    if (text) |raw| {
        const trimmed = std.mem.trim(u8, raw, " \t");
        if (trimmed.len == 0) return error.AccBadPartition;
        var n: usize = 0;
        var it = std.mem.splitScalar(u8, trimmed, ',');
        while (it.next()) |tok| {
            const t = std.mem.trim(u8, tok, " \t");
            if (t.len == 0) continue;
            const v = std.fmt.parseInt(u32, t, 10) catch return error.AccBadPartition;
            if (v != 4 and v != 8) return error.AccBadPartition;
            if (n >= out.len) return error.AccBadPartition;
            out[n] = v;
            n += 1;
        }
        if (n == 0) return error.AccBadPartition;
        var sum: u32 = 0;
        for (out[0..n]) |s| sum += s;
        if (sum != n_fine) return error.AccBadPartition;

        return out[0..n];
    }
    if (nfe < 4) return error.AccBadPartition;
    if (n_fine % nfe != 0) return error.AccBadPartition;
    const size = n_fine / nfe;
    if (size != 4 and size != 8) return error.AccBadPartition;
    if (nfe > out.len) return error.AccBadPartition;
    for (out[0..nfe]) |*s| s.* = size;

    return out[0..nfe];
}

pub fn blockBoundaries(n_fine: u32, sizes: []const u32, out: []f64) void {
    var fine: [FINE_STEPS + 1]f64 = undefined;
    fineSigmas(VIDEO_SHIFT, n_fine, fine[0 .. n_fine + 1]);
    out[0] = fine[0];
    var k: u32 = 0;
    for (sizes, 1..) |sz, i| {
        k += sz;
        out[i] = fine[k];
    }
}

pub fn selectBlock(sigma: f64, bounds: []const f64, mode: OffGrid) AccError!u32 {
    const nfe = bounds.len - 1;
    if (nfe == 0) return error.AccOffGrid;
    // Terminal knot (0) maps to the last block; it is never evaluated.
    if (@abs(sigma - bounds[nfe]) <= KNOT_TOLERANCE) return @intCast(nfe - 1);
    var i: usize = 0;
    while (i < nfe) : (i += 1) {
        if (@abs(sigma - bounds[i]) <= KNOT_TOLERANCE) return @intCast(i);
    }
    return switch (mode) {
        .err => error.AccOffGrid,
        .clamp => blk: {
            if (sigma >= bounds[0]) break :blk 0;
            var b: u32 = 0;
            while (b < nfe) : (b += 1) {
                if (sigma <= bounds[b] and sigma > bounds[b + 1]) break :blk b;
            }
            break :blk @intCast(nfe - 1);
        },
    };
}

/// Fuse 32 fine heads → `sizes.len` block heads. `w` is `[N, out, in]`, `b`
/// is `[N, out]`, both row-major f32. Output `out_w` is `[nfe, out, in]`.
pub fn fuseHeads(
    w: []const f32,
    b: []const f32,
    n: usize,
    out_dim: usize,
    in_dim: usize,
    fine: []const f64,
    sizes: []const u32,
    out_w: []f32,
    out_b: []f32,
) void {
    std.debug.assert(fine.len == n + 1);
    const win = out_dim * in_dim;
    var start: usize = 0;
    for (sizes, 0..) |sz, blk| {
        var span: f64 = 0;
        var k: usize = 0;
        while (k < sz) : (k += 1) {
            span += fine[start + k] - fine[start + k + 1];
        }
        const dst_w = out_w[blk * win ..][0..win];
        const dst_b = out_b[blk * out_dim ..][0..out_dim];
        @memset(dst_w, 0);
        @memset(dst_b, 0);
        k = 0;
        while (k < sz) : (k += 1) {
            const wt: f32 = @floatCast((fine[start + k] - fine[start + k + 1]) / span);
            const src_w = w[(start + k) * win ..][0..win];
            const src_b = b[(start + k) * out_dim ..][0..out_dim];
            for (dst_w, src_w) |*d, s| d.* += wt * s;
            for (dst_b, src_b) |*d, s| d.* += wt * s;
        }
        start += sz;
    }
}

pub fn concatRows(a: std.mem.Allocator, parts: []const []const f32, rows: []const usize, cols: usize) ![]f32 {
    var n_rows: usize = 0;
    for (rows) |r| n_rows += r;
    const out = try a.alloc(f32, n_rows * cols);
    var off: usize = 0;
    for (parts, rows) |p, r| {
        const n = r * cols;
        @memcpy(out[off..][0..n], p[0..n]);
        off += n;
    }

    return out;
}

/// Block-diagonal pack of `parts` each `[o, r]` → `[n*o, n*r]`.
pub fn blockDiag(a: std.mem.Allocator, parts: []const []const f32, o: usize, r: usize) ![]f32 {
    const n = parts.len;
    const out = try a.alloc(f32, n * o * n * r);
    @memset(out, 0);
    const stride = n * r;
    for (parts, 0..) |p, i| {
        var row: usize = 0;
        while (row < o) : (row += 1) {
            const dst_row = (i * o + row) * stride + i * r;
            @memcpy(out[dst_row..][0..r], p[row * r ..][0..r]);
        }
    }

    return out;
}

/// SwiGLU `[value; gate]` → `[gate; value]`: half-swap on axis 0.
pub fn halfSwapRows(a: std.mem.Allocator, src: []const f32, rows: usize, cols: usize) ![]f32 {
    std.debug.assert(rows % 2 == 0);
    const half = rows / 2;
    const out = try a.alloc(f32, rows * cols);
    @memcpy(out[0 .. half * cols], src[half * cols ..]);
    @memcpy(out[half * cols ..], src[0..half * cols]);

    return out;
}

pub const ConvertedLora = struct {
    module: []const u8, // owned
    a: []f32, // [r, in] or [3r, in] for qkv
    a_rows: usize,
    a_cols: usize,
    b: []f32, // [out, r] or [3o, 3r]
    b_rows: usize,
    b_cols: usize,
    alpha: f32,

    pub fn deinit(self: *ConvertedLora, allocator: std.mem.Allocator) void {
        allocator.free(self.module);
        allocator.free(self.a);
        allocator.free(self.b);
    }
};

pub const HostMat = struct {
    data: []const f32,
    rows: usize,
    cols: usize,
};

/// qkv: concat lora_A rows [q;k;v], block-diagonal lora_B, alpha×3.
pub fn convertQkv(
    allocator: std.mem.Allocator,
    downs: [3]HostMat, // each [r, in]
    ups: [3]HostMat, // each [o, r]
    alpha: f32,
) !struct { a: []f32, b: []f32, a_rows: usize, a_cols: usize, b_rows: usize, b_cols: usize, alpha: f32 } {
    const r = downs[0].rows;
    const in_dim = downs[0].cols;
    const o = ups[0].rows;
    var parts_a: [3][]const f32 = undefined;
    var rows_a: [3]usize = undefined;
    var parts_b: [3][]const f32 = undefined;
    for (0..3) |i| {
        parts_a[i] = downs[i].data;
        rows_a[i] = downs[i].rows;
        parts_b[i] = ups[i].data;
    }
    const a = try concatRows(allocator, &parts_a, &rows_a, in_dim);
    errdefer allocator.free(a);
    const b = try blockDiag(allocator, &parts_b, o, r);

    return .{
        .a = a,
        .b = b,
        .a_rows = 3 * r,
        .a_cols = in_dim,
        .b_rows = 3 * o,
        .b_cols = 3 * r,
        .alpha = alpha * 3.0,
    };
}

pub fn convertFc1Up(allocator: std.mem.Allocator, b: HostMat) ![]f32 {
    return halfSwapRows(allocator, b.data, b.rows, b.cols);
}

/// Safetensors header peek: 8-byte LE length + JSON. No mlx.
pub const HeaderPeek = struct {
    format: ?[]u8 = null,
    pdd_partition: ?[]u8 = null,
    lora_alpha: f32 = 64.0,
    keys: [][]u8,
    json: []u8,

    pub fn deinit(self: *HeaderPeek, allocator: std.mem.Allocator) void {
        if (self.format) |f| allocator.free(f);
        if (self.pdd_partition) |p| allocator.free(p);
        for (self.keys) |k| allocator.free(k);
        allocator.free(self.keys);
        allocator.free(self.json);
    }

    pub fn keySlices(self: *const HeaderPeek, buf: [][]const u8) []const []const u8 {
        const n = @min(self.keys.len, buf.len);
        for (self.keys[0..n], 0..) |k, i| buf[i] = k;

        return buf[0..n];
    }
};

pub fn peekSafetensors(allocator: std.mem.Allocator, path: []const u8) !HeaderPeek {
    const io = std.Io.Threaded.global_single_threaded.io();
    const f = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return error.AccFileMissing;
    defer f.close(io);
    var rbuf: [4096]u8 = undefined;
    var rs = f.reader(io, &rbuf);
    var len_buf: [8]u8 = undefined;
    rs.interface.readSliceAll(&len_buf) catch return error.AccFileMissing;
    const header_len = std.mem.readInt(u64, &len_buf, .little);
    if (header_len == 0 or header_len > 64 * 1024 * 1024) return error.AccFileMissing;
    const json = try allocator.alloc(u8, @intCast(header_len));
    errdefer allocator.free(json);
    rs.interface.readSliceAll(json) catch return error.AccFileMissing;

    var keys: std.ArrayList([]u8) = .empty;
    errdefer {
        for (keys.items) |k| allocator.free(k);
        keys.deinit(allocator);
    }
    var format: ?[]u8 = null;
    var pdd_partition: ?[]u8 = null;
    var lora_alpha: f32 = 64.0;

    // Top-level keys only: a safetensors header is `{ "name": { ... }, ... }`.
    // Nested fields (`dtype`, `shape`, `data_offsets`) are not tensor names.
    var i: usize = 0;
    while (i < json.len) : (i += 1) {
        if (json[i] != '"') continue;
        const start = i + 1;
        var j = start;
        var escaped = false;
        while (j < json.len) : (j += 1) {
            if (escaped) {
                escaped = false;
                continue;
            }
            if (json[j] == '\\') {
                escaped = true;
                continue;
            }
            if (json[j] == '"') break;
        }
        if (j >= json.len) break;
        const key = json[start..j];
        var k = j + 1;
        while (k < json.len and (json[k] == ' ' or json[k] == '\t' or json[k] == '\n')) k += 1;
        if (k >= json.len or json[k] != ':') {
            i = j;
            continue;
        }
        k += 1;
        while (k < json.len and (json[k] == ' ' or json[k] == '\t' or json[k] == '\n')) k += 1;
        if (k >= json.len or json[k] != '{') {
            i = j;
            continue;
        }
        i = j;
        if (std.mem.eql(u8, key, "__metadata__")) {
            const meta_end = std.mem.indexOfPos(u8, json, k, "}") orelse json.len;
            if (metaString(json[k..meta_end], "format")) |v| format = try allocator.dupe(u8, v);
            if (metaString(json[k..meta_end], "pdd_partition")) |v| pdd_partition = try allocator.dupe(u8, v);
            if (metaString(json[k..meta_end], "lora_alpha")) |v| {
                lora_alpha = std.fmt.parseFloat(f32, v) catch lora_alpha;
            }
            continue;
        }
        try keys.append(allocator, try allocator.dupe(u8, key));
    }

    return .{
        .format = format,
        .pdd_partition = pdd_partition,
        .lora_alpha = lora_alpha,
        .keys = try keys.toOwnedSlice(allocator),
        .json = json,
    };
}

fn metaString(obj: []const u8, field: []const u8) ?[]const u8 {
    var pat_buf: [64]u8 = undefined;
    const pat = std.fmt.bufPrint(&pat_buf, "\"{s}\"", .{field}) catch return null;
    const ki = std.mem.indexOf(u8, obj, pat) orelse return null;
    var i = ki + pat.len;
    while (i < obj.len and (obj[i] == ' ' or obj[i] == ':' or obj[i] == '\t')) i += 1;
    if (i >= obj.len or obj[i] != '"') return null;
    i += 1;
    const start = i;
    while (i < obj.len) : (i += 1) {
        if (obj[i] == '\\') {
            i += 1;
            continue;
        }
        if (obj[i] == '"') return obj[start..i];
    }

    return null;
}

pub fn detectPddPath(allocator: std.mem.Allocator, path: []const u8) bool {
    const base = std.fs.path.basename(path);
    if (detectFromFilename(base)) return true;
    var peek = peekSafetensors(allocator, path) catch return false;
    defer peek.deinit(allocator);
    var buf: [64][]const u8 = undefined;
    const n = @min(peek.keys.len, buf.len);
    for (peek.keys[0..n], 0..) |k, i| buf[i] = k;

    return detectPdd(base, peek.format, buf[0..n]);
}

pub fn missingAccMessage(buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "Acc file missing — download MiniMax-H3-FL2VA-Acc-8Step.safetensors / MiniMax-H3-Ref2VA-Acc-8Step.safetensors from {s}", .{HF_REPO_URL}) catch buf[0..0];
}

// ════════════════════════════════════════════════════════════════════════
const testing = std.testing;

test "pdd acc: split_pdd_state_dict peels HEAD_KEYS and leaves lora" {
    const keys = [_][]const u8{
        "proj_out.weight",
        "proj_out.bias",
        "audio_proj_out.weight",
        "audio_proj_out.bias",
        "transformer_blocks.0.attn.to_q.lora_down",
        "transformer_blocks.0.attn.to_q.lora_up",
        "transformer_blocks.0.attn.to_k.lora_down",
        "transformer_blocks.0.attn.to_k.lora_up",
        "transformer_blocks.0.attn.to_v.lora_down",
        "transformer_blocks.0.attn.to_v.lora_up",
    };
    const s = splitKeySet(&keys);
    try testing.expectEqual(@as(u32, 4), s.n_heads);
    try testing.expectEqual(@as(u32, 6), s.n_lora);
    try testing.expectEqual(@as(u32, 0), s.n_leftover);
    try testing.expect(s.all_heads);
    try testing.expectEqual(SourceFormat.original, s.source.?);
}

test "pdd acc: convert_pdd_lora qkv concat + block-diag and fc1 half-swap" {
    const a = testing.allocator;
    // q/k/v: A [2, 4], B [3, 2]
    var aq = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var ak = [_]f32{ 9, 10, 11, 12, 13, 14, 15, 16 };
    var av = [_]f32{ 17, 18, 19, 20, 21, 22, 23, 24 };
    var bq = [_]f32{ 1, 0, 0, 1, 2, 0 };
    var bk = [_]f32{ 3, 0, 0, 3, 4, 0 };
    var bv = [_]f32{ 5, 0, 0, 5, 6, 0 };
    const qkv = try convertQkv(a, .{
        .{ .data = &aq, .rows = 2, .cols = 4 },
        .{ .data = &ak, .rows = 2, .cols = 4 },
        .{ .data = &av, .rows = 2, .cols = 4 },
    }, .{
        .{ .data = &bq, .rows = 3, .cols = 2 },
        .{ .data = &bk, .rows = 3, .cols = 2 },
        .{ .data = &bv, .rows = 3, .cols = 2 },
    }, 64.0);
    defer a.free(qkv.a);
    defer a.free(qkv.b);
    try testing.expectEqual(@as(usize, 6), qkv.a_rows); // 3*r
    try testing.expectEqual(@as(usize, 4), qkv.a_cols);
    try testing.expectEqual(@as(usize, 9), qkv.b_rows); // 3*o
    try testing.expectEqual(@as(usize, 6), qkv.b_cols); // 3*r
    try testing.expectEqual(@as(f32, 192), qkv.alpha); // 64*3
    // A is [q; k; v] row concat
    try testing.expectEqual(@as(f32, 1), qkv.a[0]);
    try testing.expectEqual(@as(f32, 9), qkv.a[8]);
    try testing.expectEqual(@as(f32, 17), qkv.a[16]);
    // B is block-diagonal: off-blocks zero
    try testing.expectEqual(@as(f32, 1), qkv.b[0]); // q block
    try testing.expectEqual(@as(f32, 0), qkv.b[2]); // q row, k cols
    try testing.expectEqual(@as(f32, 3), qkv.b[3 * 6 + 2]); // k block (1,0) of k → row 3, col 2

    var bd = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 }; // [4, 2] = [value; gate] with f=2
    const swapped = try convertFc1Up(a, .{ .data = &bd, .rows = 4, .cols = 2 });
    defer a.free(swapped);
    // [gate; value] = rows 2,3 then 0,1
    try testing.expectEqual(@as(f32, 5), swapped[0]);
    try testing.expectEqual(@as(f32, 6), swapped[1]);
    try testing.expectEqual(@as(f32, 1), swapped[4]);
}

test "pdd acc: fuse_heads ones-bank 8×size-4 equals the uniform mean" {
    const n: usize = 32;
    const out_d: usize = 2;
    const in_d: usize = 3;
    var w: [32 * 2 * 3]f32 = @splat(1);
    var b: [32 * 2]f32 = @splat(1);
    var sizes: [8]u32 = @splat(4);
    var fine: [33]f64 = undefined;
    fineSigmas(VIDEO_SHIFT, 32, &fine);
    var fw: [8 * 2 * 3]f32 = undefined;
    var fb: [8 * 2]f32 = undefined;
    fuseHeads(&w, &b, n, out_d, in_d, &fine, &sizes, &fw, &fb);
    for (fw) |v| try testing.expectApproxEqAbs(@as(f32, 1), v, 1e-6);
    for (fb) |v| try testing.expectApproxEqAbs(@as(f32, 1), v, 1e-6);
}

test "pdd acc: select_block exact knots and off-grid" {
    var sizes: [8]u32 = @splat(4);
    var bounds: [9]f64 = undefined;
    blockBoundaries(32, &sizes, &bounds);
    for (0..8) |i| {
        try testing.expectEqual(@as(u32, @intCast(i)), try selectBlock(bounds[i], &bounds, .err));
        try testing.expectEqual(@as(u32, @intCast(i)), try selectBlock(bounds[i] + 5e-7, &bounds, .err));
        try testing.expectEqual(@as(u32, @intCast(i)), try selectBlock(bounds[i] - 5e-7, &bounds, .err));
    }
    try testing.expectEqual(@as(u32, 7), try selectBlock(0.0, &bounds, .err));
    const mid = (bounds[2] + bounds[3]) / 2.0;
    try testing.expectError(error.AccOffGrid, selectBlock(mid, &bounds, .err));
    try testing.expectEqual(@as(u32, 2), try selectBlock(mid, &bounds, .clamp));
    try testing.expectEqual(@as(u32, 0), try selectBlock(1.5, &bounds, .clamp));
    try testing.expectEqual(@as(u32, 7), try selectBlock(1e-4, &bounds, .clamp));
}

test "pdd acc: nfe=8 sigma knots match 12t/(1+11t) linspace(1,0,9)" {
    var got: [9]f64 = undefined;
    fineSigmas(VIDEO_SHIFT, 8, &got);
    var sizes: [8]u32 = @splat(4);
    var bounds: [9]f64 = undefined;
    blockBoundaries(32, &sizes, &bounds);
    for (0..9) |i| {
        const t = @as(f64, @floatFromInt(8 - i)) / 8.0;
        const want = 12.0 * t / (1.0 + 11.0 * t);
        try testing.expect(@abs(got[i] - want) < 1e-6);
        try testing.expect(@abs(bounds[i] - want) < 1e-6);
    }
    try testing.expectEqual(@as(f64, 1.0), got[0]);
    try testing.expectEqual(@as(f64, 0.0), got[8]);
}

test "pdd acc: Acc+Turbo together is refused" {
    try refuseTurbo(false, true);
    try refuseTurbo(true, false);
    try testing.expectError(error.AccTurboExclusive, refuseTurbo(true, true));
}

test "pdd acc: wrong partition name is refused" {
    try checkPartitionPairing(.fl2va, .fl2va);
    try checkPartitionPairing(null, .fl2va); // unknown → do not guess
    try testing.expectError(error.AccPartitionMismatch, checkPartitionPairing(.fl2va, .ref2va));
    try testing.expectError(error.AccPartitionMismatch, checkPartitionPairing(.ref2va, .fl2va));
    try testing.expectEqual(Partition.fl2va, partitionFromName(null, FL2VA_FILENAME).?);
    try testing.expectEqual(Partition.ref2va, partitionFromName(null, REF2VA_FILENAME).?);
    try testing.expect(partitionFromName(null, "fl2va_vs_ref2va_merge.safetensors") == null);
    try testing.expectEqual(Partition.ref2va, partitionFromName("ref2va", "ignored_fl2va.safetensors").?);
}

test "pdd acc: packed bank onto native head is the Comfy crash class" {
    try testing.expect(bankWouldOverwriteNative(&.{ 3072, 5376 }, 96));
    try testing.expect(bankWouldOverwriteNative(&.{ 1024, 5376 }, 32));
    try testing.expect(!bankWouldOverwriteNative(&.{ 96, 5376 }, 96)); // native itself
    try testing.expect(!bankWouldOverwriteNative(&.{ 32, 96, 5376 }, 96));
    const packed_w = try parseHeadWeightShape(&.{ 3072, 5376 }, 96);
    try testing.expectEqual(@as(usize, 32), packed_w.n);
    try testing.expectEqual(@as(usize, 96), packed_w.out);
    try testing.expectError(error.AccNativeHeadOverwrite, parseHeadWeightShape(&.{ 96, 5376 }, 96));
    const ok3 = try parseHeadWeightShape(&.{ 32, 96, 5376 }, 96);
    try testing.expectEqual(@as(usize, 32), ok3.n);
}

test "pdd acc: Acc convert hits 258 trunk targets, not Turbo's 259" {
    // Recorded reason: convert_pdd_lora walks 50 blocks × (qkv, out, fc1, fc2,
    // adaln) + 2 refiner × (qkv, out, fc1, fc2). Turbo's 259th is
    // final_layer.adaln_proj.linear, which Acc files do not ship.
    try testing.expectEqual(@as(u32, 258), ACC_TRUNK_TARGETS);
    try testing.expectEqual(@as(u32, 259), ACC_TRUNK_TARGETS + 1);
}

test "pdd acc: resolve_partition 4/8 only, explicit text overrides nfe" {
    var buf: [16]u32 = undefined;
    const eight = try resolvePartition(32, 8, null, &buf);
    try testing.expectEqual(@as(usize, 8), eight.len);
    for (eight) |s| try testing.expectEqual(@as(u32, 4), s);
    const four = try resolvePartition(32, 4, null, &buf);
    try testing.expectEqual(@as(usize, 4), four.len);
    for (four) |s| try testing.expectEqual(@as(u32, 8), s);
    try testing.expectError(error.AccBadPartition, resolvePartition(32, 6, null, &buf));
    try testing.expectError(error.AccBadPartition, resolvePartition(32, 7, null, &buf));
    try testing.expectError(error.AccBadPartition, resolvePartition(32, 16, null, &buf));
    const custom = try resolvePartition(32, 8, "8,8,4,4,4,4", &buf);
    try testing.expectEqual(@as(usize, 6), custom.len);
    try testing.expectEqual(@as(u32, 8), custom[0]);
    try testing.expectError(error.AccBadPartition, resolvePartition(32, 8, "16,16", &buf));
    try testing.expectError(error.AccBadPartition, resolvePartition(32, 8, "2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2", &buf));
}

test "pdd acc: detect filename and converted format" {
    try testing.expect(detectFromFilename(FL2VA_FILENAME));
    try testing.expect(detectFromFilename("minimax_h3_fl2va_pdd_acc_8step_comfyui.safetensors"));
    try testing.expect(!detectFromFilename("some_style_lora.safetensors"));
    try testing.expect(detectFromMetadataFormat(CONVERTED_FORMAT));
    try testing.expect(detectPdd("x.safetensors", null, &.{"proj_out.weight"}));
}

test "pdd acc: audio identity s·Δσv/(ci·cj) = Δσa on the 8-step grid" {
    var sizes: [8]u32 = @splat(4);
    var bounds: [9]f64 = undefined;
    blockBoundaries(32, &sizes, &bounds);
    var fine_a: [33]f64 = undefined;
    fineSigmas(AUDIO_SHIFT, 32, &fine_a);
    const s = VIDEO_SHIFT / AUDIO_SHIFT; // 4
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        const sv_i = bounds[i];
        const sv_j = bounds[i + 1];
        const sa_i = fine_a[i * 4];
        const sa_j = fine_a[(i + 1) * 4];
        const c_i = s - (s - 1.0) * sv_i;
        const c_j = s - (s - 1.0) * sv_j;
        const mapped = s * (sv_j - sv_i) / (c_i * c_j);
        try testing.expect(@abs(mapped - (sa_j - sa_i)) < 1e-12);
    }
}
