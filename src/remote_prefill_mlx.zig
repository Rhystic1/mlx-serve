//! MLX-side import of a remote-prefill blob (v2 of the NVIDIA prefix cache).
//!
//! The worker prefills a GGUF on llama.cpp and ships `llama_state_seq_get_data`.
//! v1 restores that into a llama.cpp session. This file restores it into OUR
//! MLX `KVCache` instead, so the Mac decodes on the MLX pack while the GPU box
//! did the prompt. Nothing GGUF-specific survives into K/V except numerics:
//! both engines store K after k_norm + RoPE and V after the parameter-free RMS
//! norm, in the same head-major row layout. The drift is that the worker's
//! weights are a different quantization of the same base model. Accepted for
//! the PoC, measured at the quality gate, never hidden.
//!
//! Blob format (llama.cpp b10472, `llama_kv_cache::state_write`), read
//! straight from the source: `[base cache][swa cache]`, each
//!
//!   u32 n_stream; per stream: u32 cell_count; if 0 -> next stream
//!     meta: cell_count x { i32 pos; u32 n_seq_id; i32 seq_id[n_seq_id] }
//!     data: u32 v_trans; u32 n_layer;
//!           n_layer x { i32 ggml_type; u64 row_size; cell_count rows }   (K)
//!           n_layer x { i32 ggml_type; u64 row_size; cell_count rows }   (V, v_trans == 0)
//!
//! The base cache holds the GLOBAL layers in ascending order, the swa cache the
//! SLIDING ones; the swa export carries only window-reachable cells. Layer
//! indices are NOT in the blob — the order is the contract. Transposed V
//! (`v_trans == 1`, flash attention off) is refused.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const transformer = @import("transformer.zig");

pub const GGML_F16: i32 = 1;
pub const GGML_Q8_0: i32 = 8;
pub const GGML_BF16: i32 = 30;

pub const Q8_BLOCK: usize = 32;
pub const Q8_BLOCK_BYTES: usize = 34;

pub const Error = error{
    Truncated,
    MultiStream,
    TransposedV,
    UnsupportedType,
    LayerCountMismatch,
    RowSizeMismatch,
    NoCells,
};

/// What the consumer model expects the blob to be shaped like.
pub const Layout = struct {
    n_global: u32,
    n_sliding: u32,
    n_kv_heads: u32,
    global_hd: u32,
    sliding_hd: u32,

    pub fn fromConfig(cfg: *const model_mod.ModelConfig) Layout {
        var g: u32 = 0;
        var s: u32 = 0;
        for (0..cfg.num_hidden_layers) |i| {
            if (cfg.isGlobalLayer(@intCast(i))) g += 1 else s += 1;
        }
        return .{
            .n_global = g,
            .n_sliding = s,
            .n_kv_heads = cfg.num_key_value_heads,
            .global_hd = if (cfg.global_head_dim > 0) cfg.global_head_dim else cfg.head_dim,
            .sliding_hd = cfg.head_dim,
        };
    }
};

pub const Rows = struct {
    ggml_type: i32,
    row_size: usize,
    data: []const u8,
};

pub const Block = struct {
    positions: []i32,
    keys: []Rows,
    values: []Rows,

    pub fn cellCount(self: Block) usize {
        return self.positions.len;
    }
};

pub const Parsed = struct {
    global: Block,
    sliding: Block,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Parsed) void {
        for ([_]*Block{ &self.global, &self.sliding }) |b| {
            self.allocator.free(b.positions);
            self.allocator.free(b.keys);
            self.allocator.free(b.values);
        }
    }
};

const Reader = struct {
    buf: []const u8,
    pos: usize = 0,

    fn take(self: *Reader, n: usize) Error![]const u8 {
        if (self.pos + n > self.buf.len) return error.Truncated;
        const s = self.buf[self.pos .. self.pos + n];
        self.pos += n;
        return s;
    }
    fn u32le(self: *Reader) Error!u32 {
        return std.mem.readInt(u32, (try self.take(4))[0..4], .little);
    }
    fn i32le(self: *Reader) Error!i32 {
        return std.mem.readInt(i32, (try self.take(4))[0..4], .little);
    }
    fn u64le(self: *Reader) Error!u64 {
        return std.mem.readInt(u64, (try self.take(8))[0..8], .little);
    }
};

pub fn rowBytes(ggml_type: i32, n_elems: usize) Error!usize {
    return switch (ggml_type) {
        GGML_F16, GGML_BF16 => n_elems * 2,
        GGML_Q8_0 => blk: {
            if (n_elems % Q8_BLOCK != 0) return error.RowSizeMismatch;
            break :blk (n_elems / Q8_BLOCK) * Q8_BLOCK_BYTES;
        },
        else => error.UnsupportedType,
    };
}

fn parseBlock(allocator: std.mem.Allocator, r: *Reader, n_layer_expect: u32, row_elems: usize) (Error || std.mem.Allocator.Error)!Block {
    const n_stream = try r.u32le();
    if (n_stream != 1) return error.MultiStream;
    const cell_count = try r.u32le();
    if (cell_count == 0) return error.NoCells;

    const positions = try allocator.alloc(i32, cell_count);
    errdefer allocator.free(positions);
    for (positions) |*p| {
        p.* = try r.i32le();
        const n_seq = try r.u32le();
        _ = try r.take(4 * n_seq);
    }

    const v_trans = try r.u32le();
    if (v_trans != 0) return error.TransposedV;
    const n_layer = try r.u32le();
    if (n_layer != n_layer_expect) return error.LayerCountMismatch;

    const keys = try allocator.alloc(Rows, n_layer);
    errdefer allocator.free(keys);
    const values = try allocator.alloc(Rows, n_layer);
    errdefer allocator.free(values);
    for ([_][]Rows{ keys, values }) |rows| {
        for (rows) |*rw| {
            const t = try r.i32le();
            const row_size: usize = @intCast(try r.u64le());
            if (row_size != try rowBytes(t, row_elems)) return error.RowSizeMismatch;
            rw.* = .{ .ggml_type = t, .row_size = row_size, .data = try r.take(row_size * cell_count) };
        }
    }
    return .{ .positions = positions, .keys = keys, .values = values };
}

pub fn parseBlob(allocator: std.mem.Allocator, blob: []const u8, layout: Layout) (Error || std.mem.Allocator.Error)!Parsed {
    var r = Reader{ .buf = blob };
    var global = try parseBlock(allocator, &r, layout.n_global, @as(usize, layout.n_kv_heads) * layout.global_hd);
    errdefer {
        allocator.free(global.positions);
        allocator.free(global.keys);
        allocator.free(global.values);
    }
    const sliding = try parseBlock(allocator, &r, layout.n_sliding, @as(usize, layout.n_kv_heads) * layout.sliding_hd);
    _ = &global;
    return .{ .global = global, .sliding = sliding, .allocator = allocator };
}

fn f16ToF32(bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(bits)));
}

/// Decode one stored row into `out` (length = elements per row).
pub fn decodeRow(ggml_type: i32, row: []const u8, out: []f32) Error!void {
    switch (ggml_type) {
        GGML_F16 => {
            for (out, 0..) |*o, i| o.* = f16ToF32(std.mem.readInt(u16, row[i * 2 ..][0..2], .little));
        },
        GGML_BF16 => {
            for (out, 0..) |*o, i| o.* = @bitCast(@as(u32, std.mem.readInt(u16, row[i * 2 ..][0..2], .little)) << 16);
        },
        GGML_Q8_0 => {
            var b: usize = 0;
            while (b * Q8_BLOCK < out.len) : (b += 1) {
                const blk = row[b * Q8_BLOCK_BYTES ..][0..Q8_BLOCK_BYTES];
                const d = f16ToF32(std.mem.readInt(u16, blk[0..2], .little));
                for (0..Q8_BLOCK) |j| {
                    out[b * Q8_BLOCK + j] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(blk[2 + j]))));
                }
            }
        },
        else => return error.UnsupportedType,
    }
}

/// Fill a `[n_tokens, row_elems]` f32 host buffer: every cell lands at its own
/// position; positions the blob does not carry (outside the sliding window)
/// stay ZERO. They are never read — the sliding view trims to the window tail
/// — but they must exist so every layer's offset equals the prompt length.
pub fn materialize(block: Block, layer: usize, which: enum { k, v }, n_tokens: usize, row_elems: usize, out: []f32) Error!void {
    @memset(out, 0);
    const rows = if (which == .k) block.keys[layer] else block.values[layer];
    for (block.positions, 0..) |pos, cell| {
        if (pos < 0 or @as(usize, @intCast(pos)) >= n_tokens) continue;
        const p: usize = @intCast(pos);
        try decodeRow(rows.ggml_type, rows.data[cell * rows.row_size ..][0..rows.row_size], out[p * row_elems ..][0..row_elems]);
    }
}

/// Push every layer of the parsed blob into `cache` through the cache's own
/// `update`, so dense and quantized schemes both work and `step` advances the
/// way a local prefill would. On return `cache.step == n_tokens`.
pub fn importIntoCache(
    allocator: std.mem.Allocator,
    cache: *transformer.KVCache,
    cfg: *const model_mod.ModelConfig,
    parsed: *const Parsed,
    n_tokens: usize,
    s: mlx.mlx_stream,
) !void {
    const H: usize = cfg.num_key_value_heads;
    var gi: usize = 0;
    var si: usize = 0;
    var buf: []f32 = &.{};
    defer allocator.free(buf);
    for (0..cfg.num_hidden_layers) |li| {
        const is_global = cfg.isGlobalLayer(@intCast(li));
        const hd: usize = cfg.layerHeadDim(@intCast(li));
        const row_elems = H * hd;
        const need = n_tokens * row_elems;
        if (buf.len < need) {
            allocator.free(buf);
            buf = try allocator.alloc(f32, need);
        }
        const block = if (is_global) &parsed.global else &parsed.sliding;
        const idx = if (is_global) gi else si;
        if (is_global) gi += 1 else si += 1;

        const k = try hostToKv(buf[0..need], block.*, idx, .k, n_tokens, H, hd, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try hostToKv(buf[0..need], block.*, idx, .v, n_tokens, H, hd, s);
        defer _ = mlx.mlx_array_free(v);
        var view = try cache.update(@intCast(li), k, v, s, 0);
        view.deinit();
    }
    try mlx.check(mlx.mlx_synchronize(s));
}

fn hostToKv(buf: []f32, block: Block, layer: usize, which: enum { k, v }, n_tokens: usize, H: usize, hd: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    try materialize(block, layer, if (which == .k) .k else .v, n_tokens, H * hd, buf);
    const shape = [_]c_int{ 1, @intCast(n_tokens), @intCast(H), @intCast(hd) };
    const host = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(host);
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    const perm = [_]c_int{ 0, 2, 1, 3 };
    try mlx.check(mlx.mlx_transpose_axes(&t, host, &perm, 4, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, t, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

// ── tests ──

const testing = std.testing;

const TestWriter = struct {
    list: std.ArrayList(u8),
    fn u32le(self: *TestWriter, v: u32) !void {
        try self.list.appendSlice(testing.allocator, std.mem.asBytes(&std.mem.nativeToLittle(u32, v)));
    }
    fn i32le(self: *TestWriter, v: i32) !void {
        try self.list.appendSlice(testing.allocator, std.mem.asBytes(&std.mem.nativeToLittle(i32, v)));
    }
    fn u64le(self: *TestWriter, v: u64) !void {
        try self.list.appendSlice(testing.allocator, std.mem.asBytes(&std.mem.nativeToLittle(u64, v)));
    }
    /// One cache block in llama.cpp's own order: meta, then all K, then all V.
    fn block(self: *TestWriter, positions: []const i32, n_layer: u32, ggml_type: i32, row_size: usize, fill: u8) !void {
        try self.u32le(1);
        try self.u32le(@intCast(positions.len));
        for (positions) |p| {
            try self.i32le(p);
            try self.u32le(1);
            try self.i32le(0);
        }
        try self.u32le(0);
        try self.u32le(n_layer);
        for (0..2) |kv| {
            for (0..n_layer) |l| {
                try self.i32le(ggml_type);
                try self.u64le(row_size);
                for (0..positions.len) |c| {
                    for (0..row_size) |b| try self.list.append(testing.allocator, fill +% @as(u8, @intCast((kv * 7 + l * 3 + c) & 0x7f)) +% @as(u8, @intCast(b & 1)));
                }
            }
        }
    }
};

test "remote-prefill mlx: parses a two-block blob and places cells by position" {
    const layout = Layout{ .n_global = 1, .n_sliding = 2, .n_kv_heads = 2, .global_hd = 4, .sliding_hd = 2 };
    var w = TestWriter{ .list = .empty };
    defer w.list.deinit(testing.allocator);
    // Global block: all 5 positions, f16 rows of 2*4 = 8 elems = 16 bytes.
    try w.block(&[_]i32{ 0, 1, 2, 3, 4 }, 1, GGML_F16, 16, 0);
    // Sliding block: only the window tail, in a scrambled cell order.
    try w.block(&[_]i32{ 4, 2, 3 }, 2, GGML_F16, 8, 0);

    var parsed = try parseBlob(testing.allocator, w.list.items, layout);
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 5), parsed.global.cellCount());
    try testing.expectEqual(@as(usize, 3), parsed.sliding.cellCount());
    try testing.expectEqual(@as(usize, 16), parsed.global.keys[0].row_size);
    try testing.expectEqual(@as(usize, 8), parsed.sliding.values[1].row_size);

    // Materialize sliding layer 1 K: rows for positions 2..4 present, 0..1 zero,
    // and cell 0 (pos 4) lands at row 4, not row 0.
    var out: [5 * 4]f32 = undefined;
    try materialize(parsed.sliding, 1, .k, 5, 4, &out);
    try testing.expectEqual(@as(f32, 0), out[0]);
    try testing.expectEqual(@as(f32, 0), out[1 * 4]);
    const cell0_row = parsed.sliding.keys[1].data[0..8];
    var want: [4]f32 = undefined;
    try decodeRow(GGML_F16, cell0_row, &want);
    try testing.expectEqualSlices(f32, &want, out[4 * 4 ..][0..4]);

    // A blob for a different layer count is refused, never misread.
    const wrong = Layout{ .n_global = 2, .n_sliding = 1, .n_kv_heads = 2, .global_hd = 4, .sliding_hd = 2 };
    try testing.expectError(error.LayerCountMismatch, parseBlob(testing.allocator, w.list.items, wrong));
    // A truncated body is refused.
    try testing.expectError(error.Truncated, parseBlob(testing.allocator, w.list.items[0 .. w.list.items.len - 3], layout));
}

test "remote-prefill mlx: q8_0 rows dequantize as d * q" {
    // One block: d = 0.5 (f16 0x3800), qs = -128..-97
    var row: [Q8_BLOCK_BYTES]u8 = undefined;
    std.mem.writeInt(u16, row[0..2], 0x3800, .little);
    for (0..Q8_BLOCK) |j| row[2 + j] = @bitCast(@as(i8, @intCast(-128 + @as(i32, @intCast(j)))));
    var out: [Q8_BLOCK]f32 = undefined;
    try decodeRow(GGML_Q8_0, &row, &out);
    try testing.expectEqual(@as(f32, -64), out[0]);
    try testing.expectEqual(@as(f32, -48.5), out[31]);
    try testing.expectEqual(@as(usize, 34), try rowBytes(GGML_Q8_0, 32));
    try testing.expectError(error.RowSizeMismatch, rowBytes(GGML_Q8_0, 33));
    try testing.expectError(error.UnsupportedType, rowBytes(2, 32));
}
