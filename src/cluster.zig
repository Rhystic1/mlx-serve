//! `GET /v1/cluster` — the mesh in one JSON object (rpc-offload-plan.md Part 3,
//! data half; m4max owns the console tab that renders it).
//!
//! PURE: `render` turns an `Input` into JSON and knows nothing about the
//! server, so it tests hermetically. `server.zig` collects the `Input` from
//! the live globals (LAN table, RPC snapshot, prefill calibration). The
//! process-wide snapshots below are written ONCE per load (`g_rpc`) or per
//! engaged/failed remote prefill (`g_prefill_last`) and only READ per
//! request — the handler never calls into llama.cpp.
//!
//! Every field is present even when its subsystem is off (null / [] / "none"),
//! so the tab renders on an empty server, like `/props` and `GET /`.

const std = @import("std");
const chat = @import("chat.zig");
const rpc_offload = @import("rpc_offload.zig");

pub const MAX_DEVICES = rpc_offload.MAX_ENDPOINTS + 16;

pub const Device = struct {
    name: []const u8, // "CUDA0" / "Metal" / "RPC0"
    endpoint: ?[]const u8 = null, // workers only
    first_layer: ?u32 = null,
    last_layer: ?u32 = null,
    layer_count: u32 = 0,

    pub fn isRemote(d: Device) bool {
        return std.mem.startsWith(u8, d.name, "RPC");
    }
};

/// Filled by the scheduler right after a llama.cpp open (from llama.cpp's own
/// `assigned to device` lines) — process-wide, one llama model at a time.
pub const RpcSnapshot = struct {
    model: ?[]const u8 = null,
    devices: [MAX_DEVICES]Device = undefined,
    n_devices: usize = 0,
    // Name storage for `Device.name` slices (llama's lines die with the engine).
    names: [MAX_DEVICES][32]u8 = undefined,
    workers: [rpc_offload.MAX_ENDPOINTS]Worker = undefined,
    n_workers: usize = 0,

    pub fn reset(s: *RpcSnapshot) void {
        s.* = .{};
    }

    /// Fold one llama.cpp assignment line in. Devices keep first-seen order,
    /// which is llama's own order; ranges are min/max of the layers seen.
    pub fn observe(s: *RpcSnapshot, line: []const u8) void {
        const a = rpc_offload.parseLayerAssign(line) orelse return;
        var idx: ?usize = null;
        for (s.devices[0..s.n_devices], 0..) |d, i| if (std.mem.eql(u8, d.name, a.device)) {
            idx = i;
        };
        if (idx == null) {
            if (s.n_devices == MAX_DEVICES) return;
            const n = @min(a.device.len, s.names[s.n_devices].len);
            @memcpy(s.names[s.n_devices][0..n], a.device[0..n]);
            s.devices[s.n_devices] = .{ .name = s.names[s.n_devices][0..n] };
            idx = s.n_devices;
            s.n_devices += 1;
        }
        var d = &s.devices[idx.?];
        d.layer_count += 1;
        d.first_layer = if (d.first_layer) |f| @min(f, a.layer) else a.layer;
        d.last_layer = if (d.last_layer) |l| @max(l, a.layer) else a.layer;
    }
};

pub const Worker = struct { endpoint: []const u8, free_bytes: u64, total_bytes: u64, reachable: bool };

pub const Serve = struct { endpoint: []const u8, device: []const u8, is_gpu: bool, free_bytes: u64, total_bytes: u64 };

pub const PrefillLast = struct { engaged: bool, tokens: u32, ms: f64, reason: ?[]const u8 };

pub const Input = struct {
    name: []const u8,
    host: []const u8,
    port: u16,
    platform: []const u8,
    engine: []const u8,
    version: []const u8,
    /// Pre-rendered `["id",...]` of what THIS node serves (self drawn like a peer).
    models_json: []const u8,
    lan_enabled: bool,
    /// Pre-rendered `[{...},...]` peer objects (rendered under the table lock
    /// by `lan_peers.Table.appendPeersJson`), or "[]".
    lan_peers_json: []const u8,
    rpc_serve: ?Serve,
    rpc: *const RpcSnapshot,
    tensor_split: []const u8,
    prefill_mode: []const u8, // none|worker|consumer
    /// "mlx" = v2 KV import into our MLX cache, "llama" = v1 llama.cpp state restore.
    prefill_consumer_engine: ?[]const u8,
    prefill_url: ?[]const u8,
    prefill_model: ?[]const u8,
    prefill_kv_type: []const u8,
    prefill_last: ?PrefillLast,
    prefill_local_tok_s: f64,
    prefill_remote_tok_s: f64,
    decode_tok_s: f64,
    requests_inflight: u32,
};

pub var g_rpc: RpcSnapshot = .{};
pub var g_prefill_last: ?PrefillLast = null;
/// Decode rate of the most recently FINISHED request (scheduler hook).
pub var g_last_decode_tok_s: f64 = 0;
var g_prefill_reason_buf: [128]u8 = undefined;

/// Record the outcome of one remote-prefill attempt (scheduler hook).
pub fn recordPrefill(engaged: bool, tokens: u32, ms: f64, reason: ?[]const u8) void {
    var r: ?[]const u8 = null;
    if (reason) |why| {
        const n = @min(why.len, g_prefill_reason_buf.len);
        @memcpy(g_prefill_reason_buf[0..n], why[0..n]);
        r = g_prefill_reason_buf[0..n];
    }
    g_prefill_last = .{ .engaged = engaged, .tokens = tokens, .ms = ms, .reason = r };
}

pub fn rpcRole(has_serve: bool, has_workers: bool) []const u8 {
    if (has_serve and has_workers) return "both";
    if (has_serve) return "worker";
    if (has_workers) return "consumer";
    return "none";
}

fn str(a: std.mem.Allocator, out: *std.ArrayList(u8), s: ?[]const u8) !void {
    if (s) |v| try chat.appendJsonString(a, out, v) else try out.appendSlice(a, "null");
}

pub fn render(a: std.mem.Allocator, in: Input) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    const w = &out;
    try w.appendSlice(a, "{\"self\":{\"name\":");
    try str(a, w, in.name);
    try w.appendSlice(a, ",\"host\":");
    try str(a, w, in.host);
    try w.print(a, ",\"port\":{d},\"platform\":", .{in.port});
    try str(a, w, in.platform);
    try w.appendSlice(a, ",\"engine\":");
    try str(a, w, in.engine);
    try w.appendSlice(a, ",\"version\":");
    try str(a, w, in.version);
    try w.print(a, ",\"models\":{s}", .{if (in.models_json.len > 0) in.models_json else "[]"});
    try w.print(a, "}},\"lan\":{{\"enabled\":{},\"peers\":{s}}}", .{ in.lan_enabled, if (in.lan_peers_json.len > 0) in.lan_peers_json else "[]" });

    // rpc
    try w.print(a, ",\"rpc\":{{\"role\":\"{s}\",\"serve\":", .{rpcRole(in.rpc_serve != null, in.rpc.n_workers > 0)});
    if (in.rpc_serve) |s| {
        try w.appendSlice(a, "{\"endpoint\":");
        try str(a, w, s.endpoint);
        try w.appendSlice(a, ",\"device\":");
        try str(a, w, s.device);
        try w.print(a, ",\"is_gpu\":{},\"free_bytes\":{d},\"total_bytes\":{d}}}", .{ s.is_gpu, s.free_bytes, s.total_bytes });
    } else try w.appendSlice(a, "null");
    try w.appendSlice(a, ",\"workers\":[");
    for (in.rpc.workers[0..in.rpc.n_workers], 0..) |wk, i| {
        if (i > 0) try w.append(a, ',');
        try w.appendSlice(a, "{\"endpoint\":");
        try str(a, w, wk.endpoint);
        try w.print(a, ",\"free_bytes\":{d},\"total_bytes\":{d},\"reachable\":{}}}", .{ wk.free_bytes, wk.total_bytes, wk.reachable });
    }
    try w.appendSlice(a, "],\"devices\":[");
    for (in.rpc.devices[0..in.rpc.n_devices], 0..) |d, i| {
        if (i > 0) try w.append(a, ',');
        try w.appendSlice(a, "{\"name\":");
        try str(a, w, d.name);
        try w.print(a, ",\"kind\":\"{s}\",\"endpoint\":", .{if (d.isRemote()) "remote" else "local"});
        // RPC<i> ↔ the i-th worker, llama's own numbering.
        var ep: ?[]const u8 = d.endpoint;
        if (ep == null and d.isRemote()) {
            const k = std.fmt.parseInt(usize, d.name[3..], 10) catch null;
            if (k) |kk| if (kk < in.rpc.n_workers) {
                ep = in.rpc.workers[kk].endpoint;
            };
        }
        try str(a, w, ep);
        if (d.first_layer) |f| {
            try w.print(a, ",\"layers\":[{d},{d}]", .{ f, d.last_layer.? });
        } else try w.appendSlice(a, ",\"layers\":[]");
        try w.print(a, ",\"layer_count\":{d}}}", .{d.layer_count});
    }
    try w.appendSlice(a, "],\"tensor_split\":");
    if (in.tensor_split.len > 0) {
        try w.append(a, '[');
        var it = std.mem.splitScalar(u8, in.tensor_split, ',');
        var first = true;
        while (it.next()) |raw| {
            const t = std.mem.trim(u8, raw, " ");
            if (t.len == 0) continue;
            if (!first) try w.append(a, ',');
            first = false;
            try w.appendSlice(a, t);
        }
        try w.append(a, ']');
    } else try w.appendSlice(a, "null");
    try w.appendSlice(a, ",\"model\":");
    try str(a, w, in.rpc.model);
    try w.append(a, '}');

    // prefill
    try w.print(a, ",\"prefill\":{{\"mode\":\"{s}\",\"consumer_engine\":", .{in.prefill_mode});
    try str(a, w, in.prefill_consumer_engine);
    try w.appendSlice(a, ",\"url\":");
    try str(a, w, in.prefill_url);
    try w.appendSlice(a, ",\"model\":");
    try str(a, w, in.prefill_model);
    try w.appendSlice(a, ",\"kv_type\":");
    try str(a, w, in.prefill_kv_type);
    try w.appendSlice(a, ",\"last\":");
    if (in.prefill_last) |l| {
        try w.print(a, "{{\"engaged\":{},\"tokens\":{d},\"ms\":{d:.1},\"reason\":", .{ l.engaged, l.tokens, l.ms });
        try str(a, w, l.reason);
        try w.append(a, '}');
    } else try w.appendSlice(a, "null");
    try w.print(a, ",\"rates\":{{\"local_tok_s\":{d:.1},\"remote_tok_s\":{d:.1}}}}}", .{ in.prefill_local_tok_s, in.prefill_remote_tok_s });

    // live + hops (hops: null until ggml gives per-boundary timings — never faked)
    try w.print(a, ",\"live\":{{\"decode_tok_s\":{d:.1},\"requests_inflight\":{d}}},\"hops\":null}}", .{ in.decode_tok_s, in.requests_inflight });
    return out.toOwnedSlice(a);
}

// ── tests ──

const testing = std.testing;

fn emptyInput(snap: *const RpcSnapshot) Input {
    return .{
        .name = "box",
        .host = "0.0.0.0",
        .port = 8080,
        .platform = "windows",
        .engine = "llama.cpp",
        .version = "test",
        .models_json = "",
        .lan_enabled = false,
        .lan_peers_json = "",
        .rpc_serve = null,
        .rpc = snap,
        .tensor_split = "",
        .prefill_mode = "none",
        .prefill_consumer_engine = null,
        .prefill_url = null,
        .prefill_model = null,
        .prefill_kv_type = "f16",
        .prefill_last = null,
        .prefill_local_tok_s = 0,
        .prefill_remote_tok_s = 0,
        .decode_tok_s = 0,
        .requests_inflight = 0,
    };
}

test "cluster: an EMPTY server renders every section, as valid JSON, with nothing invented" {
    const snap = RpcSnapshot{};
    const body = try render(testing.allocator, emptyInput(&snap));
    defer testing.allocator.free(body);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try testing.expectEqualStrings("none", o.get("rpc").?.object.get("role").?.string);
    try testing.expectEqual(@as(usize, 0), o.get("rpc").?.object.get("devices").?.array.items.len);
    try testing.expect(o.get("rpc").?.object.get("serve").? == .null);
    try testing.expect(o.get("lan").?.object.get("enabled").?.bool == false);
    try testing.expectEqual(@as(usize, 0), o.get("lan").?.object.get("peers").?.array.items.len);
    try testing.expect(o.get("prefill").?.object.get("last").? == .null);
    try testing.expect(o.get("hops").? == .null);
}

test "cluster: the split snapshot folds llama.cpp's lines into per-device layer ranges, RPC<i> mapped to worker i" {
    var snap = RpcSnapshot{};
    snap.model = "qwen";
    snap.workers[0] = .{ .endpoint = "192.168.0.150:50052", .free_bytes = 10, .total_bytes = 16, .reachable = true };
    snap.n_workers = 1;
    snap.observe("load_tensors: layer   0 assigned to device CUDA0, is_swa = 0\n");
    snap.observe("load_tensors: layer   1 assigned to device CUDA0, is_swa = 0\n");
    snap.observe("load_tensors: loading model tensors, this can take a while... (load_mode = mmap)\n");
    snap.observe("load_tensors: layer   2 assigned to device RPC0, is_swa = 1\n");
    snap.observe("load_tensors: layer   3 assigned to device RPC0, is_swa = 0\n");
    snap.observe("load_tensors: layer   4 assigned to device RPC0, is_swa = 0\n");
    try testing.expectEqual(@as(usize, 2), snap.n_devices);
    try testing.expectEqual(@as(u32, 3), snap.devices[1].layer_count);

    var in = emptyInput(&snap);
    in.tensor_split = "0.4, 0.6";
    in.rpc_serve = .{ .endpoint = "0.0.0.0:50052", .device = "CUDA0", .is_gpu = true, .free_bytes = 1, .total_bytes = 2 };
    in.lan_enabled = true;
    in.lan_peers_json = "[{\"name\":\"mini\"}]";
    in.prefill_mode = "consumer";
    in.prefill_consumer_engine = "llama";
    in.models_json = "[\"qwen\"]";
    in.prefill_url = "http://192.168.0.150:8080";
    in.prefill_last = .{ .engaged = true, .tokens = 4096, .ms = 812.5, .reason = null };
    const body = try render(testing.allocator, in);
    defer testing.allocator.free(body);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const rpc = parsed.value.object.get("rpc").?.object;
    try testing.expectEqualStrings("both", rpc.get("role").?.string);
    const devs = rpc.get("devices").?.array.items;
    try testing.expectEqual(@as(usize, 2), devs.len);
    try testing.expectEqualStrings("local", devs[0].object.get("kind").?.string);
    try testing.expectEqual(@as(i64, 1), devs[0].object.get("layers").?.array.items[1].integer);
    try testing.expectEqualStrings("remote", devs[1].object.get("kind").?.string);
    try testing.expectEqualStrings("192.168.0.150:50052", devs[1].object.get("endpoint").?.string);
    try testing.expectEqual(@as(i64, 2), devs[1].object.get("layers").?.array.items[0].integer);
    try testing.expectEqual(@as(i64, 4), devs[1].object.get("layers").?.array.items[1].integer);
    try testing.expectEqual(@as(usize, 2), rpc.get("tensor_split").?.array.items.len);
    try testing.expectEqualStrings("mini", parsed.value.object.get("lan").?.object.get("peers").?.array.items[0].object.get("name").?.string);
    try testing.expectEqualStrings("llama", parsed.value.object.get("prefill").?.object.get("consumer_engine").?.string);
    try testing.expectEqualStrings("qwen", parsed.value.object.get("self").?.object.get("models").?.array.items[0].string);
    const last = parsed.value.object.get("prefill").?.object.get("last").?.object;
    try testing.expect(last.get("engaged").?.bool);
    try testing.expectEqual(@as(i64, 4096), last.get("tokens").?.integer);
}

test "cluster: recordPrefill keeps the reason by COPY (the scheduler's slice does not outlive the request)" {
    var why_buf = [_]u8{ 'd', 'e', 'a', 'd' };
    recordPrefill(false, 0, 0, why_buf[0..]);
    why_buf[0] = 'X';
    try testing.expectEqualStrings("dead", g_prefill_last.?.reason.?);
    g_prefill_last = null;
}
