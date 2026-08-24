//! Layer offload over llama.cpp's ggml RPC backend (rpc-offload-plan.md Part 1).
//!
//! PURE half: flag parsing, the memory preflight arithmetic, and the refusal
//! text. The engine-touching half lives in `src/arch/llama.zig`
//! (`LlamaEngine.open` with `OpenOptions.rpc_endpoints`, `serveRpc`) and the
//! wiring in main.zig / scheduler.zig. Keeping this file free of FFI is what
//! lets its tests run on every host, including the iOS stub build.
//!
//! Contract with the plan: fallback is NOT silent. An unreachable worker is a
//! NAMED load failure (the shim probes each endpoint before `llama_model_load`);
//! a model half-loaded on whatever is left is worse than no model.

const std = @import("std");

/// `--rpc host:port[,host:port...]`, set once at boot (borrowed from argv, like
/// `remote_prefill_client.g_remote_prefill_url`). Empty = feature off.
pub var g_endpoints: []const u8 = "";
/// `--tensor-split a,b[,c...]` — one weight per device in [local GPUs..., rpc...]
/// order. Empty = llama.cpp splits by advertised free memory.
pub var g_tensor_split: []const u8 = "";

/// `--rpc-serve [host:]port`. Empty = not a worker.
pub var g_rpc_serve: []const u8 = "";

pub const MAX_ENDPOINTS = 16; // GGML_RPC_MAX_SERVERS

/// `--rpc-serve` value → the `host:port` ggml's server binds. A bare port
/// binds every interface: a worker exists to be reached from another machine.
pub fn serveEndpoint(buf: []u8, spec: []const u8) ParseError![]const u8 {
    const t = std.mem.trim(u8, spec, " \t");
    if (t.len == 0) return ParseError.Empty;
    if (std.mem.lastIndexOfScalar(u8, t, ':')) |c| {
        if (c == 0 or c + 1 == t.len) return ParseError.BadEndpoint;
        _ = std.fmt.parseInt(u16, t[c + 1 ..], 10) catch return ParseError.BadEndpoint;
        return t;
    }
    _ = std.fmt.parseInt(u16, t, 10) catch return ParseError.BadEndpoint;
    return std.fmt.bufPrint(buf, "0.0.0.0:{s}", .{t}) catch return ParseError.BadEndpoint;
}

pub const ParseError = error{ Empty, TooMany, BadEndpoint, BadSplit };

/// Split a comma list of `host:port` into borrowed slices. Every entry must
/// carry a numeric port; a bare host is refused here rather than by ggml's
/// connect (which would read as "unreachable" and send the user to the network).
pub fn parseEndpoints(spec: []const u8, out: *[MAX_ENDPOINTS][]const u8) ParseError![]const []const u8 {
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, spec, ',');
    while (it.next()) |raw| {
        const e = std.mem.trim(u8, raw, " \t");
        if (e.len == 0) continue;
        if (n == MAX_ENDPOINTS) return ParseError.TooMany;
        const colon = std.mem.lastIndexOfScalar(u8, e, ':') orelse return ParseError.BadEndpoint;
        if (colon == 0 or colon + 1 == e.len) return ParseError.BadEndpoint;
        _ = std.fmt.parseInt(u16, e[colon + 1 ..], 10) catch return ParseError.BadEndpoint;
        out[n] = e;
        n += 1;
    }
    if (n == 0) return ParseError.Empty;
    return out[0..n];
}

/// `--tensor-split` weights. `n_devices` is the count the caller will hand
/// llama.cpp; a list of a different length is refused because llama.cpp
/// silently zero-fills the tail (a device with weight 0 gets NO layers).
pub fn parseTensorSplit(spec: []const u8, n_devices: usize, out: *[MAX_ENDPOINTS + 16]f32) ParseError![]const f32 {
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, spec, ',');
    while (it.next()) |raw| {
        const e = std.mem.trim(u8, raw, " \t");
        if (e.len == 0) continue;
        if (n == out.len) return ParseError.BadSplit;
        const v = std.fmt.parseFloat(f32, e) catch return ParseError.BadSplit;
        if (!(v >= 0)) return ParseError.BadSplit;
        out[n] = v;
        n += 1;
    }
    if (n == 0) return ParseError.Empty;
    if (n != n_devices) return ParseError.BadSplit;
    var sum: f32 = 0;
    for (out[0..n]) |v| sum += v;
    if (sum <= 0) return ParseError.BadSplit;
    return out[0..n];
}

/// Memory preflight for a split load. The requirement is the same helper the
/// local refusal uses (`scheduler.loadRequirementBytes`), compared against
/// local + every worker's FREE device bytes. Remote counts as capacity — that
/// is the whole reason this feature exists (plan step 4) — but a worker
/// reports only what the DEVICE has free, so the sum is honest.
pub const Capacity = struct {
    local_avail: u64,
    remote_free: u64,
    remote_count: usize,

    pub fn total(self: Capacity) u64 {
        return self.local_avail +| self.remote_free;
    }
};

pub fn insufficient(requirement: u64, cap: Capacity) bool {
    // Local 0 means "could not query" (never blocks, same as the local rule);
    // with workers present the remote number is real and stands on its own.
    if (cap.local_avail == 0 and cap.remote_count == 0) return false;
    return cap.total() < requirement;
}

/// A refusal names BOTH numbers (plan step 4): the local and the remote side.
pub fn refusalMessage(buf: []u8, requirement: u64, weights: u64, cap: Capacity) []const u8 {
    const gb = 1024.0 * 1024.0 * 1024.0;
    return std.fmt.bufPrint(buf, "Insufficient memory to load model across {d} RPC worker(s): needs ~{d:.1} GB ({d:.1} GB of weights plus headroom) but local has {d:.1} GB available and the workers report {d:.1} GB free ({d:.1} GB combined). Add a worker, pick a smaller quant, or pass --skip-mem-preflight to override.", .{
        cap.remote_count,
        @as(f64, @floatFromInt(requirement)) / gb,
        @as(f64, @floatFromInt(weights)) / gb,
        @as(f64, @floatFromInt(cap.local_avail)) / gb,
        @as(f64, @floatFromInt(cap.remote_free)) / gb,
        @as(f64, @floatFromInt(cap.total())) / gb,
    }) catch buf[0..0];
}

const testing = std.testing;

test "parseEndpoints: comma list, trimmed, every entry needs a numeric port" {
    var out: [MAX_ENDPOINTS][]const u8 = undefined;
    const eps = try parseEndpoints("192.168.0.150:50052, mini.local:50052 ,", &out);
    try testing.expectEqual(@as(usize, 2), eps.len);
    try testing.expectEqualStrings("192.168.0.150:50052", eps[0]);
    try testing.expectEqualStrings("mini.local:50052", eps[1]);
    try testing.expectError(ParseError.Empty, parseEndpoints("", &out));
    try testing.expectError(ParseError.Empty, parseEndpoints(" , ", &out));
    try testing.expectError(ParseError.BadEndpoint, parseEndpoints("192.168.0.150", &out));
    try testing.expectError(ParseError.BadEndpoint, parseEndpoints("host:", &out));
    try testing.expectError(ParseError.BadEndpoint, parseEndpoints(":50052", &out));
    try testing.expectError(ParseError.BadEndpoint, parseEndpoints("host:abc", &out));
    try testing.expectError(ParseError.BadEndpoint, parseEndpoints("host:70000", &out));
}

test "parseEndpoints: refuses more than GGML_RPC_MAX_SERVERS" {
    var out: [MAX_ENDPOINTS][]const u8 = undefined;
    var buf: [512]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    var i: usize = 0;
    while (i < MAX_ENDPOINTS + 1) : (i += 1) try w.print("h{d}:1,", .{i});
    try testing.expectError(ParseError.TooMany, parseEndpoints(w.buffered(), &out));
}

test "parseTensorSplit: length must equal the device count; weights non-negative, sum > 0" {
    var out: [MAX_ENDPOINTS + 16]f32 = undefined;
    const s = try parseTensorSplit("0.6,0.4", 2, &out);
    try testing.expectEqual(@as(usize, 2), s.len);
    try testing.expectApproxEqAbs(@as(f32, 0.6), s[0], 1e-6);
    try testing.expectError(ParseError.BadSplit, parseTensorSplit("1", 2, &out)); // short list = zero-filled tail = a device with no layers
    try testing.expectError(ParseError.BadSplit, parseTensorSplit("1,1,1", 2, &out));
    try testing.expectError(ParseError.BadSplit, parseTensorSplit("0,0", 2, &out));
    try testing.expectError(ParseError.BadSplit, parseTensorSplit("-1,2", 2, &out));
    try testing.expectError(ParseError.BadSplit, parseTensorSplit("x,1", 2, &out));
    try testing.expectError(ParseError.Empty, parseTensorSplit("", 2, &out));
}

test "preflight: remote free memory counts as capacity, and the refusal names both sides" {
    const GB: u64 = 1024 * 1024 * 1024;
    // 20 GB requirement: refused alone on a 12 GB box, admitted with a 16 GB worker.
    const alone = Capacity{ .local_avail = 12 * GB, .remote_free = 0, .remote_count = 0 };
    const split = Capacity{ .local_avail = 12 * GB, .remote_free = 16 * GB, .remote_count = 1 };
    try testing.expect(insufficient(20 * GB, alone));
    try testing.expect(!insufficient(20 * GB, split));
    try testing.expect(insufficient(30 * GB, split));
    // Unknown local (0) with no workers never blocks — same rule as the local gate.
    try testing.expect(!insufficient(20 * GB, .{ .local_avail = 0, .remote_free = 0, .remote_count = 0 }));
    // Unknown local WITH a worker: the worker's number is real and is compared.
    try testing.expect(insufficient(20 * GB, .{ .local_avail = 0, .remote_free = 16 * GB, .remote_count = 1 }));

    var buf: [512]u8 = undefined;
    const msg = refusalMessage(&buf, 30 * GB, 26 * GB, split);
    try testing.expect(std.mem.indexOf(u8, msg, "local has 12.0 GB") != null);
    try testing.expect(std.mem.indexOf(u8, msg, "workers report 16.0 GB") != null);
    try testing.expect(std.mem.indexOf(u8, msg, "28.0 GB combined") != null);
    try testing.expect(std.mem.indexOf(u8, msg, "needs ~30.0 GB") != null);
}

test "serveEndpoint: bare port binds all interfaces; host:port passes through; junk refused" {
    var buf: [64]u8 = undefined;
    try testing.expectEqualStrings("0.0.0.0:50052", try serveEndpoint(&buf, "50052"));
    try testing.expectEqualStrings("192.168.0.150:50052", try serveEndpoint(&buf, "192.168.0.150:50052"));
    try testing.expectError(ParseError.BadEndpoint, serveEndpoint(&buf, "abc"));
    try testing.expectError(ParseError.BadEndpoint, serveEndpoint(&buf, "host:"));
    try testing.expectError(ParseError.Empty, serveEndpoint(&buf, ""));
}
