//! LAN model sharing (v1 — LAN only; WAN rooms layer onto these same seams).
//!
//! Two independent halves, both OFF by default:
//!   • SHARE (`--lan-share <all|id,...>`): advertise this server as a Bonjour
//!     `_mlxserve._tcp` service and open the SHARED INFERENCE SURFACE to
//!     non-loopback clients — a route allowlist (`routeClass`) plus a
//!     shared-model check enforced in server.zig's LAN gate. Admin endpoints
//!     (load/unload, metrics, status page, stored responses) stay host-local.
//!   • DISCOVER (`--lan-discover`): browse for peers, mirror their shared
//!     models into `/v1/models` as `<id>@<peer>` entries, and transparently
//!     proxy any request naming one to its host (byte-for-byte streaming
//!     tunnel). Claude Code / any localhost client gets LAN models for free.
//!
//! Design rules:
//!   - The proxy is a TRANSPORT: no scheduler, no MLX, no inference-thread
//!     involvement. Tunnels run on the calling connection thread; discovery
//!     runs on one browser thread; dns_sd (mDNSResponder, in libSystem) does
//!     all mDNS work — no hand-rolled multicast.
//!   - Loops are impossible by construction: remote entries are never
//!     included in the model list served to LAN peers, and `@peer` ids from
//!     non-loopback clients are denied at the gate (no multi-hop).
//!   - The host sees tunneled prompts in plaintext (it computes on them) —
//!     the Swift Settings pane carries the disclosure.

const std = @import("std");
const log = @import("log.zig");

pub const SERVICE_TYPE = "_mlxserve._tcp";

// ─────────────────────────────────────────────────────────────────────────────
// Pure policy + codec helpers (hermetic tests at the bottom of this file)
// ─────────────────────────────────────────────────────────────────────────────

/// `<bare>@<peer>` — the id form remote models take in /v1/models. Local
/// registry ids never contain '@' (HF org/repo + dir basenames), and a
/// registered local id that does still wins at the dispatch site (registry
/// peek runs before the remote interception).
pub const RemoteId = struct { bare: []const u8, peer: []const u8 };

pub fn splitRemoteId(id: []const u8) ?RemoteId {
    const at = std.mem.lastIndexOfScalar(u8, id, '@') orelse return null;
    if (at == 0 or at + 1 == id.len) return null;
    return .{ .bare = id[0..at], .peer = id[at + 1 ..] };
}

/// What a non-loopback, non-API-key client may reach while sharing is on.
pub const RouteClass = enum { open, model_gated, denied };

pub fn routeClass(method: []const u8, path: []const u8) RouteClass {
    const eql = std.mem.eql;
    if (eql(u8, method, "OPTIONS")) return .open; // CORS preflight
    if (eql(u8, method, "GET")) {
        for ([_][]const u8{ "/health", "/v1/models", "/api/version" }) |p|
            if (eql(u8, path, p)) return .open;
        return .denied;
    }
    if (eql(u8, method, "POST")) {
        for ([_][]const u8{
            "/v1/chat/completions",        "/v1/completions",
            "/v1/messages",                "/v1/responses",
            "/v1/embeddings",              "/v1/images/generations",
            "/v1/images/edits",            "/v1/audio/speech",
            "/v1/audio/music-generations", "/v1/video/generations",
            "/v1/3d/generations",          "/api/chat",
            "/api/generate",               "/api/embed",
            "/api/embeddings",
        }) |p| if (eql(u8, path, p)) return .model_gated;
        return .denied;
    }
    return .denied;
}

/// Which local models `--lan-share` exposes: `all`, or a csv of ids.
/// Registry ids are `basename` or `org/name`; the app's share list sends
/// basenames — matching is symmetric basename-tolerant so the two can't drift.
pub const SharedSet = struct {
    all: bool = false,
    ids: []const []const u8 = &.{},

    pub fn parse(alloc: std.mem.Allocator, spec: []const u8) !SharedSet {
        const trimmed = std.mem.trim(u8, spec, " \t");
        if (std.ascii.eqlIgnoreCase(trimmed, "all")) return .{ .all = true };
        var list: std.ArrayList([]const u8) = .empty;
        errdefer {
            for (list.items) |s| alloc.free(s);
            list.deinit(alloc);
        }
        var it = std.mem.splitScalar(u8, trimmed, ',');
        while (it.next()) |raw| {
            const id = std.mem.trim(u8, raw, " \t");
            if (id.len > 0) try list.append(alloc, try alloc.dupe(u8, id));
        }
        return .{ .ids = try list.toOwnedSlice(alloc) };
    }

    pub fn deinit(self: *SharedSet, alloc: std.mem.Allocator) void {
        for (self.ids) |id| alloc.free(id);
        alloc.free(self.ids);
        self.* = .{};
    }

    pub fn empty(self: SharedSet) bool {
        return !self.all and self.ids.len == 0;
    }

    pub fn allows(self: SharedSet, id: []const u8) bool {
        if (self.all) return true;
        if (id.len == 0) return false;
        for (self.ids) |entry| {
            if (std.mem.eql(u8, entry, id) or
                std.mem.eql(u8, entry, basename(id)) or
                std.mem.eql(u8, basename(entry), id)) return true;
        }
        return false;
    }

    fn basename(s: []const u8) []const u8 {
        const slash = std.mem.lastIndexOfScalar(u8, s, '/') orelse return s;
        return s[slash + 1 ..];
    }
};

/// Bonjour instance names are arbitrary UTF-8, but the peer name doubles as a
/// model-id suffix — collapse anything outside [A-Za-z0-9._-] to '-' ('@'
/// would break the suffix split; spaces annoy CLIs). Never empty.
pub fn sanitizeName(buf: []u8, raw: []const u8) []const u8 {
    var n: usize = 0;
    var pending_dash = false;
    for (raw) |c| {
        if (std.ascii.isAlphanumeric(c) or c == '.' or c == '_' or c == '-') {
            if (pending_dash and n > 0 and n < buf.len) {
                buf[n] = '-';
                n += 1;
            }
            pending_dash = false;
            if (n < buf.len) {
                buf[n] = c;
                n += 1;
            }
        } else pending_dash = true;
    }
    return if (n == 0) "mac" else buf[0..n];
}

/// Rebuild `body` with the model value replaced by `bare_id`. `model_value`
/// MUST be the slice `server.parseModelFromBody` returned — i.e. alias `body`
/// — so the decide and rewrite layers can never disagree on which bytes are
/// the model field.
pub fn rewriteModelValue(alloc: std.mem.Allocator, body: []const u8, model_value: []const u8, bare_id: []const u8) ![]u8 {
    const start = @intFromPtr(model_value.ptr) - @intFromPtr(body.ptr);
    std.debug.assert(start + model_value.len <= body.len);
    const out = try alloc.alloc(u8, body.len - model_value.len + bare_id.len);
    @memcpy(out[0..start], body[0..start]);
    @memcpy(out[start..][0..bare_id.len], bare_id);
    @memcpy(out[start + bare_id.len ..], body[start + model_value.len ..]);
    return out;
}

/// Collapse JSON's optional `\/` escape to `/`. Swift's JSONSerialization
/// (and PHP's json_encode) escape every slash, so a remote id's org prefix
/// arrives as `ddalcu\/gemma…` from the app while the peer table stores the
/// canonical form — the same `\/` class the load-model handler documents.
/// Returns the input verbatim (zero-copy) when there is nothing to collapse
/// or the scratch buffer is too small.
pub fn unescapeJsonSlashes(buf: []u8, s: []const u8) []const u8 {
    if (std.mem.indexOf(u8, s, "\\/") == null or s.len > buf.len) return s;
    var n: usize = 0;
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        if (s[i] == '\\' and i + 1 < s.len and s[i + 1] == '/') continue;
        buf[n] = s[i];
        n += 1;
    }
    return buf[0..n];
}

/// What a peer can do for us beyond serving chat (rpc-offload-plan.md Part 2).
/// Advertised two ways: cheaply in the TXT record (`rpc=<port>`, `pf=1`) and in
/// full from the peer's own `GET /v1/cluster` at install time (kv type, model).
pub const PeerCaps = struct {
    /// ggml RPC worker port (`--rpc-serve`), null when not a worker.
    rpc_port: ?u16 = null,
    /// Serves `POST /v1/prefill` (every llama.cpp build does).
    prefill: bool = false,
    /// The prefill worker's KV type name ("f16" / "q8_0"); a consumer only
    /// picks a worker whose type matches its own (a mismatch is a restore ERROR).
    prefill_kv: [8]u8 = [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0 },
    prefill_kv_len: u8 = 0,

    pub fn kv(self: *const PeerCaps) []const u8 {
        return self.prefill_kv[0..self.prefill_kv_len];
    }
    pub fn setKv(self: *PeerCaps, name: []const u8) void {
        const n: u8 = @intCast(@min(name.len, self.prefill_kv.len));
        @memcpy(self.prefill_kv[0..n], name[0..n]);
        self.prefill_kv_len = n;
    }
};

/// THIS server's capabilities, set once at boot (main/server) and folded into
/// every advertisement by `txtBuild`. A plain var so the builder keeps its
/// one-argument shape at both transports' call sites.
pub var g_local_caps: PeerCaps = .{};

fn txtAppend(buf: []u8, at: usize, entry: []const u8) usize {
    if (at + 1 + entry.len > buf.len or entry.len > 255) return at;
    buf[at] = @intCast(entry.len);
    @memcpy(buf[at + 1 ..][0..entry.len], entry);
    return at + 1 + entry.len;
}

/// TXT record wire format: length-prefixed `key=value` strings. `v=1` and
/// the process token first (every reader keys on those), then the optional
/// capability keys — a reader that does not know them skips them.
pub fn txtBuild(buf: []u8, token: []const u8) []const u8 {
    std.debug.assert(buf.len >= 5 + 2 + token.len and token.len <= 253);
    buf[0] = 3;
    @memcpy(buf[1..4], "v=1");
    buf[4] = @intCast(2 + token.len);
    @memcpy(buf[5..7], "t=");
    @memcpy(buf[7..][0..token.len], token);
    var n: usize = 7 + token.len;
    if (g_local_caps.rpc_port) |port| {
        var tmp: [16]u8 = undefined;
        const e = std.fmt.bufPrint(&tmp, "rpc={d}", .{port}) catch unreachable;
        n = txtAppend(buf, n, e);
    }
    if (g_local_caps.prefill) n = txtAppend(buf, n, "pf=1");
    return buf[0..n];
}

/// Caps out of a peer's `GET /v1/cluster` body. Anything unreadable is
/// simply "no capability" — a peer on an older build answers 404 here.
pub fn parsePeerCaps(alloc: std.mem.Allocator, body: []const u8) PeerCaps {
    var caps: PeerCaps = .{};
    var parsed = std.json.parseFromSlice(std.json.Value, alloc, body, .{}) catch return caps;
    defer parsed.deinit();
    if (parsed.value != .object) return caps;
    const root = parsed.value.object;
    if (root.get("rpc")) |rpc| if (rpc == .object) if (rpc.object.get("serve")) |serve| if (serve == .object) {
        if (serve.object.get("endpoint")) |ep| if (ep == .string) {
            if (std.mem.lastIndexOfScalar(u8, ep.string, ':')) |c| {
                caps.rpc_port = std.fmt.parseInt(u16, ep.string[c + 1 ..], 10) catch null;
            }
        };
    };
    if (root.get("prefill")) |pf| if (pf == .object) {
        if (pf.object.get("mode")) |m| if (m == .string) {
            caps.prefill = std.mem.eql(u8, m.string, "worker");
        };
        if (pf.object.get("kv_type")) |k| if (k == .string) caps.setKv(k.string);
    };
    return caps;
}

pub fn txtFind(txt: []const u8, key_eq: []const u8) ?[]const u8 {
    var i: usize = 0;
    while (i < txt.len) {
        const len: usize = txt[i];
        i += 1;
        if (i + len > txt.len) return null; // truncated/hostile record
        const entry = txt[i .. i + len];
        i += len;
        if (std.mem.startsWith(u8, entry, key_eq)) return entry[key_eq.len..];
    }
    return null;
}

/// One shared model as advertised by a peer: its bare id (for request
/// routing) and the ready-to-emit /v1/models entry JSON (id rewritten to
/// `<id>@<peer>`, plus a top-level `"lan_peer"` marker clients badge on).
pub const PeerModel = struct { id: []const u8, entry_json: []const u8 };

pub fn freePeerModels(alloc: std.mem.Allocator, models: []PeerModel) void {
    for (models) |m| {
        alloc.free(m.id);
        alloc.free(m.entry_json);
    }
    alloc.free(models);
}

/// Parse a peer's /v1/models response body into PeerModels. Entries that
/// aren't objects with a string id are skipped; a body without a `data`
/// array is an error (not an mlx-serve peer).
pub fn parsePeerModels(alloc: std.mem.Allocator, body: []const u8, peer: []const u8) ![]PeerModel {
    var parsed = std.json.parseFromSlice(std.json.Value, alloc, body, .{}) catch return error.BadPeerJson;
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadPeerJson;
    const data = parsed.value.object.get("data") orelse return error.BadPeerJson;
    if (data != .array) return error.BadPeerJson;

    var out: std.ArrayList(PeerModel) = .empty;
    errdefer {
        for (out.items) |m| {
            alloc.free(m.id);
            alloc.free(m.entry_json);
        }
        out.deinit(alloc);
    }
    for (data.array.items) |*item| {
        if (item.* != .object) continue;
        const id_v = item.object.get("id") orelse continue;
        if (id_v != .string) continue;
        // Never mirror the peer's own REMOTE stubs (entries badged lan_peer):
        // a peer that fails to filter its list — old binary, or a loopback
        // fetch between two servers on one Mac bypassing the non-loopback
        // gate — would otherwise chain re-exports into `@a@b` ids.
        if (item.object.get("lan_peer") != null) continue;
        const bare = try alloc.dupe(u8, id_v.string);
        errdefer alloc.free(bare);
        // Mutations of the parsed tree allocate from ITS arena (freed
        // wholesale by parsed.deinit) — never mix the caller's gpa in.
        const arena = parsed.arena.allocator();
        const full = try std.fmt.allocPrint(arena, "{s}@{s}", .{ bare, peer });
        try item.object.put(arena, "id", .{ .string = full });
        try item.object.put(arena, "lan_peer", .{ .string = peer });
        const entry_json = try std.json.Stringify.valueAlloc(alloc, item.*, .{});
        errdefer alloc.free(entry_json);
        try out.append(alloc, .{ .id = bare, .entry_json = entry_json });
    }
    return out.toOwnedSlice(alloc);
}

// ─────────────────────────────────────────────────────────────────────────────
// dns_sd FFI (mDNSResponder client API — exported by libSystem, no extra link)
// ─────────────────────────────────────────────────────────────────────────────

// ── Portable half ──────────────────────────────────────────────────────────
//
// Everything above the Bonjour transport: id parsing, the route allowlist, the
// shared-model set, name sanitizing, TXT records and peer-model rewriting.
// None of it touches dns_sd or a socket, so it builds and is tested on every
// host. `src/lan_bonjour.zig` holds the Apple-only transport that consumes it,
// and `src/lan.zig` re-exports the pair. See src/lan.zig for the split's why.

const t = std.testing;

test "lan: splitRemoteId splits on the LAST @ and rejects degenerate forms" {
    const r = splitRemoteId("gemma-4-e4b-it-4bit@Davids-Mac").?;
    try t.expectEqualStrings("gemma-4-e4b-it-4bit", r.bare);
    try t.expectEqualStrings("Davids-Mac", r.peer);
    // A bare id that itself contains '@' resolves to the last suffix.
    const r2 = splitRemoteId("weird@name@peer1").?;
    try t.expectEqualStrings("weird@name", r2.bare);
    try t.expectEqualStrings("peer1", r2.peer);
    try t.expect(splitRemoteId("no-at-here") == null);
    try t.expect(splitRemoteId("@peer") == null);
    try t.expect(splitRemoteId("model@") == null);
    try t.expect(splitRemoteId("") == null);
}

test "lan: routeClass allows exactly the shared inference surface" {
    // Open probes.
    try t.expectEqual(RouteClass.open, routeClass("OPTIONS", "/v1/chat/completions"));
    try t.expectEqual(RouteClass.open, routeClass("GET", "/health"));
    try t.expectEqual(RouteClass.open, routeClass("GET", "/v1/models"));
    try t.expectEqual(RouteClass.open, routeClass("GET", "/api/version"));
    // Inference is model-gated on every surface, media included.
    for ([_][]const u8{
        "/v1/chat/completions",        "/v1/completions",
        "/v1/messages",                "/v1/responses",
        "/v1/embeddings",              "/v1/images/generations",
        "/v1/images/edits",            "/v1/audio/speech",
        "/v1/audio/music-generations", "/v1/video/generations",
        "/v1/3d/generations",          "/api/chat",
        "/api/generate",               "/api/embed",
        "/api/embeddings",
    }) |p| try t.expectEqual(RouteClass.model_gated, routeClass("POST", p));
    // Admin/host-local stays denied.
    try t.expectEqual(RouteClass.denied, routeClass("POST", "/v1/load-model"));
    try t.expectEqual(RouteClass.denied, routeClass("POST", "/v1/unload-model"));
    try t.expectEqual(RouteClass.denied, routeClass("POST", "/v1/responses/compact"));
    try t.expectEqual(RouteClass.denied, routeClass("GET", "/"));
    try t.expectEqual(RouteClass.denied, routeClass("GET", "/metrics"));
    try t.expectEqual(RouteClass.denied, routeClass("GET", "/metrics.json"));
    try t.expectEqual(RouteClass.denied, routeClass("GET", "/v1/responses/resp_123"));
    try t.expectEqual(RouteClass.denied, routeClass("DELETE", "/v1/responses/resp_123"));
    try t.expectEqual(RouteClass.denied, routeClass("GET", "/api/tags"));
    try t.expectEqual(RouteClass.denied, routeClass("POST", "/api/pull"));
    try t.expectEqual(RouteClass.denied, routeClass("GET", "/props"));
}

test "lan: SharedSet parses all|csv and matches basename-tolerantly" {
    const a = t.allocator;
    var all = try SharedSet.parse(a, "all");
    defer all.deinit(a);
    try t.expect(all.all);
    try t.expect(!all.empty());
    try t.expect(all.allows("anything"));

    var set = try SharedSet.parse(a, "gemma-4-e4b-it-4bit, mlx-community/bge-small-en-v1.5-8bit");
    defer set.deinit(a);
    try t.expect(!set.empty());
    // Exact.
    try t.expect(set.allows("gemma-4-e4b-it-4bit"));
    // Registry id is org/name, share entry is the basename.
    try t.expect(set.allows("some-org/gemma-4-e4b-it-4bit"));
    // Share entry is org/name, registry id is the basename.
    try t.expect(set.allows("bge-small-en-v1.5-8bit"));
    try t.expect(!set.allows("qwen3.6-27b"));
    try t.expect(!set.allows(""));

    var none = try SharedSet.parse(a, " , ");
    defer none.deinit(a);
    try t.expect(none.empty());
    try t.expect(!none.allows("gemma-4-e4b-it-4bit"));
}

test "lan: sanitizeName collapses hostile chars and never returns empty" {
    var buf: [64]u8 = undefined;
    try t.expectEqualStrings("David-s-MacBook-Pro", sanitizeName(&buf, "David's MacBook Pro"));
    try t.expectEqualStrings("mac-2", sanitizeName(&buf, "mac (2)"));
    // '@' must never survive — it's the remote-id delimiter.
    try t.expectEqualStrings("a-b.local", sanitizeName(&buf, "a@b.local"));
    try t.expectEqualStrings("mac", sanitizeName(&buf, "!!!"));
    try t.expectEqualStrings("mac", sanitizeName(&buf, ""));
    try t.expectEqualStrings("plain-name_1.local", sanitizeName(&buf, "plain-name_1.local"));
}

test "lan: rewriteModelValue splices the aliased model value in place" {
    const a = t.allocator;
    const body = "{\"model\":\"qwen3.6-27b@Studio\",\"messages\":[{\"role\":\"user\",\"content\":\"hi @Studio\"}]}";
    const val_start = std.mem.indexOf(u8, body, "qwen3.6-27b@Studio").?;
    const model_value = body[val_start .. val_start + "qwen3.6-27b@Studio".len];
    const out = try rewriteModelValue(a, body, model_value, "qwen3.6-27b");
    defer a.free(out);
    try t.expectEqualStrings(
        "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi @Studio\"}]}",
        out,
    );
}

test "lan: TXT record round-trips the instance token" {
    var buf: [64]u8 = undefined;
    const txt = txtBuild(&buf, "deadbeefcafef00d");
    try t.expect(txt.len > 0);
    try t.expectEqualStrings("deadbeefcafef00d", txtFind(txt, "t=").?);
    try t.expectEqualStrings("1", txtFind(txt, "v=").?);
    try t.expect(txtFind(txt, "x=") == null);
    // Truncated/hostile TXT never panics.
    try t.expect(txtFind(&[_]u8{200}, "t=") == null);
    try t.expect(txtFind(&[_]u8{}, "t=") == null);
}

test "lan: TXT record advertises rpc port + prefill only when this server has them" {
    var buf: [96]u8 = undefined;
    const saved = g_local_caps;
    defer g_local_caps = saved;
    g_local_caps = .{};
    var txt = txtBuild(&buf, "deadbeefcafef00d");
    try t.expect(txtFind(txt, "rpc=") == null);
    try t.expect(txtFind(txt, "pf=") == null);
    g_local_caps = .{ .rpc_port = 50052, .prefill = true };
    txt = txtBuild(&buf, "deadbeefcafef00d");
    try t.expectEqualStrings("50052", txtFind(txt, "rpc=").?);
    try t.expectEqualStrings("1", txtFind(txt, "pf=").?);
    try t.expectEqualStrings("deadbeefcafef00d", txtFind(txt, "t=").?); // token still first
}

test "lan: parsePeerCaps reads a /v1/cluster body; garbage and old peers mean no caps" {
    const body =
        \\{"self":{},"rpc":{"role":"worker","serve":{"endpoint":"0.0.0.0:50052","device":"CUDA0"}},"prefill":{"mode":"worker","kv_type":"q8_0"}}
    ;
    const c = parsePeerCaps(t.allocator, body);
    try t.expectEqual(@as(?u16, 50052), c.rpc_port);
    try t.expect(c.prefill);
    try t.expectEqualStrings("q8_0", c.kv());
    const none = parsePeerCaps(t.allocator, "{\"rpc\":{\"serve\":null},\"prefill\":{\"mode\":\"consumer\"}}");
    try t.expect(none.rpc_port == null and !none.prefill);
    const junk = parsePeerCaps(t.allocator, "<html>404</html>");
    try t.expect(junk.rpc_port == null and !junk.prefill);
}

test "lan: parsePeerModels rewrites ids, adds lan_peer, keeps meta" {
    const a = t.allocator;
    const body =
        \\{"object":"list","data":[
        \\ {"id":"gemma-4-e4b-it-4bit","object":"model","loaded":true,"capabilities":["chat","vision"],"meta":{"context_length":94000}},
        \\ {"id":"flux2-klein-4bit","object":"model","loaded":false,"capabilities":["image"]},
        \\ {"id":"qwen3.6-27b@SomeoneElse","object":"model","loaded":true,"lan_peer":"SomeoneElse"},
        \\ {"id":42,"object":"junk"}
        \\]}
    ;
    // The lan_peer entry above is the peer's OWN remote stub — a peer that
    // fails to filter its list (old binary, loopback fetch between two
    // servers on one Mac) would otherwise chain re-exports (`@a@b` ids;
    // live 2026-07-21: a third server on the test Mac leaked its mirrors
    // into test_lan_share's servers). Never mirror someone else's mirror.
    const models = try parsePeerModels(a, body, "Studio");
    defer freePeerModels(a, models);
    try t.expectEqual(@as(usize, 2), models.len);
    try t.expectEqualStrings("gemma-4-e4b-it-4bit", models[0].id);
    // Entry JSON carries the suffixed id + the lan_peer badge + original meta.
    try t.expect(std.mem.indexOf(u8, models[0].entry_json, "\"id\":\"gemma-4-e4b-it-4bit@Studio\"") != null);
    try t.expect(std.mem.indexOf(u8, models[0].entry_json, "\"lan_peer\":\"Studio\"") != null);
    try t.expect(std.mem.indexOf(u8, models[0].entry_json, "\"context_length\":94000") != null);
    try t.expect(std.mem.indexOf(u8, models[1].entry_json, "\"id\":\"flux2-klein-4bit@Studio\"") != null);
    // Not an mlx-serve shape → error, not a crash.
    try t.expectError(error.BadPeerJson, parsePeerModels(a, "{\"nope\":true}", "x"));
    try t.expectError(error.BadPeerJson, parsePeerModels(a, "not json", "x"));
}

test "lan: JSON-escaped slashes canonicalize (Swift clients send org\\/name)" {
    var buf: [256]u8 = undefined;
    // JSONSerialization (Swift, PHP, …) legally escapes '/' as '\/'. The org/
    // prefix of a remote id then misses the byte-compare — live 404 "no longer
    // shares this model" from the app on ddalcu\/gemma-4-e2b…@Davids-Mac-mini.
    try t.expectEqualStrings("ddalcu/gemma-e2b@Mini", unescapeJsonSlashes(&buf, "ddalcu\\/gemma-e2b@Mini"));
    // No escapes → the INPUT slice comes back verbatim (zero-copy).
    const plain = "ddalcu/gemma-e2b@Mini";
    try t.expect(unescapeJsonSlashes(&buf, plain).ptr == plain.ptr);
    // Only the two-byte sequence `\/` collapses; other backslashes survive.
    try t.expectEqualStrings("a\\b/c", unescapeJsonSlashes(&buf, "a\\b\\/c"));
    // Oversized input degrades to verbatim rather than truncating.
    var tiny: [4]u8 = undefined;
    try t.expectEqualStrings("x\\/y", unescapeJsonSlashes(tiny[0..2], "x\\/y"));
}
