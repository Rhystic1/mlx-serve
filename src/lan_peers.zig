//! The transport-independent half of LAN sharing: the discovered-peer table,
//! the peer model fetch, and the streaming proxy tunnel.
//!
//! Split out of `lan_bonjour.zig` when the hand-rolled mDNS transport landed —
//! none of this is Bonjour-specific, and a second copy of the tunnel would be a
//! second contract. What stays per-transport is only what the transport itself
//! shapes: how a service is discovered, and how a "known" service is retried
//! (dns_sd re-resolves by name+domain, mDNS re-queries by name).
//!
//! `lan_policy.zig` remains the wire spec above this; `lan_net.zig` is the
//! socket layer below it.

const std = @import("std");
const net = @import("lan_net.zig");
const log = @import("log.zig");
const policy = @import("lan_policy.zig");
const platform = @import("platform.zig");

const PeerModel = policy.PeerModel;
const freePeerModels = policy.freePeerModels;
const parsePeerModels = policy.parsePeerModels;
const splitRemoteId = policy.splitRemoteId;

pub const Remote = struct { ip4: [4]u8, port: u16 };

pub const Peer = struct {
    display: []u8, // sanitized instance name — also the hash key
    ip4: [4]u8,
    port: u16,
    models: []PeerModel,

    pub fn deinit(p: *Peer, alloc: std.mem.Allocator) void {
        freePeerModels(alloc, p.models);
        alloc.free(p.display);
    }
};

/// Consecutive failed resolves before a service is forgotten (~4 min at the
/// 10 s refresh cadence). Until then the service keeps retrying quietly.
pub const KNOWN_MAX_FAILS: u8 = 24;
/// Consecutive failed resolves before an INSTALLED peer leaves the table
/// (~20-30 s at the refresh cadence). One transient discovery hiccup — a busy
/// responder, an interface appearing/vanishing (VM or docker bridge), a 3 s
/// resolve timeout while the peer's GPU is pinned by a load — must not evict a
/// live peer: its cached ip4:port still tunnels, and eviction turns the next
/// chat into a "LAN peer for this model is offline" 404 (live 2026-07-19:
/// chats through a proxy alternated success/404 while the peer stayed up and
/// advertising the whole time). A genuinely-gone peer still leaves within
/// PEER_DROP_FAILS refreshes; the tunnel answers 502 honestly if it is picked
/// during the grace window.
pub const PEER_DROP_FAILS: u8 = 3;

pub const KnownFailureAction = enum { retain, drop_peer, drop_and_forget };

/// Pure policy: what a known service's consecutive-failure count (AFTER
/// incrementing for the current failure) does to the peer table + registry.
pub fn knownFailureAction(fails: u8) KnownFailureAction {
    if (fails >= KNOWN_MAX_FAILS) return .drop_and_forget;
    if (fails >= PEER_DROP_FAILS) return .drop_peer;
    return .retain;
}

pub const RemoteLookup = union(enum) { found: Remote, peer_unknown, model_unlisted };

/// The discovered-peer table. Read from connection threads and written from
/// the discovery thread, hence the lock.
pub const Table = struct {
    alloc: std.mem.Allocator,
    /// `platform.Mutex`, not `std.Io.Mutex`: lookups run on conn threads and
    /// the discovery thread, none of which carries an `Io` handle to block
    /// through for a micro critical section.
    mu: platform.Mutex = .{},
    map: std.StringHashMap(Peer),

    pub fn init(alloc: std.mem.Allocator) Table {
        return .{ .alloc = alloc, .map = std.StringHashMap(Peer).init(alloc) };
    }

    pub fn deinit(tbl: *Table) void {
        var it = tbl.map.valueIterator();
        while (it.next()) |p| p.deinit(tbl.alloc);
        tbl.map.deinit();
    }

    fn lock(tbl: *Table) void {
        tbl.mu.lock();
    }
    fn unlock(tbl: *Table) void {
        tbl.mu.unlock();
    }

    /// Tri-state on purpose. `model_unlisted` is definitive — the peer
    /// answered recently and does not serve this model, so a typo or a probe
    /// fails fast. `peer_unknown` means the peer is not in the table (yet):
    /// offline, mid-restart, or discovery still converging — the proxy WAITS
    /// briefly and retries instead of failing instantly (live: a chat fired
    /// while the peer Mac was redeploying — or right after a local restart —
    /// got an instant misleading 404). A peer installed with an EMPTY model
    /// list (mid-boot) counts as unknown so the wait covers it too.
    pub fn lookupRemote(tbl: *Table, id: []const u8) RemoteLookup {
        const rid = splitRemoteId(id) orelse return .peer_unknown;
        tbl.lock();
        defer tbl.unlock();
        const p = tbl.map.getPtr(rid.peer) orelse return .peer_unknown;
        for (p.models) |m|
            if (std.mem.eql(u8, m.id, rid.bare)) return .{ .found = .{ .ip4 = p.ip4, .port = p.port } };
        return if (p.models.len == 0) .peer_unknown else .model_unlisted;
    }

    /// Owned copy of the /v1/models entry JSON for a remote id (the
    /// load-model no-op renders it so app flows work unchanged).
    pub fn remoteEntryFor(tbl: *Table, alloc: std.mem.Allocator, id: []const u8) ?[]u8 {
        const rid = splitRemoteId(id) orelse return null;
        tbl.lock();
        defer tbl.unlock();
        const p = tbl.map.getPtr(rid.peer) orelse return null;
        for (p.models) |m|
            if (std.mem.eql(u8, m.id, rid.bare)) return alloc.dupe(u8, m.entry_json) catch null;
        return null;
    }

    /// Append every discovered remote model's entry JSON to a /v1/models
    /// `data` array under construction (comma-managed by buffer length).
    /// `[{"name","ip","port","models":[ids]},...]` for `GET /v1/cluster`.
    pub fn appendPeersJson(tbl: *Table, alloc: std.mem.Allocator, buf: *std.ArrayList(u8)) !void {
        const chat = @import("chat.zig");
        tbl.lock();
        defer tbl.unlock();
        try buf.append(alloc, '[');
        var it = tbl.map.valueIterator();
        var first = true;
        while (it.next()) |p| {
            if (!first) try buf.append(alloc, ',');
            first = false;
            try buf.appendSlice(alloc, "{\"name\":");
            try chat.appendJsonString(alloc, buf, p.display);
            try buf.print(alloc, ",\"ip\":\"{d}.{d}.{d}.{d}\",\"port\":{d},\"models\":[", .{ p.ip4[0], p.ip4[1], p.ip4[2], p.ip4[3], p.port });
            for (p.models, 0..) |m, i| {
                if (i > 0) try buf.append(alloc, ',');
                try chat.appendJsonString(alloc, buf, m.id);
            }
            try buf.appendSlice(alloc, "]}");
        }
        try buf.append(alloc, ']');
    }

    pub fn appendRemoteEntries(tbl: *Table, alloc: std.mem.Allocator, buf: *std.ArrayList(u8)) !void {
        tbl.lock();
        defer tbl.unlock();
        var it = tbl.map.valueIterator();
        while (it.next()) |p| for (p.models) |m| {
            if (buf.items.len > 0) try buf.append(alloc, ',');
            try buf.appendSlice(alloc, m.entry_json);
        };
    }

    pub fn remove(tbl: *Table, display: []const u8) void {
        tbl.lock();
        defer tbl.unlock();
        if (tbl.map.fetchRemove(display)) |kv| {
            var p = kv.value;
            p.deinit(tbl.alloc);
        }
    }

    /// Takes ownership of `models` either way.
    pub fn install(tbl: *Table, display: []const u8, ip4: [4]u8, port: u16, models: []PeerModel) void {
        const owned = tbl.alloc.dupe(u8, display) catch {
            freePeerModels(tbl.alloc, models);
            return;
        };
        tbl.lock();
        defer tbl.unlock();
        if (tbl.map.fetchRemove(display)) |kv| {
            var old = kv.value;
            old.deinit(tbl.alloc);
        }
        const p = Peer{ .display = owned, .ip4 = ip4, .port = port, .models = models };
        tbl.map.put(p.display, p) catch {
            var tmp = p;
            tmp.deinit(tbl.alloc);
        };
    }

    /// Would installing `count` models for `display` change what we report?
    /// Used only to keep the "peer shares N models" log line to real changes.
    pub fn modelCountDiffers(tbl: *Table, display: []const u8, count: usize) bool {
        tbl.lock();
        defer tbl.unlock();
        const p = tbl.map.getPtr(display) orelse return true;
        return p.models.len != count;
    }

    pub fn contains(tbl: *Table, display: []const u8) bool {
        tbl.lock();
        defer tbl.unlock();
        return tbl.map.contains(display);
    }
};

// ── Peer model fetch ────────────────────────────────────────────────────────

/// Case-insensitive header lookup in a raw HTTP head. `name_lower` must be
/// lowercase; matches at line starts only. Returns the trimmed value.
pub fn headerValueCI(head: []const u8, name_lower: []const u8) ?[]const u8 {
    var it = std.mem.splitSequence(u8, head, "\r\n");
    while (it.next()) |line| {
        if (line.len < name_lower.len + 1) continue;
        var matches = true;
        for (name_lower, 0..) |c, i| {
            if (std.ascii.toLower(line[i]) != c) {
                matches = false;
                break;
            }
        }
        if (!matches or line[name_lower.len] != ':') continue;
        return std.mem.trim(u8, line[name_lower.len + 1 ..], " \t");
    }
    return null;
}

/// `error.SelfFetch` when the response carries OUR OWN process token
/// (`X-MLX-LAN-Token`): a stale record of a former self — same name and port,
/// different TXT token, so the discovery-time TXT check can't catch it —
/// resolves back to this very server, and the loopback-first fetch would
/// happily install our own models as a "peer" (live test_lan_share self-mirror
/// after the peer-restart section, 2026-07-21).
pub fn fetchPeerModels(alloc: std.mem.Allocator, ip4: [4]u8, port: u16, peer_display: []const u8, own_token: []const u8) ![]PeerModel {
    const s = try net.connectTimeout(ip4, port, 3000);
    defer net.close(s);
    try net.writeAll(s, "GET /v1/models HTTP/1.1\r\nHost: mlx-serve\r\nConnection: close\r\nX-MLX-LAN: 1\r\n\r\n");
    var resp: std.ArrayList(u8) = .empty;
    defer resp.deinit(alloc);
    var chunk: [16 * 1024]u8 = undefined;
    while (resp.items.len < 8 * 1024 * 1024) {
        // A peer that accepts the connection and then stalls must not hold the
        // discovery thread forever; the poll is the read deadline.
        if (!net.waitReadable(s, 5000)) break;
        const n = net.read(s, &chunk) catch break;
        if (n == 0) break;
        try resp.appendSlice(alloc, chunk[0..n]);
    }
    const raw = resp.items;
    const line_end = std.mem.indexOf(u8, raw, "\r\n") orelse return error.BadPeerJson;
    if (std.mem.indexOf(u8, raw[0..line_end], " 200") == null) return error.BadPeerJson;
    const header_end = std.mem.indexOf(u8, raw, "\r\n\r\n") orelse return error.BadPeerJson;
    if (headerValueCI(raw[0..header_end], "x-mlx-lan-token")) |tok| {
        if (std.mem.eql(u8, tok, own_token)) return error.SelfFetch;
    }
    return parsePeerModels(alloc, raw[header_end + 4 ..], peer_display);
}

// ── Proxy tunnel ────────────────────────────────────────────────────────────

/// Proxy one request to `remote` and pump the response back byte-for-byte.
/// `conn` needs `writeAll([]const u8) !void` + `peerClosed() bool` — the
/// server's `*Conn` fits, and the duck typing keeps this file
/// server-independent and the pump hermetically testable.
/// `error.PeerUnreachable` is returned BEFORE anything is written to `conn`
/// (the caller can still send a clean 502); any later failure just ends the
/// stream — the client sees a closed socket, the peer sees a disconnect and
/// cancels its slot.
pub fn tunnel(remote: Remote, method: []const u8, raw_path: []const u8, body: []const u8, conn: anytype) error{PeerUnreachable}!void {
    const s = net.connectTimeout(remote.ip4, remote.port, 3000) catch return error.PeerUnreachable;
    defer net.close(s);
    var head_buf: [1024]u8 = undefined;
    const head = std.fmt.bufPrint(
        &head_buf,
        "{s} {s} HTTP/1.1\r\nHost: {d}.{d}.{d}.{d}:{d}\r\nContent-Type: application/json\r\nAccept: */*\r\nConnection: close\r\nX-MLX-LAN: 1\r\nContent-Length: {d}\r\n\r\n",
        .{ method, raw_path, remote.ip4[0], remote.ip4[1], remote.ip4[2], remote.ip4[3], remote.port, body.len },
    ) catch return error.PeerUnreachable;
    net.writeAll(s, head) catch return error.PeerUnreachable;
    net.writeAll(s, body) catch return error.PeerUnreachable;

    // Pump peer → client until peer EOF. The 1 s poll tick doubles as the
    // client-disconnect probe so an abandoned generation is torn down on the
    // peer too (its own disconnect-cancel machinery fires when we close).
    var buf: [16 * 1024]u8 = undefined;
    while (true) {
        if (!net.waitReadable(s, 1000)) {
            if (conn.peerClosed()) return;
            continue;
        }
        const n = net.read(s, &buf) catch return;
        if (n == 0) return;
        conn.writeAll(buf[0..n]) catch return;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
//
// These moved here with the code they cover. They ran on macOS only while the
// tunnel lived in the Bonjour file; the peer table, the fetch and the tunnel
// are portable, so now they run everywhere — which is how the Linux port found
// out they pass there too.
// ─────────────────────────────────────────────────────────────────────────────

const t = std.testing;

test "lan: headerValueCI finds a header case-insensitively and trims the value" {
    const head = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nX-MLX-LAN-Token: deadbeefcafef00d\r\n";
    try t.expectEqualStrings("deadbeefcafef00d", headerValueCI(head, "x-mlx-lan-token").?);
    try t.expectEqualStrings("application/json", headerValueCI(head, "content-type").?);
    try t.expect(headerValueCI(head, "x-missing") == null);
    // Name must match at line start, not mid-header.
    try t.expect(headerValueCI("X-Foo-Bar: 1\r\n", "bar") == null);
}

test "lan: transient resolve failures retain a live peer; only persistent failure drops it" {
    // Grace policy for the browse thread's failure bookkeeping. dns_sd
    // resolves hiccup transiently on a LIVE peer (busy mDNSResponder, a
    // VM/docker bridge interface appearing or vanishing mid-toggle, a 3 s
    // resolve timeout while the peer's GPU is pinned) — one such hiccup
    // must NOT evict the peer from the table: the entry's cached ip4:port
    // still tunnels fine, and eviction turns the next chat into a
    // user-visible "LAN peer for this model is offline" 404. Only a
    // PERSISTENT failure streak drops the peer, and only KNOWN_MAX_FAILS
    // forgets the service name entirely.
    try t.expectEqual(KnownFailureAction.retain, knownFailureAction(1));
    try t.expectEqual(KnownFailureAction.retain, knownFailureAction(PEER_DROP_FAILS - 1));
    try t.expectEqual(KnownFailureAction.drop_peer, knownFailureAction(PEER_DROP_FAILS));
    try t.expectEqual(KnownFailureAction.drop_peer, knownFailureAction(KNOWN_MAX_FAILS - 1));
    try t.expectEqual(KnownFailureAction.drop_and_forget, knownFailureAction(KNOWN_MAX_FAILS));
    try t.expectEqual(KnownFailureAction.drop_and_forget, knownFailureAction(255));
}

test "lan: lookupRemote distinguishes found / unlisted / unknown" {
    const a = t.allocator;
    var l = Table.init(a);
    defer l.deinit();

    // Unknown peer (and non-remote ids) → unknown: the proxy waits for
    // discovery to converge instead of failing instantly.
    try t.expect(l.lookupRemote("gemma@ghost") == .peer_unknown);
    try t.expect(l.lookupRemote("local-model") == .peer_unknown);

    const models = try a.alloc(PeerModel, 1);
    models[0] = .{ .id = try a.dupe(u8, "gemma"), .entry_json = try a.dupe(u8, "{}") };
    l.install("studio", .{ 127, 0, 0, 1 }, 1234, models);
    try t.expect(l.lookupRemote("gemma@studio") == .found);
    // The peer answered recently and does NOT offer this model — definitive,
    // fail fast (probes/typos must not burn the wait).
    try t.expect(l.lookupRemote("other@studio") == .model_unlisted);

    // A mid-boot empty install (peer reachable, models not served yet)
    // counts as unknown so the wait covers it too.
    l.install("booting", .{ 127, 0, 0, 1 }, 1235, &.{});
    try t.expect(l.lookupRemote("anything@booting") == .peer_unknown);
}

/// Duck-typed stand-in for server.Conn in tunnel tests.
const TestSink = struct {
    alloc: std.mem.Allocator,
    buf: std.ArrayList(u8) = .empty,

    pub fn writeAll(self: *TestSink, data: []const u8) !void {
        try self.buf.appendSlice(self.alloc, data);
    }
    pub fn peerClosed(self: *TestSink) bool {
        _ = self;
        return false;
    }
};

fn testListener(port_out: *u16) !net.Socket {
    return net.listenLoopback(port_out);
}

test "lan: tunnel forwards the rewritten request and pumps a chunked streaming response" {
    const a = t.allocator;
    var port: u16 = 0;
    const lst = try testListener(&port);
    defer net.close(lst);

    const FakePeer = struct {
        fn say(c: net.Socket, msg: []const u8) void {
            net.writeAll(c, msg) catch {};
        }
        fn run(listener: net.Socket) void {
            const c = net.acceptOne(listener);
            if (c == net.invalid_socket) return;
            defer net.close(c);
            var req: [4096]u8 = undefined;
            var got: usize = 0;
            while (got < req.len) {
                const n = net.read(c, req[got..]) catch return;
                if (n == 0) break;
                got += n;
                if (std.mem.indexOf(u8, req[0..got], "\"messages\":[]}") != null) break;
            }
            // The peer must see the BARE id and no trace of the @peer suffix.
            const rewritten = std.mem.indexOf(u8, req[0..got], "\"model\":\"bare-model\"") != null and
                std.mem.indexOf(u8, req[0..got], "@Studio") == null and
                std.mem.indexOf(u8, req[0..got], "POST /v1/chat/completions HTTP/1.1") != null;
            say(c, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n");
            say(c, if (rewritten) "data: ok\n\n" else "data: WRONG-REQUEST\n\n");
            platform.sleepMs(20); // force a second pump iteration
            say(c, "data: [DONE]\n\n");
        }
    };
    const th = try std.Thread.spawn(.{}, FakePeer.run, .{lst});
    defer th.join();

    const body = "{\"model\":\"bare-model@Studio\",\"messages\":[]}";
    const vs = std.mem.indexOf(u8, body, "bare-model@Studio").?;
    const rewritten = try policy.rewriteModelValue(a, body, body[vs .. vs + "bare-model@Studio".len], "bare-model");
    defer a.free(rewritten);

    var sink = TestSink{ .alloc = a };
    defer sink.buf.deinit(a);
    try tunnel(.{ .ip4 = .{ 127, 0, 0, 1 }, .port = port }, "POST", "/v1/chat/completions", rewritten, &sink);

    try t.expect(std.mem.indexOf(u8, sink.buf.items, "HTTP/1.1 200 OK") != null);
    try t.expect(std.mem.indexOf(u8, sink.buf.items, "data: ok") != null);
    try t.expect(std.mem.indexOf(u8, sink.buf.items, "data: [DONE]") != null);
    try t.expect(std.mem.indexOf(u8, sink.buf.items, "WRONG-REQUEST") == null);
}

test "lan: tunnel to a dead peer fails before writing anything to the client" {
    const a = t.allocator;
    var port: u16 = 0;
    const lst = try testListener(&port);
    net.close(lst); // port now refuses connections

    var sink = TestSink{ .alloc = a };
    defer sink.buf.deinit(a);
    try t.expectError(error.PeerUnreachable, tunnel(.{ .ip4 = .{ 127, 0, 0, 1 }, .port = port }, "POST", "/v1/messages", "{}", &sink));
    try t.expectEqual(@as(usize, 0), sink.buf.items.len);
}
