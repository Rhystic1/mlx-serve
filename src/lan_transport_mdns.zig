//! LAN transport for hosts without Bonjour (Windows, Linux), over the
//! hand-rolled responder in `lan_mdns.zig`.
//!
//! Mirrors `lan_bonjour.zig`'s public shape exactly so `lan.zig` swaps the two
//! at comptime and no caller branches on the host. Everything below the
//! discovery loop — the peer table, the model fetch, the proxy tunnel — is
//! `lan_peers.zig`, shared with the Apple path: only HOW a service is found
//! differs, and duplicating the rest would be duplicating the contract.
//!
//! The security gate is unchanged and lives above this (`routeClass` x
//! `SharedSet` in `lan_policy.zig`), because it always ran on every host.

const std = @import("std");
const log = @import("log.zig");
const platform = @import("platform.zig");
const policy = @import("lan_policy.zig");
const peers_mod = @import("lan_peers.zig");
const mdns = @import("lan_mdns.zig");

const SERVICE_TYPE = policy.SERVICE_TYPE;
const SharedSet = policy.SharedSet;
const sanitizeName = policy.sanitizeName;
const PeerModel = policy.PeerModel;

pub const Remote = peers_mod.Remote;
pub const tunnel = peers_mod.tunnel;

const KnownFailureAction = peers_mod.KnownFailureAction;
const knownFailureAction = peers_mod.knownFailureAction;

/// How often the browser re-queries. Matches the Bonjour path's refresh
/// cadence, which the two-tier failure counters are calibrated against
/// (PEER_DROP_FAILS x this ≈ the 20-30 s grace those constants document).
const REFRESH_MS: i64 = 10_000;

pub const Options = struct {
    port: u16,
    /// `--lan-share` value (`all` | csv of ids); null = sharing off.
    share_spec: ?[]const u8 = null,
    /// Advertised instance name; null → hostname (".local" stripped).
    name: ?[]const u8 = null,
    discover: bool = false,
};

/// One service this browser has ever seen, with its consecutive-failure count.
/// THE retry mechanism: a transient fetch hiccup at first sight must not lose
/// the peer until the next time it happens to re-announce.
const Known = struct { fails: u8 = 0 };

pub const Lan = struct {
    alloc: std.mem.Allocator,
    port: u16,
    discover: bool,
    share: ?SharedSet = null,
    name_buf: [96]u8 = undefined,
    name: []const u8 = "",
    /// Random per-process token in the TXT record — how a browser recognizes
    /// (and skips) its own advertisement.
    token_hex: [16]u8 = undefined,
    table: peers_mod.Table,
    thread: ?std.Thread = null,
    stop_flag: std.atomic.Value(bool) = .init(false),
    /// Set by `pokeDiscovery` (conn threads); consumed by the browser loop.
    refresh_asap: std.atomic.Value(bool) = .init(false),
    // Browser-thread only, no lock:
    known: std.StringHashMap(Known),
    responder: mdns.Responder = undefined,
    responder_up: bool = false,

    pub const RemoteLookup = peers_mod.RemoteLookup;

    pub fn start(alloc: std.mem.Allocator, opts: Options) !*Lan {
        const l = try alloc.create(Lan);
        errdefer alloc.destroy(l);
        l.* = .{
            .alloc = alloc,
            .port = opts.port,
            .discover = opts.discover,
            .table = .init(alloc),
            .known = .init(alloc),
        };
        var rnd: [8]u8 = undefined;
        platform.randomBytes(&rnd);
        _ = std.fmt.bufPrint(&l.token_hex, "{x:0>16}", .{std.mem.readInt(u64, &rnd, .big)}) catch unreachable;

        var host_buf: [std.posix.HOST_NAME_MAX]u8 = undefined;
        var raw_name: []const u8 = opts.name orelse std.posix.gethostname(&host_buf) catch "mlx-serve";
        if (std.mem.endsWith(u8, raw_name, ".local")) raw_name = raw_name[0 .. raw_name.len - ".local".len];
        l.name = sanitizeName(&l.name_buf, raw_name);

        if (opts.share_spec) |spec| {
            var set = try SharedSet.parse(alloc, spec);
            if (set.empty()) {
                set.deinit(alloc);
                log.warn("[lan] --lan-share matched no models; sharing disabled\n", .{});
            } else {
                l.share = set;
            }
        }
        if (l.share != null or l.discover)
            l.thread = try std.Thread.spawn(.{}, threadMain, .{l});
        return l;
    }

    pub fn shutdown(l: *Lan) void {
        l.stop_flag.store(true, .release);
        if (l.thread) |th| th.join();
        l.table.deinit();
        var kit = l.known.keyIterator();
        while (kit.next()) |k| l.alloc.free(k.*);
        l.known.deinit();
        if (l.share) |*s| s.deinit(l.alloc);
        const alloc = l.alloc;
        alloc.destroy(l);
    }

    pub fn sharing(l: *const Lan) bool {
        return l.share != null;
    }

    pub fn sharedAllows(l: *const Lan, id: []const u8) bool {
        return if (l.share) |s| s.allows(id) else false;
    }

    pub fn lookupRemote(l: *Lan, id: []const u8) RemoteLookup {
        return l.table.lookupRemote(id);
    }

    /// Ask the browser thread to re-query NOW instead of at the next tick —
    /// the proxy's convergence wait uses this so a rebooted peer is picked up
    /// within a poll cycle, not a refresh cycle.
    pub fn pokeDiscovery(l: *Lan) void {
        l.refresh_asap.store(true, .release);
    }

    pub fn remoteEntryFor(l: *Lan, alloc: std.mem.Allocator, id: []const u8) ?[]u8 {
        return l.table.remoteEntryFor(alloc, id);
    }

    pub fn appendRemoteEntries(l: *Lan, alloc: std.mem.Allocator, buf: *std.ArrayList(u8)) !void {
        return l.table.appendRemoteEntries(alloc, buf);
    }

    /// Attempt one discovered service: fetch its models and install it, with
    /// the failure bookkeeping that keeps a live peer through a transient
    /// hiccup. Mirrors `lan_bonjour.attemptKnown` — the grace policy is
    /// shared, only the discovery data differs.
    fn attempt(l: *Lan, svc: *const mdns.Service) void {
        const display = svc.name();
        const ip4 = svc.ip4 orelse return;

        // Our own advertisement: recognized by the token in its TXT, forgotten
        // outright rather than retried. This is the first of the two barriers
        // that make a proxy loop impossible by construction.
        if (policy.txtFind(svc.txt(), "t=")) |tok| {
            if (std.mem.eql(u8, tok, &l.token_hex)) {
                l.forget(display);
                return;
            }
        }

        const models = peers_mod.fetchPeerModels(l.alloc, ip4, svc.port, display, &l.token_hex) catch |e| {
            // The SECOND barrier: a stale record of a former self resolves
            // back to this very server, which the TXT check above cannot see
            // because the token in it is the OLD process's.
            if (e == error.SelfFetch) {
                l.forget(display);
                return;
            }
            l.recordFailure(display);
            return;
        };
        const changed = l.table.modelCountDiffers(display, models.len);
        const count = models.len;
        l.table.install(display, ip4, svc.port, models);
        l.noteKnown(display);
        l.resetFailures(display);
        if (changed)
            log.info("[lan] peer \"{s}\" at {d}.{d}.{d}.{d}:{d} shares {d} models\n", .{ display, ip4[0], ip4[1], ip4[2], ip4[3], svc.port, count });
    }

    fn recordFailure(l: *Lan, display: []const u8) void {
        l.noteKnown(display);
        const k = l.known.getPtr(display) orelse return;
        k.fails +|= 1;
        switch (knownFailureAction(k.fails)) {
            .retain => {},
            .drop_peer => l.table.remove(display),
            .drop_and_forget => {
                l.table.remove(display);
                l.forget(display);
            },
        }
    }

    fn resetFailures(l: *Lan, display: []const u8) void {
        if (l.known.getPtr(display)) |k| k.fails = 0;
    }

    /// Remember a service so a later sweep that does NOT see it can age it out.
    /// Without this, `known` only ever gained entries on failure and a peer
    /// that simply stopped answering was never retried and never dropped: the
    /// sweep iterates what it FOUND, and a dead peer is found by nobody
    /// (caught by tests/test_lan_share.sh's peer-offline section).
    fn noteKnown(l: *Lan, display: []const u8) void {
        const gop = l.known.getOrPut(display) catch return;
        if (!gop.found_existing) {
            gop.key_ptr.* = l.alloc.dupe(u8, display) catch {
                _ = l.known.remove(display);
                return;
            };
            gop.value_ptr.* = .{};
        }
    }

    fn forget(l: *Lan, display: []const u8) void {
        if (l.known.fetchRemove(display)) |kv| l.alloc.free(kv.key);
    }
};

/// Browser/advertiser thread. One responder serves both halves: the same
/// socket that answers a peer's query is the one our own query comes back on.
fn threadMain(l: *Lan) void {
    var name_buf: [96]u8 = undefined;
    var txt_buf: [64]u8 = undefined;
    const txt = policy.txtBuild(&txt_buf, &l.token_hex);

    l.responder.init(.{
        .advertise = null, // claimed below, once the name is settled
        .discover = l.discover,
        .service_type = SERVICE_TYPE,
        .host_name = l.name,
    }) catch |e| {
        log.warn("[lan] mDNS socket unavailable ({s}); LAN sharing is off this run\n", .{@errorName(e)});
        return;
    };
    l.responder_up = true;
    defer {
        l.responder.deinit();
        l.responder_up = false;
    }

    if (l.share != null) {
        // Settle the instance name BEFORE advertising: dns_sd renames a
        // colliding instance for free, and without this two servers on one
        // box would both answer for one name.
        l.name = l.responder.claimName(l.name, &l.token_hex, &name_buf);
        l.responder.setAdvertisement(.{ .instance = l.name, .port = l.port, .txt = txt });
        l.responder.announceBurst();
        const what = if (l.share.?.all) "all models" else "selected models";
        log.info("[lan] sharing {s} as \"{s}\" ({s} port {d})\n", .{ what, l.name, SERVICE_TYPE, l.port });
    }

    var next_refresh: i64 = 0; // sweep immediately on entry
    while (!l.stop_flag.load(.acquire)) {
        // Answers peers' queries and folds any responses in. Also the loop's
        // only sleep, so a stop is noticed within a second.
        l.responder.pump(1000);

        const now = mdns_now();
        const poked = l.refresh_asap.swap(false, .acq_rel);
        if (!l.discover or (now < next_refresh and !poked)) continue;
        next_refresh = now + REFRESH_MS;

        // A fresh sweep each time: the responder's table describes THIS sweep,
        // while `l.table` (with its failure counters) is the durable one. That
        // is what lets a peer that moved or renamed be seen at its new address
        // instead of being pinned to a stale entry.
        l.responder.clearServices();
        l.responder.sendQuery();
        l.responder.pump(750);

        for (l.responder.found()) |*svc| {
            if (svc.gone) {
                // An explicit goodbye is definitive; no grace period.
                l.table.remove(svc.name());
                l.forget(svc.name());
                continue;
            }
            if (!svc.resolved()) continue;
            l.attempt(svc);
        }

        // A peer that dies without a goodbye — power off, kill -9, a crash —
        // is not in this sweep's results at all, so nothing above touches it.
        // Age every known service the sweep did not resolve, which is what
        // eventually drops it (after the same grace the fetch failures get).
        ageUnseen(l);
    }
}

/// Count a missed sweep against every known service that did not answer it.
/// Names are copied out first: `recordFailure` can remove entries, and the
/// map's iterator does not survive that.
fn ageUnseen(l: *Lan) void {
    var unseen: [mdns.MAX_SERVICES][96]u8 = undefined;
    var lens: [mdns.MAX_SERVICES]usize = undefined;
    var n: usize = 0;
    var it = l.known.keyIterator();
    outer: while (it.next()) |k| {
        if (n == unseen.len) break;
        for (l.responder.found()) |*svc| {
            if (svc.resolved() and std.mem.eql(u8, svc.name(), k.*)) continue :outer;
        }
        lens[n] = @min(k.len, unseen[n].len);
        @memcpy(unseen[n][0..lens[n]], k.*[0..lens[n]]);
        n += 1;
    }
    for (0..n) |i| l.recordFailure(unseen[i][0..lens[i]]);
}

fn mdns_now() i64 {
    return @import("lan_net.zig").monoMs();
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

const t = std.testing;

test "lan mdns transport: an instance that shares nothing and discovers nothing runs no thread" {
    const l = try Lan.start(t.allocator, .{ .port = 1234 });
    defer l.shutdown();
    try t.expect(l.thread == null);
    try t.expect(!l.sharing());
    // The token is real regardless: it is echoed in X-MLX-LAN-Token on every
    // /v1/models response, and a fixed one would be a cross-instance identity
    // collision for anything that reads it.
    try t.expect(!std.mem.allEqual(u8, &l.token_hex, 0));
    try t.expect(l.lookupRemote("gemma@ghost") == .peer_unknown);
}

test "lan mdns transport: the failure ladder drops a peer, then forgets the service" {
    const a = t.allocator;
    const l = try Lan.start(a, .{ .port = 1234 });
    defer l.shutdown();

    const models = try a.alloc(PeerModel, 1);
    models[0] = .{ .id = try a.dupe(u8, "gemma"), .entry_json = try a.dupe(u8, "{}") };
    l.table.install("studio", .{ 127, 0, 0, 1 }, 1234, models);
    try t.expect(l.lookupRemote("gemma@studio") == .found);

    // Transient failures must NOT evict a live peer: its cached ip4:port still
    // tunnels, and eviction turns the next chat into a misleading 404.
    var i: u8 = 0;
    while (i < peers_mod.PEER_DROP_FAILS - 1) : (i += 1) l.recordFailure("studio");
    try t.expect(l.lookupRemote("gemma@studio") == .found);

    l.recordFailure("studio"); // now at PEER_DROP_FAILS
    try t.expect(l.lookupRemote("gemma@studio") == .peer_unknown);
    // Still KNOWN, so it keeps being retried rather than waiting for a
    // re-announcement that a never-departed service will not send.
    try t.expect(l.known.contains("studio"));

    while (l.known.getPtr("studio")) |k| {
        if (k.fails >= peers_mod.KNOWN_MAX_FAILS) break;
        l.recordFailure("studio");
    }
    try t.expect(!l.known.contains("studio"));
}

test "lan mdns transport: a success resets the failure count" {
    const l = try Lan.start(t.allocator, .{ .port = 1234 });
    defer l.shutdown();
    l.recordFailure("studio");
    l.recordFailure("studio");
    try t.expectEqual(@as(u8, 2), l.known.getPtr("studio").?.fails);
    l.resetFailures("studio");
    try t.expectEqual(@as(u8, 0), l.known.getPtr("studio").?.fails);
}

test "lan mdns transport: a peer that vanishes without a goodbye still ages out" {
    // Regression: the sweep iterates services it FOUND, and a peer that is
    // powered off (or kill -9'd) appears in no sweep at all — so nothing ever
    // counted a failure against it and its models stayed listed forever.
    // tests/test_lan_share.sh's peer-offline section is the live version.
    const a = t.allocator;
    const l = try Lan.start(a, .{ .port = 1234 });
    defer l.shutdown();
    // An empty responder table stands in for "the sweep found nothing".
    l.responder = .{ .service_type = SERVICE_TYPE };

    const models = try a.alloc(PeerModel, 1);
    models[0] = .{ .id = try a.dupe(u8, "gemma"), .entry_json = try a.dupe(u8, "{}") };
    l.table.install("studio", .{ 127, 0, 0, 1 }, 1234, models);
    l.noteKnown("studio");
    try t.expect(l.lookupRemote("gemma@studio") == .found);

    // Grace first: a single missed sweep must not evict a live peer.
    ageUnseen(l);
    try t.expect(l.lookupRemote("gemma@studio") == .found);

    var i: u8 = 1;
    while (i < peers_mod.PEER_DROP_FAILS) : (i += 1) ageUnseen(l);
    try t.expect(l.lookupRemote("gemma@studio") == .peer_unknown);
}
