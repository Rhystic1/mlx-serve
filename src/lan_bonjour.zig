//! Bonjour/dns_sd LAN DISCOVERY — Apple only.
//!
//! Split out of lan.zig so the policy half stays portable, and reduced again
//! when the hand-rolled transport landed: the peer table, the peer-model fetch
//! and the proxy tunnel were never Bonjour-specific and now live in
//! `lan_peers.zig`, shared with `lan_transport_mdns.zig`. What is left here is
//! what actually binds to Apple's dns_sd — browse, resolve, address lookup —
//! plus the retry bookkeeping shaped by it (a re-resolve keys on name+domain).
//!
//! Apple keeps dns_sd rather than adopting the hand-rolled responder:
//! mDNSResponder owns and defends `<host>.local` on this machine, and going
//! around it would mean a second responder competing with the system one
//! (windows-plan.md §3.5).

const std = @import("std");
const log = @import("log.zig");

// Policy names used unqualified throughout the transport below. Aliased rather
// than re-spelled at each use so this file stayed a pure move.
const policy = @import("lan_policy.zig");
/// The peer table, the peer-model fetch and the proxy tunnel: portable,
/// shared with the hand-rolled mDNS transport. Only DISCOVERY is dns_sd.
const peers_mod = @import("lan_peers.zig");
const net = @import("lan_net.zig");
const platform = @import("platform.zig");
const SERVICE_TYPE = policy.SERVICE_TYPE;
const RemoteId = policy.RemoteId;
const splitRemoteId = policy.splitRemoteId;
const RouteClass = policy.RouteClass;
const routeClass = policy.routeClass;
const SharedSet = policy.SharedSet;
const sanitizeName = policy.sanitizeName;
const rewriteModelValue = policy.rewriteModelValue;
const unescapeJsonSlashes = policy.unescapeJsonSlashes;
const txtBuild = policy.txtBuild;
const txtFind = policy.txtFind;
const PeerModel = policy.PeerModel;
const freePeerModels = policy.freePeerModels;
const parsePeerModels = policy.parsePeerModels;

const DNSServiceRef = ?*anyopaque;
const kDNSServiceFlagsMoreComing: u32 = 0x1;
const kDNSServiceFlagsAdd: u32 = 0x2;
const kDNSServiceProtocol_IPv4: u32 = 0x01;

const BrowseReply = *const fn (DNSServiceRef, u32, u32, i32, ?[*:0]const u8, ?[*:0]const u8, ?[*:0]const u8, ?*anyopaque) callconv(.c) void;
const ResolveReply = *const fn (DNSServiceRef, u32, u32, i32, ?[*:0]const u8, ?[*:0]const u8, u16, u16, ?[*]const u8, ?*anyopaque) callconv(.c) void;
const AddrInfoReply = *const fn (DNSServiceRef, u32, u32, i32, ?[*:0]const u8, ?*const std.posix.sockaddr, u32, ?*anyopaque) callconv(.c) void;

extern "c" fn DNSServiceRegister(ref: *DNSServiceRef, flags: u32, interface: u32, name: ?[*:0]const u8, regtype: [*:0]const u8, domain: ?[*:0]const u8, host: ?[*:0]const u8, port_be: u16, txt_len: u16, txt: ?*const anyopaque, cb: ?*const anyopaque, ctx: ?*anyopaque) i32;
extern "c" fn DNSServiceBrowse(ref: *DNSServiceRef, flags: u32, interface: u32, regtype: [*:0]const u8, domain: ?[*:0]const u8, cb: BrowseReply, ctx: ?*anyopaque) i32;
extern "c" fn DNSServiceResolve(ref: *DNSServiceRef, flags: u32, interface: u32, name: [*:0]const u8, regtype: [*:0]const u8, domain: [*:0]const u8, cb: ResolveReply, ctx: ?*anyopaque) i32;
extern "c" fn DNSServiceGetAddrInfo(ref: *DNSServiceRef, flags: u32, interface: u32, protocol: u32, hostname: [*:0]const u8, cb: AddrInfoReply, ctx: ?*anyopaque) i32;
extern "c" fn DNSServiceRefSockFD(ref: DNSServiceRef) i32;
extern "c" fn DNSServiceProcessResult(ref: DNSServiceRef) i32;
extern "c" fn DNSServiceRefDeallocate(ref: DNSServiceRef) void;

// ─────────────────────────────────────────────────────────────────────────────
// Runtime: advertiser + browser thread + peer table + proxy tunnel
// ─────────────────────────────────────────────────────────────────────────────

pub const Remote = peers_mod.Remote;

const BrowseEvent = struct { name: [:0]u8, domain: [:0]u8 };

const Known = struct { domain: [:0]u8, fails: u8 = 0 };

/// The failure grace policy and the peer table itself are transport-
/// independent; only the RETRY is Bonjour-shaped (a re-resolve by name +
/// domain), so only that stays here.
const KNOWN_MAX_FAILS = peers_mod.KNOWN_MAX_FAILS;
const PEER_DROP_FAILS = peers_mod.PEER_DROP_FAILS;
const KnownFailureAction = peers_mod.KnownFailureAction;
const knownFailureAction = peers_mod.knownFailureAction;

pub const Options = struct {
    port: u16,
    /// `--lan-share` value (`all` | csv of ids); null = sharing off.
    share_spec: ?[]const u8 = null,
    /// Advertised instance name; null → hostname (".local" stripped).
    name: ?[]const u8 = null,
    discover: bool = false,
};

pub const Lan = struct {
    alloc: std.mem.Allocator,
    port: u16,
    discover: bool,
    share: ?SharedSet = null,
    name_buf: [64]u8 = undefined,
    name: []const u8 = "",
    /// Random per-process token in the TXT record — how a browser recognizes
    /// (and skips) its own advertisement.
    token_hex: [16]u8 = undefined,
    reg_ref: DNSServiceRef = null,
    thread: ?std.Thread = null,
    stop_flag: std.atomic.Value(bool) = .init(false),
    table: peers_mod.Table,
    /// Set by `pokeDiscovery` (conn threads); consumed by the browser loop.
    refresh_asap: std.atomic.Value(bool) = .init(false),
    // Browser-thread only (events + refresh both run there — no lock):
    events: std.ArrayList(BrowseEvent) = .empty,
    /// Every service name browse has ever reported, with its consecutive
    /// resolve-failure count. THE retry mechanism: a transient resolve/fetch
    /// hiccup at ADD time must not lose the peer forever (browse won't
    /// re-announce a service that never left), so refresh re-attempts every
    /// known service — installed or not — and only a service that keeps
    /// failing (or turns out to be our own advertisement) is forgotten.
    known: std.StringHashMap(Known),

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
        std.c.arc4random_buf(&rnd, rnd.len);
        _ = std.fmt.bufPrint(&l.token_hex, "{x:0>16}", .{std.mem.readInt(u64, &rnd, .big)}) catch unreachable;

        var host_buf: [256]u8 = undefined;
        var raw_name: []const u8 = opts.name orelse platform.hostName(&host_buf) orelse "mac";
        if (std.mem.endsWith(u8, raw_name, ".local")) raw_name = raw_name[0 .. raw_name.len - ".local".len];
        l.name = sanitizeName(&l.name_buf, raw_name);

        if (opts.share_spec) |spec| {
            var set = try SharedSet.parse(alloc, spec);
            if (set.empty()) {
                set.deinit(alloc);
                log.warn("[lan] --lan-share matched no models; sharing disabled\n", .{});
            } else {
                l.share = set;
                l.startAdvertise();
            }
        }
        // Spawn whenever sharing was REQUESTED (not only when the initial
        // registration succeeded) — the browser thread's revive loop can
        // heal a registration that failed at boot or died with the daemon.
        if (l.share != null or l.discover)
            l.thread = try std.Thread.spawn(.{}, threadMain, .{l});
        return l;
    }

    pub fn shutdown(l: *Lan) void {
        l.stop_flag.store(true, .release);
        if (l.thread) |th| th.join();
        if (l.reg_ref != null) DNSServiceRefDeallocate(l.reg_ref); // unregisters
        l.table.deinit();
        for (l.events.items) |ev| {
            l.alloc.free(ev.name);
            l.alloc.free(ev.domain);
        }
        l.events.deinit(l.alloc);
        var kit = l.known.iterator();
        while (kit.next()) |e| {
            l.alloc.free(e.key_ptr.*);
            l.alloc.free(e.value_ptr.domain);
        }
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

    pub const RemoteLookup = peers_mod.RemoteLookup;

    /// Three-state lookup for a `<bare>@<peer>` id. `found` → tunnel it.
    /// `model_unlisted` → the peer answered recently and does NOT offer this
    /// model: definitive, fail fast. `peer_unknown` → the peer isn't in the
    /// table (yet): offline, mid-restart, or discovery still converging —
    /// the proxy WAITS briefly and retries instead of failing instantly
    /// (live: a chat fired while the peer Mac was redeploying — or right
    /// after a local restart — got an instant misleading 404). A peer
    /// installed with an EMPTY model list (mid-boot) counts as unknown so
    /// the wait covers it too.
    pub fn lookupRemote(l: *Lan, id: []const u8) RemoteLookup {
        return l.table.lookupRemote(id);
    }

    /// Ask the browser thread to re-attempt every known service NOW instead
    /// of at the next 10 s tick — the proxy's convergence wait uses this so
    /// a rebooted peer is picked up within a poll cycle, not a refresh cycle.
    pub fn pokeDiscovery(l: *Lan) void {
        l.refresh_asap.store(true, .release);
    }

    /// Owned copy of the /v1/models entry JSON for a remote id (the
    /// load-model no-op renders it so app flows work unchanged).
    pub fn remoteEntryFor(l: *Lan, alloc: std.mem.Allocator, id: []const u8) ?[]u8 {
        return l.table.remoteEntryFor(alloc, id);
    }

    /// Append every discovered remote model's entry JSON to a /v1/models
    /// `data` array under construction (comma-managed by buffer length).
    pub fn appendRemoteEntries(l: *Lan, alloc: std.mem.Allocator, buf: *std.ArrayList(u8)) !void {
        return l.table.appendRemoteEntries(alloc, buf);
    }

    fn startAdvertise(l: *Lan) void {
        var txt_buf: [32]u8 = undefined;
        const txt = txtBuild(&txt_buf, &l.token_hex);
        var name_z: [72]u8 = undefined;
        @memcpy(name_z[0..l.name.len], l.name);
        name_z[l.name.len] = 0;
        const err = DNSServiceRegister(&l.reg_ref, 0, 0, @ptrCast(&name_z), SERVICE_TYPE, null, null, std.mem.nativeToBig(u16, l.port), @intCast(txt.len), txt.ptr, null, null);
        if (err != 0) {
            log.warn("[lan] Bonjour registration failed ({d}); sharing not advertised\n", .{err});
            l.reg_ref = null;
        } else {
            const n = if (l.share.?.all) "all models" else "selected models";
            log.info("[lan] sharing {s} as \"{s}\" ({s} port {d})\n", .{ n, l.name, SERVICE_TYPE, l.port });
        }
    }

    fn removePeer(l: *Lan, display: []const u8) void {
        l.table.remove(display);
    }

    fn installPeer(l: *Lan, display: []const u8, ip4: [4]u8, port: u16, models: []PeerModel) void {
        l.table.install(display, ip4, port, models);
    }

    const Attempt = enum { installed, self_ad, failed };

    /// Re-resolve + re-fetch one service. Serves both browse ADDs and the
    /// periodic refresh; a REMOVE event also routes here. A failure NEVER
    /// removes the peer directly — `attemptKnown` owns the removal decision
    /// (grace of PEER_DROP_FAILS consecutive failures), so one transient
    /// dns_sd hiccup can't evict a live peer whose cached ip4:port still
    /// tunnels fine.
    fn resolveAndInstall(l: *Lan, service_name: [:0]const u8, domain: [:0]const u8) Attempt {
        var disp_buf: [64]u8 = undefined;
        const display = sanitizeName(&disp_buf, service_name);

        var res: ResolveOut = .{};
        var ref: DNSServiceRef = null;
        if (DNSServiceResolve(&ref, 0, 0, service_name.ptr, SERVICE_TYPE, domain.ptr, onResolve, &res) != 0) return .failed;
        const resolved = pumpUntil(ref, &res.done, 3000);
        DNSServiceRefDeallocate(ref);
        if (!resolved) {
            log.debug("[lan] resolve timed out for \"{s}\"\n", .{display});
            return .failed;
        }
        if (std.mem.eql(u8, res.token[0..res.token_len], &l.token_hex)) return .self_ad;

        var addr: AddrOut = .{};
        var aref: DNSServiceRef = null;
        res.host[res.host_len] = 0;
        if (DNSServiceGetAddrInfo(&aref, 0, 0, kDNSServiceProtocol_IPv4, @ptrCast(&res.host), onAddr, &addr) != 0) return .failed;
        const addressed = pumpUntil(aref, &addr.done, 3000);
        DNSServiceRefDeallocate(aref);
        if (!addressed or addr.count == 0) {
            log.debug("[lan] no IPv4 for \"{s}\" host \"{s}\"\n", .{ display, res.host[0..res.host_len] });
            return .failed;
        }

        // Try each address, loopback first; the first one that ACCEPTS is the
        // peer's address from now on (the tunnel reuses it).
        var order_buf: [4][4]u8 = undefined;
        const candidates = addr.ordered(&order_buf);
        var ip4: [4]u8 = candidates[0];
        var models: ?[]PeerModel = null;
        var reachable = false;
        for (candidates) |cand| {
            models = fetchPeerModels(l.alloc, cand, res.port, display, &l.token_hex) catch |err| switch (err) {
                error.PeerUnreachable => {
                    log.debug("[lan] \"{s}\" unreachable at {d}.{d}.{d}.{d}:{d}\n", .{ display, cand[0], cand[1], cand[2], cand[3], res.port });
                    continue;
                },
                // The fetch landed on OURSELVES (stale record of a former
                // self — resolve-time TXT check can't see it). Forget the
                // service; the real record, if any, re-registers via browse.
                error.SelfFetch => return .self_ad,
                else => blk: {
                    // Connected but no usable answer (booting, mid-restart):
                    // keep the peer listed empty; the next refresh heals it.
                    log.debug("[lan] model fetch from \"{s}\" failed: {s}\n", .{ display, @errorName(err) });
                    break :blk null;
                },
            };
            ip4 = cand;
            reachable = true;
            break;
        }
        if (!reachable) return .failed;
        const fetched = models orelse {
            l.installPeer(display, ip4, res.port, &.{});
            return .installed;
        };
        const changed = l.table.modelCountDiffers(display, fetched.len);
        const count = fetched.len;
        l.installPeer(display, ip4, res.port, fetched);
        if (changed)
            log.info("[lan] peer \"{s}\" at {d}.{d}.{d}.{d}:{d} shares {d} models\n", .{ display, ip4[0], ip4[1], ip4[2], ip4[3], res.port, count });
        return .installed;
    }

    /// Attempt one known service and keep its failure bookkeeping. Peer
    /// removal AND forgetting both happen here ONLY (self-ads forget
    /// immediately; an installed peer survives PEER_DROP_FAILS-1 transient
    /// failures; KNOWN_MAX_FAILS forgets the service) — a fresh browse ADD
    /// always re-registers.
    fn attemptKnown(l: *Lan, name: []const u8) void {
        const entry = l.known.getPtr(name) orelse return;
        const name_z = l.alloc.dupeSentinel(u8, name, 0) catch return;
        defer l.alloc.free(name_z);
        switch (l.resolveAndInstall(name_z, entry.domain)) {
            .installed => entry.fails = 0,
            .self_ad => l.forgetKnown(name),
            .failed => {
                entry.fails +|= 1;
                switch (knownFailureAction(entry.fails)) {
                    .retain => {},
                    .drop_peer => {
                        var disp_buf: [64]u8 = undefined;
                        l.removePeer(sanitizeName(&disp_buf, name));
                    },
                    .drop_and_forget => {
                        var disp_buf: [64]u8 = undefined;
                        l.removePeer(sanitizeName(&disp_buf, name));
                        l.forgetKnown(name);
                    },
                }
            },
        }
    }

    fn forgetKnown(l: *Lan, name: []const u8) void {
        if (l.known.fetchRemove(name)) |kv| {
            l.alloc.free(kv.key);
            l.alloc.free(kv.value.domain);
        }
    }

    fn refreshKnown(l: *Lan) void {
        var names: std.ArrayList([]u8) = .empty;
        defer {
            for (names.items) |n| l.alloc.free(n);
            names.deinit(l.alloc);
        }
        var it = l.known.keyIterator();
        while (it.next()) |k|
            names.append(l.alloc, l.alloc.dupe(u8, k.*) catch continue) catch break;
        for (names.items) |name| {
            if (l.stop_flag.load(.acquire)) return;
            l.attemptKnown(name);
        }
    }
};

const ResolveOut = struct {
    done: bool = false,
    port: u16 = 0,
    host: [256]u8 = undefined,
    host_len: usize = 0,
    token: [17]u8 = undefined,
    token_len: usize = 0,
};

/// Resolve callbacks fire once PER INTERFACE — only a SUCCESS may complete
/// the wait (an lo0 error arriving first must not abort a resolve that en0
/// would have answered a millisecond later; `pumpUntil`'s deadline covers
/// the all-interfaces-failed case).
fn onResolve(ref: DNSServiceRef, flags: u32, if_idx: u32, err: i32, fullname: ?[*:0]const u8, hosttarget: ?[*:0]const u8, port_be: u16, txt_len: u16, txt: ?[*]const u8, ctx: ?*anyopaque) callconv(.c) void {
    _ = ref;
    _ = flags;
    _ = if_idx;
    _ = fullname;
    const out: *ResolveOut = @ptrCast(@alignCast(ctx orelse return));
    if (err != 0 or hosttarget == null) return;
    const host = std.mem.span(hosttarget.?);
    if (host.len >= out.host.len) return;
    @memcpy(out.host[0..host.len], host);
    out.host_len = host.len;
    out.port = std.mem.bigToNative(u16, port_be);
    if (txt) |txt_ptr| {
        if (txtFind(txt_ptr[0..txt_len], "t=")) |token| {
            out.token_len = @min(token.len, out.token.len);
            @memcpy(out.token[0..out.token_len], token[0..out.token_len]);
        }
    }
    out.done = true;
}

/// A multi-homed host answers with one A record per interface — collect them
/// ALL (`MoreComing` clearing marks the batch end) so the fetch can try each.
/// Order matters downstream: loopback connects are exempt from macOS Local
/// Network privacy, so for a same-machine peer the lo0 record must win even
/// when the en0 record arrives first (live flake: en0-first resolution had
/// the SYN silently blackholed → 3 s poll timeout, peer never installed).
const AddrOut = struct {
    done: bool = false,
    count: usize = 0,
    ip4s: [4][4]u8 = undefined,

    fn add(out: *AddrOut, ip: [4]u8) void {
        for (out.ip4s[0..out.count]) |seen| if (std.mem.eql(u8, &seen, &ip)) return;
        if (out.count < out.ip4s.len) {
            out.ip4s[out.count] = ip;
            out.count += 1;
        }
    }

    /// Addresses in connect-attempt order: loopback first, then as resolved.
    fn ordered(out: *const AddrOut, buf: *[4][4]u8) []const [4]u8 {
        var n: usize = 0;
        for (out.ip4s[0..out.count]) |ip| if (ip[0] == 127) {
            buf[n] = ip;
            n += 1;
        };
        for (out.ip4s[0..out.count]) |ip| if (ip[0] != 127) {
            buf[n] = ip;
            n += 1;
        };
        return buf[0..n];
    }
};

/// Accumulates every IPv4 record; the batch completes when a callback
/// arrives with `MoreComing` clear AND at least one address landed (errors
/// alone never complete — `pumpUntil`'s deadline covers the all-failed case).
fn onAddr(ref: DNSServiceRef, flags: u32, if_idx: u32, err: i32, hostname: ?[*:0]const u8, address: ?*const std.posix.sockaddr, ttl: u32, ctx: ?*anyopaque) callconv(.c) void {
    _ = ref;
    _ = if_idx;
    _ = hostname;
    _ = ttl;
    const out: *AddrOut = @ptrCast(@alignCast(ctx orelse return));
    blk: {
        const sa = address orelse break :blk;
        if (err != 0 or sa.family != std.posix.AF.INET) break :blk;
        const sin: *const std.posix.sockaddr.in = @ptrCast(@alignCast(sa));
        out.add(@bitCast(sin.addr));
    }
    if (flags & kDNSServiceFlagsMoreComing == 0 and out.count > 0) out.done = true;
}

fn onBrowse(ref: DNSServiceRef, flags: u32, if_idx: u32, err: i32, name: ?[*:0]const u8, regtype: ?[*:0]const u8, domain: ?[*:0]const u8, ctx: ?*anyopaque) callconv(.c) void {
    _ = ref;
    _ = if_idx;
    _ = regtype;
    if (err != 0) return;
    const l: *Lan = @ptrCast(@alignCast(ctx orelse return));
    // ADD and REMOVE both route through resolveAndInstall (see its doc), so
    // the event only needs the service identity.
    log.debug("[lan] browse event: \"{s}\" (flags 0x{x})\n", .{ std.mem.span(name orelse return), flags });
    const n = l.alloc.dupeSentinel(u8, std.mem.span(name.?), 0) catch return;
    const d = l.alloc.dupeSentinel(u8, std.mem.span(domain orelse "local."), 0) catch {
        l.alloc.free(n);
        return;
    };
    l.events.append(l.alloc, .{ .name = n, .domain = d }) catch {
        l.alloc.free(n);
        l.alloc.free(d);
    };
}

/// How often a dead dns_sd ref (browse or advertise) is re-created. Also the
/// retry cadence while mDNSResponder itself is down.
const REVIVE_INTERVAL_MS: i64 = 5_000;

fn threadMain(l: *Lan) void {
    var browse_ref: DNSServiceRef = null;
    defer if (browse_ref != null) DNSServiceRefDeallocate(browse_ref);

    var last_refresh = monoMs();
    // Dead refs are REVIVED, never left null forever: an mDNSResponder
    // restart (macOS update, daemon crash) or a sleep/wake cycle can
    // invalidate every dns_sd connection at once, and a permanently-dead
    // browse leaves this server blind to peers that are up and advertising
    // — every remote chat then 404s "peer offline" until a manual restart.
    // `revive_at = 0` makes the first loop iteration do the initial starts.
    var revive_at: i64 = 0;
    while (!l.stop_flag.load(.acquire)) {
        const now_ms = monoMs();
        if (now_ms >= revive_at) {
            revive_at = now_ms + REVIVE_INTERVAL_MS;
            if (l.discover and browse_ref == null) {
                if (DNSServiceBrowse(&browse_ref, 0, 0, SERVICE_TYPE, null, onBrowse, l) != 0) {
                    log.warn("[lan] Bonjour browse failed to start; retrying in {d} s\n", .{@divTrunc(REVIVE_INTERVAL_MS, 1000)});
                    browse_ref = null;
                } else {
                    log.info("[lan] discovering peers ({s})\n", .{SERVICE_TYPE});
                }
            }
            // Sharing was requested but the registration is gone (failed at
            // boot, or the daemon dropped it) — re-advertise, or peers see
            // this host vanish while it keeps serving.
            if (l.share != null and l.reg_ref == null) l.startAdvertise();
        }
        var fds: [2]std.posix.pollfd = undefined;
        var refs: [2]DNSServiceRef = undefined;
        var n: usize = 0;
        for ([_]DNSServiceRef{ l.reg_ref, browse_ref }) |r| {
            if (r != null) {
                fds[n] = .{ .fd = DNSServiceRefSockFD(r), .events = std.posix.POLL.IN, .revents = 0 };
                refs[n] = r;
                n += 1;
            }
        }
        if (n == 0) {
            // std.time.sleep was removed in Zig 0.16; this thread has no Io.
            const ts = std.c.timespec{ .sec = 0, .nsec = 500_000_000 };
            _ = std.c.nanosleep(&ts, null);
            continue;
        }
        const ready = std.posix.poll(fds[0..n], 1000) catch break;
        if (ready > 0) for (fds[0..n], refs[0..n]) |fd, r| {
            if (fd.revents == 0) continue;
            // IN with a clean ProcessResult = normal traffic. Anything else
            // (ProcessResult error, or HUP/ERR/NVAL with no data) means the
            // daemon dropped this connection — tear the ref down so the
            // revive above re-creates it, instead of hot-spinning on a dead
            // fd or silently losing discovery/advertising.
            const alive = fd.revents & std.posix.POLL.IN != 0 and DNSServiceProcessResult(r) == 0;
            if (alive) continue;
            if (r == browse_ref) {
                log.warn("[lan] dns_sd browse connection lost; will re-browse\n", .{});
                DNSServiceRefDeallocate(browse_ref);
                browse_ref = null;
            } else if (r == l.reg_ref) {
                log.warn("[lan] dns_sd advertise connection lost; will re-register\n", .{});
                DNSServiceRefDeallocate(l.reg_ref);
                l.reg_ref = null;
            }
        };
        // Drain browse events (appended by onBrowse inside ProcessResult —
        // same thread, so plain iteration is safe even though the resolves
        // below block for seconds). ADD and REMOVE both upsert `known` and
        // attempt immediately; resolvability decides what survives.
        while (l.events.pop()) |ev| {
            defer l.alloc.free(ev.name);
            if (!l.known.contains(ev.name)) blk: {
                const key = l.alloc.dupe(u8, ev.name) catch break :blk;
                l.known.put(key, .{ .domain = ev.domain }) catch {
                    l.alloc.free(key);
                    break :blk;
                };
                l.attemptKnown(ev.name);
                continue;
            }
            l.alloc.free(ev.domain);
            l.attemptKnown(ev.name);
        }
        const now = monoMs();
        const poked = l.refresh_asap.swap(false, .acq_rel);
        if (l.discover and (poked or now - last_refresh > 10_000)) {
            last_refresh = now;
            l.refreshKnown();
        }
    }
}

/// Monotonic milliseconds without an `Io` handle (this thread has none —
/// same rationale as log.zig's raw-libc sink).
const monoMs = net.monoMs;

/// One dns_sd ref pumped until its callback flips `done` or the deadline hits.
fn pumpUntil(ref: DNSServiceRef, done: *const bool, timeout_ms: i64) bool {
    const fd = DNSServiceRefSockFD(ref);
    if (fd < 0) return false;
    const deadline = monoMs() + timeout_ms;
    while (!done.*) {
        const remain = deadline - monoMs();
        if (remain <= 0) return false;
        if (net.waitReadable(fd, @intCast(@min(remain, 1000))) and DNSServiceProcessResult(ref) != 0) return false;
    }
    return true;
}

/// Re-exported so `lan.zig`'s facade keeps its flat API: the tunnel and the
/// header helper are portable and live in `lan_peers.zig` now, along with the
/// peer-model fetch and the socket helpers all three used.
pub const headerValueCI = peers_mod.headerValueCI;
pub const tunnel = peers_mod.tunnel;
const fetchPeerModels = peers_mod.fetchPeerModels;
