//! Bonjour/dns_sd LAN transport — macOS only.
//!
//! Split out of lan.zig so the policy half stays portable. Everything here
//! binds to Apple's dns_sd (DNSServiceBrowse/Resolve/GetAddrInfo, 49 call
//! sites) or to raw BSD sockets; there is no Windows equivalent and the Linux
//! one would be Avahi, a different API. `src/lan_transport_stub.zig` is the
//! no-op twin that other hosts get instead.

const std = @import("std");
const log = @import("log.zig");

// Policy names used unqualified throughout the transport below. Aliased rather
// than re-spelled at each use so this file stayed a pure move.
const policy = @import("lan_policy.zig");
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

pub const Remote = struct { ip4: [4]u8, port: u16 };

const Peer = struct {
    display: []u8, // sanitized instance name — also the hash key
    ip4: [4]u8,
    port: u16,
    models: []PeerModel,

    fn deinit(p: *Peer, alloc: std.mem.Allocator) void {
        freePeerModels(alloc, p.models);
        alloc.free(p.display);
    }
};

const BrowseEvent = struct { name: [:0]u8, domain: [:0]u8 };

const Known = struct { domain: [:0]u8, fails: u8 = 0 };
/// Consecutive failed resolves before a service is forgotten (~4 min at the
/// 10 s refresh cadence). Until then the service keeps retrying quietly.
const KNOWN_MAX_FAILS: u8 = 24;
/// Consecutive failed resolves before an INSTALLED peer leaves the table
/// (~20-30 s at the refresh cadence). One transient dns_sd hiccup — a busy
/// mDNSResponder, an interface appearing/vanishing (VM or docker bridge), a
/// 3 s resolve timeout while the peer's GPU is pinned by a load — must not
/// evict a live peer: its cached ip4:port still tunnels, and eviction turns
/// the next chat into a "LAN peer for this model is offline" 404 (live
/// 2026-07-19: chats through a proxy alternated success/404 while the peer
/// stayed up and advertising the whole time). A genuinely-gone peer still
/// leaves within PEER_DROP_FAILS refreshes; the tunnel answers 502 honestly
/// if it is picked during the grace window.
const PEER_DROP_FAILS: u8 = 3;

const KnownFailureAction = enum { retain, drop_peer, drop_and_forget };

/// Pure policy: what a known service's consecutive-failure count (AFTER
/// incrementing for the current failure) does to the peer table + registry.
fn knownFailureAction(fails: u8) KnownFailureAction {
    if (fails >= KNOWN_MAX_FAILS) return .drop_and_forget;
    if (fails >= PEER_DROP_FAILS) return .drop_peer;
    return .retain;
}

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
    /// pthread mutex, not `std.Io.Mutex`: lookups run on conn threads and the
    /// browser thread, none of which should block through an Io handle for a
    /// micro critical section (same rationale as log.zig's sink_mutex).
    mu: std.c.pthread_mutex_t = .{},
    peers: std.StringHashMap(Peer),
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
            .peers = .init(alloc),
            .known = .init(alloc),
        };
        var rnd: [8]u8 = undefined;
        std.c.arc4random_buf(&rnd, rnd.len);
        _ = std.fmt.bufPrint(&l.token_hex, "{x:0>16}", .{std.mem.readInt(u64, &rnd, .big)}) catch unreachable;

        var host_buf: [std.posix.HOST_NAME_MAX]u8 = undefined;
        var raw_name: []const u8 = opts.name orelse std.posix.gethostname(&host_buf) catch "mac";
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
        var it = l.peers.valueIterator();
        while (it.next()) |p| p.deinit(l.alloc);
        l.peers.deinit();
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

    pub const RemoteLookup = union(enum) { found: Remote, peer_unknown, model_unlisted };

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
        const rid = splitRemoteId(id) orelse return .peer_unknown;
        _ = std.c.pthread_mutex_lock(&l.mu);
        defer _ = std.c.pthread_mutex_unlock(&l.mu);
        const p = l.peers.getPtr(rid.peer) orelse return .peer_unknown;
        for (p.models) |m|
            if (std.mem.eql(u8, m.id, rid.bare)) return .{ .found = .{ .ip4 = p.ip4, .port = p.port } };
        return if (p.models.len == 0) .peer_unknown else .model_unlisted;
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
        const rid = splitRemoteId(id) orelse return null;
        _ = std.c.pthread_mutex_lock(&l.mu);
        defer _ = std.c.pthread_mutex_unlock(&l.mu);
        const p = l.peers.getPtr(rid.peer) orelse return null;
        for (p.models) |m|
            if (std.mem.eql(u8, m.id, rid.bare)) return alloc.dupe(u8, m.entry_json) catch null;
        return null;
    }

    /// Append every discovered remote model's entry JSON to a /v1/models
    /// `data` array under construction (comma-managed by buffer length).
    pub fn appendRemoteEntries(l: *Lan, alloc: std.mem.Allocator, buf: *std.ArrayList(u8)) !void {
        _ = std.c.pthread_mutex_lock(&l.mu);
        defer _ = std.c.pthread_mutex_unlock(&l.mu);
        var it = l.peers.valueIterator();
        while (it.next()) |p| for (p.models) |m| {
            if (buf.items.len > 0) try buf.append(alloc, ',');
            try buf.appendSlice(alloc, m.entry_json);
        };
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
        _ = std.c.pthread_mutex_lock(&l.mu);
        defer _ = std.c.pthread_mutex_unlock(&l.mu);
        if (l.peers.fetchRemove(display)) |kv| {
            var p = kv.value;
            p.deinit(l.alloc);
        }
    }

    fn installPeer(l: *Lan, display: []const u8, ip4: [4]u8, port: u16, models: []PeerModel) void {
        const owned = l.alloc.dupe(u8, display) catch {
            freePeerModels(l.alloc, models);
            return;
        };
        _ = std.c.pthread_mutex_lock(&l.mu);
        defer _ = std.c.pthread_mutex_unlock(&l.mu);
        if (l.peers.fetchRemove(display)) |kv| {
            var old = kv.value;
            old.deinit(l.alloc);
        }
        const p = Peer{ .display = owned, .ip4 = ip4, .port = port, .models = models };
        l.peers.put(p.display, p) catch {
            var tmp = p;
            tmp.deinit(l.alloc);
        };
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
        const changed = blk: {
            _ = std.c.pthread_mutex_lock(&l.mu);
            defer _ = std.c.pthread_mutex_unlock(&l.mu);
            const p = l.peers.getPtr(display) orelse break :blk true;
            break :blk p.models.len != fetched.len;
        };
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
fn monoMs() i64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i64, @intCast(ts.sec)) * 1000 + @divTrunc(@as(i64, @intCast(ts.nsec)), 1_000_000);
}

/// One dns_sd ref pumped until its callback flips `done` or the deadline hits.
fn pumpUntil(ref: DNSServiceRef, done: *const bool, timeout_ms: i64) bool {
    const fd = DNSServiceRefSockFD(ref);
    if (fd < 0) return false;
    const deadline = monoMs() + timeout_ms;
    while (!done.*) {
        const remain = deadline - monoMs();
        if (remain <= 0) return false;
        var fds = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
        const ready = std.posix.poll(&fds, @intCast(@min(remain, 1000))) catch return false;
        if (ready > 0 and DNSServiceProcessResult(ref) != 0) return false;
    }
    return true;
}

// ── Outbound sockets (raw libc — self-contained, no std.Io runtime) ──

const fd_t = std.c.fd_t;

/// Non-blocking connect with a real deadline: a blocking connect to a
/// powered-off host can hang for the kernel's full SYN-retry budget (~75 s).
fn connectTimeout(ip4: [4]u8, port: u16, timeout_ms: i32) !fd_t {
    const fd = std.c.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
    if (fd < 0) return error.PeerUnreachable;
    errdefer _ = std.c.close(fd);
    const nonblock: c_int = @bitCast(std.c.O{ .NONBLOCK = true });
    const flags = std.c.fcntl(fd, std.c.F.GETFL);
    _ = std.c.fcntl(fd, std.c.F.SETFL, flags | nonblock);
    var sa: std.posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = @bitCast(ip4) };
    if (std.c.connect(fd, @ptrCast(&sa), @sizeOf(std.posix.sockaddr.in)) != 0) {
        const eno = std.c._errno().*;
        if (eno != @backingInt(std.c.E.INPROGRESS)) {
            log.debug("[lan] connect({d}.{d}.{d}.{d}:{d}) errno={d}\n", .{ ip4[0], ip4[1], ip4[2], ip4[3], port, eno });
            return error.PeerUnreachable;
        }
        var fds = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.OUT, .revents = 0 }};
        const ready = std.posix.poll(&fds, timeout_ms) catch return error.PeerUnreachable;
        if (ready == 0) {
            log.debug("[lan] connect({d}.{d}.{d}.{d}:{d}) poll timeout\n", .{ ip4[0], ip4[1], ip4[2], ip4[3], port });
            return error.PeerUnreachable;
        }
        var so_err: c_int = 0;
        var so_len: std.posix.socklen_t = @sizeOf(c_int);
        if (std.c.getsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.ERROR, &so_err, &so_len) != 0 or so_err != 0) {
            log.debug("[lan] connect({d}.{d}.{d}.{d}:{d}) SO_ERROR={d}\n", .{ ip4[0], ip4[1], ip4[2], ip4[3], port, so_err });
            return error.PeerUnreachable;
        }
    }
    _ = std.c.fcntl(fd, std.c.F.SETFL, flags); // back to blocking
    return fd;
}

fn writeAllFd(fd: fd_t, data: []const u8) !void {
    var off: usize = 0;
    while (off < data.len) {
        const n = std.c.write(fd, data.ptr + off, data.len - off);
        if (n <= 0) return error.PeerUnreachable;
        off += @intCast(n);
    }
}

/// Read once; 0 on EOF, error on failure/timeout.
fn readFd(fd: fd_t, buf: []u8) !usize {
    const n = std.c.read(fd, buf.ptr, buf.len);
    if (n < 0) return error.ReadFailed;
    return @intCast(n);
}

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

/// Discovery fetch: GET the peer's (shared-filtered) /v1/models.
/// `error.SelfFetch` when the response carries OUR OWN process token
/// (`X-MLX-LAN-Token`): a stale Bonjour record of a former self — same name
/// and port, different TXT token, so the resolve-time TXT check can't catch
/// it — resolves back to this very server, and the loopback-first fetch
/// would happily install our own models as a "peer" (live test_lan_share
/// self-mirror after the peer-restart section, 2026-07-21).
fn fetchPeerModels(alloc: std.mem.Allocator, ip4: [4]u8, port: u16, peer_display: []const u8, own_token: []const u8) ![]PeerModel {
    const fd = try connectTimeout(ip4, port, 3000);
    defer _ = std.c.close(fd);
    const tv = std.c.timeval{ .sec = 5, .usec = 0 };
    std.posix.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, std.mem.asBytes(&tv)) catch {};
    try writeAllFd(fd, "GET /v1/models HTTP/1.1\r\nHost: mlx-serve\r\nConnection: close\r\nX-MLX-LAN: 1\r\n\r\n");
    var resp: std.ArrayList(u8) = .empty;
    defer resp.deinit(alloc);
    var chunk: [16 * 1024]u8 = undefined;
    while (resp.items.len < 8 * 1024 * 1024) {
        const n = readFd(fd, &chunk) catch break;
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

/// Proxy one request to `remote` and pump the response back byte-for-byte.
/// `conn` needs `writeAll([]const u8) !void` + `peerClosed() bool` — the
/// server's `*Conn` fits, and the duck typing keeps this file
/// server-independent and the pump hermetically testable.
/// `error.PeerUnreachable` is returned BEFORE anything is written to `conn`
/// (the caller can still send a clean 502); any later failure just ends the
/// stream — the client sees a closed socket, the peer sees a disconnect and
/// cancels its slot.
pub fn tunnel(remote: Remote, method: []const u8, raw_path: []const u8, body: []const u8, conn: anytype) error{PeerUnreachable}!void {
    const fd = connectTimeout(remote.ip4, remote.port, 3000) catch return error.PeerUnreachable;
    defer _ = std.c.close(fd);
    var head_buf: [1024]u8 = undefined;
    const head = std.fmt.bufPrint(
        &head_buf,
        "{s} {s} HTTP/1.1\r\nHost: {d}.{d}.{d}.{d}:{d}\r\nContent-Type: application/json\r\nAccept: */*\r\nConnection: close\r\nX-MLX-LAN: 1\r\nContent-Length: {d}\r\n\r\n",
        .{ method, raw_path, remote.ip4[0], remote.ip4[1], remote.ip4[2], remote.ip4[3], remote.port, body.len },
    ) catch return error.PeerUnreachable;
    writeAllFd(fd, head) catch return error.PeerUnreachable;
    writeAllFd(fd, body) catch return error.PeerUnreachable;

    // Pump peer → client until peer EOF. The 1 s poll tick doubles as the
    // client-disconnect probe so an abandoned generation is torn down on the
    // peer too (its own disconnect-cancel machinery fires when we close).
    var buf: [16 * 1024]u8 = undefined;
    while (true) {
        var fds = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
        const ready = std.posix.poll(&fds, 1000) catch return;
        if (ready == 0) {
            if (conn.peerClosed()) return;
            continue;
        }
        const n = readFd(fd, &buf) catch return;
        if (n == 0) return;
        conn.writeAll(buf[0..n]) catch return;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
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
    var l = Lan{ .alloc = a, .port = 0, .discover = true, .peers = .init(a), .known = .init(a) };
    defer {
        var it = l.peers.valueIterator();
        while (it.next()) |p| p.deinit(a);
        l.peers.deinit();
        l.known.deinit();
    }

    // Unknown peer (and non-remote ids) → unknown: the proxy waits for
    // discovery to converge instead of failing instantly.
    try t.expect(l.lookupRemote("gemma@ghost") == .peer_unknown);
    try t.expect(l.lookupRemote("local-model") == .peer_unknown);

    const models = try a.alloc(PeerModel, 1);
    models[0] = .{ .id = try a.dupe(u8, "gemma"), .entry_json = try a.dupe(u8, "{}") };
    l.installPeer("studio", .{ 127, 0, 0, 1 }, 1234, models);
    try t.expect(l.lookupRemote("gemma@studio") == .found);
    // The peer answered recently and does NOT offer this model — definitive,
    // fail fast (probes/typos must not burn the wait).
    try t.expect(l.lookupRemote("other@studio") == .model_unlisted);

    // A mid-boot empty install (peer reachable, models not served yet)
    // counts as unknown so the wait covers it too.
    l.installPeer("booting", .{ 127, 0, 0, 1 }, 1235, &.{});
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

fn testListener(port_out: *u16) !fd_t {
    const lst = std.c.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
    if (lst < 0) return error.Sock;
    errdefer _ = std.c.close(lst);
    var sa: std.posix.sockaddr.in = .{ .port = 0, .addr = @bitCast([4]u8{ 127, 0, 0, 1 }) };
    if (std.c.bind(lst, @ptrCast(&sa), @sizeOf(std.posix.sockaddr.in)) != 0) return error.Sock;
    var sa_len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr.in);
    if (std.c.getsockname(lst, @ptrCast(&sa), &sa_len) != 0) return error.Sock;
    port_out.* = std.mem.bigToNative(u16, sa.port);
    if (std.c.listen(lst, 1) != 0) return error.Sock;
    return lst;
}

test "lan: tunnel forwards the rewritten request and pumps a chunked streaming response" {
    const a = t.allocator;
    var port: u16 = 0;
    const lst = try testListener(&port);
    defer _ = std.c.close(lst);

    const FakePeer = struct {
        fn say(c: fd_t, msg: []const u8) void {
            _ = std.c.write(c, msg.ptr, msg.len);
        }
        fn run(listener: fd_t) void {
            const c = std.c.accept(listener, null, null);
            if (c < 0) return;
            defer _ = std.c.close(c);
            var req: [4096]u8 = undefined;
            var got: usize = 0;
            while (got < req.len) {
                const n = readFd(c, req[got..]) catch return;
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
            const ts = std.c.timespec{ .sec = 0, .nsec = 20_000_000 };
            _ = std.c.nanosleep(&ts, null); // force a second pump iteration
            say(c, "data: [DONE]\n\n");
        }
    };
    const th = try std.Thread.spawn(.{}, FakePeer.run, .{lst});
    defer th.join();

    const body = "{\"model\":\"bare-model@Studio\",\"messages\":[]}";
    const vs = std.mem.indexOf(u8, body, "bare-model@Studio").?;
    const rewritten = try rewriteModelValue(a, body, body[vs .. vs + "bare-model@Studio".len], "bare-model");
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
    _ = std.c.close(lst); // port now refuses connections

    var sink = TestSink{ .alloc = a };
    defer sink.buf.deinit(a);
    try t.expectError(error.PeerUnreachable, tunnel(.{ .ip4 = .{ 127, 0, 0, 1 }, .port = port }, "POST", "/v1/messages", "{}", &sink));
    try t.expectEqual(@as(usize, 0), sink.buf.items.len);
}
