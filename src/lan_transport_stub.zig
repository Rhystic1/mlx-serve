//! No-op LAN transport for hosts without Bonjour (Windows, Linux).
//!
//! Mirrors `src/lan_bonjour.zig`'s public shape exactly so `src/lan.zig` can
//! swap the two at comptime and no caller branches on the host. Sharing is
//! simply never on here: `Lan.start` returns an inert instance, `sharing()` is
//! false, and every lookup reports the peer as unknown — which is the same
//! answer a real Bonjour instance gives before it has discovered anything, so
//! the call sites already handle it.
//!
//! This is a real functionality gap, not a hidden one: `--lan-share` and
//! `--lan-discover` do nothing off Apple. Discovery would need Avahi on Linux
//! and has no Windows equivalent; the streaming proxy itself is portable, but
//! without discovery there is nothing to proxy to.

const std = @import("std");
const log = @import("log.zig");

var token_seq: std.atomic.Value(u64) = .init(0);

/// Named (not anonymous) so the field declaration and the `init` call refer
/// to the SAME type -- two `struct {}` literals are two distinct types.
const PeerEntry = struct {};

pub const Remote = struct { ip4: [4]u8, port: u16 };

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
    /// Mirrors the Bonjour struct's public fields. `port`/`discover`/`name` are
    /// read directly by server.zig (not through a method), so they must exist
    /// with the same names and types even though nothing here acts on them.
    port: u16 = 0,
    discover: bool = false,
    name: []const u8 = "",
    /// Discovered peers. Always empty: nothing browses.
    peers: std.StringHashMap(PeerEntry) = undefined,
    /// Peers seen previously and retried by the browser loop. Also empty.
    known: std.StringHashMap(PeerEntry) = undefined,
    /// The `--lan-share` allowlist. Parsed and stored like the real one so the
    /// gate tests still construct a Lan; sharing is off regardless.
    share: ?@import("lan_policy.zig").SharedSet = null,
    /// This instance's identity token, echoed in the `X-MLX-LAN-Token` response
    /// header. Real on purpose even though nothing here discovers peers: the
    /// header is emitted unconditionally by `/v1/models`, and a fixed value
    /// would be a (small) cross-instance identity collision for anything that
    /// reads it. Bonjour's copy is generated the same way.
    token_hex: [16]u8 = undefined,

    pub const RemoteLookup = union(enum) { found: Remote, peer_unknown, model_unlisted };

    pub fn start(alloc: std.mem.Allocator, opts: Options) !*Lan {
        // Warn only when the user actually asked for it — an unconditional
        // line would fire on every boot, since the server constructs a Lan
        // regardless of whether sharing was requested.
        if (opts.share_spec != null or opts.discover) {
            log.warn("[lan] LAN sharing is unavailable on this platform (needs Bonjour); --lan-share/--lan-discover ignored\n", .{});
        }
        const l = try alloc.create(Lan);
        l.* = .{ .alloc = alloc, .port = opts.port, .discover = opts.discover, .name = opts.name orelse "" };
        l.peers = std.StringHashMap(PeerEntry).init(alloc);
        l.known = std.StringHashMap(PeerEntry).init(alloc);
        // Bonjour's copy uses arc4random_buf, which is POSIX-only;
        // std.crypto.random and std.time.nanoTimestamp no longer exist in this
        // Zig, and every remaining clock needs an `Io` this call does not have.
        //
        // So: the instance's own heap address (ASLR-varied) mixed with a
        // process-lifetime counter. That is adequate ONLY because this build
        // never discovers peers -- the token is echoed in a header and never
        // compared against another instance's. Do NOT copy this into the
        // Bonjour path, where self-ad detection depends on unguessability.
        const seq = token_seq.fetchAdd(1, .monotonic);
        const mixed = std.hash.Wyhash.hash(seq, std.mem.asBytes(&@intFromPtr(l)));
        _ = std.fmt.bufPrint(&l.token_hex, "{x:0>16}", .{mixed}) catch unreachable;
        return l;
    }

    pub fn shutdown(l: *Lan) void {
        l.peers.deinit();
        l.known.deinit();
        l.alloc.destroy(l);
    }

    pub fn sharing(_: *const Lan) bool {
        return false;
    }

    pub fn sharedAllows(_: *const Lan, _: []const u8) bool {
        return false;
    }

    pub fn lookupRemote(_: *Lan, _: []const u8) RemoteLookup {
        return .peer_unknown;
    }

    pub fn pokeDiscovery(_: *Lan) void {}

    pub fn remoteEntryFor(_: *Lan, _: std.mem.Allocator, _: []const u8) ?[]u8 {
        return null;
    }

    pub fn appendRemoteEntries(_: *Lan, _: std.mem.Allocator, _: *std.ArrayList(u8)) !void {}
};

/// Unreachable by construction: nothing is ever discovered, so no caller
/// obtains a `Remote` to tunnel to. Kept so the symbol exists.
pub fn tunnel(_: Remote, _: []const u8, _: []const u8, _: []const u8, _: anytype) error{PeerUnreachable}!void {
    return error.PeerUnreachable;
}
