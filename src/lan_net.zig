//! The small socket layer under LAN sharing: UDP multicast for mDNS, and
//! outbound TCP for the model fetch and the proxy tunnel.
//!
//! Raw libc / ws2_32 rather than `std.Io`: these run on the browser thread and
//! on connection threads, none of which carries an `Io` handle — the same
//! constraint that shapes `platform.zig`.
//!
//! The POSIX arm is what `lan_bonjour.zig` has always used, moved here
//! unchanged so macOS keeps the exact code it shipped with; the Windows arm is
//! new. Linux exercises the POSIX arm, which is the one macOS depends on.

const std = @import("std");
const builtin = @import("builtin");
const platform = @import("platform.zig");
const log = @import("log.zig");

const is_windows = builtin.os.tag == .windows;

/// A SOCKET on Windows is an unsigned handle, not a file descriptor; keeping
/// our own alias avoids casting through `std.posix.fd_t` (a pointer there).
pub const Socket = if (is_windows) usize else i32;
pub const invalid_socket: Socket = if (is_windows) ~@as(usize, 0) else -1;

pub const Error = error{ SocketFailed, PeerUnreachable, ReadFailed };

// ── Windows entry points (ws2_32 is already linked by platform.zig's externs) ─

extern "ws2_32" fn socket(af: c_int, kind: c_int, proto: c_int) callconv(.winapi) usize;
extern "ws2_32" fn bind(s: usize, addr: *const anyopaque, len: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn connect(s: usize, addr: *const anyopaque, len: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn setsockopt(s: usize, level: c_int, opt: c_int, val: [*]const u8, len: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn getsockopt(s: usize, level: c_int, opt: c_int, val: [*]u8, len: *c_int) callconv(.winapi) c_int;
extern "ws2_32" fn sendto(s: usize, buf: [*]const u8, len: c_int, flags: c_int, to: *const anyopaque, tolen: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn recvfrom(s: usize, buf: [*]u8, len: c_int, flags: c_int, from: ?*anyopaque, fromlen: ?*c_int) callconv(.winapi) c_int;
extern "ws2_32" fn send(s: usize, buf: [*]const u8, len: c_int, flags: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn recv(s: usize, buf: [*]u8, len: c_int, flags: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn closesocket(s: usize) callconv(.winapi) c_int;
extern "ws2_32" fn ioctlsocket(s: usize, cmd: c_long, arg: *c_ulong) callconv(.winapi) c_int;
extern "ws2_32" fn WSAPoll(fds: [*]WsaPollFd, n: u32, timeout: i32) callconv(.winapi) i32;
extern "ws2_32" fn gethostname(name: [*]u8, len: c_int) callconv(.winapi) c_int;

const WsaPollFd = extern struct { fd: usize, events: i16, revents: i16 };
const FIONBIO: c_long = @bitCast(@as(u32, 0x8004667e));
const WSAEWOULDBLOCK: c_int = 10035;

// ── Constants that differ per host ───────────────────────────────────────────

const AF_INET: c_int = 2;
const SOCK_DGRAM: c_int = if (builtin.os.tag == .linux) 2 else 2;
const SOCK_STREAM: c_int = 1;
const IPPROTO_IP: c_int = 0;

// Numeric rather than via `std.c`: these constants are a plain integer on
// some targets and an enum on others, so one spelling cannot serve all three
// hosts. Linux renumbered the IP-level options; BSD did not, and Windows
// copied BSD.
const SOL_SOCKET: i32 = if (is_windows) 0xffff else if (builtin.os.tag == .linux) 1 else 0xffff;
const SO_REUSEADDR: u32 = if (is_windows) 0x0004 else if (builtin.os.tag == .linux) 2 else 0x0004;
const SO_REUSEPORT: ?u32 = if (is_windows) null else if (builtin.os.tag == .linux) 15 else 0x0200;
const SO_ERROR: u32 = if (is_windows) 0x1007 else if (builtin.os.tag == .linux) 4 else 0x1007;

const IP_MULTICAST_IF: u32 = if (builtin.os.tag == .linux) 32 else 9;
const IP_MULTICAST_TTL: u32 = if (builtin.os.tag == .linux) 33 else 10;
const IP_MULTICAST_LOOP: u32 = if (builtin.os.tag == .linux) 34 else 11;
const IP_ADD_MEMBERSHIP: u32 = if (builtin.os.tag == .linux) 35 else 12;

const IpMreq = extern struct { multiaddr: [4]u8, interface: [4]u8 };

fn sockaddrIn(ip4: [4]u8, port: u16) std.posix.sockaddr.in {
    return .{ .port = std.mem.nativeToBig(u16, port), .addr = @bitCast(ip4) };
}

// ── Lifetime ────────────────────────────────────────────────────────────────

pub fn close(s: Socket) void {
    if (s == invalid_socket) return;
    if (is_windows) _ = closesocket(s) else _ = std.c.close(s);
}

pub fn monoMs() i64 {
    if (is_windows) return @intCast(std.os.windows.kernel32.GetTickCount64());
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i64, @intCast(ts.sec)) * 1000 + @divTrunc(@as(i64, @intCast(ts.nsec)), 1_000_000);
}

fn setOpt(s: Socket, level: i32, opt: u32, bytes: []const u8) void {
    if (is_windows) {
        _ = setsockopt(s, @intCast(level), @intCast(opt), bytes.ptr, @intCast(bytes.len));
    } else {
        _ = std.c.setsockopt(s, level, opt, bytes.ptr, @intCast(bytes.len));
    }
}

/// Readable within `timeout_ms`? Used by both the mDNS pump and the TCP
/// connect path.
pub fn waitReadable(s: Socket, timeout_ms: i32) bool {
    return poll(s, timeout_ms, true);
}

fn poll(s: Socket, timeout_ms: i32, for_read: bool) bool {
    if (is_windows) {
        var fds = [_]WsaPollFd{.{ .fd = s, .events = if (for_read) 0x0100 else 0x0010, .revents = 0 }};
        return WSAPoll(&fds, 1, timeout_ms) > 0;
    }
    var fds = [_]std.posix.pollfd{.{
        .fd = s,
        .events = if (for_read) std.posix.POLL.IN else std.posix.POLL.OUT,
        .revents = 0,
    }};
    const ready = std.posix.poll(&fds, timeout_ms) catch return false;
    return ready > 0;
}

// ── UDP multicast (mDNS) ────────────────────────────────────────────────────

/// Bind UDP `port` on every interface, ready to join a multicast group.
///
/// SO_REUSEADDR is mandatory: 5353 is a shared port by design (RFC 6762 §15.1)
/// and something else may already hold it. BSD additionally wants SO_REUSEPORT
/// for a second binder, and setting it where it exists costs nothing.
pub fn udpBindShared(port: u16) Error!Socket {
    if (is_windows) platform.ensureWinsock();
    const s: Socket = if (is_windows)
        socket(AF_INET, SOCK_DGRAM, 0)
    else
        std.c.socket(AF_INET, SOCK_DGRAM, 0);
    if (s == invalid_socket or (!is_windows and s < 0)) return error.SocketFailed;
    errdefer close(s);

    const on: c_int = 1;
    setOpt(s, SOL_SOCKET, SO_REUSEADDR, std.mem.asBytes(&on));
    if (comptime SO_REUSEPORT) |rp| setOpt(s, SOL_SOCKET, rp, std.mem.asBytes(&on));

    var sa = sockaddrIn(.{ 0, 0, 0, 0 }, port);
    const rc = if (is_windows)
        bind(s, &sa, @sizeOf(std.posix.sockaddr.in))
    else
        std.c.bind(s, @ptrCast(&sa), @sizeOf(std.posix.sockaddr.in));
    if (rc != 0) return error.SocketFailed;
    return s;
}

/// Join `group` on ONE interface. Called per interface on purpose: Windows
/// joins only a single (kernel-chosen) interface for INADDR_ANY, and on a box
/// with Hyper-V / WSL / VPN adapters that is frequently the wrong one.
pub fn joinMulticast(s: Socket, group: [4]u8, iface: [4]u8) void {
    const mreq = IpMreq{ .multiaddr = group, .interface = iface };
    setOpt(s, IPPROTO_IP, IP_ADD_MEMBERSHIP, std.mem.asBytes(&mreq));
}

pub fn setMulticastInterface(s: Socket, iface: [4]u8) void {
    setOpt(s, IPPROTO_IP, IP_MULTICAST_IF, &iface);
}

pub fn setMulticastTtl(s: Socket, ttl: u8) void {
    // Link-local by definition, but the option's width is not: Windows and
    // BSD read an int here, and a 1-byte write leaves the rest undefined.
    const v: c_int = ttl;
    setOpt(s, IPPROTO_IP, IP_MULTICAST_TTL, std.mem.asBytes(&v));
}

/// Loopback delivery of our own multicast. On by default in the kernel, set
/// explicitly because two instances on ONE box discovering each other is a
/// supported case (and the only one a single-host test can cover).
pub fn setMulticastLoop(s: Socket, on: bool) void {
    const v: c_int = if (on) 1 else 0;
    setOpt(s, IPPROTO_IP, IP_MULTICAST_LOOP, std.mem.asBytes(&v));
}

pub fn sendTo(s: Socket, ip4: [4]u8, port: u16, data: []const u8) void {
    var sa = sockaddrIn(ip4, port);
    if (is_windows) {
        _ = sendto(s, data.ptr, @intCast(data.len), 0, &sa, @sizeOf(std.posix.sockaddr.in));
    } else {
        _ = std.c.sendto(s, data.ptr, data.len, 0, @ptrCast(&sa), @sizeOf(std.posix.sockaddr.in));
    }
}

pub const Datagram = struct { len: usize, from_ip4: [4]u8 };

pub fn recvFrom(s: Socket, buf: []u8) ?Datagram {
    var sa: std.posix.sockaddr.in = undefined;
    var sl: c_int = @sizeOf(std.posix.sockaddr.in);
    const n = if (is_windows)
        recvfrom(s, buf.ptr, @intCast(buf.len), 0, &sa, &sl)
    else blk: {
        var sl_p: std.posix.socklen_t = @sizeOf(std.posix.sockaddr.in);
        break :blk std.c.recvfrom(s, buf.ptr, buf.len, 0, @ptrCast(&sa), &sl_p);
    };
    if (n <= 0) return null;
    const addr: [4]u8 = @bitCast(sa.addr);
    return .{ .len = @intCast(n), .from_ip4 = addr };
}

/// Every IPv4 address this host holds, for joining the group on each. Best
/// effort: a host whose enumeration comes back empty still works through the
/// unspecified-interface join the caller also performs.
pub fn localIp4Addresses(out: [][4]u8) [][4]u8 {
    var n: usize = 0;
    if (is_windows) {
        // gethostname + the resolver, rather than iphlpapi: it needs no extra
        // link dependency and returns the host's own addresses, which is all
        // the join loop wants.
        var host: [256]u8 = undefined;
        if (gethostname(&host, host.len) != 0) return out[0..0];
        const name = std.mem.sliceTo(&host, 0);
        var buf: [64]u8 = undefined;
        const zname = std.fmt.bufPrintZ(&buf, "{s}", .{name}) catch return out[0..0];
        const list = std.net.getAddressList(std.heap.page_allocator, zname, 0) catch return out[0..0];
        defer list.deinit();
        for (list.addrs) |a| {
            if (a.any.family != AF_INET or n == out.len) continue;
            out[n] = @bitCast(a.in.sa.addr);
            n += 1;
        }
        return out[0..n];
    }
    var ifap: ?*Ifaddrs = null;
    if (getifaddrs(&ifap) != 0) return out[0..0];
    defer freeifaddrs(ifap);
    var it = ifap;
    while (it) |ia| : (it = ia.next) {
        const addr = ia.addr orelse continue;
        if (addr.family != AF_INET) continue;
        if (n == out.len) break;
        const sin: *const std.posix.sockaddr.in = @ptrCast(@alignCast(addr));
        out[n] = @bitCast(sin.addr);
        n += 1;
    }
    return out[0..n];
}

/// `struct ifaddrs` — same layout on Linux and Darwin. Not in this Zig's
/// `std.c`, so declared here.
const Ifaddrs = extern struct {
    next: ?*Ifaddrs,
    name: [*:0]u8,
    flags: c_uint,
    addr: ?*std.posix.sockaddr,
    netmask: ?*std.posix.sockaddr,
    dstaddr: ?*std.posix.sockaddr,
    data: ?*anyopaque,
};
extern "c" fn getifaddrs(out: *?*Ifaddrs) c_int;
extern "c" fn freeifaddrs(p: ?*Ifaddrs) void;

// ── Outbound TCP (peer model fetch + proxy tunnel) ──────────────────────────

/// Non-blocking connect with a real deadline: a blocking connect to a
/// powered-off host can hang for the kernel's full SYN-retry budget (~75 s).
pub fn connectTimeout(ip4: [4]u8, port: u16, timeout_ms: i32) Error!Socket {
    if (is_windows) platform.ensureWinsock();
    const s: Socket = if (is_windows)
        socket(AF_INET, SOCK_STREAM, 0)
    else
        std.c.socket(AF_INET, SOCK_STREAM, 0);
    if (s == invalid_socket or (!is_windows and s < 0)) return error.PeerUnreachable;
    errdefer close(s);

    var sa = sockaddrIn(ip4, port);
    var in_progress = false;
    if (is_windows) {
        var nb: c_ulong = 1;
        _ = ioctlsocket(s, FIONBIO, &nb);
        if (connect(s, &sa, @sizeOf(std.posix.sockaddr.in)) != 0) {
            if (WSAGetLastError() != WSAEWOULDBLOCK) return error.PeerUnreachable;
            in_progress = true;
        }
    } else {
        const nonblock: c_int = @bitCast(std.c.O{ .NONBLOCK = true });
        const flags = std.c.fcntl(s, std.c.F.GETFL);
        _ = std.c.fcntl(s, std.c.F.SETFL, flags | nonblock);
        if (std.c.connect(s, @ptrCast(&sa), @sizeOf(std.posix.sockaddr.in)) != 0) {
            const eno = std.c._errno().*;
            if (eno != @intFromEnum(std.c.E.INPROGRESS)) {
                log.debug("[lan] connect({d}.{d}.{d}.{d}:{d}) errno={d}\n", .{ ip4[0], ip4[1], ip4[2], ip4[3], port, eno });
                return error.PeerUnreachable;
            }
            in_progress = true;
        }
        if (!in_progress) _ = std.c.fcntl(s, std.c.F.SETFL, flags);
    }
    if (in_progress) {
        if (!poll(s, timeout_ms, false)) {
            log.debug("[lan] connect({d}.{d}.{d}.{d}:{d}) poll timeout\n", .{ ip4[0], ip4[1], ip4[2], ip4[3], port });
            return error.PeerUnreachable;
        }
        var so_err: c_int = 0;
        var so_len: c_int = @sizeOf(c_int);
        const got = if (is_windows)
            getsockopt(s, @intCast(SOL_SOCKET), @intCast(SO_ERROR), @ptrCast(&so_err), &so_len)
        else blk: {
            var l: std.posix.socklen_t = @sizeOf(c_int);
            break :blk std.c.getsockopt(s, SOL_SOCKET, SO_ERROR, &so_err, &l);
        };
        if (got != 0 or so_err != 0) {
            log.debug("[lan] connect({d}.{d}.{d}.{d}:{d}) SO_ERROR={d}\n", .{ ip4[0], ip4[1], ip4[2], ip4[3], port, so_err });
            return error.PeerUnreachable;
        }
        if (is_windows) {
            var nb: c_ulong = 0;
            _ = ioctlsocket(s, FIONBIO, &nb);
        } else {
            const flags = std.c.fcntl(s, std.c.F.GETFL);
            const nonblock: c_int = @bitCast(std.c.O{ .NONBLOCK = true });
            _ = std.c.fcntl(s, std.c.F.SETFL, flags & ~nonblock);
        }
    }
    return s;
}

pub fn writeAll(s: Socket, data: []const u8) Error!void {
    var off: usize = 0;
    while (off < data.len) {
        const n = if (is_windows)
            send(s, data.ptr + off, @intCast(data.len - off), 0)
        else
            std.c.write(s, data.ptr + off, data.len - off);
        if (n <= 0) return error.PeerUnreachable;
        off += @intCast(n);
    }
}

/// Read once; 0 on EOF, error on failure.
pub fn read(s: Socket, buf: []u8) Error!usize {
    const n = if (is_windows)
        recv(s, buf.ptr, @intCast(buf.len), 0)
    else
        std.c.read(s, buf.ptr, buf.len);
    if (n < 0) return error.ReadFailed;
    return @intCast(n);
}

extern "ws2_32" fn WSAGetLastError() callconv(.winapi) c_int;
