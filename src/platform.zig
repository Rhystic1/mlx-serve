//! Host primitives that have no portable form in this Zig's standard library.
//!
//! Everything here has exactly one job: give the rest of the server ONE spelling
//! for something the three supported hosts express differently. It is the place
//! for "POSIX has it, Windows doesn't" (and vice versa), not a general utility
//! bin — anything with a real `std.Io` equivalent belongs on that instead.
//!
//! Why these two live here rather than at their call sites: both are reached
//! from threads that carry no `Io` handle (the signal path is an actual signal
//! handler; the gauge sampler is a bare `std.Thread`), so the `std.Io` versions
//! are unavailable by construction.

const std = @import("std");
const builtin = @import("builtin");

// ── Sleep ──────────────────────────────────────────────────────────────────

extern "kernel32" fn Sleep(dwMilliseconds: u32) callconv(.winapi) void;

/// Block the calling thread for at least `ms` milliseconds.
///
/// `std.time.sleep` was removed in Zig 0.16 and the replacement (`std.Io.sleep`)
/// needs an `Io`. Callers here are plain `std.Thread`s that have none.
pub fn sleepMs(ms: u32) void {
    switch (builtin.os.tag) {
        .windows => Sleep(ms),
        else => {
            const ts = std.c.timespec{
                .sec = @intCast(ms / 1000),
                .nsec = @intCast((ms % 1000) * 1_000_000),
            };
            _ = std.c.nanosleep(&ts, null);
        },
    }
}

// ── Graceful shutdown ──────────────────────────────────────────────────────

/// Set by the platform's interrupt/termination notification. Polled by the
/// accept loop; never cleared.
var shutdown_flag: ?*std.atomic.Value(bool) = null;

/// Loopback port to poke so a BLOCKING accept() returns. Windows only, and
/// only because WSAPoll cannot take std.Io.net's socket handle (WSAENOTSOCK),
/// so the accept loop has no 1-second timeout there to notice the flag on.
/// Without this, Ctrl+C on an idle server sets the flag and then hangs, since
/// the handler claims the event and suppresses the default terminator.
var wake_port: u16 = 0;

fn posixHandler(_: std.posix.SIG) callconv(.c) void {
    if (shutdown_flag) |f| f.store(true, .release);
}

const CTRL_C_EVENT: u32 = 0;
const CTRL_BREAK_EVENT: u32 = 1;
const CTRL_CLOSE_EVENT: u32 = 2;
const CTRL_LOGOFF_EVENT: u32 = 5;
const CTRL_SHUTDOWN_EVENT: u32 = 6;

extern "kernel32" fn SetConsoleCtrlHandler(
    handler: ?*const fn (u32) callconv(.winapi) c_int,
    add: c_int,
) callconv(.winapi) c_int;

fn windowsHandler(ctrl_type: u32) callconv(.winapi) c_int {
    switch (ctrl_type) {
        CTRL_C_EVENT, CTRL_BREAK_EVENT, CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT => {
            if (shutdown_flag) |f| f.store(true, .release);
            wakeAcceptLoop();
            // Returning TRUE means "handled": the default handler, which
            // terminates the process immediately, does not run. That is the
            // whole point -- the accept loop needs to observe the flag and
            // unwind. Note Windows gives a CLOSE/LOGOFF/SHUTDOWN handler only a
            // few seconds before killing the process anyway, so shutdown work
            // must stay bounded.
            return 1;
        },
        else => return 0,
    }
}

/// Connect to our own listener so a blocked accept() returns and the loop can
/// observe the shutdown flag. Best-effort: a failure just means the process
/// waits for the next real connection, which is where it already was.
fn wakeAcceptLoop() void {
    if (builtin.os.tag != .windows or wake_port == 0) return;
    ensureWinsock();
    const AF_INET: c_int = 2;
    const SOCK_STREAM: c_int = 1;
    const INVALID: usize = ~@as(usize, 0);
    const sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID) return;
    defer _ = closesocket(sock);
    var addr = SockaddrIn{
        .family = @intCast(AF_INET),
        .port = std.mem.nativeToBig(u16, wake_port),
        .addr = std.mem.nativeToBig(u32, 0x7F000001),
    };
    _ = connect(sock, &addr, @sizeOf(SockaddrIn));
}

/// Route interrupt / termination notifications into `flag`.
///
/// POSIX: SIGINT + SIGTERM. Windows has no signals in the POSIX sense
/// (`std.posix.Sigaction` does not even exist there); the equivalent is a
/// console control handler, which runs on its own thread rather than
/// interrupting one.
/// `listen_port` is the server's own port, used on Windows to unblock accept()
/// (see `wake_port`). Pass 0 when there is no listener to wake.
pub fn installShutdownHandler(flag: *std.atomic.Value(bool), listen_port: u16) void {
    shutdown_flag = flag;
    wake_port = listen_port;
    switch (builtin.os.tag) {
        .windows => {
            _ = SetConsoleCtrlHandler(windowsHandler, 1);
        },
        else => {
            const sigact = std.posix.Sigaction{
                .handler = .{ .handler = posixHandler },
                .mask = std.posix.sigemptyset(),
                .flags = 0,
            };
            std.posix.sigaction(std.posix.SIG.INT, &sigact, null);
            std.posix.sigaction(std.posix.SIG.TERM, &sigact, null);
        },
    }
}

// ── Tests ──

const testing = std.testing;

test "sleepMs waits at least the requested duration" {
    // Guards the shape of the shim, not the scheduler's precision: a wrong
    // unit conversion (the ms -> {sec, nsec} split) is the failure this
    // catches, and it would show up as a near-zero elapsed time.
    const io = std.testing.io;
    const start = std.Io.Timestamp.now(io, .boot);
    sleepMs(25);
    const elapsed_ms = std.Io.Timestamp.now(io, .boot).toMilliseconds() - start.toMilliseconds();
    try testing.expect(elapsed_ms >= 20); // allow a little timer slack
    try testing.expect(elapsed_ms < 5000); // and catch a unit error the other way
}

test "installShutdownHandler binds the flag it is given" {
    var flag = std.atomic.Value(bool).init(false);
    installShutdownHandler(&flag, 0);
    try testing.expect(shutdown_flag == &flag);
    try testing.expect(!flag.load(.acquire));
}

// ── Socket readiness ───────────────────────────────────────────────────────
//
// `std.posix.poll` does not exist on Windows in this Zig: `std.posix.pollfd`
// resolves through `std.c` to `os.windows.ws2_32.pollfd`, which is not
// declared. Winsock's equivalent is WSAPoll, whose struct differs (the fd is a
// SOCKET-width integer, not an int), so the whole probe is expressed once here
// rather than shimmed field by field at the call sites.

pub const Handle = std.posix.fd_t;

var winsock_started: std.atomic.Value(bool) = .init(false);

/// Idempotent WSAStartup. Winsock refcounts it, and the flag only avoids the
/// repeat call cost. Public because `lan_net.zig` opens sockets of its own and
/// must not depend on something else here having run first (§6.2: WSAPoll
/// answers WSANOTINITIALISED until this has run in the process).
pub fn ensureWinsock() void {
    if (builtin.os.tag != .windows) return;
    if (winsock_started.swap(true, .monotonic)) return;
    var wsa: [400]u8 = undefined;
    _ = WSAStartup(0x0202, &wsa);
}

var poll_failed_logged: std.atomic.Value(bool) = .init(false);

/// One-shot, because a failing poll fails on EVERY loop iteration and would
/// otherwise fill the log faster than anything else could be read.
fn pollFailedOnce(err: c_int) void {
    if (poll_failed_logged.swap(true, .monotonic)) return;
    @import("log.zig").warn("[platform] WSAPoll failed (WSA error {d}); falling back to blocking accept\n", .{err});
}

const WsaPollFd = extern struct {
    fd: usize,
    events: i16,
    revents: i16,
};

// Winsock poll bits. POLLIN is (POLLRDNORM | POLLRDBAND); HUP/ERR/NVAL are
// output-only, exactly as in POSIX.
const WSA_POLLRDNORM: i16 = 0x0100;
const WSA_POLLRDBAND: i16 = 0x0200;
const WSA_POLLIN: i16 = WSA_POLLRDNORM | WSA_POLLRDBAND;
const WSA_POLLERR: i16 = 0x0001;
const WSA_POLLHUP: i16 = 0x0002;
const WSA_POLLNVAL: i16 = 0x0004;
const WSA_MSG_PEEK: c_int = 0x2;

extern "ws2_32" fn WSAGetLastError() callconv(.winapi) c_int;
extern "ws2_32" fn WSAPoll(fd_array: [*]WsaPollFd, fds: u32, timeout: i32) callconv(.winapi) i32;
extern "ws2_32" fn recv(s: usize, buf: [*]u8, len: c_int, flags: c_int) callconv(.winapi) c_int;

pub const SocketState = enum {
    /// The poll itself failed. Distinct from `.idle` so a caller can fall
    /// through to a blocking operation instead of spinning on "nothing yet".
    poll_failed,
    /// Nothing to read, and the peer has not gone away.
    idle,
    /// Readable: data is buffered, OR the peer sent FIN (a read would return 0).
    readable,
    /// Hangup, error, or a handle that is not a socket.
    closed,
};

/// Poll ONE socket. `timeout_ms` of 0 returns immediately; negative blocks.
///
/// A poll error reports `.idle` rather than `.closed`: the callers treat
/// `.closed` as "give up on this peer", and a transient EINTR must not be
/// allowed to cancel a live request.
pub fn pollSocket(handle: Handle, timeout_ms: i32) SocketState {
    if (builtin.os.tag == .windows) {
        // WSAPoll needs Winsock started in THIS process. std.Io.net starts it
        // lazily for its own sockets, but nothing guarantees that has happened
        // before the accept loop's first poll -- and it had not: the first
        // build of this returned WSANOTINITIALISED (10093) on every iteration,
        // so the loop never called accept() and clients sat in CLOSE_WAIT
        // (Windows completes the handshake from the backlog, so the connection
        // looks established while the server never sees it). WSAStartup is
        // refcounted and idempotent.
        ensureWinsock();
        var fds = [_]WsaPollFd{.{
            .fd = @intFromPtr(handle),
            .events = WSA_POLLIN,
            .revents = 0,
        }};
        const n = WSAPoll(&fds, 1, timeout_ms);
        if (n < 0) {
            // A FAILED poll is not "nothing to read". Reporting .idle here made
            // the accept loop skip accept() forever: Windows completes the TCP
            // handshake from the backlog before the app accepts, so clients
            // connected, timed out, and the listener sat in CLOSE_WAIT.
            @branchHint(.cold);
            pollFailedOnce(WSAGetLastError());
            return .poll_failed;
        }
        if (n == 0) return .idle;
        const re = fds[0].revents;
        if ((re & (WSA_POLLHUP | WSA_POLLERR | WSA_POLLNVAL)) != 0) return .closed;
        if ((re & WSA_POLLIN) != 0) return .readable;
        return .idle;
    } else {
        var fds = [_]std.posix.pollfd{.{
            .fd = handle,
            .events = std.posix.POLL.IN,
            .revents = 0,
        }};
        const n = std.posix.poll(&fds, timeout_ms) catch return .poll_failed;
        if (n == 0) return .idle;
        const re = fds[0].revents;
        if ((re & (std.posix.POLL.HUP | std.posix.POLL.ERR | std.posix.POLL.NVAL)) != 0) return .closed;
        if ((re & std.posix.POLL.IN) != 0) return .readable;
        return .idle;
    }
}

/// Has the peer gone away? Non-blocking, and never consumes a byte.
///
/// `.readable` alone is not an answer: it means EITHER buffered data OR a
/// received FIN. Only a zero-length peek distinguishes them, which is why this
/// is one function and not `pollSocket` plus a caller-side check.
pub fn peerClosed(handle: Handle) bool {
    switch (pollSocket(handle, 0)) {
        // A failed poll is not evidence the peer left: `.closed` cancels a live
        // request, so an unknown answer must err toward "still there".
        .idle, .poll_failed => return false,
        .closed => return true,
        .readable => {},
    }
    var peek: [1]u8 = undefined;
    const r: isize = if (builtin.os.tag == .windows)
        recv(@intFromPtr(handle), &peek, peek.len, WSA_MSG_PEEK)
    else
        std.c.recv(handle, &peek, peek.len, std.posix.MSG.PEEK | std.posix.MSG.DONTWAIT);
    if (r == 0) return true; // FIN: peer closed cleanly
    return false; // negative (EAGAIN) or data available -> assume alive
}

// ── Home directory ─────────────────────────────────────────────────────────

/// The user's home directory, or null when the host does not name one.
///
/// `HOME` is a POSIX convention that Windows does not set; the Win32 equivalent
/// is `USERPROFILE`, with `HOMEDRIVE` + `HOMEPATH` as the older fallback that
/// some locked-down/domain profiles still rely on. Everything the server keeps
/// per-user (`~/.mlx-serve/models`, `logs/`, `skills/`) hangs off this, so a
/// wrong answer here is not cosmetic -- it silently relocates the model
/// library.
///
/// Writes into `buf` and returns the slice, because the Windows fallback has to
/// concatenate two variables.
pub fn homeDir(buf: []u8) ?[]const u8 {
    // An explicitly set HOME wins on every host. On POSIX that is simply the
    // convention; on Windows it is what makes the suite's isolation idiom
    // (`HOME=$(mktemp -d) mlx-serve ...`) mean anything -- without it a faked
    // home is ignored and the run silently reads and writes the real
    // ~/.mlx-serve. Git Bash sets HOME to the same directory USERPROFILE names,
    // so a normal Windows launch resolves identically either way, and a native
    // process that sets no HOME falls through to the Windows variables below.
    if (std.c.getenv("HOME")) |h| {
        const s = std.mem.span(h);
        if (s.len > 0 and s.len <= buf.len) {
            @memcpy(buf[0..s.len], s);
            return buf[0..s.len];
        }
    }
    if (builtin.os.tag == .windows) {
        if (std.c.getenv("USERPROFILE")) |p| {
            const s = std.mem.span(p);
            if (s.len > 0 and s.len <= buf.len) {
                @memcpy(buf[0..s.len], s);
                return buf[0..s.len];
            }
        }
        const drive = std.c.getenv("HOMEDRIVE") orelse return null;
        const path = std.c.getenv("HOMEPATH") orelse return null;
        const d = std.mem.span(drive);
        const p = std.mem.span(path);
        if (d.len + p.len == 0 or d.len + p.len > buf.len) return null;
        @memcpy(buf[0..d.len], d);
        @memcpy(buf[d.len..][0..p.len], p);
        return buf[0 .. d.len + p.len];
    }
    // POSIX: HOME is the only source, and it was already tried above. Reaching
    // here means it was unset, empty, or longer than `buf`.
    return null;
}

test "homeDir honours an explicitly set HOME on every host" {
    // The POSIX branch always read HOME; the Windows one read USERPROFILE (then
    // HOMEDRIVE+HOMEPATH) and nothing else. That made the suite's standard
    // isolation idiom -- `HOME=$(mktemp -d) ./zig-out/bin/mlx-serve ...` -- a
    // silent no-op here: six test scripts fake a home that way, and on Windows
    // every one of them was exercising the DEVELOPER'S real ~/.mlx-serve
    // instead, both reading its contents and being able to write into it. A
    // fixture that does not take effect does not fail; it passes vacuously
    // (test_run_quiet, 2026-08-20).
    //
    // Explicit HOME wins on every host now. Git Bash sets HOME to the same
    // place USERPROFILE points at, so an ordinary Windows run is unchanged,
    // and a native process that sets no HOME still falls through to the
    // Windows variables below.
    var buf: [1024]u8 = undefined;
    const original = homeDir(&buf);
    var original_copy: [1024]u8 = undefined;
    var original_len: usize = 0;
    if (original) |o| {
        @memcpy(original_copy[0..o.len], o);
        original_len = o.len;
    }

    setEnv("HOME", "/tmp/mlx-serve-home-probe", true);
    defer {
        if (original_len > 0 and builtin.os.tag != .windows) {
            var restore: [1025]u8 = undefined;
            @memcpy(restore[0..original_len], original_copy[0..original_len]);
            restore[original_len] = 0;
            setEnv("HOME", restore[0..original_len :0], true);
        } else if (builtin.os.tag == .windows) {
            unsetEnv("HOME");
        }
    }

    var buf2: [1024]u8 = undefined;
    const got = homeDir(&buf2) orelse return error.NoHomeDirectory;
    try std.testing.expectEqualStrings("/tmp/mlx-serve-home-probe", got);
}

test "homeDir reports a non-empty path on this host" {
    // Not asserting a VALUE (it differs per host and per user); asserting that
    // the host's own convention resolves at all. A silent null here is what
    // relocates the whole model library to a fallback path.
    var buf: [1024]u8 = undefined;
    const home = homeDir(&buf) orelse return error.NoHomeDirectory;
    try testing.expect(home.len > 0);
}

// ── Randomness ─────────────────────────────────────────────────────────────

/// Cryptographically-strong bytes, without an `Io` handle.
///
/// `std.crypto.random` does not exist in this Zig, and `std.c.arc4random_buf`
/// is defined for Darwin and glibc>=2.36 only — nothing on Windows. The LAN
/// identity token depends on this being unguessable: it is what lets a browser
/// recognize and skip its OWN advertisement, so a predictable one is a peer
/// that can be impersonated into a proxy loop.
pub fn randomBytes(buf: []u8) void {
    if (builtin.os.tag == .windows) {
        // RtlGenRandom, exported under its undecorated ordinal name. Present
        // on every supported Windows and the documented way to do this
        // without dragging in the CNG headers.
        _ = SystemFunction036(buf.ptr, @intCast(buf.len));
        return;
    }
    std.c.arc4random_buf(buf.ptr, buf.len);
}

extern "advapi32" fn SystemFunction036(buf: [*]u8, len: u32) callconv(.winapi) u8;

test "randomBytes fills the buffer and does not repeat itself" {
    var a: [32]u8 = @splat(0);
    var b: [32]u8 = @splat(0);
    randomBytes(&a);
    randomBytes(&b);
    // An all-zero return is what a stubbed-out source looks like, and it is
    // the failure mode that matters: every instance would share a token.
    try testing.expect(!std.mem.allEqual(u8, &a, 0));
    try testing.expect(!std.mem.eql(u8, &a, &b));
}

// ── Environment ────────────────────────────────────────────────────────────

extern "c" fn _putenv_s(name: [*:0]const u8, value: [*:0]const u8) c_int;
// setenv/unsetenv are declared here rather than reached through the std C
// bindings: this nightly carries neither on any target (the only "*setenv" in
// std/c.zig is OpenBSD's auth_setenv), so the POSIX arm below did not compile
// at all off Windows -- and nothing caught it, because Windows takes the other
// arm and macOS had not been built since. They are plain libc everywhere we
// build, and a direct extern also stops this moving under us again (§6.7).
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

/// Set an environment variable for THIS process.
///
/// POSIX `setenv` does not exist in the Windows CRT; the equivalent is
/// `_putenv_s`, which has no `overwrite` flag (it always overwrites). The
/// callers here all pass overwrite=1, so nothing is lost — but the parameter is
/// kept so the signature reads the same at every call site.
pub fn setEnv(name: [*:0]const u8, value: [*:0]const u8, overwrite: bool) void {
    if (builtin.os.tag == .windows) {
        if (!overwrite and std.c.getenv(name) != null) return;
        _ = _putenv_s(name, value);
    } else {
        _ = setenv(name, value, @intFromBool(overwrite));
    }
}

/// Remove an environment variable. The Windows CRT spells "unset" as setting
/// the value to an empty string.
pub fn unsetEnv(name: [*:0]const u8) void {
    if (builtin.os.tag == .windows) {
        _ = _putenv_s(name, "");
    } else {
        _ = unsetenv(name);
    }
}

test "setEnv/unsetEnv round-trip on this host" {
    // Guards the platform SELECTION, not the CRT: a build that picked the
    // POSIX arm on Windows would not link at all, but one that picked a
    // no-op arm would silently make every env-driven kill switch inert.
    setEnv("MLX_SERVE_PLATFORM_TEST", "1", true);
    const got = std.c.getenv("MLX_SERVE_PLATFORM_TEST") orelse return error.SetEnvDidNothing;
    try testing.expectEqualStrings("1", std.mem.span(got));
    unsetEnv("MLX_SERVE_PLATFORM_TEST");
}

// ── Connected socket pair (tests) ──────────────────────────────────────────

const ws2 = std.os.windows.ws2_32;

extern "ws2_32" fn WSAStartup(version: u16, data: *[400]u8) callconv(.winapi) c_int;
extern "ws2_32" fn socket(af: c_int, kind: c_int, proto: c_int) callconv(.winapi) usize;
extern "ws2_32" fn bind(s: usize, addr: *const anyopaque, len: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn listen(s: usize, backlog: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn getsockname(s: usize, addr: *anyopaque, len: *c_int) callconv(.winapi) c_int;
extern "ws2_32" fn connect(s: usize, addr: *const anyopaque, len: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn accept(s: usize, addr: ?*anyopaque, len: ?*c_int) callconv(.winapi) usize;
extern "ws2_32" fn closesocket(s: usize) callconv(.winapi) c_int;

const SockaddrIn = extern struct {
    family: u16,
    port: u16, // network byte order
    addr: u32, // network byte order
    zero: [8]u8 = @splat(0),
};

/// Two connected stream sockets, for tests that need a real peer to close.
///
/// POSIX has `socketpair(AF_UNIX, SOCK_STREAM)`; Windows has no equivalent, so
/// this builds the same thing over TCP loopback (bind :0, listen, connect,
/// accept). Written rather than skipping the tests: `peerClosed` is the probe
/// this port REWROTE, and a skipped arm on the host whose implementation is new
/// would read as a pass while testing nothing.
pub fn connectedPair() !struct { a: Handle, b: Handle } {
    if (builtin.os.tag == .windows) {
        ensureWinsock();

        const AF_INET: c_int = 2;
        const SOCK_STREAM: c_int = 1;
        const INVALID: usize = ~@as(usize, 0);

        const srv = socket(AF_INET, SOCK_STREAM, 0);
        if (srv == INVALID) return error.SocketFailed;
        defer _ = closesocket(srv);

        // 127.0.0.1:0 — the kernel picks the port, which getsockname reports.
        var addr = SockaddrIn{ .family = @intCast(AF_INET), .port = 0, .addr = std.mem.nativeToBig(u32, 0x7F000001) };
        if (bind(srv, &addr, @sizeOf(SockaddrIn)) != 0) return error.BindFailed;
        if (listen(srv, 1) != 0) return error.ListenFailed;

        var bound: SockaddrIn = undefined;
        var blen: c_int = @sizeOf(SockaddrIn);
        if (getsockname(srv, &bound, &blen) != 0) return error.GetSockNameFailed;

        const client = socket(AF_INET, SOCK_STREAM, 0);
        if (client == INVALID) return error.SocketFailed;
        errdefer _ = closesocket(client);
        if (connect(client, &bound, @sizeOf(SockaddrIn)) != 0) return error.ConnectFailed;

        const server_side = accept(srv, null, null);
        if (server_side == INVALID) return error.AcceptFailed;

        return .{ .a = @ptrFromInt(client), .b = @ptrFromInt(server_side) };
    }

    var sv: [2]std.posix.fd_t = undefined;
    const AF_UNIX: c_uint = 1;
    const SOCK_STREAM: c_uint = 1;
    if (std.c.socketpair(AF_UNIX, SOCK_STREAM, 0, &sv) != 0) return error.SocketPairFailed;
    return .{ .a = sv[0], .b = sv[1] };
}

/// Close one half of a `connectedPair`.
pub fn closeSocket(h: Handle) void {
    if (builtin.os.tag == .windows) {
        _ = closesocket(@intFromPtr(h));
    } else {
        _ = std.c.close(h);
    }
}

test "connectedPair: a live peer is not closed, a hung-up peer is" {
    // Exercises the REWRITTEN peerClosed probe end to end on this host --
    // the WSAPoll path on Windows, poll(2) elsewhere.
    const pair = try connectedPair();
    try testing.expect(!peerClosed(pair.a));
    closeSocket(pair.b);
    try testing.expect(peerClosed(pair.a));
    closeSocket(pair.a);
}

// ── Test helpers ───────────────────────────────────────────────────────────

/// Absolute path of a `std.testing.tmpDir`, built with the NATIVE separator.
///
/// Several tests need one because the code under test opens by absolute path
/// (the `openDirAbsolute` UB class). Hand-formatting `"{s}/.zig-cache/tmp/{s}"`
/// works on POSIX but produces a mixed-separator path on Windows, where
/// `getcwd` already returns backslashes.
pub fn tmpDirPath(buf: []u8, sub_path: []const u8) ![]const u8 {
    var cwd_buf: [4096]u8 = undefined;
    const raw = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(raw)));
    const s = std.fs.path.sep_str;
    return std.fmt.bufPrint(buf, "{s}" ++ s ++ ".zig-cache" ++ s ++ "tmp" ++ s ++ "{s}", .{ cwd, sub_path });
}

/// Allocating variant of `tmpDirPath`, plus optional trailing components.
pub fn tmpDirPathAlloc(allocator: std.mem.Allocator, sub_path: []const u8) ![]u8 {
    var buf: [4096]u8 = undefined;
    return allocator.dupe(u8, try tmpDirPath(&buf, sub_path));
}

test "tmpDirPath uses the native separator and is absolute" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [4096]u8 = undefined;
    const p = try tmpDirPath(&buf, &tmp.sub_path);
    try testing.expect(std.fs.path.isAbsolute(p));
    if (builtin.os.tag == .windows) {
        try testing.expect(std.mem.indexOfScalar(u8, p, '/') == null);
    }
}

// ── Absolute-path recognition ──────────────────────────────────────────────

/// Does `p` look like an ABSOLUTE path on this host?
///
/// Exists because `std.mem.startsWith(u8, p, "/")` is a POSIX-only spelling
/// that reads as host-neutral. Two dispatch guards in server.zig used it to
/// decide whether a `model` field naming a directory should go through
/// register-by-path, so on Windows -- where an absolute path is `C:/...`, not
/// `/...` -- that whole feature was unreachable and every load-by-path answered
/// "Unknown model id" (found 2026-08-20 via tests/test_model_rescan.sh).
///
/// Deliberately a LOOKS-like test, not a validity test: the only job is to
/// separate "this is a path" from "this is a model id" (`org/name`,
/// `name:tag`, `id#file.gguf`). Whether the path exists or is loadable is
/// `registerByPath`'s answer, and it already speaks both forms.
pub fn looksAbsolutePath(p: []const u8) bool {
    if (p.len == 0) return false;
    // A leading slash is absolute on every host, and Windows accepts it as
    // root-of-current-drive -- so it stays a path there too rather than being
    // mistaken for an id.
    if (p[0] == '/') return true;
    if (builtin.os.tag != .windows) return false;
    if (p[0] == '\\') return true; // UNC, or root-relative backslash
    // Drive-qualified: X:\... or X:/...  A bare "C:" (drive-relative) is not
    // something a client can mean as a model directory, so it does not count.
    if (p.len >= 3 and p[1] == ':' and std.ascii.isAlphabetic(p[0]) and
        (p[2] == '/' or p[2] == '\\')) return true;
    return false;
}

test "looksAbsolutePath: ids are never paths, POSIX roots always are" {
    // Model ids must NOT be mistaken for paths on any host -- these are the
    // real id shapes (discovery org/name, Ollama name:tag, GGUF id#file).
    try std.testing.expect(!looksAbsolutePath("org/name"));
    try std.testing.expect(!looksAbsolutePath("qwen3:latest"));
    try std.testing.expect(!looksAbsolutePath("org/name#model.gguf"));
    try std.testing.expect(!looksAbsolutePath("mlx-serve"));
    try std.testing.expect(!looksAbsolutePath(""));
    // A POSIX absolute path is a path everywhere, including Windows.
    try std.testing.expect(looksAbsolutePath("/Users/x/models/foo"));
    try std.testing.expect(looksAbsolutePath("/"));
}

test "looksAbsolutePath: Windows drive and UNC forms, only on Windows" {
    const win = builtin.os.tag == .windows;
    // Both slash directions -- Git Bash hands over C:/..., the shell and the
    // Windows API both produce C:\...
    try std.testing.expectEqual(win, looksAbsolutePath("C:/Users/x/models/foo"));
    try std.testing.expectEqual(win, looksAbsolutePath("C:" ++ [_]u8{'\\'} ++ "Users"));
    try std.testing.expectEqual(win, looksAbsolutePath("z:/lower/drive/letter"));
    // UNC share.
    try std.testing.expectEqual(win, looksAbsolutePath([_]u8{ '\\', '\\' } ++ "server/share/model"));
    // A drive letter with NO separator is drive-RELATIVE; nobody names a model
    // directory that way, and treating it as a path would swallow the id.
    try std.testing.expect(!looksAbsolutePath("C:"));
}

/// Strip trailing path separators from `p`, leaving a root ("/" or "C:/")
/// alone. Windows accepts BOTH separators, so a hand-rolled `while (p[last] ==
/// '/')` silently keeps a trailing backslash and hands `basename` an empty
/// string.
pub fn trimTrailingSeps(p: []const u8) []const u8 {
    var t = p;
    while (t.len > 1 and isSep(t[t.len - 1])) {
        // Do not eat the separator that IS the root: "/" stays "/", and
        // "C:/" stays "C:/" rather than collapsing to a drive-relative "C:".
        if (t.len == 3 and t[1] == ':' and builtin.os.tag == .windows) break;
        t = t[0 .. t.len - 1];
    }
    return t;
}

fn isSep(c: u8) bool {
    if (c == '/') return true;
    return builtin.os.tag == .windows and c == '\\';
}

test "trimTrailingSeps: both separators, roots preserved" {
    try std.testing.expectEqualStrings("/a/b", trimTrailingSeps("/a/b/"));
    try std.testing.expectEqualStrings("/a/b", trimTrailingSeps("/a/b///"));
    try std.testing.expectEqualStrings("/a/b", trimTrailingSeps("/a/b"));
    // A lone root must survive -- trimming it to "" makes basename fail with
    // InvalidModelPath on a path that was merely odd.
    try std.testing.expectEqualStrings("/", trimTrailingSeps("/"));
    try std.testing.expectEqualStrings("", trimTrailingSeps(""));
    if (builtin.os.tag == .windows) {
        try std.testing.expectEqualStrings("C:/a", trimTrailingSeps("C:/a" ++ [_]u8{'\\'}));
        try std.testing.expectEqualStrings("C:/a", trimTrailingSeps("C:/a/"));
        try std.testing.expectEqualStrings("C:/", trimTrailingSeps("C:/"));
    } else {
        // Off Windows a backslash is an ordinary filename byte and must NOT
        // be eaten -- doing so would rename the directory being asked for.
        try std.testing.expectEqualStrings("/a/b" ++ [_]u8{'\\'}, trimTrailingSeps("/a/b" ++ [_]u8{'\\'}));
    }
}
