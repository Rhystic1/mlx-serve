//! Leveled logging with an optional persistent file sink.
//!
//! stderr is always written (the macOS app scrapes it into an in-memory
//! rolling buffer). When a file sink is open every emitted line is ALSO
//! appended to disk, so a server that ran hours ago can still be diagnosed —
//! the rolling buffer is gone the moment the app quits, and a crashed or
//! restarted server takes its whole history with it.
//!
//! The sink uses raw libc `write(2)` rather than `std.Io`: log calls come from
//! the accept loop, every connection thread and the inference thread, none of
//! which carry an `Io` handle. A mutex serializes writes + rotation; a failed
//! write is swallowed (logging must never take the server down).

const std = @import("std");
const builtin = @import("builtin");

pub const Level = enum {
    err,
    warn,
    info,
    debug,

    pub fn fromString(s: []const u8) ?Level {
        if (std.mem.eql(u8, s, "error")) return .err;
        if (std.mem.eql(u8, s, "warn")) return .warn;
        if (std.mem.eql(u8, s, "info")) return .info;
        if (std.mem.eql(u8, s, "debug")) return .debug;
        return null;
    }
};

var current_level: Level = .info;

pub fn setLevel(level: Level) void {
    current_level = level;
}

pub fn isDebug() bool {
    return @backingInt(current_level) >= @backingInt(Level.debug);
}

// ── File sink ──

/// Rotate to `<path>.1` once the live file passes this. Two files, so the
/// worst case on disk is bounded at 2x. `--log-level debug` on an agent
/// workload writes a few MB an hour, so this holds days of history.
pub const default_max_bytes: u64 = 32 * 1024 * 1024;

/// Longest single log line written to the sink. The biggest real line is the
/// debug raw-tool-parse dump (4 KB of model output plus a prefix); anything
/// past this is truncated with a visible marker so a clipped line can never be
/// mistaken for what the model actually emitted.
const line_buf_len = 16 * 1024;
const truncation_marker = "…[mlx-serve: log line truncated]\n";

/// The sink's `Io`, captured at `openFile`. Log calls arrive from the accept
/// loop, every connection thread and the inference thread, none of which carry
/// an `Io` of their own — so the sink stores the one its opener had. Reading it
/// from other threads is safe because the handle is a plain interface value
/// (vtable + context) and the implementation behind it is thread-safe.
///
/// This replaced a raw-libc (`open`/`write`/`pthread_mutex_t`) sink. That layer
/// does not exist off POSIX: this Zig has no `std.fs`, no `std.posix.open`, and
/// `std.c.open` is not even declarable under the Windows calling convention
/// ("parameter of type 'void' not allowed"). `std.Io` is the one file API that
/// spans all three hosts.
var sink_io: ?std.Io = null;

/// Serializes writes + rotation. `std.Io.Mutex` is the codebase-wide idiom.
var sink_mutex: std.Io.Mutex = .init;

var sink_file: ?std.Io.File = null;
/// Doubles as the rotation counter AND the write offset: the sink writes
/// positionally rather than opening in append mode, because a tracked offset is
/// portable while O_APPEND is not, and every write already holds the mutex.
var sink_bytes: u64 = 0;
var sink_max_bytes: u64 = default_max_bytes;
var sink_path_buf: [1024]u8 = undefined;
var sink_path_len: usize = 0;
/// Read outside the mutex on every log call, so keep it atomic. When false the
/// sink costs one relaxed load and nothing else.
var sink_active: bool = false;

/// Build the default log path for a server listening on `port`:
/// `<home>/.mlx-serve/logs/mlx-serve-<port>.log`. Per-port so the app's server
/// and a test server never interleave into one file. Returns the slice written
/// into `buf`.
pub fn defaultLogPath(buf: []u8, home: []const u8, port: u16) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/.mlx-serve/logs/mlx-serve-{d}.log", .{ home, port });
}

/// Pure rotation policy: would appending `incoming` bytes push the live file
/// past its cap? An empty file never rotates, so a single line larger than the
/// cap is written rather than looping forever.
pub fn shouldRotate(current_bytes: u64, incoming: usize, max_bytes: u64) bool {
    if (max_bytes == 0) return false;
    if (current_bytes == 0) return false;
    return current_bytes + incoming > max_bytes;
}

/// Directory portion of `path`, or null when it has none.
///
/// Splits on BOTH separators, not just '/': `defaultLogPath` builds forward
/// slashes but `--log-file` on Windows is typed with backslashes, and a
/// '/'-only scan would treat `C:\...\logs\x.log` as having no parent and then
/// fail to create it.
pub fn parentDir(path: []const u8) ?[]const u8 {
    const fwd = std.mem.lastIndexOfScalar(u8, path, '/');
    const back = std.mem.lastIndexOfScalar(u8, path, '\\');
    const cut = if (fwd != null and back != null)
        @max(fwd.?, back.?)
    else
        fwd orelse back orelse return null;
    if (cut == 0) return null; // a root-relative path has no parent to create
    return path[0..cut];
}

/// `mkdir -p` for the directory holding `path`. An already-existing tree is
/// fine; a genuinely unwritable path surfaces later as OpenLogFileFailed.
fn makeParentDirs(io: std.Io, path: []const u8) void {
    const parent = parentDir(path) orelse return;
    std.Io.Dir.cwd().createDirPath(io, parent) catch {};
}

/// Open (create/append) a log file, creating parent directories as needed.
/// `max_bytes` of 0 disables rotation. Errors are returned so the caller can
/// warn — a failure here is never fatal to the server.
pub fn openFile(io: std.Io, path: []const u8, max_bytes: u64) !void {
    if (path.len == 0 or path.len >= sink_path_buf.len) return error.InvalidLogPath;

    makeParentDirs(io, path);

    // truncate=false so reopening APPENDS: a restarted server keeps its history.
    const file = std.Io.Dir.cwd().createFile(io, path, .{ .truncate = false }) catch
        return error.OpenLogFileFailed;
    const end = file.length(io) catch 0;

    sink_mutex.lock(io) catch return error.OpenLogFileFailed;
    defer sink_mutex.unlock(io);
    if (sink_file) |f| f.close(io);
    sink_io = io;
    sink_file = file;
    sink_max_bytes = max_bytes;
    // Start the rotation counter (and the write offset) at the file's size.
    sink_bytes = end;
    @memcpy(sink_path_buf[0..path.len], path);
    sink_path_len = path.len;
    @atomicStore(bool, &sink_active, true, .release);
}

pub fn closeFile() void {
    const io = sink_io orelse return;
    sink_mutex.lock(io) catch return;
    defer sink_mutex.unlock(io);
    @atomicStore(bool, &sink_active, false, .release);
    if (sink_file) |f| f.close(io);
    sink_file = null;
    sink_bytes = 0;
    sink_path_len = 0;
}

/// Path of the live log file, or null when no sink is open.
pub fn filePath() ?[]const u8 {
    if (!@atomicLoad(bool, &sink_active, .acquire)) return null;
    return sink_path_buf[0..sink_path_len];
}

/// Caller holds `sink_mutex`.
fn rotateLocked(io: std.Io) void {
    if (sink_path_len == 0) return;
    const live = sink_path_buf[0..sink_path_len];
    var prev_buf: [1024 + 2]u8 = undefined;
    @memcpy(prev_buf[0..sink_path_len], live);
    @memcpy(prev_buf[sink_path_len..][0..2], ".1");
    const prev = prev_buf[0 .. sink_path_len + 2];

    if (sink_file) |f| f.close(io);
    sink_file = null;
    // `rename` rather than `renameAbsolute`: the latter asserts both paths are
    // absolute (it is otherwise the identical cwd-relative call), and a log path
    // may legitimately be relative -- `--log-file server.log`, and the tests.
    const cwd = std.Io.Dir.cwd();
    cwd.rename(live, cwd, prev, io) catch {};
    sink_file = std.Io.Dir.cwd().createFile(io, live, .{ .truncate = true }) catch null;
    sink_bytes = 0;
    if (sink_file == null) @atomicStore(bool, &sink_active, false, .release);
}

fn writeToSink(bytes: []const u8) void {
    const io = sink_io orelse return;
    sink_mutex.lock(io) catch return;
    defer sink_mutex.unlock(io);
    if (sink_file == null) return;
    if (shouldRotate(sink_bytes, bytes.len, sink_max_bytes)) rotateLocked(io);
    const file = sink_file orelse return;

    // Positional, at the tracked offset — see `sink_bytes`. A failed write is
    // swallowed (ENOSPC / EIO): logging must never take the server down.
    file.writePositionalAll(io, bytes, sink_bytes) catch return;
    sink_bytes += bytes.len;
}

/// Server builds log to stderr; TEST builds do not.
///
/// A unit test that exercises a logging code path (the tool-call parse layer
/// alone emits hundreds of lines) sprays them into the suite's stderr, where
/// they do two kinds of damage: they bury the one line that matters when a test
/// actually fails, and Zig's build runner echoes any test stderr back tagged
/// with a `failed command: …/test --listen=-` line — which READS AS A FAILURE
/// even though the build exits 0 and every test passed. That false signal cost
/// real debugging time on both sides of a genuine CI break (2026-07-14). Test
/// failures themselves are unaffected: they surface through the test runner and
/// through each test's own `std.debug.print`, neither of which goes through here.
///
/// Flip it locally (see the file-sink test) when a test must assert on logging.
var stderr_enabled: bool = !builtin.is_test;

/// Env-gated live-test harnesses (generation runs driven through the test
/// binary) opt back into stderr logging: the phase/step lines ARE the
/// harness's observable output, and a profile run without them is blind.
pub fn enableStderr() void {
    stderr_enabled = true;
}

/// Format once, fan out to stderr and (when open) the file.
fn emit(comptime fmt: []const u8, args: anytype) void {
    if (stderr_enabled) std.debug.print(fmt, args);
    if (!@atomicLoad(bool, &sink_active, .acquire)) return;

    var buf: [line_buf_len]u8 = undefined;
    // Format into everything but the marker's reserved tail, so an overlong
    // line stays ONE `write(2)` — two writes could interleave with another
    // thread's line between them.
    var w = std.Io.Writer.fixed(buf[0 .. buf.len - truncation_marker.len]);
    if (w.print(fmt, args)) |_| {
        writeToSink(w.buffered());
    } else |_| {
        const kept = w.buffered().len;
        @memcpy(buf[kept..][0..truncation_marker.len], truncation_marker);
        writeToSink(buf[0 .. kept + truncation_marker.len]);
    }
}

pub fn info(comptime fmt: []const u8, args: anytype) void {
    if (@backingInt(current_level) >= @backingInt(Level.info)) {
        emit(fmt, args);
    }
}

pub fn warn(comptime fmt: []const u8, args: anytype) void {
    if (@backingInt(current_level) >= @backingInt(Level.warn)) {
        emit(fmt, args);
    }
}

pub fn err(comptime fmt: []const u8, args: anytype) void {
    if (@backingInt(current_level) >= @backingInt(Level.err)) {
        emit(fmt, args);
    }
}

pub fn debug(comptime fmt: []const u8, args: anytype) void {
    if (@backingInt(current_level) >= @backingInt(Level.debug)) {
        emit(fmt, args);
    }
}

// ── Tests ──

const testing = std.testing;

test "Level.fromString valid levels" {
    try testing.expectEqual(Level.err, Level.fromString("error").?);
    try testing.expectEqual(Level.warn, Level.fromString("warn").?);
    try testing.expectEqual(Level.info, Level.fromString("info").?);
    try testing.expectEqual(Level.debug, Level.fromString("debug").?);
}

test "Level.fromString invalid returns null" {
    try testing.expect(Level.fromString("verbose") == null);
    try testing.expect(Level.fromString("") == null);
    try testing.expect(Level.fromString("INFO") == null);
}

test "Level ordering" {
    // err < warn < info < debug
    try testing.expect(@backingInt(Level.err) < @backingInt(Level.warn));
    try testing.expect(@backingInt(Level.warn) < @backingInt(Level.info));
    try testing.expect(@backingInt(Level.info) < @backingInt(Level.debug));
}

test "setLevel changes current level" {
    const original = current_level;
    defer setLevel(original); // restore

    setLevel(.debug);
    try testing.expectEqual(Level.debug, current_level);
    try testing.expect(isDebug());
    setLevel(.err);
    try testing.expectEqual(Level.err, current_level);
    try testing.expect(!isDebug());
}

test "defaultLogPath is per-port under ~/.mlx-serve/logs" {
    var buf: [256]u8 = undefined;
    const p = try defaultLogPath(&buf, "/Users/x", 11234);
    try testing.expectEqualStrings("/Users/x/.mlx-serve/logs/mlx-serve-11234.log", p);
    // Per-port: the app's server and a test server never share a file.
    const q = try defaultLogPath(&buf, "/Users/x", 8098);
    try testing.expectEqualStrings("/Users/x/.mlx-serve/logs/mlx-serve-8098.log", q);
}

test "shouldRotate: caps growth, never loops on an oversized first line" {
    try testing.expect(!shouldRotate(0, 100, 1000)); // empty file: always accept
    try testing.expect(!shouldRotate(900, 100, 1000)); // exactly at the cap
    try testing.expect(shouldRotate(901, 100, 1000)); // one past
    try testing.expect(!shouldRotate(5000, 100, 0)); // 0 = rotation disabled
    // A single line larger than the whole cap is written to an empty file
    // rather than rotating forever.
    try testing.expect(!shouldRotate(0, 999_999, 1000));
}

// NOTE: the sink is a process-global singleton, so every assertion about it
// lives in ONE test — the build runner executes tests in parallel and two
// tests calling `openFile` would fight over `sink_fd`.
test "parentDir splits on both separators" {
    // `--log-file` on Windows is typed with backslashes; a '/'-only scan would
    // report "no parent" and then silently fail to create the directory.
    try testing.expectEqualStrings("/a/b", parentDir("/a/b/c.log").?);
    try testing.expectEqualStrings("C:\\a\\b", parentDir("C:\\a\\b\\c.log").?);
    try testing.expectEqualStrings("C:/a\\b", parentDir("C:/a\\b/c.log").?);
    try testing.expect(parentDir("c.log") == null);
    try testing.expect(parentDir("/c.log") == null); // nothing to create
}

test "file sink: writes lines, honors level, survives reopen, rotates at the cap" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    // The sink takes a PATH (it opens through cwd), so build one pointing into
    // the tmp dir rather than handing it a Dir. `tmpDir` lives at a known
    // cwd-relative location, which keeps this free of any absolute-path or
    // /tmp assumption -- neither of which exists on Windows.
    var base_buf: [256]u8 = undefined;
    const sep = std.fs.path.sep_str;
    const base = try std.fmt.bufPrint(&base_buf, ".zig-cache" ++ sep ++ "tmp" ++ sep ++ "{s}", .{tmp.sub_path});

    var path_buf: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/sink.log", .{base});
    var prev_buf: [512]u8 = undefined;
    const rotated = try std.fmt.bufPrint(&prev_buf, "{s}/sink.log.1", .{base});

    const original = current_level;
    defer setLevel(original);
    setLevel(.info);
    const original_stderr = stderr_enabled;
    stderr_enabled = false; // this test drives info/debug for real
    defer stderr_enabled = original_stderr;
    defer closeFile();

    // No sink yet -> nothing on disk, and filePath reports that.
    try testing.expect(filePath() == null);

    try openFile(io, path, 0); // rotation disabled
    try testing.expectEqualStrings(path, filePath().?);
    info("hello {d}\n", .{42});
    info("second line\n", .{});
    // Below the threshold: stderr and disk both stay quiet.
    debug("this must not reach disk\n", .{});
    closeFile();
    try testing.expect(filePath() == null);

    var buf: [4096]u8 = undefined;
    const first = readFileForTest(io, path, &buf);
    try testing.expect(std.mem.indexOf(u8, first, "hello 42\n") != null);
    try testing.expect(std.mem.indexOf(u8, first, "second line\n") != null);
    try testing.expect(std.mem.indexOf(u8, first, "must not reach disk") == null);

    // Reopening appends rather than truncating: history survives a restart.
    try openFile(io, path, 0);
    info("after restart\n", .{});
    closeFile();
    const second = readFileForTest(io, path, &buf);
    try testing.expect(std.mem.indexOf(u8, second, "hello 42\n") != null);
    try testing.expect(std.mem.indexOf(u8, second, "after restart\n") != null);

    // A tiny cap forces a rotation; the old bytes land in `.log.1`.
    try openFile(io, path, 64);
    var i: usize = 0;
    while (i < 40) : (i += 1) info("filler line {d}\n", .{i});
    closeFile();
    const live = readFileForTest(io, path, &buf);
    try testing.expect(live.len > 0);
    try testing.expect(live.len <= 64 + 32); // bounded: the last line may overshoot
    var buf2: [4096]u8 = undefined;
    try testing.expect(readFileForTest(io, rotated, &buf2).len > 0); // rotated backup exists

    // Nested parents are created: the real default path is
    // ~/.mlx-serve/logs/… and BOTH components can be missing on a fresh HOME.
    var deep_buf: [512]u8 = undefined;
    const deep = try std.fmt.bufPrint(&deep_buf, "{s}/a/b/c/deep.log", .{base});
    try openFile(io, deep, 0);
    info("deep\n", .{});
    closeFile();
    try testing.expect(readFileForTest(io, deep, &buf2).len > 0);

    // An overlong line is clipped and SAYS SO — a silently-clipped model dump
    // must never read as the model's actual output.
    std.Io.Dir.cwd().deleteFile(io, deep) catch {};
    try openFile(io, deep, 0);
    const huge: [line_buf_len + 500]u8 = @splat('x');
    info("{s}\n", .{huge});
    closeFile();
    var big: [line_buf_len * 2]u8 = undefined;
    const wrote = readFileForTest(io, deep, &big);
    try testing.expectEqual(@as(usize, line_buf_len), wrote.len); // never exceeds the buffer
    try testing.expect(std.mem.endsWith(u8, wrote, truncation_marker));
}

fn readFileForTest(io: std.Io, path: []const u8, buf: []u8) []u8 {
    return std.Io.Dir.cwd().readFile(io, path, buf) catch buf[0..0];
}
