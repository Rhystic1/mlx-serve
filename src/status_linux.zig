//! Linux implementation of the process/system stat probes in status.zig.
//!
//! These are NOT cosmetic. `getTotalMemBytes` / `getAvailableMemBytes` feed the
//! model-load admission gate and the auto-context sizer, so returning zero here
//! would not merely blank a panel — it would make every load decision on wrong
//! numbers. That is why this is a real implementation rather than a stub, for
//! the same reason status_windows.zig is.
//!
//! Everything comes out of procfs rather than a syscall, because the syscall
//! that looks right is wrong: `sysinfo(2)`'s `freeram` is FREE memory, not
//! AVAILABLE memory, and it counts the page cache as used. The macOS path
//! deliberately does not subtract file-backed pages (the kernel evicts them the
//! instant a big allocation lands), and #45 is the bug you get when the guard
//! disagrees with the kernel about what is reclaimable. `MemAvailable` is the
//! kernel's OWN estimate of exactly that quantity, so it is the honest analogue
//! of `computeAvailableBytes` — and it has the property that matters for #45: a
//! resident model's anonymous pages are not reclaimable, so they correctly
//! count as used and a second large load is refused.
//!
//! GPU utilization goes through NVML, loaded at RUNTIME via dlopen for the same
//! reason as Windows: a box with the CUDA build always has libnvidia-ml.so.1
//! beside the driver, but link-time dependence would make the server refuse to
//! start without one. A missing/failed NVML reports 0, matching the
//! "unavailable" answer the macOS IOKit path gives.

const std = @import("std");

// ── procfs helpers ─────────────────────────────────────────────────────────

/// Read a small procfs/sysfs file into `buf`. Everything here is a synthetic
/// file the kernel renders on read, so a short read is the whole file and the
/// sizes are bounded (a few KB at most).
///
/// This deliberately uses the raw syscalls rather than `std.Io`: the probes are
/// called from the status thread and from the load gate, neither of which
/// carries an `Io` handle — the same constraint that shapes platform.zig.
fn readProcFile(path: [*:0]const u8, buf: []u8) ?[]const u8 {
    const fd = std.os.linux.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
    if (@as(isize, @bitCast(fd)) < 0) return null;
    const ifd: i32 = @intCast(fd);
    defer _ = std.os.linux.close(ifd);

    var total: usize = 0;
    while (total < buf.len) {
        const n = std.os.linux.read(ifd, buf[total..].ptr, buf.len - total);
        const in: isize = @bitCast(n);
        if (in < 0) return null;
        if (in == 0) break;
        total += @intCast(in);
    }
    return buf[0..total];
}

/// Value of a `Key:  <number> kB`-style line, in bytes. Used for both
/// /proc/meminfo and /proc/self/status, which share the format.
///
/// The unit is part of the KEY's contract, not a global: /proc/meminfo is kB
/// throughout, but a sibling file with a bare-number line would silently be
/// scaled by 1024 if this helper were reused blindly — hence `scale` is
/// explicit at each call site rather than baked in here.
fn fieldValue(text: []const u8, key: []const u8, scale: u64) ?u64 {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, key)) continue;
        // Match the key at a boundary: "MemFree" must not answer for
        // "MemFreeFoo", and "Vm" prefixes collide readily in /proc/self/status.
        if (line.len <= key.len or line[key.len] != ':') continue;
        const rest = std.mem.trimStart(u8, line[key.len + 1 ..], " \t");
        var end: usize = 0;
        while (end < rest.len and rest[end] >= '0' and rest[end] <= '9') end += 1;
        if (end == 0) return null;
        const n = std.fmt.parseInt(u64, rest[0..end], 10) catch return null;
        return n * scale;
    }
    return null;
}

fn meminfoField(key: []const u8) ?u64 {
    var buf: [4096]u8 = undefined;
    const text = readProcFile("/proc/meminfo", &buf) orelse return null;
    return fieldValue(text, key, 1024);
}

fn selfStatusField(key: []const u8) ?u64 {
    var buf: [4096]u8 = undefined;
    const text = readProcFile("/proc/self/status", &buf) orelse return null;
    return fieldValue(text, key, 1024);
}

// ── Memory ─────────────────────────────────────────────────────────────────

/// Resident set: VmRSS is the direct analogue of macOS's `resident_size`.
pub fn getAppRssMb() u32 {
    const rss = selfStatusField("VmRSS") orelse return 0;
    return @intCast(rss / (1024 * 1024));
}

/// macOS reports a `phys_footprint` that includes memory the compressor holds
/// on the process's behalf — pages it still owns but that are not resident.
/// The Linux analogue is VmRSS + VmSwap: swapped-out anonymous pages are
/// exactly the pages this process still owns and would have to fault back in.
/// VmSwap is absent on a kernel built without swap, which is a zero, not a
/// failure — so only VmRSS is required.
pub fn getAppMemFootprintMb() u32 {
    const rss = selfStatusField("VmRSS") orelse return 0;
    const swap = selfStatusField("VmSwap") orelse 0;
    return @intCast((rss + swap) / (1024 * 1024));
}

pub fn getTotalMemBytes() u64 {
    return meminfoField("MemTotal") orelse 0;
}

/// The kernel's own estimate of what a new allocation can have without
/// swapping. See the file header for why this is `MemAvailable` and not
/// `MemFree` or `sysinfo().freeram`.
pub fn getAvailableMemBytes() u64 {
    return meminfoField("MemAvailable") orelse 0;
}

/// Bytes still available to THIS process's cgroup, or 0 when no ceiling
/// applies.
///
/// The caller (`scheduler.effectiveAvailableBytes`) lets a nonzero answer
/// REPLACE the host figure, so this must return 0 whenever the concept does not
/// apply — an unlimited cgroup, cgroup v1, or an unreadable hierarchy — exactly
/// as macOS does for jetsam. It is not "how much is free": under a container
/// memory limit the host's MemAvailable is a number this process can never
/// reach, and a load gated on it OOM-kills the container instead of refusing.
///
/// `memory.current` includes the cgroup's page cache, which the kernel reclaims
/// under pressure rather than OOM-killing for. Counting it as used would
/// reintroduce the file-cache error the macOS path exists to avoid, so the
/// reclaimable set is subtracted back out — the same rule, one layer down.
pub fn getProcAvailableMemBytes() u64 {
    var cg_buf: [512]u8 = undefined;
    const cg_text = readProcFile("/proc/self/cgroup", &cg_buf) orelse return 0;
    // cgroup v2 is a single "0::<path>" line. A v1 hierarchy has numbered
    // controllers and no unified memory.max; we do not attempt it.
    const path = blk: {
        var lines = std.mem.splitScalar(u8, cg_text, '\n');
        while (lines.next()) |line| {
            if (std.mem.startsWith(u8, line, "0::")) break :blk std.mem.trimEnd(u8, line[3..], " \t\r");
        }
        return 0;
    };
    if (path.len == 0) return 0;

    var path_buf: [512]u8 = undefined;
    const max_path = std.fmt.bufPrintSentinel(&path_buf, "/sys/fs/cgroup{s}/memory.max", .{path}, 0) catch return 0;
    var val_buf: [64]u8 = undefined;
    const max_text = readProcFile(max_path, &val_buf) orelse return 0;
    const max_trimmed = std.mem.trim(u8, max_text, " \n\t\r");
    // "max" is the common case outside a container: no ceiling, so no answer.
    const limit = std.fmt.parseInt(u64, max_trimmed, 10) catch return 0;

    const cur_path = std.fmt.bufPrintSentinel(&path_buf, "/sys/fs/cgroup{s}/memory.current", .{path}, 0) catch return 0;
    var cur_buf: [64]u8 = undefined;
    const cur_text = readProcFile(cur_path, &cur_buf) orelse return 0;
    const current = std.fmt.parseInt(u64, std.mem.trim(u8, cur_text, " \n\t\r"), 10) catch return 0;

    const stat_path = std.fmt.bufPrintSentinel(&path_buf, "/sys/fs/cgroup{s}/memory.stat", .{path}, 0) catch return 0;
    var stat_buf: [8192]u8 = undefined;
    const reclaimable: u64 = if (readProcFile(stat_path, &stat_buf)) |stat_text|
        (statField(stat_text, "inactive_file") orelse 0) +
            (statField(stat_text, "active_file") orelse 0) +
            (statField(stat_text, "slab_reclaimable") orelse 0)
    else
        0;

    return cgroupHeadroom(limit, current, reclaimable);
}

/// A `key value` line from memory.stat (space-separated, bytes, no unit
/// suffix — hence not `fieldValue`, which expects `Key:` and kB).
fn statField(text: []const u8, key: []const u8) ?u64 {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, key)) continue;
        if (line.len <= key.len or line[key.len] != ' ') continue;
        return std.fmt.parseInt(u64, std.mem.trim(u8, line[key.len + 1 ..], " \t\r"), 10) catch null;
    }
    return null;
}

/// Pure: headroom under a cgroup ceiling, with the reclaimable page cache
/// counted as free rather than used. Saturating throughout — a momentarily
/// inconsistent trio must never wrap into a huge "available".
fn cgroupHeadroom(limit: u64, current: u64, reclaimable: u64) u64 {
    const used = current -| reclaimable;
    return limit -| used;
}

pub fn getSysMemPct() u32 {
    const total = meminfoField("MemTotal") orelse return 0;
    if (total == 0) return 0;
    const avail = meminfoField("MemAvailable") orelse return 0;
    const used = total -| avail;
    return @intCast(used * 100 / total);
}

// ── CPU ────────────────────────────────────────────────────────────────────

var prev_total: u64 = 0;
var prev_idle: u64 = 0;

/// System-wide CPU percentage since the previous call, matching the macOS and
/// Windows tick-delta approach (the first call has no baseline and reports 0
/// rather than a meaningless since-boot average).
///
/// iowait counts as idle: the CPU is not executing anything during it, and
/// treating a model streaming off disk as 100% CPU busy would misreport exactly
/// the phase people watch the status bar during.
pub fn getCpuPct() u32 {
    var buf: [512]u8 = undefined;
    const text = readProcFile("/proc/stat", &buf) orelse return 0;
    const nl = std.mem.indexOfScalar(u8, text, '\n') orelse text.len;
    const line = text[0..nl];
    if (!std.mem.startsWith(u8, line, "cpu ")) return 0;

    var total: u64 = 0;
    var idle: u64 = 0;
    var it = std.mem.tokenizeScalar(u8, line[4..], ' ');
    var i: usize = 0;
    while (it.next()) |tok| : (i += 1) {
        const v = std.fmt.parseInt(u64, tok, 10) catch break;
        total += v;
        // Fields are user, nice, system, idle, iowait, irq, softirq, steal, ...
        if (i == 3 or i == 4) idle += v;
    }
    if (total == 0) return 0;

    const d_total = total -| prev_total;
    const d_idle = idle -| prev_idle;
    prev_total = total;
    prev_idle = idle;
    if (d_total == 0) return 0;
    return @intCast((d_total -| d_idle) * 100 / d_total);
}

// ── GPU (NVML) ─────────────────────────────────────────────────────────────

const NvmlUtilization = extern struct { gpu: u32, memory: u32 };

var nvml_tried: bool = false;
var nvml_ok: bool = false;
var nvml_device: ?*anyopaque = null;
var nvmlDeviceGetUtilizationRates: ?*const fn (?*anyopaque, *NvmlUtilization) callconv(.c) c_int = null;

/// One-shot NVML bring-up. Failure is permanent for the process lifetime — a
/// driver does not appear mid-run, and retrying per sample would put a dlopen
/// on a 500 ms timer.
fn nvmlInit() void {
    nvml_tried = true;
    // SONAME, not the -dev symlink: the development package that provides a
    // bare "libnvidia-ml.so" is not installed on a machine that merely RUNS the
    // driver, which is every deployment target.
    const lib = std.c.dlopen("libnvidia-ml.so.1", .{ .LAZY = true }) orelse return;

    const init_fn: *const fn () callconv(.c) c_int =
        @ptrCast(std.c.dlsym(lib, "nvmlInit_v2") orelse return);
    if (init_fn() != 0) return;

    const by_index: *const fn (c_uint, *?*anyopaque) callconv(.c) c_int =
        @ptrCast(std.c.dlsym(lib, "nvmlDeviceGetHandleByIndex_v2") orelse return);
    var dev: ?*anyopaque = null;
    // Device 0: the server pins no device, and llama.cpp defaults to 0 as well.
    if (by_index(0, &dev) != 0) return;

    nvmlDeviceGetUtilizationRates = @ptrCast(std.c.dlsym(lib, "nvmlDeviceGetUtilizationRates") orelse return);
    nvml_device = dev;
    nvml_ok = true;
}

pub fn getGpuPct() u32 {
    if (!nvml_tried) nvmlInit();
    if (!nvml_ok) return 0;
    const f = nvmlDeviceGetUtilizationRates orelse return 0;
    var util: NvmlUtilization = std.mem.zeroes(NvmlUtilization);
    if (f(nvml_device, &util) != 0) return 0;
    return util.gpu;
}

// ── Tests ──────────────────────────────────────────────────────────────────

test "fieldValue reads kB lines and matches keys at a boundary" {
    const meminfo =
        \\MemTotal:       15988360 kB
        \\MemFree:         9945832 kB
        \\MemAvailable:   10378476 kB
        \\
    ;
    try std.testing.expectEqual(@as(?u64, 15988360 * 1024), fieldValue(meminfo, "MemTotal", 1024));
    try std.testing.expectEqual(@as(?u64, 10378476 * 1024), fieldValue(meminfo, "MemAvailable", 1024));
    // A prefix of a longer key must not answer: "MemFree" is a prefix of
    // "MemFreeFoo", and /proc/self/status is full of "Vm*" siblings.
    try std.testing.expectEqual(@as(?u64, null), fieldValue("MemFreeFoo: 12 kB\n", "MemFree", 1024));
    try std.testing.expectEqual(@as(?u64, null), fieldValue(meminfo, "VmRSS", 1024));
    // A bare-number file (scale 1) must not be silently scaled by kB.
    try std.testing.expectEqual(@as(?u64, 42), fieldValue("Thing: 42\n", "Thing", 1));
}

test "statField reads space-separated byte lines" {
    const stat =
        \\anon 1048576
        \\file 2097152
        \\inactive_file 198664192
        \\slab_reclaimable 41609072
        \\
    ;
    try std.testing.expectEqual(@as(?u64, 198664192), statField(stat, "inactive_file"));
    // "file" must not answer for "inactive_file", and vice versa.
    try std.testing.expectEqual(@as(?u64, 2097152), statField(stat, "file"));
    try std.testing.expectEqual(@as(?u64, null), statField(stat, "active_file"));
}

test "cgroupHeadroom counts reclaimable cache as free and never wraps" {
    const GB: u64 = 1024 * 1024 * 1024;
    // 8 GB limit, 5 GB charged of which 2 GB is page cache → 5 GB headroom.
    try std.testing.expectEqual(5 * GB, cgroupHeadroom(8 * GB, 5 * GB, 2 * GB));
    // No cache: the charge stands.
    try std.testing.expectEqual(3 * GB, cgroupHeadroom(8 * GB, 5 * GB, 0));
    // Over the limit, or momentarily inconsistent counters, saturate to 0/limit
    // rather than wrapping into a huge "available" that would wave a load
    // through the gate.
    try std.testing.expectEqual(@as(u64, 0), cgroupHeadroom(4 * GB, 9 * GB, 0));
    try std.testing.expectEqual(8 * GB, cgroupHeadroom(8 * GB, 1 * GB, 5 * GB));
}

test "live probes return plausible values on this host" {
    // These read real procfs. A wrong key or a missing boundary check shows up
    // as 0 or as garbage, both of which the load gate would act on.
    const total = getTotalMemBytes();
    try std.testing.expect(total > 256 * 1024 * 1024); // no Linux box we target has less
    try std.testing.expect(total < 64 * 1024 * 1024 * 1024 * 1024); // < 64 TB sanity bound

    const avail = getAvailableMemBytes();
    try std.testing.expect(avail > 0);
    try std.testing.expect(avail <= total);

    const rss = getAppRssMb();
    try std.testing.expect(rss > 0); // the test process itself is resident
    try std.testing.expect(getAppMemFootprintMb() >= rss); // footprint = rss + swap

    try std.testing.expect(getSysMemPct() <= 100);
    // First call establishes the tick baseline and reports 0; the second is a
    // real delta. Both must stay in range.
    _ = getCpuPct();
    try std.testing.expect(getCpuPct() <= 100);
    try std.testing.expect(getGpuPct() <= 100); // 0 when there is no NVML
}
