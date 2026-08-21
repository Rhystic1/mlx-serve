//! Windows implementation of the process/system stat probes in status.zig.
//!
//! These are NOT cosmetic. `getTotalMemBytes` / `getAvailableMemBytes` feed the
//! model-load admission gate and the auto-context sizer, so returning zero here
//! would not merely blank a panel — it would make every load decision on wrong
//! numbers. That is why this is a real implementation rather than a stub.
//!
//! GPU utilization goes through NVML, loaded at RUNTIME via LoadLibrary: a
//! machine with the CUDA build installed always has nvml.dll beside the driver,
//! but link-time dependence would make the server refuse to start without a
//! driver. A missing/failed NVML reports 0, matching the "unavailable" answer
//! the macOS IOKit path gives.

const std = @import("std");
const windows = std.os.windows;

const HANDLE = windows.HANDLE;
const DWORD = windows.DWORD;
const BOOL = c_int;

// ── Memory ─────────────────────────────────────────────────────────────────

const MEMORYSTATUSEX = extern struct {
    dwLength: DWORD,
    dwMemoryLoad: DWORD,
    ullTotalPhys: u64,
    ullAvailPhys: u64,
    ullTotalPageFile: u64,
    ullAvailPageFile: u64,
    ullTotalVirtual: u64,
    ullAvailVirtual: u64,
    ullAvailExtendedVirtual: u64,
};

const PROCESS_MEMORY_COUNTERS_EX = extern struct {
    cb: DWORD,
    PageFaultCount: DWORD,
    PeakWorkingSetSize: usize,
    WorkingSetSize: usize,
    QuotaPeakPagedPoolUsage: usize,
    QuotaPagedPoolUsage: usize,
    QuotaPeakNonPagedPoolUsage: usize,
    QuotaNonPagedPoolUsage: usize,
    PagefileUsage: usize,
    PeakPagefileUsage: usize,
    PrivateUsage: usize,
};

const FILETIME = extern struct { low: DWORD, high: DWORD };

extern "kernel32" fn GlobalMemoryStatusEx(buffer: *MEMORYSTATUSEX) callconv(.winapi) BOOL;
extern "kernel32" fn GetCurrentProcess() callconv(.winapi) HANDLE;
extern "kernel32" fn GetSystemTimes(idle: *FILETIME, kernel: *FILETIME, user: *FILETIME) callconv(.winapi) BOOL;
// K32-prefixed so it resolves from kernel32 directly rather than needing psapi.
extern "kernel32" fn K32GetProcessMemoryInfo(
    process: HANDLE,
    counters: *PROCESS_MEMORY_COUNTERS_EX,
    cb: DWORD,
) callconv(.winapi) BOOL;

fn memStatus() ?MEMORYSTATUSEX {
    var ms: MEMORYSTATUSEX = std.mem.zeroes(MEMORYSTATUSEX);
    ms.dwLength = @sizeOf(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&ms) == 0) return null;
    return ms;
}

fn procMem() ?PROCESS_MEMORY_COUNTERS_EX {
    var pmc: PROCESS_MEMORY_COUNTERS_EX = std.mem.zeroes(PROCESS_MEMORY_COUNTERS_EX);
    pmc.cb = @sizeOf(PROCESS_MEMORY_COUNTERS_EX);
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, pmc.cb) == 0) return null;
    return pmc;
}

/// Resident set: the Windows working set is the direct analogue of macOS's
/// `resident_size`.
pub fn getAppRssMb() u32 {
    const pmc = procMem() orelse return 0;
    return @intCast(pmc.WorkingSetSize / (1024 * 1024));
}

/// macOS reports a "phys_footprint" that includes compressed memory; the
/// closest Windows equivalent is private commit (`PrivateUsage`), which counts
/// pages this process owns whether or not they are currently resident.
pub fn getAppMemFootprintMb() u32 {
    const pmc = procMem() orelse return 0;
    return @intCast(pmc.PrivateUsage / (1024 * 1024));
}

pub fn getTotalMemBytes() u64 {
    const ms = memStatus() orelse return 0;
    return ms.ullTotalPhys;
}

pub fn getAvailableMemBytes() u64 {
    const ms = memStatus() orelse return 0;
    return ms.ullAvailPhys;
}

/// What THIS process can still commit. Windows enforces a per-process address
/// space and a system commit limit; the binding one for a large model load is
/// whichever is smaller.
pub fn getProcAvailableMemBytes() u64 {
    const ms = memStatus() orelse return 0;
    return @min(ms.ullAvailPhys, ms.ullAvailVirtual);
}

pub fn getSysMemPct() u32 {
    const ms = memStatus() orelse return 0;
    return @intCast(ms.dwMemoryLoad);
}

// ── CPU ────────────────────────────────────────────────────────────────────

fn ftU64(ft: FILETIME) u64 {
    return (@as(u64, ft.high) << 32) | @as(u64, ft.low);
}

var prev_idle: u64 = 0;
var prev_busy: u64 = 0;

/// System-wide CPU percentage since the previous call, matching the macOS
/// tick-delta approach (the first call after boot has no baseline and reports
/// 0 rather than a meaningless since-boot average).
pub fn getCpuPct() u32 {
    var idle: FILETIME = undefined;
    var kernel: FILETIME = undefined;
    var user: FILETIME = undefined;
    if (GetSystemTimes(&idle, &kernel, &user) == 0) return 0;

    // Windows counts idle time INSIDE kernel time, so busy = kernel+user-idle.
    const idle_t = ftU64(idle);
    const total_t = ftU64(kernel) + ftU64(user);
    const busy_t = total_t -| idle_t;

    const d_idle = idle_t -| prev_idle;
    const d_busy = busy_t -| prev_busy;
    prev_idle = idle_t;
    prev_busy = busy_t;

    const denom = d_idle + d_busy;
    if (denom == 0) return 0;
    return @intCast(d_busy * 100 / denom);
}

// ── GPU (NVML) ─────────────────────────────────────────────────────────────

const NvmlUtilization = extern struct { gpu: u32, memory: u32 };

var nvml_tried: bool = false;
var nvml_ok: bool = false;
var nvml_device: ?*anyopaque = null;
var nvmlDeviceGetUtilizationRates: ?*const fn (?*anyopaque, *NvmlUtilization) callconv(.winapi) c_int = null;

extern "kernel32" fn LoadLibraryA(name: [*:0]const u8) callconv(.winapi) ?windows.HMODULE;
extern "kernel32" fn GetProcAddress(module: windows.HMODULE, name: [*:0]const u8) callconv(.winapi) ?*anyopaque;

/// One-shot NVML bring-up. Failure is permanent for the process lifetime — a
/// driver does not appear mid-run, and retrying per sample would put a
/// LoadLibrary on a 500 ms timer.
fn nvmlInit() void {
    nvml_tried = true;
    const lib = LoadLibraryA("nvml.dll") orelse return;

    const init_fn: *const fn () callconv(.winapi) c_int =
        @ptrCast(GetProcAddress(lib, "nvmlInit_v2") orelse return);
    if (init_fn() != 0) return;

    const by_index: *const fn (c_uint, *?*anyopaque) callconv(.winapi) c_int =
        @ptrCast(GetProcAddress(lib, "nvmlDeviceGetHandleByIndex_v2") orelse return);
    var dev: ?*anyopaque = null;
    // Device 0: the server pins no device, and llama.cpp defaults to 0 as well.
    if (by_index(0, &dev) != 0) return;

    nvmlDeviceGetUtilizationRates = @ptrCast(GetProcAddress(lib, "nvmlDeviceGetUtilizationRates") orelse return);
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
