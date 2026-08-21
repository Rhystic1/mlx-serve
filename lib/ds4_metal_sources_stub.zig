// Stub for builds without the ds4 engine (see src/build_cfg.zig).
//
// The real module @embedFiles ~19 .metal kernel sources out of the lib/ds4
// submodule. Those files exist only when that submodule is checked out, and
// they are Metal shaders regardless -- so a GGUF-only or iOS build cannot
// reference them at all. Same const names, empty bodies: `src/arch/ds4.zig`
// is itself stubbed in those builds, so nothing ever reads these.

pub const argsort: []const u8 = "";
pub const bin: []const u8 = "";
pub const concat: []const u8 = "";
pub const cpy: []const u8 = "";
pub const dense: []const u8 = "";
pub const dsv4_hc: []const u8 = "";
pub const dsv4_kv: []const u8 = "";
pub const dsv4_misc: []const u8 = "";
pub const dsv4_rope: []const u8 = "";
pub const flash_attn: []const u8 = "";
pub const get_rows: []const u8 = "";
pub const glu: []const u8 = "";
pub const moe: []const u8 = "";
pub const norm: []const u8 = "";
pub const repeat: []const u8 = "";
pub const set_rows: []const u8 = "";
pub const softmax: []const u8 = "";
pub const sum_rows: []const u8 = "";
pub const unary: []const u8 = "";
