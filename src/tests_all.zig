//! Full test root: the portable suite plus every backend-specific one.
//!
//! build.zig picks this as the test root when MLX is compiled in, and
//! src/tests.zig (portable only) otherwise. Selecting by FILE is what actually
//! excludes the MLX suites: a dead `if (cond) _ = @import(x)` branch still
//! registers x's tests, because the collector walks the reference graph rather
//! than the post-analysis one.

const build_cfg = @import("build_cfg.zig");

test {
    _ = @import("tests.zig");
    _ = @import("tests_mlx.zig");
    _ = if (build_cfg.ds4_enabled) @import("tests_ds4.zig") else struct {};
    _ = if (build_cfg.ane_enabled) @import("tests_ane.zig") else struct {};
}
