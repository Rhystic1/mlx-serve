//! ds4 engine test roots. See src/tests_mlx.zig for why this is a separate
//! file rather than an `if` block.

test {
    _ = @import("ds4_ffi.zig");
    _ = @import("arch/ds4.zig");
}
