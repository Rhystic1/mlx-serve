// Test root — imports all modules to run their embedded tests.
// Run with: zig build test
//
// This is the PORTABLE root: everything
// whose tests are hermetic with respect to the inference backend: templating,
// tool-call parsing, the HTTP surfaces, discovery, tokenization, and the
// pure-Zig mesh/image helpers.
//
// The MLX-backed roots live in src/tests_all.zig, which build.zig selects as
// the test root INSTEAD of this file when MLX is compiled in. They are
// selected by FILE rather than by a comptime `if` because a dead
// `if (cond) _ = @import(x)` branch still registers x's tests -- the test
// collector works on the reference graph, not on which branch survives
// analysis -- so the MLX imports must not appear in this file at all.
//
// Keeping the portable list explicit (rather than deriving it by subtraction)
// is deliberate: a new module lands in exactly one list on purpose, and a
// module that quietly grows an MLX dependency breaks the Windows/Linux build
// at compile time instead of at run time.

const build_cfg = @import("build_cfg.zig");

test {
    // ── Portable: no MLX, no Metal, no Apple frameworks ──
    _ = @import("log.zig");
    _ = @import("version.zig");
    _ = @import("platform.zig");
    _ = if (build_cfg.mlx_enabled) struct {} else @import("transformer_stub.zig");
    _ = @import("chat.zig");
    _ = @import("format_corpus_test.zig");
    _ = @import("tool_traffic_replay_test.zig");
    _ = @import("regex.zig");
    _ = @import("json_schema.zig");
    _ = @import("json_grammar.zig");
    _ = @import("token_mask.zig");
    _ = @import("responses.zig");
    _ = @import("ws.zig");
    _ = @import("pld_index.zig");
    _ = @import("loop_detect.zig");
    _ = @import("tokenizer.zig");
    _ = @import("metrics.zig");
    _ = @import("status.zig");
    _ = @import("model_discovery.zig");
    _ = @import("gguf_meta.zig");
    _ = @import("llama_ffi.zig");
    _ = @import("wav.zig");
    _ = @import("png.zig");
    _ = @import("marching_cubes.zig");
    _ = @import("glb.zig");
    _ = @import("uvwrap.zig");
    _ = @import("rasterize.zig");
    _ = @import("texinpaint.zig");
    _ = @import("multipart.zig");
    _ = @import("gen_sse.zig");
    _ = @import("ollama.zig");
    _ = @import("cli.zig");
    _ = @import("launch.zig");
    _ = @import("lan.zig");
    _ = @import("model_registry.zig");
    _ = @import("scheduler.zig");
    _ = @import("server.zig");

    // ── MLX-backed: the native transformer, spec decode, media generation ──

    _ = if (build_cfg.llama_enabled) @import("arch/llama.zig") else struct {};
}
