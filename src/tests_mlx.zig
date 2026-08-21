//! MLX-backed test roots: the native transformer, speculative decode, and
//! every media-generation backend. Referenced from src/tests.zig only when
//! `build_cfg.mlx_enabled`.
//!
//! Separate FILE rather than an `if` block inside tests.zig: a statement-level
//! `if (cond) { _ = @import(...) }` still registers the imported file's tests
//! when the branch is dead, which drags the whole MLX surface into a
//! -Dgguf-only test binary and fails at link. Selecting a module with the
//! `if/else` expression form is what actually excludes it.

test {
    _ = @import("mlx.zig");
    _ = @import("model.zig");
    _ = @import("generate.zig");
    _ = @import("transformer.zig");
    _ = @import("vision.zig");
    _ = @import("qwen_vision.zig");
    _ = @import("qwen_vision_parity_test.zig");
    _ = @import("muse_vision.zig");
    _ = @import("muse_vision_parity_test.zig");
    _ = @import("lfm2_vision.zig");
    _ = @import("lfm2_vision_parity_test.zig");
    _ = @import("mrope.zig");
    _ = @import("kv_quant.zig");
    _ = @import("drafter.zig");
    _ = @import("dflash.zig");
    _ = @import("mtp.zig");
    _ = @import("diffusion.zig");
    _ = @import("deepseek_v4.zig");
    _ = @import("kokoro.zig");
    _ = @import("kokoro_g2p.zig");
    _ = @import("prefix_cache.zig");
    _ = @import("kv_disk_cache.zig");
    _ = @import("tts.zig");
    _ = @import("flux.zig");
    _ = @import("krea.zig");
    _ = @import("mage_flow.zig");
    _ = @import("lora.zig");
    _ = @import("nsfw.zig");
    _ = @import("ltx_video.zig");
    _ = @import("ltx_diffvae.zig");
    _ = @import("ltx_diffvae_kernel.zig");
    _ = @import("ltx_diffvae_forward.zig");
    _ = @import("ltx_audio.zig");
    _ = @import("minimax_h3.zig");
    _ = @import("minimax_h3_vision.zig");
    _ = @import("minimax_h3_vae.zig");
    _ = @import("minimax_h3_audio.zig");
    _ = @import("hunyuan3d.zig");
    _ = @import("acestep.zig");
    _ = @import("music3.zig");
    _ = @import("hunyuan3d_paint.zig");
    _ = @import("hunyuan3d_paint_unet.zig");
    _ = @import("bake.zig");
    _ = @import("gen.zig");
}
