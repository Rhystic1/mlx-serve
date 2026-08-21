//! Media-model classification and residency estimation.
//!
//! The MLX-free half of gen.zig, split out for the Windows/Linux port. These
//! are the parts a GGUF-only build still needs: `model_discovery.zig` has to
//! recognise an image/audio/video/3D pack so the server can refuse it BY NAME
//! rather than routing it into the text loader, which is the "incomplete media
//! pack" failure class (a marker-less dir otherwise falls through and dies on
//! the first missing weight). The residency estimators are pure directory
//! arithmetic and feed the same admission gate.
//!
//! `gen.zig` (engines) and `gen_stub.zig` both re-export everything here, so
//! callers see one flat API either way.

const std = @import("std");
const log = @import("log.zig");
const tok_mod = @import("tokenizer.zig");
const model_mod = @import("model.zig");
const chat_mod = @import("chat.zig");
const discovery = @import("model_discovery.zig");
const platform = @import("platform.zig");

pub const Modality = enum {
    image,
    audio,
    video,
    mesh,

    pub fn capability(self: Modality) []const u8 {
        return switch (self) {
            .image => "image",
            .audio => "audio",
            .video => "video",
            .mesh => "3d",
        };
    }

    /// Static, borrowed-static `ModelConfig.model_type` marker for each
    /// modality. Stable string literals (never freed) — `ModelConfig`
    /// treats `model_type` as borrowed-static, so a heap dupe is wrong here.
    pub fn modelType(self: Modality) []const u8 {
        return switch (self) {
            .image => "flux2",
            .audio => "qwen3_tts",
            .video => "AudioVideo",
            .mesh => "hunyuan3d_2_1",
        };
    }
};

/// Classify a `model_type` string into a media modality, or null for a
/// regular LM/embedding arch. Pure — the load arms dispatch on this off the
/// (stub) config's `model_type`, so it must accept the markers from
/// `Modality.modelType` AND the raw config strings discovery peeks
/// ("flux2-klein-4b", "qwen3_tts", "AudioVideo").
/// Every media `model_type` this server serves. The ONE list the two
/// duplicated predicates below are checked against.
///
/// `model_discovery.isMediaModelType` cannot call `modalityFromType` — that
/// module stays filesystem-only so it never pulls in mlx — so the duplication
/// is deliberate and documented. What was missing was a guard: `minimax_h3`
/// was registered here and NOT there, so discovery rejected the model with
/// "unsupported model_type" while the engine that serves it was ready and
/// waiting. The test at the bottom of this file pins them together.
pub const media_model_types = [_][]const u8{
    "flux2",     "krea",       "mage_flow",      "mageflow",
    "qwen3_tts", "acestep",    "kokoro",         "AudioVideo",
    "hunyuan3d", "minimax_h3", "minimax_music3",
};

pub fn modalityFromType(model_type: []const u8) ?Modality {
    if (std.mem.startsWith(u8, model_type, "flux2")) return .image;
    if (std.mem.startsWith(u8, model_type, "krea")) return .image;
    if (std.mem.startsWith(u8, model_type, "mage_flow") or std.mem.eql(u8, model_type, "mageflow")) return .image;
    if (std.mem.eql(u8, model_type, "qwen3_tts")) return .audio;
    if (std.mem.eql(u8, model_type, "acestep")) return .audio;
    if (std.mem.eql(u8, model_type, "minimax_music3")) return .audio;
    if (std.mem.eql(u8, model_type, "kokoro")) return .audio;
    if (std.mem.eql(u8, model_type, "AudioVideo")) return .video;
    if (std.mem.eql(u8, model_type, "minimax_h3")) return .video;
    if (std.mem.startsWith(u8, model_type, "hunyuan3d")) return .mesh;
    return null;
}

/// Endpoint-level media route. `.speech` and `.music` share the `.audio`
/// modality/engine slot — the loaded `AudioBackend` arm decides which endpoint
/// is valid (wrong pairing → explicit 400, never a silent misinterpretation).
pub const GenRoute = enum {
    image,
    speech,
    music,
    video,
    mesh,

    pub fn modality(self: GenRoute) Modality {
        return switch (self) {
            .image => .image,
            .speech, .music => .audio,
            .video => .video,
            .mesh => .mesh,
        };
    }
};

/// Which audio backend a `model_type` selects (pure; pins the dispatch the
/// `AudioEngine.load` re-peek performs).
pub fn audioBackendKindForType(model_type: []const u8) AudioBackendKind {
    if (std.mem.eql(u8, model_type, "acestep")) return .music;
    if (std.mem.eql(u8, model_type, "minimax_music3")) return .music3;
    if (std.mem.eql(u8, model_type, "kokoro")) return .kokoro;
    return .tts;
}

/// Which arm of `AudioBackend` a checkpoint loads into. `.tts` (Qwen3-TTS) and
/// `.kokoro` both serve `/v1/audio/speech` but have DISJOINT controls: Qwen3-TTS
/// clones from `ref_audio` and has no voice list, Kokoro has 54 named blendable
/// voices and no cloning. The handler refuses the wrong control rather than
/// ignoring it.
pub const AudioBackendKind = enum {
    tts,
    music,
    music3,
    kokoro,

    /// Music-generation backends serve /v1/audio/music-generations and
    /// advertise "music" beside "audio"; the TTS arms never do.
    pub fn servesMusic(self: AudioBackendKind) bool {
        return self == .music or self == .music3;
    }
};

/// Peek `model_dir/config.json` for its `model_type` string (owned dupe, caller
/// frees) or null on any read/parse error. Cheap — used both to route to a media
/// modality and to pick the image backend (FLUX vs Krea).
pub fn peekModelType(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) ?[]u8 {
    // Guard the openFileAbsolute assert (ReleaseFast UB on relative/empty paths).
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return null;
    if (readConfigModelType(io, allocator, model_dir)) |mt| return mt;
    // Diffusers-style repos (Mage-Flow) have no root config.json / model_type —
    // the pipeline identity lives in model_index.json's `_class_name`. Synthesize
    // the "mage_flow" marker so routing + the backend dispatch light up.
    if (isMageFlowRepo(io, allocator, model_dir)) return allocator.dupe(u8, "mage_flow") catch null;
    // Same for an mflux FLUX.2 conversion with no config.json at all (the only
    // MLX build of klein 9B). Identified by the DiT's own weight names, through
    // the SAME predicate discovery uses — a private copy here is how `list` and
    // the loader end up disagreeing about whether a dir is a model.
    if (isMfluxFlux2Repo(io, allocator, model_dir)) return allocator.dupe(u8, "flux2-klein") catch null;
    return null;
}

/// True when `model_dir` holds FLUX.2 DiT weights but no config.json to say so.
/// Thin path→Dir wrapper over `model_discovery.peekMfluxFlux2`.
fn isMfluxFlux2Repo(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) bool {
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{}) catch return false;
    defer dir.close(io);
    return discovery.peekMfluxFlux2(io, allocator, dir);
}

/// Read `model_dir/config.json`'s `model_type` (owned dupe) or null on any
/// read/parse error or when the field is absent.
fn readConfigModelType(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) ?[]u8 {
    const path = std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir}) catch return null;
    defer allocator.free(path);
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return null;
    defer file.close(io);
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const content = rs.interface.allocRemaining(allocator, .limited(4 * 1024 * 1024)) catch return null;
    defer allocator.free(content);
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const mt = parsed.value.object.get("model_type") orelse return null;
    if (mt != .string) return null;
    return allocator.dupe(u8, mt.string) catch null;
}

/// True when `model_dir/model_index.json` marks a MageFlow pipeline (its
/// `_class_name` is "MageFlowPipeline", or the `_mage_flow_version` tag exists).
fn isMageFlowRepo(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) bool {
    const path = std.fmt.allocPrint(allocator, "{s}/model_index.json", .{model_dir}) catch return false;
    defer allocator.free(path);
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return false;
    defer file.close(io);
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const content = rs.interface.allocRemaining(allocator, .limited(1024 * 1024)) catch return false;
    defer allocator.free(content);
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch return false;
    defer parsed.deinit();
    if (parsed.value != .object) return false;
    if (parsed.value.object.get("_mage_flow_version") != null) return true;
    const cn = parsed.value.object.get("_class_name") orelse return false;
    return cn == .string and std.mem.eql(u8, cn.string, "MageFlowPipeline");
}

/// Classify a model dir into a media modality (reads its `model_type`), or null
/// for a regular LM/embedding arch. The video (LTX "AudioVideo") branch
/// additionally requires `connector.safetensors` so a generic "AudioVideo"
/// config without the LTX bundle isn't misrouted.
/// A file that must be present for `model_type` to be accepted as that media
/// backend, or null when the `model_type` alone is sufficient.
///
/// This is keyed on the TYPE, not the modality. It used to be
/// `if (modality == .video) require connector.safetensors` — a marker that
/// belongs to LTX only. The moment a second video backend existed, that guard
/// rejected it, `detectModality` returned null, and the loader fell through to
/// the MLX TEXT path: it globbed all four of MiniMax-H3's safetensors into one
/// weight map and died on `model.norm.weight`. A per-MODALITY guard cannot
/// survive a modality growing a second backend.
pub fn requiredMarkerFor(model_type: []const u8) ?[]const u8 {
    // The table lives in model_discovery (fs-only, so discovery and
    // register-by-path apply the SAME completeness rule) — this module can
    // import that one, just not the other way around.
    return discovery.requiredMediaMarker(model_type);
}

pub fn detectModality(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) ?Modality {
    const mt = peekModelType(io, allocator, model_dir) orelse return null;
    defer allocator.free(mt);
    const modality = modalityFromType(mt) orelse return null;
    if (requiredMarkerFor(mt)) |marker| {
        const p = std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, marker }, 0) catch return null;
        defer allocator.free(p);
        if (!fileExists(io, p)) {
            log.warn("[gen] {s} at {s} is missing {s}; not treating it as a media model\n", .{ mt, model_dir, marker });
            return null;
        }
    }
    return modality;
}

/// True when `model_dir` declares a media `model_type` whose required marker
/// is missing — an incomplete pack (an in-flight or interrupted download, or
/// a stray fragment). The load path refuses these BY NAME: falling through to
/// the text loader globs whatever safetensors ARE present and dies on the
/// first missing weight (`unreachable` in ReleaseFast — live 2026-08-08, a
/// turbo-lora fragment killed the server on a plain Generate).
pub fn incompleteMediaDir(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) bool {
    const mt = peekModelType(io, allocator, model_dir) orelse return false;
    defer allocator.free(mt);
    if (modalityFromType(mt) == null) return false;
    const marker = requiredMarkerFor(mt) orelse return false;
    const p = std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, marker }, 0) catch return false;
    defer allocator.free(p);
    return !fileExists(io, p);
}

// ════════════════════════════════════════════════════════════════════════
// Engine wrappers — own the backend sub-models. Allocated on the heap so the
// `?*Engine` slot on `LoadedModel` is a stable pointer (mirrors `ds4_engine`).
// load() + every generate() run on the inference thread.
// ════════════════════════════════════════════════════════════════════════

pub const PAD_TOKEN_FLUX: i32 = 151643; // Qwen2/3 pad token
pub const FLUX_SEQ_LEN: usize = 512; // mflux Qwen3 tokenizer max_length

pub const StubCpuState = struct {
    config: *model_mod.ModelConfig,
    tok: *tok_mod.Tokenizer,
    chat_config: *chat_mod.ChatConfig,
};

/// Build heap-allocated stub config/tokenizer/chat_config for `modality`.
/// Ownership transfers to the LoadedModel on a successful load (mirrors the
/// ds4/llama stubs). `freeStubCpuState` frees them on the failure path.
pub fn buildStubCpuState(allocator: std.mem.Allocator, modality: Modality) !StubCpuState {
    const config = try allocator.create(model_mod.ModelConfig);
    errdefer allocator.destroy(config);
    config.* = model_mod.ModelConfig{
        .model_type = modality.modelType(),
        .weight_prefix = "model",
        .num_hidden_layers = 1,
        .hidden_size = 1,
        .head_dim = 1,
        .num_attention_heads = 1,
        .num_key_value_heads = 1,
        .max_position_embeddings = 4096,
        .is_encoder_only = false,
    };

    const tok = try allocator.create(tok_mod.Tokenizer);
    errdefer allocator.destroy(tok);
    var byte_map: [256]u21 = undefined;
    var b: usize = 0;
    while (b < 256) : (b += 1) byte_map[b] = @intCast(b);
    tok.* = .{
        .vocab = std.StringHashMap(u32).init(allocator),
        .id_to_token = std.AutoHashMap(u32, []const u8).init(allocator),
        .merge_ranks = @TypeOf(tok.merge_ranks).init(allocator),
        .allocator = allocator,
        .special_tokens = std.StringHashMap(u32).init(allocator),
        .tok_type = .byte_level_bpe,
        .byte_to_unicode = byte_map,
        .unicode_to_byte = std.AutoHashMap(u21, u8).init(allocator),
        .bos_id = null,
        .eos_id = null,
        .parsed_json = null,
    };
    errdefer tok.deinit();

    const cc = try allocator.create(chat_mod.ChatConfig);
    errdefer allocator.destroy(cc);
    cc.* = .{
        .chat_template = try allocator.dupe(u8, ""),
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    return .{ .config = config, .tok = tok, .chat_config = cc };
}

pub fn freeStubCpuState(allocator: std.mem.Allocator, s: *StubCpuState) void {
    allocator.destroy(s.config);
    s.tok.deinit();
    allocator.destroy(s.tok);
    s.chat_config.deinit();
    allocator.destroy(s.chat_config);
}

/// Sum the safetensors footprint of a media model dir for the eviction gate.
/// Walks the top level + one level of subdirs (FLUX keeps weights in
/// transformer/, vae/, text_encoder/; LTX keeps them top-level). Returns 0 on
/// any read failure (treated as "unknown" → the registry skips the byte cap).
pub fn estimateResidentBytes(io: std.Io, model_dir: []const u8) u64 {
    if (model_dir.len == 0 or model_dir[0] != '/') return 0; // openDirAbsolute UB class
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true }) catch return 0;
    defer dir.close(io);
    return sumSafetensorsIn(io, dir);
}

fn sumSafetensorsIn(io: std.Io, dir: std.Io.Dir) u64 {
    // Symlinked weights count (statFile follows) — an HF hub-cache snapshot
    // is ALL symlinks into ../../blobs; skipping them billed a pack at 0.
    var total: u64 = 0;
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if ((entry.kind == .file or entry.kind == .sym_link) and std.mem.endsWith(u8, entry.name, ".safetensors")) {
            const st = dir.statFile(io, entry.name, .{}) catch continue;
            if (st.kind != .file) continue;
            total += @intCast(st.size);
        } else if (entry.kind == .directory) {
            var sub = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
            defer sub.close(io);
            var sit = sub.iterate();
            while (sit.next(io) catch null) |se| {
                if (se.kind != .file and se.kind != .sym_link) continue;
                if (!std.mem.endsWith(u8, se.name, ".safetensors")) continue;
                const st = sub.statFile(io, se.name, .{}) catch continue;
                if (st.kind != .file) continue;
                total += @intCast(st.size);
            }
        }
    }
    return total;
}

/// Peak resident bytes for a backend whose parts do NOT all coexist.
/// `resident` is what the engine holds for its whole lifetime; `stages` are
/// DISJOINT — each is loaded, used and released before the next runs, so only
/// the biggest is ever on top of `resident`. A stage that GENERATES carries its
/// own transients inside its number, because they are not uniform across stages
/// (a text-encoder pass over a few hundred rows allocates nothing like a
/// 124-frame denoise).
///
/// Sum-of-directory is what a backend gets when it declares no plan, and it is
/// the RIGHT answer for the resident-engine backends (krea, mage_flow,
/// hunyuan3d, acestep, tts all hold text-encoder + DiT + VAE on one struct for
/// the engine's lifetime). It goes wrong in BOTH directions the moment a
/// backend loads something it later frees, or reads weights from outside its
/// own directory — see `ltxPeakBytes` for one backend doing each.
pub fn stagedPeakBytes(resident: u64, stages: []const u64) u64 {
    var biggest: u64 = 0;
    for (stages) |st| biggest = @max(biggest, st);
    return resident + biggest;
}

/// LTX's plan. Its engine is RESIDENT (transformer + connector + VAEs + audio
/// stay on `LtxVideoEngine` for its lifetime), with two corrections the
/// directory sum cannot make:
///
///   `spare_transformer` — packs ship `transformer-dev` AND
///   `transformer-distilled` (~10.5 GiB each) and `ensureTransformer` frees one
///   BEFORE loading the other, precisely so they never coexist. The sum bills a
///   phantom.
///
///   `text_encoder` — the Gemma encoder is a SEPARATE shared repo, loaded by
///   `ltx_video.gemmaCapture` per generation and freed when it returns. Being
///   outside the model dir, the sum bills 7.5 GiB at zero, and it is resident
///   on top of the whole engine while it runs.
///
/// No activation term: nothing has been measured for this backend, and
/// inventing one would newly refuse loads that work today.
pub fn ltxPeakBytes(dir_sum: u64, spare_transformer: u64, text_encoder: u64) u64 {
    return stagedPeakBytes(dir_sum -| spare_transformer, &.{text_encoder});
}

/// Percent of `transformer.safetensors` still resident once `precomputeAdaln`
/// has tabled and FREED the 13B modulation weights (~39% of the DiT's
/// parameters, so the share barely moves with quant width). Measured 0.615 on
/// the 8-bit pack (32.83 → 20.19 GiB) and 0.623 on the 4-bit (17.41 → 10.84);
/// billed at 0.65 so a pack whose AdaLN share is smaller than ours still fails
/// safe.
pub const H3_DIT_RESIDENT_PCT: u64 = 65;

/// Transients the two GENERATING stages carry on top of their weights: the
/// packed [text|cond|audio|video] sequence's activations while sampling, and
/// the VAE decode's frame buffers. Measured 4.0-5.0 GiB at 768x448 / 124f
/// (process peak minus self-reported DiT residency, both packs); billed at 6.
/// It scales with pixels x frames, which a per-MODEL load gate cannot see —
/// bounding a specific request is not something this estimator can do, and
/// the old formula's incidental margin was the same order.
///
/// The TEXT-ENCODER stage gets none of it: that is one forward over a few
/// hundred prompt rows, so a shared "+ activations" on the max of all three
/// stages bills the biggest stage for transients it never allocates — which
/// is what refused the 8-bit pack on every Mac under ~96 GB.
pub const H3_ACTIVATION_BYTES: u64 = 6 * 1024 * 1024 * 1024;

/// MiniMax Music 3's non-weight working set at the request caps: batch-2 KV
/// cache for 36 layers at 9000 frames + 5000 prompt tokens (~4.1 GB), the
/// bf16 frame-hidden buffer (~0.6 GB), and DiT/vocoder window transients.
pub const MUSIC3_GEN_BUFFER_BYTES: u64 = 6 * 1024 * 1024 * 1024;

/// The DiT term of the H3 bill. `precompute` mirrors
/// MINIMAX_H3_ADALN_PRECOMPUTE: with it off the modulation weights are never
/// freed and the whole file stays resident, so the shed size would under-bill
/// by ~12 GiB into an uncatchable Metal OOM.
pub fn h3DitResidentBytes(dit_file: u64, precompute: bool) u64 {
    if (!precompute) return dit_file;
    return dit_file * H3_DIT_RESIDENT_PCT / 100;
}

/// MiniMax-H3's staged residency plan, as a bill. `minimax_h3.generate` runs
/// three DISJOINT stages: the text encoder is loaded, run and FREED before the
/// DiT loads (`Model.load` is scoped), and the DiT is released before the VAEs
/// load — so the peak is the BIGGEST stage, never a sum. The two VAEs are one
/// stage: the video decoder is still resident when the audio one loads.
/// `dit_resident` is post-AdaLN-precompute (`h3DitResidentBytes`), which the
/// file size overstates by ~39%.
pub fn h3PeakBytes(te: u64, dit_resident: u64, video_vae: u64, audio_vae: u64) u64 {
    const vaes = video_vae + audio_vae;
    const generating = @max(dit_resident, vaes);
    if (te == 0 and generating == 0) return 0; // unknown dir → never block
    return stagedPeakBytes(0, &.{ te, generating + H3_ACTIVATION_BYTES });
}

/// Per-backend generation-peak estimate for the media load preflight. A
/// backend with a STAGED residency plan declares it here; every other type
/// keeps the sum-of-safetensors default — over-billing fails safe (a refused
/// load names its numbers), under-billing kills the process mid-request.
pub fn estimatePeakResidentBytesIn(io: std.Io, dir: std.Io.Dir, model_type: []const u8) u64 {
    const sz = struct {
        fn f(io_: std.Io, d: std.Io.Dir, name: []const u8) u64 {
            const st = d.statFile(io_, name, .{}) catch return 0;
            return @intCast(st.size);
        }
    }.f;
    if (std.mem.eql(u8, model_type, "minimax_h3")) {
        // The Turbo LoRA (when the pack ships one) is resident ALONGSIDE the
        // DiT and precompute does not free it, so it rides the DiT term at
        // full size — billed whenever present, since the gate estimate is
        // per-model, not per-request.
        const dit = h3DitResidentBytes(
            sz(io, dir, "transformer.safetensors"),
            h3AdalnPrecomputeOn(),
        ) + sz(io, dir, "turbo_lora.safetensors");
        return h3PeakBytes(
            sz(io, dir, "text_encoder.safetensors"),
            dit,
            sz(io, dir, "video_vae.safetensors"),
            sz(io, dir, "audio_vae.safetensors"),
        );
    }
    if (std.mem.eql(u8, model_type, "minimax_music3")) {
        // The whole engine is resident for its lifetime (no staging), so the
        // sum is the right weight bill — plus the AR stage's working set the
        // directory cannot see: the batch-2 KV cache (~4.1 GB at the 9000-frame
        // + 5000-token caps), the frame-hidden buffer (~0.6 GB bf16), and the
        // DiT/vocoder window transients.
        const sum = sumSafetensorsIn(io, dir);
        if (sum == 0) return 0; // unknown dir -> never block
        return sum + MUSIC3_GEN_BUFFER_BYTES;
    }
    if (std.mem.eql(u8, model_type, "AudioVideo")) {
        // Both variants ship; only one is ever loaded. Subtract the smaller so
        // an asymmetric future pack still bills its larger one.
        const spare = @min(
            sz(io, dir, "transformer-dev.safetensors"),
            sz(io, dir, "transformer-distilled.safetensors"),
        );
        return ltxPeakBytes(sumSafetensorsIn(io, dir), spare, 0);
    }
    return sumSafetensorsIn(io, dir);
}

/// Sum of the `.safetensors` under an absolute path, or 0 if it is not
/// readable — 0 means "unknown", which every caller treats as "do not block".
fn sumSafetensorsAt(io: std.Io, path: []const u8) u64 {
    if (path.len == 0 or path[0] != '/') return 0; // openDirAbsolute UB class
    var d = std.Io.Dir.openDirAbsolute(io, path, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    return sumSafetensorsIn(io, d);
}

/// LTX's text encoder, resolved the way `resolveGemmaDir` does but without an
/// allocator (this runs inside the load gate). Absent → 0.
fn ltxTextEncoderBytes(io: std.Io) u64 {
    var buf: [1024]u8 = undefined;
    if (std.c.getenv("LTX_GEMMA_DIR")) |env| {
        const e = std.mem.span(env);
        if (std.fs.path.isAbsolute(e)) return sumSafetensorsAt(io, e);
    }
    // HOME is POSIX-only; see platform.homeDir.
    var home_buf: [1024]u8 = undefined;
    const home = platform.homeDir(&home_buf) orelse return 0;
    for ([_][]const u8{ LTX_GEMMA_REPO_DIR, "gemma-3-12b-it-4bit" }) |rel| {
        const p = std.fmt.bufPrint(&buf, "{s}/.mlx-serve/models/{s}", .{ home, rel }) catch continue;
        const n = sumSafetensorsAt(io, p);
        if (n > 0) return n;
    }
    return 0;
}

pub fn estimatePeakResidentBytes(io: std.Io, model_dir: []const u8, model_type: []const u8) u64 {
    if (model_dir.len == 0 or model_dir[0] != '/') return 0; // openDirAbsolute UB class
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true }) catch return 0;
    defer dir.close(io);
    const in_dir = estimatePeakResidentBytesIn(io, dir, model_type);
    // LTX reads its text encoder from a DIFFERENT repo, so it is a stage the
    // model dir cannot see. Every other backend's weights are all in its own
    // directory; if that stops being true, it belongs here beside this one.
    if (in_dir > 0 and std.mem.eql(u8, model_type, "AudioVideo"))
        return stagedPeakBytes(in_dir, &.{ltxTextEncoderBytes(io)});
    return in_dir;
}

pub fn fileExists(io: std.Io, path: [:0]const u8) bool {
    // openFileAbsolute ASSERTS the path is absolute — a failed assert is
    // `unreachable`, i.e. ReleaseFast UB that can miscompile the CALLER (see
    // the openDirAbsolute gotcha in CLAUDE.md). Paths here come from --model /
    // $LTX_AUDIO_DIR / $LTX_GEMMA_DIR, all user-controlled, so guard first.
    if (path.len == 0 or !std.fs.path.isAbsolute(path)) return false;
    if (std.Io.Dir.openFileAbsolute(io, path, .{})) |f| {
        f.close(io);
        return true;
    } else |_| return false;
}

/// LTX's text encoder is Gemma-3-12B (4-bit). It's a normal downloadable model
/// the app pulls into `~/.mlx-serve/models` (as the LTX bundle dependency, and
/// selectable as a chat model). The repo id maps to a `<author>/<name>` dir.
pub const LTX_GEMMA_REPO_DIR = "mlx-community/gemma-3-12b-it-4bit";

/// Whether MiniMax-H3 precomputes its AdaLN tables -- a pure env read, but the
/// residency estimate depends on it (precompute holds a second copy alongside
/// the DiT), so it has to be answerable without the H3 backend compiled in.
/// `minimax_h3.adalnPrecomputeOn` delegates here so there is ONE answer.
pub fn h3AdalnPrecomputeOn() bool {
    const raw = std.c.getenv("MINIMAX_H3_ADALN_PRECOMPUTE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(raw), "0");
}
