//! Media-generation surface for builds without MLX (see src/build_cfg.zig).
//!
//! Every media backend — FLUX, Krea, MageFlow, LTX, MiniMax-H3, Hunyuan3D,
//! ACE-Step, Music 3, Qwen3-TTS, Kokoro — is written against MLX, so media
//! generation stands or falls with it. This build has none.
//!
//! What it DOES keep is `gen_common.zig`: classification and residency
//! estimation are pure, and the server needs them precisely because it cannot
//! serve these models. Recognising an image pack is what lets `/v1/images/…`
//! answer with a named refusal, and what stops a media directory from falling
//! through into the text loader (the "incomplete media pack" class, which dies
//! on the first missing weight instead of reporting anything useful).
//!
//! The engine types are `opaque` on purpose: `LoadedModel` holds them as
//! `?*Engine` and they are always null here, so there is no representable
//! value that could make a media path look loaded.

const std = @import("std");

pub const common = @import("gen_common.zig");

// ── Classification + residency (real, shared with the MLX build) ──
pub const Modality = common.Modality;
pub const modalityFromType = common.modalityFromType;
pub const peekModelType = common.peekModelType;
pub const detectModality = common.detectModality;
pub const incompleteMediaDir = common.incompleteMediaDir;
pub const requiredMarkerFor = common.requiredMarkerFor;
pub const media_model_types = common.media_model_types;
pub const GenRoute = common.GenRoute;
pub const AudioBackendKind = common.AudioBackendKind;
pub const audioBackendKindForType = common.audioBackendKindForType;
pub const StubCpuState = common.StubCpuState;
pub const buildStubCpuState = common.buildStubCpuState;
pub const freeStubCpuState = common.freeStubCpuState;
pub const estimateResidentBytes = common.estimateResidentBytes;
pub const stagedPeakBytes = common.stagedPeakBytes;
pub const ltxPeakBytes = common.ltxPeakBytes;
pub const h3DitResidentBytes = common.h3DitResidentBytes;
pub const h3PeakBytes = common.h3PeakBytes;
pub const estimatePeakResidentBytesIn = common.estimatePeakResidentBytesIn;
pub const estimatePeakResidentBytes = common.estimatePeakResidentBytes;
pub const H3_DIT_RESIDENT_PCT = common.H3_DIT_RESIDENT_PCT;
pub const H3_ACTIVATION_BYTES = common.H3_ACTIVATION_BYTES;
pub const MUSIC3_GEN_BUFFER_BYTES = common.MUSIC3_GEN_BUFFER_BYTES;

// ── Engines: absent ──
pub const Error = error{MlxUnavailable};

/// One shape per modality. `LoadedModel` holds each as `?*Engine` and the
/// scheduler calls `load` / `deinit`, so they need to be sized types with those
/// methods -- an `opaque` cannot be embedded or optional.
///
/// `load` refuses BY NAME. The scheduler turns that into a load failure the
/// client sees as a named error, which is the whole point: a media request on
/// this build must say media generation is unavailable, not fail somewhere
/// deeper with a missing weight.
fn MediaEngine(comptime what: []const u8) type {
    return struct {
        const Self = @This();

        /// The audio arm is asked which backend it is when advertising
        /// capabilities (`has_music_backend`). Present on every arm so the
        /// generic shape stays uniform; always `.tts`, which is the "no music
        /// backend" answer. Unreachable regardless -- `load` never returns.
        backend: common.AudioBackendKind = .tts,

        pub fn load(_: std.Io, _: std.mem.Allocator, _: []const u8) anyerror!*Self {
            @branchHint(.cold);
            _ = what; // named in the error path's message at the call site
            return Error.MlxUnavailable;
        }

        pub fn deinit(_: *Self) void {}
    };
}

pub const ImageEngine = MediaEngine("image");
pub const AudioEngine = MediaEngine("audio");
pub const VideoEngine = MediaEngine("video");
pub const MeshEngine = MediaEngine("mesh");

/// Alias kept because `estimatePeakResident` is spelled both ways at different
/// call sites in the shared plumbing.
pub const estimatePeakResident = common.estimatePeakResidentBytes;

/// Text embeddings run through the MLX encoder path, which is absent. GGUF
/// embedding models are a separate, planned route through llama.cpp's own
/// embedding API — not this function.
/// Returns the per-sequence embedding vectors. Refuses: the MLX encoder is
/// absent. GGUF embedding models are a separate planned route through
/// llama.cpp's own embedding API.
pub fn computeEmbeddingsBatch(_: std.mem.Allocator, _: anytype, _: anytype) anyerror![][]f32 {
    return Error.MlxUnavailable;
}

// ── HTTP handlers ──────────────────────────────────────────────────────────
//
// The media endpoints stay ROUTED (endpoint existence must never depend on
// model state -- see the route-404 rule), so these entry points must exist.
// They are unreachable in practice: each is called only with a loaded engine
// of its modality, and no engine can load. They refuse rather than being
// removed so the dispatch chain is unchanged and `ROUTE_PATHS` still matches.

pub fn handleImage(_: std.mem.Allocator, _: anytype, _: []const u8, _: *ImageEngine) anyerror!void {
    return Error.MlxUnavailable;
}
pub fn handleAudio(_: std.mem.Allocator, _: anytype, _: []const u8, _: *AudioEngine) anyerror!void {
    return Error.MlxUnavailable;
}
pub fn handleMusic(_: std.mem.Allocator, _: anytype, _: []const u8, _: *AudioEngine) anyerror!void {
    return Error.MlxUnavailable;
}
pub fn handleVideo(_: std.Io, _: std.mem.Allocator, _: anytype, _: []const u8, _: *VideoEngine) anyerror!void {
    return Error.MlxUnavailable;
}
pub fn handleMesh(_: std.mem.Allocator, _: anytype, _: []const u8, _: *MeshEngine) anyerror!void {
    return Error.MlxUnavailable;
}

/// `POST /v1/images/edits` multipart -> the `mode:"edit"` JSON body. Pure
/// translation with no inference in it, but it only ever feeds an image
/// engine, so it refuses here too rather than producing a body nothing can
/// consume. Error set mirrors gen.zig so the named 400s still compile.
pub const EditFormError = error{
    NotMultipart,
    MissingPrompt,
    MissingImage,
    TooManyImages,
    MaskUnsupported,
    MultipleChoicesUnsupported,
    UrlResponseUnsupported,
    OutputFormatUnsupported,
    StreamUnsupported,
    OutOfMemory,
};

pub fn openaiEditFormToJson(_: std.mem.Allocator, _: []const u8, _: []const u8) EditFormError![]u8 {
    return EditFormError.NotMultipart;
}

pub fn editFormErrorMessage(err: EditFormError) []const u8 {
    return switch (err) {
        error.NotMultipart => "image generation is unavailable in this build",
        error.MissingPrompt => "missing required field: prompt",
        error.MissingImage => "missing required field: image",
        error.TooManyImages => "too many images",
        error.MaskUnsupported => "mask is not supported",
        error.MultipleChoicesUnsupported => "n>1 is not supported",
        error.UrlResponseUnsupported => "response_format=url is not supported",
        error.OutputFormatUnsupported => "output_format is not supported",
        error.StreamUnsupported => "stream is not supported",
        error.OutOfMemory => "out of memory",
    };
}
