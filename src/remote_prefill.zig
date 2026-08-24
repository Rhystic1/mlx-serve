//! Remote prefill v1 -- the SHARED, pure half of the wire contract between a
//! prefill worker (`POST /v1/prefill`, server.zig, Linux/CUDA box) and a
//! consumer (`--remote-prefill <url>`, the decoding Mac). Nothing here touches
//! a socket or an engine: the server parses its request through
//! `parseRequest`, the consumer checks a reply through `validateResponse`, and
//! both spell the header names from the ONE table below.
//!
//! Wire format (v1):
//!   Request  POST /v1/prefill, application/json
//!            {"model": "<id>", "tokens": [int, ...]}
//!   Success  200 application/octet-stream, body = the raw llama.cpp
//!            sequence-state blob (`llama_state_seq_get_data`), plus headers:
//!              X-Prefill-Version: 2
//!              X-Prefill-Model:   <request model, verbatim>
//!              X-Prefill-Tokens:  <len(tokens)>
//!              X-Prefill-Bytes:   <blob length == body length>
//!              X-Prefill-Vocab:   <llama_n_vocab>
//!              X-Prefill-Model-Bytes: <GGUF file size>
//!              X-Prefill-Kv-Type: f16 | q8_0 | q4_0   (ggml type NAME of the worker's KV cache)
//!              X-Prefill-Swa:     full | windowed     (the worker's SWA cache mode)
//!   Failure  any non-200; the consumer never parses the body.
//!
//! v2 added the last two: a q8_0 blob read into an f16 cache is not an error,
//! it is garbage KV, so the cache type is part of the fingerprint; the SWA
//! mode says how much of a sliding layer the blob carries (a windowed blob
//! restores into a FULL consumer cache -- llama.cpp's state_read is a plain
//! find_slot of cell_count cells and the mask is by position -- so the
//! consumer compares these against its OWN session, never assumes).
//!
//! Every header is REQUIRED: the blob is coupled to the llama.cpp build and
//! the GGUF weights, and restoring the wrong one is the failure that produces
//! garbage rather than an error. Model id is a string both sides resolved on
//! their own, so vocab size + file size are the cheap structural checks that
//! catch "same name, different quant". A consumer that cannot prove the reply
//! matches falls back to local prefill -- remote prefill must never be able to
//! fail a request.
const std = @import("std");

pub const VERSION: u32 = 2;

pub const H_VERSION = "X-Prefill-Version";
pub const H_MODEL = "X-Prefill-Model";
pub const H_TOKENS = "X-Prefill-Tokens";
pub const H_BYTES = "X-Prefill-Bytes";
pub const H_VOCAB = "X-Prefill-Vocab";
pub const H_MODEL_BYTES = "X-Prefill-Model-Bytes";
pub const H_KV_TYPE = "X-Prefill-Kv-Type";
pub const H_SWA = "X-Prefill-Swa";

pub const CONTENT_TYPE = "application/octet-stream";

/// The worker prefills ALL BUT THE LAST token, and the consumer decodes that
/// one itself. Two reasons, both load-bearing: the blob carries no logits, so
/// the consumer must decode at least one token before sampling anyway -- and
/// on a RECURRENT/hybrid checkpoint (LFM2, Mamba, GatedDeltaNet) a fully
/// resident prompt cannot be backed off one position (a partial trim is
/// refused and the whole session cleared), so a blob describing N tokens
/// would cost the consumer a cold prefill of all N. A blob describing N-1
/// lets `sync(prompt)` decode exactly one token on every arch. Both sides
/// spell this through the ONE helper; `X-Prefill-Tokens` still echoes N.
pub fn prefillSpan(tokens: []const i32) []const i32 {
    return tokens[0 .. tokens.len - 1];
}

/// A request must carry at least two tokens: one to prefill, one to keep.
pub const MIN_TOKENS: usize = 2;

/// Hard cap on tokens per request: a runaway body must not be able to ask a
/// worker for a context it never configured. The worker still clamps to its
/// own context size.
pub const MAX_TOKENS: usize = 1 << 20;

pub const Request = struct {
    model: []const u8,
    tokens: []i32,

    pub fn deinit(self: *Request, allocator: std.mem.Allocator) void {
        allocator.free(self.model);
        allocator.free(self.tokens);
    }
};

pub const ParseError = error{
    InvalidJson,
    NotAnObject,
    MissingModel,
    MissingTokens,
    TooFewTokens,
    TooManyTokens,
    BadToken,
    OutOfMemory,
};

/// Parse the request body. `model` and `tokens` are owned copies.
pub fn parseRequest(allocator: std.mem.Allocator, body: []const u8) ParseError!Request {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch return ParseError.InvalidJson;
    defer parsed.deinit();
    if (parsed.value != .object) return ParseError.NotAnObject;
    const obj = parsed.value.object;

    const model_v = obj.get("model") orelse return ParseError.MissingModel;
    if (model_v != .string or model_v.string.len == 0) return ParseError.MissingModel;

    const toks_v = obj.get("tokens") orelse return ParseError.MissingTokens;
    if (toks_v != .array) return ParseError.MissingTokens;
    const arr = toks_v.array.items;
    if (arr.len < MIN_TOKENS) return ParseError.TooFewTokens;
    if (arr.len > MAX_TOKENS) return ParseError.TooManyTokens;

    const tokens = try allocator.alloc(i32, arr.len);
    errdefer allocator.free(tokens);
    for (arr, 0..) |v, i| {
        if (v != .integer) return ParseError.BadToken;
        if (v.integer < 0 or v.integer > std.math.maxInt(i32)) return ParseError.BadToken;
        tokens[i] = @intCast(v.integer);
    }
    const model = try allocator.dupe(u8, model_v.string);
    return .{ .model = model, .tokens = tokens };
}

/// The reply headers a consumer collected (any missing = null) beside what it
/// already knows about its OWN model. Strings are compared verbatim.
pub const ResponseHeaders = struct {
    version: ?[]const u8 = null,
    model: ?[]const u8 = null,
    tokens: ?[]const u8 = null,
    bytes: ?[]const u8 = null,
    vocab: ?[]const u8 = null,
    model_bytes: ?[]const u8 = null,
};

pub const Expected = struct {
    model: []const u8,
    n_tokens: usize,
    vocab: u32,
    model_bytes: u64,
    body_len: usize,
};

/// Null when the reply is safe to restore; otherwise the reason to log beside
/// `fell back:`. Every check is a named string so the log explains itself.
pub fn validateResponse(h: ResponseHeaders, e: Expected) ?[]const u8 {
    const ver = h.version orelse return "missing " ++ H_VERSION;
    if (parseU64(ver) != VERSION) return "unsupported " ++ H_VERSION;
    const model = h.model orelse return "missing " ++ H_MODEL;
    if (!std.mem.eql(u8, model, e.model)) return "model id mismatch";
    const toks = h.tokens orelse return "missing " ++ H_TOKENS;
    if (parseU64(toks) != e.n_tokens) return "token count mismatch";
    const bytes = h.bytes orelse return "missing " ++ H_BYTES;
    if (parseU64(bytes) != e.body_len) return "blob length mismatch";
    const vocab = h.vocab orelse return "missing " ++ H_VOCAB;
    if (parseU64(vocab) != e.vocab) return "vocab size mismatch";
    const mb = h.model_bytes orelse return "missing " ++ H_MODEL_BYTES;
    if (parseU64(mb) != e.model_bytes) return "model file size mismatch";
    if (e.body_len == 0) return "empty blob";
    return null;
}

fn parseU64(s: []const u8) ?u64 {
    return std.fmt.parseInt(u64, std.mem.trim(u8, s, " \t"), 10) catch null;
}

/// Format the eight reply headers (server side). Lines end in CRLF, ready to
/// be spliced into a response head. `kv_type` / `swa` are the NAMES from
/// `arch/llama.zig` (`kvTypeName`, `swaModeName`).
pub fn formatHeaders(buf: []u8, model: []const u8, n_tokens: usize, blob_len: usize, vocab: u32, model_bytes: u64, kv_type: []const u8, swa: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, H_VERSION ++ ": {d}\r\n" ++ H_MODEL ++ ": {s}\r\n" ++ H_TOKENS ++ ": {d}\r\n" ++ H_BYTES ++ ": {d}\r\n" ++ H_VOCAB ++ ": {d}\r\n" ++ H_MODEL_BYTES ++ ": {d}\r\n" ++ H_KV_TYPE ++ ": {s}\r\n" ++ H_SWA ++ ": {s}\r\n", .{ VERSION, model, n_tokens, blob_len, vocab, model_bytes, kv_type, swa });
}

test "remote_prefill: parseRequest accepts the v1 body and refuses each malformed shape by name" {
    const a = std.testing.allocator;
    var req = try parseRequest(a, "{\"model\":\"gemma-4-12b\",\"tokens\":[2,106,17]}");
    defer req.deinit(a);
    try std.testing.expectEqualStrings("gemma-4-12b", req.model);
    try std.testing.expectEqualSlices(i32, &.{ 2, 106, 17 }, req.tokens);

    try std.testing.expectError(ParseError.InvalidJson, parseRequest(a, "{nope"));
    try std.testing.expectError(ParseError.NotAnObject, parseRequest(a, "[1,2]"));
    try std.testing.expectError(ParseError.MissingModel, parseRequest(a, "{\"tokens\":[1]}"));
    try std.testing.expectError(ParseError.MissingModel, parseRequest(a, "{\"model\":\"\",\"tokens\":[1]}"));
    try std.testing.expectError(ParseError.MissingTokens, parseRequest(a, "{\"model\":\"m\"}"));
    try std.testing.expectError(ParseError.MissingTokens, parseRequest(a, "{\"model\":\"m\",\"tokens\":\"1,2\"}"));
    try std.testing.expectError(ParseError.TooFewTokens, parseRequest(a, "{\"model\":\"m\",\"tokens\":[]}"));
    try std.testing.expectError(ParseError.TooFewTokens, parseRequest(a, "{\"model\":\"m\",\"tokens\":[7]}"));
    try std.testing.expectError(ParseError.BadToken, parseRequest(a, "{\"model\":\"m\",\"tokens\":[1,-3]}"));
    try std.testing.expectError(ParseError.BadToken, parseRequest(a, "{\"model\":\"m\",\"tokens\":[1,\"x\"]}"));
    try std.testing.expectError(ParseError.BadToken, parseRequest(a, "{\"model\":\"m\",\"tokens\":[1,1.5]}"));
}

test "remote_prefill: validateResponse passes a matching reply and names every mismatch" {
    const e = Expected{ .model = "gemma-4-12b", .n_tokens = 3, .vocab = 262144, .model_bytes = 7_000_000, .body_len = 4096 };
    const ok = ResponseHeaders{ .version = "2", .model = "gemma-4-12b", .tokens = "3", .bytes = "4096", .vocab = "262144", .model_bytes = "7000000" };
    try std.testing.expect(validateResponse(ok, e) == null);

    var h = ok;
    h.version = null;
    try std.testing.expectEqualStrings("missing " ++ H_VERSION, validateResponse(h, e).?);
    h = ok;
    h.version = "1";
    try std.testing.expectEqualStrings("unsupported " ++ H_VERSION, validateResponse(h, e).?);
    h = ok;
    h.model = "gemma-4-12b-q8";
    try std.testing.expectEqualStrings("model id mismatch", validateResponse(h, e).?);
    h = ok;
    h.tokens = "4";
    try std.testing.expectEqualStrings("token count mismatch", validateResponse(h, e).?);
    h = ok;
    h.bytes = "4095";
    try std.testing.expectEqualStrings("blob length mismatch", validateResponse(h, e).?);
    h = ok;
    h.vocab = "32000";
    try std.testing.expectEqualStrings("vocab size mismatch", validateResponse(h, e).?);
    h = ok;
    h.model_bytes = "1";
    try std.testing.expectEqualStrings("model file size mismatch", validateResponse(h, e).?);
    h = ok;
    h.vocab = "abc";
    try std.testing.expectEqualStrings("vocab size mismatch", validateResponse(h, e).?);

    // A 200 with no bytes is not a state.
    var e0 = e;
    e0.body_len = 0;
    var h0 = ok;
    h0.bytes = "0";
    try std.testing.expectEqualStrings("empty blob", validateResponse(h0, e0).?);
}

test "remote_prefill: formatHeaders emits exactly the eight v2 headers" {
    var buf: [512]u8 = undefined;
    const out = try formatHeaders(&buf, "gemma-4-12b", 3, 4096, 262144, 7_000_000, "q8_0", "windowed");
    try std.testing.expectEqualStrings("X-Prefill-Version: 2\r\nX-Prefill-Model: gemma-4-12b\r\nX-Prefill-Tokens: 3\r\nX-Prefill-Bytes: 4096\r\nX-Prefill-Vocab: 262144\r\nX-Prefill-Model-Bytes: 7000000\r\nX-Prefill-Kv-Type: q8_0\r\nX-Prefill-Swa: windowed\r\n", out);
}

test "remote_prefill: prefillSpan leaves exactly the last token for the consumer" {
    const ids = [_]i32{ 2, 106, 17, 9 };
    try std.testing.expectEqualSlices(i32, &.{ 2, 106, 17 }, prefillSpan(&ids));
    const two = [_]i32{ 5, 6 };
    try std.testing.expectEqualSlices(i32, &.{5}, prefillSpan(&two));
}
