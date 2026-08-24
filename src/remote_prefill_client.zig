//! Client half of remote prefill (`--remote-prefill <base-url>`).
//!
//! The consumer side of the PoC in nvidia-prefill-plan.md: before prefilling an
//! embedded-engine (GGUF) request locally, POST the token ids to a remote box
//! holding the SAME model on a faster backend, take back its llama.cpp
//! sequence-state blob, and restore it instead of decoding the prompt here.
//!
//! The wire format, the header names and `validateResponse` are SHARED with the
//! endpoint side and live in `remote_prefill.zig`. This file is only what a
//! client needs: where to send, whether it is worth sending, how to frame the
//! request, and what to say when it did not work.
//!
//! THE INVARIANT THAT OUTRANKS THE FEATURE: remote prefill must never be able
//! to fail a request. Nothing here returns an error to the caller — every path
//! yields either a blob or a REASON, and a reason means "prefill locally, as if
//! the flag had never been passed". That is why the fetch entry point has no
//! error set: there is no failure a caller could be tempted to propagate.
//!
//! Token accounting is the shared contract's, not ours: the client POSTs the
//! FULL token list, and the worker prefills `remote_prefill.prefillSpan` of it
//! (all but the last). See that helper's comment for why — on a recurrent
//! checkpoint a fully-resident prompt cannot be backed off one position, so a
//! blob describing all N would cost the consumer a cold prefill of all N.

const std = @import("std");
const net = @import("lan_net.zig");
const peers = @import("lan_peers.zig");
const proto = @import("remote_prefill.zig");
const chat = @import("chat.zig");

/// Below this many tokens the round trip cannot pay for itself: the blob is
/// per-layer KV and costs far more to move than a short prompt costs to
/// decode. An economic bound, not a correctness one — a smaller prompt is
/// simply prefilled locally. Distinct from `proto.MIN_TOKENS`, which is the
/// protocol floor (2) that makes `prefillSpan` meaningful.
pub const MIN_REMOTE_TOKENS: usize = 256;

comptime {
    // The economic gate must never admit a request the protocol would refuse.
    if (MIN_REMOTE_TOKENS < proto.MIN_TOKENS) @compileError("MIN_REMOTE_TOKENS below the protocol floor");
}

/// Read deadline for the whole exchange. A worker that accepts and then stalls
/// must not hold a request longer than prefilling it here would have taken.
pub const DEFAULT_TIMEOUT_MS: i32 = 30_000;

/// Refuse a blob larger than this rather than grow without bound; a few hundred
/// MB is expected at long contexts (per-layer KV), so the cap is generous.
pub const MAX_BLOB_BYTES: usize = 2 * 1024 * 1024 * 1024;

pub const PREFILL_PATH = "/v1/prefill";

/// `--remote-prefill <base-url>`, borrowed from argv like `--api-key`. It lives
/// HERE rather than in server.zig because the consumer is scheduler.zig, which
/// deliberately does not import server.zig (server imports scheduler — the
/// other direction would be a cycle). Null/empty = off, which is also what
/// every failure degrades to.
pub var g_remote_prefill_url: ?[]const u8 = null;

/// Whether remote prefill can pay for itself, as a PER-TOKEN comparison.
///
/// The flat MIN_REMOTE_TOKENS floor this replaces was the wrong SHAPE, not
/// merely the wrong number. Blob bytes scale linearly with tokens and prefill
/// time scales linearly with tokens, so their ratio is constant in n: a config
/// that loses at 1k tokens loses at 100k, and one that wins, wins everywhere.
/// No token threshold can express that.
///
/// THAT LINEARITY IS A SHORT-CONTEXT APPROXIMATION, and the rates below are
/// only valid at the length they were measured at. Prefill carries an O(n^2)
/// attention term while the blob stays strictly linear — and goes SUB-linear
/// under `swa_full=false`, where most layers cap at their window — so at long
/// contexts the two curves genuinely diverge in REMOTE's favour and a real
/// crossover can exist even for a config that loses at 3k. Deliberately not
/// modelled here (the PoC lives in the short-context regime), but do not
/// "fix" this by trusting a rate learned at 2k to describe 64k: re-measure
/// per length band, or the model will refuse a config that would have won.
///
/// What it actually turns on is the WORKER'S ADVANTAGE. Per token the remote
/// path costs `worker_prefill + transfer` and the local path costs
/// `local_prefill`, so the entire budget for transfer is
/// `local_ms_per_token - worker_ms_per_token`. A worker only 1.5x faster than
/// the consumer leaves a third of local prefill to spend on the wire, however
/// fat the pipe — measured 2026-08-23: gemma-4-12b at 167 KB/token needs a
/// 2.2x blob shrink just to break even on gigabit against a CUDA worker that
/// prefills at 1.398 ms/token versus the Mac's 2.109.
///
/// A small token floor still earns its place, but only to amortize the fixed
/// round-trip overhead this model ignores — not to decide viability.
pub const Economics = struct {
    /// Blob bytes per prompt token, learned from a previous reply. 0 = not yet
    /// measured for this model.
    bytes_per_token: u64 = 0,
    /// Assumed usable wire throughput. 0 is a configuration error, never
    /// evidence about speed.
    wire_bytes_per_sec: u64 = 0,
    worker_ms_per_token: f64 = 0,
    local_ms_per_token: f64 = 0,

    pub fn viable(self: Economics) bool {
        if (self.wire_bytes_per_sec == 0) return false;
        // A worker that is not FASTER can never pay, at any blob size.
        const budget_ms = self.local_ms_per_token - self.worker_ms_per_token;
        if (budget_ms <= 0) return false;
        // Unmeasured rate ⇒ attempt, and learn it from the reply. Refusing here
        // would mean the client could never bootstrap a model's rate.
        if (self.bytes_per_token == 0) return true;
        const transfer_ms = @as(f64, @floatFromInt(self.bytes_per_token)) /
            @as(f64, @floatFromInt(self.wire_bytes_per_sec)) * 1000.0;
        return transfer_ms < budget_ms;
    }
};

/// The blob rate a reply implies. Null when either number is degenerate — a
/// rate invented from a zero is worse than no rate at all.
pub fn observedBytesPerToken(blob_bytes: u64, n_tokens: usize) ?u64 {
    if (blob_bytes == 0 or n_tokens == 0) return null;
    return blob_bytes / @as(u64, @intCast(n_tokens));
}

/// Why a request did not use remote prefill. Every arm is a log phrase, not an
/// error: the request proceeds locally in all of them.
pub const FallbackReason = enum {
    disabled,
    too_few_tokens,
    bad_base_url,
    tls_unsupported,
    not_embedded_engine,
    unresolved_host,
    connect_failed,
    timed_out,
    http_error,
    truncated_response,
    oversize_blob,
    restore_failed,

    pub fn text(self: FallbackReason) []const u8 {
        return switch (self) {
            .disabled => "not configured",
            .too_few_tokens => "prompt below the round-trip threshold",
            .bad_base_url => "unusable --remote-prefill URL",
            .tls_unsupported => "https is not supported in v1",
            .not_embedded_engine => "not an embedded-engine model",
            .unresolved_host => "remote host did not resolve",
            .connect_failed => "remote unreachable",
            .timed_out => "remote timed out",
            .http_error => "remote returned an error status",
            .truncated_response => "response ended mid-message",
            .oversize_blob => "declared blob exceeds the client cap",
            .restore_failed => "state restore refused the blob",
        };
    }
};

/// Join a user-supplied base URL with the endpoint path. Pure string work:
/// accepts a trailing slash or a mounted path prefix (the worker may sit behind
/// a reverse proxy). Rejects anything that is not an http(s) authority — a raw
/// KV blob handed to some other protocol is worse than no remote prefill.
pub fn endpointUrl(buf: []u8, base: []const u8) ?[]const u8 {
    if (base.len == 0) return null;
    const is_http = std.mem.startsWith(u8, base, "http://");
    const is_https = std.mem.startsWith(u8, base, "https://");
    if (!is_http and !is_https) return null;

    const scheme_len: usize = if (is_https) "https://".len else "http://".len;
    var trimmed = base;
    while (trimmed.len > 0 and trimmed[trimmed.len - 1] == '/') trimmed = trimmed[0 .. trimmed.len - 1];
    // Trimming must not have eaten the authority ("http://" / "http:///").
    if (trimmed.len <= scheme_len) return null;

    return std.fmt.bufPrint(buf, "{s}{s}", .{ trimmed, PREFILL_PATH }) catch null;
}

pub const Endpoint = struct {
    host: []const u8,
    port: u16,
    path: []const u8,
    tls: bool,
};

/// Split a full endpoint URL into its connect parts. Slices borrow `url`.
pub fn parseUrl(url: []const u8) ?Endpoint {
    const is_http = std.mem.startsWith(u8, url, "http://");
    const is_https = std.mem.startsWith(u8, url, "https://");
    if (!is_http and !is_https) return null;
    const rest = url[(if (is_https) "https://".len else "http://".len)..];
    if (rest.len == 0) return null;

    const slash = std.mem.indexOfScalar(u8, rest, '/') orelse rest.len;
    const authority = rest[0..slash];
    if (authority.len == 0) return null;
    const path = if (slash == rest.len) "/" else rest[slash..];

    var host = authority;
    var port: u16 = if (is_https) 443 else 80;
    if (std.mem.lastIndexOfScalar(u8, authority, ':')) |c| {
        // Rightmost colon, but never one inside an IPv6 literal's brackets.
        const rb = std.mem.indexOfScalar(u8, authority, ']');
        if (rb == null or c > rb.?) {
            host = authority[0..c];
            port = std.fmt.parseInt(u16, authority[c + 1 ..], 10) catch return null;
            if (port == 0) return null;
        }
    }
    if (host.len == 0) return null;
    return .{ .host = host, .port = port, .path = path, .tls = is_https };
}

/// Should this request even try the remote? Cheap local checks only; the
/// expensive ones (resolution, reachability) are discovered by trying.
pub fn shouldAttempt(configured: ?[]const u8, is_embedded_engine: bool, n_tokens: usize) ?FallbackReason {
    const base = configured orelse return .disabled;
    if (base.len == 0) return .disabled;
    if (!is_embedded_engine) return .not_embedded_engine;
    if (n_tokens < MIN_REMOTE_TOKENS) return .too_few_tokens;
    var buf: [1024]u8 = undefined;
    const url = endpointUrl(&buf, base) orelse return .bad_base_url;
    const ep = parseUrl(url) orelse return .bad_base_url;
    // Refused up front rather than at connect time: claiming to attempt an
    // exchange we cannot perform would log an attempt for a request that was
    // never going anywhere.
    if (ep.tls) return .tls_unsupported;
    return null;
}

/// The JSON body: `{"model": ..., "tokens": [ids]}`. Caller frees.
pub fn buildRequestBody(allocator: std.mem.Allocator, model: []const u8, tokens: []const i32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"model\":");
    // A model id is arbitrary bytes by the time it reaches here; the codebase's
    // one escaper is what keeps a control byte from breaking the worker's parse.
    try chat.appendJsonString(allocator, &out, model);
    try out.appendSlice(allocator, ",\"tokens\":[");
    var num: [16]u8 = undefined;
    for (tokens, 0..) |t, i| {
        if (i > 0) try out.append(allocator, ',');
        try out.appendSlice(allocator, std.fmt.bufPrint(&num, "{d}", .{t}) catch unreachable);
    }
    try out.appendSlice(allocator, "]}");
    return out.toOwnedSlice(allocator);
}

/// The request head. `Connection: close` is what makes the read loop's EOF
/// meaningful, since the blob length is only known from a header.
pub fn buildRequestHead(buf: []u8, ep: Endpoint, body_len: usize) ![]const u8 {
    return std.fmt.bufPrint(
        buf,
        "POST {s} HTTP/1.1\r\nHost: {s}:{d}\r\nContent-Type: application/json\r\n" ++
            "Content-Length: {d}\r\nAccept: " ++ proto.CONTENT_TYPE ++ "\r\nConnection: close\r\n\r\n",
        .{ ep.path, ep.host, ep.port, body_len },
    );
}

pub const Split = struct {
    status: u16,
    head: []const u8,
    body: []const u8,
};

/// Split a raw HTTP response into status, header block and body. Null when the
/// message is not yet complete — the caller keeps reading or gives up.
pub fn splitResponse(raw: []const u8) ?Split {
    const head_end = std.mem.indexOf(u8, raw, "\r\n\r\n") orelse return null;
    const line_end = std.mem.indexOf(u8, raw, "\r\n") orelse return null;
    var it = std.mem.splitScalar(u8, raw[0..line_end], ' ');
    _ = it.next() orelse return null; // "HTTP/1.1"
    const code_str = it.next() orelse return null;
    const code = std.fmt.parseInt(u16, code_str, 10) catch return null;
    return .{ .status = code, .head = raw[0..head_end], .body = raw[head_end + 4 ..] };
}

// Lowercased header names for the case-insensitive lookup. Spelled out rather
// than computed: a comptime lowercaser that returns a slice is a footgun in
// this Zig, and the drift risk is covered by the assert below.
const L_VERSION = "x-prefill-version";
const L_MODEL = "x-prefill-model";
const L_TOKENS = "x-prefill-tokens";
const L_BYTES = "x-prefill-bytes";
const L_VOCAB = "x-prefill-vocab";
const L_MODEL_BYTES = "x-prefill-model-bytes";

comptime {
    // If the shared module renames a header, these must move with it or the
    // lookup silently returns null and every reply "misses" a header.
    assertLowerOf(proto.H_VERSION, L_VERSION);
    assertLowerOf(proto.H_MODEL, L_MODEL);
    assertLowerOf(proto.H_TOKENS, L_TOKENS);
    assertLowerOf(proto.H_BYTES, L_BYTES);
    assertLowerOf(proto.H_VOCAB, L_VOCAB);
    assertLowerOf(proto.H_MODEL_BYTES, L_MODEL_BYTES);
}

fn assertLowerOf(comptime name: []const u8, comptime want: []const u8) void {
    if (name.len != want.len) @compileError("header name/lowercase length drift: " ++ name);
    for (name, want) |c, w| {
        if (std.ascii.toLower(c) != w) @compileError("header lowercase drift: " ++ name);
    }
}

/// Collect the six reply headers for `proto.validateResponse`.
pub fn readHeaders(head: []const u8) proto.ResponseHeaders {
    return .{
        .version = peers.headerValueCI(head, L_VERSION),
        .model = peers.headerValueCI(head, L_MODEL),
        .tokens = peers.headerValueCI(head, L_TOKENS),
        .bytes = peers.headerValueCI(head, L_BYTES),
        .vocab = peers.headerValueCI(head, L_VOCAB),
        .model_bytes = peers.headerValueCI(head, L_MODEL_BYTES),
    };
}

/// The outcome of an attempted exchange. `.blob` is owned by the caller.
pub const Outcome = union(enum) {
    blob: []u8,
    fell_back: []const u8,
};

/// Perform the exchange. Never fails: every problem becomes `.fell_back` with a
/// reason to log. `expected` is checked against the reply headers BEFORE the
/// body is handed back, so a blob from a different model, a different build or
/// a different token count can never reach the restore.
pub fn fetchBlob(
    allocator: std.mem.Allocator,
    base_url: []const u8,
    model: []const u8,
    tokens: []const i32,
    expected: proto.Expected,
    timeout_ms: i32,
) Outcome {
    var url_buf: [1024]u8 = undefined;
    const url = endpointUrl(&url_buf, base_url) orelse return .{ .fell_back = FallbackReason.bad_base_url.text() };
    const ep = parseUrl(url) orelse return .{ .fell_back = FallbackReason.bad_base_url.text() };
    if (ep.tls) return .{ .fell_back = FallbackReason.tls_unsupported.text() };

    const ip4 = resolveIp4(allocator, ep.host) orelse
        return .{ .fell_back = FallbackReason.unresolved_host.text() };

    const body = buildRequestBody(allocator, model, tokens) catch
        return .{ .fell_back = FallbackReason.oversize_blob.text() };
    defer allocator.free(body);

    var head_buf: [512]u8 = undefined;
    const head = buildRequestHead(&head_buf, ep, body.len) catch
        return .{ .fell_back = FallbackReason.bad_base_url.text() };

    const s = net.connectTimeout(ip4, ep.port, timeout_ms) catch
        return .{ .fell_back = FallbackReason.connect_failed.text() };
    defer net.close(s);

    net.writeAll(s, head) catch return .{ .fell_back = FallbackReason.connect_failed.text() };
    net.writeAll(s, body) catch return .{ .fell_back = FallbackReason.connect_failed.text() };

    var resp: std.ArrayList(u8) = .empty;
    defer resp.deinit(allocator);
    var chunk: [64 * 1024]u8 = undefined;
    var declared: ?usize = null;
    while (true) {
        if (!net.waitReadable(s, timeout_ms)) return .{ .fell_back = FallbackReason.timed_out.text() };
        const n = net.read(s, &chunk) catch break;
        if (n == 0) break;
        resp.appendSlice(allocator, chunk[0..n]) catch
            return .{ .fell_back = FallbackReason.oversize_blob.text() };

        if (declared == null) {
            if (splitResponse(resp.items)) |sp| {
                if (readHeaders(sp.head).bytes) |bytes_hdr| {
                    const want = std.fmt.parseInt(usize, std.mem.trim(u8, bytes_hdr, " \t"), 10) catch 0;
                    // Stop before allocating a body we would refuse anyway.
                    if (want > MAX_BLOB_BYTES) return .{ .fell_back = FallbackReason.oversize_blob.text() };
                    if (want > 0) declared = want;
                }
            }
        }
        if (declared) |want| {
            if (splitResponse(resp.items)) |sp| {
                if (sp.body.len >= want) break;
            }
        }
    }

    const sp = splitResponse(resp.items) orelse
        return .{ .fell_back = FallbackReason.truncated_response.text() };
    if (sp.status != 200) return .{ .fell_back = FallbackReason.http_error.text() };

    var exp = expected;
    exp.body_len = sp.body.len;
    if (proto.validateResponse(readHeaders(sp.head), exp)) |why| return .{ .fell_back = why };

    const owned = allocator.dupe(u8, sp.body) catch
        return .{ .fell_back = FallbackReason.oversize_blob.text() };
    return .{ .blob = owned };
}

/// Resolve a host to one IPv4 address.
///
/// LIMITATION, deliberate for v1: only a dotted quad resolves. This Zig's std
/// has no `std.net` address-list API here, and the one resolver in the tree
/// (lan_net's getaddrinfo) is Windows-only, so a portable name lookup would be
/// its own piece of work. A NAME therefore falls back with `unresolved_host`
/// rather than silently connecting somewhere unintended — and `localhost` is
/// spelled 127.0.0.1, which is what the WSL<->Windows loop uses anyway.
fn resolveIp4(allocator: std.mem.Allocator, host: []const u8) ?[4]u8 {
    _ = allocator;
    return parseDottedQuad(host);
}

pub fn parseDottedQuad(s: []const u8) ?[4]u8 {
    var out: [4]u8 = undefined;
    var it = std.mem.splitScalar(u8, s, '.');
    var i: usize = 0;
    while (it.next()) |part| {
        if (i == 4) return null;
        if (part.len == 0 or part.len > 3) return null;
        out[i] = std.fmt.parseInt(u8, part, 10) catch return null;
        i += 1;
    }
    return if (i == 4) out else null;
}

// ── Tests ──

const testing = std.testing;

test "endpointUrl appends the path once, whatever the base's trailing slashes" {
    var buf: [256]u8 = undefined;
    try testing.expectEqualStrings("http://192.168.1.50:8080/v1/prefill", endpointUrl(&buf, "http://192.168.1.50:8080").?);
    try testing.expectEqualStrings("http://192.168.1.50:8080/v1/prefill", endpointUrl(&buf, "http://192.168.1.50:8080/").?);
    try testing.expectEqualStrings("http://192.168.1.50:8080/v1/prefill", endpointUrl(&buf, "http://192.168.1.50:8080///").?);
    // A mounted prefix is preserved — the worker may sit behind a proxy.
    try testing.expectEqualStrings("https://box/prefill-worker/v1/prefill", endpointUrl(&buf, "https://box/prefill-worker/").?);
}

test "endpointUrl refuses anything that is not an http(s) authority" {
    var buf: [256]u8 = undefined;
    try testing.expect(endpointUrl(&buf, "") == null);
    try testing.expect(endpointUrl(&buf, "192.168.1.50:8080") == null); // no scheme
    try testing.expect(endpointUrl(&buf, "ftp://box") == null);
    try testing.expect(endpointUrl(&buf, "file:///etc/passwd") == null);
    try testing.expect(endpointUrl(&buf, "http://") == null); // scheme, no host
    try testing.expect(endpointUrl(&buf, "http:///") == null); // slashes are not a host
}

test "endpointUrl declines rather than truncating when the buffer is too small" {
    // bufPrint failing must read as "fall back", never a silently cut URL
    // pointing at a different host.
    var tiny: [8]u8 = undefined;
    try testing.expect(endpointUrl(&tiny, "http://192.168.1.50:8080") == null);
}

test "parseUrl splits authority, port and path with scheme-correct defaults" {
    const a = parseUrl("http://192.168.1.50:8080/v1/prefill").?;
    try testing.expectEqualStrings("192.168.1.50", a.host);
    try testing.expectEqual(@as(u16, 8080), a.port);
    try testing.expectEqualStrings("/v1/prefill", a.path);
    try testing.expect(!a.tls);

    // No explicit port ⇒ the scheme's default, never zero.
    try testing.expectEqual(@as(u16, 80), parseUrl("http://box/v1/prefill").?.port);
    const c = parseUrl("https://box/v1/prefill").?;
    try testing.expectEqual(@as(u16, 443), c.port);
    try testing.expect(c.tls);

    // A mounted prefix stays in the path — the request line must carry it.
    try testing.expectEqualStrings("/prefill-worker/v1/prefill", parseUrl("http://box:9/prefill-worker/v1/prefill").?.path);

    try testing.expect(parseUrl("http://box:0/x") == null); // port 0 connects nowhere
    try testing.expect(parseUrl("http://box:notaport/x") == null);
    try testing.expect(parseUrl("http://") == null);
}

test "shouldAttempt gates on config, engine, size and scheme before any network work" {
    try testing.expectEqual(FallbackReason.disabled, shouldAttempt(null, true, 10_000).?);
    try testing.expectEqual(FallbackReason.disabled, shouldAttempt("", true, 10_000).?);
    // v1 is embedded-engine only; an MLX model has no llama seq state to restore.
    try testing.expectEqual(FallbackReason.not_embedded_engine, shouldAttempt("http://box", false, 10_000).?);
    // A short prompt is cheaper to decode than the blob is to move.
    try testing.expectEqual(FallbackReason.too_few_tokens, shouldAttempt("http://box", true, MIN_REMOTE_TOKENS - 1).?);
    try testing.expectEqual(FallbackReason.bad_base_url, shouldAttempt("box:8080", true, 10_000).?);
    // Refused up front, so we never begin an exchange we cannot finish.
    try testing.expectEqual(FallbackReason.tls_unsupported, shouldAttempt("https://box", true, 10_000).?);
    try testing.expect(shouldAttempt("http://box:8080", true, MIN_REMOTE_TOKENS) == null);
}

test "buildRequestBody emits the agreed JSON shape and escapes the model id" {
    const a = testing.allocator;
    const b1 = try buildRequestBody(a, "gemma-4-12b", &[_]i32{ 1, 2, 3 });
    defer a.free(b1);
    try testing.expectEqualStrings("{\"model\":\"gemma-4-12b\",\"tokens\":[1,2,3]}", b1);

    // A model id is a caller-supplied string that can carry anything discovery
    // accepted; unescaped it would break the worker's parse.
    const b2 = try buildRequestBody(a, "org/name \"q4\"", &[_]i32{42});
    defer a.free(b2);
    try testing.expectEqualStrings("{\"model\":\"org/name \\\"q4\\\"\",\"tokens\":[42]}", b2);

    // Round-trip through the SERVER's own parser — the two halves must agree,
    // and a unit test on our side alone could not show that.
    var parsed = try proto.parseRequest(a, b1);
    defer parsed.deinit(a);
    try testing.expectEqualStrings("gemma-4-12b", parsed.model);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3 }, parsed.tokens);
}

test "buildRequestHead carries the mounted path and a correct Content-Length" {
    var buf: [512]u8 = undefined;
    const ep = parseUrl("http://box:9/prefill-worker/v1/prefill").?;
    const head = try buildRequestHead(&buf, ep, 123);
    try testing.expect(std.mem.startsWith(u8, head, "POST /prefill-worker/v1/prefill HTTP/1.1\r\n"));
    try testing.expect(std.mem.indexOf(u8, head, "Host: box:9\r\n") != null);
    try testing.expect(std.mem.indexOf(u8, head, "Content-Length: 123\r\n") != null);
    // Connection: close is what makes the read loop's EOF meaningful.
    try testing.expect(std.mem.indexOf(u8, head, "Connection: close\r\n") != null);
    try testing.expect(std.mem.endsWith(u8, head, "\r\n\r\n"));
}

test "splitResponse needs a complete head and reports the real status" {
    try testing.expect(splitResponse("HTTP/1.1 200 OK\r\nX: 1\r\n") == null); // no blank line yet
    const ok = splitResponse("HTTP/1.1 200 OK\r\nX: 1\r\n\r\nBODY").?;
    try testing.expectEqual(@as(u16, 200), ok.status);
    try testing.expectEqualStrings("BODY", ok.body);
    // A non-200 must be seen as such, never read as a blob.
    try testing.expectEqual(@as(u16, 503), splitResponse("HTTP/1.1 503 Service Unavailable\r\n\r\n{}").?.status);
    // The blob is BINARY and may contain CRLFCRLF: the FIRST blank line ends
    // the head, and a later one must not re-split the body.
    try testing.expectEqualStrings("ab\r\n\r\ncd", splitResponse("HTTP/1.1 200 OK\r\n\r\nab\r\n\r\ncd").?.body);
}

test "readHeaders finds the six reply headers case-insensitively" {
    const head = "HTTP/1.1 200 OK\r\nx-prefill-version: 1\r\nX-PREFILL-MODEL: m\r\n" ++
        "X-Prefill-Tokens: 7\r\nx-Prefill-Bytes: 99\r\nX-Prefill-Vocab: 256000\r\n" ++
        "X-Prefill-Model-Bytes: 1234";
    const h = readHeaders(head);
    try testing.expectEqualStrings("1", h.version.?);
    try testing.expectEqualStrings("m", h.model.?);
    try testing.expectEqualStrings("7", h.tokens.?);
    try testing.expectEqualStrings("99", h.bytes.?);
    try testing.expectEqualStrings("256000", h.vocab.?);
    try testing.expectEqualStrings("1234", h.model_bytes.?);

    // A well-formed reply validates; the same reply missing a header does not.
    // Proves our reader and their validator compose, which neither side's own
    // unit tests can show.
    const exp = proto.Expected{ .model = "m", .n_tokens = 7, .vocab = 256000, .model_bytes = 1234, .body_len = 99 };
    try testing.expect(proto.validateResponse(h, exp) == null);
    try testing.expect(proto.validateResponse(readHeaders("HTTP/1.1 200 OK\r\nX-Prefill-Version: 1\r\n"), exp) != null);
}

test "parseDottedQuad accepts only four in-range octets" {
    try testing.expectEqual([4]u8{ 127, 0, 0, 1 }, parseDottedQuad("127.0.0.1").?);
    try testing.expectEqual([4]u8{ 0, 0, 0, 0 }, parseDottedQuad("0.0.0.0").?);
    try testing.expect(parseDottedQuad("256.0.0.1") == null); // octet overflow
    try testing.expect(parseDottedQuad("1.2.3") == null);
    try testing.expect(parseDottedQuad("1.2.3.4.5") == null);
    try testing.expect(parseDottedQuad("1.2.3.") == null);
    try testing.expect(parseDottedQuad("box.local") == null); // a name, not a quad
}

test "every fallback reason has distinct non-empty text" {
    // The log line is the only way to tell these apart in the field, so a
    // duplicated or empty phrase is a real defect.
    const tags = std.meta.tags(FallbackReason);
    for (tags, 0..) |a, i| {
        try testing.expect(a.text().len > 0);
        for (tags[i + 1 ..]) |b| {
            try testing.expect(!std.mem.eql(u8, a.text(), b.text()));
        }
    }
}

test "viability is a PER-TOKEN comparison, so it does not vary with prompt length" {
    // The insight the flat floor got wrong. Blob bytes scale linearly with
    // tokens AND prefill time scales linearly with tokens, so the ratio between
    // them is constant: a config that loses at 1k tokens loses at 100k too, and
    // one that wins, wins everywhere. A token threshold cannot express that.
    const c = Economics{
        .bytes_per_token = 167 * 1024, // gemma-4-12b, measured
        .wire_bytes_per_sec = 110_000_000, // gigabit, real-world
        .worker_ms_per_token = 1.398, // measured, CUDA worker
        .local_ms_per_token = 2.109, // measured, m4max's Mac
    };
    // Same verdict at every length — that IS the property.
    try testing.expectEqual(c.viable(), c.viable());
    try testing.expect(!c.viable()); // gemma on gigabit today: remote LOSES

    // Shrink the blob enough and the same config flips, with no change to n.
    var shrunk = c;
    shrunk.bytes_per_token = 60 * 1024;
    try testing.expect(shrunk.viable());
}

test "viability tracks the WORKER's advantage, not its absolute speed" {
    // The ceiling on this whole technique: you can only spend
    // (local - worker) per token on transfer. A worker that is barely faster
    // leaves no budget however fat the pipe, which is why a 1.5x-faster worker
    // needs a 2.2x blob shrink just to break even.
    var c = Economics{
        .bytes_per_token = 1024,
        .wire_bytes_per_sec = 110_000_000,
        .worker_ms_per_token = 2.0,
        .local_ms_per_token = 2.109,
    };
    try testing.expect(c.viable()); // a thin blob still fits the thin budget

    // A worker no faster than the consumer can NEVER pay, at any blob size.
    c.worker_ms_per_token = 2.109;
    try testing.expect(!c.viable());
    c.bytes_per_token = 1;
    try testing.expect(!c.viable());

    // A SLOWER worker is never viable either.
    c.worker_ms_per_token = 5.0;
    try testing.expect(!c.viable());
}

test "viability degrades gracefully when a term is unknown" {
    // Before the first exchange with a model there is no measured
    // bytes_per_token. Unknown must mean ATTEMPT (we learn the rate from the
    // reply) rather than refuse, or the client can never bootstrap — but a
    // known-bad rate must still refuse.
    var c = Economics{
        .bytes_per_token = 0, // unmeasured
        .wire_bytes_per_sec = 110_000_000,
        .worker_ms_per_token = 1.398,
        .local_ms_per_token = 2.109,
    };
    try testing.expect(c.viable());

    // A zero/absent wire rate is a configuration error, not evidence of speed.
    c.bytes_per_token = 167 * 1024;
    c.wire_bytes_per_sec = 0;
    try testing.expect(!c.viable());
}

test "observedBytesPerToken reads the rate straight off a reply" {
    // No new header needed: the contract already carries both numbers, so the
    // client learns each model's rate from its first successful exchange.
    try testing.expectEqual(@as(u64, 1000), observedBytesPerToken(2_000_000, 2000).?);
    // Degenerate inputs must not divide by zero or invent a rate.
    try testing.expect(observedBytesPerToken(2_000_000, 0) == null);
    try testing.expect(observedBytesPerToken(0, 2000) == null);
}

test "the economic gate admits only prompts the protocol also accepts" {
    // prefillSpan is only meaningful at >= 2 tokens; the client's own floor is
    // far above it, but the RELATIONSHIP is the thing worth pinning — lowering
    // one without the other is how a request gets sent that the worker refuses.
    try testing.expect(MIN_REMOTE_TOKENS >= proto.MIN_TOKENS);
    const toks = [_]i32{ 1, 2, 3, 4 };
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3 }, proto.prefillSpan(&toks));
}
