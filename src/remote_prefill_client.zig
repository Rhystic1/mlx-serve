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

/// Whether remote prefill can pay for itself at a given prompt length.
///
/// The blob is NOT proportional to the prompt. Measured on gemma-4-12b
/// (linux-x64, b10472, 2026-08-23) it is `fixed + slope x tokens`: a sliding-
/// window arch exports a WINDOW-worth of cells for its 43 sliding layers
/// whatever the prompt length, and only its 8 global layers grow with it.
///
///   f16   fixed ~335 MB   slope 16.4 KB/token
///   q8_0  fixed ~178 MB   slope  8.7 KB/token
///   (below the 1024 window the whole thing is still filling, ~344 KB/token)
///
/// That fixed term is why a token threshold turns out to be the RIGHT shape
/// after all — my earlier "no crossover can exist" reasoning was correct only
/// for the strictly-linear blob I assumed. The fixed cost amortizes over the
/// prompt, so there is a genuine break-even length:
///
///   transfer(n) = (fixed + slope*n) / wire
///   saving(n)   = n * (local_ms_per_token - worker_ms_per_token)
///   viable  <=>  fixed/wire  <  n * (budget - slope/wire)
///
/// It still turns on the WORKER'S ADVANTAGE: when `budget - slope/wire <= 0`
/// no prompt is long enough, because every extra token costs more to ship than
/// it saves. A worker only 1.5x faster leaves a third of local prefill to spend
/// on the wire however fat the pipe.
///
/// The model reproduces both independently measured results, which is the only
/// reason to trust it: M4 Max f16 needs 5455 tokens and measured a LOSS at
/// 3329; M4 mini f16 needs 595 and measured a WIN at 3525.
///
/// Still an approximation past the window in one direction: prefill carries an
/// O(n^2) attention term while the blob stays linear, so at long contexts the
/// real crossover arrives EARLIER than this predicts. Erring toward refusing a
/// config that would have won is the safe direction; do not "fix" it by
/// extrapolating rates measured at one length to another.
pub const Economics = struct {
    /// Window payload, paid whatever the prompt length. 0 with `slope` also 0
    /// means "not yet measured for this model".
    fixed_bytes: u64 = 0,
    /// Marginal blob bytes per prompt token past the window.
    slope_bytes_per_token: u64 = 0,
    /// Assumed usable wire throughput. 0 is a configuration error, never
    /// evidence about speed.
    wire_bytes_per_sec: u64 = 0,
    worker_ms_per_token: f64 = 0,
    local_ms_per_token: f64 = 0,

    fn measured(self: Economics) bool {
        return self.fixed_bytes != 0 or self.slope_bytes_per_token != 0;
    }

    /// Milliseconds of local prefill each token saves, net of shipping it.
    fn netPerTokenMs(self: Economics) f64 {
        const budget = self.local_ms_per_token - self.worker_ms_per_token;
        const slope_ms = @as(f64, @floatFromInt(self.slope_bytes_per_token)) /
            @as(f64, @floatFromInt(self.wire_bytes_per_sec)) * 1000.0;
        return budget - slope_ms;
    }

    /// Shortest prompt at which remote prefill pays. Null when no length does —
    /// either the worker is not faster, or each token costs more to ship than
    /// it saves.
    pub fn minViableTokens(self: Economics) ?usize {
        if (self.wire_bytes_per_sec == 0) return null;
        const net_ms = self.netPerTokenMs();
        if (net_ms <= 0) return null;
        const fixed_ms = @as(f64, @floatFromInt(self.fixed_bytes)) /
            @as(f64, @floatFromInt(self.wire_bytes_per_sec)) * 1000.0;
        return @intFromFloat(@ceil(fixed_ms / net_ms));
    }

    pub fn viable(self: Economics, n_tokens: usize) bool {
        if (self.wire_bytes_per_sec == 0) return false;
        // Unmeasured shape ⇒ attempt, and learn it from the replies. Refusing
        // here would stop the client ever bootstrapping a model's numbers.
        if (!self.measured()) return true;
        const min = self.minViableTokens() orelse return false;
        return n_tokens >= min;
    }
};

/// A model's LOCAL prefill rate on THIS machine, learned from its own local
/// prefills rather than configured.
///
/// viable() needs local ms/token, and neither a flag nor a load-time probe is
/// right: a flag is a number the operator cannot know (it is per model, per
/// machine, and moves with thermal state), and a probe pays a prefill nobody
/// asked for. The server already times every local prefill, so the rate is
/// free — an EMA over observations, exactly how the blob shape is learned from
/// replies.
///
/// Unmeasured means ATTEMPT, mirroring the blob-shape rule: refusing until
/// calibrated would mean a server that never remote-prefills until it has
/// already done the local work remote prefill exists to avoid.
///
/// The EMA is deliberately slow (1/8) because the thing it tracks is slow —
/// thermal state and memory pressure drift over minutes — while individual
/// prefills are noisy, especially a short one dominated by fixed overhead.
pub const LocalRateEma = struct {
    ms_per_token: f64 = 0,

    /// Samples below this are dominated by per-request overhead rather than
    /// per-token work, and would bias the rate upward.
    pub const MIN_SAMPLE_TOKENS: usize = 64;

    pub fn observe(self: *LocalRateEma, prefill_ms: f64, tokens_computed: usize) void {
        if (tokens_computed < MIN_SAMPLE_TOKENS or prefill_ms <= 0) return;
        const sample = prefill_ms / @as(f64, @floatFromInt(tokens_computed));
        self.ms_per_token = if (self.ms_per_token == 0)
            sample
        else
            self.ms_per_token * 0.875 + sample * 0.125;
    }

    pub fn known(self: LocalRateEma) bool {
        return self.ms_per_token > 0;
    }
};

/// Solve `fixed + slope*n` from two replies at different lengths.
///
/// One reply cannot separate the two terms, and treating a single
/// bytes/token average as the slope is wrong by 20x across the window (344
/// KB/token while filling vs 16.4 past it). Null when the samples cannot
/// determine a shape — same length, out of order, or an implied negative slope
/// (which means the two came from different configs and neither describes the
/// model).
pub fn solveBlobShape(bytes_a: u64, n_a: usize, bytes_b: u64, n_b: usize) ?struct { fixed: u64, slope: u64 } {
    if (n_a == n_b or n_a == 0 or n_b == 0) return null;
    const lo_n: u64 = @intCast(@min(n_a, n_b));
    const hi_n: u64 = @intCast(@max(n_a, n_b));
    const lo_b = if (n_a < n_b) bytes_a else bytes_b;
    const hi_b = if (n_a < n_b) bytes_b else bytes_a;
    if (hi_b < lo_b) return null; // a bigger prompt cannot ship fewer bytes
    const slope = (hi_b - lo_b) / (hi_n - lo_n);
    const grown = slope * hi_n;
    if (grown > hi_b) return null;
    return .{ .fixed = hi_b - grown, .slope = slope };
}

/// What to report as `cached_n` after a prefill.
///
/// `cached_n` means "reused for FREE from local KV", and prefill throughput is
/// (prompt - cached) / prefill_ms. A remote restore leaves the session holding
/// N-1 tokens, so reporting them as cached divides ONE token by the entire
/// round trip and reports a rate ~3000x too low — m4mini measured 0.093 tok/s
/// on a path whose real effective rate was ~300 (2026-08-23), the kind of
/// broken instrument that gets a working feature diagnosed as a catastrophic
/// regression.
///
/// Remotely-prefilled tokens were not free: they cost the exchange, and that
/// time IS billed into prefill_ns. The honest report is zero cache hits and a
/// prefill that paid for the whole prompt, which makes prompt_per_second the
/// effective end-to-end rate of the remote path.
pub fn cachedTokensAfterPrefill(remote_engaged: bool, local_reuse: usize) usize {
    return if (remote_engaged) 0 else local_reuse;
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
const L_KV_TYPE = "x-prefill-kv-type";
const L_SWA = "x-prefill-swa";

comptime {
    // If the shared module renames a header, these must move with it or the
    // lookup silently returns null and every reply "misses" a header.
    assertLowerOf(proto.H_VERSION, L_VERSION);
    assertLowerOf(proto.H_MODEL, L_MODEL);
    assertLowerOf(proto.H_TOKENS, L_TOKENS);
    assertLowerOf(proto.H_BYTES, L_BYTES);
    assertLowerOf(proto.H_VOCAB, L_VOCAB);
    assertLowerOf(proto.H_MODEL_BYTES, L_MODEL_BYTES);
    assertLowerOf(proto.H_KV_TYPE, L_KV_TYPE);
    assertLowerOf(proto.H_SWA, L_SWA);
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
        .kv_type = peers.headerValueCI(head, L_KV_TYPE),
        .swa = peers.headerValueCI(head, L_SWA),
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

test "readHeaders finds all eight reply headers case-insensitively" {
    const head = "HTTP/1.1 200 OK\r\nx-prefill-version: 2\r\nX-PREFILL-MODEL: m\r\n" ++
        "X-Prefill-Tokens: 7\r\nx-Prefill-Bytes: 99\r\nX-Prefill-Vocab: 256000\r\n" ++
        "X-Prefill-Model-Bytes: 1234\r\nx-PREFILL-kv-type: q8_0\r\nX-Prefill-Swa: windowed";
    const h = readHeaders(head);
    try testing.expectEqualStrings("2", h.version.?);
    try testing.expectEqualStrings("m", h.model.?);
    try testing.expectEqualStrings("7", h.tokens.?);
    try testing.expectEqualStrings("99", h.bytes.?);
    try testing.expectEqualStrings("256000", h.vocab.?);
    try testing.expectEqualStrings("1234", h.model_bytes.?);
    try testing.expectEqualStrings("q8_0", h.kv_type.?);
    try testing.expectEqualStrings("windowed", h.swa.?);

    // A well-formed reply validates; the same reply missing a header does not.
    // Proves our reader and their validator compose, which neither side's own
    // unit tests can show.
    const exp = proto.Expected{ .model = "m", .n_tokens = 7, .vocab = 256000, .model_bytes = 1234, .body_len = 99, .kv_type = "q8_0" };
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

// Measured on gemma-4-12b (linux-x64, b10472, 2026-08-23) and on the two Mac
// consumers. Every case below is a real number, so a change that breaks the
// model breaks against reality rather than against an invented threshold.
const GEMMA_F16 = Economics{
    .fixed_bytes = 335_000_000,
    .slope_bytes_per_token = 16 * 1024 + 410,
    .wire_bytes_per_sec = 110_000_000, // gigabit, real-world
    .worker_ms_per_token = 1.398, // RTX 5060 Ti CUDA worker
    .local_ms_per_token = 2.109, // M4 Max consumer
};

test "the model reproduces both independently measured outcomes" {
    // This is the only reason to trust it. The M4 Max measured a LOSS at 3329
    // tokens on f16, and the M4 mini measured a WIN at 3525 on the same config
    // and the same worker — a model that cannot tell those apart is worthless.
    const max_f16 = GEMMA_F16;
    try testing.expect(!max_f16.viable(3329)); // m4max: measured loss
    try testing.expect(max_f16.minViableTokens().? > 3329);

    var mini_f16 = GEMMA_F16;
    mini_f16.local_ms_per_token = 6.67; // M4 mini, measured 150 tok/s
    try testing.expect(mini_f16.viable(3525)); // m4mini: measured win
    try testing.expect(mini_f16.minViableTokens().? < 3525);
}

test "the FIXED term is what creates a crossover, so viability varies with n" {
    // The correction to my earlier model: I argued no token threshold could
    // express this, which was true only for the strictly-linear blob I assumed.
    // A sliding-window arch ships a window payload whatever the prompt length,
    // and that cost amortizes — so there IS a break-even length.
    const e = GEMMA_F16;
    const min = e.minViableTokens().?;
    try testing.expect(!e.viable(min - 1));
    try testing.expect(e.viable(min));
    try testing.expect(e.viable(min * 10));
}

test "no prompt is long enough when the worker has no advantage to sell" {
    // The ceiling on the whole technique. If each token costs more to ship than
    // it saves, amortizing the fixed cost cannot rescue it at any length.
    var e = GEMMA_F16;
    e.worker_ms_per_token = e.local_ms_per_token; // no faster
    try testing.expect(e.minViableTokens() == null);
    try testing.expect(!e.viable(1_000_000));

    e.worker_ms_per_token = 5.0; // slower
    try testing.expect(e.minViableTokens() == null);

    // Slope alone can also sink it: a per-token wire cost above the budget.
    e = GEMMA_F16;
    e.slope_bytes_per_token = 1024 * 1024; // 1 MB/token
    try testing.expect(e.minViableTokens() == null);
}

test "quantized KV moves the crossover, which is the point of the shrink round" {
    // q8_0 roughly halves both terms, so it should roughly halve the break-even
    // length — the lever that makes this viable on a strong consumer.
    const f16_min = GEMMA_F16.minViableTokens().?;
    var q8 = GEMMA_F16;
    q8.fixed_bytes = 178_000_000;
    q8.slope_bytes_per_token = 8 * 1024 + 700;
    const q8_min = q8.minViableTokens().?;
    try testing.expect(q8_min < f16_min);
    // And it brings the M4 Max's 3329-token prompt inside the viable range,
    // which f16 did not.
    try testing.expect(!GEMMA_F16.viable(3329));
    try testing.expect(q8.viable(3329));
}

test "viability degrades gracefully when the shape is unknown" {
    // Before any exchange with a model there is no measured shape. Unknown must
    // mean ATTEMPT — the shape is learned from replies, so refusing would stop
    // the client bootstrapping. A zero wire rate is a config error, not
    // evidence about speed, and must refuse.
    var e = Economics{ .wire_bytes_per_sec = 110_000_000, .worker_ms_per_token = 1.4, .local_ms_per_token = 6.67 };
    try testing.expect(e.viable(1000));
    e.wire_bytes_per_sec = 0;
    try testing.expect(!e.viable(1000));
    try testing.expect(e.minViableTokens() == null);
}

test "LocalRateEma learns the consumer's own rate and ignores unusable samples" {
    var ema = LocalRateEma{};
    try testing.expect(!ema.known());

    // A short prefill is dominated by fixed overhead, not per-token work —
    // taking it would bias the rate upward and refuse remote prefill that
    // would have paid.
    ema.observe(500.0, 10);
    try testing.expect(!ema.known());
    // Degenerate timings are not evidence either.
    ema.observe(0.0, 1000);
    ema.observe(-5.0, 1000);
    try testing.expect(!ema.known());

    // First real sample IS the estimate — there is nothing to blend with, and
    // starting from an invented prior would take many samples to shake off.
    ema.observe(23_820.0, 3525); // m4mini, measured
    try testing.expectApproxEqAbs(@as(f64, 6.757), ema.ms_per_token, 0.01);

    // Later samples move it slowly: the thing being tracked (thermal state) is
    // slow, while individual prefills are noisy.
    const before = ema.ms_per_token;
    ema.observe(100_000.0, 3525); // a wild outlier
    try testing.expect(ema.ms_per_token > before);
    try testing.expect(ema.ms_per_token < before * 1.6); // but not dragged to it
}

test "a learned local rate drives viability without any configuration" {
    // The end state: nothing is configured. The blob shape comes from replies,
    // the local rate from local prefills, and the two together decide.
    var ema = LocalRateEma{};
    ema.observe(23_820.0, 3525); // M4 mini
    var e = Economics{
        .fixed_bytes = 178_000_000, // q8_0, measured
        .slope_bytes_per_token = 8 * 1024 + 700,
        .wire_bytes_per_sec = 110_000_000,
        .worker_ms_per_token = 1.398,
        .local_ms_per_token = ema.ms_per_token,
    };
    try testing.expect(e.viable(3525));

    const mini_min = e.minViableTokens().?;

    // The SAME worker against a much faster consumer has less advantage to
    // sell, so it needs a longer prompt to amortize the fixed blob.
    var fast = LocalRateEma{};
    fast.observe(7_020.0, 3329); // M4 Max, measured
    e.local_ms_per_token = fast.ms_per_token;
    const max_min = e.minViableTokens().?;
    try testing.expect(max_min > mini_min);
    // ...but q8_0 still brings the M4 Max's real 3329-token prompt inside the
    // viable range, which is what m4max measured: remote won by ~0.8 s where
    // f16 had lost by ~2 s.
    try testing.expect(e.viable(3329));
}

test "solveBlobShape separates the window payload from the per-token slope" {
    // One reply cannot separate them, and a single bytes/token average is wrong
    // by 20x across the window — 344 KB/token while filling vs 16.4 past it.
    // Two of linux-x64's measured points must recover the shape they came from.
    const sol = solveBlobShape(360_000_000, 1518, 385_000_000, 3000).?;
    try testing.expect(sol.slope > 15_000 and sol.slope < 18_000); // ~16.4 KB
    try testing.expect(sol.fixed > 320_000_000 and sol.fixed < 345_000_000);

    // Degenerate or contradictory samples must yield nothing rather than a
    // confident wrong shape.
    try testing.expect(solveBlobShape(360_000_000, 1518, 385_000_000, 1518) == null);
    try testing.expect(solveBlobShape(360_000_000, 0, 385_000_000, 3000) == null);
    // A longer prompt shipping FEWER bytes means the two came from different
    // configs; neither describes the model.
    try testing.expect(solveBlobShape(385_000_000, 3000, 360_000_000, 4000) == null);
}

test "remotely-prefilled tokens are NOT reported as cache hits" {
    // `cached_n` means "reused for free from local KV", and prefill throughput
    // is (prompt - cached) / prefill_ms. A remote restore leaves the session
    // holding N-1 tokens, so reporting those as cached divides ONE token by the
    // whole round trip: m4mini measured 0.093 tok/s on a path whose real
    // effective rate was ~300 (2026-08-23). Those tokens were not free — they
    // cost the remote exchange, which is billed in prefill_ns — so the honest
    // report is that nothing was cached and the prefill paid for all of them.
    try testing.expectEqual(@as(usize, 0), cachedTokensAfterPrefill(true, 3524));
    // Without remote prefill the local reuse count is the truth, untouched.
    try testing.expectEqual(@as(usize, 3524), cachedTokensAfterPrefill(false, 3524));
    try testing.expectEqual(@as(usize, 0), cachedTokensAfterPrefill(false, 0));
}

test "the economic gate admits only prompts the protocol also accepts" {
    // prefillSpan is only meaningful at >= 2 tokens; the client's own floor is
    // far above it, but the RELATIONSHIP is the thing worth pinning — lowering
    // one without the other is how a request gets sent that the worker refuses.
    try testing.expect(MIN_REMOTE_TOKENS >= proto.MIN_TOKENS);
    const toks = [_]i32{ 1, 2, 3, 4 };
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3 }, proto.prefillSpan(&toks));
}
