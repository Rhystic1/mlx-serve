// Zig-side wrapper around llama.cpp's libllama, via the C shim in
// `lib/llama_shim/`. The C ABI lives in `src/llama_ffi.zig`; this module owns
// lifetimes and error mapping and exposes the same `sync`/`eval`/`argmax`/
// `sample` session shape as `src/arch/ds4.zig`, so the scheduler can drive a
// GGUF slot through either embedded engine with the same per-token loop.
//
// Routing: `src/main.zig` sends DeepSeek-V4-Flash GGUFs to ds4 (a bespoke
// engine for that architecture) and every other `.gguf` here.

const std = @import("std");
const ffi = @import("../llama_ffi.zig");
const log = @import("../log.zig");

pub const Error = error{
    EngineOpenFailed,
    SessionCreateFailed,
    SessionSyncFailed,
    SessionEvalFailed,
    EmbedFailed,
    TokenizeFailed,
    StateExportFailed,
    StateImportFailed,
    OutOfMemory,
};

pub const OpenOptions = struct {
    /// Layers to offload to Metal. 999 = "all" (every real model has fewer).
    n_gpu_layers: i32 = 999,
};

pub const LlamaEngine = struct {
    allocator: std.mem.Allocator,
    handle: *ffi.Engine,
    model_path_owned: [:0]u8,

    pub fn open(allocator: std.mem.Allocator, model_path: []const u8, opts: OpenOptions) Error!*LlamaEngine {
        const path_z = allocator.dupeSentinel(u8, model_path, 0) catch return Error.OutOfMemory;
        errdefer allocator.free(path_z);

        var err_buf: [256]u8 = undefined;
        const raw = ffi.mlx_llama_open(path_z.ptr, opts.n_gpu_layers, &err_buf, err_buf.len);
        if (raw == null) {
            log.err("[llama] open failed: {s} (model={s})\n", .{ std.mem.sliceTo(&err_buf, 0), model_path });
            return Error.EngineOpenFailed;
        }

        const wrapper = allocator.create(LlamaEngine) catch {
            ffi.mlx_llama_close(raw);
            return Error.OutOfMemory;
        };
        wrapper.* = .{
            .allocator = allocator,
            .handle = raw.?,
            .model_path_owned = path_z,
        };
        return wrapper;
    }

    pub fn close(self: *LlamaEngine) void {
        ffi.mlx_llama_close(self.handle);
        self.allocator.free(self.model_path_owned);
        self.allocator.destroy(self);
    }

    pub fn eosToken(self: *LlamaEngine) i32 {
        return ffi.mlx_llama_eos_token(self.handle);
    }

    pub fn isEog(self: *LlamaEngine, token: i32) bool {
        return ffi.mlx_llama_is_eog(self.handle, token);
    }

    pub fn nVocab(self: *LlamaEngine) i32 {
        return ffi.mlx_llama_n_vocab(self.handle);
    }

    /// Tokenize free-form text. `add_special` controls BOS/special insertion.
    /// Caller owns the returned slice and frees with `allocator.free`.
    pub fn tokenizeText(
        self: *LlamaEngine,
        allocator: std.mem.Allocator,
        text: []const u8,
        add_special: bool,
    ) Error![]i32 {
        if (text.len == 0) return allocator.alloc(i32, 0) catch Error.OutOfMemory;

        // First attempt with a generous estimate; on -required, grow and retry.
        var cap: i32 = @intCast(@min(text.len + 16, std.math.maxInt(i32)));
        while (true) {
            const buf = allocator.alloc(i32, @intCast(cap)) catch return Error.OutOfMemory;
            const n = ffi.mlx_llama_tokenize(
                self.handle,
                text.ptr,
                @intCast(@min(text.len, std.math.maxInt(i32))),
                add_special,
                true, // parse_special: honor template special tokens already in `text`
                buf.ptr,
                cap,
            );
            if (n >= 0) {
                return allocator.realloc(buf, @intCast(n)) catch buf[0..@intCast(n)];
            }
            // n < 0: -n is the required capacity. Grow and retry.
            allocator.free(buf);
            const needed = -n;
            if (needed <= cap) return Error.TokenizeFailed; // shouldn't happen; guard against a loop
            cap = needed;
        }
    }

    /// Single token → bytes lookup. Caller owns the returned buffer.
    /// Bytes are NOT NUL-terminated; the slice length is authoritative.
    pub fn detokenizeOne(self: *LlamaEngine, allocator: std.mem.Allocator, token_id: i32) Error![]u8 {
        var cap: i32 = 64;
        while (true) {
            const buf = allocator.alloc(u8, @intCast(cap)) catch return Error.OutOfMemory;
            const n = ffi.mlx_llama_token_to_piece(self.handle, token_id, buf.ptr, cap);
            if (n >= 0) {
                return allocator.realloc(buf, @intCast(n)) catch buf[0..@intCast(n)];
            }
            allocator.free(buf);
            const needed = -n;
            if (needed <= cap) return allocator.dupe(u8, "") catch Error.OutOfMemory;
            cap = needed;
        }
    }

    /// The model's embedded chat-template string (jinja source), or null.
    /// Borrowed; valid for the engine's lifetime. Prefer rendering this through
    /// mlx-serve's jinja engine (chat.zig) over `applyChatTemplate`.
    pub fn chatTemplate(self: *LlamaEngine) ?[]const u8 {
        const raw = ffi.mlx_llama_chat_template(self.handle);
        if (raw == null) return null;
        return std.mem.sliceTo(raw.?, 0);
    }

    pub const ChatTurn = struct { role: []const u8, content: []const u8 };

    /// Fallback chat rendering via llama_chat_apply_template (recognized formats
    /// only — not a full jinja parser). Returns the formatted prompt string;
    /// caller owns it. Use only when the jinja path is unavailable.
    pub fn applyChatTemplate(
        self: *LlamaEngine,
        allocator: std.mem.Allocator,
        turns: []const ChatTurn,
        add_assistant: bool,
    ) Error![]u8 {
        var roles = std.ArrayList([*:0]const u8).empty;
        defer roles.deinit(allocator);
        var contents = std.ArrayList([*:0]const u8).empty;
        defer contents.deinit(allocator);
        // Track the duped C strings so we can free them all at the end.
        var owned = std.ArrayList([:0]u8).empty;
        defer {
            for (owned.items) |s| allocator.free(s);
            owned.deinit(allocator);
        }

        for (turns) |t| {
            const role_z = allocator.dupeSentinel(u8, t.role, 0) catch return Error.OutOfMemory;
            owned.append(allocator, role_z) catch return Error.OutOfMemory;
            const content_z = allocator.dupeSentinel(u8, t.content, 0) catch return Error.OutOfMemory;
            owned.append(allocator, content_z) catch return Error.OutOfMemory;
            roles.append(allocator, role_z.ptr) catch return Error.OutOfMemory;
            contents.append(allocator, content_z.ptr) catch return Error.OutOfMemory;
        }

        var cap: i32 = 4096;
        while (true) {
            const buf = allocator.alloc(u8, @intCast(cap)) catch return Error.OutOfMemory;
            const n = ffi.mlx_llama_apply_chat_template(
                self.handle,
                roles.items.ptr,
                contents.items.ptr,
                @intCast(turns.len),
                add_assistant,
                buf.ptr,
                cap,
            );
            if (n < 0) {
                allocator.free(buf);
                return Error.TokenizeFailed;
            }
            if (n <= cap) {
                return allocator.realloc(buf, @intCast(n)) catch buf[0..@intCast(n)];
            }
            // n > cap: required size returned; grow and retry.
            allocator.free(buf);
            cap = n;
        }
    }

    pub fn createSession(self: *LlamaEngine, ctx_size: i32) Error!*LlamaSession {
        return self.createSessionWithKvQuant(ctx_size, 0, 0);
    }

    /// Plan 5 #2: create a session with non-default ggml types for the K and
    /// V halves of the KV cache. `type_k` / `type_v` are ggml_type integers
    /// (see `llama_ffi.GgmlType`). Passing 0 keeps the libllama default
    /// (F16 in current versions). Non-default values automatically enable
    /// flash attention in the shim (required for quantized KV).
    pub fn createSessionWithKvQuant(
        self: *LlamaEngine,
        ctx_size: i32,
        type_k: i32,
        type_v: i32,
    ) Error!*LlamaSession {
        var err_buf: [256]u8 = undefined;
        const raw = ffi.mlx_llama_session_create_kv_quant(
            self.handle,
            ctx_size,
            type_k,
            type_v,
            &err_buf,
            err_buf.len,
        );
        if (raw == null) {
            log.err("[llama] session_create_kv_quant failed: {s} (ctx={d}, type_k={d}, type_v={d})\n", .{
                std.mem.sliceTo(&err_buf, 0), ctx_size, type_k, type_v,
            });
            return Error.SessionCreateFailed;
        }
        const sess = self.allocator.create(LlamaSession) catch {
            ffi.mlx_llama_session_free(raw);
            return Error.OutOfMemory;
        };
        sess.* = .{
            .allocator = self.allocator,
            .engine = self,
            .handle = raw.?,
            .ctx_size = ctx_size,
            .resident = .empty,
        };
        return sess;
    }

    /// Width of one embedding vector for this checkpoint.
    pub fn nEmbd(self: *LlamaEngine) i32 {
        return ffi.mlx_llama_n_embd(self.handle);
    }

    /// Context length this checkpoint was trained at, or 0 if it does not say.
    pub fn nCtxTrain(self: *LlamaEngine) i32 {
        return ffi.mlx_llama_n_ctx_train(self.handle);
    }

    /// Is this an EMBEDDING checkpoint rather than a generative one? Answered
    /// from the POOLING type it declares (see the shim -- NOT
    /// `llama_model_has_encoder`, which answers false for every BERT-family
    /// embedder). Drives what the model ADVERTISES: nomic-embed cannot chat,
    /// and a server that lists it as a chat model sends every discovery client
    /// somewhere it will only get nonsense.
    pub fn isEncoderOnly(self: *LlamaEngine) bool {
        return ffi.mlx_llama_is_encoder_only(self.handle);
    }

    /// Create a session for embeddings. Separate from a generation session by
    /// necessity: `embeddings` and `pooling_type` are context-creation
    /// parameters, and a pooled context exposes no per-token logits to sample.
    /// Freed with the same `free()`.
    pub fn createEmbedSession(self: *LlamaEngine, ctx_size: i32) Error!*LlamaSession {
        var err_buf: [256]u8 = undefined;
        const raw = ffi.mlx_llama_embed_session_create(self.handle, ctx_size, &err_buf, err_buf.len);
        if (raw == null) {
            log.err("[llama] embed_session_create failed: {s} (ctx={d})\n", .{
                std.mem.sliceTo(&err_buf, 0), ctx_size,
            });
            return Error.SessionCreateFailed;
        }
        const sess = self.allocator.create(LlamaSession) catch {
            ffi.mlx_llama_session_free(raw);
            return Error.OutOfMemory;
        };
        sess.* = .{
            .allocator = self.allocator,
            .engine = self,
            .handle = raw.?,
            .ctx_size = ctx_size,
            .resident = .empty,
        };
        return sess;
    }
};

/// Plan 5 #2: KV quant for the llama.cpp engine. Mapped from the CLI/body
/// flag onto ggml types; F16 is the dense default, Q8_0 halves KV bytes,
/// Q4_0 quarters them. See `mlx_llama_session_create_kv_quant`.
pub const LlamaKvQuant = enum(u8) {
    off,    // F16 (default)
    q8,     // Q8_0 (~2x compression, near-lossless on most archs)
    q4,     // Q4_0 (~4x compression, some quality impact)

    pub fn fromString(s: []const u8) ?LlamaKvQuant {
        if (std.mem.eql(u8, s, "off") or std.mem.eql(u8, s, "f16") or std.mem.eql(u8, s, "F16")) return .off;
        if (std.mem.eql(u8, s, "8") or std.mem.eql(u8, s, "q8") or std.mem.eql(u8, s, "Q8_0") or std.mem.eql(u8, s, "q8_0")) return .q8;
        if (std.mem.eql(u8, s, "4") or std.mem.eql(u8, s, "q4") or std.mem.eql(u8, s, "Q4_0") or std.mem.eql(u8, s, "q4_0")) return .q4;
        return null;
    }

    pub fn ggmlType(self: LlamaKvQuant) i32 {
        return switch (self) {
            .off => 0, // shim treats 0 as "use libllama default"
            .q8 => ffi.GgmlType.Q8_0,
            .q4 => ffi.GgmlType.Q4_0,
        };
    }

    pub fn label(self: LlamaKvQuant) []const u8 {
        return switch (self) {
            .off => "F16",
            .q8 => "Q8_0",
            .q4 => "Q4_0",
        };
    }
};

/// Decode `mlx_llama_session_trim`'s return value: did it leave the first
/// `n_keep` tokens resident (false), or refuse and clear the whole session
/// (true)? Split out so the "1 means start over" half of the contract has a
/// guard that does not need a hybrid GGUF on disk -- the failure it exists for
/// only reproduces on a recurrent checkpoint, and reading the value as a plain
/// ok/error code silently reinstates the bug.
pub fn trimClearedSession(rc: i32) bool {
    return rc == 1;
}

test "trimClearedSession: 1 means the session was cleared, 0 means trimmed" {
    try std.testing.expect(trimClearedSession(1));
    try std.testing.expect(!trimClearedSession(0));
    // A negative code is not a "cleared" report -- the session is untouched.
    try std.testing.expect(!trimClearedSession(-1));
}

/// Length of the longest common prefix of two token sequences. Pure helper so
/// the prompt-prefix reuse logic in `LlamaSession.sync` is unit-testable without
/// a model. An off-by-one here would corrupt KV reuse, so it's covered directly.
pub fn commonPrefixLen(a: []const i32, b: []const i32) usize {
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i < n and a[i] == b[i]) : (i += 1) {}
    return i;
}

pub const LlamaSession = struct {
    allocator: std.mem.Allocator,
    engine: *LlamaEngine,
    handle: *ffi.Session,
    ctx_size: i32,
    /// Token ids currently resident in the KV cache (prompt + every token fed via
    /// `eval`), in position order. `sync` diffs the next prompt against this to
    /// reuse the common prefix; it always mirrors the C session's KV exactly.
    resident: std.ArrayList(i32),
    /// How many times a partial trim was REFUSED by the underlying memory and
    /// the session cleared instead (see `trimClearedSession`). Non-zero means
    /// this checkpoint's memory is recurrent/hybrid and cannot rewind, so a
    /// diverging prompt cold-prefills by necessity. Kept as a counter rather
    /// than a bool so a test can tell "never happened" from "happened once".
    trim_refusals: u32 = 0,

    pub fn free(self: *LlamaSession) void {
        self.resident.deinit(self.allocator);
        ffi.mlx_llama_session_free(self.handle);
        self.allocator.destroy(self);
    }

    pub fn pos(self: *LlamaSession) i32 {
        return ffi.mlx_llama_session_pos(self.handle);
    }

    /// Serialize the resident sequence state (remote prefill, server side).
    /// llama.cpp's own backend-neutral blob: what a CUDA box prefilled, a Metal
    /// decoder can `importState`. Coupled to the llama.cpp BUILD + GGUF file.
    /// Caller frees. Fails on an empty session -- an empty blob is not a state.
    pub fn exportState(self: *LlamaSession, allocator: std.mem.Allocator) Error![]u8 {
        const need = ffi.mlx_llama_session_state_size(self.handle);
        if (need == 0) return Error.StateExportFailed;
        const buf = try allocator.alloc(u8, need);
        errdefer allocator.free(buf);
        const got = ffi.mlx_llama_session_state_get(self.handle, buf.ptr, buf.len);
        if (got == 0) return Error.StateExportFailed;
        return buf[0..got];
    }

    /// Restore a blob from `exportState` as THIS session's whole state, with
    /// `tokens` = the ids it was prefilled from. The session is cleared first.
    /// Sets the C-side position to `tokens.len` (the shim owns that counter
    /// separately from llama's KV -- see llama_shim.h) AND the `resident`
    /// mirror to `tokens`, so a following `sync(prompt)` finds the prompt
    /// resident, backs off one position and re-decodes the last token for
    /// logits (the blob carries none). On failure the session is left EMPTY.
    pub fn importState(self: *LlamaSession, blob: []const u8, tokens: []const i32) Error!void {
        self.resident.clearRetainingCapacity();
        if (blob.len == 0 or tokens.len == 0) return Error.StateImportFailed;
        var err_buf: [256]u8 = undefined;
        const rc = ffi.mlx_llama_session_state_set(self.handle, blob.ptr, blob.len, @intCast(tokens.len), &err_buf, err_buf.len);
        if (rc != 0) {
            log.warn("[llama] state import failed: {s}\n", .{std.mem.sliceTo(&err_buf, 0)});
            return Error.StateImportFailed;
        }
        try self.resident.appendSlice(self.allocator, tokens);
    }

    /// Sync the KV cache to `prompt_ids`, reusing the longest prefix already
    /// resident from a previous request. Trims the divergent tail, decodes only
    /// the suffix, and updates the resident-token mirror. Returns the number of
    /// tokens reused (the cached prefix length) so the caller can report it.
    ///
    /// At least the final prompt token is always (re)decoded so fresh logits
    /// exist for sampling — even when the whole prompt is already resident we
    /// back off one position. Decoding the suffix continues from the trim point
    /// because libllama tracks positions from the KV state.
    pub fn sync(self: *LlamaSession, prompt_ids: []const i32) Error!i32 {
        var common = commonPrefixLen(self.resident.items, prompt_ids);
        if (common == prompt_ids.len and common > 0) common -= 1;

        if (common < self.resident.items.len) {
            const rc = ffi.mlx_llama_session_trim(self.handle, @intCast(common));
            if (trimClearedSession(rc)) {
                // Recurrent/hybrid memory refused the partial trim and cleared
                // itself instead (see llama_shim.h). The prefix we were about
                // to reuse is gone, so this request cold-prefills. Logged at
                // debug because it is the NORMAL outcome for such a checkpoint
                // whenever a prompt diverges from what is resident -- a
                // straight continuation still matches as a pure prefix and
                // never reaches this branch at all.
                log.debug("[llama] partial trim refused (recurrent memory); cold prefill\n", .{});
                self.trim_refusals +|= 1;
                self.resident.clearRetainingCapacity();
                common = 0;
            } else {
                self.resident.shrinkRetainingCapacity(common);
            }
        }

        const suffix = prompt_ids[common..];
        if (suffix.len > 0) {
            var err_buf: [256]u8 = undefined;
            const rc = ffi.mlx_llama_session_sync(
                self.handle,
                suffix.ptr,
                @intCast(suffix.len),
                &err_buf,
                err_buf.len,
            );
            if (rc != 0) {
                log.err("[llama] session_sync rc={d} err={s}\n", .{ rc, std.mem.sliceTo(&err_buf, 0) });
                return Error.SessionSyncFailed;
            }
            self.resident.appendSlice(self.allocator, suffix) catch return Error.OutOfMemory;
        }
        return @intCast(common);
    }

    /// Drop all resident KV (and the mirror). Used when the model wants a clean
    /// slate before a cold prefill.
    pub fn reset(self: *LlamaSession) void {
        ffi.mlx_llama_session_reset(self.handle);
        self.resident.clearRetainingCapacity();
    }

    /// `sync` with a one-shot defense against libllama transients (the
    /// `init_batch: failed to prepare attention ubatches` / `decode: failed to
    /// find a memory slot for batch of size N` class). On any failure we drop
    /// the resident state and retry once with a cold prefill — we lose this
    /// single request's prefix-reuse savings but the client gets a normal
    /// response instead of a 500. If the retry also fails the session is left
    /// clean so the *next* request can cold-prefill into a known-good state.
    ///
    /// Behaves identically to `sync` on the happy path, including the returned
    /// cached-prefix length.
    pub fn syncWithFallback(self: *LlamaSession, prompt_ids: []const i32) Error!i32 {
        return self.sync(prompt_ids) catch |err| blk: {
            log.warn(
                "[llama] sync failed ({s}); resetting session and retrying cold\n",
                .{@errorName(err)},
            );
            self.reset();
            break :blk self.sync(prompt_ids) catch |err2| {
                self.reset();
                return err2;
            };
        };
    }

    /// Advance the KV cache by one already-sampled token.
    pub fn eval(self: *LlamaSession, token: i32) Error!void {
        var err_buf: [256]u8 = undefined;
        const rc = ffi.mlx_llama_session_eval(self.handle, token, &err_buf, err_buf.len);
        if (rc != 0) {
            log.err("[llama] session_eval rc={d} err={s}\n", .{ rc, std.mem.sliceTo(&err_buf, 0) });
            return Error.SessionEvalFailed;
        }
        self.resident.append(self.allocator, token) catch return Error.OutOfMemory;
    }

    /// Embed one token sequence. Allocates and returns the vector.
    ///
    /// Only valid on a session from `createEmbedSession`. The session's memory
    /// is cleared inside the shim first, so the result depends on `ids` alone
    /// and a batch of inputs cannot bleed into each other -- which also means
    /// an embed session keeps no `resident` mirror worth maintaining.
    pub fn embed(self: *LlamaSession, allocator: std.mem.Allocator, ids: []const i32) Error![]f32 {
        const n_embd = self.engine.nEmbd();
        if (n_embd <= 0) return Error.EmbedFailed;
        const out = allocator.alloc(f32, @intCast(n_embd)) catch return Error.OutOfMemory;
        errdefer allocator.free(out);
        var err_buf: [256]u8 = undefined;
        const n = ffi.mlx_llama_session_embed(
            self.handle,
            ids.ptr,
            @intCast(ids.len),
            out.ptr,
            n_embd,
            &err_buf,
            err_buf.len,
        );
        if (n != n_embd) {
            log.err("[llama] session_embed failed: {s}\n", .{std.mem.sliceTo(&err_buf, 0)});
            return Error.EmbedFailed;
        }
        // L2-normalize. The MLX encoder normalizes every row it returns, and
        // `/v1/embeddings` truncates-then-renormalizes for the `dimensions`
        // option assuming unit input, so an unnormalized vector here would make
        // the same server's embeddings incomparable across backends. llama.cpp
        // pools without normalizing (nomic-embed measures ~22), so it is ours
        // to do.
        //
        // Accumulate in f64: a 4096-wide sum of squares in f32 loses enough to
        // move the last digits of every component.
        var mag: f64 = 0;
        for (out) |v| mag += @as(f64, v) * @as(f64, v);
        // A genuinely zero vector cannot be normalized; hand it back as-is
        // rather than dividing by zero and returning NaNs that look like data.
        if (mag > 0) {
            const inv = 1.0 / @sqrt(mag);
            for (out) |*v| v.* = @floatCast(@as(f64, v.*) * inv);
        }

        // The shim cleared the memory, so nothing of `ids` is resident and the
        // mirror must say so -- a stale mirror would make a later `sync` on
        // this handle reuse a prefix that is not there.
        self.resident.clearRetainingCapacity();
        return out;
    }

    pub fn argmax(self: *LlamaSession) i32 {
        return ffi.mlx_llama_session_argmax(self.handle);
    }

    pub fn sample(self: *LlamaSession, temperature: f32, top_k: i32, top_p: f32, min_p: f32, rng: *u64) i32 {
        return ffi.mlx_llama_session_sample(self.handle, temperature, top_k, top_p, min_p, rng);
    }
};

// ── Tests ────────────────────────────────────────────────────────────────
// Real-model tests gate on LLAMA_TEST_MODEL (a path to a small .gguf), matching
// the UD_MOE_MODEL / PLD_TEST_MODEL convention. Without it they skip so CI
// without the fixture stays green.

fn testModelPath() ?[]const u8 {
    // libc getenv (same idiom as src/generate.zig / src/arch/ds4.zig); the
    // returned pointer is owned by the environment, so no free is needed.
    const raw = std.c.getenv("LLAMA_TEST_MODEL") orelse return null;
    const slice = std.mem.sliceTo(raw, 0);
    return if (slice.len == 0) null else slice;
}

test "llama: tokenize round-trip and short greedy decode" {
    const allocator = std.testing.allocator;
    const path = testModelPath() orelse return error.SkipZigTest;

    var engine = try LlamaEngine.open(allocator, path, .{});
    defer engine.close();

    try std.testing.expect(engine.nVocab() > 0);
    try std.testing.expect(engine.eosToken() >= 0);

    const ids = try engine.tokenizeText(allocator, "The capital of France is", true);
    defer allocator.free(ids);
    try std.testing.expect(ids.len > 0);

    // Every token detokenizes to some (possibly empty) byte slice without error.
    const piece = try engine.detokenizeOne(allocator, ids[ids.len - 1]);
    defer allocator.free(piece);

    var sess = try engine.createSession(2048);
    defer sess.free();

    const cached0 = try sess.sync(ids);
    try std.testing.expectEqual(@as(i32, 0), cached0); // cold session: nothing reused
    const first = sess.argmax();
    try std.testing.expect(first >= 0 and first < engine.nVocab());

    // Decode a few greedy tokens; the loop must advance position and stay valid.
    var produced: usize = 0;
    var tok = first;
    while (produced < 5 and !engine.isEog(tok)) : (produced += 1) {
        try sess.eval(tok);
        tok = sess.argmax();
        try std.testing.expect(tok >= 0 and tok < engine.nVocab());
    }
    try std.testing.expect(sess.pos() >= @as(i32, @intCast(ids.len)));
}

// Model-gated: `syncWithFallback` is the public entry used by the scheduler.
// Happy path — it must return the same cached-prefix length and leave the
// session in the same state as plain `sync`. The retry-on-error branch is
// covered by the integration test `tests/test_llama_persistent_session.sh`,
// which exercises the full HTTP path through the scheduler.
test "llama: syncWithFallback matches sync on the happy path" {
    const allocator = std.testing.allocator;
    const path = testModelPath() orelse return error.SkipZigTest;

    var engine = try LlamaEngine.open(allocator, path, .{});
    defer engine.close();

    const a = try engine.tokenizeText(allocator, "Once upon a time, in a", true);
    defer allocator.free(a);
    const b = try engine.tokenizeText(allocator, "Once upon a time, in a galaxy far away", true);
    defer allocator.free(b);

    var sess_plain = try engine.createSession(2048);
    defer sess_plain.free();
    var sess_fb = try engine.createSession(2048);
    defer sess_fb.free();

    // Cold: both return 0, leave the resident mirror == the prompt.
    try std.testing.expectEqual(
        try sess_plain.sync(a),
        try sess_fb.syncWithFallback(a),
    );
    try std.testing.expectEqualSlices(i32, sess_plain.resident.items, sess_fb.resident.items);

    // Warm: prefix reuse, same cached-prefix count and same final resident.
    const c1 = try sess_plain.sync(b);
    const c2 = try sess_fb.syncWithFallback(b);
    try std.testing.expectEqual(c1, c2);
    try std.testing.expectEqualSlices(i32, sess_plain.resident.items, sess_fb.resident.items);

    // After reset both go back to cold, and syncWithFallback still works.
    sess_fb.reset();
    try std.testing.expectEqual(@as(usize, 0), sess_fb.resident.items.len);
    const c3 = try sess_fb.syncWithFallback(a);
    try std.testing.expectEqual(@as(i32, 0), c3);
}

// Model-gated regression: the SAME prompt served twice. This is the shape every
// multi-turn conversation and every retry hits, and it is the one `sync` branch
// the extend-the-prefix test above never reaches — there, the new prompt is
// LONGER than what is resident, so `common < prompt.len` and the back-off never
// runs. Here the whole prompt is already resident AND a generated tail sits
// past it, so `sync` must back off one position, trim the tail plus that
// position, and re-decode it. Get the trim wrong and the request answers from
// the wrong position: live on Windows (2026-08-20) the second identical request
// echoed PROMPT tokens back and the third returned an empty completion with
// `finish_reason: "stop"`.
//
// The bar is the FIRST token, not the whole continuation: it is the token the
// restored position is supposed to predict, so it convicts a position error on
// its own, and it cannot drift the way a longer greedy run can.
test "llama: re-serving an identical prompt over a generated tail restores the same logits" {
    const allocator = std.testing.allocator;
    const path = testModelPath() orelse return error.SkipZigTest;

    var engine = try LlamaEngine.open(allocator, path, .{});
    defer engine.close();

    const prompt = try engine.tokenizeText(allocator, "The history of the Roman Empire is long and", true);
    defer allocator.free(prompt);

    var sess = try engine.createSession(4096);
    defer sess.free();

    // Turn 1: cold prefill, then generate a few tokens so the resident KV holds
    // prompt + generated — the state a finished request leaves behind.
    _ = try sess.sync(prompt);
    const cold_first = sess.argmax();
    var tok = cold_first;
    var i: usize = 0;
    while (i < 4 and !engine.isEog(tok)) : (i += 1) {
        try sess.eval(tok);
        tok = sess.argmax();
    }
    try std.testing.expect(sess.resident.items.len > prompt.len);

    // Turn 2: the identical prompt. Every token of it is resident, so this is
    // the back-off-and-trim path.
    const cached = try sess.sync(prompt);
    // How much is reused is memory-dependent and not the invariant: a KV cache
    // keeps everything but the backed-off position, while a recurrent/hybrid
    // one cannot rewind at all and legitimately starts from zero. What must
    // hold either way is the state below.
    try std.testing.expect(cached == @as(i32, @intCast(prompt.len - 1)) or cached == 0);
    // The session must be back to exactly the prompt — no generated tail left.
    try std.testing.expectEqualSlices(i32, prompt, sess.resident.items);
    try std.testing.expectEqual(@as(i32, @intCast(prompt.len)), sess.pos());
    // ...and predict what the cold prefill predicted.
    try std.testing.expectEqual(cold_first, sess.argmax());
}

// Model-gated: embeddings through llama.cpp's own embedding API.
//
// Set LLAMA_TEST_EMBED_MODEL to an EMBEDDING gguf (e.g. nomic-embed-text);
// pointing it at a chat model exercises the no-pooling arm instead, which is a
// different contract, so the two are separate variables on purpose.
fn testEmbedModelPath() ?[]const u8 {
    const raw = std.c.getenv("LLAMA_TEST_EMBED_MODEL") orelse return null;
    const slice = std.mem.sliceTo(raw, 0);
    return if (slice.len == 0) null else slice;
}

test "llama: embeddings are deterministic, input-dependent, and isolated" {
    const allocator = std.testing.allocator;
    const path = testEmbedModelPath() orelse return error.SkipZigTest;

    var engine = try LlamaEngine.open(allocator, path, .{});
    defer engine.close();

    const n_embd = engine.nEmbd();
    try std.testing.expect(n_embd > 0);

    var sess = try engine.createEmbedSession(512);
    defer sess.free();

    const a1 = try engine.tokenizeText(allocator, "the quick brown fox", true);
    defer allocator.free(a1);
    const b1 = try engine.tokenizeText(allocator, "quantum chromodynamics lattice gauge theory", true);
    defer allocator.free(b1);

    const ea = try sess.embed(allocator, a1);
    defer allocator.free(ea);
    const eb = try sess.embed(allocator, b1);
    defer allocator.free(eb);
    try std.testing.expectEqual(@as(usize, @intCast(n_embd)), ea.len);

    // Finite, and L2-NORMALIZED. Unit length is the server's embedding
    // contract, not a detail: the MLX encoder normalizes every row, and
    // `/v1/embeddings`'s `dimensions` option truncates and then RE-normalizes
    // on the assumption that it started from a unit vector. A raw llama.cpp
    // pooled vector is not normalized (nomic-embed measures ~22), so without
    // this the same server hands out two incomparable kinds of embedding
    // depending on which backend loaded the model.
    var mag: f64 = 0;
    for (ea) |v| {
        try std.testing.expect(std.math.isFinite(v));
        mag += @as(f64, v) * @as(f64, v);
    }
    try std.testing.expect(mag > 0);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), @sqrt(mag), 1e-4);

    // Different inputs must differ. Without this an embedder that returns a
    // constant (wrong pooling, uninitialised buffer) passes everything else.
    var same = true;
    for (ea, eb) |x, y| {
        if (x != y) {
            same = false;
            break;
        }
    }
    try std.testing.expect(!same);

    // The SAME input must reproduce, and -- the actual regression risk -- it
    // must reproduce AFTER another input has been embedded on this session. If
    // the shim did not clear memory between calls, this second `a` would be
    // embedded as "b then a" and quietly differ.
    const ea2 = try sess.embed(allocator, a1);
    defer allocator.free(ea2);
    try std.testing.expectEqualSlices(f32, ea, ea2);
}

test "commonPrefixLen: shared prefix, divergence, and bounds" {
    const a = [_]i32{ 1, 2, 3, 4, 5 };
    // Identical → full length.
    try std.testing.expectEqual(@as(usize, 5), commonPrefixLen(&a, &a));
    // Shared 3-token prefix then diverge.
    try std.testing.expectEqual(@as(usize, 3), commonPrefixLen(&a, &[_]i32{ 1, 2, 3, 9, 9 }));
    // b is a strict prefix of a → bounded by b.len.
    try std.testing.expectEqual(@as(usize, 2), commonPrefixLen(&a, &[_]i32{ 1, 2 }));
    // Diverge at token 0 → 0.
    try std.testing.expectEqual(@as(usize, 0), commonPrefixLen(&a, &[_]i32{ 9, 2, 3 }));
    // Empty inputs → 0, no out-of-bounds.
    try std.testing.expectEqual(@as(usize, 0), commonPrefixLen(&a, &[_]i32{}));
    try std.testing.expectEqual(@as(usize, 0), commonPrefixLen(&[_]i32{}, &a));
}

// Model-gated: prove prompt-prefix reuse produces byte-identical greedy output
// to a cold decode, and that the second request reuses the shared prefix. Guards
// the KV-trim off-by-one (the dangerous failure mode of Phase 3).
test "llama: prefix reuse is byte-identical to cold decode" {
    const allocator = std.testing.allocator;
    const path = testModelPath() orelse return error.SkipZigTest;

    var engine = try LlamaEngine.open(allocator, path, .{});
    defer engine.close();

    const shared = try engine.tokenizeText(allocator, "The history of the Roman Empire is long and", true);
    defer allocator.free(shared);
    // A divergent prompt that extends the shared prefix.
    const full = try engine.tokenizeText(allocator, "The history of the Roman Empire is long and storied, beginning with", true);
    defer allocator.free(full);
    try std.testing.expect(commonPrefixLen(shared, full) >= shared.len - 1);

    const N = 12; // first-N greedy tokens; short enough to stay byte-stable

    // Cold reference: a fresh session decodes `full` from scratch.
    var cold_out: [N]i32 = undefined;
    {
        var sess = try engine.createSession(4096);
        defer sess.free();
        _ = try sess.sync(full);
        var tok = sess.argmax();
        var i: usize = 0;
        while (i < N) : (i += 1) {
            cold_out[i] = tok;
            if (engine.isEog(tok)) break;
            try sess.eval(tok);
            tok = sess.argmax();
        }
    }

    // Warm: one session, prime with `shared`, then sync `full` (reuses prefix).
    var sess = try engine.createSession(4096);
    defer sess.free();

    _ = try sess.sync(shared);
    // Generate a couple tokens so the resident KV holds prompt + generated, the
    // realistic multi-turn shape the reuse must survive.
    var warm_tok = sess.argmax();
    try sess.eval(warm_tok);
    warm_tok = sess.argmax();
    try sess.eval(warm_tok);

    const cached = try sess.sync(full);
    // Reuse is a property of the MEMORY, not of correctness: a recurrent/hybrid
    // checkpoint refuses the partial trim and cold-prefills instead, which is
    // exactly why it must still produce the cold tokens below. On a plain KV
    // cache a zero here would be the trim bug this test exists for.
    try std.testing.expect(cached > 0 or sess.trim_refusals > 0);
    try std.testing.expect(cached <= @as(i32, @intCast(full.len)));

    var tok = sess.argmax();
    var i: usize = 0;
    while (i < N) : (i += 1) {
        try std.testing.expectEqual(cold_out[i], tok);
        if (engine.isEog(tok)) break;
        try sess.eval(tok);
        tok = sess.argmax();
    }
}

test "llama: sequence state round-trips between sessions (remote prefill interchange)" {
    const allocator = std.testing.allocator;
    const path = testModelPath() orelse return error.SkipZigTest;
    const remote_prefill = @import("../remote_prefill.zig");

    var engine = try LlamaEngine.open(allocator, path, .{});
    defer engine.close();
    const ids = try engine.tokenizeText(allocator, "The capital of France is the city of", true);
    defer allocator.free(ids);
    try std.testing.expect(ids.len >= 3);

    // Reference: one session, whole prompt, the next token it wants.
    var reference = try engine.createSession(2048);
    defer reference.free();
    _ = try reference.sync(ids);
    const expect_next = reference.argmax();

    // Worker: prefill ALL BUT THE LAST token (the wire contract -- see
    // remote_prefill.prefillSpan), export. pos must equal what was fed or the
    // consumer's next decode lands at the wrong position.
    const span = remote_prefill.prefillSpan(ids);
    var producer = try engine.createSession(2048);
    defer producer.free();
    _ = try producer.sync(span);
    try std.testing.expectEqual(@as(i32, @intCast(span.len)), producer.pos());
    const blob = try producer.exportState(allocator);
    defer allocator.free(blob);
    try std.testing.expect(blob.len > 0);

    // An empty session has no state to export.
    var empty = try engine.createSession(2048);
    defer empty.free();
    try std.testing.expectError(Error.StateExportFailed, empty.exportState(allocator));

    // Consumer: import, then the ORDINARY sync of the full prompt. It must
    // reuse exactly N-1 (no trim -- a recurrent checkpoint like LFM2 refuses
    // one and would cold-prefill instead) and decode only the last token,
    // landing on the SAME next token as the reference.
    var consumer = try engine.createSession(2048);
    defer consumer.free();
    try consumer.importState(blob, span);
    try std.testing.expectEqual(@as(i32, @intCast(span.len)), consumer.pos());
    try std.testing.expectEqualSlices(i32, span, consumer.resident.items);
    const reused = try consumer.sync(ids);
    try std.testing.expectEqual(@as(i32, @intCast(ids.len - 1)), reused);
    try std.testing.expectEqual(@as(u32, 0), consumer.trim_refusals);
    try std.testing.expectEqual(expect_next, consumer.argmax());

    // Garbage is refused and leaves the session empty, never half-restored.
    var victim = try engine.createSession(2048);
    defer victim.free();
    const junk = [_]u8{ 0xde, 0xad, 0xbe, 0xef, 0, 0, 0, 0 };
    try std.testing.expectError(Error.StateImportFailed, victim.importState(&junk, span));
    try std.testing.expectEqual(@as(i32, 0), victim.pos());
    try std.testing.expectEqual(@as(usize, 0), victim.resident.items.len);
}
