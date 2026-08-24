// Clean C API over llama.cpp's libllama, in the spirit of ds4.h.
//
// libllama exposes large, ABI-fragile structs (llama_model_params,
// llama_context_params) and a verbose decode/sample protocol. Rather than mirror
// those structs in Zig (where a field drift silently corrupts memory), this shim
// compiles against the real llama.h (so the ABI is always correct) and exports a
// small, stable surface that src/llama_ffi.zig mirrors 1:1 — exactly how
// src/ds4_ffi.zig mirrors ds4.h.
//
// Lifetimes & threading: the engine wraps a loaded model + vocab; a session
// wraps a context + KV state. The llama backend is initialized once per process
// (pthread_once). Sessions are single-threaded (the scheduler drives one slot at
// a time), matching the ds4 contract.
#ifndef MLX_LLAMA_SHIM_H
#define MLX_LLAMA_SHIM_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mlx_llama_engine mlx_llama_engine;
typedef struct mlx_llama_session mlx_llama_session;

// Load a GGUF model. n_gpu_layers: number of layers to offload to Metal (pass a
// large value, e.g. 999, for "all"). Returns NULL on failure with `err` filled.
mlx_llama_engine *mlx_llama_open(const char *gguf_path, int32_t n_gpu_layers, char *err, size_t errlen);
void mlx_llama_close(mlx_llama_engine *e);

int32_t mlx_llama_eos_token(mlx_llama_engine *e);
bool    mlx_llama_is_eog(mlx_llama_engine *e, int32_t token); // EOS or any end-of-generation token
int32_t mlx_llama_n_vocab(mlx_llama_engine *e);

// Tokenize `text` (length `text_len`). Writes up to `out_cap` ids into `out`.
// Returns the number of tokens written, or a negative value (= -required) when
// `out_cap` is too small (caller re-allocates and retries). Mirrors llama_tokenize.
int32_t mlx_llama_tokenize(mlx_llama_engine *e, const char *text, int32_t text_len,
                           bool add_special, bool parse_special,
                           int32_t *out, int32_t out_cap);

// One token -> bytes (NOT NUL-terminated). Returns #bytes written, or a negative
// value (= -required) when `buf_cap` is too small. Mirrors llama_token_to_piece.
int32_t mlx_llama_token_to_piece(mlx_llama_engine *e, int32_t token, char *buf, int32_t buf_cap);

// Raw GGUF chat-template string (jinja source), or NULL if the model has none.
// Borrowed pointer owned by the model; valid for the engine's lifetime. Callers
// that want robust rendering should feed this to mlx-serve's own jinja engine.
const char *mlx_llama_chat_template(mlx_llama_engine *e);

// Apply the model's built-in chat template via llama_chat_apply_template.
// NOTE: this is NOT a full jinja parser — it only recognizes a fixed set of
// known template formats. Prefer rendering mlx_llama_chat_template() through
// mlx-serve's jinja engine; this is a fallback. Returns the formatted byte count
// (may exceed buf_cap -> grow and retry), or a negative value on error.
int32_t mlx_llama_apply_chat_template(mlx_llama_engine *e,
                                      const char **roles, const char **contents, int32_t n_msgs,
                                      bool add_assistant, char *buf, int32_t buf_cap);

// Create a context/session sized to n_ctx (0 = model default). NULL on failure.
mlx_llama_session *mlx_llama_session_create(mlx_llama_engine *e, int32_t n_ctx, char *err, size_t errlen);

// Like mlx_llama_session_create but lets the caller pick ggml types for the K
// and V KV-cache halves. Values are ggml_type enum integers (F16=1, Q8_0=8,
// Q4_0=2, Q4_1=3, etc.); pass 0 to keep the libllama default (typically F16).
// Use F16/F16 for parity with mlx_llama_session_create; Q8_0/Q8_0 cuts KV by
// ~2× with near-lossless quality on most archs; Q4_0/Q4_0 cuts KV by ~4× at
// some accuracy cost.
mlx_llama_session *mlx_llama_session_create_kv_quant(mlx_llama_engine *e,
                                                     int32_t n_ctx,
                                                     int32_t type_k,
                                                     int32_t type_v,
                                                     char *err, size_t errlen);
// Like mlx_llama_session_create_kv_quant plus the SWA cache mode: swa_full
// true = every sliding-window layer keeps a full-context cache (what every
// persistent session needs -- a windowed cache aborts the next decode after
// a partial seq_rm); false = window-worth of cells only, for a session whose
// exported state must be SMALL (remote-prefill worker: reset per request,
// never trimmed).
mlx_llama_session *mlx_llama_session_create_ex(mlx_llama_engine *e,
                                               int32_t n_ctx,
                                               int32_t type_k,
                                               int32_t type_v,
                                               bool swa_full,
                                               char *err, size_t errlen);
void mlx_llama_session_free(mlx_llama_session *s);

// ── Embeddings ─────────────────────────────────────────────────────────────
//
// An embedding context is a SEPARATE context, not a mode of a generation one:
// `llama_context_params.embeddings` and `pooling_type` are fixed at creation,
// and a pooled context produces no per-token logits to sample from. So an
// embedding model gets its own session and never shares one with chat.

// Width of one embedding vector for this model.
int32_t mlx_llama_n_embd(mlx_llama_engine *e);

// The context length this checkpoint was TRAINED at. The GGUF stub config is
// built before the engine opens and has to guess (8192), which is wrong in
// both directions: it over-promises on a 2048-window embedder and
// under-reports a 262144 one. Once the engine is open the model itself knows.
int32_t mlx_llama_n_ctx_train(mlx_llama_engine *e);

// Is this checkpoint an EMBEDDING model rather than a generative one?
// Answered from the POOLING type it declares -- an embedding model reduces its
// token states to one vector per sequence, a generative model does not.
// Probed with a 32-token context, so the KV allocation it costs is noise.
//
// The answer drives what the server ADVERTISES: an encoder-only checkpoint
// must not claim "chat", and must claim "embeddings".
bool mlx_llama_is_encoder_only(mlx_llama_engine *e);

// Create a session for embeddings. Same lifetime and free function as a
// generation session (mlx_llama_session_free). NULL on failure, with the
// reason in `err`.
mlx_llama_session *mlx_llama_embed_session_create(mlx_llama_engine *e, int32_t n_ctx,
                                                 char *err, size_t errlen);

// Embed one token sequence into `out` (which must hold at least
// mlx_llama_n_embd floats). Returns the number of floats written, or -1 with
// the reason in `err`.
//
// The session's memory is CLEARED first: an embedding is a property of the
// input alone, so carrying KV across calls would make the answer depend on
// what was embedded before it.
int32_t mlx_llama_session_embed(mlx_llama_session *s,
                                const int32_t *tokens, int32_t n_tokens,
                                float *out, int32_t out_cap,
                                char *err, size_t errlen);

// Decode a contiguous run of tokens, appending to the KV cache from the current
// position. Chunked internally. 0 ok, -1 on failure. Caller drives prompt-prefix
// reuse by trimming first (see mlx_llama_session_trim) and passing only the
// divergent suffix here; positions continue automatically from the KV state.
int32_t mlx_llama_session_sync(mlx_llama_session *s, const int32_t *tokens, int32_t n_tokens,
                               char *err, size_t errlen);

// Drop all KV entries at positions >= n_keep (i.e. keep the first n_keep tokens),
// leaving the cache at position n_keep so the next sync continues from there.
// No-op when n_keep >= current position. Used for prompt-prefix reuse: keep the
// common prefix, discard the divergent tail.
//
// Returns 0 when the session now holds exactly the first n_keep tokens.
// Returns 1 when a partial trim was REFUSED and the whole session was cleared
// instead: a recurrent/hybrid memory (GatedDeltaNet, Mamba, RWKV) has no
// history to rewind to. The caller must then treat the session as EMPTY and
// re-feed the whole prompt -- reusing a prefix it no longer holds serves the
// request from the wrong position.
int32_t mlx_llama_session_trim(mlx_llama_session *s, int32_t n_keep);

// Clear the entire KV cache (position → 0). For a prompt that shares nothing with
// the resident tokens or exceeds the context — caller rebuilds from scratch.
void mlx_llama_session_reset(mlx_llama_session *s);

// Advance the KV cache by one token. 0 ok, -1 on failure.
int32_t mlx_llama_session_eval(mlx_llama_session *s, int32_t token, char *err, size_t errlen);

// Sample the next token from the current (last-decoded) logits.
// temperature <= 0 => greedy argmax. `rng` is advanced so repeated draws differ
// while staying reproducible from the seed (matches the ds4 sample contract).
int32_t mlx_llama_session_sample(mlx_llama_session *s, float temperature, int32_t top_k,
                                 float top_p, float min_p, uint64_t *rng);
// Greedy argmax of the current logits.
int32_t mlx_llama_session_argmax(mlx_llama_session *s);
// Number of tokens currently in the KV cache.
int32_t mlx_llama_session_pos(mlx_llama_session *s);

// ── Sequence-state interchange (remote prefill) ────────────────────────────
//
// The session holds ONE sequence (seq_id 0). Its KV state serializes through
// llama.cpp's own llama_state_seq_get_data / set_data, which is the ONLY thing
// that makes a CUDA-prefilled cache loadable by a Metal decoder: upstream owns
// the per-backend layout and the blob is backend-neutral. The blob is coupled
// to the llama.cpp BUILD and the GGUF file -- both sides must run the same
// pin and the same weights, or a restore silently produces garbage.

// Exact size in bytes of the serialized state. 0 when the session is empty.
size_t mlx_llama_session_state_size(mlx_llama_session *s);

// Serialize the sequence state into `dst` (`cap` bytes). Returns the number of
// bytes written, 0 on failure (cap too small or libllama refused).
size_t mlx_llama_session_state_get(mlx_llama_session *s, uint8_t *dst, size_t cap);

// Restore a blob produced by mlx_llama_session_state_get on the SAME llama.cpp
// build + GGUF. The session is CLEARED first. On success the shim's own
// position counter is set to `n_tokens` -- the token count the blob was
// prefilled with, which llama.cpp does not carry -- because a restore that
// left `pos` behind would make the next decode write at the wrong position:
// coherent-looking output from a corrupt cache, no error anywhere. The blob
// carries NO logits: decode at least one token before sampling. 0 ok, -1 on
// failure (session left cleared; `err` names the reason).
int32_t mlx_llama_session_state_set(mlx_llama_session *s, const uint8_t *src, size_t n,
                                    int32_t n_tokens, char *err, size_t errlen);

#ifdef __cplusplus
}
#endif

#endif // MLX_LLAMA_SHIM_H
