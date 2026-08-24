// Implementation of the mlx-serve llama.cpp shim. See llama_shim.h.
//
// Compiled by build.zig against the staged headers in lib/llama/include and
// linked against lib/llama/lib/libllama.dylib (scripts/fetch-llama.sh stages
// both). This is the only place that touches llama.cpp's real structs.
#include "llama_shim.h"

#include "llama.h"
#include "ggml-backend.h"
// The only thing this shim needs from ggml-rpc.h is the server cap; every RPC
// entry point is reached through the registry proc table at runtime, never
// linked. Including the header dragged in a full ggml.h by a second path — on a
// host with a stale brew ggml on the include search path (webp pulls in
// /opt/homebrew/include) that is a hard redefinition against the vendored
// ggml.h, and the macOS asset ships no ggml-rpc.h at all. Define the constant
// (ggml's own value) and drop the dependency on every platform.
#define GGML_RPC_MAX_SERVERS 16

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct mlx_llama_engine {
    struct llama_model *model;
    const struct llama_vocab *vocab;
};

struct mlx_llama_session {
    struct llama_context *ctx;
    struct mlx_llama_engine *engine;
    int32_t pos; // tokens decoded into the KV cache so far
};

// Prefill chunk size. The default logical batch (n_batch) is 2048, so 512-token
// chunks are always accepted by llama_decode regardless of context params.
#define MLX_LLAMA_PREFILL_CHUNK 512

static pthread_once_t g_backend_once = PTHREAD_ONCE_INIT;

static void backend_init_once(void) {
    llama_backend_init();
    // ggml dlopens its compute backends (CUDA, the per-uarch CPU variants) by
    // FILENAME at runtime; `llama_backend_init` does not do it for us. The
    // registry's own auto-load searches the directory of the RUNNING
    // executable, which is right for the installed server (the whole DLL set
    // ships beside mlx-serve.exe) and wrong for anything else — a `zig build
    // test` binary lives under .zig-cache and loaded ZERO backends, so every
    // model-gated llama test died `no backends are loaded` no matter what was
    // on PATH. Calling it explicitly is the documented contract, and
    // MLX_LLAMA_BACKEND_DIR points it at the staged tree (lib/llama/bin) for
    // any binary that is not the installed one. Loading is additive and
    // idempotent — a backend already registered is not registered twice — so
    // doing both is safe.
#ifndef __APPLE__
    // Apple only: the XCFramework merges llama + ggml + ggml-metal into ONE
    // dylib with its backends compiled IN, so there is nothing to dlopen and
    // nothing to link for it. Everywhere else ggml ships the backends as
    // separate shared objects and this call is what registers them.
    const char *dir = getenv("MLX_LLAMA_BACKEND_DIR");
    if (dir && dir[0] != '\0') {
        ggml_backend_load_all_from_path(dir);
    } else {
        ggml_backend_load_all();
    }
#endif
}

static void copy_err(char *err, size_t errlen, const char *msg) {
    if (err && errlen > 0) {
        strncpy(err, msg, errlen - 1);
        err[errlen - 1] = '\0';
    }
}

mlx_llama_engine *mlx_llama_open(const char *gguf_path, int32_t n_gpu_layers, char *err, size_t errlen) {
    return mlx_llama_open_ex(gguf_path, n_gpu_layers, NULL, 0, NULL, err, errlen);
}

// ggml RPC entry points are reached through the registry's proc table, NOT
// linked: ggml-rpc is a dlopen'd backend everywhere but the Apple
// XCFramework, and linking its symbols would make the exe refuse to start on
// a host whose build has no RPC (the stock rpc-server tool does the same).
typedef ggml_backend_reg_t (*rpc_add_server_fn)(const char *endpoint);
typedef void (*rpc_get_device_memory_fn)(const char *endpoint, uint32_t device, size_t *free, size_t *total);
typedef void (*rpc_start_server_fn)(const char *endpoint, const char *cache_dir,
                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t *devices);

static void ensure_backend_init(void) { pthread_once(&g_backend_once, backend_init_once); }

static void *rpc_proc(const char *name) {
    ensure_backend_init();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("RPC");
    if (!reg) return NULL;
    return ggml_backend_reg_get_proc_address(reg, name);
}

// One registry per endpoint, created on first contact and reused by
// mlx_llama_open_ex: adding the same server twice would register its device
// twice, and llama.cpp would then see two "GPUs" that are one card.
static struct { char endpoint[256]; ggml_backend_reg_t reg; } g_rpc_regs[GGML_RPC_MAX_SERVERS];
static int g_rpc_regs_n = 0;
static pthread_mutex_t g_rpc_regs_mu = PTHREAD_MUTEX_INITIALIZER;

// NULL = unreachable, no RPC backend, or the worker exports no device. The
// proc table at b10472 exposes add_server/start_server, NOT get_device_memory,
// which is why reachability rides add_server (the path common.cpp uses too).
static ggml_backend_reg_t rpc_reg_for(const char *endpoint) {
    pthread_mutex_lock(&g_rpc_regs_mu);
    for (int i = 0; i < g_rpc_regs_n; i++) {
        if (strcmp(g_rpc_regs[i].endpoint, endpoint) == 0) {
            ggml_backend_reg_t r = g_rpc_regs[i].reg;
            pthread_mutex_unlock(&g_rpc_regs_mu);
            return r;
        }
    }
    ggml_backend_reg_t reg = NULL;
    rpc_add_server_fn add = (rpc_add_server_fn)rpc_proc("ggml_backend_rpc_add_server");
    if (add) reg = add(endpoint);
    if (reg && ggml_backend_reg_dev_count(reg) == 0) reg = NULL;
    if (reg && g_rpc_regs_n < GGML_RPC_MAX_SERVERS) {
        strncpy(g_rpc_regs[g_rpc_regs_n].endpoint, endpoint, sizeof g_rpc_regs[0].endpoint - 1);
        g_rpc_regs[g_rpc_regs_n].endpoint[sizeof g_rpc_regs[0].endpoint - 1] = '\0';
        g_rpc_regs[g_rpc_regs_n].reg = reg;
        g_rpc_regs_n++;
    }
    pthread_mutex_unlock(&g_rpc_regs_mu);
    return reg;
}

bool mlx_llama_rpc_device_memory(const char *endpoint, uint64_t *free_out, uint64_t *total_out) {
    ggml_backend_reg_t reg = rpc_reg_for(endpoint);
    if (!reg) return false;
    size_t f = 0, t = 0;
    ggml_backend_dev_memory(ggml_backend_reg_dev_get(reg, 0), &f, &t);
    if (free_out) *free_out = (uint64_t)f;
    if (total_out) *total_out = (uint64_t)t;
    // A worker that dies between add_server and now answers 0/0.
    return t > 0;
}

// First GPU-class device (CUDA0 / Metal), else the CPU device.
static ggml_backend_dev_t best_local_device(void) {
    ensure_backend_init();
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) return d;
    }
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) return d;
    }
    return NULL;
}

mlx_llama_engine *mlx_llama_open_ex(const char *gguf_path, int32_t n_gpu_layers,
                                    const char **rpc_endpoints, int32_t n_rpc,
                                    const float *tensor_split,
                                    char *err, size_t errlen) {
    ensure_backend_init();

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;

    // Explicit device list: every local GPU (registry order) followed by the
    // RPC devices in the order the user named them; `tensor_split` is indexed
    // the same way. Without RPC devices mp.devices stays NULL and llama.cpp
    // picks exactly what it always did.
    ggml_backend_dev_t devs[GGML_RPC_MAX_SERVERS + 17];
    size_t n_devs = 0;
    if (n_rpc > 0) {
        if (!rpc_proc("ggml_backend_rpc_add_server")) {
            copy_err(err, errlen, "ggml RPC backend is not loaded (ggml-rpc missing beside the executable)");
            return NULL;
        }
        for (size_t i = 0; i < ggml_backend_dev_count() && n_devs < 16; i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) devs[n_devs++] = d;
        }
        for (int32_t i = 0; i < n_rpc && i < GGML_RPC_MAX_SERVERS; i++) {
            // Reachability FIRST: rpc-offload-plan.md says a dead peer is a
            // named load failure, never a model half-loaded on whatever is left.
            uint64_t f = 0, t = 0;
            char msg[320];
            if (!mlx_llama_rpc_device_memory(rpc_endpoints[i], &f, &t)) {
                snprintf(msg, sizeof msg, "RPC worker unreachable: %s", rpc_endpoints[i]);
                copy_err(err, errlen, msg);
                return NULL;
            }
            ggml_backend_reg_t reg = rpc_reg_for(rpc_endpoints[i]);
            if (!reg) {
                snprintf(msg, sizeof msg, "RPC add_server failed: %s", rpc_endpoints[i]);
                copy_err(err, errlen, msg);
                return NULL;
            }
            for (size_t j = 0; j < ggml_backend_reg_dev_count(reg) && n_devs < GGML_RPC_MAX_SERVERS + 16; j++)
                devs[n_devs++] = ggml_backend_reg_dev_get(reg, j);
        }
        devs[n_devs] = NULL;
        mp.devices = devs;
        mp.tensor_split = tensor_split;
        mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
    }

    struct llama_model *model = llama_model_load_from_file(gguf_path, mp);
    if (!model) {
        copy_err(err, errlen, "llama_model_load_from_file failed");
        return NULL;
    }

    mlx_llama_engine *e = (mlx_llama_engine *)calloc(1, sizeof(*e));
    if (!e) {
        llama_model_free(model);
        copy_err(err, errlen, "out of memory allocating engine");
        return NULL;
    }
    e->model = model;
    e->vocab = llama_model_get_vocab(model);
    return e;
}

void mlx_llama_close(mlx_llama_engine *e) {
    if (!e) return;
    if (e->model) llama_model_free(e->model);
    free(e);
}

int32_t mlx_llama_eos_token(mlx_llama_engine *e) {
    return (int32_t)llama_vocab_eos(e->vocab);
}

bool mlx_llama_is_eog(mlx_llama_engine *e, int32_t token) {
    return llama_vocab_is_eog(e->vocab, (llama_token)token);
}

int32_t mlx_llama_n_vocab(mlx_llama_engine *e) {
    return llama_vocab_n_tokens(e->vocab);
}

int32_t mlx_llama_tokenize(mlx_llama_engine *e, const char *text, int32_t text_len,
                           bool add_special, bool parse_special,
                           int32_t *out, int32_t out_cap) {
    return llama_tokenize(e->vocab, text, text_len, (llama_token *)out, out_cap,
                          add_special, parse_special);
}

int32_t mlx_llama_token_to_piece(mlx_llama_engine *e, int32_t token, char *buf, int32_t buf_cap) {
    // lstrip=0, special=false: render the literal piece bytes.
    return llama_token_to_piece(e->vocab, (llama_token)token, buf, buf_cap, 0, false);
}

const char *mlx_llama_chat_template(mlx_llama_engine *e) {
    return llama_model_chat_template(e->model, NULL);
}

int32_t mlx_llama_apply_chat_template(mlx_llama_engine *e,
                                      const char **roles, const char **contents, int32_t n_msgs,
                                      bool add_assistant, char *buf, int32_t buf_cap) {
    const char *tmpl = llama_model_chat_template(e->model, NULL);
    size_t n = (size_t)(n_msgs > 0 ? n_msgs : 0);
    struct llama_chat_message *msgs =
        (struct llama_chat_message *)calloc(n ? n : 1, sizeof(struct llama_chat_message));
    if (!msgs) return -1;
    for (size_t i = 0; i < n; i++) {
        msgs[i].role = roles[i];
        msgs[i].content = contents[i];
    }
    int32_t r = llama_chat_apply_template(tmpl, msgs, n, add_assistant, buf, buf_cap);
    free(msgs);
    return r;
}

mlx_llama_session *mlx_llama_session_create(mlx_llama_engine *e, int32_t n_ctx, char *err, size_t errlen) {
    return mlx_llama_session_create_kv_quant(e, n_ctx, 0, 0, err, errlen);
}

mlx_llama_session *mlx_llama_session_create_kv_quant(mlx_llama_engine *e,
                                                    int32_t n_ctx,
                                                    int32_t type_k,
                                                    int32_t type_v,
                                                    char *err, size_t errlen) {
    return mlx_llama_session_create_ex(e, n_ctx, type_k, type_v, true, err, errlen);
}

mlx_llama_session *mlx_llama_session_create_ex(mlx_llama_engine *e,
                                               int32_t n_ctx,
                                               int32_t type_k,
                                               int32_t type_v,
                                               bool swa_full,
                                               char *err, size_t errlen) {
    struct llama_context_params cp = llama_context_default_params();
    if (n_ctx > 0) cp.n_ctx = (uint32_t)n_ctx;
    // swa_full=true is the default for every PERSISTENT session (below); a
    // remote-prefill WORKER session passes false so the exported state carries
    // only window-worth of cells per sliding layer (gemma 4: 43/49 layers at
    // window 1024 -- the blob shrinks ~10x at long contexts). A windowed blob
    // restores into a full-size consumer cache: state_read is a plain
    // find_slot of cell_count cells, the mask is by position.
    //
    // Force the full-size SWA cache. With swa_full=false (the libllama default
    // since b73xx) sliding-window-attention layers only expose `window`-many KV
    // slots per sequence; after `llama_memory_seq_rm` trims a divergent tail in
    // a persistent prompt-prefix-reuse session, the next `llama_decode` can
    // fail to find a contiguous block of free slots and abort the prefill with
    //   init_batch: failed to prepare attention ubatches
    //   decode: failed to find a memory slot for batch of size 512
    // mlx-serve owns its own ctx-size cap up the stack, so the extra KV that
    // swa_full=true costs (window→full per SWA layer) is exactly what we
    // already accounted for. Matches `llama-server --swa-full` and addresses
    // llama.cpp issues #19794 / #21831 / #17196 for hybrid/SWA GGUFs.
    cp.swa_full = swa_full;
    // Flash attention. The llama.cpp default is AUTO, which on Metal already
    // enables FA when beneficial AND safely falls back when a model's head_dim
    // isn't supported by the FA kernel — measured equivalent to forcing ENABLED
    // (e.g. Gemma E4B long-context decode: auto≈on≈86 tok/s, off≈75), so we keep
    // AUTO rather than forcing ENABLED (which would drop the fallback safety).
    // Quantized K/V is the one case that *requires* FA — the plain SDPA path is
    // F16/F32 only — so force it on whenever a non-default KV type is requested.
    if (type_k != 0 || type_v != 0) {
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    if (type_k != 0) cp.type_k = (enum ggml_type)type_k;
    if (type_v != 0) cp.type_v = (enum ggml_type)type_v;

    struct llama_context *ctx = llama_init_from_model(e->model, cp);
    if (!ctx) {
        copy_err(err, errlen, "llama_init_from_model failed");
        return NULL;
    }
    mlx_llama_session *s = (mlx_llama_session *)calloc(1, sizeof(*s));
    if (!s) {
        llama_free(ctx);
        copy_err(err, errlen, "out of memory allocating session");
        return NULL;
    }
    s->ctx = ctx;
    s->engine = e;
    s->pos = 0;
    return s;
}

void mlx_llama_session_free(mlx_llama_session *s) {
    if (!s) return;
    if (s->ctx) llama_free(s->ctx);
    free(s);
}

// Decode a contiguous run of tokens. llama_batch_get_one tracks positions
// automatically from the KV state and (logits == NULL) outputs logits for the
// last token only. Advances s->pos on success.
static int32_t decode_run(mlx_llama_session *s, const int32_t *tokens, int32_t n) {
    struct llama_batch batch = llama_batch_get_one((llama_token *)tokens, n);
    int32_t rc = llama_decode(s->ctx, batch);
    if (rc != 0) return rc;
    s->pos += n;
    return 0;
}

int32_t mlx_llama_session_sync(mlx_llama_session *s, const int32_t *tokens, int32_t n_tokens,
                               char *err, size_t errlen) {
    int32_t off = 0;
    while (off < n_tokens) {
        int32_t n = n_tokens - off;
        if (n > MLX_LLAMA_PREFILL_CHUNK) n = MLX_LLAMA_PREFILL_CHUNK;
        if (decode_run(s, tokens + off, n) != 0) {
            copy_err(err, errlen, "llama_decode failed during prefill");
            return -1;
        }
        off += n;
    }
    return 0;
}

// ── Embeddings ─────────────────────────────────────────────────────────────

int32_t mlx_llama_n_embd(mlx_llama_engine *e) {
    return llama_model_n_embd(e->model);
}

int32_t mlx_llama_n_ctx_train(mlx_llama_engine *e) {
    return llama_model_n_ctx_train(e->model);
}

// Build the context params an embedding session needs. Split out so
// mlx_llama_has_pooling can ask the same question the session will answer.
static struct llama_context_params embed_ctx_params(int32_t n_ctx) {
    struct llama_context_params cp = llama_context_default_params();
    if (n_ctx > 0) cp.n_ctx = (uint32_t)n_ctx;
    cp.embeddings = true;
    // UNSPECIFIED means "use whatever the checkpoint declares". Naming a
    // pooling type here would override the model's own -- a CLS model asked to
    // mean-pool returns a plausible vector that is simply not this model's
    // embedding, and nothing about it looks wrong.
    cp.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;
    // No causal mask: an embedding encoder is bidirectional. Generative
    // checkpoints used as embedders keep causal attention, which llama.cpp
    // decides from the model's own architecture, so this only relaxes it where
    // the model says it is an encoder.
    cp.n_ubatch = cp.n_batch;
    return cp;
}

bool mlx_llama_is_encoder_only(mlx_llama_engine *e) {
    // NOT llama_model_has_encoder(): llama.cpp reserves that for true
    // encoder-decoder architectures (T5), and reports every BERT-family
    // embedding model as a plain decoder -- so it answers false for exactly
    // the checkpoints this question is about (measured on nomic-embed).
    //
    // The authoritative signal is the POOLING type the checkpoint declares:
    // an embedding model pools its token states into one vector per sequence,
    // a generative one does not. That is only readable from a context, so
    // probe with a deliberately tiny one -- n_ctx is what sizes the KV
    // allocation, and 32 tokens of it costs nothing even on a 27B.
    struct llama_context_params cp = embed_ctx_params(32);
    struct llama_context *ctx = llama_init_from_model(e->model, cp);
    if (!ctx) return false;
    const enum llama_pooling_type pt = llama_pooling_type(ctx);
    llama_free(ctx);
    return pt != LLAMA_POOLING_TYPE_NONE;
}

mlx_llama_session *mlx_llama_embed_session_create(mlx_llama_engine *e, int32_t n_ctx,
                                                 char *err, size_t errlen) {
    struct llama_context *ctx = llama_init_from_model(e->model, embed_ctx_params(n_ctx));
    if (!ctx) {
        copy_err(err, errlen, "llama_init_from_model failed (embeddings)");
        return NULL;
    }
    mlx_llama_session *s = (mlx_llama_session *)calloc(1, sizeof(*s));
    if (!s) {
        llama_free(ctx);
        copy_err(err, errlen, "out of memory allocating embed session");
        return NULL;
    }
    s->ctx = ctx;
    s->engine = e;
    s->pos = 0;
    return s;
}

int32_t mlx_llama_session_embed(mlx_llama_session *s,
                                const int32_t *tokens, int32_t n_tokens,
                                float *out, int32_t out_cap,
                                char *err, size_t errlen) {
    const int32_t n_embd = llama_model_n_embd(s->engine->model);
    if (n_tokens <= 0) {
        copy_err(err, errlen, "empty token sequence");
        return -1;
    }
    if (out_cap < n_embd) {
        copy_err(err, errlen, "output buffer smaller than n_embd");
        return -1;
    }

    // An embedding depends on its input alone. Without this clear the previous
    // sequence is still resident and positions continue from it, so the second
    // call in a batch embeds "previous text + this text".
    llama_memory_clear(llama_get_memory(s->ctx), true);
    s->pos = 0;

    // An explicit batch rather than llama_batch_get_one: pooling reduces over
    // the tokens whose logits flag is set, and get_one sets only the LAST one.
    // With that, a mean-pooled model returns the mean of a single token.
    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int32_t i = 0; i < n_tokens; i++) {
        batch.token[i] = (llama_token)tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = n_tokens;

    const int32_t rc = llama_decode(s->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        copy_err(err, errlen, "llama_decode failed during embedding");
        return -1;
    }
    s->pos = n_tokens;

    const float *src = llama_get_embeddings_seq(s->ctx, 0);
    if (!src) {
        // No pooling: the model produces one vector PER TOKEN and there is no
        // single sequence embedding. Take the last token's, which is what a
        // generative checkpoint used as an embedder means.
        src = llama_get_embeddings_ith(s->ctx, n_tokens - 1);
    }
    if (!src) {
        copy_err(err, errlen, "no embeddings produced (is this an embedding model?)");
        return -1;
    }
    memcpy(out, src, (size_t)n_embd * sizeof(float));
    return n_embd;
}

int32_t mlx_llama_session_trim(mlx_llama_session *s, int32_t n_keep) {
    if (n_keep < 0) n_keep = 0;
    if (n_keep >= s->pos) return 0; // nothing resident beyond n_keep
    // Single-sequence (seq 0) usage: remove positions [n_keep, inf).
    //
    // This return value is LOAD-BEARING and was previously discarded on the
    // belief that removing a whole tail never fails. That holds for a KV cache
    // and is false for a RECURRENT one: a hybrid checkpoint (GatedDeltaNet,
    // Mamba, RWKV) keeps a fixed-size rolling state per layer with no history
    // to roll back to, so `llama_memory_recurrent::seq_rm` refuses any p0 > 0
    // and leaves the state exactly as it was. Ignoring that produced a session
    // whose recurrent layers still held the OLD tail while `pos` claimed they
    // did not, and the request answered from the wrong position -- live on
    // Qwen3.8-27B (2026-08-20) an identical repeat request echoed prompt tokens
    // back, and the one after it returned an empty completion.
    //
    // A refusal is not an error: the only correct recovery is to drop
    // everything and let the caller cold-prefill, which is what `1` reports.
    if (!llama_memory_seq_rm(llama_get_memory(s->ctx), 0, n_keep, -1)) {
        llama_memory_clear(llama_get_memory(s->ctx), true);
        s->pos = 0;
        return 1;
    }
    s->pos = n_keep;
    return 0;
}

void mlx_llama_session_reset(mlx_llama_session *s) {
    llama_memory_clear(llama_get_memory(s->ctx), true);
    s->pos = 0;
}

int32_t mlx_llama_session_eval(mlx_llama_session *s, int32_t token, char *err, size_t errlen) {
    int32_t t = token;
    if (decode_run(s, &t, 1) != 0) {
        copy_err(err, errlen, "llama_decode failed");
        return -1;
    }
    return 0;
}

int32_t mlx_llama_session_argmax(mlx_llama_session *s) {
    const float *logits = llama_get_logits_ith(s->ctx, -1);
    if (!logits) return -1;
    int32_t n = llama_vocab_n_tokens(s->engine->vocab);
    int32_t best = 0;
    float best_v = logits[0];
    for (int32_t i = 1; i < n; i++) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return best;
}

int32_t mlx_llama_session_sample(mlx_llama_session *s, float temperature, int32_t top_k,
                                 float top_p, float min_p, uint64_t *rng) {
    if (temperature <= 0.0f) return mlx_llama_session_argmax(s);

    struct llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    sp.no_perf = true;
    struct llama_sampler *chain = llama_sampler_chain_init(sp);
    if (top_k > 0) llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
    if (top_p > 0.0f && top_p < 1.0f) llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
    if (min_p > 0.0f) llama_sampler_chain_add(chain, llama_sampler_init_min_p(min_p, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));

    uint64_t state = (rng && *rng) ? *rng : 0x106689D45497FDB5ULL;
    uint32_t seed = (uint32_t)(state ^ ((uint64_t)s->pos * 0x9E3779B97F4A7C15ULL));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));

    int32_t tok = (int32_t)llama_sampler_sample(chain, s->ctx, -1);
    llama_sampler_free(chain);

    if (rng) {
        // xorshift64 so the next draw uses a fresh seed (reproducible chain).
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *rng = state;
    }
    return tok;
}

int32_t mlx_llama_session_pos(mlx_llama_session *s) {
    return s->pos;
}

// ── Sequence-state interchange (remote prefill) ────────────────────────────

size_t mlx_llama_session_state_size(mlx_llama_session *s) {
    if (s->pos == 0) return 0;
    return llama_state_seq_get_size(s->ctx, 0);
}

size_t mlx_llama_session_state_get(mlx_llama_session *s, uint8_t *dst, size_t cap) {
    if (s->pos == 0) return 0;
    size_t need = llama_state_seq_get_size(s->ctx, 0);
    if (need == 0 || need > cap) return 0;
    return llama_state_seq_get_data(s->ctx, dst, cap, 0);
}

int32_t mlx_llama_session_state_set(mlx_llama_session *s, const uint8_t *src, size_t n,
                                    int32_t n_tokens, char *err, size_t errlen) {
    // Clear first: set_data appends into whatever the sequence already holds,
    // and the caller's contract is "this blob IS the sequence".
    mlx_llama_session_reset(s);
    if (n_tokens <= 0 || n == 0) {
        copy_err(err, errlen, "state_set: empty blob or non-positive token count");
        return -1;
    }
    size_t got = llama_state_seq_set_data(s->ctx, src, n, 0);
    if (got == 0) {
        // llama.cpp reports failure as 0 and may have partially populated the
        // sequence before giving up -- never leave that behind.
        mlx_llama_session_reset(s);
        copy_err(err, errlen, "llama_state_seq_set_data refused the blob (build/model mismatch?)");
        return -1;
    }
    s->pos = n_tokens;
    return 0;
}

// -- Backend registry queries --------------------------------------------------
bool mlx_llama_backend_present(const char *name) {
    ensure_backend_init();
    return ggml_backend_reg_by_name(name) != NULL;
}

int32_t mlx_llama_backend_names(char *buf, size_t cap) {
    ensure_backend_init();
    if (!buf || cap == 0) return 0;
    size_t used = 0;
    buf[0] = '\0';
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        const char *n = ggml_backend_reg_name(ggml_backend_reg_get(i));
        size_t len = strlen(n);
        if (used + len + 2 > cap) break;
        if (used) buf[used++] = ',';
        memcpy(buf + used, n, len);
        used += len;
        buf[used] = '\0';
    }
    return (int32_t)used;
}

// -- RPC worker --------------------------------------------------------------
int32_t mlx_llama_rpc_local_device(char *name_buf, size_t cap, uint64_t *free_out, uint64_t *total_out) {
    ggml_backend_dev_t best = best_local_device();
    if (!best) return -1;
    size_t f = 0, t = 0;
    ggml_backend_dev_memory(best, &f, &t);
    if (free_out) *free_out = f;
    if (total_out) *total_out = t;
    if (name_buf && cap) { strncpy(name_buf, ggml_backend_dev_name(best), cap - 1); name_buf[cap - 1] = '\0'; }
    return (int32_t)(ggml_backend_dev_type(best) == GGML_BACKEND_DEVICE_TYPE_GPU);
}

bool mlx_llama_rpc_serve(const char *endpoint, const char *cache_dir, int32_t n_threads, char *err, size_t errlen) {
    rpc_start_server_fn start = (rpc_start_server_fn)rpc_proc("ggml_backend_rpc_start_server");
    if (!start) {
        copy_err(err, errlen, "ggml RPC backend is not loaded (ggml-rpc missing beside the executable)");
        return false;
    }
    ggml_backend_dev_t best = best_local_device();
    if (!best) { copy_err(err, errlen, "no ggml device to serve"); return false; }
    ggml_backend_dev_t devs[1] = { best };
    // Blocks for the life of the process (accept loop); the caller runs it on
    // its own thread.
    start(endpoint, (cache_dir && cache_dir[0]) ? cache_dir : NULL,
          (size_t)(n_threads > 0 ? n_threads : 1), 1, devs);
    copy_err(err, errlen, "rpc server loop returned");
    return false;
}

int32_t mlx_llama_local_gpu_count(void) {
    ensure_backend_init();
    int32_t n = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++)
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) n++;
    return n;
}
