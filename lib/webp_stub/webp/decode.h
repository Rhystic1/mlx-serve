// Minimal libwebp `decode.h` surface for builds that do not ship libwebp.
//
// Homebrew's libwebp is a macOS build dependency; there is no equivalent on the
// Windows/Linux GGUF-only path, and vendoring libwebp (~100 translation units)
// is not worth it for one decode helper. The matching lib/webp_stub/webp_stub.c
// implements these two symbols so `src/server.zig` compiles and links
// unchanged, with WebPDecodeRGB returning NULL — which the single call site
// (`server.zig`, `... orelse return null`) already treats as "cannot decode".
//
// The effect is that WebP image input is unsupported off Apple; PNG and JPEG
// still decode via stb_image. Replace this stub with a real vendored libwebp to
// lift that.
#ifndef MLX_SERVE_WEBP_STUB_DECODE_H
#define MLX_SERVE_WEBP_STUB_DECODE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

unsigned char *WebPDecodeRGB(const unsigned char *data, size_t data_size,
                             int *width, int *height);
void WebPFree(void *ptr);

#ifdef __cplusplus
}
#endif

#endif
