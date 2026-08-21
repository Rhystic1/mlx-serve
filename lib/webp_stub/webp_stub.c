// See webp/decode.h in this directory: an honest "no WebP decoder in this
// build" implementation, not a placeholder that pretends to work. Returning
// NULL is exactly the signal the caller already handles.
#include "webp/decode.h"

unsigned char *WebPDecodeRGB(const unsigned char *data, size_t data_size,
                             int *width, int *height) {
    (void)data; (void)data_size;
    if (width) *width = 0;
    if (height) *height = 0;
    return 0;
}

void WebPFree(void *ptr) { (void)ptr; }
