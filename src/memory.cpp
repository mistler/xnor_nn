#include <mm_malloc.h>

#include "xnor_nn.h"

xnor_nn_status_t xnor_nn_memory_allocate(void **ptr, size_t sz) {
    if (ptr == NULL) return xnor_nn_error_invalid_input;
    void *mem = _mm_malloc(sz, 64);
    if (mem == NULL) {
        *ptr = NULL;
        return xnor_nn_error_memory;
    }
    *ptr = mem;
    return xnor_nn_success;
}

void xnor_nn_memory_free(void *ptr) {
    if (ptr == NULL) return;
    _mm_free(ptr);
}
