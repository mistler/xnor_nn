#include "xnor_nn.h"

xnor_nn_status_t xnor_nn_memory_allocate(void **ptr, size_t sz) {
    (void)ptr;
    (void)sz;
    return xnor_nn_success;
}

void xnor_nn_memory_free(void *ptr) {
    (void)ptr;
}
