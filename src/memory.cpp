#include "xnor_nn.h"

#include <cstdlib>

namespace {

void xnor_nn_memory_allocate(void **ptr, size_t sz) {
    void *mem = aligned_alloc(64, sz);
    if (mem == NULL) {
        *ptr = NULL;
        throw 0;
    }
    *ptr = mem;
}

void xnor_nn_memory_free(void *ptr) {
    if (ptr == NULL) return;
    free(ptr);
}

}

xnor_nn_status_t xnor_nn_allocate_resources(const xnor_nn_convolution_t *c,
        xnor_nn_resources_t res) {
    try {
        for (int i = xnor_nn_resource_internal; i < xnor_nn_resource_number; i++)
            xnor_nn_memory_allocate(res + i, c->resource_size[i]);
    } catch (int err) {
        xnor_nn_free_resources(res);
        return xnor_nn_error_memory;
    }
    return xnor_nn_success;
}

void xnor_nn_free_resources(xnor_nn_resources_t res) {
    for (int i = xnor_nn_resource_internal; i < xnor_nn_resource_number; i++) {
        xnor_nn_memory_free(res[i]);
        res[i] = nullptr;
    }
} // namespace
