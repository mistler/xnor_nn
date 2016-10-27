#include <cstdlib>

#include "xnor_nn.h"

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
        xnor_nn_memory_allocate(res + xnor_nn_resource_bin_src,
                c->resource_size[xnor_nn_resource_bin_src]);
        xnor_nn_memory_allocate(res + xnor_nn_resource_bin_weights,
                c->resource_size[xnor_nn_resource_bin_weights]);
        xnor_nn_memory_allocate(res + xnor_nn_resource_a,
                c->resource_size[xnor_nn_resource_a]);
        xnor_nn_memory_allocate(res + xnor_nn_resource_k,
                c->resource_size[xnor_nn_resource_k]);
    } catch (int err) {
        xnor_nn_free_resources(res);
        return xnor_nn_error_memory;
    }
    return xnor_nn_success;
}

void xnor_nn_free_resources(xnor_nn_resources_t res) {
        xnor_nn_memory_free(res[xnor_nn_resource_bin_src]);
        xnor_nn_memory_free(res[xnor_nn_resource_bin_weights]);
        xnor_nn_memory_free(res[xnor_nn_resource_a]);
        xnor_nn_memory_free(res[xnor_nn_resource_k]);
        res[xnor_nn_resource_bin_src] = nullptr;
        res[xnor_nn_resource_bin_weights] = nullptr;
        res[xnor_nn_resource_a] = nullptr;
        res[xnor_nn_resource_k] = nullptr;
}
