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
        switch (c->algorithm) {
        case xnor_nn_algorithm_reference: {
            xnor_nn_memory_allocate(res + xnor_nn_resource_bin_src,
                    c->mb * c->ic * c->ih * c->iw * sizeof(float));
            xnor_nn_memory_allocate(res + xnor_nn_resource_bin_weights,
                    c->oc * c->ic * c->kh * c->kw * sizeof(float));
            xnor_nn_memory_allocate(res + xnor_nn_resource_a,
                    c->ih * c->iw * sizeof(float));
            xnor_nn_memory_allocate(res + xnor_nn_resource_k,
                    c->oh * c->ow * sizeof(float));
            break;
        }
        case xnor_nn_algorithm_optimized: {
            const size_t BIC = (c->ic + 8 - 1) / 8;
            xnor_nn_memory_allocate(res + xnor_nn_resource_bin_src,
                    c->mb * BIC * c->ih * c->iw);
            xnor_nn_memory_allocate(res + xnor_nn_resource_bin_weights,
                    c->oc * c->ic * c->kh * c->kw * sizeof(float));
            xnor_nn_memory_allocate(res + xnor_nn_resource_a,
                    c->ih * c->iw * sizeof(float));
            xnor_nn_memory_allocate(res + xnor_nn_resource_k,
                    c->oh * c->ow * sizeof(float));
            break;
        }
        }
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
