#include "xnor_nn.h"

static size_t sz(const void *s){
    const xnor_nn_weights_binarizer_t *self =
        (const xnor_nn_weights_binarizer_t*)s;
    size_t elems = self->sz[0] * self->sz[1] * self->sz[2] * self->sz[3];
    return elems * sizeof(float);
}

static xnor_nn_status_t copy_on_float(
        const void *s, const void *from, void *to) {
    const xnor_nn_weights_binarizer_t *self =
        (const xnor_nn_weights_binarizer_t*)s;
    const float *f = (const float*)from;
    float *t = (float*)to;
    int elems = self->sz[0] * self->sz[1] * self->sz[2] * self->sz[3];

#   pragma omp parallel for
    for(int i = 0; i < elems; i++) t[i] = f[i];

    return xnor_nn_success;
}

xnor_nn_status_t xnor_nn_init_weights_binarizer(xnor_nn_weights_binarizer_t *b,
        int oc, int ic, int kh, int kw) {
    b->sz[0] = kw;
    b->sz[1] = kh;
    b->sz[2] = ic;
    b->sz[3] = oc;

    b->size = sz;
    b->execute = copy_on_float;

    return xnor_nn_success;
}
