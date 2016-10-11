#include "xnor_nn.h"

static size_t sz(const void *s){
    const xnor_nn_data_binarizer_t *self = (const xnor_nn_data_binarizer_t*)s;
    size_t elems = self->mb * self->sz[0] * self->sz[1] * self->sz[2];
    return elems * sizeof(float);
}

static xnor_nn_status_t copy_on_float(
        const void *s, const void *from, void *to) {
    const xnor_nn_data_binarizer_t *self = (const xnor_nn_data_binarizer_t*)s;
    const float *f = (const float*)from;
    float *t = (float*)to;
    int elems = self->mb * self->sz[0] * self->sz[1] * self->sz[2];
    for(int i = 0; i < elems; i++)
        t[i] = f[i];
    return xnor_nn_success;
}

xnor_nn_status_t xnor_nn_init_data_binarizer(xnor_nn_data_binarizer_t *b,
        int mb, int c, int h, int w) {
    b->mb = mb;
    b->sz[0] = w;
    b->sz[1] = h;
    b->sz[2] = c;

    b->size = sz;
    b->execute = copy_on_float;

    return xnor_nn_success;
}
