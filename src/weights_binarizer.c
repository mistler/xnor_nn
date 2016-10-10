#include "xnor_nn.h"

static size_t sz(const void *s){
    const xnor_nn_weights_binarizer_t *self = s;
    (void)self;
    return 1;
}

static xnor_nn_status_t exec(const void *s, const void *from, void *to) {
    const xnor_nn_weights_binarizer_t *self = s;
    (void)self;
    (void)from;
    (void)to;
    return xnor_nn_success;
}

xnor_nn_status_t xnor_nn_init_weights_binarizer(xnor_nn_weights_binarizer_t *b,
        int oc, int ic, int kh, int kw) {
    (void)b;
    (void)oc;
    (void)ic;
    (void)kh;
    (void)kw;

    b->size = sz;
    b->execute = exec;
    return xnor_nn_success;
}
