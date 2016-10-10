#include "xnor_nn.h"

static size_t sz(const void *s){
    const xnor_nn_data_binarizer_t *self = s;
    (void)self;
    return 1;
}

static xnor_nn_status_t exec(const void *s, const void *from, void *to) {
    const xnor_nn_data_binarizer_t *self = s;
    (void)self;
    (void)from;
    (void)to;
    return xnor_nn_success;
}

xnor_nn_status_t xnor_nn_init_data_binarizer(xnor_nn_data_binarizer_t *b,
        int mb, int c, int h, int w) {
    (void)b;
    (void)mb;
    (void)c;
    (void)h;
    (void)w;

    b->size = sz;
    b->execute = exec;
    return xnor_nn_success;
}
