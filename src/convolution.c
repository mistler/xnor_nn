#include "xnor_nn.h"

static xnor_nn_status_t fwd(const void *s,
        const void *src, const void *weights, void *dst) {
    const xnor_nn_convolution_t *self = s;
    (void)self;
    (void)src;
    (void)weights;
    (void)dst;
    return xnor_nn_success;
}

xnor_nn_status_t xnor_nn_init_convolution(xnor_nn_convolution_t *c,
        int mb, int oc, int ic, int ih, int iw,
        int kh, int kw, int sh, int sw, int ph, int pw) {
    (void)c;
    (void)mb;
    (void)ic;
    (void)ih;
    (void)iw;
    (void)oc;
    (void)kh;
    (void)kw;
    (void)sh;
    (void)sw;
    (void)ph;
    (void)pw;

    c->forward = fwd;
    return xnor_nn_success;
}
