#include "xnor_nn.h"

void xnor_nn_get_status_message(char *msg, xnor_nn_status_t status) {
    (void)msg;
    (void)status;
}


xnor_nn_status_t xnor_nn_memory_allocate(void **ptr, size_t sz) {
    (void)ptr;
    (void)sz;
    return xnor_nn_success;
}

void xnor_nn_memory_free(void *ptr) {
    (void)ptr;
}

xnor_nn_status_t xnor_nn_init_data_binarizer(xnor_nn_data_binarizer_t *b,
        int mb, int c, int h, int w) {
    (void)b;
    (void)mb;
    (void)c;
    (void)h;
    (void)w;
    return xnor_nn_success;
}

xnor_nn_status_t xnor_nn_init_weights_binarizer(xnor_nn_weights_binarizer_t *b,
        int oc, int ic, int kh, int kw) {
    (void)b;
    (void)oc;
    (void)ic;
    (void)kh;
    (void)kw;
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
    return xnor_nn_success;
}
