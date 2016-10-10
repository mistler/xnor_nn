#ifndef XNOR_NN_H
#define XNOR_NN_H

#include <stddef.h>

#include "xnor_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void xnor_nn_get_status_message(char *msg, xnor_nn_status_t status);

xnor_nn_status_t xnor_nn_memory_allocate(void **ptr, size_t sz);
void xnor_nn_memory_free(void *ptr);

xnor_nn_status_t xnor_nn_init_data_binarizer(xnor_nn_data_binarizer_t *b,
        int mb, int c, int h, int w);

xnor_nn_status_t xnor_nn_init_weights_binarizer(xnor_nn_weights_binarizer_t *b,
        int oc, int ic, int kh, int kw);

xnor_nn_status_t xnor_nn_init_convolution(xnor_nn_convolution_t *c,
        int mb, int ic, int ih, int iw, int oc, int oh, int ow,
        int kh, int kw, int sh, int sw, int ph, int pw);

#ifdef __cplusplus
}
#endif
#endif
