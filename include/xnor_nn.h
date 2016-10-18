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

xnor_nn_status_t xnor_nn_init_convolution(xnor_nn_convolution_t *c,
        int mb, int oc, int ic, int ih, int iw,
        int kh, int kw, int sh, int sw, int ph, int pw);

xnor_nn_status_t xnor_nn_init_data_binarizer(xnor_nn_data_binarizer_t *b,
        const xnor_nn_convolution_t *c);

xnor_nn_status_t xnor_nn_init_weights_binarizer(xnor_nn_weights_binarizer_t *b,
        const xnor_nn_convolution_t *c);

#ifdef __cplusplus
}
#endif
#endif
