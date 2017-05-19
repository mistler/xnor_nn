#ifndef XNOR_NN_H
#define XNOR_NN_H

#include <stddef.h>

#include "xnor_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void xnor_nn_get_status_message(char *msg, xnor_nn_status_t status);

xnor_nn_status_t xnor_nn_allocate_resources(const xnor_nn_convolution_t *c,
        xnor_nn_resources_t res);
void xnor_nn_free_resources(xnor_nn_resources_t res);

void xnor_nn_binarize_weights_float(const xnor_nn_convolution_t *c,
        const float *from, float *to);

xnor_nn_status_t xnor_nn_init_convolution(xnor_nn_convolution_t *c,
        const xnor_nn_algorithm_t algorithm,
        const xnor_nn_tensor_format_t src_fmt,
        const xnor_nn_tensor_format_t weights_fmt,
        const xnor_nn_tensor_format_t dst_fmt,
        int mb, int oc, int ic, int ih, int iw,
        int kh, int kw, int sh, int sw, int ph, int pw);

void xnor_nn_destroy_convolution(xnor_nn_convolution_t *c);

#ifdef __cplusplus
}
#endif
#endif
