#ifndef XNOR_NN_TYPES_H
#define XNOR_NN_TYPES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    xnor_nn_success,
    xnor_nn_error_memory,
    xnor_nn_error_invalid_input,
} xnor_nn_status_t;

typedef struct {
    int mb, c, h, w;

    xnor_nn_status_t (*execute)(const void *self,
            const void *from, void *to);
    size_t (*size)(const void *self);
} xnor_nn_data_binarizer_t;

typedef struct {
    int oc, ic, kh, kw;

    xnor_nn_status_t (*execute)(const void *self,
            const void *from, void *to);
    size_t (*size)(const void *self);
} xnor_nn_weights_binarizer_t;

typedef struct {
    int mb;
    int ic, ih, iw;
    int oc, oh, ow;
    int sh, sw;
    int kh, kw;
    int ph, pw;

    void *workspace;

    xnor_nn_status_t (*forward)(const void *self,
            const void *src, const void *weights, void *dst);
} xnor_nn_convolution_t;

#ifdef __cplusplus
}
#endif
#endif
