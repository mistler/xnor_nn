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

typedef enum {
    xnor_nn_algorithm_reference,
    xnor_nn_algorithm_optimized,
} xnor_nn_algorithm_t;

typedef struct xnor_nn_data_binarizer_ xnor_nn_data_binarizer_t;
typedef struct xnor_nn_weights_binarizer_ xnor_nn_weights_binarizer_t;
typedef struct xnor_nn_convolution_ xnor_nn_convolution_t;

struct xnor_nn_data_binarizer_ {
    const xnor_nn_convolution_t *c;

    xnor_nn_status_t (*binarize)(const xnor_nn_data_binarizer_t *self,
            const void *from, void *to);
    xnor_nn_status_t (*calculate_k)(const xnor_nn_data_binarizer_t *self,
            const void *from, void *to);
    size_t (*size)(const xnor_nn_data_binarizer_t *self);
};

struct xnor_nn_weights_binarizer_ {
    const xnor_nn_convolution_t *c;

    xnor_nn_status_t (*execute)(const xnor_nn_weights_binarizer_t *self,
            const void *from, void *to);
    size_t (*size)(const xnor_nn_weights_binarizer_t *self);
};

struct xnor_nn_convolution_ {
    xnor_nn_algorithm_t algorithm;

    int mb;
    int ic, ih, iw;
    int oc, oh, ow;
    int sh, sw;
    int kh, kw;
    int ph, pw;

    void *workspace;

    xnor_nn_status_t (*forward)(const xnor_nn_convolution_t *self,
            const void *src, const void *weights, void *dst);
};

#ifdef __cplusplus
}
#endif
#endif
