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
    xnor_nn_unimplemented,
} xnor_nn_status_t;

typedef enum {
    xnor_nn_algorithm_reference,
    xnor_nn_algorithm_optimized,
} xnor_nn_algorithm_t;

typedef enum {
    xnor_nn_resource_number = 8,

    xnor_nn_resource_user_src = 0,
    xnor_nn_resource_user_weights = 1,
    xnor_nn_resource_user_dst = 2,

    xnor_nn_resource_bin_src = 3,
    xnor_nn_resource_bin_weights = 4,
    xnor_nn_resource_a = 5,
    xnor_nn_resource_k = 6,
    xnor_nn_resource_alpha = 7,
} xnor_nn_resource_type_t;

typedef void *xnor_nn_resources_t[xnor_nn_resource_number];

typedef struct xnor_nn_data_binarizer_ xnor_nn_data_binarizer_t;
typedef struct xnor_nn_weights_binarizer_ xnor_nn_weights_binarizer_t;
typedef struct xnor_nn_convolution_ xnor_nn_convolution_t;

typedef xnor_nn_status_t(*executor)(const xnor_nn_convolution_t *self,
        xnor_nn_resources_t res);

struct xnor_nn_convolution_ {
    xnor_nn_algorithm_t algorithm;

    int mb;
    int ic, ih, iw;
    int oc, oh, ow;
    int sh, sw;
    int kh, kw;
    int ph, pw;

    size_t resource_size[xnor_nn_resource_number];

    executor binarize_weights;
    executor binarize_data;
    executor calculate_k;
    executor forward;
};

#ifdef __cplusplus
}
#endif
#endif
