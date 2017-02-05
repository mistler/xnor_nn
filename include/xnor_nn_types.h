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
    xnor_nn_algorithm_reference = 0,
    xnor_nn_algorithm_direct = 1,
    xnor_nn_algorithm_bcast = 2,

    xnor_nn_algorithm_fastest = 3,
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

typedef enum {
    xnor_nn_operation_convolution_forward = 0,
    xnor_nn_operation_binarize_weights = 1,
    xnor_nn_operation_binarize_data = 2,
    xnor_nn_operation_calculate_k = 3,
} xnor_nn_operation_t;

typedef void *xnor_nn_resources_t[xnor_nn_resource_number];

typedef struct xnor_nn_convolution_ xnor_nn_convolution_t;

typedef xnor_nn_status_t(*xnor_nn_executor_t)(const xnor_nn_convolution_t *self,
        xnor_nn_resources_t res);

struct xnor_nn_convolution_ {
    xnor_nn_algorithm_t algorithm;

    int mb;
    int ic, ih, iw;
    int oc, oh, ow;
    int sh, sw;
    int kh, kw;
    int ph, pw;

    int vlen; // for testing purposes only

    size_t resource_size[xnor_nn_resource_number];

    xnor_nn_executor_t binarize_weights;
    xnor_nn_executor_t binarize_data;
    xnor_nn_executor_t calculate_k;
    xnor_nn_executor_t forward;

    void *state;
};

#ifdef __cplusplus
}
#endif
#endif
