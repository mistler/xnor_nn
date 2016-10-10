#ifndef XNOR_NN_TYPES_H
#define XNOR_NN_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    xnor_nn_success,
    xnor_nn_error_memory,
    xnor_nn_error_invalid_input,
} xnor_nn_status_t;

typedef struct {
    int mb;
    int sz[3];

    xnor_nn_status_t (*execute)(const void *self,
            const void *from, void *to);
    size_t (*size)(const void *self);
} xnor_nn_data_binarizer_t;

typedef struct {
    int sz[4];

    xnor_nn_status_t (*execute)(const void *self,
            const void *from, void *to);
    size_t (*size)(const void *self);
} xnor_nn_weights_binarizer_t;

typedef struct {
    int mb;
    int in[3];
    int out[3];
    int stride[2];
    int padding[2];

    xnor_nn_status_t (*forward)(const void *self,
            const void *src, const void *weights, void *dst);
} xnor_nn_convolution_t;

#ifdef __cplusplus
}
#endif
#endif
