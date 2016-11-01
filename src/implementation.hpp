#ifndef IMPLEMENTATION_HPP
#define IMPLEMENTATION_HPP

#include <unordered_map>
#include <string>

#include "xnor_nn_types.h"

typedef enum {
    operation_binarize_weights,
    operation_binarize_data,
    operation_calculate_k,
    operation_convolution_forward,
} operation_t;

xnor_nn_status_t reference_data_copy_on_float(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res);

xnor_nn_status_t direct_binarize_data_char(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res);

xnor_nn_status_t reference_calculate_k(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res);

xnor_nn_status_t reference_weights_copy_on_float(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res);

xnor_nn_status_t direct_binarize_weights_char(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res);

xnor_nn_status_t reference_convolution_forward(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res);

xnor_nn_status_t direct_convolution_forward(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res);

xnor_nn_executor_t get_executor(const xnor_nn_algorithm_t &algorithm,
        const operation_t &operation);

#endif
