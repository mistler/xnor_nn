#include <map>

#include "implementation.hpp"

namespace {
std::map<xnor_nn_algorithm_t, xnor_nn_executor_t> map_bin_weights{
    { xnor_nn_algorithm_reference, reference_weights_copy_on_float },
    { xnor_nn_algorithm_optimized, direct_binarize_weights_char },
};
std::map<xnor_nn_algorithm_t, xnor_nn_executor_t> map_bin_data{
    { xnor_nn_algorithm_reference, reference_data_copy_on_float },
    { xnor_nn_algorithm_optimized, direct_binarize_data_char },
};
std::map<xnor_nn_algorithm_t, xnor_nn_executor_t> map_calculate_k{
    { xnor_nn_algorithm_reference, reference_calculate_k },
    { xnor_nn_algorithm_optimized, reference_calculate_k },
};
std::map<xnor_nn_algorithm_t, xnor_nn_executor_t> map_conv_fwd{
    { xnor_nn_algorithm_reference, reference_convolution_forward },
    { xnor_nn_algorithm_optimized, direct_convolution_forward },
};
}

xnor_nn_executor_t get_executor(const xnor_nn_algorithm_t &algorithm,
        const operation_t &operation) {
    switch (operation) {
        case operation_binarize_weights:
            return map_bin_weights[algorithm];
        case operation_binarize_data:
            return map_bin_data[algorithm];
        case operation_calculate_k:
            return map_calculate_k[algorithm];
        case operation_convolution_forward:
            return map_conv_fwd[algorithm];
    }
    return nullptr;
}
