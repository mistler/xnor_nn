#include "reference_convolution.hpp"

#include <cmath>

#include "xnor_nn_types.h"
#include "logger.hpp"

namespace xnor_nn {
namespace implementation {

xnor_nn_status_t ReferenceConvolution::binarize_weights(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    if (
        res[xnor_nn_resource_user_weights] == nullptr
        || res[xnor_nn_resource_bin_weights] == nullptr
        || c == nullptr
    ) return xnor_nn_error_invalid_input;
    const float *from = (float*)res[xnor_nn_resource_user_weights];
    float *to = (float*)res[xnor_nn_resource_bin_weights];
    float *alpha = (float*)res[xnor_nn_resource_alpha];

    const int OC = c->oc;
    const int IC = c->ic;
    const int KH = c->kh;
    const int KW = c->kw;

    const int elems = OC*IC*KH*KW;

    LOG_INFO("binarize_weights:", "execute:",
            "[", OC, "][", IC, "][", KH, "][", KW, "]",
            "->",
            "[", OC, "][", IC, "][", KH, "][", KW, "]",
            "Algorithm:", "reference");

#   pragma omp parallel for schedule(static)
    for(int i = 0; i < elems; i++) to[i] = from[i];

    const float chw = 1.f / (IC*KH*KW);

    // Calculate alpha
#   pragma omp parallel for schedule(static)
    for (int oc = 0; oc < OC; oc++) {
        float *curr_alpha = alpha + oc;
        *curr_alpha = 0.f;
        for (int ic = 0; ic < IC; ic++)
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++)
            *curr_alpha +=
                std::fabs(from[((oc*IC + ic)*KH + kh)*KW + kw]) * chw;
    }

    return xnor_nn_success;
}

} // namespace implementation
} // namespace xnor_nn
