#include "reference_convolution.hpp"

#include <cmath>

#include "xnor_nn_types.h"
#include "logger.hpp"

namespace xnor_nn {
namespace implementation {

xnor_nn_status_t ReferenceConvolution::binarize_data(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    if (
        res[xnor_nn_resource_user_src] == nullptr
        || res[xnor_nn_resource_bin_src] == nullptr
        || c == nullptr
    ) return xnor_nn_error_invalid_input;
    const float *from = (float*)res[xnor_nn_resource_user_src];
    float *to = (float *)res[xnor_nn_resource_bin_src];

    const int MB = c->mb;
    const int IC = c->ic;
    const int IH = c->ih;
    const int IW = c->iw;

    const int elems = MB*IC*IH*IW;

    LOG_INFO("binarize_data:\t", "execute:",
            "[", MB, "][", IC, "][", IH, "][", IW, "]",
            "->",
            "[", MB, "][", IC, "][", IH, "][", IW, "]",
            "Algorithm:", "reference");

#   pragma omp parallel for schedule(static)
    for(int i = 0; i < elems; i++) to[i] = from[i];

    return xnor_nn_success;
}

} // namespace implementation
} // namespace xnor_nn
