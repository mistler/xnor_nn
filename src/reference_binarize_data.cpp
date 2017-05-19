#include "reference_convolution.hpp"

#include <cmath>

#include "xnor_nn_types.h"
#include "convolution_logger.hpp"

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

    const auto sfmt = c->src_format;
    if (sfmt != xnor_nn_data_format_nchw && sfmt != xnor_nn_data_format_nhwc)
        return xnor_nn_unimplemented;

    using namespace xnor_nn::utils;
    logger::log<logger::exec, logger::data>::info(c);

#   pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int ih = 0; ih < IH; ih++)
    for (int iw = 0; iw < IW; iw++)
    for (int ic = 0; ic < IC; ic++) {
        int src_idx = -1;
        switch (sfmt) {
        case xnor_nn_data_format_nchw:
            src_idx = ((mb*IC + ic)*IH + ih)*IW + iw; break;
        case xnor_nn_data_format_nhwc:
            src_idx = ((mb*IH + ih)*IW + iw)*IC + ic; break;
        default: break;
        }
        to[((mb*IH + ih)*IW + iw)*IC + ic] = from[src_idx];
    }

    return xnor_nn_success;
}

} // namespace implementation
} // namespace xnor_nn
