#include "bcast_convolution.hpp"

#include <functional>

#include "convolution_logger.hpp"
#include "xnor_nn_types.h"
#include "convolution_traits.hpp"

namespace xnor_nn {
namespace implementation {

template<typename Traits>
xnor_nn_status_t BcastConvolution<Traits>::binarize_data(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    if (
        res[xnor_nn_resource_user_src] == nullptr
        || res[xnor_nn_resource_bin_src] == nullptr
        || c == nullptr
    ) return xnor_nn_error_invalid_input;
    const unsigned int *from = (unsigned int*)res[xnor_nn_resource_user_src];
    unsigned char *to = (unsigned char*)res[xnor_nn_resource_bin_src];

    const int MB = c->mb;
    const int IC = c->ic;
    const int IH = c->ih;
    const int IW = c->iw;

    auto *state = reinterpret_cast<BcastConvolution<ConvolutionTraits<
        RuntimeConvolutionTraits>>*>(getState(c));

    const int SZ = state->SZ;

    const int BIC = state->BIC;
    const int ABIC = state->ABIC;

    const auto sfmt = c->src_format;
    if (sfmt != xnor_nn_data_format_nchw && sfmt != xnor_nn_data_format_nhwc)
        return xnor_nn_unimplemented;

    using namespace xnor_nn::utils;
    logger::log<logger::exec, logger::data>::info(c);

#   pragma omp parallel for collapse(3) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int ih = 0; ih < IH; ih++)
    for (int iw = 0; iw < IW; iw++) {
        for (int bic = 0; bic < BIC; bic++) {
            unsigned char out{0};
            int LEN = bic == BIC - 1 ? (IC % SZ) : SZ;
            if (LEN == 0) LEN = SZ;
            for (int ic = 0; ic < LEN; ic++) {
                int from_idx = -1;
                switch (sfmt) {
                case xnor_nn_data_format_nchw:
                    from_idx = ((mb*IC + bic*SZ + ic)*IH + ih)*IW + iw; break;
                case xnor_nn_data_format_nhwc:
                    from_idx = ((mb*IH + ih)*IW + iw)*IC + bic*SZ + ic; break;
                default: break;
                }

                char tmp = (~from[from_idx]) >> 31;
                out <<= 1;
                out |= tmp;
            }
            if (LEN != SZ) out <<= SZ-LEN;
            int to_idx = ((mb*IH + ih)*IW + iw)*ABIC + bic;
            to[to_idx] = out;
        }
        for (int r = 0; r < ABIC - BIC; r++) {
            int to_idx = ((mb*IH + ih)*IW + iw)*ABIC + BIC + r;
            to[to_idx] = 0x00u;
        }
    }

    return xnor_nn_success;
}

template xnor_nn_status_t BcastConvolution<ConvolutionTraits<
    IntConvolutionTraits>>::binarize_data(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res);
template xnor_nn_status_t BcastConvolution<ConvolutionTraits<
    ShortConvolutionTraits>>::binarize_data(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res);

template<> xnor_nn_status_t BcastConvolution<ConvolutionTraits<
            RuntimeConvolutionTraits>>::binarize_data(
    const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    (void)c; (void)res; return xnor_nn_unimplemented;
}

} // namespace implementation
} // namespace xnor_nn
