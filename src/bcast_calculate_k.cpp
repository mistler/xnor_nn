#include "bcast_convolution.hpp"

#include <cmath>

#include "convolution_logger.hpp"
#include "xnor_nn_types.h"
#include "convolution_traits.hpp"

namespace xnor_nn {
namespace implementation {

template<typename Traits>
xnor_nn_status_t BcastConvolution<Traits>::calculate_k(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    if (
        res[xnor_nn_resource_user_src] == nullptr
        || res[xnor_nn_resource_a] == nullptr
        || res[xnor_nn_resource_k] == nullptr
        || c == nullptr
    ) return xnor_nn_error_invalid_input;
    const float *from = (float*)res[xnor_nn_resource_user_src];
    float *a = (float*)res[xnor_nn_resource_a];
    float *k = (float*)res[xnor_nn_resource_k];

    const int MB = c->mb;
    const int IC = c->ic;
    const int IH = c->ih;
    const int IW = c->iw;
    const int OH = c->oh;
    const int OW = c->ow;
    const int KH = c->kh;
    const int KW = c->kw;
    const int SH = c->sh;
    const int SW = c->sw;
    const int PH = c->ph;
    const int PW = c->pw;

    const auto sfmt = c->src_format;
    if (sfmt != xnor_nn_data_format_nchw && sfmt != xnor_nn_data_format_nhwc)
        return xnor_nn_unimplemented;

    const float C = 1.f / IC;
    const float KHW = 1.f / KH / KW;

    using namespace xnor_nn::utils;
    logger::log<logger::exec, logger::k>::info(c);

    // Calculate A
#   pragma omp parallel for collapse(3) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int ih = 0; ih < IH; ih++)
    for (int iw = 0; iw < IW; iw++) {
        float *a_curr = a + (mb*IH + ih)*IW + iw;
        *a_curr = 0.f;
        for (int ic = 0; ic < IC; ic++) {
            int from_idx = -1;
            switch (sfmt) {
            case xnor_nn_data_format_nchw:
                from_idx = ((mb*IC + ic)*IH + ih)*IW + iw; break;
            case xnor_nn_data_format_nhwc:
                from_idx = ((mb*IH + ih)*IW + iw)*IC + ic; break;
            default: break;
            }
            *a_curr += std::fabs(from[from_idx]) * C;
        }
    }

    // Calculate K
#   pragma omp parallel for collapse(3) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float *k_curr = k + (mb*OH + oh)*OW + ow;
        *k_curr = 0.f;
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            const int ih = oh*SH - PH + kh;
            const int iw = ow*SW - PW + kw;

            if (ih < 0 || iw < 0) continue;
            if (ih >= IH || iw >= IW) continue;

            *k_curr += a[(mb*IH + ih)*IW + iw] * KHW;
        }
    }

    return xnor_nn_success;
}

template xnor_nn_status_t BcastConvolution<ConvolutionTraits<
    IntConvolutionTraits>>::calculate_k(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res);
template xnor_nn_status_t BcastConvolution<ConvolutionTraits<
    ShortConvolutionTraits>>::calculate_k(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res);

template<> xnor_nn_status_t BcastConvolution<ConvolutionTraits<
        RuntimeConvolutionTraits>>::calculate_k(
    const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    (void)c; (void)res; return xnor_nn_unimplemented;
}

} // namespace implementation
} // namespace xnor_nn
