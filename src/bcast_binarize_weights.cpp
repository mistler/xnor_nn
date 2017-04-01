#include "bcast_convolution.hpp"

#include <cmath>

#include "logger.hpp"
#include "xnor_nn_types.h"
#include "convolution_traits.hpp"

namespace xnor_nn {
namespace implementation {

template<typename Traits>
xnor_nn_status_t BcastConvolution<Traits>::binarize_weights(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    if (
        res[xnor_nn_resource_user_weights] == nullptr
        || res[xnor_nn_resource_bin_weights] == nullptr
        || res[xnor_nn_resource_operations_count] == nullptr
        || res[xnor_nn_resource_alpha] == nullptr
        || c == nullptr
    ) return xnor_nn_error_invalid_input;

    typedef typename Traits::data_t data_t;

    const float *from = (float*)res[xnor_nn_resource_user_weights];
    unsigned char *to = (unsigned char*)res[xnor_nn_resource_bin_weights];
    float *alpha = (float*)res[xnor_nn_resource_alpha];
    data_t *op_c = (data_t*)res[xnor_nn_resource_operations_count];
    const unsigned int *f = (unsigned int*)from;


    const int OC = c->oc;
    const int IC = c->ic;
    const int KH = c->kh;
    const int KW = c->kw;

    auto *state = reinterpret_cast<BcastConvolution<ConvolutionTraits<
        RuntimeConvolutionTraits>>*>(getState(c));

    const int SZ = state->SZ;
    const int BICI = state->BICI;

    const int OCI = state->OCI;
    const int ICO = state->ICO;
    const int OCO = state->OCO;

    LOG_INFO("binarize_weights:", "execute:",
            "[", OC, "][", IC, "][", KH, "][", KW, "]",
            "->",
            "[", OCO, "][", KH, "][", KW, "][", ICO, "][", OCI, "][", BICI, "]",
            "Algorithm:", "bcast");

#   pragma omp parallel for collapse(3) schedule(static)
    for (int oco = 0; oco < OCO; oco++)
    for (int kh = 0; kh < KH; kh++)
    for (int kw = 0; kw < KW; kw++)
    for (int ico = 0; ico < ICO; ico++)
    for (int oci = 0; oci < OCI; oci++) {
        for (int bici = 0; bici < BICI; bici++) {
            unsigned char out{0};
            const int current_ic = (ico*BICI + bici)*SZ;
            int ic = 0;
            for (; ic < SZ && current_ic + ic < IC; ic++) {
                const int from_oc = oco*OCI + oci;
                const int from_ic = current_ic + ic;
                const int from_idx = ((from_oc*IC + from_ic)*KH + kh)*KW + kw;
                char tmp = (~f[from_idx]) >> 31;
                out <<= 1;
                out |= tmp;
            }
            if (ic != SZ) {
                // Dirty hack! As this data is fake we want it to have
                // zero influence to the dst after convolution, so lets fill it
                // with ONES so ~(src^weights) = ~(0^1) = 0
                for (int i = 0; i < SZ - ic; i++) {
                    out <<= 1;
                    out |= (unsigned char)1;
                }
            }
            const int to_idx =
                ((((oco*KH + kh)*KW + kw)*ICO + ico)*OCI + oci)*BICI + bici;
            to[to_idx] = out;
        }
    }

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

    const int IH = c->ih;
    const int IW = c->iw;
    const int OH = c->oh;
    const int OW = c->ow;
    const int SH = c->sh;
    const int SW = c->sw;
    const int PH = c->ph;
    const int PW = c->pw;

#   pragma omp parallel for collapse(2) schedule(static)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        data_t *curr_op_c = op_c + oh*OW + ow;
        *curr_op_c = 0;

        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            const int ih = oh*SH - PH + kh;
            const int iw = ow*SW - PW + kw;

            if (ih < 0 || iw < 0) continue;
            if (ih >= IH || iw >= IW) continue;

            *curr_op_c += IC;
        }
    }

    return xnor_nn_success;
}

template xnor_nn_status_t BcastConvolution<ConvolutionTraits<
    IntConvolutionTraits>>::binarize_weights(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res);
template xnor_nn_status_t BcastConvolution<ConvolutionTraits<
    ShortConvolutionTraits>>::binarize_weights(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res);

template<> xnor_nn_status_t BcastConvolution<ConvolutionTraits<
        RuntimeConvolutionTraits>>::binarize_weights(
    const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    (void)c; (void)res; return xnor_nn_unimplemented;
}

} // namespace implementation
} // namespace xnor_nn
