#include "bcast_convolution.hpp"

#include <cstdint>

#include "utils.hpp"
#include "logger.hpp"

#include "isa_traits.hpp"
#include "convolution_traits.hpp"

namespace xnor_nn {
namespace implementation {

template<typename Traits>
template<typename isa_traits, int OC, int IC, int IH, int IW, int KH, int KW,
    int SH, int SW, int PH, int PW>
xnor_nn_status_t BcastConvolution<Traits>::exec(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    typedef typename Traits::data_t data_t;
    typedef typename Traits::udata_t udata_t;

    constexpr int VLEN = isa_traits::vlen;
    (void)VLEN;

    if (
        res[xnor_nn_resource_bin_src] == nullptr
        || res[xnor_nn_resource_bin_weights] == nullptr
        || res[xnor_nn_resource_user_dst] == nullptr
        || res[xnor_nn_resource_k] == nullptr
        || res[xnor_nn_resource_alpha] == nullptr
        || res[xnor_nn_resource_operations_count] == nullptr
        || c == nullptr
    ) return xnor_nn_error_invalid_input;

    const data_t *src = (data_t*)res[xnor_nn_resource_bin_src];
    const data_t *weights = (data_t*)res[xnor_nn_resource_bin_weights];
    const float *alpha = (const float*)res[xnor_nn_resource_alpha];
    const float *k = (const float*)res[xnor_nn_resource_k];
    const data_t *op_c = (const data_t*)res[xnor_nn_resource_operations_count];
    float *dst = (float*)res[xnor_nn_resource_user_dst];

    const int MB = c->mb;
    constexpr int OH = utils::getOH(IH, KH, SH, PH);
    constexpr int OW = utils::getOW(IW, KW, SW, PW);
    constexpr int ICO = getICO(IC);
    constexpr int OCO = getOCO(OC, VLEN);
    constexpr int OCI = getOCI(VLEN);

    LOG_INFO("convolution:\t", "execute:",
            "[", MB, "][", IC, "][", IH, "][", IW, "]",
            "x",
            "[", OC, "][", IC, "][", KH, "][", KW, "]",
            "=",
            "[", MB, "][", OC, "][", OH, "][", OW, "]",
            "stride: [", SH, "][", SW, "]",
            "pad: [", PH, "][", PW, "]",
            sizeof(data_t) == sizeof(int32_t) ? "bcast_int" : "bcast_short",
            "ISA:", "default" , "Templated");

    // TODO: potentially loops can be reordered
    // TODO: check collapse value for performance
#   pragma omp parallel for collapse(3) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oco = 0; oco < OCO; oco++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        int d_arr[16] = { 0 };
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            const int ih = oh*SH - PH + kh;
            const int iw = ow*SW - PW + kw;

            if (ih < 0 || iw < 0) continue;
            if (ih >= IH || iw >= IW) continue;

            const data_t *src_ic =
                src + ((mb*IH + ih)*IW + iw)*ICO;
            const data_t *weights_ic_oci =
                weights + ((oco*KH +kh)*KW + kw)*ICO*OCI;

            for (int ico = 0; ico < ICO; ico++)
            for (int oci = 0; oci < OCI; oci++) {
                const int src_idx = ico;
                const int weights_idx = ico*OCI + oci;

                const data_t bsrc = src_ic[src_idx];
                const data_t bweights = weights_ic_oci[weights_idx];

                const udata_t result = ~(bsrc ^ bweights);
                d_arr[oci] += __builtin_popcount(result);
            }
        }
        for (int oci = 0; oci < OCI; oci++)
            dst[((mb*OC + oco*OCI + oci)*OH + oh)*OW + ow] =
                (d_arr[oci]*2 - op_c[oh*OW + ow])
                * alpha[oco*OCI + oci] * k[(mb*OH + oh)*OW + ow];
    }

    return xnor_nn_success;
}

template<typename Traits>
template<typename isa_traits>
xnor_nn_status_t BcastConvolution<Traits>::exec(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    typedef typename Traits::data_t data_t;
    typedef typename Traits::udata_t udata_t;

    constexpr int VLEN = isa_traits::vlen;
    (void)VLEN;

    if (
        res[xnor_nn_resource_bin_src] == nullptr
        || res[xnor_nn_resource_bin_weights] == nullptr
        || res[xnor_nn_resource_user_dst] == nullptr
        || res[xnor_nn_resource_k] == nullptr
        || res[xnor_nn_resource_alpha] == nullptr
        || res[xnor_nn_resource_operations_count] == nullptr
        || c == nullptr
    ) return xnor_nn_error_invalid_input;

    const data_t *src = (data_t*)res[xnor_nn_resource_bin_src];
    const data_t *weights = (data_t*)res[xnor_nn_resource_bin_weights];
    const float *alpha = (const float*)res[xnor_nn_resource_alpha];
    const float *k = (const float*)res[xnor_nn_resource_k];
    const data_t *op_c = (const data_t*)res[xnor_nn_resource_operations_count];
    float *dst = (float*)res[xnor_nn_resource_user_dst];

    const int MB = c->mb;

    auto *state = reinterpret_cast<BcastConvolution<Traits>*>(getState(c));

    const int OC = c->oc;
    const int OH = c->oh;
    const int OW = c->ow;
    const int IC = c->ic;
    const int IH = c->ih;
    const int IW = c->iw;
    const int KH = c->kh;
    const int KW = c->kw;
    const int SH = c->sh;
    const int SW = c->sw;
    const int PH = c->ph;
    const int PW = c->pw;

    const int ICO = state->ICO;
    const int OCO = state->OCO;
    const int OCI = state->OCI;

    LOG_INFO("convolution:\t", "execute:",
            "[", MB, "][", IC, "][", IH, "][", IW, "]",
            "x",
            "[", OC, "][", IC, "][", KH, "][", KW, "]",
            "=",
            "[", MB, "][", OC, "][", OH, "][", OW, "]",
            "stride: [", SH, "][", SW, "]",
            "pad: [", PH, "][", PW, "]",
            sizeof(data_t) == sizeof(int32_t) ? "bcast_int" : "bcast_short",
            "ISA:", "default");

    // TODO: potentially loops can be reordered
    // TODO: check collapse value for performance
#   pragma omp parallel for collapse(3) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oco = 0; oco < OCO; oco++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        data_t d_arr[16] = { 0 };
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            const int ih = oh*SH - PH + kh;
            const int iw = ow*SW - PW + kw;

            if (ih < 0 || iw < 0) continue;
            if (ih >= IH || iw >= IW) continue;

            const data_t *src_ic =
                src + ((mb*IH + ih)*IW + iw)*ICO;
            const data_t *weights_ic_oci =
                weights + ((oco*KH +kh)*KW + kw)*ICO*OCI;

            for (int ico = 0; ico < ICO; ico++)
            for (int oci = 0; oci < OCI; oci++) {
                const int src_idx = ico;
                const int weights_idx = ico*OCI + oci;

                const data_t bsrc = src_ic[src_idx];
                const data_t bweights = weights_ic_oci[weights_idx];

                const udata_t result = ~(bsrc ^ bweights);
                d_arr[oci] += __builtin_popcount(result);
            }
        }
        for (int oci = 0; oci < OCI; oci++)
            dst[((mb*OC + oco*OCI + oci)*OH + oh)*OW + ow] =
                (d_arr[oci]*2 - op_c[oh*OW + ow])
                * alpha[oco*OCI + oci] * k[(mb*OH + oh)*OW + ow];
    }

    return xnor_nn_success;
}

using isa = xnor_nn::isa::isa_traits<xnor_nn::isa::isa_default>;
#include "bcast_instantiator.hxx"

} // namespace implementation
} // namespace xnor_nn
