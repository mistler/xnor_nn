#include "bcast_convolution.hpp"

#include "utils.hpp"
#include "logger.hpp"

#include "isa_traits.hpp"

namespace xnor_nn {
namespace implementation {

inline static void execute(const int MB, const int OC,
        const int OH, const int OW, const int IC, const int IH, const int IW,
        const int KH, const int KW, const int SH, const int SW,
        const int PH, const int PW,
        const int ICO, const int OCO, const int OCI, const int VLEN,
        const int *src, const int *weights, const float *alpha, const float *k,
        const int *op_c, float *dst) {
    (void)VLEN;

    // TODO: potentially loops can be reordered
    // TODO: check collapse value for performance
#   pragma omp parallel for collapse(4) schedule(static)
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

            const int *src_ic = src + ((mb*IH + ih)*IW + iw)*ICO;
            const int *weights_ic_oci = weights + ((oco*KH +kh)*KW + kw)*ICO*OCI;

            for (int ico = 0; ico < ICO; ico++)
            for (int oci = 0; oci < OCI; oci++) {
                int src_idx = ico;
                int weights_idx = ico*OCI + oci;

                int bsrc = src_ic[src_idx];
                int bweights = weights_ic_oci[weights_idx];

                int result = ~(bsrc ^ bweights);
                d_arr[oci] += __builtin_popcount(result);
            }
        }
        const int oi = oh*OW + ow;
        for (int i = 0; i < OCI; i++)
            dst[((mb*OC + oco*OCI + i)*OH + oh)*OW + ow] =
                (d_arr[i]*2 - op_c[oi]) * *alpha * k[oi];
    }
}

template<typename isa_traits, int OC, int IC, int IH, int IW, int KH, int KW,
    int SH, int SW, int PH, int PW>
xnor_nn_status_t BcastConvolution::exec(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    constexpr int VLEN = isa_traits::vlen;
    if (
        res[xnor_nn_resource_bin_src] == nullptr
        || res[xnor_nn_resource_bin_weights] == nullptr
        || res[xnor_nn_resource_user_dst] == nullptr
        || res[xnor_nn_resource_k] == nullptr
        || res[xnor_nn_resource_operations_count] == nullptr
        || c == nullptr
    ) return xnor_nn_error_invalid_input;
    const int *src = (int*)res[xnor_nn_resource_bin_src];
    const int *weights = (int*)res[xnor_nn_resource_bin_weights];
    const float *alpha = (const float*)&res[xnor_nn_resource_alpha];
    const float *k = (const float*)res[xnor_nn_resource_k];
    const int *op_c = (const int*)res[xnor_nn_resource_operations_count];
    float *dst = (float*)res[xnor_nn_resource_user_dst];

    const int MB = c->mb;
    constexpr int OH = utils::getOH(IH, KH, SH, PH);
    constexpr int OW = utils::getOW(IW, KW, SW, PW);
    constexpr int ICO = constexpr_getICO(IC);
    constexpr int OCO = constexpr_getOCO(OC, VLEN);
    constexpr int OCI = constexpr_getOCI(VLEN);

    LOG_INFO("convolution:\t", "execute:",
            "[", MB, "][", IC, "][", IH, "][", IW, "]",
            "x",
            "[", OC, "][", IC, "][", KH, "][", KW, "]",
            "=",
            "[", MB, "][", OC, "][", OH, "][", OW, "]",
            "stride: [", SH, "][", SW, "]",
            "pad: [", PH, "][", PW, "]",
            "Algorithm:", "bcast", "ISA:", "default" , "Templated");

    execute(MB, OC, OH, OW, IC, IH, IW, KH, KW, SH, SW, PH, PW,
        ICO, OCO, OCI, VLEN, src, weights, alpha, k, op_c, dst);

    return xnor_nn_success;
}

template<typename isa_traits>
xnor_nn_status_t BcastConvolution::exec(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    constexpr int VLEN = isa_traits::vlen;
    if (
        res[xnor_nn_resource_bin_src] == nullptr
        || res[xnor_nn_resource_bin_weights] == nullptr
        || res[xnor_nn_resource_user_dst] == nullptr
        || res[xnor_nn_resource_k] == nullptr
        || res[xnor_nn_resource_operations_count] == nullptr
        || c == nullptr
    ) return xnor_nn_error_invalid_input;
    const int *src = (int*)res[xnor_nn_resource_bin_src];
    const int *weights = (int*)res[xnor_nn_resource_bin_weights];
    const float *alpha = (const float*)&res[xnor_nn_resource_alpha];
    const float *k = (const float*)res[xnor_nn_resource_k];
    const int *op_c = (const int*)res[xnor_nn_resource_operations_count];
    float *dst = (float*)res[xnor_nn_resource_user_dst];

    const int MB = c->mb;

    BcastConvolution *state = reinterpret_cast<BcastConvolution*>(getState(c));

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
    const int OCI = state->getOCI();

    LOG_INFO("convolution:\t", "execute:",
            "[", MB, "][", IC, "][", IH, "][", IW, "]",
            "x",
            "[", OC, "][", IC, "][", KH, "][", KW, "]",
            "=",
            "[", MB, "][", OC, "][", OH, "][", OW, "]",
            "stride: [", SH, "][", SW, "]",
            "pad: [", PH, "][", PW, "]",
            "Algorithm:", "bcast", "ISA:", "AVX");

    execute(MB, OC, OH, OW, IC, IH, IW, KH, KW, SH, SW, PH, PW,
        ICO, OCO, OCI, VLEN, src, weights, alpha, k, op_c, dst);

    return xnor_nn_success;
}

using isa = xnor_nn::isa::isa_traits<xnor_nn::isa::isa_default>;
using algorithm = BcastConvolution;
#include "instantiator.hxx"

} // namespace implementation
} // namespace xnor_nn
