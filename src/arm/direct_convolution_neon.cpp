#include "direct_convolution.hpp"

#include <arm_neon.h>

#include "utils.hpp"
#include "logger.hpp"

#include "direct_template_parameters.hpp"

namespace xnor_nn {
namespace implementation {

#ifdef TEMPLATED
template<int OC, int IC, int IH, int IW, int KH, int KW, int SH, int SW,
    int PH, int PW>
xnor_nn_status_t DirectConvolution::exec_neon_template(
#else
xnor_nn_status_t DirectConvolution::exec_neon_simple(
#endif
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    if (
        res[xnor_nn_resource_bin_src] == nullptr
        || res[xnor_nn_resource_bin_weights] == nullptr
        || res[xnor_nn_resource_user_dst] == nullptr
        || res[xnor_nn_resource_k] == nullptr
        || c == nullptr
    ) return xnor_nn_error_invalid_input;
    const int *src = (int*)res[xnor_nn_resource_bin_src];
    const int *weights = (int*)res[xnor_nn_resource_bin_weights];
    float *dst = (float*)res[xnor_nn_resource_user_dst];
    float *alpha = (float*)&res[xnor_nn_resource_alpha];
    const float *k = (float*)res[xnor_nn_resource_k];

    const int MB = c->mb;

    DirectConvolution *state = reinterpret_cast<DirectConvolution*>(
            getState(c, xnor_nn_operation_convolution_forward));

#ifdef TEMPLATED
    constexpr int OH = getOH(IH, KH, SH, PH);
    constexpr int OW = getOW(IW, KW, SW, PW);
    constexpr int ABIC = state->constexpr_getABIC(IC);
#else
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

    const int ABIC = state->ABIC;
#endif

    constexpr int ELEM_SIZE = 32;
    const int VECTORS_IN_ABIC = ABIC / VLEN;

    LOG_INFO("convolution:\t", "execute:",
            "[", MB, "][", IC, "][", IH, "][", IW, "]",
            "x",
            "[", OC, "][", IC, "][", KH, "][", KW, "]",
            "=",
            "[", MB, "][", OC, "][", OH, "][", OW, "]",
            "stride: [", SH, "][", SW, "]",
            "pad: [", PH, "][", PW, "]",
            "Algorithm:", "direct", "ISA:", "NEON"
#ifdef TEMPLATED
            , "Templated"
#endif
            );

    // TODO: potentially loops can be reordered
#   pragma omp parallel for collapse(4) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oc = 0; oc < OC; oc++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        int dst_idx = ((mb*OC + oc)*OH + oh)*OW + ow;
        float *d = dst + dst_idx;
        *d = 0.f;
        int32x4_t v_accum = veorq_s32(v_accum, v_accum);
        int8x16_t v_eight = vmovq_n_s8(8);
        int8x16_t v_two = vmovq_n_s8(2);
        long long int dst_i = 0;

        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            const int ih = oh*SH - PH + kh;
            const int iw = ow*SW - PW + kw;

            if (ih < 0 || iw < 0) continue;
            if (ih >= IH || iw >= IW) continue;

            const int *src_ic = src + ((mb*IH + ih)*IW + iw)*ABIC/ELEM_SIZE;
            const int *weights_ic =
                weights + ((kh*KW + kw)*OC + oc)*ABIC/ELEM_SIZE;

            for (int vabic = 0; vabic < VECTORS_IN_ABIC; vabic++) {
                int32x4_t v_src = vld1q_s32(src_ic + vabic*VLEN/ELEM_SIZE);
                int32x4_t v_weights =
                    vld1q_s32(weights_ic + vabic*VLEN/ELEM_SIZE);

                int32x4_t v_xor = veorq_s32(v_src, v_weights);
                int32x4_t v_xnor = vmvnq_s32(v_xor);

                int8x16_t v_cnt16 = vcntq_s8(vreinterpretq_s8_s32(v_xnor));
                v_cnt16 = vmulq_s8(v_cnt16, v_two);
                v_cnt16 = vsubq_s8(v_cnt16, v_eight);
                int16x8_t v_cnt8 = vpaddlq_s8(v_cnt16);
                int32x4_t v_cnt4 = vpaddlq_s16(v_cnt8);
                v_accum = vaddq_s32(v_cnt4, v_accum);
            }
        }
        int64x2_t v_cnt2 = vpaddlq_s32(v_accum);
        dst_i += vgetq_lane_s64(v_cnt2, 0);
        dst_i += vgetq_lane_s64(v_cnt2, 1);
        *d = (float)dst_i * *alpha * k[oh*OW + ow];
    }

    return xnor_nn_success;
}

#ifdef TEMPLATED

DIRECT_TEMPLATE_INSTANTIATE(neon);

#endif

} // namespace implementation
} // namespace xnor_nn
