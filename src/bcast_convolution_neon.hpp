#include "bcast_convolution.hpp"

#include <arm_neon.h>

#include "utils.h"

// TODO: log execution

namespace xnor_nn {
namespace implementation {

#ifdef TEMPLATE_CONVOLUTION
template<int OC, int IC, int IH, int IW, int KH, int KW,
    int SH, int SW, int PH, int PW, int OH, int OW, int OCO, int ICO, int OCI>
xnor_nn_status_t BcastConvolution::exec_template(
#else
xnor_nn_status_t BcastConvolution::exec_simple(
#endif
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    if (
        res[xnor_nn_resource_bin_src] == nullptr
        || res[xnor_nn_resource_bin_weights] == nullptr
        || res[xnor_nn_resource_user_dst] == nullptr
        || res[xnor_nn_resource_k] == nullptr
        || c == nullptr
    ) return xnor_nn_error_invalid_input;
    const unsigned int *src = (unsigned int*)res[xnor_nn_resource_bin_src];
    const unsigned int *weights =
        (unsigned int*)res[xnor_nn_resource_bin_weights];
    float *dst = (float*)res[xnor_nn_resource_user_dst];
    float *alpha = (float*)&res[xnor_nn_resource_alpha];
    const float *k = (float*)res[xnor_nn_resource_k];

    const int MB = c->mb;

#ifdef TEMPLATE_CONVOLUTION
#else
    const int OC = c->oc;
    const int OH = c->oh;
    const int OW = c->ow;
    const int IH = c->ih;
    const int IW = c->iw;
    const int KH = c->kh;
    const int KW = c->kw;
    const int SH = c->sh;
    const int SW = c->sw;
    const int PH = c->ph;
    const int PW = c->pw;

    BcastConvolution *state = reinterpret_cast<BcastConvolution*>(
            getState(c, xnor_nn_operation_convolution_forward));

    constexpr int OCI = state->OCI;

    const int ICO = state->ICO;
    const int OCO = state->OCO;
#endif

    // TODO: potentially loops can be reordered
    // TODO: check collapse value for performance
#   pragma omp parallel for collapse(4) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oco = 0; oco < OCO; oco++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        uint32x4_t v_accum = veorq_u32(v_accum, v_accum);

        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            const int ih = oh*SH - PH + kh;
            const int iw = ow*SW - PW + kw;

            if (ih < 0 || iw < 0) continue;
            if (ih >= IH || iw >= IW) continue;

            const unsigned int *src_ic =
                src + ((mb*IH + ih)*IW + iw)*ICO;
            const unsigned int *weights_ic_oci =
                weights + ((oco*KH +kh)*KW + kw)*ICO*OCI;

            for (int ico = 0; ico < ICO; ico++) {
                uint32x4_t v_src = vdupq_n_u32(src_ic[ico]);
                uint32x4_t v_weights = vld1q_u32(weights_ic_oci + ico*OCI);

                uint32x4_t v_xor = veorq_u32(v_src, v_weights);
                uint32x4_t v_xnor = vmvnq_u32(v_xor);

                uint8x16_t v_cnt16 = vcntq_u8(vreinterpretq_u8_u32(v_xnor));
                uint16x8_t v_cnt8 = vpaddlq_u8(v_cnt16);
                uint32x4_t v_cnt4 = vpaddlq_u16(v_cnt8);
                v_accum = vaddq_u32(v_cnt4, v_accum);
            }
        }
        // TODO: single instruction mov
        unsigned int d_arr[OCI] = { 0u };
        vst1q_u32(d_arr, v_accum);
        for (int i = 0; i < OCI; i++)
            dst[((mb*OC + oco*OCI + i)*OH + oh)*OW + ow] =
                d_arr[i] * *alpha * k[oh*OW + ow];
    }

    return xnor_nn_success;
}

} // namespace implementation
} // namespace xnor_nn
