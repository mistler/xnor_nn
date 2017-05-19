#include "bcast_convolution.hpp"

#include <arm_neon.h>
#include <cstdint>

#include "utils.hpp"
#include "convolution_logger.hpp"

#include "isa_traits.hpp"
#include "convolution_traits.hpp"
#include "unroller.hpp"

namespace xnor_nn {
namespace implementation {
template<typename Traits>

template<typename isa_traits, int OC, int IC, int IH, int IW, int KH, int KW,
    int SH, int SW, int PH, int PW>
xnor_nn_status_t BcastConvolution<Traits>::exec(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    typedef typename Traits::data_t data_t;

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

    const data_t *src = (const data_t*)res[xnor_nn_resource_bin_src];
    const data_t *weights = (const data_t*)res[xnor_nn_resource_bin_weights];
    const float *alpha = (const float*)res[xnor_nn_resource_alpha];
    const float *k = (const float*)res[xnor_nn_resource_k];
    const data_t *op_c = (const data_t*)res[xnor_nn_resource_operations_count];
    float *dst = (float*)res[xnor_nn_resource_user_dst];

    const int MB = c->mb;
    const auto dfmt = c->dst_format;
    if (dfmt != xnor_nn_data_format_nchw && dfmt != xnor_nn_data_format_nhwc)
        return xnor_nn_unimplemented;

    constexpr int OH = utils::getOH(IH, KH, SH, PH);
    constexpr int OW = utils::getOW(IW, KW, SW, PW);
    constexpr int ICO = getICO(IC);
    constexpr int OCO = getOCO(OC, VLEN);
    constexpr int OCI = getOCI(VLEN);

    constexpr int MAX_ICO_UNROLL = 4;
    constexpr int MAX_OW_UNROLL = 4;

    constexpr int unroll_ow = get_unroll_factor(OW, MAX_OW_UNROLL);
    constexpr int unroll_ico = get_unroll_factor(ICO, MAX_ICO_UNROLL);

    using namespace xnor_nn::utils;
    logger::log<logger::exec, logger::convolution>::info(c,
            sizeof(data_t) == sizeof(int32_t) ? "bcast_int" : "bcast_short",
            "NEON", "uow:", unroll_ow, "uico:", unroll_ico);

    // TODO: try to use one of 32x4 and 16x8
#   pragma omp parallel for collapse(3) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oco = 0; oco < OCO; oco++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow += unroll_ow) {
        int32x4_t v_accum32x4[unroll_ow];
        int16x8_t v_accum16x8[unroll_ow];
        for (int uow = 0; uow < unroll_ow; uow++) {
            v_accum32x4[uow] = veorq_s32(v_accum32x4[uow], v_accum32x4[uow]);
            v_accum16x8[uow] = veorq_s16(v_accum16x8[uow], v_accum16x8[uow]);
        }

        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            const data_t *weights_ic_oci = weights + ((oco*KH +kh)*KW + kw)*ICO*OCI;

            for (int ico = 0; ico < ICO; ico += unroll_ico) {
                int32x4_t v_weights32x4[unroll_ico];
                int16x8_t v_weights16x8[unroll_ico];
                auto load_v_weights = [&](const int uico) {
                    if (sizeof(data_t) == sizeof(int32_t)) {
                        v_weights32x4[uico] = vld1q_s32((const int32_t*)(weights_ic_oci + (ico+uico)*OCI));
                    } else {
                        v_weights16x8[uico] = vld1q_s16((const int16_t*)(weights_ic_oci + (ico+uico)*OCI));
                    }
                };
                unroller<unroll_ico>::unroll(load_v_weights);

                for (int uow = 0; uow < unroll_ow; uow++) {
                    const int ih = oh*SH - PH + kh;
                    const int iw = (ow + uow)*SW - PW + kw;

                    if (PH != 0 || PW != 0) { // May be incorrect for assymet pad
                        if (ih < 0 || iw < 0) continue;
                        if (ih >= IH || iw >= IW) continue;
                    }
                    const data_t *src_ic = src + ((mb*IH + ih)*IW + iw)*ICO;

                    auto kernel = [&](const int uico) {
                        if (sizeof(data_t) == sizeof(int32_t)) {
                            int32x4_t v_src = vld1q_dup_s32((const int32_t*)(src_ic + (ico+uico)));

                            int32x4_t v_xor = veorq_s32(v_src, v_weights32x4[uico]);
                            int32x4_t v_xnor = vmvnq_s32(v_xor);

                            int8x16_t v_cnt16 = vcntq_s8(vreinterpretq_s8_s32(v_xnor));
                            int16x8_t v_cnt8 = vpaddlq_s8(v_cnt16);
                            int32x4_t v_cnt4 = vpaddlq_s16(v_cnt8);
                            v_accum32x4[uow] = vaddq_s32(v_cnt4, v_accum32x4[uow]);
                        } else {
                            int16x8_t v_src = vld1q_dup_s16((const int16_t*)(src_ic + (ico+uico)));

                            int16x8_t v_xor = veorq_s16(v_src, v_weights16x8[uico]);
                            int16x8_t v_xnor = vmvnq_s16(v_xor);

                            int8x16_t v_cnt16 = vcntq_s8(vreinterpretq_s8_s16(v_xnor));
                            int16x8_t v_cnt8 = vpaddlq_s8(v_cnt16);
                            v_accum16x8[uow] = vaddq_s16(v_cnt8, v_accum16x8[uow]);
                        }
                    };
                    unroller<unroll_ico>::unroll(kernel);
                }
            }
        }
        // TODO: single instruction mov
        for (int uow = 0; uow < unroll_ow; uow++) {
            data_t d_arr[OCI] = { 0u };
            if (sizeof(data_t) == sizeof(int32_t)) {
                vst1q_s32((int32_t*)d_arr, v_accum32x4[uow]);
            } else {
                vst1q_s16((int16_t*)d_arr, v_accum16x8[uow]);
            }
            for (int oci = 0; oci < OCI; oci++) {
                int dst_idx = -1;
                switch (dfmt) {
                case xnor_nn_data_format_nchw:
                    dst_idx = ((mb*OC + oco*OCI + oci)*OH + oh)*OW + (ow+uow);
                    break;
                case xnor_nn_data_format_nhwc:
                    dst_idx = ((mb*OH + oh)*OW + (ow+uow))*OC + oco*OCI + oci;
                    break;
                default: break;
                }
                dst[dst_idx] = (d_arr[oci]*2 - op_c[oh*OW + (ow+uow)]) * alpha[oco*OCI + oci] * k[(mb*OH + oh)*OW + (ow+uow)];
            }
        }
    }

    return xnor_nn_success;
}

template<typename Traits>
template<typename isa_traits>
xnor_nn_status_t BcastConvolution<Traits>::exec(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    typedef typename Traits::data_t data_t;

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

    const data_t *src = (const data_t*)res[xnor_nn_resource_bin_src];
    const data_t *weights = (const data_t*)res[xnor_nn_resource_bin_weights];
    const float *alpha = (const float*)res[xnor_nn_resource_alpha];
    const float *k = (const float*)res[xnor_nn_resource_k];
    const data_t *op_c = (const data_t*)res[xnor_nn_resource_operations_count];
    float *dst = (float*)res[xnor_nn_resource_user_dst];

    auto *state = reinterpret_cast<BcastConvolution<Traits>*>(getState(c));

    const int MB = c->mb;
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

    const auto dfmt = c->dst_format;
    if (dfmt != xnor_nn_data_format_nchw && dfmt != xnor_nn_data_format_nhwc)
        return xnor_nn_unimplemented;

    const int ICO = state->ICO;
    const int OCO = state->OCO;
    const int OCI = state->OCI;

    using namespace xnor_nn::utils;
    logger::log<logger::exec, logger::convolution>::info(c,
            sizeof(data_t) == sizeof(int32_t) ? "bcast_int" : "bcast_short",
            "ISA:", "NEON");

#   pragma omp parallel for collapse(3) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oco = 0; oco < OCO; oco++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        int16x8_t v_accum16x8 = veorq_s16(v_accum16x8, v_accum16x8);
        int32x4_t v_accum32x4 = veorq_s32(v_accum32x4, v_accum32x4);

        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            const int ih = oh*SH - PH + kh;
            const int iw = ow*SW - PW + kw;

            if (ih < 0 || iw < 0) continue;
            if (ih >= IH || iw >= IW) continue;

            const data_t *src_ic = src + ((mb*IH + ih)*IW + iw)*ICO;
            const data_t *weights_ic_oci = weights + ((oco*KH +kh)*KW + kw)*ICO*OCI;

            for (int ico = 0; ico < ICO; ico++) {
                if (sizeof(data_t) == sizeof(int32_t)) {
                    int32x4_t v_src = vld1q_dup_s32((const int32_t*)(src_ic + ico));
                    int32x4_t v_weights = vld1q_s32((const int32_t*)(weights_ic_oci + ico*OCI));

                    int32x4_t v_xor = veorq_s32(v_src, v_weights);
                    int32x4_t v_xnor = vmvnq_s32(v_xor);

                    int8x16_t v_cnt16 = vcntq_s8(vreinterpretq_s8_s32(v_xnor));
                    int16x8_t v_cnt8 = vpaddlq_s8(v_cnt16);
                    int32x4_t v_cnt4 = vpaddlq_s16(v_cnt8);
                    v_accum32x4 = vaddq_s32(v_cnt4, v_accum32x4);
                } else {
                    int16x8_t v_src = vld1q_dup_s16((const int16_t*)(src_ic + ico));
                    int16x8_t v_weights = vld1q_s16((const int16_t*)(weights_ic_oci + ico*OCI));

                    int16x8_t v_xor = veorq_s16(v_src, v_weights);
                    int16x8_t v_xnor = vmvnq_s16(v_xor);

                    int8x16_t v_cnt16 = vcntq_s8(vreinterpretq_s8_s16(v_xnor));
                    int16x8_t v_cnt8 = vpaddlq_s8(v_cnt16);
                    v_accum16x8 = vaddq_s16(v_cnt8, v_accum16x8);
                }
            }
        }
        // TODO: single instruction mov
        data_t d_arr[OCI] = { 0u };
        if (sizeof(data_t) == sizeof(int32_t)) {
            vst1q_s32((int32_t*)d_arr, v_accum32x4);
        } else {
            vst1q_s16((int16_t*)d_arr, v_accum16x8);
        }
        for (int oci = 0; oci < OCI; oci++) {
            int dst_idx = -1;
            switch (dfmt) {
            case xnor_nn_data_format_nchw:
                dst_idx = ((mb*OC + oco*OCI + oci)*OH + oh)*OW + ow;
                break;
            case xnor_nn_data_format_nhwc:
                dst_idx = ((mb*OH + oh)*OW + ow)*OC + oco*OCI + oci;
                break;
            default: break;
            }
            dst[dst_idx] = (d_arr[oci]*2 - op_c[oh*OW + ow]) * alpha[oco*OCI + oci] * k[(mb*OH + oh)*OW + ow];
        }
    }

    return xnor_nn_success;
}

using isa = xnor_nn::isa::isa_traits<xnor_nn::isa::isa_neon>;
#include "bcast_instantiator.hxx"

} // namespace implementation
} // namespace xnor_nn
