#include "bcast_convolution.hpp"

#include <immintrin.h>
#include <cstdint>

#include "utils.hpp"
#include "logger.hpp"

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
    constexpr int OH = utils::getOH(IH, KH, SH, PH);
    constexpr int OW = utils::getOW(IW, KW, SW, PW);
    constexpr int ICO = getICO(IC);
    constexpr int OCO = getOCO(OC, VLEN);
    constexpr int OCI = getOCI(VLEN);

    constexpr int MAX_ICO_UNROLL = 4;
    constexpr int MAX_OW_UNROLL = 4;

    constexpr int unroll_ow = get_unroll_factor(OW, MAX_OW_UNROLL);
    constexpr int unroll_ico = get_unroll_factor(ICO, MAX_ICO_UNROLL);

    LOG_INFO("convolution:\t", "execute:",
            "[", MB, "][", IC, "][", IH, "][", IW, "]",
            "x",
            "[", OC, "][", IC, "][", KH, "][", KW, "]",
            "=",
            "[", MB, "][", OC, "][", OH, "][", OW, "]",
            "stride: [", SH, "][", SW, "]",
            "pad: [", PH, "][", PW, "]",
            sizeof(data_t) == sizeof(int32_t) ? "bcast_int" : "bcast_short",
            "ATOM", "uow:", unroll_ow, "uico:", unroll_ico);

#   pragma omp parallel for collapse(3) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oco = 0; oco < OCO; oco++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow += unroll_ow) {
        __m128i d_arr[unroll_ow];
        for (int i = 0; i < unroll_ow; i++) d_arr[i] = _mm_set1_epi8(0);
        const __m128i v_ones = _mm_set1_epi32(-1);

        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            const data_t *weights_ic_oci = weights + ((oco*KH +kh)*KW + kw)*ICO*OCI;

            for (int ico = 0; ico < ICO; ico += unroll_ico) {
                __m128i v_weights[unroll_ico];
                auto load_v_weights = [&](const int uico) {
                    v_weights[uico] = _mm_castps_si128(_mm_load_ps((const float*)(weights_ic_oci + (ico+uico)*OCI)));
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
                        const __m128i v_src = sizeof(data_t) == sizeof(int32_t)
                            ? _mm_set1_epi32(src_ic[ico + uico]) : _mm_set1_epi16(src_ic[ico + uico]);

                        const __m128i v_xor = _mm_xor_si128(v_src, v_weights[uico]);
                        const __m128i v_xnor = _mm_xor_si128(v_xor, v_ones);

                        const __m128i v_mask_8 = _mm_set1_epi16(0x000F);
                        const __m128i v_mask_16 = _mm_set1_epi32(0x0000FFFF);
                        const __m128i v_popcnt_table = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);

                        const __m128i v_popcnt_8hi = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(v_xnor, v_mask_8));
                        const __m128i v_popcnt_8lo = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(_mm_srli_epi16(v_xnor, 4), v_mask_8));
                        const __m128i v_popcnt_8 = _mm_add_epi16(v_popcnt_8lo, v_popcnt_8hi);
                        const __m128i v_popcnt_16hi = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(_mm_srli_epi16(v_xnor, 8), v_mask_8));
                        const __m128i v_popcnt_16lo = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(_mm_srli_epi16(v_xnor, 12), v_mask_8));
                        const __m128i v_popcnt_16 = _mm_add_epi16(_mm_add_epi16(v_popcnt_16lo, v_popcnt_16hi), v_popcnt_8);

                        if (sizeof(data_t) == sizeof(int32_t)) {
                            const __m128i v_popcnt_32hi = _mm_and_si128(v_popcnt_16, v_mask_16);
                            const __m128i v_popcnt_32lo = _mm_and_si128(_mm_srli_epi32(v_popcnt_16, 16), v_mask_16);
                            d_arr[uow] = _mm_add_epi32(d_arr[uow], v_popcnt_32lo);
                            d_arr[uow] = _mm_add_epi32(d_arr[uow], v_popcnt_32hi);
                        } else {
                            d_arr[uow] = _mm_add_epi16(d_arr[uow], v_popcnt_16);
                        }
                    };
                    unroller<unroll_ico>::unroll(kernel);
                }
            }
        }

        auto store = [&](const int uow) {
            if (sizeof(data_t) == sizeof(int32_t)) {
                dst[((mb*OC + oco*OCI + 0)*OH + oh)*OW + (ow+uow)] =
                    ((_mm_extract_epi16(d_arr[uow], 0)+_mm_extract_epi16(d_arr[uow], 1))*2 - op_c[oh*OW + (ow+uow)]) * alpha[oco*OCI + 0] * k[(mb*OH + oh)*OW + (ow+uow)];
                dst[((mb*OC + oco*OCI + 1)*OH + oh)*OW + (ow+uow)] =
                    ((_mm_extract_epi16(d_arr[uow], 2)+_mm_extract_epi16(d_arr[uow], 3))*2 - op_c[oh*OW + (ow+uow)]) * alpha[oco*OCI + 1] * k[(mb*OH + oh)*OW + (ow+uow)];
                dst[((mb*OC + oco*OCI + 2)*OH + oh)*OW + (ow+uow)] =
                    ((_mm_extract_epi16(d_arr[uow], 4)+_mm_extract_epi16(d_arr[uow], 5))*2 - op_c[oh*OW + (ow+uow)]) * alpha[oco*OCI + 2] * k[(mb*OH + oh)*OW + (ow+uow)];
                dst[((mb*OC + oco*OCI + 3)*OH + oh)*OW + (ow+uow)] =
                    ((_mm_extract_epi16(d_arr[uow], 6)+_mm_extract_epi16(d_arr[uow], 7))*2 - op_c[oh*OW + (ow+uow)]) * alpha[oco*OCI + 3] * k[(mb*OH + oh)*OW + (ow+uow)];
            } else {
                dst[((mb*OC + oco*OCI + 0)*OH + oh)*OW + (ow+uow)] =
                    (_mm_extract_epi16(d_arr[uow], 0)*2 - op_c[oh*OW + (ow+uow)]) * alpha[oco*OCI + 0] * k[(mb*OH + oh)*OW + (ow+uow)];
                dst[((mb*OC + oco*OCI + 1)*OH + oh)*OW + (ow+uow)] =
                    (_mm_extract_epi16(d_arr[uow], 1)*2 - op_c[oh*OW + (ow+uow)]) * alpha[oco*OCI + 1] * k[(mb*OH + oh)*OW + (ow+uow)];
                dst[((mb*OC + oco*OCI + 2)*OH + oh)*OW + (ow+uow)] =
                    (_mm_extract_epi16(d_arr[uow], 2)*2 - op_c[oh*OW + (ow+uow)]) * alpha[oco*OCI + 2] * k[(mb*OH + oh)*OW + (ow+uow)];
                dst[((mb*OC + oco*OCI + 3)*OH + oh)*OW + (ow+uow)] =
                    (_mm_extract_epi16(d_arr[uow], 3)*2 - op_c[oh*OW + (ow+uow)]) * alpha[oco*OCI + 3] * k[(mb*OH + oh)*OW + (ow+uow)];
                dst[((mb*OC + oco*OCI + 4)*OH + oh)*OW + (ow+uow)] =
                    (_mm_extract_epi16(d_arr[uow], 4)*2 - op_c[oh*OW + (ow+uow)]) * alpha[oco*OCI + 4] * k[(mb*OH + oh)*OW + (ow+uow)];
                dst[((mb*OC + oco*OCI + 5)*OH + oh)*OW + (ow+uow)] =
                    (_mm_extract_epi16(d_arr[uow], 5)*2 - op_c[oh*OW + (ow+uow)]) * alpha[oco*OCI + 5] * k[(mb*OH + oh)*OW + (ow+uow)];
                dst[((mb*OC + oco*OCI + 6)*OH + oh)*OW + (ow+uow)] =
                    (_mm_extract_epi16(d_arr[uow], 6)*2 - op_c[oh*OW + (ow+uow)]) * alpha[oco*OCI + 6] * k[(mb*OH + oh)*OW + (ow+uow)];
                dst[((mb*OC + oco*OCI + 7)*OH + oh)*OW + (ow+uow)] =
                    (_mm_extract_epi16(d_arr[uow], 7)*2 - op_c[oh*OW + (ow+uow)]) * alpha[oco*OCI + 7] * k[(mb*OH + oh)*OW + (ow+uow)];
            }
        };

        unroller<unroll_ow>::unroll(store);
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
            "ISA:", "ATOM");

#   pragma omp parallel for collapse(3) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oco = 0; oco < OCO; oco++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        __m128i d_arr = _mm_set1_epi8(0);
        const __m128i v_ones = _mm_set1_epi32(-1);

        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            const int ih = oh*SH - PH + kh;
            const int iw = ow*SW - PW + kw;

            if (ih < 0 || iw < 0) continue;
            if (ih >= IH || iw >= IW) continue;

            const data_t *src_ic = src + ((mb*IH + ih)*IW + iw)*ICO;
            const data_t *weights_ic_oci = weights + ((oco*KH +kh)*KW + kw)*ICO*OCI;

            for (int ico = 0; ico < ICO; ico++) {
                const __m128i v_src = sizeof(data_t) == sizeof(int32_t)
                    ? _mm_set1_epi32(src_ic[ico]) : _mm_set1_epi16(src_ic[ico]);
                const __m128i v_weights = _mm_castps_si128(_mm_load_ps((const float*)(weights_ic_oci + ico*OCI)));

                const __m128i v_xor = _mm_xor_si128(v_src, v_weights);
                const __m128i v_xnor = _mm_xor_si128(v_xor, v_ones);

                const __m128i v_mask_4 = _mm_set1_epi8(0x0F);
                const __m128i v_mask_8hi = _mm_set1_epi16(0x00FF);
                const __m128i v_mask_16 = _mm_set1_epi32(0x0000FFFF);
                const __m128i v_popcnt_table = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);

                const __m128i v_popcnt_8lo = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(v_xnor, v_mask_4));
                const __m128i v_popcnt_8hi = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(_mm_srli_epi16(v_xnor, 4), v_mask_4));
                const __m128i v_popcnt_8 = _mm_add_epi32(v_popcnt_8hi, v_popcnt_8lo);
                const __m128i v_popcnt_16hi = _mm_and_si128(v_popcnt_8, v_mask_8hi);
                const __m128i v_popcnt_16lo = _mm_and_si128(_mm_srli_epi16(v_popcnt_8, 8), v_mask_8hi);

                if (sizeof(data_t) == sizeof(int32_t)) {
                    const __m128i v_popcnt_16 = _mm_add_epi16(v_popcnt_16lo, v_popcnt_16hi);
                    const __m128i v_popcnt_32hi = _mm_and_si128(v_popcnt_16, v_mask_16);
                    const __m128i v_popcnt_32lo = _mm_and_si128(_mm_srli_epi32(v_popcnt_16, 16), v_mask_16);
                    d_arr = _mm_add_epi32(d_arr, v_popcnt_32lo);
                    d_arr = _mm_add_epi32(d_arr, v_popcnt_32hi);
                } else {
                    d_arr = _mm_add_epi16(d_arr, v_popcnt_16lo);
                    d_arr = _mm_add_epi16(d_arr, v_popcnt_16hi);
                }
            }
        }
        if (sizeof(data_t) == sizeof(int32_t)) {
            dst[((mb*OC + oco*OCI + 0)*OH + oh)*OW + ow] =
                ((_mm_extract_epi16(d_arr, 0)+_mm_extract_epi16(d_arr, 1))*2 - op_c[oh*OW + ow]) * alpha[oco*OCI + 0] * k[(mb*OH + oh)*OW + ow];
            dst[((mb*OC + oco*OCI + 1)*OH + oh)*OW + ow] =
                ((_mm_extract_epi16(d_arr, 2)+_mm_extract_epi16(d_arr, 3))*2 - op_c[oh*OW + ow]) * alpha[oco*OCI + 1] * k[(mb*OH + oh)*OW + ow];
            dst[((mb*OC + oco*OCI + 2)*OH + oh)*OW + ow] =
                ((_mm_extract_epi16(d_arr, 4)+_mm_extract_epi16(d_arr, 5))*2 - op_c[oh*OW + ow]) * alpha[oco*OCI + 2] * k[(mb*OH + oh)*OW + ow];
            dst[((mb*OC + oco*OCI + 3)*OH + oh)*OW + ow] =
                ((_mm_extract_epi16(d_arr, 6)+_mm_extract_epi16(d_arr, 7))*2 - op_c[oh*OW + ow]) * alpha[oco*OCI + 3] * k[(mb*OH + oh)*OW + ow];
        } else {
            dst[((mb*OC + oco*OCI + 0)*OH + oh)*OW + ow] =
                (_mm_extract_epi16(d_arr, 0)*2 - op_c[oh*OW + ow]) * alpha[oco*OCI + 0] * k[(mb*OH + oh)*OW + ow];
            dst[((mb*OC + oco*OCI + 1)*OH + oh)*OW + ow] =
                (_mm_extract_epi16(d_arr, 1)*2 - op_c[oh*OW + ow]) * alpha[oco*OCI + 1] * k[(mb*OH + oh)*OW + ow];
            dst[((mb*OC + oco*OCI + 2)*OH + oh)*OW + ow] =
                (_mm_extract_epi16(d_arr, 2)*2 - op_c[oh*OW + ow]) * alpha[oco*OCI + 2] * k[(mb*OH + oh)*OW + ow];
            dst[((mb*OC + oco*OCI + 3)*OH + oh)*OW + ow] =
                (_mm_extract_epi16(d_arr, 3)*2 - op_c[oh*OW + ow]) * alpha[oco*OCI + 3] * k[(mb*OH + oh)*OW + ow];
            dst[((mb*OC + oco*OCI + 4)*OH + oh)*OW + ow] =
                (_mm_extract_epi16(d_arr, 4)*2 - op_c[oh*OW + ow]) * alpha[oco*OCI + 4] * k[(mb*OH + oh)*OW + ow];
            dst[((mb*OC + oco*OCI + 5)*OH + oh)*OW + ow] =
                (_mm_extract_epi16(d_arr, 5)*2 - op_c[oh*OW + ow]) * alpha[oco*OCI + 5] * k[(mb*OH + oh)*OW + ow];
            dst[((mb*OC + oco*OCI + 6)*OH + oh)*OW + ow] =
                (_mm_extract_epi16(d_arr, 6)*2 - op_c[oh*OW + ow]) * alpha[oco*OCI + 6] * k[(mb*OH + oh)*OW + ow];
            dst[((mb*OC + oco*OCI + 7)*OH + oh)*OW + ow] =
                (_mm_extract_epi16(d_arr, 7)*2 - op_c[oh*OW + ow]) * alpha[oco*OCI + 7] * k[(mb*OH + oh)*OW + ow];
        }
    }

    return xnor_nn_success;
}

using isa = xnor_nn::isa::isa_traits<xnor_nn::isa::isa_sse3>;
#include "bcast_instantiator.hxx"

} // namespace implementation
} // namespace xnor_nn
