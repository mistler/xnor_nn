#include "bcast_convolution.hpp"

#include <immintrin.h>

#include "utils.hpp"
#include "logger.hpp"

#include "isa_traits.hpp"

namespace xnor_nn {
namespace implementation {

constexpr int get_unroll_factor(const int total, const int max_unroll) {
    int unroll = 0;
    for (int u = 1; u < max_unroll; u++)
        if (total % u == 0) unroll = u;
    return unroll;
}

template<typename isa_traits, int OC, int IC, int IH, int IW, int KH, int KW,
    int SH, int SW, int PH, int PW>
xnor_nn_status_t BcastConvolution::exec(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    constexpr int VLEN = isa_traits::vlen;
    (void)VLEN;
    if (
        res[xnor_nn_resource_bin_src] == nullptr
        || res[xnor_nn_resource_bin_weights] == nullptr
        || res[xnor_nn_resource_user_dst] == nullptr
        || res[xnor_nn_resource_k] == nullptr
        || c == nullptr
    ) return xnor_nn_error_invalid_input;
    const int *src = (int*)res[xnor_nn_resource_bin_src];
    const int *weights = (int*)res[xnor_nn_resource_bin_weights];
    const float alpha = *((const float*)&res[xnor_nn_resource_alpha]);
    const float *k = (const float*)res[xnor_nn_resource_k];
    float *dst = (float*)res[xnor_nn_resource_user_dst];

    const int MB = c->mb;
    constexpr int OH = getOH(IH, KH, SH, PH);
    constexpr int OW = getOW(IW, KW, SW, PW);
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
            "Algorithm:", "bcast", "ISA:", "ATOM", "Templated");

    constexpr int MAX_ICO_UNROLL = 12;
    constexpr int MAX_OW_UNROLL = 4;

    constexpr int unroll_ow = get_unroll_factor(OW, MAX_OW_UNROLL);
    //constexpr int unroll_ico = get_unroll_factor(ICO, MAX_ICO_UNROLL);

#   pragma omp parallel for collapse(3) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oco = 0; oco < OCO; oco++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow += unroll_ow) {
        int operations_counter[MAX_OW_UNROLL] __attribute__ ((aligned(64))) = {0};

        __m128i d_arr[MAX_OW_UNROLL];
        for (int i = 0; i < MAX_OW_UNROLL; i++) d_arr[i] = _mm_set1_epi8(0);
        const __m128i v_ones = _mm_set1_epi32(-1);

        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            const int *weights_ic_oci = weights + ((oco*KH +kh)*KW + kw)*ICO*OCI;

            __m128i v_weights[MAX_ICO_UNROLL];
            for (int ico = 0; ico < ICO; ico++)
                v_weights[ico] = _mm_castps_si128(_mm_load_ps((float*)weights_ic_oci + ico*OCI));

            for (int uow = 0; uow < unroll_ow; uow++) {
                const int ih = oh*SH - PH + kh;
                const int iw = (ow + uow)*SW - PW + kw;

                if (PH != 0 || PW != 0) { // May be incorrect for assymet pad
                    if (ih < 0 || iw < 0) continue;
                    if (ih >= IH || iw >= IW) continue;
                }
                operations_counter[uow] += IC;

                const int *src_ic = src + ((mb*IH + ih)*IW + iw)*ICO;

                for (int ico = 0; ico < ICO; ico++) {
                    __m128 s_src = _mm_load_ss((float*)src_ic + ico);
                    __m128i v_src = _mm_shuffle_epi32(_mm_castps_si128(s_src), 0);

                    const __m128i v_xor = _mm_xor_si128(v_src, v_weights[ico]);
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
                    const __m128i v_popcnt_32hi = _mm_and_si128(v_popcnt_16, v_mask_16);
                    const __m128i v_popcnt_32lo = _mm_and_si128(_mm_srli_epi32(v_popcnt_16, 16), v_mask_16);
                    d_arr[uow] = _mm_add_epi32(d_arr[uow], v_popcnt_32lo);
                    d_arr[uow] = _mm_add_epi32(d_arr[uow], v_popcnt_32hi);
                }
            }
        }

        for (int uow = 0; uow < unroll_ow; uow++) {
            dst[((mb*OC + oco*OCI + 0)*OH + oh)*OW + (ow+uow)] =
                ((_mm_extract_epi16(d_arr[uow], 0)+_mm_extract_epi16(d_arr[uow], 1))*2 - operations_counter[uow]) * alpha * k[oh*OW + (ow+uow)];
            dst[((mb*OC + oco*OCI + 1)*OH + oh)*OW + (ow+uow)] =
                ((_mm_extract_epi16(d_arr[uow], 2)+_mm_extract_epi16(d_arr[uow], 3))*2 - operations_counter[uow]) * alpha * k[oh*OW + (ow+uow)];
            dst[((mb*OC + oco*OCI + 2)*OH + oh)*OW + (ow+uow)] =
                ((_mm_extract_epi16(d_arr[uow], 4)+_mm_extract_epi16(d_arr[uow], 5))*2 - operations_counter[uow]) * alpha * k[oh*OW + (ow+uow)];
            dst[((mb*OC + oco*OCI + 3)*OH + oh)*OW + (ow+uow)] =
                ((_mm_extract_epi16(d_arr[uow], 6)+_mm_extract_epi16(d_arr[uow], 7))*2 - operations_counter[uow]) * alpha * k[oh*OW + (ow+uow)];
        }
    }

    return xnor_nn_success;
}

template<typename isa_traits>
xnor_nn_status_t BcastConvolution::exec(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    constexpr int VLEN = isa_traits::vlen;
    (void)VLEN;
    if (
        res[xnor_nn_resource_bin_src] == nullptr
        || res[xnor_nn_resource_bin_weights] == nullptr
        || res[xnor_nn_resource_user_dst] == nullptr
        || res[xnor_nn_resource_k] == nullptr
        || c == nullptr
    ) return xnor_nn_error_invalid_input;
    const int *src = (int*)res[xnor_nn_resource_bin_src];
    const int *weights = (int*)res[xnor_nn_resource_bin_weights];
    const float alpha = *((const float*)&res[xnor_nn_resource_alpha]);
    const float *k = (const float*)res[xnor_nn_resource_k];
    float *dst = (float*)res[xnor_nn_resource_user_dst];

    const int MB = c->mb;

    BcastConvolution *state = reinterpret_cast<BcastConvolution*>(
            getState(c, xnor_nn_operation_convolution_forward));

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
            "Algorithm:", "bcast", "ISA:", "ATOM");

#   pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oco = 0; oco < OCO; oco++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        int operations_counter = 0;
        __m128i d_arr = _mm_set1_epi8(0);
        const __m128i v_ones = _mm_set1_epi32(-1);

        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            const int ih = oh*SH - PH + kh;
            const int iw = ow*SW - PW + kw;

            if (ih < 0 || iw < 0) continue;
            if (ih >= IH || iw >= IW) continue;

            const int *src_ic = src + ((mb*IH + ih)*IW + iw)*ICO;
            const int *weights_ic_oci = weights + ((oco*KH +kh)*KW + kw)*ICO*OCI;

            operations_counter += IC;
            for (int ico = 0; ico < ICO; ico++) {
                const __m128 s_src = _mm_load_ss((float*)src_ic + ico);
                const __m128i v_src = _mm_castps_si128(_mm_shuffle_ps(s_src, s_src, 0));
                const __m128i v_weights = _mm_castps_si128(_mm_load_ps((float*)weights_ic_oci + ico*OCI));

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
                const __m128i v_popcnt_16 = _mm_add_epi16(v_popcnt_16lo, v_popcnt_16hi);
                const __m128i v_popcnt_32hi = _mm_and_si128(v_popcnt_16, v_mask_16);
                const __m128i v_popcnt_32lo = _mm_and_si128(_mm_srli_epi32(v_popcnt_16, 16), v_mask_16);
                d_arr = _mm_add_epi32(d_arr, v_popcnt_32lo);
                d_arr = _mm_add_epi32(d_arr, v_popcnt_32hi);
            }
        }
        dst[((mb*OC + oco*OCI + 0)*OH + oh)*OW + ow] =
            ((_mm_extract_epi16(d_arr, 0)+_mm_extract_epi16(d_arr, 1))*2 - operations_counter) * alpha * k[oh*OW + ow];
        dst[((mb*OC + oco*OCI + 1)*OH + oh)*OW + ow] =
            ((_mm_extract_epi16(d_arr, 2)+_mm_extract_epi16(d_arr, 3))*2 - operations_counter) * alpha * k[oh*OW + ow];
        dst[((mb*OC + oco*OCI + 2)*OH + oh)*OW + ow] =
            ((_mm_extract_epi16(d_arr, 4)+_mm_extract_epi16(d_arr, 5))*2 - operations_counter) * alpha * k[oh*OW + ow];
        dst[((mb*OC + oco*OCI + 3)*OH + oh)*OW + ow] =
            ((_mm_extract_epi16(d_arr, 6)+_mm_extract_epi16(d_arr, 7))*2 - operations_counter) * alpha * k[oh*OW + ow];
    }

    return xnor_nn_success;
}

using isa = xnor_nn::isa::isa_traits<xnor_nn::isa::isa_sse3>;
using algorithm = BcastConvolution;
#include "instantiator.hxx"

} // namespace implementation
} // namespace xnor_nn
