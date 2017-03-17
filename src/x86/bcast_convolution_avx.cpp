#include "bcast_convolution.hpp"

#include <immintrin.h>

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
        const float alpha,
        const int *src, const int *weights, const float *k, float *dst) {
    (void)VLEN;

    constexpr int MAX_ICO_UNROLL = 12;
    constexpr int MAX_OW_UNROLL = 4;

    int unroll_ow = 1;
    for (int uow = 0; uow < MAX_OW_UNROLL; uow++)
        if (OW % uow == 0) unroll_ow = uow;

#   pragma omp parallel for collapse(3) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oco = 0; oco < OCO; oco++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow += unroll_ow) {
        int operations_counter[MAX_OW_UNROLL] = {0};

        __m128i d_arr[MAX_OW_UNROLL][2];
        for (int i = 0; i < MAX_OW_UNROLL; i++)
        for (int k = 0; k < 2; k++) d_arr[i][k] = _mm_set1_epi8(0);

        __m256 v_ones = _mm256_castsi256_ps(_mm256_set1_epi32(-1));

        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            const int *weights_ic_oci = weights + ((oco*KH +kh)*KW + kw)*ICO*OCI;

            __m256 v_weights[MAX_ICO_UNROLL];
            for (int ico = 0; ico < ICO; ico++)
                v_weights[ico] = _mm256_load_ps((float*)weights_ic_oci + ico*OCI);

            for (int uow = 0; uow < unroll_ow; uow++) {
                const int ih = oh*SH - PH + kh;
                const int iw = (ow + uow)*SW - PW + kw;

                if (PH != 0 || PW != 0) { // May be incorrect for non symmetr p
                    if (ih < 0 || iw < 0) continue;
                    if (ih >= IH || iw >= IW) continue;
                }
                operations_counter[uow] += IC;

                const int *src_ic = src + ((mb*IH + ih)*IW + iw)*ICO;

                for (int ico = 0; ico < ICO; ico++) {
                    __m256 v_src = _mm256_broadcast_ss((float*)src_ic + ico);

                    const __m256 v_xor = _mm256_xor_ps(v_src, v_weights[ico]);
                    const __m256i v_xnor = _mm256_castps_si256(_mm256_xor_ps(v_xor, v_ones));

                    /*
                    int tmp[8] __attribute__ ((aligned(64)));
                    tmp[0] = __builtin_popcount(_mm256_extract_epi32(v_xnor, 0));
                    tmp[1] = __builtin_popcount(_mm256_extract_epi32(v_xnor, 1));
                    tmp[2] = __builtin_popcount(_mm256_extract_epi32(v_xnor, 2));
                    tmp[3] = __builtin_popcount(_mm256_extract_epi32(v_xnor, 3));
                    tmp[4] = __builtin_popcount(_mm256_extract_epi32(v_xnor, 4));
                    tmp[5] = __builtin_popcount(_mm256_extract_epi32(v_xnor, 5));
                    tmp[6] = __builtin_popcount(_mm256_extract_epi32(v_xnor, 6));
                    tmp[7] = __builtin_popcount(_mm256_extract_epi32(v_xnor, 7));
                    d_arr[uow][0] = _mm_add_epi32(d_arr[uow][0], _mm_load_si128((const __m128i*)(tmp + 0)));
                    d_arr[uow][1] = _mm_add_epi32(d_arr[uow][1], _mm_load_si128((const __m128i*)(tmp + 4)));
                    */

                    // 1x6 + 2x4
                    /*
                    const __m128i v_mask_4 = _mm_set1_epi8(0x0F);
                    const __m128i v_mask_8 = _mm_set1_epi16(0x00FF);
                    const __m128i v_mask_16 = _mm_set1_epi32(0x0000FFFF);
                    const __m128i v_popcnt_table = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);

                    const __m128i lo_v_xnor = _mm256_extractf128_si256(v_xnor, 0);
                    const __m128i lo_v_popcnt_8hi = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(lo_v_xnor, v_mask_4));
                    const __m128i lo_v_popcnt_8lo = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(_mm_srli_epi16(lo_v_xnor, 4), v_mask_4));
                    const __m128i lo_v_popcnt_8 = _mm_add_epi8(lo_v_popcnt_8lo, lo_v_popcnt_8hi);
                    const __m128i lo_v_popcnt_16hi = _mm_and_si128(lo_v_popcnt_8, v_mask_8);
                    const __m128i lo_v_popcnt_16lo = _mm_and_si128(_mm_srli_epi16(lo_v_popcnt_8, 8), v_mask_8);
                    const __m128i lo_v_popcnt_16 = _mm_add_epi16(lo_v_popcnt_16lo, lo_v_popcnt_16hi);
                    const __m128i lo_v_popcnt_32hi = _mm_and_si128(lo_v_popcnt_16, v_mask_16);
                    const __m128i lo_v_popcnt_32lo = _mm_and_si128(_mm_srli_epi32(lo_v_popcnt_16, 16), v_mask_16);
                    d_arr[uow][0] = _mm_add_epi32(d_arr[uow][0], lo_v_popcnt_32lo);
                    d_arr[uow][0] = _mm_add_epi32(d_arr[uow][0], lo_v_popcnt_32hi);

                    const __m128i hi_v_xnor = _mm256_extractf128_si256(v_xnor, 1);
                    const __m128i hi_v_popcnt_8lo = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(hi_v_xnor, v_mask_4));
                    const __m128i hi_v_popcnt_8hi = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(_mm_srli_epi16(hi_v_xnor, 4), v_mask_4));
                    const __m128i hi_v_popcnt_8 = _mm_add_epi32(hi_v_popcnt_8hi, hi_v_popcnt_8lo);
                    const __m128i hi_v_popcnt_16hi = _mm_and_si128(hi_v_popcnt_8, v_mask_8);
                    const __m128i hi_v_popcnt_16lo = _mm_and_si128(_mm_srli_epi16(hi_v_popcnt_8, 8), v_mask_8);
                    const __m128i hi_v_popcnt_16 = _mm_add_epi16(hi_v_popcnt_16lo, hi_v_popcnt_16hi);
                    const __m128i hi_v_popcnt_32hi = _mm_and_si128(hi_v_popcnt_16, v_mask_16);
                    const __m128i hi_v_popcnt_32lo = _mm_and_si128(_mm_srli_epi32(hi_v_popcnt_16, 16), v_mask_16);
                    d_arr[uow][1] = _mm_add_epi32(d_arr[uow][1], hi_v_popcnt_32lo);
                    d_arr[uow][1] = _mm_add_epi32(d_arr[uow][1], hi_v_popcnt_32hi);
                    */

                    // 2x6 + 1x4
                    /*
                    //const __m128i v_mask_4 = _mm_set1_epi8(0x0F);
                    const __m128i v_mask_8 = _mm_set1_epi16(0x000F);
                    const __m128i v_mask_16 = _mm_set1_epi32(0x0000FFFF);
                    const __m128i v_popcnt_table = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);

                    const __m128i lo_v_xnor = _mm256_extractf128_si256(v_xnor, 0);
                    const __m128i lo_v_popcnt_8hi = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(lo_v_xnor, v_mask_8));
                    const __m128i lo_v_popcnt_8lo = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(_mm_srli_epi16(lo_v_xnor, 4), v_mask_8));
                    const __m128i lo_v_popcnt_8 = _mm_add_epi16(lo_v_popcnt_8lo, lo_v_popcnt_8hi);
                    const __m128i lo_v_popcnt_16hi = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(_mm_srli_epi16(lo_v_xnor, 8), v_mask_8));
                    const __m128i lo_v_popcnt_16lo = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(_mm_srli_epi16(lo_v_xnor, 12), v_mask_8));
                    const __m128i lo_v_popcnt_16 = _mm_add_epi16(_mm_add_epi16(lo_v_popcnt_16lo, lo_v_popcnt_16hi), lo_v_popcnt_8);
                    const __m128i lo_v_popcnt_32hi = _mm_and_si128(lo_v_popcnt_16, v_mask_16);
                    const __m128i lo_v_popcnt_32lo = _mm_and_si128(_mm_srli_epi32(lo_v_popcnt_16, 16), v_mask_16);
                    d_arr[uow][0] = _mm_add_epi32(d_arr[uow][0], lo_v_popcnt_32lo);
                    d_arr[uow][0] = _mm_add_epi32(d_arr[uow][0], lo_v_popcnt_32hi);

                    const __m128i hi_v_xnor = _mm256_extractf128_si256(v_xnor, 1);
                    const __m128i hi_v_popcnt_8lo = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(hi_v_xnor, v_mask_8));
                    const __m128i hi_v_popcnt_8hi = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(_mm_srli_epi16(hi_v_xnor, 4), v_mask_8));
                    const __m128i hi_v_popcnt_8 = _mm_add_epi16(hi_v_popcnt_8hi, hi_v_popcnt_8lo);
                    const __m128i hi_v_popcnt_16hi = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(_mm_srli_epi16(hi_v_xnor, 8), v_mask_8));
                    const __m128i hi_v_popcnt_16lo = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(_mm_srli_epi16(hi_v_xnor, 12), v_mask_8));
                    const __m128i hi_v_popcnt_16 = _mm_add_epi16(_mm_add_epi16(hi_v_popcnt_16lo, hi_v_popcnt_16hi), hi_v_popcnt_8);
                    const __m128i hi_v_popcnt_32hi = _mm_and_si128(hi_v_popcnt_16, v_mask_16);
                    const __m128i hi_v_popcnt_32lo = _mm_and_si128(_mm_srli_epi32(hi_v_popcnt_16, 16), v_mask_16);
                    d_arr[uow][1] = _mm_add_epi32(d_arr[uow][1], hi_v_popcnt_32lo);
                    d_arr[uow][1] = _mm_add_epi32(d_arr[uow][1], hi_v_popcnt_32hi);
                    */

                    // mix
                    const __m128i v_mask_4 = _mm_set1_epi8(0x0F);
                    const __m128i v_mask_8 = _mm_set1_epi16(0x000F);
                    const __m128i v_mask_8hi = _mm_set1_epi16(0x00FF);
                    const __m128i v_mask_16 = _mm_set1_epi32(0x0000FFFF);
                    const __m128i v_popcnt_table = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);

                    const __m128i lo_v_xnor = _mm256_extractf128_si256(v_xnor, 0);
                    const __m128i lo_v_popcnt_8hi = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(lo_v_xnor, v_mask_8));
                    const __m128i lo_v_popcnt_8lo = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(_mm_srli_epi16(lo_v_xnor, 4), v_mask_8));
                    const __m128i lo_v_popcnt_8 = _mm_add_epi16(lo_v_popcnt_8lo, lo_v_popcnt_8hi);
                    const __m128i lo_v_popcnt_16hi = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(_mm_srli_epi16(lo_v_xnor, 8), v_mask_8));
                    const __m128i lo_v_popcnt_16lo = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(_mm_srli_epi16(lo_v_xnor, 12), v_mask_8));
                    const __m128i lo_v_popcnt_16 = _mm_add_epi16(_mm_add_epi16(lo_v_popcnt_16lo, lo_v_popcnt_16hi), lo_v_popcnt_8);
                    const __m128i lo_v_popcnt_32hi = _mm_and_si128(lo_v_popcnt_16, v_mask_16);
                    const __m128i lo_v_popcnt_32lo = _mm_and_si128(_mm_srli_epi32(lo_v_popcnt_16, 16), v_mask_16);
                    d_arr[uow][0] = _mm_add_epi32(d_arr[uow][0], lo_v_popcnt_32lo);
                    d_arr[uow][0] = _mm_add_epi32(d_arr[uow][0], lo_v_popcnt_32hi);

                    const __m128i hi_v_xnor = _mm256_extractf128_si256(v_xnor, 1);
                    const __m128i hi_v_popcnt_8lo = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(hi_v_xnor, v_mask_4));
                    const __m128i hi_v_popcnt_8hi = _mm_shuffle_epi8(v_popcnt_table, _mm_and_si128(_mm_srli_epi16(hi_v_xnor, 4), v_mask_4));
                    const __m128i hi_v_popcnt_8 = _mm_add_epi32(hi_v_popcnt_8hi, hi_v_popcnt_8lo);
                    const __m128i hi_v_popcnt_16hi = _mm_and_si128(hi_v_popcnt_8, v_mask_8hi);
                    const __m128i hi_v_popcnt_16lo = _mm_and_si128(_mm_srli_epi16(hi_v_popcnt_8, 8), v_mask_8hi);
                    const __m128i hi_v_popcnt_16 = _mm_add_epi16(hi_v_popcnt_16lo, hi_v_popcnt_16hi);
                    const __m128i hi_v_popcnt_32hi = _mm_and_si128(hi_v_popcnt_16, v_mask_16);
                    const __m128i hi_v_popcnt_32lo = _mm_and_si128(_mm_srli_epi32(hi_v_popcnt_16, 16), v_mask_16);
                    d_arr[uow][1] = _mm_add_epi32(d_arr[uow][1], hi_v_popcnt_32lo);
                    d_arr[uow][1] = _mm_add_epi32(d_arr[uow][1], hi_v_popcnt_32hi);
                }
            }
        }

        for (int uow = 0; uow < unroll_ow; uow++) {
            dst[((mb*OC + oco*OCI + 0)*OH + oh)*OW + (ow+uow)] =
                (_mm_extract_epi32(d_arr[uow][0], 0)*2 - operations_counter[uow]) * alpha * k[oh*OW + (ow+uow)];
            dst[((mb*OC + oco*OCI + 1)*OH + oh)*OW + (ow+uow)] =
                (_mm_extract_epi32(d_arr[uow][0], 1)*2 - operations_counter[uow]) * alpha * k[oh*OW + (ow+uow)];
            dst[((mb*OC + oco*OCI + 2)*OH + oh)*OW + (ow+uow)] =
                (_mm_extract_epi32(d_arr[uow][0], 2)*2 - operations_counter[uow]) * alpha * k[oh*OW + (ow+uow)];
            dst[((mb*OC + oco*OCI + 3)*OH + oh)*OW + (ow+uow)] =
                (_mm_extract_epi32(d_arr[uow][0], 3)*2 - operations_counter[uow]) * alpha * k[oh*OW + (ow+uow)];
            dst[((mb*OC + oco*OCI + 4)*OH + oh)*OW + (ow+uow)] =
                (_mm_extract_epi32(d_arr[uow][1], 0)*2 - operations_counter[uow]) * alpha * k[oh*OW + (ow+uow)];
            dst[((mb*OC + oco*OCI + 5)*OH + oh)*OW + (ow+uow)] =
                (_mm_extract_epi32(d_arr[uow][1], 1)*2 - operations_counter[uow]) * alpha * k[oh*OW + (ow+uow)];
            dst[((mb*OC + oco*OCI + 6)*OH + oh)*OW + (ow+uow)] =
                (_mm_extract_epi32(d_arr[uow][1], 2)*2 - operations_counter[uow]) * alpha * k[oh*OW + (ow+uow)];
            dst[((mb*OC + oco*OCI + 7)*OH + oh)*OW + (ow+uow)] =
                (_mm_extract_epi32(d_arr[uow][1], 3)*2 - operations_counter[uow]) * alpha * k[oh*OW + (ow+uow)];
        }
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
        || c == nullptr
    ) return xnor_nn_error_invalid_input;
    const int *src = (int*)res[xnor_nn_resource_bin_src];
    const int *weights = (int*)res[xnor_nn_resource_bin_weights];
    const float *alpha = (const float*)&res[xnor_nn_resource_alpha];
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
            "Algorithm:", "bcast", "ISA:", "AVX", "Templated");

    execute(MB, OC, OH, OW, IC, IH, IW, KH, KW, SH, SW, PH, PW,
        ICO, OCO, OCI, VLEN, *alpha, src, weights, k, dst);

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
        || c == nullptr
    ) return xnor_nn_error_invalid_input;
    const int *src = (int*)res[xnor_nn_resource_bin_src];
    const int *weights = (int*)res[xnor_nn_resource_bin_weights];
    const float *alpha = (const float*)&res[xnor_nn_resource_alpha];
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
            "Algorithm:", "bcast", "ISA:", "AVX");

    execute(MB, OC, OH, OW, IC, IH, IW, KH, KW, SH, SW, PH, PW,
        ICO, OCO, OCI, VLEN, *alpha, src, weights, k, dst);

    return xnor_nn_success;
}

using isa = xnor_nn::isa::isa_traits<xnor_nn::isa::isa_avx>;
using algorithm = BcastConvolution;
#include "instantiator.hxx"

} // namespace implementation
} // namespace xnor_nn
