#include "bcast_convolution.hpp"

#include <immintrin.h>

#include "utils.hpp"
#include "logger.hpp"

#include "bcast_template_parameters.hpp"

namespace xnor_nn {
namespace implementation {

#ifdef TEMPLATED
template<int OC, int IC, int IH, int IW, int KH, int KW, int SH, int SW,
    int PH, int PW>
xnor_nn_status_t BcastConvolution::exec_avx_template(
#else
xnor_nn_status_t BcastConvolution::exec_avx_simple(
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

    BcastConvolution *state = reinterpret_cast<BcastConvolution*>(
            getState(c, xnor_nn_operation_convolution_forward));

#ifdef TEMPLATED
    constexpr int OH = getOH(IH, KH, SH, PH);
    constexpr int OW = getOW(IW, KW, SW, PW);
    constexpr int OCO = state->constexpr_getOCO(OC);
    constexpr int ICO = state->constexpr_getICO(IC);
    constexpr int OCI = state->constexpr_getOCI();

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

    const int OCO = state->OCO;
    const int ICO = state->ICO;
    const int OCI = state->getOCI();
#endif

    LOG_INFO("convolution:\t", "execute:",
            "[", MB, "][", IC, "][", IH, "][", IW, "]",
            "x",
            "[", OC, "][", IC, "][", KH, "][", KW, "]",
            "=",
            "[", MB, "][", OC, "][", OH, "][", OW, "]",
            "stride: [", SH, "][", SW, "]",
            "pad: [", PH, "][", PW, "]",
            "Algorithm:", "bcast", "ISA:", "AVX"
#ifdef TEMPLATED
            , "Templated"
#endif
            );

    const int ones[8] = {-1,-1,-1,-1,-1,-1,-1,-1};

    // TODO: potentially loops can be reordered
    // TODO: check collapse value for performance
#   pragma omp parallel for collapse(4) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oco = 0; oco < OCO; oco++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        unsigned int d_arr[OCI] = { 0u };
        __m256 v_ones = _mm256_loadu_ps((float*)ones);

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
                __m256 v_src = _mm256_broadcast_ss((float*)src_ic + ico);
                __m256 v_weights = _mm256_load_ps(
                        (float*)weights_ic_oci + ico*OCI);

                __m256 v_xor = _mm256_xor_ps(v_src, v_weights);
                __m256i v_xnor =
                    _mm256_castps_si256(_mm256_xor_ps(v_xor, v_ones));

                d_arr[0] += __builtin_popcount(_mm256_extract_epi32(v_xnor, 0));
                d_arr[1] += __builtin_popcount(_mm256_extract_epi32(v_xnor, 1));
                d_arr[2] += __builtin_popcount(_mm256_extract_epi32(v_xnor, 2));
                d_arr[3] += __builtin_popcount(_mm256_extract_epi32(v_xnor, 3));
                d_arr[4] += __builtin_popcount(_mm256_extract_epi32(v_xnor, 4));
                d_arr[5] += __builtin_popcount(_mm256_extract_epi32(v_xnor, 5));
                d_arr[6] += __builtin_popcount(_mm256_extract_epi32(v_xnor, 6));
                d_arr[7] += __builtin_popcount(_mm256_extract_epi32(v_xnor, 7));
            }
        }
        for (int i = 0; i < OCI; i++)
            dst[((mb*OC + oco*OCI + i)*OH + oh)*OW + ow] =
                d_arr[i] * *alpha * k[oh*OW + ow];
    }

    return xnor_nn_success;
}

#ifdef TEMPLATED

BCAST_TEMPLATE_INSTANTIATE(avx);

#undef INSTANTIATE

#endif

} // namespace implementation
} // namespace xnor_nn
