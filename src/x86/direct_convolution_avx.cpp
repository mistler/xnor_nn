#include "direct_convolution.hpp"

#include <immintrin.h>

#include "utils.hpp"
#include "logger.hpp"

#include "direct_template_parameters.hpp"

namespace xnor_nn {
namespace implementation {

#ifdef TEMPLATED
template<int OC, int IC, int IH, int IW, int KH, int KW, int SH, int SW,
    int PH, int PW>
xnor_nn_status_t DirectConvolution::exec_avx_template(
#else
xnor_nn_status_t DirectConvolution::exec_avx_simple(
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
            "Algorithm:", "direct", "ISA:", "AVX"
#ifdef TEMPLATED
            , "Templated"
#endif
            );

    /*
    unsigned long long int t = xnor_nn::utils::Timer::rdtsc();
    */

    const int ones[8] = {-1,-1,-1,-1,-1,-1,-1,-1};

    // TODO: potentially loops can be reordered
#   pragma omp parallel for collapse(4) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oc = 0; oc < OC; oc++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        int dst_idx = ((mb*OC + oc)*OH + oh)*OW + ow;
        float *d = dst + dst_idx;
        *d = 0.f;
        long long int dst_i = 0;
        __m256 v_ones = _mm256_loadu_ps((float*)ones);
        /*
        asm volatile (
            "vmovups (%0), %%ymm2\n\t"
            :
            : "r" ((float*)ones)
            : "ymm2"
        );
        */
#ifdef TEMPLATED
        constexpr int kernel_size = KW * KH * VECTORS_IN_ABIC;
        __m256 kernel[kernel_size];
        if (kernel_size < 10) {
            for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++)
            for (int vabic = 0; vabic < VECTORS_IN_ABIC; vabic++)
                kernel[(vabic*KH + kh)*KW + kw] = _mm256_load_ps(
                        (const float*) weights
                        + ((kh*KW + kw)*OC + oc)*ABIC/ELEM_SIZE
                        + vabic*VLEN/ELEM_SIZE);
        }
#endif
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
                __m256 v_src = _mm256_load_ps(
                        (float*)src_ic + vabic*VLEN/ELEM_SIZE);
#ifdef TEMPLATED
                __m256 v_weights;
                __m256 v_xor;
                if (kernel_size < 10) {
                    v_xor = _mm256_xor_ps(v_src,
                            kernel[(vabic*KH + kh)*KW + kw]);
                } else {
                    v_weights = _mm256_load_ps(
                        (float*)weights_ic + vabic*VLEN/ELEM_SIZE);
                    v_xor = _mm256_xor_ps(v_src, v_weights);
                }
#else
                __m256 v_weights = _mm256_load_ps(
                    (float*)weights_ic + vabic*VLEN/ELEM_SIZE);
                __m256 v_xor = _mm256_xor_ps(v_src, v_weights);
#endif
                __m256i v_xnor =
                    _mm256_castps_si256(_mm256_xor_ps(v_xor, v_ones));

                dst_i += __builtin_popcountll(_mm256_extract_epi64(v_xnor, 0))
                    *2-64;
                dst_i += __builtin_popcountll(_mm256_extract_epi64(v_xnor, 1))
                    *2-64;
                dst_i += __builtin_popcountll(_mm256_extract_epi64(v_xnor, 2))
                    *2-64;
                dst_i += __builtin_popcountll(_mm256_extract_epi64(v_xnor, 3))
                    *2-64;
                /*
                float *src_ptr = (float*)src_ic + abic*VLEN/ELEM_SIZE;
                float *weights_ptr = (float*)weights_ic + abic*VLEN/ELEM_SIZE;
                asm volatile (
                    "vmovaps (%1), %%ymm0\n\t"
                    "vxorps (%2), %%ymm0, %%ymm0\n\t"
                    "vxorps %%ymm2, %%ymm0, %%ymm0\n\t"
                    "vextractf128 $0x1,%%ymm0,%%xmm1\n\t"
                    "vpextrq $0x1,%%xmm1,%%rcx\n\t"
                    "vpextrq $0x1,%%xmm0,%%rax\n\t"
                    "vmovq %%xmm1,%%rdx\n\t"
                    "vmovq %%xmm0, %%rsi\n\t"
                    "popcnt %%rsi,%%rsi\n\t"
                    "popcnt %%rax,%%rax\n\t"
                    "popcnt %%rcx,%%rcx\n\t"
                    "popcnt %%rdx,%%rdx\n\t"
                    "add %%rax, %0\n\t"
                    "add %%rcx, %0\n\t"
                    "add %%rdx, %0\n\t"
                    "add %%rsi, %0\n\t"
                    : "+r" (dst_i)
                    : "r" (src_ptr), "r" (weights_ptr)
                    : "rax", "rdx", "rcx", "rsi",
                    "ymm0", "ymm1", "ymm2"
                );
                */
            }
        }
        *d = (float)dst_i * *alpha * k[oh*OW + ow];
    }

    /*
    t = xnor_nn::utils::Timer::rdtsc() - t;

    xnor_nn::utils::Logger::info("Time: ", t);
    */

    return xnor_nn_success;
}

#ifdef TEMPLATED

DIRECT_TEMPLATE_INSTANTIATE(avx);

#endif

} // namespace implementation
} // namespace xnor_nn
