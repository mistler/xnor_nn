#include <cstdlib>

#include <algorithm>
#include <iostream>

#include <immintrin.h>

#include "utils.hpp"

typedef int data_t;

template<
    int SIMD_W,
    int INT_SZ,
    int MB,
    int OC, int OH, int OW,
    int IC, int IH, int IW,
    int KH, int KW,
    int SH, int SW,
    int PH, int PW>
void exec(const data_t *__restrict__ src,
        const data_t *__restrict__ weights,
        data_t *__restrict__ dst){
    data_t ones[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    __m128i v_ones = _mm_load_si128((__m128i*)ones);
#   pragma omp parallel for collapse(2) schedule(static)
    for(int mb = 0; mb < MB; ++mb)
    for(int oc = 0; oc < OC/SIMD_W; ++oc)
    for(int ic = 0; ic < IC/SIMD_W/INT_SZ; ++ic)
    for(int oh = 0; oh < OH; ++oh)
    for(int ow = 0; ow < OW; ++ow){
        size_t dst_idx = ((mb*OC/SIMD_W + oc)*OH + oh)*OW + ow;
        __m128i *v_ptr_dst = (__m128i*)(dst + dst_idx);
        __m128i v_dst = _mm_xor_si128(v_ones, v_ones);

        for (int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw){
            if (oh*SH + kh < std::max(0, PH)) continue;
            if (ow*SW + kw < std::max(0, PW)) continue;

            if (oh*SH + kh >= IH + PH) continue;
            if (ow*SW + kw >= IW + PW) continue;

            const int ih = oh * SH - PH + kh;
            const int iw = ow * SW - PW + kw;

            size_t src_idx =
                (((mb*IC/SIMD_W/INT_SZ + ic)*IH + ih)*IW + iw)*SIMD_W;
            __m128i *v_ptr_src = (__m128i*)(src + src_idx);
            __m128i v_src = _mm_load_si128(v_ptr_src);
            __m128i v_dst_addition = _mm_xor_si128(v_ones, v_ones);
            for(int b_oc = 0; b_oc < SIMD_W; b_oc++){
                size_t weights_idx =
                    ((((oc*IC/SIMD_W/INT_SZ + ic)*KH + kh)*KW + kw)*SIMD_W
                    + b_oc)*SIMD_W;
                __m128i *v_ptr_weights = (__m128i*)(weights + weights_idx);
                __m128i v_weights = _mm_load_si128(v_ptr_weights);
                __m128i v_xor = _mm_xor_si128(v_src, v_weights);
                __m128i v_xnor = _mm_xor_si128(v_xor, v_ones);
                for(int b_ic = 0; b_ic < SIMD_W; b_ic++){
                    data_t element = _mm_extract_epi32(v_xnor, b_ic);
                    data_t popcount = __builtin_popcount(element);
                    v_dst_addition = _mm_insert_epi32(v_dst, popcount, b_ic);
                }
            }
            v_dst = _mm_add_epi32(v_dst, v_dst_addition);
        }
        _mm_store_si128(v_ptr_dst, v_dst);
    }
}

// AlexNet conv4
int main(){
    const size_t MB = 256,
          OC = 384, OH = 13, OW = 13,
          IC = 384, IH = 13, IW = 13,
          KH = 3, KW = 3,
          SH = 1, SW = 1,
          PH = 1, PW = 1;
    const int SIMD_W = 4;  // SSE4.2
    const int INT_SZ = 32;
    data_t *src = (data_t *)aligned_alloc(64, MB*IC*IH*IW * sizeof(data_t));
    data_t *weights = (data_t *)aligned_alloc(64, OC*IC*KH*KW * sizeof(data_t));
    data_t *dst = (data_t *)aligned_alloc(64, MB*OC*OH*OW * sizeof(data_t));

    const int N_RUNS = 8;

    xnor_nn::utils::Timer timer;
    timer.start();
    for(int i = 0; i < N_RUNS; i++){
        exec<SIMD_W, INT_SZ, MB, OC, OH, OW, IC, IH, IW,
            KH, KW, SH, SW, PH, PW>(src, weights, dst);
    }
    timer.stop();

    std::cout << timer.micros() / N_RUNS << std::endl;
    return 0;
}
