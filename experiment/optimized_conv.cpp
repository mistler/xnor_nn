#include <cstdlib>

#include <algorithm>
#include <iostream>

#include "utils.hpp"

typedef int data_t;

template<
    int VEC_LEN_IN_INTS,
    int MB,
    int OC, int OH, int OW,
    int IC, int IH, int IW,
    int KH, int KW,
    int SH, int SW,
    int PH, int PW>
void exec(const data_t *__restrict__ src,
        const data_t *__restrict__ weights,
        data_t *__restrict__ dst){
#   pragma omp parallel for collapse(4) schedule(static)
    for(int mb = 0; mb < MB; ++mb)
    for(int oc = 0; oc < OC/VEC_LEN_IN_INTS; ++oc)
    for(int oh = 0; oh < OH; ++oh)
    for(int ow = 0; ow < OW; ++ow){
        size_t dst_idx = ((mb*OC/VEC_LEN_IN_INTS + oc)*OH + oh)*OW + ow;
        data_t *d = dst + dst_idx;
        for(int b_oc = 0; b_oc < VEC_LEN_IN_INTS; b_oc++)
            *(d + b_oc) = data_t(0);

        for (int ic = 0; ic < IC/VEC_LEN_IN_INTS; ++ic)
        for (int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw){
            if (oh*SH + kh < std::max(0, PH)) continue;
            if (ow*SW + kw < std::max(0, PW)) continue;

            if (oh*SH + kh >= IH + PH) continue;
            if (ow*SW + kw >= IW + PW) continue;

            const int ih = oh * SH - PH + kh;
            const int iw = ow * SW - PW + kw;

            for(int b_oc = 0; b_oc < VEC_LEN_IN_INTS; b_oc++){
                for(int b_ic = 0; b_ic < VEC_LEN_IN_INTS; b_ic++){
                    size_t src_idx =
                        (((mb*IC/VEC_LEN_IN_INTS + ic)*IH + ih)*IW + iw)
                        * VEC_LEN_IN_INTS + b_ic;
                    size_t weights_idx =
                        ((((oc*IC/VEC_LEN_IN_INTS + ic)*KH + kh)*KW + kw)
                        * VEC_LEN_IN_INTS + b_oc)
                        * VEC_LEN_IN_INTS + b_ic;
                    data_t xnor_result = ~(src[src_idx] ^ weights[weights_idx]);
                    //*(d+b_oc) += __builtin_popcountll(xnor_result);
                    *(d+b_oc) += xnor_result;
                }
            }
        }
    }
}

// AlexNet conv4
int main(){
    const size_t MB = 256,
          OC = 384, OH = 13, OW = 13,
          IC = 384/32, IH = 13, IW = 13,
          KH = 3, KW = 3,
          SH = 1, SW = 1,
          PH = 1, PW = 1;
    const int VEC_LEN_IN_INTS = 4;  // SSE4.2
    data_t *src = (data_t *)aligned_alloc(64, MB*IC*IH*IW * sizeof(data_t));
    data_t *weights = (data_t *)aligned_alloc(64, OC*IC*KH*KW * sizeof(data_t));
    data_t *dst = (data_t *)aligned_alloc(64, MB*OC*OH*OW * sizeof(data_t));

    const int N_RUNS = 8;

    xnor_nn::utils::Timer timer;
    timer.start();
    for(int i = 0; i < N_RUNS; i++){
        exec<VEC_LEN_IN_INTS, MB, OC, OH, OW, IC, IH, IW,
            KH, KW, SH, SW, PH, PW>(src, weights, dst);
    }
    timer.stop();

    std::cout << timer.ms() / N_RUNS << std::endl;
    return 0;
}
