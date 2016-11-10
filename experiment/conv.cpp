#include <cstdlib>
#include <algorithm>

typedef int data_t;

void exec(const data_t *src, const data_t *weights, data_t *dst,
        int MB,
        int OC, int OH, int OW,
        int IC, int IH, int IW,
        int KH, int KW,
        int SH, int SW,
        int PH, int PW){
//#   pragma omp parallel for collapse(4) schedule(static)
    for(int mb = 0; mb < MB; ++mb)
    for(int oc = 0; oc < OC; ++oc)
    for(int oh = 0; oh < OH; ++oh)
    for(int ow = 0; ow < OW; ++ow){
        size_t dst_idx = ((mb*OC + oc)*OH + oh)*OW + ow;
        data_t *d = dst + dst_idx;
        *d = data_t(0);
        for (int ic = 0; ic < IC; ++ic)
        for (int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw){
            if (oh*SH + kh < std::max(0, PH)) continue;
            if (ow*SW + kw < std::max(0, PW)) continue;

            if (oh*SH + kh >= IH + PH) continue;
            if (ow*SW + kw >= IW + PW) continue;

            const int ih = oh * SH - PH + kh;
            const int iw = ow * SW - PW + kw;

            size_t src_idx = ((mb*IC + ic)*IH + ih)*IW + iw;
            size_t weights_idx = ((oc*IC + ic)*KH + kh)*KW + kw;
            *d += src[src_idx] * weights[weights_idx];
        }
    }
}
