#include "binarize_data.h"

xnor_nn_status_t direct_convolution_forward(
        const unsigned char *src, const unsigned char *weights, float *dst,
        float alpha, const float *k,
        int MB, int IC, int IH, int IW,
        int OC, int OH, int OW,
        int KH, int KW, int SH, int SW, int PH, int PW) {
    const int BIC = (IC + 8 - 1) / 8;

    // TODO: potentially loops can be reordered
#   pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++)
    for (int oc = 0; oc < OC; oc++) {
        int dst_idx = ((mb*OC + oc)*OH + oh)*OW + ow;
        float *d = dst + dst_idx;
        *d = 0.f;
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            if (oh*SH + kh < (PH > 0 ? PH : 0)) continue;
            if (ow*SW + kw < (PW > 0 ? PW : 0)) continue;

            if (oh*SH + kh >= IH + PH) continue;
            if (ow*SW + kw >= IW + PW) continue;

            const int ih = oh * SH - PH + kh;
            const int iw = ow * SW - PW + kw;

            for (int bic = 0; bic < BIC; bic++) {
                int src_idx = ((mb*IH + ih)*IW + iw)*BIC + bic;
                int weights_idx = ((kh*KW + kw)*OC + oc)*BIC + bic;

                unsigned char bsrc = src[src_idx];
                unsigned char bweights = weights[weights_idx];

                unsigned char result = ~(bsrc ^ bweights);
                *d += __builtin_popcount((unsigned int)result);
            }
        }
        *d *= alpha;
        *d *= k[oh*OW + ow];
    }

    return xnor_nn_success;
}
