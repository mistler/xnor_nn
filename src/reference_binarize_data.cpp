#include <cmath>

#include "binarize_data.h"

xnor_nn_status_t reference_data_copy_on_float(const float *from, float *to,
        int MB, int IC, int IH, int IW) {
    const int elems = MB*IC*IH*IW;

#   pragma omp parallel for schedule(static)
    for(int i = 0; i < elems; i++) to[i] = from[i];

    return xnor_nn_success;
}

xnor_nn_status_t reference_calculate_k(const float *from, float *a, float *k,
        int MB, int IC, int IH, int IW, int OH, int OW,
        int KH, int KW, int SH, int SW, int PH, int PW) {
    const float c = 1.f / IC;
    const float khw = 1.f / KH / KW;

    // Calculate A
#   pragma omp parallel for collapse(2) schedule(static)
    for (int ih = 0; ih < IH; ih++)
    for (int iw = 0; iw < IW; iw++) {
        float *a_curr = a + ih*IW + iw;
        *a_curr = 0.f;
        for (int mb = 0; mb < MB; mb++)
        for (int ic = 0; ic < IC; ic++) {
            int src_idx = ((mb*IC + ic)*IH + ih)*IW + iw;
            *a_curr += std::fabs(from[src_idx]) * c;
        }
    }

    // Calculate K
#   pragma omp parallel for collapse(2) schedule(static)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float *k_curr = k + oh*OW + ow;
        *k_curr = 0.f;
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            if (oh*SH + kh < (PH > 0 ? PH : 0)) continue;
            if (ow*SW + kw < (PW > 0 ? PW : 0)) continue;

            if (oh*SH + kh >= IH + PH) continue;
            if (ow*SW + kw >= IW + PW) continue;

            const int ih = oh * SH - PH + kh;
            const int iw = ow * SW - PW + kw;

            *k_curr += a[ih*IW + iw] * khw;
        }
    }

    return xnor_nn_success;
}