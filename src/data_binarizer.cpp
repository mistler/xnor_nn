#include "xnor_nn.h"

#include <cmath>

#include "utils/logger.hpp"
#include "utils/timer.hpp"

using Logger = xnor_nn::utils::Logger;

namespace {

size_t data_size(const xnor_nn_data_binarizer_t *s) {
    return s->c->mb * s->c->ic * s->c->ih * s->c->iw * sizeof(float);
}

size_t a_offset(const xnor_nn_data_binarizer_t *s) {
    return data_size(s);
}

size_t a_size(const xnor_nn_data_binarizer_t *s) {
    return s->c->ih * s->c->iw * sizeof(float);
}

size_t k_offset(const xnor_nn_data_binarizer_t *s) {
    return a_offset(s) + a_size(s);
}

size_t k_size(const xnor_nn_data_binarizer_t *s) {
    return s->c->oh * s->c->ow * sizeof(float);
}

size_t sz(const xnor_nn_data_binarizer_t *s){
    return data_size(s) + a_size(s) + k_size(s);
}

xnor_nn_status_t copy_on_float(const float *from, float *to,
        int MB, int IC, int IH, int IW) {
    const int elems = MB*IC*IH*IW;

#   pragma omp parallel for schedule(static)
    for(int i = 0; i < elems; i++) to[i] = from[i];

    return xnor_nn_success;
}

xnor_nn_status_t binarize_char(const unsigned int *from, unsigned char *to,
        int MB, int IC, int IH, int IW) {
    const int SZ = 8;
    const int OC = (IC + SZ - 1) / SZ;

#   pragma omp parallel for collapse(3) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int ih = 0; ih < IH; ih++)
    for (int iw = 0; iw < IW; iw++)
    for (int oc = 0; oc < OC; oc++) {
        unsigned char out{0};
        const int LEN = oc == OC - 1 ? (IC % SZ) : SZ;
        for (int ic = 0; ic < LEN; ic++) {
            int from_idx = (ic*IH + ih)*IW + iw;
            char tmp = from[from_idx] >> 31;
            out <<= 1;
            out |= tmp;
        }
        if (LEN != SZ) out <<= SZ-LEN;
        int to_idx = (ih*IW + iw)*OC + oc;
        to[to_idx] = out;
    }

    return xnor_nn_success;
}

xnor_nn_status_t calculate_k(const float *from, float *a, float *k,
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

// TODO: dispatch at init time
xnor_nn_status_t binarize_dispatch(const xnor_nn_data_binarizer_t *s,
        const void *from, void *to) {
    const int MB = s->c->mb;
    const int IC = s->c->ic;
    const int IH = s->c->ih;
    const int IW = s->c->iw;

    xnor_nn_status_t st;

    xnor_nn::utils::Timer timer;
    timer.start();

    switch (s->c->algorithm) {
    case xnor_nn_algorithm_reference:
    {
        st = copy_on_float((float*)from, (float*)to, MB, IC, IH, IW);
        break;
    }
    case xnor_nn_algorithm_optimized:
    {
        st = binarize_char((unsigned int*)from, (unsigned char*)to,
                MB, IC, IH, IW);
        break;
    }
    }

    timer.stop();
    Logger::info("data_binarizer:", "binarize:",
            "MB:", MB, "IC:", IC, "IH:", IH, "IW:", IW,
            "time:", timer.millis(), "ms");

    return st;
}

// TODO: dispatch at init time
xnor_nn_status_t calculate_k_dispatch(const xnor_nn_data_binarizer_t *s,
        const void *from, void *to) {
    (void)calculate_k;

    const int MB = s->c->mb;
    const int IC = s->c->ic;
    const int IH = s->c->ih;
    const int IW = s->c->iw;

    const int OH = s->c->oh;
    const int OW = s->c->ow;
    const int KH = s->c->kh;
    const int KW = s->c->kw;
    const int SH = s->c->sh;
    const int SW = s->c->sw;
    const int PH = s->c->ph;
    const int PW = s->c->pw;

    xnor_nn_status_t st;

    xnor_nn::utils::Timer timer;
    timer.start();

    switch (s->c->algorithm) {
    case xnor_nn_algorithm_reference:
    {
        float *a_ptr = (float*)to + a_offset(s)/sizeof(float);
        float *k_ptr = (float*)to + k_offset(s)/sizeof(float);
        st = calculate_k((float*)from, a_ptr, k_ptr,
            MB, IC, IH, IW, OH, OW, KH, KW, SH, SW, PH, PW);
        break;
    }
    case xnor_nn_algorithm_optimized:
    {
        break;
    }
    }

    timer.stop();
    Logger::info("data_binarizer:", "calculate_k:",
            "MB:", MB, "IC:", IC, "IH:", IH, "IW:", IW, "OH:", OH, "OW:", OW,
            "KH:", KH, "KW:", KW, "SH:", SH, "SW:", SW, "PH:", PH, "PW:", PW,
            "time:", timer.millis(), "ms");

    return st;
}

}

xnor_nn_status_t xnor_nn_init_data_binarizer(xnor_nn_data_binarizer_t *b,
        const xnor_nn_convolution_t *c) {
    b->c = c;

    b->size = sz;
    b->binarize = binarize_dispatch;
    b->calculate_k = calculate_k_dispatch;

    Logger::info("data_binarizer:", "create:",
            "MB:", c->mb, "IC:", c->ic, "IH:", c->ih, "IW:", c->iw,
            "OH:", c->oh, "OW:", c->ow);

    return xnor_nn_success;
}
