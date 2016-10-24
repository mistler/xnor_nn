#include "xnor_nn.h"

#include <cmath>

#include "utils/logger.hpp"
#include "utils/timer.hpp"

using Logger = xnor_nn::utils::Logger;

namespace {

size_t sz(const void *s){
    const xnor_nn_data_binarizer_t *self = (const xnor_nn_data_binarizer_t*)s;

    size_t elems = self->mb * self->ic * self->ih * self->iw; // Data
    elems += self->ih * self->iw; // A
    elems += self->oh * self->ow; // K
    return elems * sizeof(float);
}

xnor_nn_status_t copy_on_float(const float *from, float *to,
        int MB, int IC, int IH, int IW) {
    int elems = MB*IC*IH*IW;
    const float c = 1.f / IC;

#   pragma omp parallel for schedule(static)
    for(int i = 0; i < elems; i++) to[i] = from[i];

    // Calculate A
#   pragma omp parallel for collapse(2) schedule(static)
    for (int ih = 0; ih < IH; ih++)
    for (int iw = 0; iw < IW; iw++) {
        float *a = to + elems + ih*IW + iw;
        *a = 0.f;
        for (int mb = 0; mb < MB; mb++)
        for (int ic = 0; ic < IC; ic++) {
            int src_idx = ((mb*IC + ic)*IH + ih)*IW + iw;
            *a += std::fabs(from[src_idx]) * c;
        }
    }

    return xnor_nn_success;
}

/*
xnor_nn_status_t binarize_char(const float *from, unsigned char *to,
        int MB, int IC, int IH, int IW) {
    const int SZ = sizeof(unsigned char);

#   pragma omp parallel for collapse(3) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int ih = 0; ih < IH; ih++)
    for (int iw = 0; iw < IW; iw++)
    for (int oc = 0; oc < OC / SZ; oc++) {
        unsigned char out{0};
        for (int ic = 0; ic < SZ; ic++) {
            int from_idx = ((mb*IC + oc*SZ + ic)*IH + ih)*IW + iw;
            char tmp = from[from_idx] >> (sizeof(float)-1);
            out |= tmp;
            out <<= 1;
        }
    }

    return xnor_nn_success;
}
*/

xnor_nn_status_t calculate_k(const float *a, float *k, int IH, int IW,
        int OH, int OW, int KH, int KW, int SH, int SW, int PH, int PW) {
    const float khw = 1.f / KH / KW;

#   pragma omp parallel for collapse(2) schedule(static)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float *k_ = k + oh*OW + ow;
        *k_ = 0.f;
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            if (oh*SH + kh < (PH > 0 ? PH : 0)) continue;
            if (ow*SW + kw < (PW > 0 ? PW : 0)) continue;

            if (oh*SH + kh >= IH + PH) continue;
            if (ow*SW + kw >= IW + PW) continue;

            const int ih = oh * SH - PH + kh;
            const int iw = ow * SW - PW + kw;

            *k_ += a[ih*IW + iw] * khw;
        }
    }

    return xnor_nn_success;
}

// TODO: dispatch at init time
xnor_nn_status_t binarize_dispatch(
        const void *s, const void *from, void *to) {
    const xnor_nn_data_binarizer_t *self = (const xnor_nn_data_binarizer_t*)s;

    const float *f = (const float*)from;
    float *t = (float*)to;

    const int MB = self->mb;
    const int IC = self->ic;
    const int IH = self->ih;
    const int IW = self->iw;

    xnor_nn::utils::Timer timer;
    timer.start();

    xnor_nn_status_t st = copy_on_float(f, t, MB, IC, IH, IW);

    timer.stop();
    Logger::info("data_binarizer:", "binarize:",
            "MB:", MB, "IC:", IC, "IH:", IH, "IW:", IW,
            "time:", timer.millis(), "ms");

    return st;
}

// TODO: dispatch at init time
xnor_nn_status_t calculate_k_dispatch(const void *s, void *to) {
    const xnor_nn_data_binarizer_t *self = (const xnor_nn_data_binarizer_t*)s;

    const int MB = self->mb;
    const int IC = self->ic;
    const int IH = self->ih;
    const int IW = self->iw;

    const int OH = self->oh;
    const int OW = self->ow;
    const int KH = self->kh;
    const int KW = self->kw;
    const int SH = self->sh;
    const int SW = self->sw;
    const int PH = self->ph;
    const int PW = self->pw;

    const int elems = MB*IC*IH*IW;
    const int a_elems = IH*IW;

    const float *a_ptr = (float*)to + elems;
    float *k_ptr = (float*)to + elems + a_elems;

    xnor_nn::utils::Timer timer;
    timer.start();

    xnor_nn_status_t st = calculate_k(a_ptr, k_ptr, IH, IW,
            OH, OW, KH, KW, SH, SW, PH, PW);

    timer.stop();
    Logger::info("data_binarizer:", "calculate_k:",
            "IH:", IH, "IW:", IW, "OH:", OH, "OW:", OW, "KH:", KH, "KW:", KW,
            "SH:", SH, "SW:", SW, "PH:", PH, "PW:", PW,
            "time:", timer.millis(), "ms");

    return st;
}

}

xnor_nn_status_t xnor_nn_init_data_binarizer(xnor_nn_data_binarizer_t *b,
        const xnor_nn_convolution_t *c) {
    b->mb = c->mb;
    b->ic = c->ic;
    b->ih = c->ih;
    b->iw = c->iw;

    b->oh = c->oh;
    b->ow = c->ow;
    b->kh = c->kh;
    b->kw = c->kw;
    b->sh = c->sh;
    b->sw = c->sw;
    b->ph = c->ph;
    b->pw = c->pw;

    b->size = sz;
    b->binarize = binarize_dispatch;
    b->calculate_k = calculate_k_dispatch;

    Logger::info("data_binarizer:", "create:",
            "MB:", b->mb, "IC:", b->ic, "IH:", b->ih, "IW:", b->iw,
            "OH:", b->oh, "OW:", b->ow);

    return xnor_nn_success;
}
