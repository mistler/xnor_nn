#include "xnor_nn.h"

#include <cmath>

#include "utils/logger.hpp"
#include "utils/timer.hpp"

using Logger = xnor_nn::utils::Logger;

namespace {

size_t sz(const void *s){
    const xnor_nn_data_binarizer_t *self = (const xnor_nn_data_binarizer_t*)s;

    size_t elems = self->mb * self->c * self->h * self->w; // Data
    elems += self->h * self->w; // A
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

// TODO: dispatch at init time
xnor_nn_status_t data_bin_dispatch(
        const void *s, const void *from, void *to) {
    const xnor_nn_data_binarizer_t *self = (const xnor_nn_data_binarizer_t*)s;

    const float *f = (const float*)from;
    float *t = (float*)to;

    const int MB = self->mb;
    const int IC = self->c;
    const int IH = self->h;
    const int IW = self->w;

    xnor_nn::utils::Timer timer;
    timer.start();

    xnor_nn_status_t st = copy_on_float(f, t, MB, IC, IH, IW);

    timer.stop();
    Logger::info("data_binarizer:", "execute:",
            "MB:", MB, "IC:", IC, "IH:", IH, "IW:", IW,
            "time:", timer.millis(), "ms");

    return st;
}

}

xnor_nn_status_t xnor_nn_init_data_binarizer(xnor_nn_data_binarizer_t *b,
        const xnor_nn_convolution_t *c) {
    b->mb = c->mb;
    b->c = c->ic;
    b->h = c->ih;
    b->w = c->iw;

    b->size = sz;
    b->execute = data_bin_dispatch;

    Logger::info("data_binarizer:", "create:",
            "MB:", b->mb, "IC:", b->c, "IH:", b->h, "IW:", b->w);

    return xnor_nn_success;
}
