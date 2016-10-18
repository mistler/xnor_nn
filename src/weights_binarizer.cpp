#include "xnor_nn.h"

#include <cmath>

#include "utils/logger.hpp"
#include "utils/timer.hpp"

using Logger = xnor_nn::utils::Logger;

namespace {

size_t sz(const void *s){
    const xnor_nn_weights_binarizer_t *self =
        (const xnor_nn_weights_binarizer_t*)s;

    size_t elems = self->oc * self->ic * self->kh * self->kw; // Kernels
    elems += 1; // Alpha
    return elems * sizeof(float);
}

xnor_nn_status_t copy_on_float(const float *from, float *to,
        int OC, int IC, int KH, int KW) {
    int elems = OC*IC*KH*KW;

#   pragma omp parallel for
    for(int i = 0; i < elems; i++) to[i] = from[i];

    const float cckhw = 1.f / elems;

    // Calculate alpha
    float *alpha = to + elems;
    *alpha = 0.f;
    for (int i = 0; i < elems; i++) *alpha += std::fabs(from[i]) * cckhw;

    return xnor_nn_success;
}

// TODO: dispatch at init time
xnor_nn_status_t weights_bin_dispatch(
        const void *s, const void *from, void *to) {
    const xnor_nn_weights_binarizer_t *self =
        (const xnor_nn_weights_binarizer_t*)s;

    const float *f = (const float*)from;
    float *t = (float*)to;

    const int OC = self->oc;
    const int IC = self->ic;
    const int KH = self->kh;
    const int KW = self->kw;

    xnor_nn::utils::Timer timer;
    timer.start();

    xnor_nn_status_t st = copy_on_float(f, t, OC, IC, KH, KW);

    timer.stop();
    Logger::info("weights_binarizer:", "execute:",
            "OC:", OC, "IC:", IC, "KH:", KH, "KW:", KW,
            "time:", timer.millis(), "ms");

    return st;
}

}

xnor_nn_status_t xnor_nn_init_weights_binarizer(xnor_nn_weights_binarizer_t *b,
        int oc, int ic, int kh, int kw) {
    b->kw = kw;
    b->kh = kh;
    b->ic = ic;
    b->oc = oc;

    b->size = sz;
    b->execute = weights_bin_dispatch;

    Logger::info("weights_binarizer:", "create:",
            "OC:", oc, "IC:", ic, "KH:", kh, "KW:", kw);

    return xnor_nn_success;
}
