#include "xnor_nn.h"

#include <cmath>

#include "utils/logger.hpp"
#include "utils/timer.hpp"

using Logger = xnor_nn::utils::Logger;

namespace {

size_t sz(const xnor_nn_weights_binarizer_t *s){
    size_t elems = s->oc * s->ic * s->kh * s->kw; // Kernels
    elems += 1; // Alpha
    return elems * sizeof(float);
}

xnor_nn_status_t copy_on_float(const float *from, float *to,
        int OC, int IC, int KH, int KW) {
    int elems = OC*IC*KH*KW;

#   pragma omp parallel for schedule(static)
    for(int i = 0; i < elems; i++) to[i] = from[i];

    const float cckhw = 1.f / elems;

    // Calculate alpha
    float *alpha = to + elems;
    *alpha = 0.f;
    for (int i = 0; i < elems; i++) *alpha += std::fabs(from[i]) * cckhw;

    return xnor_nn_success;
}

// TODO: dispatch at init time
xnor_nn_status_t weights_bin_dispatch(const xnor_nn_weights_binarizer_t *s,
        const void *from, void *to) {
    const float *f = (const float*)from;
    float *t = (float*)to;

    const int OC = s->oc;
    const int IC = s->ic;
    const int KH = s->kh;
    const int KW = s->kw;

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
        const xnor_nn_convolution_t *c) {
    b->oc = c->oc;
    b->ic = c->ic;
    b->kh = c->kh;
    b->kw = c->kw;

    b->size = sz;
    b->execute = weights_bin_dispatch;

    Logger::info("weights_binarizer:", "create:",
            "OC:", b->oc, "IC:", b->ic, "KH:", b->kh, "KW:", b->kw);

    return xnor_nn_success;
}
