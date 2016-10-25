#include "xnor_nn.h"

#include <cmath>

#include "utils/logger.hpp"
#include "utils/timer.hpp"

using Logger = xnor_nn::utils::Logger;

namespace {

size_t sz(const xnor_nn_weights_binarizer_t *s){
    size_t elems = s->c->oc * s->c->ic * s->c->kh * s->c->kw; // Kernels
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
    const int OC = s->c->oc;
    const int IC = s->c->ic;
    const int KH = s->c->kh;
    const int KW = s->c->kw;

    xnor_nn_status_t st;

    xnor_nn::utils::Timer timer;
    timer.start();

    switch (s->c->algorithm) {
    case xnor_nn_algorithm_reference:
    {
        const float *f = (const float*)from;
        float *t = (float*)to;
        st = copy_on_float(f, t, OC, IC, KH, KW);
        break;
    }
    case xnor_nn_algorithm_optimized:
    {
        break;
    }
    }

    timer.stop();
    Logger::info("weights_binarizer:", "execute:",
            "OC:", OC, "IC:", IC, "KH:", KH, "KW:", KW,
            "time:", timer.millis(), "ms");

    return st;
}

}

xnor_nn_status_t xnor_nn_init_weights_binarizer(xnor_nn_weights_binarizer_t *b,
        const xnor_nn_convolution_t *c) {
    b->c = c;

    b->size = sz;
    b->execute = weights_bin_dispatch;

    Logger::info("weights_binarizer:", "create:",
            "OC:", c->oc, "IC:", c->ic, "KH:", c->kh, "KW:", c->kw);

    return xnor_nn_success;
}
