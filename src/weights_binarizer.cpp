#include "xnor_nn.h"

#include "utils/logger.hpp"
#include "utils/timer.hpp"

using Logger = xnor_nn::utils::Logger;

namespace {

size_t sz(const void *s){
    const xnor_nn_weights_binarizer_t *self =
        (const xnor_nn_weights_binarizer_t*)s;

    size_t elems = self->sz[0] * self->sz[1] * self->sz[2] * self->sz[3];
    return elems * sizeof(float);
}

xnor_nn_status_t copy_on_float(const float *from, float *to,
        int OC, int IC, int KH, int KW) {
    int elems = OC*IC*KH*KW;

#   pragma omp parallel for
    for(int i = 0; i < elems; i++) to[i] = from[i];

    return xnor_nn_success;
}

// TODO: dispatch at init time
xnor_nn_status_t weights_bin_dispatch(
        const void *s, const void *from, void *to) {
    const xnor_nn_weights_binarizer_t *self =
        (const xnor_nn_weights_binarizer_t*)s;

    const float *f = (const float*)from;
    float *t = (float*)to;

    const int OC = self->sz[3];
    const int IC = self->sz[2];
    const int KH = self->sz[1];
    const int KW = self->sz[0];

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
    b->sz[0] = kw;
    b->sz[1] = kh;
    b->sz[2] = ic;
    b->sz[3] = oc;

    b->size = sz;
    b->execute = weights_bin_dispatch;

    return xnor_nn_success;
}
