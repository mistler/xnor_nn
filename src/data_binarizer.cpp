#include "xnor_nn.h"

#include "utils/logger.hpp"
#include "utils/timer.hpp"

using Logger = xnor_nn::utils::Logger;

namespace {

size_t sz(const void *s){
    const xnor_nn_data_binarizer_t *self = (const xnor_nn_data_binarizer_t*)s;

    size_t elems = self->mb * self->sz[0] * self->sz[1] * self->sz[2];
    return elems * sizeof(float);
}

xnor_nn_status_t copy_on_float(const float *from, float *to,
        int MB, int IC, int IH, int IW) {
    int elems = MB*IC*IH*IW;

#   pragma omp parallel for
    for(int i = 0; i < elems; i++) to[i] = from[i];

    return xnor_nn_success;
}

// TODO: dispatch at init time
xnor_nn_status_t data_bin_dispatch(
        const void *s, const void *from, void *to) {
    const xnor_nn_data_binarizer_t *self = (const xnor_nn_data_binarizer_t*)s;

    const float *f = (const float*)from;
    float *t = (float*)to;

    const int MB = self->mb;
    const int IC = self->sz[2];
    const int IH = self->sz[1];
    const int IW = self->sz[0];

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
        int mb, int ic, int ih, int iw) {
    b->mb = mb;
    b->sz[0] = iw;
    b->sz[1] = ih;
    b->sz[2] = ic;

    b->size = sz;
    b->execute = data_bin_dispatch;

    return xnor_nn_success;
}
