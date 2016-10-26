#include "xnor_nn.h"

#include <cmath>

#include "utils/logger.hpp"
#include "utils/timer.hpp"

using Logger = xnor_nn::utils::Logger;

namespace {

xnor_nn_status_t copy_on_float(const float *from, float *to, float *alpha,
        int OC, int IC, int KH, int KW) {
    int elems = OC*IC*KH*KW;

#   pragma omp parallel for schedule(static)
    for(int i = 0; i < elems; i++) to[i] = from[i];

    const float cckhw = 1.f / elems;

    // Calculate alpha
    *alpha = 0.f;
    for (int i = 0; i < elems; i++) *alpha += std::fabs(from[i]) * cckhw;

    return xnor_nn_success;
}

// TODO: dispatch at init time
xnor_nn_status_t weights_bin_dispatch(const xnor_nn_weights_binarizer_t *s,
        xnor_nn_resources_t res) {
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
        const float *f = (float*)res[xnor_nn_resource_user_weights];
        float *t = (float*)res[xnor_nn_resource_bin_weights];
        float *alpha = (float*)&(res[xnor_nn_resource_alpha]);
        st = copy_on_float(f, t, alpha, OC, IC, KH, KW);
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

    b->execute = weights_bin_dispatch;

    Logger::info("weights_binarizer:", "create:",
            "OC:", c->oc, "IC:", c->ic, "KH:", c->kh, "KW:", c->kw);

    return xnor_nn_success;
}
