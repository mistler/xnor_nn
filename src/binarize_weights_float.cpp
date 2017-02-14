#include "xnor_nn.h"

void xnor_nn_binarize_weights_float(const xnor_nn_convolution_t *c,
        const float *from, float *to) {
    const int OC = c->oc;
    const int IC = c->ic;
    const int KH = c->kh;
    const int KW = c->kw;

    const int elems = OC*IC*KH*KW;

    const float cckhw = 1.f / elems;
    // TODO: use correct binarization to {-1,1} instead of {0,1}

#   pragma omp parallel for schedule(static)
    for (int i = 0; i < elems; i++)
        to[i] = (from[i] > .0f ? 1.f : 0.f) * cckhw;
}
