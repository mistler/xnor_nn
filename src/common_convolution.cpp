#include <cmath>
#include <vector>

#include "xnor_nn.h"
#include "implementation.hpp"
#include "utils.hpp"
#include "logger.hpp"

xnor_nn_status_t xnor_nn_init_convolution(xnor_nn_convolution_t *c,
        const xnor_nn_algorithm_t algorithm,
        int mb, int oc, int ic, int ih, int iw,
        int kh, int kw, int sh, int sw, int ph, int pw) {
    const int oh = getOH(ih, kh, sh, pw);
    const int ow = getOW(iw, kw, sw, pw);

    c->algorithm = algorithm;

    c->mb = mb;

    c->ic = ic;
    c->ih = ih;
    c->iw = iw;

    c->oc = oc;
    c->oh = oh;
    c->ow = ow;

    c->kw = kw;
    c->kh = kh;
    c->sw = sw;
    c->sh = sh;
    c->pw = pw;
    c->ph = ph;

    c->binarize_data = nullptr;
    c->binarize_weights = nullptr;
    c->calculate_k = nullptr;
    c->forward = nullptr;

    c->state =
        (void*)new std::vector<xnor_nn::implementation::Implementation*>();

    for (xnor_nn::implementation::Implementation *impl
            : xnor_nn::implementation::Implementations()) {
        if (impl->isApplicable(c)) {
            impl->setupConvolution(c);
        }
    }

    LOG_INFO("convolution:\t", "create: ",
            "[", mb, "][", ic, "][", ih, "][", iw, "]",
            "x",
            "[", oc, "][", ic, "][", kh, "][", kw, "]",
            "=",
            "[", mb, "][", oc, "][", oh, "][", ow, "]",
            "stride: [", sh, "][", sw, "]",
            "pad: [", ph, "][", pw, "]",
            "Algorithm:", algorithm);

    return xnor_nn_success;
}

void xnor_nn_destroy_convolution(xnor_nn_convolution_t *c){
    std::vector<xnor_nn::implementation::Implementation*> *vec =
        (std::vector<xnor_nn::implementation::Implementation*>*)c->state;
    while (!vec->empty()) {
        delete vec->back();
        vec->pop_back();
    }
    delete (std::vector<xnor_nn::implementation::Implementation*>*)c->state;
}
