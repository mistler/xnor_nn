#include <cmath>
#include <vector>

#include "xnor_nn.h"

#include "implementation.hpp"

#include "utils/logger.hpp"

using Logger = xnor_nn::utils::Logger;

xnor_nn_status_t xnor_nn_init_convolution(xnor_nn_convolution_t *c,
        const xnor_nn_algorithm_t algorithm,
        int mb, int oc, int ic, int ih, int iw,
        int kh, int kw, int sh, int sw, int ph, int pw) {
    const int oh = (ih + 2*ph - kh) / sh + 1;
    const int ow = (iw + 2*pw - kw) / sw + 1;

    c->algorithm = algorithm;

    c->mb = mb;

    c->iw = iw;
    c->ih = ih;
    c->ic = ic;

    c->ow = ow;
    c->oh = oh;
    c->oc = oc;

    c->sw = sw;
    c->sh = sh;
    c->kw = kw;
    c->kh = kh;
    c->pw = pw;
    c->ph = ph;

    c->aic = ic;
    c->bic = ic;

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

    Logger::info("convolution:", "create:",
            "MB:", mb, "IC:", ic, "IH:", ih, "IW:", iw,
            "OC:", oc, "OH:", oh, "OW:", ow,
            "KH:", kh, "KW:", kw, "SH:", sh, "SW:", sw, "PH:", ph, "PW:", pw,
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
