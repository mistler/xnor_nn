#include <cmath>
#include <vector>

#include "xnor_nn.h"
#include "implementation.hpp"
#include "utils.hpp"
#include "convolution_logger.hpp"

xnor_nn_status_t xnor_nn_init_convolution(xnor_nn_convolution_t *c,
        const xnor_nn_algorithm_t algorithm,
        const xnor_nn_tensor_format_t src_fmt,
        const xnor_nn_tensor_format_t weights_fmt,
        const xnor_nn_tensor_format_t dst_fmt,
        int mb, int oc, int ic, int ih, int iw,
        int kh, int kw, int sh, int sw, int ph, int pw) {
    const int oh = xnor_nn::utils::getOH(ih, kh, sh, pw);
    const int ow = xnor_nn::utils::getOW(iw, kw, sw, pw);

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

    c->src_format = src_fmt;
    c->weights_format = weights_fmt;
    c->dst_format = dst_fmt;

    c->binarize_data = nullptr;
    c->binarize_weights = nullptr;
    c->calculate_k = nullptr;
    c->forward = nullptr;

    c->state = nullptr;

    for (int i = xnor_nn_resource_internal; i < xnor_nn_resource_number; i++)
        c->resource_size[i] = 0;

    using namespace xnor_nn::utils;
    logger::log<logger::init, logger::convolution>::info(c);

    for (xnor_nn::implementation::Implementation *impl
            : xnor_nn::implementation::Implementations()) {
        if (impl->isApplicable(c)) {
            impl->setupConvolution(c);
        }
    }

    return xnor_nn_success;
}

void xnor_nn_destroy_convolution(xnor_nn_convolution_t *c){
    delete reinterpret_cast<xnor_nn::implementation::Implementation*>(c->state);
}
