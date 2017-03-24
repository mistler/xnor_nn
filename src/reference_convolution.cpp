#include "reference_convolution.hpp"

#include "logger.hpp"

namespace xnor_nn {
namespace implementation {

bool ReferenceConvolution::isApplicable(
        const xnor_nn_convolution_t *c) const {
    if (c->forward != nullptr) return false;
    return true;
}

void ReferenceConvolution::setupConvolution(xnor_nn_convolution_t *c) {
    const int ELEM_SIZE = sizeof(float);

    c->resource_size[xnor_nn_resource_bin_src] =
        c->mb * c->ic * c->ih * c->iw * ELEM_SIZE;
    c->resource_size[xnor_nn_resource_bin_weights] =
        c->oc * c->ic * c->kh * c->kw * ELEM_SIZE;
    c->resource_size[xnor_nn_resource_a] = c->ih * c->iw * sizeof(float);
    c->resource_size[xnor_nn_resource_k] = c->oh * c->ow * sizeof(float);

    c->binarize_data = binarize_data;
    c->binarize_weights = binarize_weights;
    c->calculate_k = calculate_k;
    c->forward = exec;

    ReferenceConvolution *op = new ReferenceConvolution(*this);
    setState(c, op);
}

ReferenceConvolution::~ReferenceConvolution() {}

xnor_nn_status_t ReferenceConvolution::exec(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    if (
        res[xnor_nn_resource_bin_src] == nullptr
        || res[xnor_nn_resource_bin_weights] == nullptr
        || res[xnor_nn_resource_user_dst] == nullptr
        || res[xnor_nn_resource_k] == nullptr
        || c == nullptr
    ) return xnor_nn_error_invalid_input;
    const float *src = (float *)res[xnor_nn_resource_bin_src];
    const float *weights = (float*)res[xnor_nn_resource_bin_weights];
    float *dst = (float*)res[xnor_nn_resource_user_dst];
    float *alpha = (float*)&res[xnor_nn_resource_alpha];
    const float *k = (float*)res[xnor_nn_resource_k];

    const int MB = c->mb;
    const int IC = c->ic;
    const int IH = c->ih;
    const int IW = c->iw;
    const int OC = c->oc;
    const int OH = c->oh;
    const int OW = c->ow;
    const int KH = c->kh;
    const int KW = c->kw;
    const int SH = c->sh;
    const int SW = c->sw;
    const int PH = c->ph;
    const int PW = c->pw;

    LOG_INFO("convolution:\t", "execute:",
            "[", MB, "][", IC, "][", IH, "][", IW, "]",
            "x",
            "[", OC, "][", IC, "][", KH, "][", KW, "]",
            "=",
            "[", MB, "][", OC, "][", OH, "][", OW, "]",
            "stride: [", SH, "][", SW, "]",
            "pad: [", PH, "][", PW, "]",
            "Algorithm:", "reference");

#   pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oc = 0; oc < OC; oc++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        int dst_idx = ((mb*OC + oc)*OH + oh)*OW + ow;
        float *d = dst + dst_idx;
        *d = 0.f;
        for (int ic = 0; ic < IC; ic++) {
            for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++) {
                const int ih = oh*SH - PH + kh;
                const int iw = ow*SW - PW + kw;

                if (ih < 0 || iw < 0) continue;
                if (ih >= IH || iw >= IW) continue;

                int src_idx = ((mb*IC + ic)*IH + ih)*IW + iw;
                int weights_idx = ((oc*IC + ic)*KH + kh)*KW + kw;

                // TODO: optimize me
                bool bsrc = src[src_idx] >= 0 ? true : false;
                bool bweights = weights[weights_idx] >= 0 ? true : false;

                float result = !(bsrc ^ bweights);
                *d += result*2.f-1.f; // {0,1} -> {-1,1}
            }
        }
        *d *= *alpha;
        *d *= k[oh*OW + ow];
    }

    return xnor_nn_success;
}

} // namespace implementation
} // namespace xnor_nn
