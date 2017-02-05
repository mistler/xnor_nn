#include "reference_calculate_k.hpp"

#include <cmath>

#include "logger.hpp"

namespace xnor_nn {
namespace implementation {

bool ReferenceCalculateK::isApplicable(
        const xnor_nn_convolution_t *c) const {
    if (c->calculate_k != nullptr) return false;
    return true;
}

void ReferenceCalculateK::setupConvolution(xnor_nn_convolution_t *c) {
    ReferenceCalculateK *op = new ReferenceCalculateK;

    c->calculate_k = op->exec;

    std::vector<Implementation*> *vec =
        (std::vector<Implementation*>*)c->state;
    vec->push_back(op);
}

ReferenceCalculateK::~ReferenceCalculateK() {}

xnor_nn_status_t ReferenceCalculateK::exec(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    if (
        res[xnor_nn_resource_user_src] == nullptr
        || res[xnor_nn_resource_a] == nullptr
        || res[xnor_nn_resource_k] == nullptr
        || c == nullptr
    ) return xnor_nn_error_invalid_input;
    const float *from = (float*)res[xnor_nn_resource_user_src];
    float *a = (float*)res[xnor_nn_resource_a];
    float *k = (float*)res[xnor_nn_resource_k];

    const int MB = c->mb;
    const int IC = c->ic;
    const int IH = c->ih;
    const int IW = c->iw;
    const int OH = c->oh;
    const int OW = c->ow;
    const int KH = c->kh;
    const int KW = c->kw;
    const int SH = c->sh;
    const int SW = c->sw;
    const int PH = c->ph;
    const int PW = c->pw;

    const float C = 1.f / IC;
    const float KHW = 1.f / KH / KW;

    LOG_INFO("calculate_k:\t", "execute:",
            "[", MB, "][", IC, "][", IH, "][", IW, "]",
            "->",
            "[", "1", "]",
            "+",
            "[", IH, "][", IW, "]",
            "Algorithm:", "reference");

    // Calculate A
#   pragma omp parallel for collapse(2) schedule(static)
    for (int ih = 0; ih < IH; ih++)
    for (int iw = 0; iw < IW; iw++) {
        float *a_curr = a + ih*IW + iw;
        *a_curr = 0.f;
        for (int mb = 0; mb < MB; mb++)
        for (int ic = 0; ic < IC; ic++) {
            int src_idx = ((mb*IC + ic)*IH + ih)*IW + iw;
            *a_curr += std::fabs(from[src_idx]) * C;
        }
    }

    // Calculate K
#   pragma omp parallel for collapse(2) schedule(static)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float *k_curr = k + oh*OW + ow;
        *k_curr = 0.f;
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            const int ih = oh*SH - PH + kh;
            const int iw = ow*SW - PW + kw;

            if (ih < 0 || iw < 0) continue;
            if (ih >= IH || iw >= IW) continue;

            *k_curr += a[ih*IW + iw] * KHW;
        }
    }

    return xnor_nn_success;
}

} // namespace implementation
} // namespace xnor_nn
