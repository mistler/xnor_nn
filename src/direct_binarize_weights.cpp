#include "direct_binarize_weights.hpp"

#include <cmath>

namespace xnor_nn {
namespace implementation {

bool DirectBinarizeWeightsChar::isApplicable(
        const xnor_nn_convolution_t *c) const {
    if (c->binarize_weights != nullptr) return false;
    if (c->algorithm != xnor_nn_algorithm_optimized) return false;
    return true;
}

void DirectBinarizeWeightsChar::setupConvolution(
        xnor_nn_convolution_t *c) {
    c->binarize_weights = exec;
    ((std::vector<Implementation*>*)c->state)->push_back(this);
}

xnor_nn_status_t DirectBinarizeWeightsChar::exec(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    const float *from = (float*)res[xnor_nn_resource_user_weights];
    unsigned char *to = (unsigned char*)res[xnor_nn_resource_bin_weights];
    float *alpha = (float*)&res[xnor_nn_resource_alpha];
    const unsigned int *f = (unsigned int*)from;

    const int OC = c->oc;
    const int IC = c->ic;
    const int KH = c->kh;
    const int KW = c->kw;

    const int elems = OC*IC*KH*KW;
    const int SZ = c->sizeof_element * 8;
    const int BIC = c->bic;
    const int AIC = c->aic;

#   pragma omp parallel for collapse(3) schedule(static)
    for (int kh = 0; kh < KH; kh++)
    for (int kw = 0; kw < KW; kw++)
    for (int oc = 0; oc < OC; oc++) {
        for (int bic = 0; bic < BIC; bic++) {
            unsigned char out{0};
            const int LEN = bic == BIC - 1 ? (IC % SZ) : SZ;
            for (int ic = 0; ic < LEN; ic++) {
                int from_idx = ((oc*IC + ic)*KH + kh)*KW + kw;
                char tmp = (~f[from_idx]) >> 31;
                out <<= 1;
                out |= tmp;
            }
            if (LEN != SZ) {
                // Dirty hack! As this data is fake we want it to have
                // zero influence to the dst after convolution, so lets fill it
                // with ONES and after ~(src^weights) it will be ZERO
                // because corresponding values in src are zeros
                // before the convolution forward.
                for (int i = 0; i < SZ-LEN; i++) {
                    out <<= 1;
                    out |= (unsigned char)1;
                }
            }
            int to_idx = ((kh*KW + kw)*OC + oc)*AIC + bic;
            to[to_idx] = out;
        }
        for (int r = 0; r < AIC - BIC; r++) {
            int to_idx = ((kh*KW + kw)*OC + oc)*AIC + BIC + r;
            to[to_idx] = 0xFFu;
        }
    }

    // Calculate alpha
    const float cckhw = 1.f / elems;
    *alpha = 0.f;
    for (int i = 0; i < elems; i++) *alpha += std::fabs(from[i]) * cckhw;

    return xnor_nn_success;
}

}
}
