#include "reference_binarize_weights.hpp"

#include <cmath>

namespace xnor_nn {
namespace implementation {

bool ReferenceBinarizeWeightsCopyOnFloat::isApplicable(
        const xnor_nn_convolution_t *c) const {
    if (c->binarize_weights != nullptr) return false;
    if (c->algorithm != xnor_nn_algorithm_reference) return false;
    return true;
}

void ReferenceBinarizeWeightsCopyOnFloat::setupConvolution(
        xnor_nn_convolution_t *c) {
    c->binarize_weights = exec;
    ((std::vector<Implementation*>*)c->state)->push_back(this);
}

xnor_nn_status_t ReferenceBinarizeWeightsCopyOnFloat::exec(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    const float *from = (float*)res[xnor_nn_resource_user_weights];
    float *to = (float*)res[xnor_nn_resource_bin_weights];
    float *alpha = (float*)&res[xnor_nn_resource_alpha];

    const int OC = c->oc;
    const int IC = c->ic;
    const int KH = c->kh;
    const int KW = c->kw;

    const int elems = OC*IC*KH*KW;

#   pragma omp parallel for schedule(static)
    for(int i = 0; i < elems; i++) to[i] = from[i];

    const float cckhw = 1.f / elems;

    // Calculate alpha
    *alpha = 0.f;
    for (int i = 0; i < elems; i++) *alpha += std::fabs(from[i]) * cckhw;

    return xnor_nn_success;
}

}
}
