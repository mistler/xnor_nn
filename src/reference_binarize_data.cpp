#include "reference_binarize_data.hpp"

#include <cmath>

#include "logger.hpp"

namespace xnor_nn {
namespace implementation {

bool ReferenceBinarizeData::isApplicable(
        const xnor_nn_convolution_t *c) const {
    if (c->binarize_data != nullptr) return false;
    return true;
}

void ReferenceBinarizeData::setupConvolution(
        xnor_nn_convolution_t *c) {
    ReferenceBinarizeData *op = new ReferenceBinarizeData;

    c->binarize_data = op->exec;

    std::vector<Implementation*> *vec =
        (std::vector<Implementation*>*)c->state;
    vec->push_back(op);
}

ReferenceBinarizeData::~ReferenceBinarizeData() {}

xnor_nn_status_t ReferenceBinarizeData::exec(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    if (
        res[xnor_nn_resource_user_src] == nullptr
        || res[xnor_nn_resource_bin_src] == nullptr
        || c == nullptr
    ) return xnor_nn_error_invalid_input;
    const float *from = (float*)res[xnor_nn_resource_user_src];
    float *to = (float *)res[xnor_nn_resource_bin_src];

    const int MB = c->mb;
    const int IC = c->ic;
    const int IH = c->ih;
    const int IW = c->iw;

    const int elems = MB*IC*IH*IW;

    LOG_INFO("binarize_data:\t", "execute:",
            "[", MB, "][", IC, "][", IH, "][", IW, "]",
            "->",
            "[", MB, "][", IC, "][", IH, "][", IW, "]",
            "Algorithm:", "reference");

#   pragma omp parallel for schedule(static)
    for(int i = 0; i < elems; i++) to[i] = from[i];

    return xnor_nn_success;
}

} // namespace implementation
} // namespace xnor_nn
