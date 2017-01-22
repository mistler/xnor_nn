#include "bcast_binarize_weights.hpp"

#include <cmath>

#include "utils.h"
#include "logger.hpp"

using Logger = xnor_nn::utils::Logger;

namespace xnor_nn {
namespace implementation {

bool BcastBinarizeWeights::isApplicable(const xnor_nn_convolution_t *c) const {
    bool ok = this->BcastBase::isApplicable(c)
        && c->binarize_weights == nullptr;
    return ok;
}

void BcastBinarizeWeights::setupConvolution(xnor_nn_convolution_t *c) {
    BcastBinarizeWeights *op = new BcastBinarizeWeights;
    op->BcastBase::setupConvolution(c);
    setState(c, op, xnor_nn_operation_binarize_weights);
    c->binarize_weights = op->exec;
}

BcastBinarizeWeights::~BcastBinarizeWeights() {}

xnor_nn_status_t BcastBinarizeWeights::exec(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    if (
        res[xnor_nn_resource_user_weights] == nullptr
        || res[xnor_nn_resource_bin_weights] == nullptr
        || c == nullptr
    ) return xnor_nn_error_invalid_input;

    const float *from = (float*)res[xnor_nn_resource_user_weights];
    unsigned char *to = (unsigned char*)res[xnor_nn_resource_bin_weights];
    float *alpha = (float*)&res[xnor_nn_resource_alpha];
    const unsigned int *f = (unsigned int*)from;

    const int OC = c->oc;
    const int IC = c->ic;
    const int KH = c->kh;
    const int KW = c->kw;

    BcastBinarizeWeights *state = reinterpret_cast<BcastBinarizeWeights*>(
            getState(c, xnor_nn_operation_binarize_weights));

    constexpr int SZ = state->SZ;
    constexpr int BICI = state->BICI;
    constexpr int OCI = state->OCI;

    const int ICO = state->ICO;
    const int OCO = state->OCO;

    const int elems = OC*IC*KH*KW;

    Logger::info("binarize_weights:", "execute:",
            "[", OC, "]",
            "[", IC, "]",
            "[", KH, "]",
            "[", KW, "]",
            " -> ",
            "[", OCO, "]",
            "[", KH, "]",
            "[", KW, "]",
            "[", ICO, "]",
            "[", OCI, "]",
            "[", BICI, "]",
            "Algorithm:", xnor_nn_algorithm_bcast);

#   pragma omp parallel for collapse(3) schedule(static)
    for (int oco = 0; oco < OCO; oco++)
    for (int kh = 0; kh < KH; kh++)
    for (int kw = 0; kw < KW; kw++)
    for (int ico = 0; ico < ICO; ico++)
    for (int oci = 0; oci < OCI; oci++) {
        for (int bici = 0; bici < BICI; bici++) {
            unsigned char out{0};
            const int current_ic = (ico*BICI + bici)*SZ;
            int ic = 0;
            for (; ic < SZ && current_ic + ic < IC; ic++) {
                const int from_oc = oco*OCI + oci;
                const int from_ic = current_ic + ic;
                const int from_idx = ((from_oc*IC + from_ic)*KH + kh)*KW + kw;
                char tmp = (~f[from_idx]) >> 31;
                out <<= 1;
                out |= tmp;
            }
            if (ic != SZ) {
                // Dirty hack! As this data is fake we want it to have
                // zero influence to the dst after convolution, so lets fill it
                // with ONES so ~(src^weights) = ~(0^1) = 0
                for (int i = 0; i < SZ - ic; i++) {
                    out <<= 1;
                    out |= (unsigned char)1;
                }
            }
            const int to_idx =
                ((((oco*KH + kh)*KW + kw)*ICO + ico)*OCI + oci)*BICI + bici;
            to[to_idx] = out;
        }
    }

    // Calculate alpha
    const float cckhw = 1.f / elems;
    *alpha = 0.f;
    for (int i = 0; i < elems; i++) *alpha += std::fabs(from[i]) * cckhw;

    return xnor_nn_success;
}

} // namespace implementation
} // namespace xnor_nn
