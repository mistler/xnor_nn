#include "bcast_binarize_weights.hpp"

#include <cmath>

#include "utils.h"
#include "logger.hpp"

using Logger = xnor_nn::utils::Logger;

namespace xnor_nn {
namespace implementation {

bool BcastBinarizeWeights::isApplicable(
        const xnor_nn_convolution_t *c) const {
    // TODO: make it one bool and log it
    if (c->binarize_weights != nullptr) return false;
    if (c->oc % (VLEN / 32) != 0) return false; // TODO constant in base class
    if (c->algorithm == xnor_nn_algorithm_bcast) return true;
    return false;
}

void BcastBinarizeWeights::setupConvolution(
        xnor_nn_convolution_t *c) {
    BcastBinarizeWeights *op = new BcastBinarizeWeights;

    c->binarize_weights = op->exec;

    std::vector<Implementation*> *vec =
        (std::vector<Implementation*>*)c->state;
    vec->push_back(op);
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

    const int SZ = 8;
    const int VLEN_BYTES = (VLEN / 8);

    const int BICI = 4;

    const int ELEM_SIZE = sizeof(char);
    const int BITS = ELEM_SIZE * SZ;

    const int BIC = (IC + BITS - 1) / BITS;
    const int ICO = (BIC + BICI - 1) / BICI;

    const int OCI = VLEN_BYTES / BICI;
    const int OCO = (OC + OCI - 1) / OCI;

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
