#include "bcast_convolution.hpp"

#include "logger.hpp"
#include "xnor_nn_types.h"

namespace xnor_nn {
namespace implementation {

xnor_nn_status_t BcastConvolution::binarize_data(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    if (
        res[xnor_nn_resource_user_src] == nullptr
        || res[xnor_nn_resource_bin_src] == nullptr
        || c == nullptr
    ) return xnor_nn_error_invalid_input;
    const unsigned int *from = (unsigned int*)res[xnor_nn_resource_user_src];
    unsigned char *to = (unsigned char*)res[xnor_nn_resource_bin_src];

    const int MB = c->mb;
    const int IC = c->ic;
    const int IH = c->ih;
    const int IW = c->iw;

    BcastConvolution *state = reinterpret_cast<BcastConvolution*>(getState(c));

    const int BIC = state->BIC;
    const int ABIC = state->ABIC;

    /*
    const int BIC = (c->ic + 8 - 1) / 8;
    const int ABIC = ((BIC + 4 - 1) / 4) * 4;
    */

    LOG_INFO("binarize_data:\t", "execute:",
            "[", MB, "][", IC, "][", IH, "][", IW, "]",
            "->",
            "[", MB, "][", IH, "][", IW, "][", ABIC, "(bytes)", "]",
            "Algorithm:", "bcast");

#   pragma omp parallel for collapse(3) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int ih = 0; ih < IH; ih++)
    for (int iw = 0; iw < IW; iw++) {
        for (int bic = 0; bic < BIC; bic++) {
            unsigned char out{0};
            int LEN = bic == BIC - 1 ? (IC % SZ) : SZ;
            if (LEN == 0) LEN = SZ;
            for (int ic = 0; ic < LEN; ic++) {
                int from_idx = ((mb*IC + bic*SZ + ic)*IH + ih)*IW + iw;
                char tmp = (~from[from_idx]) >> 31;
                out <<= 1;
                out |= tmp;
            }
            if (LEN != SZ) out <<= SZ-LEN;
            int to_idx = ((mb*IH + ih)*IW + iw)*ABIC + bic;
            to[to_idx] = out;
        }
        for (int r = 0; r < ABIC - BIC; r++) {
            int to_idx = ((mb*IH + ih)*IW + iw)*ABIC + BIC + r;
            to[to_idx] = 0x00u;
        }
    }

    return xnor_nn_success;
}

} // namespace implementation
} // namespace xnor_nn
