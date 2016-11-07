#include "implementation.hpp"

xnor_nn_status_t direct_binarize_data_char(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    const unsigned int *from = (unsigned int*)res[xnor_nn_resource_user_src];
    unsigned char *to = (unsigned char*)res[xnor_nn_resource_bin_src];

    const int MB = c->mb;
    const int IC = c->ic;
    const int IH = c->ih;
    const int IW = c->iw;

    const int SZ = c->sizeof_element * 8;
    const int BIC = c->bic;
    const int AIC = c->aic;

#   pragma omp parallel for collapse(3) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int ih = 0; ih < IH; ih++)
    for (int iw = 0; iw < IW; iw++) {
        for (int bic = 0; bic < BIC; bic++) {
            unsigned char out{0};
            const int LEN = bic == BIC - 1 ? (IC % SZ) : SZ;
            for (int ic = 0; ic < LEN; ic++) {
                int from_idx = ((mb*IC + ic)*IH + ih)*IW + iw;
                char tmp = (~from[from_idx]) >> 31;
                out <<= 1;
                out |= tmp;
            }
            if (LEN != SZ) out <<= SZ-LEN;
            int to_idx = ((mb*IH + ih)*IW + iw)*AIC + bic;
            to[to_idx] = out;
        }
        for (int r = 0; r < AIC - BIC; r++) {
            int to_idx = ((mb*IH + ih)*IW + iw)*AIC + BIC + r;
            to[to_idx] = 0x00u;
        }
    }

    return xnor_nn_success;
}
