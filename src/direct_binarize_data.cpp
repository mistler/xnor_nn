#include "binarize_data.h"

xnor_nn_status_t direct_binarize_char(const unsigned int *from, unsigned char *to,
        int MB, int IC, int IH, int IW) {
    const int SZ = 8;
    const int OC = (IC + SZ - 1) / SZ;

#   pragma omp parallel for collapse(3) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int ih = 0; ih < IH; ih++)
    for (int iw = 0; iw < IW; iw++)
    for (int oc = 0; oc < OC; oc++) {
        unsigned char out{0};
        const int LEN = oc == OC - 1 ? (IC % SZ) : SZ;
        for (int ic = 0; ic < LEN; ic++) {
            int from_idx = (ic*IH + ih)*IW + iw;
            char tmp = (~from[from_idx]) >> 31;
            out <<= 1;
            out |= tmp;
        }
        if (LEN != SZ) out <<= SZ-LEN;
        int to_idx = (ih*IW + iw)*OC + oc;
        to[to_idx] = out;
    }

    return xnor_nn_success;
}
