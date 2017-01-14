#include "bcast_convolution.hpp"

#include "utils.h"

// TODO: log execution

namespace xnor_nn {
namespace implementation {

bool BcastConvolution::isApplicable(
        const xnor_nn_convolution_t *c) const {
    if (c->forward != nullptr) return false;
    if (c->oc % (VLEN / 32) != 0) return false; // TODO constant in base class
    if (c->algorithm != xnor_nn_algorithm_bcast) return false;
    return true;
}

void BcastConvolution::setupConvolution(
        xnor_nn_convolution_t *c) {
    BcastConvolution *op = new BcastConvolution;

    const int VLEN_BYTES = (VLEN / 8);

    const int ELEM_SIZE = sizeof(char);
    const int BITS = ELEM_SIZE * 8;
    const int BIC = ((c->ic + BITS - 1) / BITS) * BITS;

    const int SZ = 8;
    const int BICI = 4;
    const int ABIC = ((BIC + BICI - 1) / BICI) * BICI;

    const int ICO = ((c->ic + BICI - 1) / BICI + SZ - 1) / SZ;

    const int OCI = VLEN_BYTES / BICI;
    const int OCO = (c->oc + OCI - 1) / OCI;

    c->resource_size[xnor_nn_resource_bin_src] =
        c->mb * ABIC * c->ih * c->iw * sizeof(char);
    c->resource_size[xnor_nn_resource_bin_weights] =
        OCO * c->kh * c->kw * ICO * OCI * sizeof(int);
    c->resource_size[xnor_nn_resource_a] = c->ih * c->iw * sizeof(float);
    c->resource_size[xnor_nn_resource_k] = c->oh * c->ow * sizeof(float);

    c->forward = op->exec;

    std::vector<Implementation*> *vec =
        (std::vector<Implementation*>*)c->state;
    vec->push_back(op);
}

BcastConvolution::~BcastConvolution() {}

xnor_nn_status_t BcastConvolution::exec(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    if (
        res[xnor_nn_resource_bin_src] == nullptr
        || res[xnor_nn_resource_bin_weights] == nullptr
        || res[xnor_nn_resource_user_dst] == nullptr
        || res[xnor_nn_resource_k] == nullptr
        || c == nullptr
    ) return xnor_nn_error_invalid_input;
    const unsigned int *src = (unsigned int*)res[xnor_nn_resource_bin_src];
    const unsigned int *weights =
        (unsigned int*)res[xnor_nn_resource_bin_weights];
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

    const int SZ = 8;
    const int VLEN_BYTES = (VLEN / SZ);

    const int BICI = 4;

    const int ELEM_SIZE = sizeof(char);
    const int BITS = ELEM_SIZE * SZ;

    const int BIC = (IC + BITS - 1) / BITS;
    const int ICO = (BIC + BICI - 1) / BICI;

    const int OCI = VLEN_BYTES / BICI;
    const int OCO = (OC + OCI - 1) / OCI;

    // TODO: potentially loops can be reordered
    // TODO: check collapse value for performance
#   pragma omp parallel for collapse(4) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oco = 0; oco < OCO; oco++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        unsigned int d_arr[OCI] = { 0u };
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            const int ih = oh*SH - PH + kh;
            const int iw = ow*SW - PW + kw;

            if (ih < 0 || iw < 0) continue;
            if (ih >= IH || iw >= IW) continue;

            const unsigned int *src_ic =
                src + ((mb*IH + ih)*IW + iw)*ICO;
            const unsigned int *weights_ic_oci =
                weights + ((oco*KH +kh)*KW + kw)*ICO*OCI;

            for (int ico = 0; ico < ICO; ico++)
            for (int oci = 0; oci < OCI; oci++) {
                int src_idx = ico;
                int weights_idx = ico*OCI + oci;

                unsigned int bsrc = src_ic[src_idx];
                unsigned int bweights = weights_ic_oci[weights_idx];

                unsigned int result = ~(bsrc ^ bweights);
                d_arr[oci] += __builtin_popcount(result);
            }
        }
        for (int i = 0; i < OCI; i++)
            dst[((mb*OC + oco*OCI + i)*OH + oh)*OW + ow] =
                d_arr[i] * *alpha * k[oh*OW + ow];
    }

    return xnor_nn_success;
}

} // namespace implementation
} // namespace xnor_nn
