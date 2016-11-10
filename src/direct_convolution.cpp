#include "implementation.hpp"

xnor_nn_status_t direct_convolution_forward(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    const unsigned int *src = (unsigned int*)res[xnor_nn_resource_bin_src];
    const unsigned int *weights =
        (unsigned int*)res[xnor_nn_resource_bin_weights];
    float *dst = (float*)res[xnor_nn_resource_user_dst];
    float *alpha = (float*)&res[xnor_nn_resource_alpha];
    const float *k = (float*)res[xnor_nn_resource_k];

    const int MB = c->mb;
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

    const int AIC = c->aic;
    const int VEC_LENGTH = c->vector_length;
    const int ELEM_SIZE = 4;

    const int VECTORS_IN_AIC = AIC / VEC_LENGTH;

    // TODO: potentially loops can be reordered
#   pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++)
    for (int oc = 0; oc < OC; oc++) {
        int dst_idx = ((mb*OC + oc)*OH + oh)*OW + ow;
        float *d = dst + dst_idx;
        *d = 0.f;
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            if (oh*SH + kh < (PH > 0 ? PH : 0)) continue;
            if (ow*SW + kw < (PW > 0 ? PW : 0)) continue;

            if (oh*SH + kh >= IH + PH) continue;
            if (ow*SW + kw >= IW + PW) continue;

            const int ih = oh * SH - PH + kh;
            const int iw = ow * SW - PW + kw;

            const unsigned int *src_ic =
                src + ((mb*IH + ih)*IW + iw)*AIC/ELEM_SIZE;
            const unsigned int *weights_ic =
                weights + ((kh*KW + kw)*OC + oc)*AIC/ELEM_SIZE;

            for (int aic = 0; aic < VECTORS_IN_AIC; aic++)
            for (int v = 0; v < VEC_LENGTH / ELEM_SIZE; v++) {
                int src_idx = aic*VEC_LENGTH/ELEM_SIZE + v;
                int weights_idx = aic*VEC_LENGTH/ELEM_SIZE + v;

                unsigned int bsrc = src_ic[src_idx];
                unsigned int bweights = weights_ic[weights_idx];

                unsigned int result = ~(bsrc ^ bweights);
                *d += __builtin_popcount(result);
            }
        }
        *d *= *alpha;
        *d *= k[oh*OW + ow];
    }

    return xnor_nn_success;
}
