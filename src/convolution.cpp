#include "xnor_nn.h"

static xnor_nn_status_t fwd_xnor_on_float(const void *s,
        const void *src_, const void *weights_, void *dst_) {
    const xnor_nn_convolution_t *self = (const xnor_nn_convolution_t*)s;

    const float *src = (const float*)src_;
    const float *weights = (const float*)weights_;
    float *dst = (float*)dst_;

    const int MB = self->mb;
    const int IW = self->src[0];
    const int IH = self->src[1];
    const int IC = self->src[2];
    const int OW = self->dst[0];
    const int OH = self->dst[1];
    const int OC = self->dst[2];
    const int SW = self->stride[0];
    const int SH = self->stride[1];
    const int KW = self->kernel[0];
    const int KH = self->kernel[1];
    const int PW = self->padding[0];
    const int PH = self->padding[1];

#   pragma omp parallel for collapse(2)
    for (int mb = 0; mb < MB; ++mb)
    for (int oc = 0; oc < OC; ++oc)
    for (int oh = 0; oh < OH; ++oh)
    for (int ow = 0; ow < OW; ++ow) {
        int dst_idx = ((mb*OC + oc)*OH + oh)*OW + ow;
        float *d = dst + dst_idx;
        *d = .0f;
        for (int ic = 0; ic < IC; ++ic)
        for (int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw){
            if (oh*SH + kh < (PH > 0 ? PH : 0)) continue;
            if (ow*SW + kw < (PW > 0 ? PH : 0)) continue;

            if (oh*SH + kh >= IH + PH) continue;
            if (ow*SW + kw >= IW + PW) continue;

            const int ih = oh * SH - PH + kh;
            const int iw = ow * SW - PW + kw;

            int src_idx = ((mb*IC + ic)*IH + ih)*IW + iw;
            int weights_idx = ((oc*IC + ic)*KH + kh)*KW + kw;

            bool bsrc = src[src_idx] >= 0 ? true : false;
            bool bweights = weights[weights_idx] >= 0 ? true : false;

            *d += ~(bsrc ^ bweights);
        }
    }

    return xnor_nn_success;
}

xnor_nn_status_t xnor_nn_init_convolution(xnor_nn_convolution_t *c,
        int mb, int oc, int ic, int ih, int iw,
        int kh, int kw, int sh, int sw, int ph, int pw) {
    const int oh = (ih + 2*ph - kh) / sh + 1;
    const int ow = (iw + 2*pw - kw) / sw + 1;

    c->mb = mb;

    c->src[0] = iw;
    c->src[1] = ih;
    c->src[2] = ic;

    c->dst[0] = ow;
    c->dst[1] = oh;
    c->dst[2] = oc;

    c->stride[0] = sw;
    c->stride[1] = sh;

    c->kernel[0] = kw;
    c->kernel[1] = kh;

    c->padding[0] = pw;
    c->padding[1] = ph;

    c->forward = fwd_xnor_on_float;

    return xnor_nn_success;
}
