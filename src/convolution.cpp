#include <cmath>
#include <vector>

#include "xnor_nn.h"

#include "utils/logger.hpp"
#include "utils/timer.hpp"

using Logger = xnor_nn::utils::Logger;

namespace {

xnor_nn_status_t fwd_xnor_on_float(
        const float *src, const float *weights, float *dst,
        float alpha, const float *k,
        int MB, int IC, int IH, int IW,
        int OC, int OH, int OW,
        int KH, int KW, int SH, int SW, int PH, int PW) {

#   pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oc = 0; oc < OC; oc++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        int dst_idx = ((mb*OC + oc)*OH + oh)*OW + ow;
        float *d = dst + dst_idx;
        *d = 0.f;
        for (int ic = 0; ic < IC; ic++) {
            for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++) {
                if (oh*SH + kh < (PH > 0 ? PH : 0)) continue;
                if (ow*SW + kw < (PW > 0 ? PW : 0)) continue;

                if (oh*SH + kh >= IH + PH) continue;
                if (ow*SW + kw >= IW + PW) continue;

                const int ih = oh * SH - PH + kh;
                const int iw = ow * SW - PW + kw;

                int src_idx = ((mb*IC + ic)*IH + ih)*IW + iw;
                int weights_idx = ((oc*IC + ic)*KH + kh)*KW + kw;

                // TODO: optimize me
                bool bsrc = src[src_idx] >= 0 ? true : false;
                bool bweights = weights[weights_idx] >= 0 ? true : false;

                float result = !(bsrc ^ bweights);
                *d += result;
            }
        }
        *d *= alpha;
        *d *= k[oh*OW + ow];
    }

    return xnor_nn_success;
}

// TODO: dispatch at init time
xnor_nn_status_t convolution_dispatch(const xnor_nn_convolution_t *s,
        xnor_nn_resources_t res) {

    const float *src = (const float*)res[xnor_nn_resource_bin_src];
    const float *weights = (const float*)res[xnor_nn_resource_bin_weights];
    float *dst = (float*)res[xnor_nn_resource_user_dst];

    const int MB = s->mb;
    const int IW = s->iw;
    const int IH = s->ih;
    const int IC = s->ic;
    const int OW = s->ow;
    const int OH = s->oh;
    const int OC = s->oc;
    const int SW = s->sw;
    const int SH = s->sh;
    const int KW = s->kw;
    const int KH = s->kh;
    const int PW = s->pw;
    const int PH = s->ph;

    xnor_nn_status_t st;

    xnor_nn::utils::Timer timer;
    timer.start();

    switch (s->algorithm) {
    case xnor_nn_algorithm_reference:
    {
        const float alpha = *(float*)(&res[xnor_nn_resource_alpha]);
        const float *k = (float*)res[xnor_nn_resource_k];
        st = fwd_xnor_on_float(src, weights, dst, alpha, k,
                MB, IC, IH, IW, OC, OH, OW, KH, KW, SH, SW, PH, PW);
        break;
    }
    case xnor_nn_algorithm_optimized:
    {
        break;
    }
    }

    timer.stop();
    Logger::info("convolution:", "execute:",
            "MB:", MB, "IC:", IC, "IH:", IH, "IW:", IW,
            "OC:", OC, "OH:", OH, "OW:", OW,
            "KH:", KH, "KW:", KW, "SH:", SH, "SW:", SW, "PH:", PH, "PW:", PW,
            "time:", timer.millis(), "ms");

    return st;
}

}

xnor_nn_status_t xnor_nn_init_convolution(xnor_nn_convolution_t *c,
        const xnor_nn_algorithm_t algorithm,
        int mb, int oc, int ic, int ih, int iw,
        int kh, int kw, int sh, int sw, int ph, int pw) {
    const int oh = (ih + 2*ph - kh) / sh + 1;
    const int ow = (iw + 2*pw - kw) / sw + 1;

    c->algorithm = algorithm;

    c->mb = mb;

    c->iw = iw;
    c->ih = ih;
    c->ic = ic;

    c->ow = ow;
    c->oh = oh;
    c->oc = oc;

    c->sw = sw;
    c->sh = sh;
    c->kw = kw;
    c->kh = kh;
    c->pw = pw;
    c->ph = ph;

    c->forward = convolution_dispatch;

    switch (c->algorithm) {
    case xnor_nn_algorithm_reference: {
        c->size_bin_src = c->mb * c->ic * c->ih * c->iw * sizeof(float);
        c->size_bin_weights = c->oc * c->ic * c->kh * c->kw * sizeof(float);
        c->size_a = c->ih * c->iw * sizeof(float);
        c->size_k = c->oh * c->ow * sizeof(float);
        break;
    }
    case xnor_nn_algorithm_optimized: {
        const size_t BIC = (c->ic + 8 - 1) / 8;
        c->size_bin_src = c->mb * BIC * c->ih * c->iw;
        c->size_bin_weights = c->oc * c->ic * c->kh * c->kw * sizeof(float);
        c->size_a = c->ih * c->iw * sizeof(float);
        c->size_k = c->oh * c->ow * sizeof(float);
        break;
    }
    }

    Logger::info("convolution:", "create:",
            "MB:", mb, "IC:", ic, "IH:", ih, "IW:", iw,
            "OC:", oc, "OH:", oh, "OW:", ow,
            "KH:", kh, "KW:", kw, "SH:", sh, "SW:", sw, "PH:", ph, "PW:", pw);

    return xnor_nn_success;
}
