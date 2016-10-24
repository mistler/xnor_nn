#include <cmath>
#include <vector>

#include "xnor_nn.h"

#include "utils/logger.hpp"
#include "utils/timer.hpp"

using Logger = xnor_nn::utils::Logger;

namespace {

xnor_nn_status_t fwd_xnor_on_float(
        const float *src, const float *weights, float *dst,
        int MB, int IC, int IH, int IW,
        int OC, int OH, int OW,
        int KH, int KW, int SH, int SW, int PH, int PW,
        float alpha, const float *k) {

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
        const void *src_, const void *weights_, void *dst_) {

    const float *src = (const float*)src_;
    const float *weights = (const float*)weights_;
    float *dst = (float*)dst_;

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

    const float alpha = src[MB*IC*IH*IW];
    const float *k = src + MB*IC*IH*IW + IH*IW;

    xnor_nn::utils::Timer timer;
    timer.start();

    xnor_nn_status_t st = fwd_xnor_on_float(src, weights, dst,
            MB, IC, IH, IW, OC, OH, OW, KH, KW, SH, SW, PH, PW, alpha, k);

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
        int mb, int oc, int ic, int ih, int iw,
        int kh, int kw, int sh, int sw, int ph, int pw) {
    const int oh = (ih + 2*ph - kh) / sh + 1;
    const int ow = (iw + 2*pw - kw) / sw + 1;

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

    Logger::info("convolution:", "create:",
            "MB:", mb, "IC:", ic, "IH:", ih, "IW:", iw,
            "OC:", oc, "OH:", oh, "OW:", ow,
            "KH:", kh, "KW:", kw, "SH:", sh, "SW:", sw, "PH:", ph, "PW:", pw);

    return xnor_nn_success;
}
