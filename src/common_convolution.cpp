#include <cmath>
#include <vector>

#include "xnor_nn.h"

#include "binarize_data.h"
#include "binarize_weights.h"
#include "convolution_forward.h"

#include "utils/logger.hpp"
#include "utils/timer.hpp"

using Logger = xnor_nn::utils::Logger;

namespace {

// TODO: dispatch at init time
xnor_nn_status_t dispatch_binarize_weights(const xnor_nn_convolution_t *c,
        xnor_nn_resources_t res) {
    const int OC = c->oc;
    const int IC = c->ic;
    const int KH = c->kh;
    const int KW = c->kw;

    xnor_nn_status_t st = xnor_nn_unimplemented;

    xnor_nn::utils::Timer timer;
    timer.start();

    switch (c->algorithm) {
    case xnor_nn_algorithm_reference:
    {
        const float *f = (float*)res[xnor_nn_resource_user_weights];
        float *t = (float*)res[xnor_nn_resource_bin_weights];
        float *alpha = (float*)&(res[xnor_nn_resource_alpha]);
        st = reference_weights_copy_on_float(f, t, alpha, OC, IC, KH, KW);
        break;
    }
    case xnor_nn_algorithm_optimized:
    {
        const float *f = (float*)res[xnor_nn_resource_user_weights];
        unsigned char *t = (unsigned char*)res[xnor_nn_resource_bin_weights];
        float *alpha = (float*)&(res[xnor_nn_resource_alpha]);
        st = direct_binarize_weights_char(f, t, alpha, OC, IC, KH, KW);
        break;
    }
    }

    timer.stop();
    Logger::info("weights_binarizer:", "execute:",
            "OC:", OC, "IC:", IC, "KH:", KH, "KW:", KW,
            "time:", timer.millis(), "ms");

    return st;
}

xnor_nn_status_t dispatch_binarize_data(const xnor_nn_convolution_t *c,
        xnor_nn_resources_t res) {
    const int MB = c->mb;
    const int IC = c->ic;
    const int IH = c->ih;
    const int IW = c->iw;

    xnor_nn_status_t st = xnor_nn_unimplemented;

    xnor_nn::utils::Timer timer;
    timer.start();

    switch (c->algorithm) {
    case xnor_nn_algorithm_reference:
    {
        const float *from = (float*)res[xnor_nn_resource_user_src];
        float *to = (float*)res[xnor_nn_resource_bin_src];
        st = reference_data_copy_on_float(from, to, MB, IC, IH, IW);
        break;
    }
    case xnor_nn_algorithm_optimized:
    {
        const unsigned int *from =
            (unsigned int*)res[xnor_nn_resource_user_src];
        unsigned char *to = (unsigned char*)res[xnor_nn_resource_bin_src];
        st = direct_binarize_char(from, to, MB, IC, IH, IW);
        break;
    }
    }

    timer.stop();
    Logger::info("data_binarizer:", "binarize:",
            "MB:", MB, "IC:", IC, "IH:", IH, "IW:", IW,
            "time:", timer.millis(), "ms");

    return st;
}

xnor_nn_status_t dispatch_calculate_k(const xnor_nn_convolution_t *c,
        xnor_nn_resources_t res) {
    const int MB = c->mb;
    const int IC = c->ic;
    const int IH = c->ih;
    const int IW = c->iw;

    const int OH = c->oh;
    const int OW = c->ow;
    const int KH = c->kh;
    const int KW = c->kw;
    const int SH = c->sh;
    const int SW = c->sw;
    const int PH = c->ph;
    const int PW = c->pw;

    xnor_nn_status_t st = xnor_nn_unimplemented;

    xnor_nn::utils::Timer timer;
    timer.start();

    switch (c->algorithm) {
    case xnor_nn_algorithm_reference:
    case xnor_nn_algorithm_optimized:
    {
        const float *from = (float*)res[xnor_nn_resource_user_src];
        float *a_ptr = (float*)res[xnor_nn_resource_a];
        float *k_ptr = (float*)res[xnor_nn_resource_k];
        st = reference_calculate_k(from, a_ptr, k_ptr,
            MB, IC, IH, IW, OH, OW, KH, KW, SH, SW, PH, PW);
        break;
    }
    }

    timer.stop();
    Logger::info("data_binarizer:", "calculate_k:",
            "MB:", MB, "IC:", IC, "IH:", IH, "IW:", IW, "OH:", OH, "OW:", OW,
            "KH:", KH, "KW:", KW, "SH:", SH, "SW:", SW, "PH:", PH, "PW:", PW,
            "time:", timer.millis(), "ms");

    return st;
}

xnor_nn_status_t dispatch_forward(const xnor_nn_convolution_t *c,
        xnor_nn_resources_t res) {
    const float *src = (const float*)res[xnor_nn_resource_bin_src];
    const float *weights = (const float*)res[xnor_nn_resource_bin_weights];
    float *dst = (float*)res[xnor_nn_resource_user_dst];

    const int MB = c->mb;
    const int IW = c->iw;
    const int IH = c->ih;
    const int IC = c->ic;
    const int OW = c->ow;
    const int OH = c->oh;
    const int OC = c->oc;
    const int SW = c->sw;
    const int SH = c->sh;
    const int KW = c->kw;
    const int KH = c->kh;
    const int PW = c->pw;
    const int PH = c->ph;

    xnor_nn_status_t st = xnor_nn_unimplemented;

    xnor_nn::utils::Timer timer;
    timer.start();

    switch (c->algorithm) {
    case xnor_nn_algorithm_reference:
    case xnor_nn_algorithm_optimized:
    {
        const float alpha = *(float*)(&res[xnor_nn_resource_alpha]);
        const float *k = (float*)res[xnor_nn_resource_k];
        st = reference_convolution_forward(src, weights, dst, alpha, k,
                MB, IC, IH, IW, OC, OH, OW, KH, KW, SH, SW, PH, PW);
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

    c->binarize_weights = dispatch_binarize_weights;
    c->binarize_data = dispatch_binarize_data;
    c->calculate_k = dispatch_calculate_k;
    c->forward = dispatch_forward;

    switch (c->algorithm) {
    case xnor_nn_algorithm_reference: {
        c->resource_size[xnor_nn_resource_bin_src] =
            c->mb * c->ic * c->ih * c->iw * sizeof(float);
        c->resource_size[xnor_nn_resource_bin_weights] =
            c->oc * c->ic * c->kh * c->kw * sizeof(float);
        c->resource_size[xnor_nn_resource_a] = c->ih * c->iw * sizeof(float);
        c->resource_size[xnor_nn_resource_k] = c->oh * c->ow * sizeof(float);
        break;
    }
    case xnor_nn_algorithm_optimized: {
        const size_t BIC = (c->ic + 8 - 1) / 8;
        c->resource_size[xnor_nn_resource_bin_src] =
            c->mb * BIC * c->ih * c->iw;
        c->resource_size[xnor_nn_resource_bin_weights] =
            c->oc * BIC * c->kh * c->kw * sizeof(float);
        c->resource_size[xnor_nn_resource_a] = c->ih * c->iw * sizeof(float);
        c->resource_size[xnor_nn_resource_k] = c->oh * c->ow * sizeof(float);
        break;
    }
    }

    Logger::info("convolution:", "create:",
            "MB:", mb, "IC:", ic, "IH:", ih, "IW:", iw,
            "OC:", oc, "OH:", oh, "OW:", ow,
            "KH:", kh, "KW:", kw, "SH:", sh, "SW:", sw, "PH:", ph, "PW:", pw);

    return xnor_nn_success;
}
