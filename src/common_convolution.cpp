#include <cmath>
#include <vector>

#include "xnor_nn.h"

#include "implementation.hpp"

#include "utils/logger.hpp"
#include "utils/timer.hpp"

#ifdef __x86_64__
#define VLEN 16
#elif defined __arm__
#define VLEN 16
#elif
#define VLEN 4
#endif

using Logger = xnor_nn::utils::Logger;

namespace {

// TODO: dispatch at init time
xnor_nn_status_t dispatch_binarize_weights(const xnor_nn_convolution_t *c,
        xnor_nn_resources_t res) {
    xnor_nn_executor_t ex = get_executor(c->algorithm,
            operation_binarize_weights);

    xnor_nn::utils::Timer timer;
    timer.start();

    xnor_nn_status_t st = ex(c, res);

    timer.stop();
    Logger::info("Binarize weights:", "MB:", c->mb, "IC:", c->ic,
            "IH:", c->ih, "IW:", c->iw,
            "OC:", c->oc, "OH:", c->oh, "OW:", c->ow,
            "KH:", c->kh, "KW:", c->kw, "SH:", c->sh, "SW:", c->sw,
            "PH:", c->ph, "PW:", c->pw,
            "execute:", "time:", timer.millis(), "ms");

    return st;
}

xnor_nn_status_t dispatch_binarize_data(const xnor_nn_convolution_t *c,
        xnor_nn_resources_t res) {
    xnor_nn_executor_t ex = get_executor(c->algorithm, operation_binarize_data);

    xnor_nn::utils::Timer timer;
    timer.start();

    xnor_nn_status_t st = ex(c, res);

    timer.stop();
    Logger::info("Binarize data:", "MB:", c->mb, "IC:", c->ic,
            "IH:", c->ih, "IW:", c->iw,
            "OC:", c->oc, "OH:", c->oh, "OW:", c->ow,
            "KH:", c->kh, "KW:", c->kw, "SH:", c->sh, "SW:", c->sw,
            "PH:", c->ph, "PW:", c->pw,
            "execute:", "time:", timer.millis(), "ms");

    return st;
}

xnor_nn_status_t dispatch_calculate_k(const xnor_nn_convolution_t *c,
        xnor_nn_resources_t res) {
    xnor_nn_executor_t ex = get_executor(c->algorithm, operation_calculate_k);

    xnor_nn::utils::Timer timer;
    timer.start();

    xnor_nn_status_t st = ex(c, res);

    timer.stop();
    Logger::info("Calculate K:", "MB:", c->mb, "IC:", c->ic,
            "IH:", c->ih, "IW:", c->iw,
            "OC:", c->oc, "OH:", c->oh, "OW:", c->ow,
            "KH:", c->kh, "KW:", c->kw, "SH:", c->sh, "SW:", c->sw,
            "PH:", c->ph, "PW:", c->pw,
            "execute:", "time:", timer.millis(), "ms");

    return st;

}

xnor_nn_status_t dispatch_forward(const xnor_nn_convolution_t *c,
        xnor_nn_resources_t res) {

    xnor_nn_executor_t ex = get_executor(c->algorithm,
            operation_convolution_forward);

    xnor_nn::utils::Timer timer;
    timer.start();

    xnor_nn_status_t st = ex(c, res);

    timer.stop();
    Logger::info("Convolution forward:", "MB:", c->mb, "IC:", c->ic,
            "IH:", c->ih, "IW:", c->iw,
            "OC:", c->oc, "OH:", c->oh, "OW:", c->ow,
            "KH:", c->kh, "KW:", c->kw, "SH:", c->sh, "SW:", c->sw,
            "PH:", c->ph, "PW:", c->pw,
            "execute:", "time:", timer.millis(), "ms");

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

    c->aic = ic;
    c->bic = ic;

    switch (c->algorithm) {
    case xnor_nn_algorithm_reference: {
        const size_t ELEM_SIZE = sizeof(float);
        const size_t VEC_LENGTH = 1;

        c->sizeof_element = ELEM_SIZE;
        c->vector_length = VEC_LENGTH;

        c->resource_size[xnor_nn_resource_bin_src] =
            c->mb * c->ic * c->ih * c->iw * ELEM_SIZE;
        c->resource_size[xnor_nn_resource_bin_weights] =
            c->oc * c->ic * c->kh * c->kw * ELEM_SIZE;
        c->resource_size[xnor_nn_resource_a] = c->ih * c->iw * sizeof(float);
        c->resource_size[xnor_nn_resource_k] = c->oh * c->ow * sizeof(float);
        break;
    }
    case xnor_nn_algorithm_optimized: {
        const size_t ELEM_SIZE = sizeof(char);
        const size_t BITS = ELEM_SIZE * 8;
        const size_t VEC_LENGTH = VLEN;
        const size_t BIC = (c->ic + BITS - 1) / BITS;

        c->sizeof_element = ELEM_SIZE;
        c->vector_length = VEC_LENGTH;
        c->bic = BIC;
        c->aic = BIC + (VEC_LENGTH - (BIC % VEC_LENGTH));

        c->resource_size[xnor_nn_resource_bin_src] =
            c->mb * c->aic * c->ih * c->iw * ELEM_SIZE;
        c->resource_size[xnor_nn_resource_bin_weights] =
            c->oc * c->aic * c->kh * c->kw * ELEM_SIZE;
        c->resource_size[xnor_nn_resource_a] = c->ih * c->iw * sizeof(float);
        c->resource_size[xnor_nn_resource_k] = c->oh * c->ow * sizeof(float);
        break;
    }
    }

    c->binarize_weights = dispatch_binarize_weights;
    c->binarize_data = dispatch_binarize_data;
    c->calculate_k = dispatch_calculate_k;
    c->forward = dispatch_forward;

    Logger::info("convolution:", "create:",
            "MB:", mb, "IC:", ic, "IH:", ih, "IW:", iw,
            "OC:", oc, "OH:", oh, "OW:", ow,
            "KH:", kh, "KW:", kw, "SH:", sh, "SW:", sw, "PH:", ph, "PW:", pw);

    return xnor_nn_success;
}
