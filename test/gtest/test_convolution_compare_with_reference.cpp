#include "xnor_nn.hpp"

#include "gtest.h"
#include "common.hpp"

typedef struct {
    xnor_nn_algorithm_t algorithm;
    int mb;
    int ic, oc;
    int ih, iw;
    int kh, kw;
    int sh, sw;
    int ph, pw;
    int oh, ow;
} params_t;

static void fill_src(float *d, const params_t &p) {
#   pragma omp parallel for collapse(3) schedule(static)
    for (int mb = 0; mb < p.mb; mb++)
    for (int ic = 0; ic < p.ic; ic++)
    for (int ih = 0; ih < p.ih; ih++)
    for (int iw = 0; iw < p.iw; iw++)
        d[((mb*p.ic + ic)*p.ih + ih)*p.iw + iw] =
            0.33f + 0.0023f*(19 - (iw+ih+ic+mb%37));
}

static void fill_weights(float *d, const params_t &p) {
#   pragma omp parallel for collapse(3) schedule(static)
    for (int oc = 0; oc < p.oc; oc++)
    for (int ic = 0; ic < p.ic; ic++)
    for (int kh = 0; kh < p.kh; kh++)
    for (int kw = 0; kw < p.kw; kw++)
        d[((oc*p.ic + ic)*p.kh + kh)*p.kw + kw] =
            0.18f + 0.017f*(11 - (kw+kh+ic+oc%23));
}

class ConvolutionForward : public ::testing::TestWithParam<params_t> {
protected:
    virtual void SetUp() {
        params_t p = ::testing::TestWithParam<params_t>::GetParam();

        p.oh = (p.ih + 2*p.ph - p.kh) / p.sh + 1;
        p.ow = (p.iw + 2*p.pw - p.kw) / p.sw + 1;

        float *src = new float[p.mb*p.ic*p.ih*p.iw];
        float *weights = new float[p.oc*p.ic*p.kh*p.kw];
        float *dst = new float[p.mb*p.oc*p.oh*p.ow];
        float *ref_dst = new float[p.mb*p.oc*p.oh*p.ow];

        fill_src(src, p);
        fill_weights(weights, p);

        // Optimized
        xnor_nn::Convolution convolution{p.algorithm,
                p.mb, p.oc, p.ic, p.ih, p.iw, p.kh, p.kw,
                p.sh, p.sw, p.ph, p.pw, weights};

        convolution.forward(src, dst);

        // Reference
        xnor_nn::Convolution reference_convolution{xnor_nn_algorithm_reference,
                p.mb, p.oc, p.ic, p.ih, p.iw, p.kh, p.kw,
                p.sh, p.sw, p.ph, p.pw, weights};

        reference_convolution.forward(src, ref_dst);

        xnor_nn::test::check_4d(p.mb, p.oc, p.oh, p.ow, dst, ref_dst);

        delete[] src;
        delete[] weights;
        delete[] dst;
        delete[] ref_dst;
    }
};

TEST_P(ConvolutionForward, compare_with_reference)
{
}

// mb ic oc ih iw kh kw sh sw ph pw
INSTANTIATE_TEST_CASE_P(ConvolutionBcastTask,
        ConvolutionForward, ::testing::Values(
params_t{ xnor_nn_algorithm_bcast, 1, 1, 32, 60, 61, 3, 3, 1, 1, 0, 0 },
params_t{ xnor_nn_algorithm_bcast, 256, 1, 32, 60, 61, 3, 3, 1, 1, 0, 0 },
params_t{ xnor_nn_algorithm_bcast, 1, 32, 32, 20, 20, 3, 3, 1, 1, 0, 0 },
params_t{ xnor_nn_algorithm_bcast, 256, 32, 32, 20, 20, 3, 3, 1, 1, 0, 0 }
));
INSTANTIATE_TEST_CASE_P(ConvolutionBcastSmall,
        ConvolutionForward, ::testing::Values(
params_t{ xnor_nn_algorithm_bcast, 3, 24, 32, 27, 27, 5, 5, 1, 1, 2, 2 },
params_t{ xnor_nn_algorithm_bcast, 3, 25, 32, 27, 27, 5, 5, 1, 1, 2, 2 },
params_t{ xnor_nn_algorithm_bcast, 3, 26, 32, 27, 27, 5, 5, 1, 1, 2, 2 },
params_t{ xnor_nn_algorithm_bcast, 3, 24, 24, 27, 27, 5, 5, 1, 1, 2, 2 },
params_t{ xnor_nn_algorithm_bcast, 3, 24, 24, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_bcast, 3, 32, 32, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_bcast, 3, 24, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_bcast, 3, 24, 16, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_bcast, 3, 8, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_bcast, 3, 8, 16, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_bcast, 3, 9, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_bcast, 3, 10, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_bcast, 3, 11, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_bcast, 3, 12, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_bcast, 3, 13, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_bcast, 3, 14, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_bcast, 3, 15, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_bcast, 3, 16, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_bcast, 3, 3, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_bcast, 3, 4, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_bcast, 3, 5, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_bcast, 3, 6, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_bcast, 3, 7, 8, 13, 13, 3, 3, 1, 1, 1, 1 }
));
INSTANTIATE_TEST_CASE_P(ConvolutionBcastAlexNet,
        ConvolutionForward, ::testing::Values(
params_t{ xnor_nn_algorithm_bcast, 2, 3, 96, 227, 227, 11, 11, 4, 4, 0, 0 },
params_t{ xnor_nn_algorithm_bcast, 2, 96, 256, 27, 27, 5, 5, 1, 1, 2, 2 },
params_t{ xnor_nn_algorithm_bcast, 2, 256, 384, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_bcast, 2, 384, 384, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_bcast, 2, 384, 256, 13, 13, 3, 3, 1, 1, 1, 1 }
));
INSTANTIATE_TEST_CASE_P(ConvolutionDirectAlexNet,
        ConvolutionForward, ::testing::Values(
params_t{ xnor_nn_algorithm_direct, 2, 3, 96, 227, 227, 11, 11, 4, 4, 0, 0 },
params_t{ xnor_nn_algorithm_direct, 2, 96, 256, 27, 27, 5, 5, 1, 1, 2, 2 },
params_t{ xnor_nn_algorithm_direct, 2, 256, 384, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_direct, 2, 384, 384, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_direct, 2, 384, 256, 13, 13, 3, 3, 1, 1, 1, 1 }
));
