#include "gtest.h"

#include "common.hpp"

#include "xnor_nn.hpp"

using namespace xnor_nn::test;

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
            p.fmt.src, p.fmt.weights, p.fmt.dst,
            p.mb, p.oc, p.ic, p.ih, p.iw, p.kh, p.kw,
            p.sh, p.sw, p.ph, p.pw, weights};

        convolution.forward(src, dst);

        // Reference
        xnor_nn::Convolution reference_convolution{xnor_nn_algorithm_reference,
            p.fmt.src, p.fmt.weights, p.fmt.dst,
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

auto bcast = xnor_nn_algorithm_bcast;

auto fmt_caffe = tensor_fmt_t{xnor_nn_data_format_nchw,
    xnor_nn_weights_format_oihw, xnor_nn_data_format_nchw};
auto fmt_tf = tensor_fmt_t{xnor_nn_data_format_nchw,
    xnor_nn_weights_format_oihw, xnor_nn_data_format_nchw};

// mb ic oc ih iw kh kw sh sw ph pw
INSTANTIATE_TEST_CASE_P(ConvolutionBcastTaskCaffe,
        ConvolutionForward, ::testing::Values(
params_t{ bcast, fmt_caffe, 1, 1, 32, 60, 61, 3, 3, 1, 1, 0, 0 },
params_t{ bcast, fmt_caffe, 64, 1, 32, 60, 61, 3, 3, 1, 1, 0, 0 },
params_t{ bcast, fmt_caffe, 1, 32, 32, 20, 20, 3, 3, 1, 1, 0, 0 },
params_t{ bcast, fmt_caffe, 64, 32, 32, 20, 20, 3, 3, 1, 1, 0, 0 }
));
INSTANTIATE_TEST_CASE_P(ConvolutionBcastTaskTf,
        ConvolutionForward, ::testing::Values(
params_t{ bcast, fmt_tf, 1, 1, 32, 60, 61, 3, 3, 1, 1, 0, 0 },
params_t{ bcast, fmt_tf, 64, 1, 32, 60, 61, 3, 3, 1, 1, 0, 0 },
params_t{ bcast, fmt_tf, 1, 32, 32, 20, 20, 3, 3, 1, 1, 0, 0 },
params_t{ bcast, fmt_tf, 64, 32, 32, 20, 20, 3, 3, 1, 1, 0, 0 }
));
INSTANTIATE_TEST_CASE_P(ConvolutionBcastSmallCaffe,
        ConvolutionForward, ::testing::Values(
params_t{ bcast, fmt_caffe, 3, 24, 32, 27, 27, 5, 5, 1, 1, 2, 2 },
params_t{ bcast, fmt_caffe, 3, 25, 32, 27, 27, 5, 5, 1, 1, 2, 2 },
params_t{ bcast, fmt_caffe, 3, 26, 32, 27, 27, 5, 5, 1, 1, 2, 2 },
params_t{ bcast, fmt_caffe, 3, 24, 24, 27, 27, 5, 5, 1, 1, 2, 2 },
params_t{ bcast, fmt_caffe, 3, 24, 24, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 3, 32, 32, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 3, 24, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 3, 24, 16, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 3, 8, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 3, 8, 16, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 3, 9, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 3, 10, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 3, 11, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 3, 12, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 3, 13, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 3, 14, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 3, 15, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 3, 16, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 3, 3, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 3, 4, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 3, 5, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 3, 6, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 3, 7, 8, 13, 13, 3, 3, 1, 1, 1, 1 }
));
INSTANTIATE_TEST_CASE_P(ConvolutionBcastSmallTf,
        ConvolutionForward, ::testing::Values(
params_t{ bcast, fmt_tf, 3, 24, 32, 27, 27, 5, 5, 1, 1, 2, 2 },
params_t{ bcast, fmt_tf, 3, 25, 32, 27, 27, 5, 5, 1, 1, 2, 2 },
params_t{ bcast, fmt_tf, 3, 26, 32, 27, 27, 5, 5, 1, 1, 2, 2 },
params_t{ bcast, fmt_tf, 3, 24, 24, 27, 27, 5, 5, 1, 1, 2, 2 },
params_t{ bcast, fmt_tf, 3, 24, 24, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 3, 32, 32, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 3, 24, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 3, 24, 16, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 3, 8, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 3, 8, 16, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 3, 9, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 3, 10, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 3, 11, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 3, 12, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 3, 13, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 3, 14, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 3, 15, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 3, 16, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 3, 3, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 3, 4, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 3, 5, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 3, 6, 8, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 3, 7, 8, 13, 13, 3, 3, 1, 1, 1, 1 }
));
INSTANTIATE_TEST_CASE_P(ConvolutionBcastAlexNetCaffe,
        ConvolutionForward, ::testing::Values(
params_t{ bcast, fmt_caffe, 2, 3, 96, 227, 227, 11, 11, 4, 4, 0, 0 },
params_t{ bcast, fmt_caffe, 2, 96, 256, 27, 27, 5, 5, 1, 1, 2, 2 },
params_t{ bcast, fmt_caffe, 2, 256, 384, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 2, 384, 384, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 2, 384, 256, 13, 13, 3, 3, 1, 1, 1, 1 }
));
INSTANTIATE_TEST_CASE_P(ConvolutionBcastAlexNetTf,
        ConvolutionForward, ::testing::Values(
params_t{ bcast, fmt_tf, 2, 3, 96, 227, 227, 11, 11, 4, 4, 0, 0 },
params_t{ bcast, fmt_tf, 2, 96, 256, 27, 27, 5, 5, 1, 1, 2, 2 },
params_t{ bcast, fmt_tf, 2, 256, 384, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 2, 384, 384, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 2, 384, 256, 13, 13, 3, 3, 1, 1, 1, 1 }
));
INSTANTIATE_TEST_CASE_P(ConvolutionBcastAlexNetOutOfTemplateCaffe,
        ConvolutionForward, ::testing::Values(
params_t{ bcast, fmt_caffe, 2, 3, 64, 227, 227, 11, 11, 4, 4, 0, 0 },
params_t{ bcast, fmt_caffe, 2, 96, 128, 27, 27, 5, 5, 1, 1, 2, 2 },
params_t{ bcast, fmt_caffe, 2, 256, 256, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 2, 12, 128, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 2, 384, 64, 13, 13, 3, 3, 1, 1, 1, 1 }
));
INSTANTIATE_TEST_CASE_P(ConvolutionBcastAlexNetOutOfTemplateTf,
        ConvolutionForward, ::testing::Values(
params_t{ bcast, fmt_tf, 2, 3, 64, 227, 227, 11, 11, 4, 4, 0, 0 },
params_t{ bcast, fmt_tf, 2, 96, 128, 27, 27, 5, 5, 1, 1, 2, 2 },
params_t{ bcast, fmt_tf, 2, 256, 256, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 2, 12, 128, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 2, 384, 64, 13, 13, 3, 3, 1, 1, 1, 1 }
));
INSTANTIATE_TEST_CASE_P(ConvolutionBcastCifar10_tmpCaffe,
        ConvolutionForward, ::testing::Values(
params_t{ bcast, fmt_caffe, 2, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 2, 128, 256, 16, 16, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 2, 256, 256, 16, 16, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 2, 256, 512, 8, 8, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 2, 512, 512, 8, 8, 3, 3, 1, 1, 1, 1 }
));
INSTANTIATE_TEST_CASE_P(ConvolutionBcastCifar10_tmpTf,
        ConvolutionForward, ::testing::Values(
params_t{ bcast, fmt_tf, 2, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 2, 128, 256, 16, 16, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 2, 256, 256, 16, 16, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 2, 256, 512, 8, 8, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 2, 512, 512, 8, 8, 3, 3, 1, 1, 1, 1 }
));
INSTANTIATE_TEST_CASE_P(ConvolutionBcastCifar10Caffe,
        ConvolutionForward, ::testing::Values(
params_t{ bcast, fmt_caffe, 2, 3, 32, 32, 32, 5, 5, 1, 1, 2, 2 },
params_t{ bcast, fmt_caffe, 2, 32, 32, 32, 32, 5, 5, 1, 1, 2, 2 },
params_t{ bcast, fmt_caffe, 2, 32, 64, 32, 32, 5, 5, 1, 1, 2, 2 }
));
INSTANTIATE_TEST_CASE_P(ConvolutionBcastCifar10Tf,
        ConvolutionForward, ::testing::Values(
params_t{ bcast, fmt_tf, 2, 3, 32, 32, 32, 5, 5, 1, 1, 2, 2 },
params_t{ bcast, fmt_tf, 2, 32, 32, 32, 32, 5, 5, 1, 1, 2, 2 },
params_t{ bcast, fmt_tf, 2, 32, 64, 32, 32, 5, 5, 1, 1, 2, 2 }
));
INSTANTIATE_TEST_CASE_P(ConvolutionBcastMnistCaffe,
        ConvolutionForward, ::testing::Values(
params_t{ bcast, fmt_caffe, 2, 1, 24, 28, 28, 5, 5, 1, 1, 0, 0 },
params_t{ bcast, fmt_caffe, 2, 24, 48, 8, 8, 5, 5, 1, 1, 0, 0 }
));
INSTANTIATE_TEST_CASE_P(ConvolutionBcastMnistTf,
        ConvolutionForward, ::testing::Values(
params_t{ bcast, fmt_tf, 2, 1, 24, 28, 28, 5, 5, 1, 1, 0, 0 },
params_t{ bcast, fmt_tf, 2, 24, 48, 8, 8, 5, 5, 1, 1, 0, 0 }
));
