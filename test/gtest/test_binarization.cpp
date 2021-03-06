#include "gtest.h"

#include "common.hpp"

#include "xnor_nn.hpp"
#include "cpuid.hpp"

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

class Binarization : public ::testing::TestWithParam<params_t> {
protected:
    virtual void SetUp() {
        params_t p = ::testing::TestWithParam<params_t>::GetParam();

        p.oh = (p.ih + 2*p.ph - p.kh) / p.sh + 1;
        p.ow = (p.iw + 2*p.pw - p.kw) / p.sw + 1;

        float *src = new float[p.mb*p.ic*p.ih*p.iw];
        float *weights = new float[p.oc*p.ic*p.kh*p.kw];

        fill_src(src, p);
        fill_weights(weights, p);

        xnor_nn_resources_t res = {0};

        xnor_nn_status_t st;
        char st_msg[16];

        xnor_nn_convolution_t convolution;
        const int bici = getBICI(p);
        const int vlen = xnor_nn::utils::Cpuid::vlen();

        st = xnor_nn_init_convolution(&convolution, p.algorithm,
            p.fmt.src, p.fmt.weights, p.fmt.dst,
            p.mb, p.oc, p.ic, p.ih, p.iw, p.kh, p.kw,
            p.sh, p.sw, p.ph, p.pw);
        if (st != xnor_nn_success) goto label;

        st = xnor_nn_allocate_resources(&convolution, res);
        if (st != xnor_nn_success) goto label;

        res[xnor_nn_resource_user_src] = (void*)src;
        res[xnor_nn_resource_user_weights] = (void*)weights;

        // Execution
        st = convolution.binarize_weights(&convolution, res);
        if (st != xnor_nn_success) goto label;

        st = convolution.binarize_data(&convolution, res);
        if (st != xnor_nn_success) goto label;

        // Check result
        xnor_nn::test::check_weights_bcast(p.fmt.weights,
                p.oc, p.ic, p.kh, p.kw, bici, vlen,
                (unsigned char*)res[xnor_nn_resource_bin_weights], weights);
        xnor_nn::test::check_data_bcast(p.fmt.src,
                p.mb, p.ic, p.ih, p.iw, bici,
                (unsigned char*)res[xnor_nn_resource_bin_src], src);
        // TODO: check alpha, k, a

label:
        delete[] src;
        delete[] weights;
        xnor_nn_free_resources(res);

        xnor_nn_destroy_convolution(&convolution);

        EXPECT_EQ(st, xnor_nn_success);
        xnor_nn_get_status_message(st_msg, st);
        printf("%s\n", st_msg);
    }
};

TEST_P(Binarization, binarization)
{
}

auto bcast = xnor_nn_algorithm_bcast;

auto fmt_caffe = tensor_fmt_t{xnor_nn_data_format_nchw,
    xnor_nn_weights_format_oihw, xnor_nn_data_format_nchw};
auto fmt_tf = tensor_fmt_t{xnor_nn_data_format_nchw,
    xnor_nn_weights_format_oihw, xnor_nn_data_format_nchw};

// mb ic oc ih iw kh kw sh sw ph pw
INSTANTIATE_TEST_CASE_P(BinarizationBcastTaskCaffe,
        Binarization, ::testing::Values(
params_t{ bcast, fmt_caffe, 1, 1, 32, 60, 61, 3, 3, 1, 1, 0, 0 },
params_t{ bcast, fmt_caffe, 256, 1, 32, 60, 61, 3, 3, 1, 1, 0, 0 },
params_t{ bcast, fmt_caffe, 1, 32, 32, 20, 20, 3, 3, 1, 1, 0, 0 },
params_t{ bcast, fmt_caffe, 256, 32, 32, 20, 20, 3, 3, 1, 1, 0, 0 }
));
INSTANTIATE_TEST_CASE_P(BinarizationBcastTaskTf,
        Binarization, ::testing::Values(
params_t{ bcast, fmt_tf, 1, 1, 32, 60, 61, 3, 3, 1, 1, 0, 0 },
params_t{ bcast, fmt_tf, 256, 1, 32, 60, 61, 3, 3, 1, 1, 0, 0 },
params_t{ bcast, fmt_tf, 1, 32, 32, 20, 20, 3, 3, 1, 1, 0, 0 },
params_t{ bcast, fmt_tf, 256, 32, 32, 20, 20, 3, 3, 1, 1, 0, 0 }
));
INSTANTIATE_TEST_CASE_P(BinarizationBcastAlexNetCaffe,
        Binarization, ::testing::Values(
params_t{ bcast, fmt_caffe, 2, 3, 96, 227, 227, 11, 11, 4, 4, 0, 0 },
params_t{ bcast, fmt_caffe, 2, 96, 256, 27, 27, 5, 5, 1, 1, 2, 2 },
params_t{ bcast, fmt_caffe, 2, 256, 384, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 2, 384, 384, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_caffe, 2, 384, 256, 13, 13, 3, 3, 1, 1, 1, 1 }
));
INSTANTIATE_TEST_CASE_P(BinarizationBcastAlexNetTf,
        Binarization, ::testing::Values(
params_t{ bcast, fmt_tf, 2, 3, 96, 227, 227, 11, 11, 4, 4, 0, 0 },
params_t{ bcast, fmt_tf, 2, 96, 256, 27, 27, 5, 5, 1, 1, 2, 2 },
params_t{ bcast, fmt_tf, 2, 256, 384, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 2, 384, 384, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ bcast, fmt_tf, 2, 384, 256, 13, 13, 3, 3, 1, 1, 1, 1 }
));
