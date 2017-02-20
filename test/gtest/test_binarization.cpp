#include "gtest.h"

#include "common.hpp"

#include "xnor_nn.hpp"
#include "cpuid.hpp"

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

        st = xnor_nn_init_convolution(&convolution, p.algorithm,
                p.mb, p.oc, p.ic, p.ih, p.iw, p.kh, p.kw,
                p.sh, p.sw, p.ph, p.pw);
        const int VLEN = xnor_nn::utils::Cpuid::vlen();
        const int BIC = ((p.ic + 8 - 1) / 8) * 8;
        const int ABIC = ((BIC + VLEN - 1) / VLEN) * VLEN;
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
        if (p.algorithm == xnor_nn_algorithm_bcast) {
            // TODO: some better checks
            xnor_nn::test::check_weights_bcast(p.oc, p.ic, p.kh, p.kw, VLEN,
                    (unsigned char*)res[xnor_nn_resource_bin_weights], weights);
            xnor_nn::test::check_data(p.mb, p.ic, p.ih, p.iw,
                    ((p.ic + 4 - 1) / 4 + 8 - 1) / 8 * 4 * 8,
                    (unsigned char*)res[xnor_nn_resource_bin_src], src);
        } else {
            xnor_nn::test::check_weights(p.oc, p.ic, p.kh, p.kw, ABIC,
                    (unsigned char*)res[xnor_nn_resource_bin_weights], weights);
            xnor_nn::test::check_data(p.mb, p.ic, p.ih, p.iw, ABIC,
                    (unsigned char*)res[xnor_nn_resource_bin_src], src);
        }

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

// mb ic oc ih iw kh kw sh sw ph pw
INSTANTIATE_TEST_CASE_P(BinarizationBcastTask,
        Binarization, ::testing::Values(
// params_t{ xnor_nn_algorithm_bcast, 1, 1, 32, 60, 61, 3, 3, 1, 1, 0, 0 },
// params_t{ xnor_nn_algorithm_bcast, 256, 1, 32, 60, 61, 3, 3, 1, 1, 0, 0 },
params_t{ xnor_nn_algorithm_bcast, 1, 32, 32, 20, 20, 3, 3, 1, 1, 0, 0 },
params_t{ xnor_nn_algorithm_bcast, 256, 32, 32, 20, 20, 3, 3, 1, 1, 0, 0 }
));
INSTANTIATE_TEST_CASE_P(BinarizationBcastAlexNet,
        Binarization, ::testing::Values(
// params_t{ xnor_nn_algorithm_bcast, 2, 3, 96, 227, 227, 11, 11, 4, 4, 0, 0 },
params_t{ xnor_nn_algorithm_bcast, 2, 96, 256, 27, 27, 5, 5, 1, 1, 2, 2 },
params_t{ xnor_nn_algorithm_bcast, 2, 256, 384, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_bcast, 2, 384, 384, 13, 13, 3, 3, 1, 1, 1, 1 },
params_t{ xnor_nn_algorithm_bcast, 2, 384, 256, 13, 13, 3, 3, 1, 1, 1, 1 }
));
INSTANTIATE_TEST_CASE_P(BinarizationDirectAlexNet,
        Binarization, ::testing::Values(
// params_t{ xnor_nn_algorithm_direct, 2, 3, 96, 227, 227, 11, 11, 4, 4, 0, 0 },
// params_t{ xnor_nn_algorithm_direct, 2, 96, 256, 27, 27, 5, 5, 1, 1, 2, 2 },
params_t{ xnor_nn_algorithm_direct, 2, 256, 384, 13, 13, 3, 3, 1, 1, 1, 1 }
// params_t{ xnor_nn_algorithm_direct, 2, 384, 384, 13, 13, 3, 3, 1, 1, 1, 1 },
// params_t{ xnor_nn_algorithm_direct, 2, 384, 256, 13, 13, 3, 3, 1, 1, 1, 1 }
));
