#include "gtest.h"

#include "common.hpp"

#include "xnor_nn.hpp"

TEST(WeightsBinarizeReference, reference_precalculated) {
    const int MB = 1;
    const int IC = 2, OC = 2;
    const int IH = 3, IW = 3;
    const int KH = 3, KW = 3;
    const int SH = 1, SW = 1;
    const int PH = 1, PW = 1;

    const float P = 1.f;
    const float N = -1.f;

    // Usr weights (just random)
    const float weights_oihw[] = {
        P, P, P,
        N, N, P,
        P, N, N,

        P, P, P,
        P, P, P,
        P, P, P,

        N, N, N,
        N, N, N,
        N, N, N,

        P, P, P,
        N, N, P,
        P, N, N
    };
    const float weights_hwio[] = {
        P,N,P,P, P,N,P,P, P,N,P,P,
        N,N,P,N, N,N,P,N, P,N,P,P,
        P,N,P,P, N,N,P,N, N,N,P,N,
    };

    // Precalculated weights
    const float expected_weights_bin[] = {
        P,N,P,P, P,N,P,P, P,N,P,P,
        N,N,P,N, N,N,P,N, P,N,P,P,
        P,N,P,P, N,N,P,N, N,N,P,N,
    };

    // Precalculated alpha
    const float alpha[OC] = {P, P};

    // Binarizer setup
    xnor_nn_resources_t res_oihw = {0};
    xnor_nn_resources_t res_hwio = {0};
    float *actual_alpha_oihw;
    float *actual_alpha_hwio;

    xnor_nn_status_t st;
    char st_msg[16];

    xnor_nn_convolution_t convolution_oihw, convolution_hwio;

    st = xnor_nn_init_convolution(&convolution_oihw,
            xnor_nn_algorithm_reference,
            xnor_nn_data_format_nchw,
            xnor_nn_weights_format_oihw,
            xnor_nn_data_format_nchw,
            MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW);
    if (st != xnor_nn_success) goto label;
    st = xnor_nn_init_convolution(&convolution_hwio,
            xnor_nn_algorithm_reference,
            xnor_nn_data_format_nhwc,
            xnor_nn_weights_format_hwio,
            xnor_nn_data_format_nchw,
            MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW);
    if (st != xnor_nn_success) goto label;

    st = xnor_nn_allocate_resources(&convolution_oihw, res_oihw);
    if (st != xnor_nn_success) goto label;
    st = xnor_nn_allocate_resources(&convolution_hwio, res_hwio);
    if (st != xnor_nn_success) goto label;

    res_oihw[xnor_nn_resource_user_weights] = (void*)weights_oihw;
    res_hwio[xnor_nn_resource_user_weights] = (void*)weights_hwio;

    // Execution
    st = convolution_oihw.binarize_weights(&convolution_oihw, res_oihw);
    if (st != xnor_nn_success) goto label;
    st = convolution_hwio.binarize_weights(&convolution_hwio, res_hwio);
    if (st != xnor_nn_success) goto label;

    // Check result
    xnor_nn::test::check_4d(OC, IC, KH, KW,
            (float*)res_oihw[xnor_nn_resource_bin_weights],
            expected_weights_bin);
    xnor_nn::test::check_4d(OC, IC, KH, KW,
            (float*)res_hwio[xnor_nn_resource_bin_weights],
            expected_weights_bin);

    actual_alpha_oihw = (float*)res_oihw[xnor_nn_resource_alpha];
    xnor_nn::test::check_arrays(OC, actual_alpha_oihw, alpha);

    actual_alpha_hwio = (float*)res_hwio[xnor_nn_resource_alpha];
    xnor_nn::test::check_arrays(OC, actual_alpha_hwio, alpha);

label:
    xnor_nn_free_resources(res_oihw);
    xnor_nn_free_resources(res_hwio);

    xnor_nn_destroy_convolution(&convolution_oihw);
    xnor_nn_destroy_convolution(&convolution_hwio);

    EXPECT_EQ(st, xnor_nn_success);
    xnor_nn_get_status_message(st_msg, st);
    printf("%s\n", st_msg);
}
