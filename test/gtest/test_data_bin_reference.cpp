#include "gtest.h"

#include "common.hpp"

#include "xnor_nn.hpp"

TEST(DataBinarizeReference, reference_precalculated) {
    const int MB = 1;
    const int IC = 2, OC = 2;
    const int IH = 3, IW = 3;
    const int OH = 3, OW = 3;
    const int KH = 3, KW = 3;
    const int SH = 1, SW = 1;
    const int PH = 1, PW = 1;

    const float P = 1.f;
    const float N = -1.f;

    // Usr data (just random)
    const float src_nchw[] = {
        P, P, P,
        N, P, N,
        N, N, N,

        N, N, N,
        N, N, N,
        P, P, P
    };
    const float src_nhwc[] = {
        P,N, P,N, P,N,
        N,N, P,N, N,N,
        N,P, N,P, N,P,
    };

    // Precalculated src
    const float expected_src_bin[] = {
        P,N, P,N, P,N,
        N,N, P,N, N,N,
        N,P, N,P, N,P,
    };

    // Precalculated A
    const float expected_a[] = {
        P, P, P,
        P, P, P,
        P, P, P,
    };
    // Precalculated K
    const float expected_k[] = {
        4.f / 9.f, 6.f / 9.f, 4.f / 9.f,
        6.f / 9.f, 9.f / 9.f, 6.f / 9.f,
        4.f / 9.f, 6.f / 9.f, 4.f / 9.f,
    };

    // Binarizer setup
    xnor_nn_resources_t res_nchw = {0};
    xnor_nn_resources_t res_nhwc = {0};

    xnor_nn_status_t st;
    char st_msg[16];

    xnor_nn_convolution_t convolution_nchw, convolution_nhwc;

    st = xnor_nn_init_convolution(&convolution_nchw,
            xnor_nn_algorithm_reference,
            xnor_nn_data_format_nchw,
            xnor_nn_weights_format_oihw,
            xnor_nn_data_format_nchw,
            MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW);
    if (st != xnor_nn_success) goto label;
    st = xnor_nn_init_convolution(&convolution_nhwc,
            xnor_nn_algorithm_reference,
            xnor_nn_data_format_nhwc,
            xnor_nn_weights_format_hwio,
            xnor_nn_data_format_nchw,
            MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW);
    if (st != xnor_nn_success) goto label;

    st = xnor_nn_allocate_resources(&convolution_nchw, res_nchw);
    if (st != xnor_nn_success) goto label;
    st = xnor_nn_allocate_resources(&convolution_nhwc, res_nhwc);
    if (st != xnor_nn_success) goto label;

    res_nchw[xnor_nn_resource_user_src] = (void*)src_nchw;
    res_nhwc[xnor_nn_resource_user_src] = (void*)src_nhwc;

    // Execution
    st = convolution_nchw.binarize_data(&convolution_nchw, res_nchw);
    if (st != xnor_nn_success) goto label;
    st = convolution_nhwc.binarize_data(&convolution_nhwc, res_nhwc);
    if (st != xnor_nn_success) goto label;

    st = convolution_nchw.calculate_k(&convolution_nchw, res_nchw);
    if (st != xnor_nn_success) goto label;
    st = convolution_nhwc.calculate_k(&convolution_nhwc, res_nhwc);
    if (st != xnor_nn_success) goto label;

    // Check result
    xnor_nn::test::check_4d(MB, IC, IH, IW,
            (float*)res_nchw[xnor_nn_resource_bin_src], expected_src_bin);
    xnor_nn::test::check_4d(MB, IC, IH, IW,
            (float*)res_nhwc[xnor_nn_resource_bin_src], expected_src_bin);

    // Check A
    xnor_nn::test::check_arrays(IH*IW, (float*)res_nchw[xnor_nn_resource_a],
            expected_a);
    xnor_nn::test::check_arrays(IH*IW, (float*)res_nhwc[xnor_nn_resource_a],
            expected_a);

    // Check K
    xnor_nn::test::check_arrays(OH*OW, (float*)res_nchw[xnor_nn_resource_k],
            expected_k);
    xnor_nn::test::check_arrays(OH*OW, (float*)res_nhwc[xnor_nn_resource_k],
            expected_k);

label:
    xnor_nn_free_resources(res_nchw);
    xnor_nn_free_resources(res_nhwc);

    xnor_nn_destroy_convolution(&convolution_nchw);
    xnor_nn_destroy_convolution(&convolution_nhwc);

    EXPECT_EQ(st, xnor_nn_success);
    xnor_nn_get_status_message(st_msg, st);
    printf("%s\n", st_msg);
}
