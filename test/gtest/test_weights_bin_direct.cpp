#include "xnor_nn.hpp"

#include "gtest.h"
#include "common.hpp"

TEST(WeightsBinarizeDirect, direct_precalculated) {
    const int MB = 1;
    const int IC = 2, OC = 2;
    const int IH = 3, IW = 3;
    const int KH = 3, KW = 3;
    const int SH = 1, SW = 1;
    const int PH = 1, PW = 1;

    const float P = 1.f;
    const float N = -1.f;

    // Usr weights (just random)
    const float weights[] = {
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

    /*
    // Precalculated weights
    const unsigned char expected_weights_bin[] = {
        0xFFu, 0xFFu, 0x7Fu, 0xFFu,
        0xFFu, 0xFFu, 0x7Fu, 0xFFu,
        0xFFu, 0xFFu, 0x7Fu, 0xFFu,

        0x7Fu, 0xFFu, 0x3Fu, 0xFFu,
        0x7Fu, 0xFFu, 0x3Fu, 0xFFu,
        0xFFu, 0xFFu, 0x7Fu, 0xFFu,

        0xFFu, 0xFFu, 0x7Fu, 0xFFu,
        0x7Fu, 0xFFu, 0x3Fu, 0xFFu,
        0x7Fu, 0xFFu, 0x3Fu, 0xFFu,
    };
    */

    // Precalculated alpha
    const float alpha = P;

    // Binarizer setup
    xnor_nn_resources_t res = {0};
    float *actual_alpha = (float*)(&res[xnor_nn_resource_alpha]);

    xnor_nn_status_t st;
    char st_msg[16];

    xnor_nn_convolution_t convolution;

    st = xnor_nn_init_convolution(&convolution, xnor_nn_algorithm_direct,
            MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW);
    const int VLEN = convolution.vlen;
    const int BIC = ((IC + 8 - 1) / 8) * 8;
    const int ABIC = ((BIC + VLEN - 1) / VLEN) * VLEN;
    if (st != xnor_nn_success) goto label;

    st = xnor_nn_allocate_resources(&convolution, res);
    if (st != xnor_nn_success) goto label;

    res[xnor_nn_resource_user_weights] = (void*)weights;

    // Execution
    st = convolution.binarize_weights(&convolution, res);
    if (st != xnor_nn_success) goto label;

    // Check result
    /*
    xnor_nn::test::check_4d(OC, convolution.aic, KH, KW,
            (unsigned char*)res[xnor_nn_resource_bin_weights],
            expected_weights_bin);
    */

    xnor_nn::test::check_weights(OC, IC, KH, KW, ABIC,
            (unsigned char*)res[xnor_nn_resource_bin_weights], weights);

    xnor_nn::test::check_value(*actual_alpha, alpha);

label:
    xnor_nn_free_resources(res);

    xnor_nn_destroy_convolution(&convolution);

    EXPECT_EQ(st, xnor_nn_success);
    xnor_nn_get_status_message(st_msg, st);
    printf("%s\n", st_msg);
}
