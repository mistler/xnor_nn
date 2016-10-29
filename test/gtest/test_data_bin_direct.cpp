#include "xnor_nn.hpp"

#include "gtest.h"
#include "common.hpp"

TEST(DataBinarizationDirectChar, optimized_precalculated) {
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
    const float src[] = {
        P, P, P,
        N, P, N,
        N, N, N,

        N, N, N,
        N, N, N,
        P, P, P
    };

    // Precalculated src
    const unsigned char expected_src_bin[] = {
        0x80u, 0x80u, 0x80u,
        0x00u, 0x80u, 0x00u,
        0x40u, 0x40u, 0x40u,
    };

    // Precalculated a
    const float expected_a[] = {
        P, P, P,
        P, P, P,
        P, P, P,
    };

    // Precalculated k
    const float expected_k[] = {
        4.f / 9.f, 6.f / 9.f, 4.f / 9.f,
        6.f / 9.f, 9.f / 9.f, 6.f / 9.f,
        4.f / 9.f, 6.f / 9.f, 4.f / 9.f,
    };

    // Binarizer setup
    xnor_nn_resources_t res = {0};

    xnor_nn_status_t st;
    char st_msg[16];

    xnor_nn_convolution_t convolution;

    st = xnor_nn_init_convolution(&convolution, xnor_nn_algorithm_optimized,
            MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW);
    if (st != xnor_nn_success) goto label;

    st = xnor_nn_allocate_resources(&convolution, res);
    if (st != xnor_nn_success) goto label;

    res[xnor_nn_resource_user_src] = (void*)src;

    // Execution
    st = convolution.binarize_data(&convolution, res);
    if (st != xnor_nn_success) goto label;

    st = convolution.calculate_k(&convolution, res);
    if (st != xnor_nn_success) goto label;

    // Check result
    xnor_nn::test::check_4d(MB, IH, IW, (IC + 8 - 1) / 8,
            (unsigned char*)res[xnor_nn_resource_bin_src], expected_src_bin);
    xnor_nn::test::check_data(MB, IC, IH, IW,
            (unsigned char*)res[xnor_nn_resource_bin_src], src);

    // Check A
    xnor_nn::test::check_arrays(IH*IW, (float*)res[xnor_nn_resource_a],
            expected_a);

    // Check K
    xnor_nn::test::check_arrays(OH*OW, (float*)res[xnor_nn_resource_k],
            expected_k);

label:
    xnor_nn_free_resources(res);

    EXPECT_EQ(st, xnor_nn_success);
    xnor_nn_get_status_message(st_msg, st);
    printf("%s\n", st_msg);
}
