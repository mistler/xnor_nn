#include "xnor_nn.hpp"

#include "gtest.h"
#include "common.hpp"

TEST(DataBinFloatToFloat, simple_precalculated) {
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
    const float src[MB*IC*IH*IW] = {
        P, P, P,
        N, P, N,
        N, N, N,

        N, N, N,
        N, N, N,
        P, P, P
    };
    // Precalculated src
    const float expected_src_bin[MB*IC*IH*IW + IH*IW + OH*OW] = {
        P, P, P,
        N, P, N,
        N, N, N,

        N, N, N,
        N, N, N,
        P, P, P,

        // A
        P, P, P,
        P, P, P,
        P, P, P,

        // K
        4.f / 9.f, 6.f / 9.f, 4.f / 9.f,
        6.f / 9.f, 9.f / 9.f, 6.f / 9.f,
        4.f / 9.f, 6.f / 9.f, 4.f / 9.f,
    };

    // Binarizer setup
    xnor_nn_status_t st;
    char st_msg[16];

    size_t sz_src_bin;
    void *actual_src_bin = NULL;

    xnor_nn_convolution_t convolution;
    xnor_nn_data_binarizer_t src_binarizer;

    st = xnor_nn_init_convolution(&convolution,
            MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW);
    if (st != xnor_nn_success) goto label;

    st = xnor_nn_init_data_binarizer(&src_binarizer, &convolution);
    if (st != xnor_nn_success) goto label;

    sz_src_bin = src_binarizer.size(&src_binarizer);

    st = xnor_nn_memory_allocate(&actual_src_bin, sz_src_bin);
    if (st != xnor_nn_success) goto label;

    // Execution
    st = src_binarizer.binarize(&src_binarizer, src, actual_src_bin);
    if (st != xnor_nn_success) goto label;

    st = src_binarizer.calculate_k(&src_binarizer, src, actual_src_bin);
    if (st != xnor_nn_success) goto label;

    // Check result
    xnor_nn::test::check_data(MB, IC, IH, IW,
            (float*)actual_src_bin, expected_src_bin);

    // Check A
    xnor_nn::test::check_arrays(IH*IW, (float*)actual_src_bin + MB*IC*IH*IW,
            expected_src_bin + MB*IC*IH*IW);

    // Check K
    xnor_nn::test::check_arrays(OH*OW,
            (float*)actual_src_bin + MB*IC*IH*IW + IH*IW,
            expected_src_bin + MB*IC*IH*IW + IH*IW);

label:
    xnor_nn_memory_free(actual_src_bin);

    EXPECT_EQ(st, xnor_nn_success);
    xnor_nn_get_status_message(st_msg, st);
    printf("%s\n", st_msg);
}
