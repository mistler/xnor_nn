#include "xnor_nn.hpp"

#include "gtest.h"
#include "common.hpp"

TEST(DataBinFloatToFloat, simple_precalculated) {
    const int MB = 1, IC = 2, IH = 3, IW = 3;

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
    const float expected_src_bin[MB*IC*IH*IW] = {
        P, P, P,
        N, P, N,
        N, N, N,

        N, N, N,
        N, N, N,
        P, P, P
    };

    // Binarizer setup
    xnor_nn_status_t st;
    char st_msg[16];

    size_t sz_src_bin;
    void *actual_src_bin = NULL;

    xnor_nn_data_binarizer_t src_binarizer;

    st = xnor_nn_init_data_binarizer(&src_binarizer, MB, IC, IH, IW);
    if (st != xnor_nn_success) goto label;

    sz_src_bin = src_binarizer.size(&src_binarizer);

    st = xnor_nn_memory_allocate(&actual_src_bin, sz_src_bin);
    if (st != xnor_nn_success) goto label;

    // Execution
    st = src_binarizer.execute(&src_binarizer, src, actual_src_bin);
    if (st != xnor_nn_success) goto label;

    // Check result
    xnor_nn::test::check_data(MB, IC, IH, IW,
            (float*)actual_src_bin, expected_src_bin);

label:
    xnor_nn_memory_free(actual_src_bin);

    EXPECT_EQ(st, xnor_nn_success);
    xnor_nn_get_status_message(st_msg, st);
    printf("%s\n", st_msg);
}
