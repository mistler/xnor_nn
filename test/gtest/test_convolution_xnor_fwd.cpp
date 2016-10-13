#include "xnor_nn.hpp"
#include "gtest.h"

TEST(ConvolutionXnorFwd, simple_precalculated) {
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
    const float weights[OC*IC*KH*KW] = {
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
    // Precalculated output
    const float expected_dst[MB*OC*OH*OW] = {
        8, 12, 8,
        48, 81, 36,
        16, 36, 20,

        16, 36, 16,
        24, 72, 36,
        16, 36, 12
    }; // * 1/9

    float actual_dst[MB*OC*OH*OW] = { 0.f };

    // Convolution setup
    xnor_nn::Convolution convolution{MB, OC, IC, IH, IW,
            KH, KW, SH, SW, PH, PW, weights};

    // Execution
    convolution.forward(src, actual_dst);

    // Check result
    int wrong = 0;
    for (int mb = 0; mb < MB; mb++)
    for (int oc = 0; oc < OC; oc++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float actual = actual_dst[((mb*OC + oc)*OH + oh)*OW + ow];
        float expected = expected_dst[((mb*OC + oc)*OH + oh)*OW + ow] / 9.f;
        EXPECT_NEAR(expected, actual, 1e-5f) << "mb: " << mb << ", oc: "
            << oc << ", oh: " << oh << ", ow: " << ow << ". wrong/total: "
            << ++wrong << "/" << MB*OC*OH*OW;
    }
}
