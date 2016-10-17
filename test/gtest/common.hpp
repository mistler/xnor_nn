#ifndef COMMON_HPP
#define COMMON_HPP

#include "xnor_nn.h"

#include "gtest.h"

namespace xnor_nn {
namespace test {

template <typename T>
void check_data(int MB, int C, int H, int W,
        const T *actual, const T *expected);

template<> void check_data<float>(int MB, int C, int H, int W,
        const float *actual, const float *expected) {
    const float ERR = 1e-5f;
    int wrong = 0;
    for (int mb = 0; mb < MB; mb++)
    for (int c = 0; c < C; c++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++) {
        float a = actual[((mb*C + c)*H + h)*W + w];
        float e = expected[((mb*C + c)*H + h)*W + w];
        EXPECT_NEAR(e, a, ERR) << "mb: " << mb << ", c: "
            << c << ", h: " << h << ", w: " << w << ". wrong/total: "
            << ++wrong << "/" << MB*C*H*W;
    }
}

}
}

#endif
