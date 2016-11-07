#ifndef COMMON_HPP
#define COMMON_HPP

#include "xnor_nn.h"

#include "gtest.h"

namespace xnor_nn {
namespace test {

template <typename A, typename E>
void check_data(int MB, int C, int H, int W, int AC,
        const A *actual, const E *expected);

template <typename A, typename E>
void check_weights(int OC, int IC, int KH, int KW, int AC,
        const A *actual, const E *expected);

template <typename T>
void check_4d(int S3, int S2, int S1, int S0,
        const T *actual, const T *expected);

template <typename A, typename E>
void check_arrays(int elems, const A *actual, const E *expected);

template <typename A, typename E>
void check_value(const A &actual, const E &expected);

template<> void check_4d<unsigned char>(int S3, int S2, int S1, int S0,
        const unsigned char *a, const unsigned char *e) {
    int wrong = 0;
    for (int s3 = 0; s3 < S3; s3++)
    for (int s2 = 0; s2 < S2; s2++)
    for (int s1 = 0; s1 < S1; s1++)
    for (int s0 = 0; s0 < S0; s0++) {
        unsigned char actual = a[((s3*S2 + s2)*S1 + s1)*S0 + s0];
        unsigned char expected = e[((s3*S2 + s2)*S1 + s1)*S0 + s0];
        EXPECT_EQ(expected, actual) << "s3: " << s3 << ", s2: "
            << s2 << ", s1: " << s1 << ", s0: " << s0 << ". wrong/total: "
            << ++wrong << "/" << S3*S2*S1*S0;
    }
}

template<typename T> void check_data(
        int MB, int C, int H, int W, int AC,
        const T *a, const float *e) {
    const int SZ = sizeof(T) * 8;

    int wrong = 0;
    for (int mb = 0; mb < MB; mb++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++)
    for (int c = 0; c < C; c++) {
        bool actual = a[((mb*H + h)*W + w)*AC + (c / SZ)] &
            ((T)1) << (SZ - 1 - (c % SZ));
        bool expected = !(bool)
            (((unsigned int*)e)[((mb*C + c)*H + h)*W + w] >> 31);
        EXPECT_EQ(expected, actual) << "mb: " << mb << ", c: "
            << c << ", h: " << h << ", w: " << w << ". wrong/total: "
            << ++wrong << "/" << MB*C*H*W;
    }
}

template<typename T> void check_weights(
        int OC, int IC, int KH, int KW, int AC,
        const T *a, const float *e) {
    const int SZ = sizeof(T) * 8;

    int wrong = 0;
    for (int kh = 0; kh < KH; kh++)
    for (int kw = 0; kw < KW; kw++)
    for (int oc = 0; oc < OC; oc++)
    for (int ic = 0; ic < IC; ic++) {
        bool actual = a[((kh*KW + kw)*OC + oc)*AC + (ic / SZ)] &
            ((T)1) << (SZ - 1 - (ic % SZ));
        bool expected = !(bool)
            (((unsigned int*)e)[((oc*IC + ic)*KH + kh)*KW + kw] >> 31);
        EXPECT_EQ(expected, actual) << "oc: " << oc << ", ic: "
            << ic << ", kh: " << kh << ", kw: " << kw << ". wrong/total: "
            << ++wrong << "/" << OC*IC*KH*KW;
    }
}

template<> void check_4d<float>(int S3, int S2, int S1, int S0,
        const float *a, const float *e) {
    const float ERR = 1e-5f;

    int wrong = 0;
    for (int s3 = 0; s3 < S3; s3++)
    for (int s2 = 0; s2 < S2; s2++)
    for (int s1 = 0; s1 < S1; s1++)
    for (int s0 = 0; s0 < S0; s0++) {
        float actual = a[((s3*S2 + s2)*S1 + s1)*S0 + s0];
        float expected = e[((s3*S2 + s2)*S1 + s1)*S0 + s0];
        EXPECT_NEAR(expected, actual, ERR) << "s3: " << s3 << ", s2: "
            << s2 << ", s1: " << s1 << ", s0: " << s0 << ". wrong/total: "
            << ++wrong << "/" << S3*S2*S1*S0;
    }
}

template<> void check_arrays<float, float>(int elems,
        const float *actual, const float *expected) {
    const float ERR = 1e-5f;
    for (int i = 0; i < elems; i++) {
        EXPECT_NEAR(expected[i], actual[i], ERR) << "i: " << i << ", elems: "
            << elems;
    }
}

template<> void check_value<float, float>(const float &actual,
        const float &expected) {
    const float ERR = 1e-5f;
    EXPECT_NEAR(expected, actual, ERR);
}

}
}

#endif
