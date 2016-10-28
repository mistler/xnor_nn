#ifndef COMMON_HPP
#define COMMON_HPP

#include "xnor_nn.h"

#include "gtest.h"

namespace xnor_nn {
namespace test {

template <typename A, typename E>
void check_data(int MB, int C, int H, int W,
        const A *actual, const E *expected);

template <typename A, typename E>
void check_weights(int OC, int IC, int KH, int KW,
        const A *actual, const E *expected);

template <typename A, typename E>
void check_arrays(int elems, const A *actual, const E *expected);

template <typename A, typename E>
void check_value(const A &actual, const E &expected);


template<> void check_data<float, float>(int MB, int C, int H, int W,
        const float *a, const float *e) {
    const float ERR = 1e-5f;
    int wrong = 0;
    for (int mb = 0; mb < MB; mb++)
    for (int c = 0; c < C; c++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++) {
        float actual = a[((mb*C + c)*H + h)*W + w];
        float expected = e[((mb*C + c)*H + h)*W + w];
        EXPECT_NEAR(expected, actual, ERR) << "mb: " << mb << ", c: "
            << c << ", h: " << h << ", w: " << w << ". wrong/total: "
            << ++wrong << "/" << MB*C*H*W;
    }
}

template<> void check_data<unsigned char, unsigned char>(
        int MB, int C, int H, int W,
        const unsigned char *a, const unsigned char *e) {
    int wrong = 0;
    for (int mb = 0; mb < MB; mb++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++)
    for (int c = 0; c < C; c++) {
        unsigned char actual = a[((mb*H + h)*W + w)*C + c];
        unsigned char expected = e[((mb*H + h)*W + w)*C + c];
        EXPECT_EQ(expected, actual) << "mb: " << mb << ", c: "
            << c << ", h: " << h << ", w: " << w << ". wrong/total: "
            << ++wrong << "/" << MB*C*H*W;
    }
}

template<> void check_data<unsigned char, float>(
        int MB, int C, int H, int W,
        const unsigned char *a, const float *e) {
    int wrong = 0;
    const int OC = (C + 8 - 1) / 8;
    for (int mb = 0; mb < MB; mb++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++)
    for (int c = 0; c < C; c++) {
        bool actual = a[((mb*H + h)*W + w)*OC + (c / 8)] &
            ((unsigned char)128) >> (c % 8);
        bool expected = !(bool)
            (((unsigned int*)e)[((mb*C + c)*H + h)*W + w] >> 31);
        EXPECT_EQ(expected, actual) << "mb: " << mb << ", c: "
            << c << ", h: " << h << ", w: " << w << ". wrong/total: "
            << ++wrong << "/" << MB*C*H*W;
    }
}

template<> void check_weights<float, float>(int OC, int IC, int KH, int KW,
        const float *a, const float *e) {
    const float ERR = 1e-5f;
    int wrong = 0;
    for (int oc = 0; oc < OC; oc++)
    for (int ic = 0; ic < IC; ic++)
    for (int kh = 0; kh < KH; kh++)
    for (int kw = 0; kw < KW; kw++) {
        float actual = a[((oc*IC + ic)*KH + kh)*KW + kw];
        float expected = e[((oc*IC + ic)*KH + kh)*KW + kw];
        EXPECT_NEAR(expected, actual, ERR) << "oc: " << oc << ", ic: "
            << ic << ", kh: " << kh << ", kw: " << kw << ". wrong/total: "
            << ++wrong << "/" << OC*IC*KH*KW;
    }
}

template<> void check_weights<unsigned char, unsigned char>(
        int OC, int IC, int KH, int KW,
        const unsigned char *a, const unsigned char *e) {
    int wrong = 0;
    for (int oc = 0; oc < OC; oc++)
    for (int ic = 0; ic < IC; ic++)
    for (int kh = 0; kh < KH; kh++)
    for (int kw = 0; kw < KW; kw++) {
        float actual = a[((kh*KW + kw)*OC + oc)*IC + ic];
        float expected = e[((kh*KW + kw)*OC + oc)*IC + ic];
        EXPECT_EQ(expected, actual) << "oc: " << oc << ", ic: "
            << ic << ", kh: " << kh << ", kw: " << kw << ". wrong/total: "
            << ++wrong << "/" << OC*IC*KH*KW;
    }
}

template<> void check_weights<unsigned char, float>(
        int OC, int IC, int KH, int KW,
        const unsigned char *a, const float *e) {
    const int BIC = (IC + 8 - 1) / 8;
    int wrong = 0;
    for (int kh = 0; kh < KH; kh++)
    for (int kw = 0; kw < KW; kw++)
    for (int oc = 0; oc < OC; oc++)
    for (int ic = 0; ic < IC; ic++) {
        bool actual = a[((kh*KW + kw)*OC + oc)*BIC + (ic / 8)] &
            ((unsigned char)128) >> (ic % 8);
        bool expected = !(bool)
            (((unsigned int*)e)[((oc*IC + ic)*KH + kh)*KW + kw] >> 31);
        EXPECT_EQ(expected, actual) << "oc: " << oc << ", ic: "
            << ic << ", kh: " << kh << ", kw: " << kw << ". wrong/total: "
            << ++wrong << "/" << OC*IC*KH*KW;
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
