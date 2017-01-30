#ifndef UTILS_HPP
#define UTILS_HPP

#ifdef __x86_64__

#ifdef __AVX__
#define VLEN 256
#else
#define VLEN 32
#endif

#elif defined __arm__

#ifdef __ARM_NEON
#define VLEN 128
#else
#define VLEN 32
#endif

#else

#error Target is not supported

#endif

constexpr int getOH(int ih, int kh, int sh, int ph) {
    return (ih + 2*ph - kh) / sh + 1;
}

constexpr int getOW(int iw, int kw, int sw, int pw) {
    return (iw + 2*pw - kw) / sw + 1;
}

// TODO: add div with round up function

#endif // UTILS_HPP
