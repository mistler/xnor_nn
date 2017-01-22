#ifndef UTILS_H
#define UTILS_H

#ifdef __x86_64__
#define VLEN 256
#define ARCH_X86
#elif defined __arm__
#define VLEN 128
#define ARCH_ARM
#else
#define VLEN 32
#define ARCH_UNDEF
#endif

constexpr int getOH(int ih, int kh, int sh, int ph) {
    return (ih + 2*ph - kh) / sh + 1;
}

constexpr int getOW(int iw, int kw, int sw, int pw) {
    return (iw + 2*pw - kw) / sw + 1;
}

// TODO: add div with round up function

#endif // UTILS_H
