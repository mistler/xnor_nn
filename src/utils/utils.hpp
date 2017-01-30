#ifndef UTILS_HPP
#define UTILS_HPP

constexpr int getOH(int ih, int kh, int sh, int ph) {
    return (ih + 2*ph - kh) / sh + 1;
}

constexpr int getOW(int iw, int kw, int sw, int pw) {
    return (iw + 2*pw - kw) / sw + 1;
}

// TODO: add div with round up function

#endif // UTILS_HPP
