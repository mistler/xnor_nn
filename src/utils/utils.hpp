#ifndef UTILS_HPP
#define UTILS_HPP

namespace xnor_nn{
namespace utils{

inline constexpr int getOH(int ih, int kh, int sh, int ph) {
    return (ih + 2*ph - kh) / sh + 1;
}

inline constexpr int getOW(int iw, int kw, int sw, int pw) {
    return (iw + 2*pw - kw) / sw + 1;
}

template<typename T>
inline constexpr T div_up(const T &a, const T &b) {
    return (a+b-1) / b;
}

} // namespace utils
} // namespace xnor_nn

// TODO: add div with round up function

#endif // UTILS_HPP
