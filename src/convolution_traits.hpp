#ifndef CONVOLUTION_TRAITS
#define CONVOLUTION_TRAITS

#include <limits>
#include <cstdint>

#include "xnor_nn_types.h"

namespace xnor_nn{
namespace implementation{

template<typename Type>
struct ConvolutionTraits{
    typedef void data_t;
    typedef void udata_t;
    static constexpr int sz = -1;
    static constexpr int bits = -1;
    static constexpr int bici = -1;
    static bool isApplicable(const xnor_nn_convolution_t *c) {
        (void)c;
        return false;
    }
};

struct ShortConvolutionTraits{};
struct IntConvolutionTraits{};
struct RuntimeConvolutionTraits{};

template<>
struct ConvolutionTraits<ShortConvolutionTraits>{
    typedef int16_t data_t;
    typedef uint16_t udata_t;
    static constexpr int sz = 8;
    static constexpr int bits = sizeof(int8_t)*sz;
    static constexpr int bici = sizeof(data_t);
    static bool isApplicable(const xnor_nn_convolution_t *c) {
#if 1
        return c->ic * c->kh * c->kw < std::numeric_limits<data_t>::max();
#else
        (void)c;
        return false;
#endif
    }
};

template<>
struct ConvolutionTraits<IntConvolutionTraits>{
    typedef int32_t data_t;
    typedef uint32_t udata_t;
    static constexpr int sz = 8;
    static constexpr int bits = sizeof(int8_t)*sz;
    static constexpr int bici = sizeof(data_t);
    static bool isApplicable(const xnor_nn_convolution_t *c) {
        return c->ic * c->kh * c->kw < std::numeric_limits<data_t>::max();
    }
};

} // namespace implementation
} // namespace xnor_nn

#endif // CONVOLUTION_TRAITS
