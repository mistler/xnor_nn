#include "convolution_traits.hpp"

namespace xnor_nn{
namespace implementation{

// TODO: check if it is necessary
template<typename T> constexpr int ConvolutionTraits<T>::sz;
template<typename T> constexpr int ConvolutionTraits<T>::bits;
template<typename T> constexpr int ConvolutionTraits<T>::bici;

constexpr int ConvolutionTraits<IntConvolutionTraits>::sz;
constexpr int ConvolutionTraits<IntConvolutionTraits>::bits;
constexpr int ConvolutionTraits<IntConvolutionTraits>::bici;

constexpr int ConvolutionTraits<ShortConvolutionTraits>::sz;
constexpr int ConvolutionTraits<ShortConvolutionTraits>::bits;
constexpr int ConvolutionTraits<ShortConvolutionTraits>::bici;

template struct ConvolutionTraits<ShortConvolutionTraits>;
template struct ConvolutionTraits<IntConvolutionTraits>;
template struct ConvolutionTraits<RuntimeConvolutionTraits>;

}; // namespace implementation
}; // namespace xnor_nn
