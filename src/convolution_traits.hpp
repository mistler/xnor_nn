#ifndef CONVOLUTION_TRAITS
#define CONVOLUTION_TRAITS

namespace xnor_nn{
namespace implementation{

template<typename Type>
struct ConvolutionTraits{};

struct RuntimeConvolutionTraits{};
struct IntConvolutionTraits{};

template<>
struct ConvolutionTraits<IntConvolutionTraits>{
    typedef int input_type;
    static constexpr int sz = 8;
    static constexpr int bits = sizeof(char)*sz;
    static constexpr int bici = sizeof(input_type);
};

template<>
struct ConvolutionTraits<RuntimeConvolutionTraits>{
    typedef void input_type;
    static constexpr int sz = 0;
    static constexpr int bits = 0;
    static constexpr int bici = 0;
};

} // namespace implementation
} // namespace xnor_nn

#endif // CONVOLUTION_TRAITS
