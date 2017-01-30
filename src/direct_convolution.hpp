#ifndef DIRECT_CONVOLUTION_HPP
#define DIRECT_CONVOLUTION_HPP

#include "direct_base.hpp"
#include "direct_template_parameters.hpp"

namespace xnor_nn {
namespace implementation {

class DirectConvolution : public DirectBase {
public:
    ~DirectConvolution();
    bool isApplicable(const xnor_nn_convolution_t *c) const;
    void setupConvolution(xnor_nn_convolution_t *c);

private:

#define EXECUTE(NAME) \
    template<int OC, int IC, int IH, int IW, int KH, int KW, \
            int SH, int SW, int PH, int PW> \
    static xnor_nn_status_t exec_##NAME##_template( \
            const xnor_nn_convolution_t *c, xnor_nn_resources_t res); \
    static xnor_nn_status_t exec_##NAME##_simple( \
            const xnor_nn_convolution_t *c, xnor_nn_resources_t res);

#ifdef __x86_64__
    EXECUTE(avx)
#elif defined __arm__
    EXECUTE(neon)
#endif
    EXECUTE(default)

#undef EXECUTE

};

} // namespace implementation
} // namespace xnor_nn

#endif // DIRECT_CONVOLUTION_HPP
