#ifndef TEMPLATE_CONVOLUTION_HPP
#define TEMPLATE_CONVOLUTION_HPP

#include "implementation.hpp"

namespace xnor_nn {
namespace implementation {

class TemplateConvolution : public Implementation {
public:
    ~TemplateConvolution();
    bool isApplicable(const xnor_nn_convolution_t *c) const;
    void setupConvolution(xnor_nn_convolution_t *c);
};

} // namespace implementation
} // namespace xnor_nn

#endif // TEMPLATE_CONVOLUTION_HPP
