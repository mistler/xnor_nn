#ifndef REFERENCE_CONVOLUTION_HPP
#define REFERENCE_CONVOLUTION_HPP

#include "implementation.hpp"

namespace xnor_nn {
namespace implementation {

class ReferenceConvolution : public Implementation {
public:
    ~ReferenceConvolution();
    bool isApplicable(const xnor_nn_convolution_t *c) const;
    void setupConvolution(xnor_nn_convolution_t *c);
private:
    static xnor_nn_status_t exec(const xnor_nn_convolution_t *c,
            xnor_nn_resources_t res);
};

} // namespace implementation
} // namespace xnor_nn

#endif // REFERENCE_CONVOLUTION_HPP
