#include "implementation.hpp"

#include <vector>

#include "xnor_nn_types.h"

#include "convolution_traits.hpp"

#include "bcast_convolution.hpp"
#include "reference_convolution.hpp"

namespace xnor_nn {
namespace implementation {

    // TODO: use class instead of pointer to object
std::vector<Implementation*> Implementations() {
    static std::vector<Implementation*> impls = {
        new BcastConvolution<ConvolutionTraits<IntConvolutionTraits>>(),

        new ReferenceConvolution(),
    };
    return impls;
};

void Implementation::setState(xnor_nn_convolution_t *c, Implementation *impl) {
    c->state = (void*)impl;
}

Implementation *Implementation::getState(const xnor_nn_convolution_t *c) {
    return reinterpret_cast<Implementation*>(c->state);
}

} // namespace implementation
} // namespace xnor_nn
