#ifndef IMPLEMENTATION_HPP
#define IMPLEMENTATION_HPP

#include <vector>

#include "xnor_nn_types.h"

namespace xnor_nn {
namespace implementation {

class Implementation {
public:
    virtual bool isApplicable(const xnor_nn_convolution_t *c) const = 0;
    virtual void setupConvolution(xnor_nn_convolution_t *c) = 0;
};

std::vector<Implementation*> Implementations();

}
}

#endif
