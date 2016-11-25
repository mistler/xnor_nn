#ifndef IMPLEMENTATION_HPP
#define IMPLEMENTATION_HPP

#include "xnor_nn_types.h"

#include <string>
#include <vector>

namespace xnor_nn {
namespace implementation {

class Implementation {
public:
    virtual ~Implementation() {};
    virtual bool isApplicable(const xnor_nn_convolution_t *c) const = 0;
    virtual void setupConvolution(xnor_nn_convolution_t *c) = 0;
};

std::vector<Implementation*> Implementations();

}
}

#endif
