#ifndef IMPLEMENTATION_HPP
#define IMPLEMENTATION_HPP

#include "xnor_nn_types.h"

#include <vector>

namespace xnor_nn {
namespace implementation {

class Implementation {
public:
    virtual ~Implementation() {};
    virtual bool isApplicable(const xnor_nn_convolution_t *c) const = 0;
    // TODO: rename setupParams
    virtual void setupConvolution(xnor_nn_convolution_t *c) = 0;

    xnor_nn_operation_t op;

protected:
    // TODO: maybe const implementation
    void setState(xnor_nn_convolution_t *c, Implementation *impl,
            const xnor_nn_operation_t operation);
    static Implementation *getState(const xnor_nn_convolution_t *c,
            const xnor_nn_operation_t operation);

};

std::vector<Implementation*> Implementations();

} // namespace implementation
} // namespace xnor_nn

#endif
