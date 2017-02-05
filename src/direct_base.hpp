#ifndef DIRECT_BASE_HPP
#define DIRECT_BASE_HPP

#include "implementation.hpp"

#include "xnor_nn_types.h"
#include "utils.hpp"

namespace xnor_nn {
namespace implementation {

class DirectBase : public Implementation {
public:
    virtual ~DirectBase();
    virtual bool isApplicable(const xnor_nn_convolution_t *c) const;
    virtual void setupConvolution(xnor_nn_convolution_t *c);

protected:
    static constexpr int getABIC(int IC) {
        return (((((IC + SZ - 1) / SZ) * SZ) + VLEN - 1) / VLEN) * VLEN;
    }

protected:
    int BIC, ABIC;

    // TODO: remove unused stuff
    static constexpr int SZ = 8;
    static constexpr int ELEM_SIZE = sizeof(char);
    static constexpr int BITS = ELEM_SIZE * SZ;
};

} // namespace implementation
} // namespace xnor_nn

#endif // DIRECT_BASE_HPP
