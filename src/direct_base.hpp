#ifndef DIRECT_BASE_HPP
#define DIRECT_BASE_HPP

#include "implementation.hpp"

#include "xnor_nn_types.h"
#include "cpuid.hpp"

namespace xnor_nn {
namespace implementation {

class DirectBase : public Implementation {
public:
    virtual ~DirectBase();
    virtual bool isApplicable(const xnor_nn_convolution_t *c) const;
    virtual void setupConvolution(xnor_nn_convolution_t *c);

protected:

#ifdef VLEN
    static constexpr int constexpr_getABIC(int IC) {
        return (((((IC + SZ - 1) / SZ) * SZ) + VLEN - 1) / VLEN) * VLEN;
    }
#endif

    static int getABIC(int IC) {
        const int vlen_ = xnor_nn::utils::Cpuid::vlen();
        return (((((IC + SZ - 1) / SZ) * SZ) + vlen_ - 1) / vlen_) * vlen_;
    }

protected:
    int BIC, ABIC;

    static constexpr int SZ = 8;
    static constexpr int BITS = sizeof(char) * SZ;
    static constexpr int ELEM_SIZE = sizeof(int);
};

} // namespace implementation
} // namespace xnor_nn

#endif // DIRECT_BASE_HPP
