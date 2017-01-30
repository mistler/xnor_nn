#ifndef BCAST_BASE_HPP
#define BCAST_BASE_HPP

#include "implementation.hpp"

#include "xnor_nn_types.h"
#include "cpuid.hpp"

namespace xnor_nn {
namespace implementation {

class BcastBase : public Implementation {
public:
    virtual ~BcastBase();
    virtual bool isApplicable(const xnor_nn_convolution_t *c) const;
    virtual void setupConvolution(xnor_nn_convolution_t *c);

protected:

#ifdef VLEN
    static constexpr int constexpr_getICO(int IC) {
        return ((IC + BICI - 1) / BICI + SZ - 1) / SZ;
    }

    static constexpr int constexpr_getOCO(int OC) {
        return (OC + constexpr_getOCI() - 1) / constexpr_getOCI();
    }

    static constexpr int constexpr_getOCI() {
        return VLEN / SZ / BICI;
    }
#endif

    static constexpr int getICO(int IC) {
        return ((IC + BICI - 1) / BICI + SZ - 1) / SZ;
    }

    static int getOCO(int OC) {
        return (OC + getOCI() - 1) / getOCI();
    }

    static int getOCI() {
        return xnor_nn::utils::Cpuid::vlen() / SZ / BICI;
    }

protected:
    int BIC, ABIC, ICO, OCO, OCI;

    static constexpr int SZ = 8;
    static constexpr int BITS = sizeof(char) * SZ;
    static constexpr int BICI = sizeof(int);
};

} // namespace implementation
} // namespace xnor_nn

#endif // BCAST_BASE_HPP
