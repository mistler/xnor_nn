#ifndef BCAST_BASE_HPP
#define BCAST_BASE_HPP

#include "implementation.hpp"

#include "xnor_nn_types.h"
#include "cpuid.hpp"
#include "utils.hpp"

namespace xnor_nn {
namespace implementation {

class BcastBase : public Implementation {
public:
    virtual ~BcastBase();
    virtual bool isApplicable(const xnor_nn_convolution_t *c) const;
    virtual void setupConvolution(xnor_nn_convolution_t *c);

protected:

    static constexpr int constexpr_getICO(int IC) {
        return utils::div_up(utils::div_up(IC, BICI), SZ);
    }

    static constexpr int constexpr_getOCO(int OC, int VLEN) {
        return utils::div_up(OC, constexpr_getOCI(VLEN));
    }

    static constexpr int constexpr_getOCI(int VLEN) {
        return VLEN / SZ / BICI;
    }

    static constexpr int getICO(int IC) {
        return utils::div_up(utils::div_up(IC, BICI), SZ);
    }

    static int getOCO(int OC) {
        return utils::div_up(OC, getOCI());
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
