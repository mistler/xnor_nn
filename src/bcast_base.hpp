#ifndef BCAST_BASE_HPP
#define BCAST_BASE_HPP

#include "implementation.hpp"

#include "xnor_nn_types.h"
#include "utils.h"

namespace xnor_nn {
namespace implementation {

class BcastBase : public Implementation {
public:
    virtual ~BcastBase();
    virtual bool isApplicable(const xnor_nn_convolution_t *c) const;
    virtual void setupConvolution(xnor_nn_convolution_t *c);

protected:
    int BIC, ABIC, ICO, OCO;

    // TODO: remove unused stuff
    static constexpr int SZ = 8;
    static constexpr int ELEM_SIZE = sizeof(char);
    static constexpr int BITS = ELEM_SIZE * SZ;
    static constexpr int BICI = sizeof(int);

    static constexpr int VLEN_BYTES = VLEN / SZ;
    static constexpr int OCI = VLEN_BYTES / BICI;

};

} // namespace implementation
} // namespace xnor_nn

#endif // BCAST_BASE_HPP
