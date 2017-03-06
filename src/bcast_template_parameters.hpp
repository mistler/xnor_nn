#ifndef BCAST_TEMPLATE_PARAMETERS
#define BCAST_TEMPLATE_PARAMETERS

#include "xnor_nn_types.h"

namespace xnor_nn {
namespace bcast {

// oc, ic, ih, iw, kh, kw, sh, sw, ph, pw

constexpr int template_parameters[][10] = {
    // AlexNet
    { 96, 3, 227, 227, 11, 11, 4, 4, 0, 0 },
    { 256, 96, 27, 27, 5, 5, 1, 1, 2, 2 },
    { 384, 256, 13, 13, 3, 3, 1, 1, 1, 1 },
    { 384, 384, 13, 13, 3, 3, 1, 1, 1, 1 },
    { 256, 384, 13, 13, 3, 3, 1, 1, 1, 1 },
    // Task
    { 32, 1, 60, 61, 3, 3, 1, 1, 0, 0 },
    { 32, 32, 20, 20, 3, 3, 1, 1, 0, 0 },
};

#define _TEMPLATE_PARAMS xnor_nn::bcast::template_parameters

#define BCAST_TEMPLATE_MATCHES(C, INDEX) \
        C->oc == _TEMPLATE_PARAMS[INDEX][0] \
        && C->ic == _TEMPLATE_PARAMS[INDEX][1] \
        && C->ih == _TEMPLATE_PARAMS[INDEX][2] \
        && C->iw == _TEMPLATE_PARAMS[INDEX][3] \
        && C->kh == _TEMPLATE_PARAMS[INDEX][4] \
        && C->kw == _TEMPLATE_PARAMS[INDEX][5] \
        && C->sh == _TEMPLATE_PARAMS[INDEX][6] \
        && C->sw == _TEMPLATE_PARAMS[INDEX][7] \
        && C->ph == _TEMPLATE_PARAMS[INDEX][8] \
        && C->pw == _TEMPLATE_PARAMS[INDEX][9]

#define BCAST_TEMPLATE_PARAMS(INDEX) \
        _TEMPLATE_PARAMS[INDEX][0], \
        _TEMPLATE_PARAMS[INDEX][1], \
        _TEMPLATE_PARAMS[INDEX][2], \
        _TEMPLATE_PARAMS[INDEX][3], \
        _TEMPLATE_PARAMS[INDEX][4], \
        _TEMPLATE_PARAMS[INDEX][5], \
        _TEMPLATE_PARAMS[INDEX][6], \
        _TEMPLATE_PARAMS[INDEX][7], \
        _TEMPLATE_PARAMS[INDEX][8], \
        _TEMPLATE_PARAMS[INDEX][9]

#define _BCAST_TEMPLATE_INSTANTIATE(ISA, INDEX) \
template xnor_nn_status_t BcastConvolution::exec_template <ISA, \
        BCAST_TEMPLATE_PARAMS(INDEX) \
        >(const xnor_nn_convolution_t *c, xnor_nn_resources_t res);

#define BCAST_TEMPLATE_INSTANTIATE(ISA) \
        _BCAST_TEMPLATE_INSTANTIATE(ISA, 0) \
        _BCAST_TEMPLATE_INSTANTIATE(ISA, 1) \
        _BCAST_TEMPLATE_INSTANTIATE(ISA, 2) \
        _BCAST_TEMPLATE_INSTANTIATE(ISA, 3) \
        _BCAST_TEMPLATE_INSTANTIATE(ISA, 4) \
        _BCAST_TEMPLATE_INSTANTIATE(ISA, 5) \
        _BCAST_TEMPLATE_INSTANTIATE(ISA, 6)

#define _BCAST_TEMPLATE_ASSIGN(CONV, ISA, INDEX) \
        if (BCAST_TEMPLATE_MATCHES(CONV, INDEX)) { \
            CONV->forward = exec_template<ISA, BCAST_TEMPLATE_PARAMS(INDEX)>; \
            return; \
        }

#define BCAST_TEMPLATE_ASSIGN(CONV, ISA) \
        _BCAST_TEMPLATE_ASSIGN(CONV, ISA, 0) \
        _BCAST_TEMPLATE_ASSIGN(CONV, ISA, 1) \
        _BCAST_TEMPLATE_ASSIGN(CONV, ISA, 2) \
        _BCAST_TEMPLATE_ASSIGN(CONV, ISA, 3) \
        _BCAST_TEMPLATE_ASSIGN(CONV, ISA, 4) \
        _BCAST_TEMPLATE_ASSIGN(CONV, ISA, 5) \
        _BCAST_TEMPLATE_ASSIGN(CONV, ISA, 6)

} // namespace bcast
} // namespace xnor_nn

#endif // BCAST_TEMPLATE_PARAMETERS
