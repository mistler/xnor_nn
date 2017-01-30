#ifndef DIRECT_TEMPLATE_PARAMETERS
#define DIRECT_TEMPLATE_PARAMETERS

#include "xnor_nn_types.h"

namespace xnor_nn {
namespace direct {

// oc, ic, ih, iw, kh, kw, sh, sw, ph, pw

constexpr int template_parameters[][10] = {
    // AlexNet
    { 96, 3, 227, 227, 11, 11, 4, 4, 0, 0 },
    { 256, 96, 27, 27, 5, 5, 1, 1, 2, 2 },
    { 384, 256, 13, 13, 3, 3, 1, 1, 1, 1 },
    { 384, 384, 13, 13, 3, 3, 1, 1, 1, 1 },
    { 256, 384, 13, 13, 3, 3, 1, 1, 1, 1 },
};

#define _TEMPLATE_PARAMS xnor_nn::direct::template_parameters

#define DIRECT_TEMPLATE_MATCHES(C, INDEX) \
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

#define DIRECT_TEMPLATE_PARAMS(INDEX) \
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

#define _DIRECT_TEMPLATE_INSTANTIATE(NAME, INDEX) \
template xnor_nn_status_t DirectConvolution::exec_##NAME##_template < \
        DIRECT_TEMPLATE_PARAMS(INDEX) \
        >(const xnor_nn_convolution_t *c, xnor_nn_resources_t res);

#define DIRECT_TEMPLATE_INSTANTIATE(NAME) \
        _DIRECT_TEMPLATE_INSTANTIATE(NAME, 0) \
        _DIRECT_TEMPLATE_INSTANTIATE(NAME, 1) \
        _DIRECT_TEMPLATE_INSTANTIATE(NAME, 2) \
        _DIRECT_TEMPLATE_INSTANTIATE(NAME, 3) \
        _DIRECT_TEMPLATE_INSTANTIATE(NAME, 4)

#define _DIRECT_TEMPLATE_ASSIGN(CONV, NAME, INDEX) \
        if (DIRECT_TEMPLATE_MATCHES(CONV, INDEX)) { \
            CONV->forward = \
                    exec_##NAME##_template<DIRECT_TEMPLATE_PARAMS(INDEX)>; \
            return; \
        }

#define DIRECT_TEMPLATE_ASSIGN(CONV, NAME) \
        _DIRECT_TEMPLATE_ASSIGN(CONV, NAME, 0) \
        _DIRECT_TEMPLATE_ASSIGN(CONV, NAME, 1) \
        _DIRECT_TEMPLATE_ASSIGN(CONV, NAME, 2) \
        _DIRECT_TEMPLATE_ASSIGN(CONV, NAME, 3) \
        _DIRECT_TEMPLATE_ASSIGN(CONV, NAME, 4)

} // namespace direct
} // namespace xnor_nn

#endif // DIRECT_TEMPLATE_PARAMETERS
