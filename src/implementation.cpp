#include "implementation.hpp"

#include "utils/logger.hpp"
#include "utils/timer.hpp"

#include "bcast_convolution.hpp"
#include "template_convolution.hpp"
#include "direct_convolution.hpp"

#include "bcast_binarize_weights.hpp"
#include "direct_binarize_weights.hpp"
#include "bcast_binarize_data.hpp"
#include "direct_binarize_data.hpp"

#include "reference_binarize_data.hpp"
#include "reference_binarize_weights.hpp"
#include "reference_convolution.hpp"
#include "reference_calculate_k.hpp"

using Logger = xnor_nn::utils::Logger;

namespace xnor_nn {
namespace implementation {

std::vector<Implementation*> Implementations() {
    static std::vector<Implementation*> impls = {
        new BcastConvolution(),
        new TemplateConvolution(),
        new DirectConvolution(),

        new BcastBinarizeWeights(),
        new DirectBinarizeWeights(),
        new BcastBinarizeData(),
        new DirectBinarizeData(),

        new ReferenceBinarizeWeights(),
        new ReferenceBinarizeData(),
        new ReferenceConvolution(),
        new ReferenceCalculateK(),
    };
    return impls;
};

} // namespace implementation
} // namespace xnor_nn
