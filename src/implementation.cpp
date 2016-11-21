#include "implementation.hpp"

#include "direct_binarize_data.hpp"
#include "direct_binarize_weights.hpp"
#include "direct_convolution.hpp"
#include "reference_binarize_data.hpp"
#include "reference_binarize_weights.hpp"
#include "reference_convolution.hpp"
#include "reference_calculate_k.hpp"

namespace xnor_nn {
namespace implementation {

std::vector<Implementation*> Implementations() {
    static std::vector<Implementation*> impls = {
        new DirectBinarizeWeightsChar(),
        new DirectBinarizeDataChar(),
        new DirectConvolution(),
        new ReferenceBinarizeWeightsCopyOnFloat(),
        new ReferenceBinarizeDataCopyOnFloat(),
        new ReferenceConvolution(),
        new ReferenceCalculateK(),
    };
    return impls;
};

}
}
