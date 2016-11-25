#include "xnor_nn.hpp"

int main(){
    const int MB = 8;
    const int IC = 64, OC = 128;
    const int IH = 13, IW = 13;
    const int OH = 13, OW = 13;
    const int KH = 3, KW = 3;
    const int SH = 1, SW = 1;
    const int PH = 1, PW = 1;

    // Usr data
    float *src = nullptr, *weights = nullptr;
    float *dst = nullptr;

    src = new float[MB*IC*IH*IW];
    weights = new float[OC*IC*KH*KW];
    dst = new float[MB*OC*OH*OW];

    // Convolution setup
    xnor_nn::Convolution convolution{xnor_nn_algorithm_direct,
            MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW, weights};
    delete[] weights;

    // Execution
    for (int i = 0; i < 3; i++) {
        convolution.forward(src, dst);
    }

    delete[] src;
    delete[] dst;

    return 0;
}
