#include <vector>
#include <iostream>
#include <cmath>

typedef struct {
    int MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW;

    void print(std::ostream &stream) const {
        stream
            << "MB: " << MB
            << ", OC: " << OC
            << ", IC: " << IC
            << ", IH: " << IH
            << ", IW: " << IW
            << ", KH: " << KH
            << ", KW: " << KW
            << ", SH: " << SH
            << ", SW: " << SW
            << ", PH: " << PH
            << ", PW: " << PW
            << std::endl;
    }
} convolution_params;

int main(){
    const std::vector<convolution_params> params =
    {
        // AlexNet
        { 1, 64, 3, 224, 224, 11, 11, 4, 4, 2, 2 }, // conv1
        { 1, 192, 64, 27, 27, 5, 5, 1, 1, 2, 2, }, // conv2
        { 1, 384, 192, 13, 13, 3, 3, 1, 1, 1, 1, }, // conv3
        { 1, 256, 384, 13, 13, 3, 3, 1, 1, 1, 1, }, // conv4
        { 1, 256, 256, 13, 13, 3, 3, 1, 1, 1, 1, }, // conv5
        { 256, 256, 384, 13, 13, 3, 3, 1, 1, 1, 1, }, // conv4
    };

    // Main loop
    int index = 0;
    for (const convolution_params &p : params) {

        std::cout << ++index << " of " << params.size() << ": ";
        p.print(std::cout);

        const int OH = (p.IH + 2*p.PH - p.KH) / p.SH + 1;
        const int OW = (p.IW + 2*p.PW - p.KW) / p.SW + 1;

        const int VEC_LENGTH = 32; // bytes
        const int BITS = 8;
        const int BIC = (p.IC + BITS - 1) / BITS;
        const int AIC = ((BIC + VEC_LENGTH - 1) / VEC_LENGTH) * VEC_LENGTH;

        const int VECTORS_IN_AIC = AIC / VEC_LENGTH;

        unsigned long long int counter = 0;

        for (int mb = 0; mb < p.MB; mb++)
        for (int oc = 0; oc < p.OC; oc++)
        for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++) {
            for (int kh = 0; kh < p.KH; kh++)
            for (int kw = 0; kw < p.KW; kw++) {
                if (oh*p.SH + kh < (p.PH > 0 ? p.PH : 0)) continue;
                if (ow*p.SW + kw < (p.PW > 0 ? p.PW : 0)) continue;

                if (oh*p.SH + kh >= p.IH + p.PH) continue;
                if (ow*p.SW + kw >= p.IW + p.PW) continue;

                const int ih = oh * p.SH - p.PH + kh;
                const int iw = ow * p.SW - p.PW + kw;

                for (int aic = 0; aic < VECTORS_IN_AIC; aic++) {
                    counter++;
                }
            }
        }

        std::cout << "Inner loop interations: " << counter << std::endl;
        std::cout << "Vector length (bytes): " << VEC_LENGTH << std::endl;

        std::cout << std::endl;

    }
    return 0;
}
