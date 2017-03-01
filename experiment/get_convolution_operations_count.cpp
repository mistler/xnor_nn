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

constexpr int BICI = sizeof(int);
constexpr int SZ = 8;
constexpr int VLEN = 256;

static constexpr int constexpr_getICO(int IC) {
    return ((IC + BICI - 1) / BICI + SZ - 1) / SZ;
}

static constexpr int constexpr_getOCI() {
    return VLEN / SZ / BICI;
}

static constexpr int constexpr_getOCO(int OC) {
    return (OC + constexpr_getOCI() - 1) / constexpr_getOCI();
}

static constexpr int getICO(int IC) {
    return ((IC + BICI - 1) / BICI + SZ - 1) / SZ;
}


int main(){
    const std::vector<convolution_params> params =
    {
        // AlexNet
        //{ 1, 64, 3, 224, 224, 11, 11, 4, 4, 2, 2 }, // conv1
        //{ 1, 192, 64, 27, 27, 5, 5, 1, 1, 2, 2, }, // conv2
        //{ 1, 384, 192, 13, 13, 3, 3, 1, 1, 1, 1, }, // conv3
        //{ 1, 256, 384, 13, 13, 3, 3, 1, 1, 1, 1, }, // conv4
        { 1, 256, 256, 13, 13, 3, 3, 1, 1, 1, 1, }, // conv5
        //{ 256, 256, 384, 13, 13, 3, 3, 1, 1, 1, 1, }, // conv4
    };

    // Main loop
    int index = 0;
    for (const convolution_params &p : params) {

        const int OH = (p.IH + 2*p.PH - p.KH) / p.SH + 1;
        const int OW = (p.IW + 2*p.PW - p.KW) / p.SW + 1;
        const int OCO = constexpr_getOCO(p.OC);
        const int OCI = constexpr_getOCI();
        const int ICO = getICO(p.IC);

        std::cout << ++index << " of " << params.size() << ": " ;
        p.print(std::cout);


        unsigned long long int counter = 0;
        unsigned long long int writes = 0;

    for (int mb = 0; mb < p.MB; mb++)
    for (int oco = 0; oco < OCO; oco++)
    {
        counter = 0;
        for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++) {
            for (int kh = 0; kh < p.KH; kh++)
            for (int kw = 0; kw < p.KW; kw++) {
                const int ih = oh*p.SH - p.PH + kh;
                const int iw = ow*p.SW - p.PW + kw;

                if (ih < 0 || iw < 0) continue;
                if (ih >= p.IH || iw >= p.IW) continue;

                for (int ico = 0; ico < ICO; ico++)
                //for (int oci = 0; oci < OCI; oci++)
                    counter += 32;
            }
            for (int i = 0; i < OCI; i++)
                writes++;
        }
    }

        unsigned long long int simple = (p.KH*p.KW*OH*OW
                - 2*p.PH*p.KW*OW - 2*p.PW*p.KH*OH + 4*p.PW*p.PH)*p.IC;

        std::cout << "Inner loop interations: " << counter << std::endl;
        std::cout << "Writes: " << writes << std::endl;
        std::cout << "Simple: " << simple << std::endl;

        std::cout << std::endl;

    }
    return 0;
}
