#include <cstdio>

int main() {
    const int IC = 256;
    const int IH = 2;
    const int IW = 2;
    const int SZ = 8;

    float * __restrict__ from = new float[IC*IH*IW];
    unsigned char * __restrict__ to = new unsigned char[IC/SZ*IH*IW];

    unsigned int * __restrict__ from_i = ((unsigned int*)from);

    float current_value = -1.f;
    for (int ih = 0; ih < IH; ih++)
    for (int iw = 0; iw < IW; iw++)
    for (int ic = 0; ic < IC; ic++)
        from[(ic*IH + ih)*IW + iw] = (current_value = -current_value);


    /*
#   pragma ivdep
    for (int ih = 0; ih < IH; ih++)
    for (int iw = 0; iw < IW; iw++)
    for (int ic = 0; ic < IC; ic++) {
        int from_idx = (ic*IH + ih)*IW + iw;
        int to_idx = (ih*IW + iw)*IC + ic;
        from2[to_idx] = from[from_idx];
    }
    */

#   pragma ivdep
    for (int ih = 0; ih < IH; ih++)
    for (int iw = 0; iw < IW; iw++)
    for (int oc = 0; oc < IC / SZ; oc++) {
        unsigned char out{0};
        for (int ic = 0; ic < SZ; ic++) {
            int from_idx = (ih*IW + iw)*IC + oc*SZ + ic;
            char tmp = from_i[from_idx] >> 31;
            out <<= 1;
            out |= tmp;
        }
        int to_idx = (oc*IH + ih)*IW + iw;
        to[to_idx] = out;
    }


    for (int ih = 0; ih < IH; ih++)
    for (int iw = 0; iw < IW; iw++) {
        for (int ic = 0; ic < IC / SZ; ic++)
            printf("%X ", to[(ic*IH + ih)*IW + iw]);
        printf("\n");
    }

    delete[] from;
    delete[] to;

    return 0;
}
