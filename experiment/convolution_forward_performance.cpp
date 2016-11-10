#include <cstdio>

int main() {
    const int IC = 256*1000;
    const int IH = 8;
    const int IW = 8;

    unsigned int * __restrict__ from1 = new unsigned int[IC*IH*IW];
    unsigned int * __restrict__ from2 = new unsigned int[IC*IH*IW];

    float * __restrict__ to = new float[IC*IH*IW];

    float current_value = -1.f;
    for (int ih = 0; ih < IH; ih++)
    for (int iw = 0; iw < IW; iw++)
    for (int ic = 0; ic < IC; ic++) {
        int from_idx = (ih*IW + iw)*IC + ic;
        from1[from_idx] = from2[from_idx] = (current_value = -current_value);
    }

    for (int ih = 0; ih < IH; ih++)
    for (int iw = 0; iw < IW; iw++) {
        int to_idx = ih*IW + iw;
        to[to_idx] = 0.f;
        for (int ic = 0; ic < IC; ic++) {
            int from_idx = (ih*IW + iw)*IC + ic;
            unsigned int f1 = from1[from_idx];
            unsigned int f2 = from2[from_idx];

            unsigned int result = ~(f1 ^ f2);
            //to[to_idx] += __builtin_popcount(result);
            to[to_idx] += result;
        }
    }

    /*
    for (int ih = 0; ih < IH; ih++)
    for (int iw = 0; iw < IW; iw++) {
        for (int ic = 0; ic < IC / SZ; ic++)
            printf("%X ", to[(ic*IH + ih)*IW + iw]);
        printf("\n");
    }
    */

    delete[] from1;
    delete[] from2;
    delete[] to;

    return 0;
}
