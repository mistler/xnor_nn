#include <cstdio>

#define DIV_R_UP(a, b) ((a) + (b) - 1) / (b)

int main() {
    const int IC = 252;
    const int IH = 2;
    const int IW = 3;
    const int SZ = 8;

    float * __restrict__ from = new float[IC*IH*IW];
    unsigned char * __restrict__ to = new unsigned char[DIV_R_UP(IC, SZ)*IH*IW];

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

    const int OC = DIV_R_UP(IC, SZ);

    for (int ih = 0; ih < IH; ih++)
    for (int iw = 0; iw < IW; iw++)
    for (int oc = 0; oc < OC; oc++) {
        unsigned char out{0};
        const int LEN = oc == OC - 1 ? (IC % SZ) : SZ;
        for (int ic = 0; ic < LEN; ic++) {
            int from_idx = (ic*IH + ih)*IW + iw;
            char tmp = from_i[from_idx] >> 31;
            out <<= 1;
            out |= tmp;
        }
        if (LEN != SZ) out <<= SZ-LEN;
        int to_idx = (ih*IW + iw)*OC + oc;
        to[to_idx] = out;
    }


    for (int ih = 0; ih < IH; ih++)
    for (int iw = 0; iw < IW; iw++) {
        for (int ic = 0; ic < OC; ic++)
        for (int c = SZ - 1; c >= 0; c--) {
            unsigned char ch = (to[(ih*IW + iw)*OC + ic] & (1 << c)) >> c;
            printf("%d", ch);
        }
        printf("\n");
    }

    delete[] from;
    delete[] to;

    return 0;
}
