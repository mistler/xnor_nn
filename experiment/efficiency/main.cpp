#include <iostream>
#include <vector>
#include <cstdlib>

#include <immintrin.h>

unsigned long long int f(unsigned long long int N, float *s, float *w, float *o);

inline unsigned long long rdtsc() {
    unsigned int lo, hi;
    asm volatile("rdtsc\n" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)hi << 32) | lo;
}

#define N 8

int main() {
    std::vector<unsigned long long int> iterations = {
// MB: 1, OC: 64, IC: 3, IH: 224, IW: 224, KH: 11, KW: 11, SH: 4, SW: 4, PH: 2, PW: 2
        23193856,
// Vector length (bytes): 32

// MB: 1, OC: 192, IC: 64, IH: 27, IW: 27, KH: 5, KW: 5, SH: 1, SW: 1, PH: 2, PW: 2
        3195072,
// Vector length (bytes): 32

// MB: 1, OC: 384, IC: 192, IH: 13, IW: 13, KH: 3, KW: 3, SH: 1, SW: 1, PH: 1, PW: 1
        525696,
// Vector length (bytes): 32

// MB: 1, OC: 256, IC: 384, IH: 13, IW: 13, KH: 3, KW: 3, SH: 1, SW: 1, PH: 1, PW: 1
        700928,
// Vector length (bytes): 32

// MB: 1, OC: 256, IC: 256, IH: 13, IW: 13, KH: 3, KW: 3, SH: 1, SW: 1, PH: 1, PW: 1
        350464,
// Vector length (bytes): 32

// MB: 256, OC: 256, IC: 384, IH: 13, IW: 13, KH: 3, KW: 3, SH: 1, SW: 1, PH: 1, PW: 1
        179437568,
// Vector length (bytes): 32
    };

    float *a = (float*)aligned_alloc(64, 256*4);
    a[0] = 3.141592f;

    for (auto i = iterations.begin(); i != iterations.end(); i++) {
        std::cout << "Operations: " << *i << std::endl;

        unsigned long long int t = rdtsc();

        for (int n = 0; n < N; n++) {
            f(*i, a, a+8, a+16);
        }

        t = rdtsc() - t;

        std::cout << "Rdtsc total: " << (double)t / N << std::endl;
        std::cout << "Rdtsc per iter: " << (double)t / N / *i << std::endl;

        std::cout << std::endl;
    }

    free(a);
    return 0;
}
