#include <cstdio>
#include "timer.hpp"

inline unsigned long long rdtsc() {
    unsigned int lo, hi;
    asm volatile("rdtsc\n" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)hi << 32) | lo;
}

int main() {
    const int ELEMS = 256 * 512; // 512k
    const int WARM_UP = 128;
    const int N = 1024;

    float * __restrict__ a = new float[ELEMS];
    float * __restrict__ b = new float[ELEMS];
    float * __restrict__ c = new float[ELEMS];

    float current_value = -1.f;
    for (int i = 0; i < ELEMS; i++)
        a[i] = (current_value = -current_value);
    for (int i = 0; i < ELEMS; i++)
        b[i] = (current_value = -current_value);
    for (int i = 0; i < ELEMS; i++)
        c[i] = (current_value = -current_value);

    xnor_nn::utils::Timer timer;

    // Warm up
    for (int w = 0; w < WARM_UP; w++)
    for (int i = 0; i < ELEMS; i++) {
        c[i] += a[i]*b[i];
    }

    timer.start();
    unsigned long long t = rdtsc();

    for (int k = 0; k < N; k++)
    for (int i = 0; i < ELEMS; i++) {
        c[i] += a[i]*b[i];
    }

    t = rdtsc() - t;
    timer.stop();

    printf("cpu: %lf ticks\n", (double)t / N / ELEMS);
    printf("timer: %lf micros\n", (double)timer.micros() / N / ELEMS);

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
