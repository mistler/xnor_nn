#include <cstdio>

inline unsigned long long rdtsc() {
    unsigned int lo, hi;
    asm volatile("rdtsc\n" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)hi << 32) | lo;
}

#if 0
inline long long get_tsc() {
    long long rc;
    asm bolatile("rdtsc\n"
        "mov %%eax, (%0)\n"
        "mov %%edx, 4(%0)\n"
        :
        : "c"(&rc), "a"(-1), "d"(-1));
    return rc;
}
#endif

int main() {
    const int ELEMS = 256 * 512; // 512k
    const int WARM_UP = 128;
    const int N = 1024 * 8;

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

    // Warm up
    for (int w = 0; w < WARM_UP; w++)
    for (int i = 0; i < ELEMS; i++)
        c[i] += a[i]*b[i];

    unsigned long long s = rdtsc();

    for (int k = 0; k < N; k++)
    for (int i = 0; i < ELEMS; i++)
        c[i] += a[i]*b[i];

    s = rdtsc() - s;

    printf("%lf\n", (double)s / ELEMS / N);

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
