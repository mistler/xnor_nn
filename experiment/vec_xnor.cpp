#include <cstdlib>

#define data_t unsigned long long int

void xnor(const data_t *__restrict__ src,
        const data_t *__restrict__ weights,
        data_t *__restrict__ dst){
    for(size_t i = 0; i < (size_t)4096 * 4096 * 2; i++){
        dst[i] = ~(src[i] ^ weights[i]);
    }
}

void popcnt(const data_t *__restrict__ src,
        const data_t *__restrict__ weights,
        data_t *__restrict__ dst){
    for(size_t i = 0; i < (size_t)4096 * 4096 * 2; i++){
        dst[i] += __builtin_popcountll(src[i]);
    }
}
