#include <cstdlib>

#define data_t unsigned long long int

void xnor(const data_t *src, const data_t *weights, data_t *dst);
void popcnt(const data_t *src, const data_t *weights, data_t *dst);

int main(void){
    data_t *src = new data_t[(size_t)4096*4096*2];
    data_t *weights = new data_t[(size_t)4096*4096*2];
    data_t *dst = new data_t[(size_t)4096*4096*2];
    xnor(src, weights, dst);
    popcnt(src, weights, dst);
    return 0;
}
