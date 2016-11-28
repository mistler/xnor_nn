#include <immintrin.h>

unsigned long long int f(unsigned long long int N, float *s, float *w, float*o) {
    unsigned long long int dst_i = 0;
    __m256 v_ones = _mm256_loadu_ps(o);
    for (unsigned long long int k = 0; k < N; k++) {
        /*
        __m256 v_src = _mm256_load_ps(a);
        __m256 v_weights = _mm256_load_ps(w);

        __m256 v_xor = _mm256_xor_ps(v_src, v_weights);
        __m256i v_xnor = _mm256_castps_si256(_mm256_xor_ps(v_xor, v_ones));

        dst_i += __builtin_popcountll(_mm256_extract_epi64(v_xnor, 0));
        dst_i += __builtin_popcountll(_mm256_extract_epi64(v_xnor, 1));
        dst_i += __builtin_popcountll(_mm256_extract_epi64(v_xnor, 2));
        dst_i += __builtin_popcountll(_mm256_extract_epi64(v_xnor, 3));
        */
        asm volatile (
            "vmovaps (%0), %%ymm0\n\t"
            "vxorps (%1), %%ymm0, %%ymm0\n\t"
            "vxorps %%ymm2, %%ymm0, %%ymm0\n\t"
            "vextractf128 $0x1,%%ymm0,%%xmm1\n\t"
            "vpextrq $0x1,%%xmm1,%%rcx\n\t"
            "vpextrq $0x1,%%xmm0,%%rax\n\t"
            "vmovq %%xmm1,%%rdx\n\t"
            "vmovq %%xmm0, %%rsi\n\t"
            "popcnt %%rsi,%%rsi\n\t"
            "popcnt %%rax,%%rax\n\t"
            "popcnt %%rcx,%%rcx\n\t"
            "popcnt %%rdx,%%rdx\n\t"
            "add %%rdx, %%rax\n\t"
            "add %%rcx, %%rax\n\t"
            "add %%rsi, %%rax\n\t"
            :
            : "r" (s), "r" (w)
            : "%ymm0", "%ymm1", "%ymm2", "%rax", "%rbx", "%rcx", "%rsi"
        );

        /*
  0,37 │       vmovap (%r15,%rax,4),%ymm0
  6,64 │       vxorps (%r14,%rax,4),%ymm0,%ymm0
 36,03 │       vxorps %ymm2,%ymm0,%ymm0
  1,19 │       vmovq  %xmm0,%rsi
  2,17 │       vpextr $0x1,%xmm0,%rax
  3,41 │       popcnt %rsi,%rsi
  3,11 │       vextra $0x1,%ymm0,%xmm0
  0,02 │       popcnt %rax,%rax
  1,68 │       vmovq  %xmm0,%rdx
  0,04 │       add    %rsi,%rax
  2,86 │       vpextr $0x1,%xmm0,%rcx
  0,30 │       popcnt %rcx,%rcx
  2,19 │       popcnt %rdx,%rdx
  0,89 │       add    %rdx,%rax
  0,26 │       add    %rcx,%rax
  1,13 │       add    %rax,%r12
  2,08 │       cmp    %r11d,%edi
  */

    }
    return dst_i;
}
