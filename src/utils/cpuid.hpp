#ifndef CPUID_HPP
#define CPUID_HPP

#include <cstdlib>
#ifdef __x86_64__
#include <cpuid.h>
#elif defined __arm__
#include <fcntl.h>
#include <elf.h>
#include <unistd.h>
#endif

namespace xnor_nn {
namespace utils {

class Cpuid {
public:

    static int vlen() {
        Cpuid &inst = instance();
#ifdef __x86_64__
        if (inst._avx512f) return 512;
        if (inst._avx) return 256;
        if (inst._sse42) return 128;
        return 64;
#elif defined __arm__
        if (inst._neon) return 128;
        return 64;
#endif
    }

#define GETTER(NAME) \
        static bool NAME() { Cpuid &inst = instance(); return inst._##NAME; }

#ifdef __x86_64__
    //  Misc.
    GETTER(mmx)
    GETTER(x64)
    GETTER(abm)
    GETTER(rdrand)
    GETTER(bmi1)
    GETTER(bmi2)
    GETTER(adx)
    GETTER(prefetchwt1)

    // SIMD: 128-bit
    GETTER(sse)
    GETTER(sse2)
    GETTER(sse3)
    GETTER(ssse3)
    GETTER(sse41)
    GETTER(sse42)
    GETTER(sse4a)
    GETTER(aes)
    GETTER(sha)

    // SIMD: 256-bit
    GETTER(avx)
    GETTER(xop)
    GETTER(fma3)
    GETTER(fma4)
    GETTER(avx2)

    // SIMD: 512-bit
    GETTER(avx512f) // AVX512 Foundation
    GETTER(avx512cd) // AVX512 Conflict Detection
    GETTER(avx512pf) // AVX512 Prefetch
    GETTER(avx512er) // AVX512 Exponential + Reciprocal
    GETTER(avx512vl) // AVX512 Vector Length Extensions
    GETTER(avx512bw) // AVX512 Byte + Word
    GETTER(avx512dq) // AVX512 Doubleword + Quadword
    GETTER(avx512ifma) // AVX512 Integer 52-bit Fused Multiply-Add
    GETTER(avx512vbmi) // AVX512 Vector Byte Manipulation Instructions
#elif defined __arm__
    GETTER(neon)
#endif

#undef GETTER

private:

    Cpuid() {
        read_cpuid();
        read_environment();
    }

    static Cpuid &instance() {
        static Cpuid instance;
        return instance;
    }

    void read_environment() {
        char *env = getenv("XNOR_NN_ISA");
        if (env != NULL && env[0] >= '0' && env[0] <= '9') {
            environment_isa_ = env[0] - '0';
#ifdef __x86_64__
            switch (environment_isa_) {
                case 0: _avx512f = false; _avx = false; break;
                case 1: _avx512f = false; break;
            }
#elif defined __arm__
            switch (environment_isa_) {
                case 0: _neon = false; break;
            }
#endif
        }
    }

#ifdef __x86_64__
    void cpuid(int info[4], int InfoType){
        __cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
    }

    void read_cpuid() {
        int info[4];
        cpuid(info, 0);
        int nIds = info[0];

        cpuid(info, 0x80000000);
        unsigned nExIds = info[0];

        //  Detect Features
        if (nIds >= 0x00000001) {
            cpuid(info, 0x00000001);
            _mmx = (info[3] & ((int)1 << 23)) != 0;
            _sse = (info[3] & ((int)1 << 25)) != 0;
            _sse2 = (info[3] & ((int)1 << 26)) != 0;
            _sse3 = (info[2] & ((int)1 <<  0)) != 0;
            _ssse3 = (info[2] & ((int)1 <<  9)) != 0;
            _sse41 = (info[2] & ((int)1 << 19)) != 0;
            _sse42 = (info[2] & ((int)1 << 20)) != 0;
            _aes = (info[2] & ((int)1 << 25)) != 0;
            _avx = (info[2] & ((int)1 << 28)) != 0;
            _fma3 = (info[2] & ((int)1 << 12)) != 0;
            _rdrand = (info[2] & ((int)1 << 30)) != 0;
        }
        if (nIds >= 0x00000007) {
            cpuid(info, 0x00000007);
            _avx2 = (info[1] & ((int)1 <<  5)) != 0;
            _bmi1 = (info[1] & ((int)1 <<  3)) != 0;
            _bmi2 = (info[1] & ((int)1 <<  8)) != 0;
            _adx = (info[1] & ((int)1 << 19)) != 0;
            _sha = (info[1] & ((int)1 << 29)) != 0;
            _prefetchwt1 = (info[2] & ((int)1 <<  0)) != 0;
            _avx512f = (info[1] & ((int)1 << 16)) != 0;
            _avx512cd = (info[1] & ((int)1 << 28)) != 0;
            _avx512pf = (info[1] & ((int)1 << 26)) != 0;
            _avx512er = (info[1] & ((int)1 << 27)) != 0;
            _avx512vl = (info[1] & ((int)1 << 31)) != 0;
            _avx512bw = (info[1] & ((int)1 << 30)) != 0;
            _avx512dq = (info[1] & ((int)1 << 17)) != 0;
            _avx512ifma = (info[1] & ((int)1 << 21)) != 0;
            _avx512vbmi = (info[2] & ((int)1 <<  1)) != 0;
        }
        if (nExIds >= 0x80000001) {
            cpuid(info, 0x80000001);
            _x64 = (info[3] & ((int)1 << 29)) != 0;
            _abm = (info[2] & ((int)1 <<  5)) != 0;
            _sse4a = (info[2] & ((int)1 <<  6)) != 0;
            _fma4 = (info[2] & ((int)1 << 16)) != 0;
            _xop = (info[2] & ((int)1 << 11)) != 0;
        }
    }
#elif defined __arm__
    void read_cpuid() {
        Elf32_auxv_t auxv;

        auto cpufile = open("/proc/self/auxv", O_RDONLY);
        if (cpufile < 0) {
            return;
        }

        const auto size_auxv_t = sizeof(Elf32_auxv_t);
        while (read(cpufile, &auxv, size_auxv_t) == size_auxv_t) {
            if (auxv.a_type == AT_HWCAP) {
                _neon = (auxv.a_un.a_val & 4096) != 0;
                break;
            }
        }
        close(cpufile);
    }
#endif

private:

    int environment_isa_;
#ifdef __x86_64__
    //  Misc.
    bool _mmx = false;
    bool _x64 = false;
    bool _abm = false; // Advanced Bit Manipulation
    bool _rdrand = false;
    bool _bmi1 = false;
    bool _bmi2 = false;
    bool _adx = false;
    bool _prefetchwt1 = false;

    // SIMD: 128-bit
    bool _sse = false;
    bool _sse2 = false;
    bool _sse3 = false;
    bool _ssse3 = false;
    bool _sse41 = false;
    bool _sse42 = false;
    bool _sse4a = false;
    bool _aes = false;
    bool _sha = false;

    // SIMD: 256-bit
    bool _avx = false;
    bool _xop = false;
    bool _fma3 = false;
    bool _fma4 = false;
    bool _avx2 = false;

    // SIMD: 512-bit
    bool _avx512f = false; // AVX512 Foundation
    bool _avx512cd = false; // AVX512 Conflict Detection
    bool _avx512pf = false; // AVX512 Prefetch
    bool _avx512er = false; // AVX512 Exponential + Reciprocal
    bool _avx512vl = false; // AVX512 Vector Length Extensions
    bool _avx512bw = false; // AVX512 Byte + Word
    bool _avx512dq = false; // AVX512 Doubleword + Quadword
    bool _avx512ifma = false; // AVX512 Integer 52-bit Fused Multiply-Add
    bool _avx512vbmi = false; // AVX512 Vector Byte Manipulation Instructions
#elif defined __arm__
    bool _neon = false;
#endif

};

} // namespace utils
} // namespace xnor_nn

#endif // CPUID_HPP
