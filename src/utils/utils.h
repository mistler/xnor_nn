#ifndef UTILS_H
#define UTILS_H

#ifdef __x86_64__
#define VLEN 32
#define ARCH_X86
#elif defined __arm__
#define VLEN 16
#define ARCH_ARM
#else
#define VLEN 4
#define ARCH_UNDEF
#endif

#endif // UTILS_H
