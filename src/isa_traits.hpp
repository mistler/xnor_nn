#ifndef ISA_TRAITS_HPP
#define ISA_TRAITS_HPP

namespace xnor_nn {
namespace isa {

template<typename isa> struct isa_traits {};

#ifdef __x86_64__
struct isa_avx512{};
struct isa_avx2{};
struct isa_avx{};
struct isa_sse4_2{};
struct isa_sse2{};

template<> struct isa_traits<isa_avx512> { enum {vlen = 512}; };
template<> struct isa_traits<isa_avx2> { enum {vlen = 256}; };
template<> struct isa_traits<isa_avx> { enum {vlen = 256}; };
template<> struct isa_traits<isa_sse4_2> { enum {vlen = 128}; };
template<> struct isa_traits<isa_sse2> { enum {vlen = 128}; };
#elif defined __arm__
struct isa_neon{};
template<> struct isa_traits<isa_neon> { enum {vlen = 128}; };
#endif
struct isa_default{};
template<> struct isa_traits<isa_default> { enum {vlen = 64}; };

} // namespace isa
} // naespace xnor_nn

#endif // ISA_TRAITS_HPP
