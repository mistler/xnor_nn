#ifndef ISA_TRAITS_HPP
#define ISA_TRAITS_HPP

namespace xnor_nn {
namespace isa {

#ifdef __x86_64__
struct isa_avx512{};
struct isa_avx2{};
struct isa_avx{};
struct isa_sse4_2{};
#elif defined __arm__
struct isa_neon{};
#endif
struct isa_default{};

} // namespace isa
} // naespace xnor_nn

#endif // ISA_TRAITS_HPP
