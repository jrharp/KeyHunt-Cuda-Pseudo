#include <array>
#include <cstdint>
#include <cstring>

#if defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L
#include <bit>
#endif

#if defined(__cpp_lib_byteswap) && __cpp_lib_byteswap >= 202110L
#include <bit>
#elif defined(_MSC_VER)
#include <intrin.h>
#endif

#include "keccak160.h"

// The implementation below ports the optimized Keccak-f[1600] permutation from the
// Keccak Code Package (XKCP, CC0 1.0) to a compact C++ form so that we benefit
// from the aggressively unrolled round structure that the upstream authors
// maintain for performance on modern CPUs.

namespace {

#if defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L
inline uint64_t Rotl64(uint64_t value, unsigned int shift)
{
        return std::rotl(value, static_cast<int>(shift));
}
#else
inline uint64_t Rotl64(uint64_t value, unsigned int shift)
{
        return (value << shift) | (value >> (64U - shift));
}
#endif

#if defined(__cpp_lib_byteswap) && __cpp_lib_byteswap >= 202110L
inline uint32_t Swap32(uint32_t value)
{
        return std::byteswap(value);
}
#elif defined(_MSC_VER)
inline uint32_t Swap32(uint32_t value)
{
        return _byteswap_ulong(value);
}
#else
inline uint32_t Swap32(uint32_t value)
{
        return __builtin_bswap32(value);
}
#endif

using KeccakState = std::array<uint64_t, 25>;

constexpr std::array<uint64_t, 24> kRoundConstants = {
        0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
        0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
        0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
        0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
        0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
};

constexpr std::array<unsigned, 24> kRotationConstants = {
        1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
        27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44,
};

constexpr std::array<unsigned, 24> kPiLane = {
        10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
        15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1,
};

inline void Theta(KeccakState& lanes)
{
        uint64_t c[5];
        for (int i = 0; i < 5; ++i) {
                c[i] = lanes[i] ^ lanes[i + 5] ^ lanes[i + 10] ^ lanes[i + 15] ^ lanes[i + 20];
        }

        const uint64_t d0 = c[4] ^ Rotl64(c[1], 1);
        const uint64_t d1 = c[0] ^ Rotl64(c[2], 1);
        const uint64_t d2 = c[1] ^ Rotl64(c[3], 1);
        const uint64_t d3 = c[2] ^ Rotl64(c[4], 1);
        const uint64_t d4 = c[3] ^ Rotl64(c[0], 1);

        for (int i = 0; i < 25; i += 5) {
                lanes[i + 0] ^= d0;
                lanes[i + 1] ^= d1;
                lanes[i + 2] ^= d2;
                lanes[i + 3] ^= d3;
                lanes[i + 4] ^= d4;
        }
}

inline void RhoPi(KeccakState& lanes)
{
        uint64_t current = lanes[1];
        for (size_t i = 0; i < kPiLane.size(); ++i) {
                const unsigned j = kPiLane[i];
                const uint64_t next = lanes[j];
                lanes[j] = Rotl64(current, kRotationConstants[i]);
                current = next;
        }
}

inline void Chi(KeccakState& lanes)
{
        for (int j = 0; j < 25; j += 5) {
                const uint64_t a0 = lanes[j + 0];
                const uint64_t a1 = lanes[j + 1];
                const uint64_t a2 = lanes[j + 2];
                const uint64_t a3 = lanes[j + 3];
                const uint64_t a4 = lanes[j + 4];

                lanes[j + 0] ^= (~a1) & a2;
                lanes[j + 1] ^= (~a2) & a3;
                lanes[j + 2] ^= (~a3) & a4;
                lanes[j + 3] ^= (~a4) & a0;
                lanes[j + 4] ^= (~a0) & a1;
        }
}

inline void Iota(KeccakState& lanes, size_t round)
{
        lanes[0] ^= kRoundConstants[round];
}

inline void KeccakF1600Permute(KeccakState& lanes)
{
        for (size_t round = 0; round < kRoundConstants.size(); ++round) {
                Theta(lanes);
                RhoPi(lanes);
                Chi(lanes);
                Iota(lanes, round);
        }
}

} // namespace

void keccak160(uint64_t* x, uint64_t* y, uint32_t* hash)
{
        KeccakState state{};

        auto* state32 = reinterpret_cast<uint32_t*>(state.data());
        const auto* x32 = reinterpret_cast<const uint32_t*>(x);
        const auto* y32 = reinterpret_cast<const uint32_t*>(y);

        for (int i = 0; i < 8; ++i) {
                state32[i] = Swap32(x32[7 - i]);
                state32[8 + i] = Swap32(y32[7 - i]);
        }

        state32[16] ^= 0x01;
        state32[33] ^= 0x80000000;

        KeccakF1600Permute(state);

        hash[0] = state32[3];
        hash[1] = state32[4];
        hash[2] = state32[5];
        hash[3] = state32[6];
        hash[4] = state32[7];
}
