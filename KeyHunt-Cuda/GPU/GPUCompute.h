/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <cuda_runtime.h>
#include <stdint.h>

#if __CUDA_ARCH__ >= 350
#define GPU_LDG(ptr) __ldg(ptr)
#else
#define GPU_LDG(ptr) (*(ptr))
#endif

__device__ uint64_t* _2Gnx = nullptr;
__device__ uint64_t* _2Gny = nullptr;

__device__ uint64_t* Gx = nullptr;
__device__ uint64_t* Gy = nullptr;

// ---------------------------------------------------------------------------------------

__device__ __forceinline__ void LoadGeneratorPoint(uint64_t* __restrict__ out,
        const uint64_t* __restrict__ src)
{
#pragma unroll
        for (int limb = 0; limb < 4; ++limb) {
                out[limb] = GPU_LDG(src + limb);
        }
}

// ---------------------------------------------------------------------------------------

__device__ __forceinline__ int Test_Bit_Set_Bit(const uint8_t* __restrict__ buf, uint32_t bit)
{
        const uint32_t byte = bit >> 3;
        const uint8_t mask = static_cast<uint8_t>(1u << (bit & 7u));
        return (GPU_LDG(buf + byte) & mask) != 0;
}

// ---------------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t MurMurHash2(const void* __restrict__ key, int len, uint32_t seed)
{
        constexpr uint32_t m = 0x5bd1e995u;
        constexpr int r = 24;

        uint32_t h = seed ^ len;
        const uint8_t* data = static_cast<const uint8_t*>(key);
        while (len >= 4) {
                uint32_t k;
#if __CUDA_ARCH__ >= 350
                k = __ldg(reinterpret_cast<const uint32_t*>(data));
#else
                k = *reinterpret_cast<const uint32_t*>(data);
#endif
                k *= m;
                k ^= k >> r;
                k *= m;
                h *= m;
                h ^= k;
		data += 4;
		len -= 4;
	}
	switch (len) {
        case 3: h ^= static_cast<uint32_t>(data[2]) << 16;
                break;
        case 2: h ^= static_cast<uint32_t>(data[1]) << 8;
                break;
        case 1: h ^= data[0];
                h *= m;
                break;
        }

	h ^= h >> 13;
	h *= m;
	h ^= h >> 15;

        return h;
}

// ---------------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t FastRange(uint64_t value, uint64_t modulus, uint64_t reciprocal,
        uint32_t mask, uint32_t isPowerOfTwo)
{
        if (isPowerOfTwo) {
                return static_cast<uint32_t>(value) & mask;
        }

        const uint64_t quotient = __umul64hi(value, reciprocal);
        uint64_t remainder = value - quotient * modulus;
        if (remainder >= modulus) {
                remainder -= modulus;
                if (remainder >= modulus) {
                        remainder -= modulus;
                }
        }
        return static_cast<uint32_t>(remainder);
}

// ---------------------------------------------------------------------------------------

__device__ __forceinline__ int BloomCheck(const uint32_t* __restrict__ hash, const uint8_t* __restrict__ inputBloomLookUp,
        uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t K_LENGTH, uint64_t bloomReciprocal,
        uint32_t bloomMask, uint32_t bloomIsPowerOfTwo)
{
        uint8_t hits = 0;
        const uint32_t seedA = MurMurHash2(hash, K_LENGTH, 0x9747b28c);
        const uint32_t seedB = MurMurHash2(hash, K_LENGTH, seedA);
        for (uint8_t i = 0; i < BLOOM_HASHES; ++i) {
                const uint64_t candidate = static_cast<uint64_t>(seedA) + static_cast<uint64_t>(seedB) * i;
                const uint32_t bitIndex = FastRange(candidate, BLOOM_BITS, bloomReciprocal, bloomMask, bloomIsPowerOfTwo);
                if (Test_Bit_Set_Bit(inputBloomLookUp, bitIndex)) {
                        ++hits;
                }
                else {
                        return 0;
                }
        }
        return hits == BLOOM_HASHES;
}

// ---------------------------------------------------------------------------------------

__device__ __forceinline__ void CheckPointSEARCH_MODE_MA(uint32_t* __restrict__ _h, uint32_t offset, int32_t mode,
        uint8_t* __restrict__ bloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t bloomReciprocal,
        uint32_t bloomMask, uint32_t bloomIsPowerOfTwo, uint32_t maxFound, uint32_t* __restrict__ out)
{
        const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (BloomCheck(_h, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, 20, bloomReciprocal, bloomMask, bloomIsPowerOfTwo) > 0) {
                const uint32_t pos = atomicAdd(out, 1u);
                if (pos < maxFound) {
                        const uint32_t base = pos * ITEM_SIZE_A32;
                        out[base + 1] = tid;
                        const uint32_t packed = (mode ? 0x80000000u : 0u) | (offset & 0x7FFFFFFFu);
                        out[base + 2] = packed;
                        #pragma unroll
                        for (int i = 0; i < 5; ++i) {
                                out[base + 3 + i] = _h[i];
                        }
                }
        }
}

// ---------------------------------------------------------------------------------------

__device__ __forceinline__ void CheckPointSEARCH_MODE_MX(uint32_t* __restrict__ _h, uint32_t offset, int32_t mode,
        uint8_t* __restrict__ bloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t bloomReciprocal,
        uint32_t bloomMask, uint32_t bloomIsPowerOfTwo, uint32_t maxFound, uint32_t* __restrict__ out)
{
        const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (BloomCheck(_h, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, 32, bloomReciprocal, bloomMask, bloomIsPowerOfTwo) > 0) {
                const uint32_t pos = atomicAdd(out, 1u);
                if (pos < maxFound) {
                        const uint32_t base = pos * ITEM_SIZE_X32;
                        out[base + 1] = tid;
                        const uint32_t packed = (mode ? 0x80000000u : 0u) | (offset & 0x7FFFFFFFu);
                        out[base + 2] = packed;
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                                out[base + 3 + i] = _h[i];
                        }
                }
	}
}

// ---------------------------------------------------------------------------------------

__device__ __forceinline__ bool MatchHash(const uint32_t* __restrict__ _h, const uint32_t* __restrict__ hash)
{
	bool match = true;
	#pragma unroll
	for (int i = 0; i < 5; ++i) {
		match &= (_h[i] == hash[i]);
	}
	return match;
}

// ---------------------------------------------------------------------------------------

__device__ __forceinline__ bool MatchXPoint(const uint32_t* __restrict__ _h, const uint32_t* __restrict__ xpoint)
{
	bool match = true;
	#pragma unroll
	for (int i = 0; i < 8; ++i) {
		match &= (_h[i] == xpoint[i]);
	}
	return match;
}

// ---------------------------------------------------------------------------------------

__device__ __forceinline__ void CheckPointSEARCH_MODE_SA(uint32_t* __restrict__ _h, uint32_t offset, int32_t mode,
        uint32_t* __restrict__ hash160, uint32_t maxFound, uint32_t* __restrict__ out)
{
        const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (MatchHash(_h, hash160)) {
                const uint32_t pos = atomicAdd(out, 1u);
                if (pos < maxFound) {
                        const uint32_t base = pos * ITEM_SIZE_A32;
                        out[base + 1] = tid;
                        const uint32_t packed = (mode ? 0x80000000u : 0u) | (offset & 0x7FFFFFFFu);
                        out[base + 2] = packed;
                        #pragma unroll
                        for (int i = 0; i < 5; ++i) {
                                out[base + 3 + i] = _h[i];
                        }
                }
        }
}

// ---------------------------------------------------------------------------------------

__device__ __forceinline__ void CheckPointSEARCH_MODE_SX(uint32_t* __restrict__ _h, uint32_t offset, int32_t mode,
        uint32_t* __restrict__ xpoint, uint32_t maxFound, uint32_t* __restrict__ out)
{
        const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (MatchXPoint(_h, xpoint)) {
                const uint32_t pos = atomicAdd(out, 1u);
                if (pos < maxFound) {
                        const uint32_t base = pos * ITEM_SIZE_X32;
                        out[base + 1] = tid;
                        const uint32_t packed = (mode ? 0x80000000u : 0u) | (offset & 0x7FFFFFFFu);
                        out[base + 2] = packed;
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                                out[base + 3 + i] = _h[i];
                        }
                }
	}
}

// -----------------------------------------------------------------------------------------

#define CHECK_POINT_SEARCH_MODE_MA(_h,offset,mode)  CheckPointSEARCH_MODE_MA(_h,offset,mode,bloomLookUp,BLOOM_BITS,BLOOM_HASHES,bloomReciprocal,bloomMask,bloomIsPowerOfTwo,maxFound,out)

__device__ __noinline__ void CheckHashCompSEARCH_MODE_MA(uint64_t* px, uint8_t isOdd, uint32_t offset,
        uint8_t* bloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t bloomReciprocal,
        uint32_t bloomMask, uint32_t bloomIsPowerOfTwo, uint32_t maxFound, uint32_t* out)
{
        uint32_t h[5];
        _GetHash160Comp(px, isOdd, (uint8_t*)h);
        CHECK_POINT_SEARCH_MODE_MA(h, offset, true);
}

// -----------------------------------------------------------------------------------------

#define CHECK_POINT_SEARCH_MODE_SA(_h,offset,mode)  CheckPointSEARCH_MODE_SA(_h,offset,mode,hash160,maxFound,out)

__device__ __noinline__ void CheckHashCompSEARCH_MODE_SA(uint64_t* px, uint8_t isOdd, uint32_t offset,
        uint32_t* hash160, uint32_t maxFound, uint32_t* out)
{
        uint32_t h[5];
        _GetHash160Comp(px, isOdd, (uint8_t*)h);
        CHECK_POINT_SEARCH_MODE_SA(h, offset, true);
}
// -----------------------------------------------------------------------------------------

__device__ __noinline__ void CheckHashUnCompSEARCH_MODE_MA(uint64_t* px, uint64_t* py, uint32_t offset,
        uint8_t* bloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t bloomReciprocal,
        uint32_t bloomMask, uint32_t bloomIsPowerOfTwo, uint32_t maxFound, uint32_t* out)
{
        uint32_t h[5];
        _GetHash160(px, py, (uint8_t*)h);
        CHECK_POINT_SEARCH_MODE_MA(h, offset, false);
}

// ---------------------------------------------------------------------------------------

__device__ __noinline__ void CheckHashUnCompSEARCH_MODE_SA(uint64_t* px, uint64_t* py, uint32_t offset,
        uint32_t* hash160, uint32_t maxFound, uint32_t* out)
{
        uint32_t h[5];
        _GetHash160(px, py, (uint8_t*)h);
        CHECK_POINT_SEARCH_MODE_SA(h, offset, false);
}

// -----------------------------------------------------------------------------------------

__device__ __noinline__ void CheckHashSEARCH_MODE_MA(uint32_t mode, uint64_t* px, uint64_t* py, uint32_t offset,
        uint8_t* bloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t bloomReciprocal,
        uint32_t bloomMask, uint32_t bloomIsPowerOfTwo, uint32_t maxFound, uint32_t* out)
{
        switch (mode) {
        case SEARCH_COMPRESSED:
                CheckHashCompSEARCH_MODE_MA(px, (uint8_t)(py[0] & 1), offset, bloomLookUp, BLOOM_BITS, BLOOM_HASHES,
                        bloomReciprocal, bloomMask, bloomIsPowerOfTwo, maxFound, out);
                break;
        case SEARCH_UNCOMPRESSED:
                CheckHashUnCompSEARCH_MODE_MA(px, py, offset, bloomLookUp, BLOOM_BITS, BLOOM_HASHES,
                        bloomReciprocal, bloomMask, bloomIsPowerOfTwo, maxFound, out);
                break;
        case SEARCH_BOTH:
                CheckHashCompSEARCH_MODE_MA(px, (uint8_t)(py[0] & 1), offset, bloomLookUp, BLOOM_BITS, BLOOM_HASHES,
                        bloomReciprocal, bloomMask, bloomIsPowerOfTwo, maxFound, out);
                CheckHashUnCompSEARCH_MODE_MA(px, py, offset, bloomLookUp, BLOOM_BITS, BLOOM_HASHES,
                        bloomReciprocal, bloomMask, bloomIsPowerOfTwo, maxFound, out);
                break;
        }
}

// -----------------------------------------------------------------------------------------

#define CHECK_POINT_SEARCH_MODE_MX(_h,offset,mode)  CheckPointSEARCH_MODE_MX(_h,offset,mode,bloomLookUp,BLOOM_BITS,BLOOM_HASHES,bloomReciprocal,bloomMask,bloomIsPowerOfTwo,maxFound,out)

__device__ __noinline__ void CheckPubCompSEARCH_MODE_MX(uint64_t* px, uint8_t isOdd, uint32_t offset,
        uint8_t* bloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t bloomReciprocal,
        uint32_t bloomMask, uint32_t bloomIsPowerOfTwo, uint32_t maxFound, uint32_t* out)
{
        uint32_t h[8];
        uint32_t* x32 = (uint32_t*)(px);

        // Compressed public key
	h[0] = __byte_perm(x32[7], 0, 0x0123);
	h[1] = __byte_perm(x32[6], 0, 0x0123);
	h[2] = __byte_perm(x32[5], 0, 0x0123);
	h[3] = __byte_perm(x32[4], 0, 0x0123);
	h[4] = __byte_perm(x32[3], 0, 0x0123);
	h[5] = __byte_perm(x32[2], 0, 0x0123);
	h[6] = __byte_perm(x32[1], 0, 0x0123);
	h[7] = __byte_perm(x32[0], 0, 0x0123);

        CHECK_POINT_SEARCH_MODE_MX(h, offset, true);
}

#define CHECK_POINT_SEARCH_MODE_SX(_h,offset,mode)  CheckPointSEARCH_MODE_SX(_h,offset,mode,xpoint,maxFound,out)

__device__ __noinline__ void CheckPubCompSEARCH_MODE_SX(uint64_t* px, uint8_t isOdd, uint32_t offset,
        uint32_t* xpoint, uint32_t maxFound, uint32_t* out)
{
        uint32_t h[8];
        uint32_t* x32 = (uint32_t*)(px);

	// Compressed public key
	h[0] = __byte_perm(x32[7], 0, 0x0123);
	h[1] = __byte_perm(x32[6], 0, 0x0123);
	h[2] = __byte_perm(x32[5], 0, 0x0123);
	h[3] = __byte_perm(x32[4], 0, 0x0123);
	h[4] = __byte_perm(x32[3], 0, 0x0123);
	h[5] = __byte_perm(x32[2], 0, 0x0123);
	h[6] = __byte_perm(x32[1], 0, 0x0123);
	h[7] = __byte_perm(x32[0], 0, 0x0123);

        CHECK_POINT_SEARCH_MODE_SX(h, offset, true);
}

// ---------------------------------------------------------------------------------------

__device__ __noinline__ void CheckPubSEARCH_MODE_MX(uint32_t mode, uint64_t* px, uint64_t* py, uint32_t offset,
        uint8_t* bloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t bloomReciprocal,
        uint32_t bloomMask, uint32_t bloomIsPowerOfTwo, uint32_t maxFound, uint32_t* out)
{

        if (mode == SEARCH_COMPRESSED) {
                CheckPubCompSEARCH_MODE_MX(px, (uint8_t)(py[0] & 1), offset, bloomLookUp, BLOOM_BITS, BLOOM_HASHES,
                        bloomReciprocal, bloomMask, bloomIsPowerOfTwo, maxFound, out);
        }
        else {
                return;
        }
}

// -----------------------------------------------------------------------------------------

__device__ __noinline__ void CheckPubSEARCH_MODE_SX(uint32_t mode, uint64_t* px, uint64_t* py, uint32_t offset,
        uint32_t* xpoint, uint32_t maxFound, uint32_t* out)
{

        if (mode == SEARCH_COMPRESSED) {
                CheckPubCompSEARCH_MODE_SX(px, (uint8_t)(py[0] & 1), offset, xpoint, maxFound, out);
        }
        else {
                return;
        }
}

// -----------------------------------------------------------------------------------------

#define CHECK_HASH_SEARCH_MODE_MA(offset) CheckHashSEARCH_MODE_MA(mode, px, py, offset, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, bloomReciprocal, bloomMask, bloomIsPowerOfTwo, maxFound, out)

__device__ void ComputeKeysSEARCH_MODE_MA(uint32_t mode, uint64_t* startx, uint64_t* starty,
        uint8_t* bloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t bloomReciprocal,
        uint32_t bloomMask, uint32_t bloomIsPowerOfTwo, uint32_t maxFound, uint32_t* out, uint32_t baseOffset)
{

        uint64_t dx[GRP_SIZE / 2 + 1][4];
        uint64_t px[4];
        uint64_t py[4];
        uint64_t pyn[4];
        uint64_t sx[4];
        uint64_t sy[4];
        uint64_t dy[4];
        uint64_t _s[4];
        uint64_t _p2[4];
        uint64_t twoGx[4];
        uint64_t twoGy[4];

        LoadGeneratorPoint(twoGx, _2Gnx);
        LoadGeneratorPoint(twoGy, _2Gny);

        // Load starting key
        __syncthreads();
        Load256A(sx, startx);
        Load256A(sy, starty);
	Load256(px, sx);
	Load256(py, sy);

	// Fill group with delta x
	uint32_t i;
        for (i = 0; i < HSIZE; i++) {
                uint64_t gxVal[4];
                LoadGeneratorPoint(gxVal, Gx + 4 * i);
                ModSub256(dx[i], gxVal, sx);
        }
        uint64_t centerGx[4];
        LoadGeneratorPoint(centerGx, Gx + 4 * i);
        ModSub256(dx[i], centerGx, sx);   // For the first point
        ModSub256(dx[i + 1], twoGx, sx); // For the next center point

	// Compute modular inverse
	_ModInvGrouped(dx);

	// We use the fact that P + i*G and P - i*G has the same deltax, so the same inverse
	// We compute key in the positive and negative way from the center of the group

        // Check starting point
        CHECK_HASH_SEARCH_MODE_MA(baseOffset + static_cast<uint32_t>(GRP_SIZE / 2));

        ModNeg256(pyn, py);

        for (i = 0; i < HSIZE; i++) {

                // P = StartPoint + i*G
                Load256(px, sx);
                Load256(py, sy);
                uint64_t gxVal[4];
                uint64_t gyVal[4];
                LoadGeneratorPoint(gxVal, Gx + 4 * i);
                LoadGeneratorPoint(gyVal, Gy + 4 * i);
                ModSub256(dy, gyVal, py);

                _ModMult(_s, dy, dx[i]);                 //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
                _ModSqr(_p2, _s);                        // _p2 = pow2(s)

                ModSub256(px, _p2, px);
                ModSub256(px, gxVal);               // px = pow2(s) - p1.x - p2.x;

                ModSub256(py, gxVal, px);
                _ModMult(py, _s);                        // py = - s*(ret.x-p2.x)
                ModSub256(py, gyVal);               // py = - p2.y - s*(ret.x-p2.x);

                CHECK_HASH_SEARCH_MODE_MA(baseOffset + static_cast<uint32_t>(GRP_SIZE / 2) + (i + 1u));

                // P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
                Load256(px, sx);
                ModSub256(dy, pyn, gyVal);

                _ModMult(_s, dy, dx[i]);                //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
                _ModSqr(_p2, _s);                       // _p = pow2(s)

                ModSub256(px, _p2, px);
                ModSub256(px, gxVal);              // px = pow2(s) - p1.x - p2.x;

                ModSub256(py, px, gxVal);
                _ModMult(py, _s);                       // py = s*(ret.x-p2.x)
                ModSub256(py, gyVal, py);          // py = - p2.y - s*(ret.x-p2.x);

                CHECK_HASH_SEARCH_MODE_MA(baseOffset + (static_cast<uint32_t>(GRP_SIZE / 2) - (i + 1u)));

        }

        // First point (startP - (GRP_SZIE/2)*G)
        Load256(px, sx);
        Load256(py, sy);
        uint64_t boundaryGx[4];
        uint64_t boundaryGy[4];
        LoadGeneratorPoint(boundaryGx, Gx + 4 * i);
        LoadGeneratorPoint(boundaryGy, Gy + 4 * i);
        ModNeg256(dy, boundaryGy);
        ModSub256(dy, py);

        _ModMult(_s, dy, dx[i]);                  //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
        _ModSqr(_p2, _s);                         // _p = pow2(s)

        ModSub256(px, _p2, px);
        ModSub256(px, boundaryGx);                // px = pow2(s) - p1.x - p2.x;

        ModSub256(py, px, boundaryGx);
        _ModMult(py, _s);                         // py = s*(ret.x-p2.x)
        ModSub256(py, boundaryGy, py);            // py = - p2.y - s*(ret.x-p2.x);

        CHECK_HASH_SEARCH_MODE_MA(baseOffset);

        i++;

        // Next start point (startP + GRP_SIZE*G)
        Load256(px, sx);
        Load256(py, sy);
        ModSub256(dy, twoGy, py);

        _ModMult(_s, dy, dx[i]);              //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
        _ModSqr(_p2, _s);                     // _p2 = pow2(s)

        ModSub256(px, _p2, px);
        ModSub256(px, twoGx);                 // px = pow2(s) - p1.x - p2.x;

        ModSub256(py, twoGx, px);
        _ModMult(py, _s);                     // py = - s*(ret.x-p2.x)
        ModSub256(py, twoGy);                 // py = - p2.y - s*(ret.x-p2.x);


	// Update starting point
	__syncthreads();
	Store256A(startx, px);
	Store256A(starty, py);

}

// -----------------------------------------------------------------------------------------

__device__ __noinline__ void CheckHashSEARCH_MODE_SA(uint32_t mode, uint64_t* px, uint64_t* py, uint32_t offset,
        uint32_t* hash160, uint32_t maxFound, uint32_t* out)
{
        switch (mode) {
        case SEARCH_COMPRESSED:
                CheckHashCompSEARCH_MODE_SA(px, (uint8_t)(py[0] & 1), offset, hash160, maxFound, out);
                break;
        case SEARCH_UNCOMPRESSED:
                CheckHashUnCompSEARCH_MODE_SA(px, py, offset, hash160, maxFound, out);
                break;
        case SEARCH_BOTH:
                CheckHashCompSEARCH_MODE_SA(px, (uint8_t)(py[0] & 1), offset, hash160, maxFound, out);
                CheckHashUnCompSEARCH_MODE_SA(px, py, offset, hash160, maxFound, out);
                break;
        }
}

// -----------------------------------------------------------------------------------------

#define CHECK_HASH_SEARCH_MODE_SA(offset) CheckHashSEARCH_MODE_SA(mode, px, py, offset, hash160, maxFound, out)

__device__ void ComputeKeysSEARCH_MODE_SA(uint32_t mode, uint64_t* startx, uint64_t* starty,
        uint32_t* hash160, uint32_t maxFound, uint32_t* out, uint32_t baseOffset)
{

	uint64_t dx[GRP_SIZE / 2 + 1][4];
	uint64_t px[4];
	uint64_t py[4];
        uint64_t pyn[4];
        uint64_t sx[4];
        uint64_t sy[4];
        uint64_t dy[4];
        uint64_t _s[4];
        uint64_t _p2[4];
        uint64_t twoGx[4];
        uint64_t twoGy[4];

        LoadGeneratorPoint(twoGx, _2Gnx);
        LoadGeneratorPoint(twoGy, _2Gny);

        // Load starting key
	__syncthreads();
	Load256A(sx, startx);
	Load256A(sy, starty);
	Load256(px, sx);
	Load256(py, sy);

	// Fill group with delta x
	uint32_t i;
        for (i = 0; i < HSIZE; i++) {
                uint64_t gxVal[4];
                LoadGeneratorPoint(gxVal, Gx + 4 * i);
                ModSub256(dx[i], gxVal, sx);
        }
        uint64_t centerGx[4];
        LoadGeneratorPoint(centerGx, Gx + 4 * i);
        ModSub256(dx[i], centerGx, sx);   // For the first point
        ModSub256(dx[i + 1], twoGx, sx); // For the next center point

	// Compute modular inverse
	_ModInvGrouped(dx);

	// We use the fact that P + i*G and P - i*G has the same deltax, so the same inverse
	// We compute key in the positive and negative way from the center of the group

        // Check starting point
        CHECK_HASH_SEARCH_MODE_SA(baseOffset + static_cast<uint32_t>(GRP_SIZE / 2));

        ModNeg256(pyn, py);

        for (i = 0; i < HSIZE; i++) {

                // P = StartPoint + i*G
                Load256(px, sx);
                Load256(py, sy);
                uint64_t gxVal[4];
                uint64_t gyVal[4];
                LoadGeneratorPoint(gxVal, Gx + 4 * i);
                LoadGeneratorPoint(gyVal, Gy + 4 * i);
                ModSub256(dy, gyVal, py);

                _ModMult(_s, dy, dx[i]);             //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
                _ModSqr(_p2, _s);                    // _p2 = pow2(s)

                ModSub256(px, _p2, px);
                ModSub256(px, gxVal);           // px = pow2(s) - p1.x - p2.x;

                ModSub256(py, gxVal, px);
                _ModMult(py, _s);                    // py = - s*(ret.x-p2.x)
                ModSub256(py, gyVal);           // py = - p2.y - s*(ret.x-p2.x);

                CHECK_HASH_SEARCH_MODE_SA(baseOffset + static_cast<uint32_t>(GRP_SIZE / 2) + (i + 1u));

                // P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
                Load256(px, sx);
                ModSub256(dy, pyn, gyVal);

                _ModMult(_s, dy, dx[i]);            //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
                _ModSqr(_p2, _s);                   // _p = pow2(s)

                ModSub256(px, _p2, px);
                ModSub256(px, gxVal);          // px = pow2(s) - p1.x - p2.x;

                ModSub256(py, px, gxVal);
                _ModMult(py, _s);                   // py = s*(ret.x-p2.x)
                ModSub256(py, gyVal, py);      // py = - p2.y - s*(ret.x-p2.x);

                CHECK_HASH_SEARCH_MODE_SA(baseOffset + (static_cast<uint32_t>(GRP_SIZE / 2) - (i + 1u)));

        }

        // First point (startP - (GRP_SZIE/2)*G)
        Load256(px, sx);
        Load256(py, sy);
        uint64_t boundaryGx[4];
        uint64_t boundaryGy[4];
        LoadGeneratorPoint(boundaryGx, Gx + 4 * i);
        LoadGeneratorPoint(boundaryGy, Gy + 4 * i);
        ModNeg256(dy, boundaryGy);
        ModSub256(dy, py);

        _ModMult(_s, dy, dx[i]);              //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
        _ModSqr(_p2, _s);                     // _p = pow2(s)

        ModSub256(px, _p2, px);
        ModSub256(px, boundaryGx);            // px = pow2(s) - p1.x - p2.x;

        ModSub256(py, px, boundaryGx);
        _ModMult(py, _s);                     // py = s*(ret.x-p2.x)
        ModSub256(py, boundaryGy, py);        // py = - p2.y - s*(ret.x-p2.x);

        CHECK_HASH_SEARCH_MODE_SA(baseOffset);

        i++;

        // Next start point (startP + GRP_SIZE*G)
        Load256(px, sx);
        Load256(py, sy);
        ModSub256(dy, twoGy, py);

        _ModMult(_s, dy, dx[i]);             //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
        _ModSqr(_p2, _s);                    // _p2 = pow2(s)

        ModSub256(px, _p2, px);
        ModSub256(px, twoGx);                // px = pow2(s) - p1.x - p2.x;

        ModSub256(py, twoGx, px);
        _ModMult(py, _s);                    // py = - s*(ret.x-p2.x)
        ModSub256(py, twoGy);                // py = - p2.y - s*(ret.x-p2.x);

	// Update starting point
	__syncthreads();
	Store256A(startx, px);
	Store256A(starty, py);

}



// -----------------------------------------------------------------------------------------

#define CHECK_PUB_SEARCH_MODE_MX(offset) CheckPubSEARCH_MODE_MX(mode, px, py, offset, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, bloomReciprocal, bloomMask, bloomIsPowerOfTwo, maxFound, out)

__device__ void ComputeKeysSEARCH_MODE_MX(uint32_t mode, uint64_t* startx, uint64_t* starty,
        uint8_t* bloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t bloomReciprocal,
        uint32_t bloomMask, uint32_t bloomIsPowerOfTwo, uint32_t maxFound, uint32_t* out, uint32_t baseOffset)
{

	uint64_t dx[GRP_SIZE / 2 + 1][4];
	uint64_t px[4];
	uint64_t py[4];
        uint64_t pyn[4];
        uint64_t sx[4];
        uint64_t sy[4];
        uint64_t dy[4];
        uint64_t _s[4];
        uint64_t _p2[4];
        uint64_t twoGx[4];
        uint64_t twoGy[4];

        LoadGeneratorPoint(twoGx, _2Gnx);
        LoadGeneratorPoint(twoGy, _2Gny);

        // Load starting key
	__syncthreads();
	Load256A(sx, startx);
	Load256A(sy, starty);
	Load256(px, sx);
	Load256(py, sy);

	// Fill group with delta x
	uint32_t i;
        for (i = 0; i < HSIZE; i++) {
                uint64_t gxVal[4];
                LoadGeneratorPoint(gxVal, Gx + 4 * i);
                ModSub256(dx[i], gxVal, sx);
        }
        uint64_t centerGx[4];
        LoadGeneratorPoint(centerGx, Gx + 4 * i);
        ModSub256(dx[i], centerGx, sx);   // For the first point
        ModSub256(dx[i + 1], twoGx, sx); // For the next center point

	// Compute modular inverse
	_ModInvGrouped(dx);

	// We use the fact that P + i*G and P - i*G has the same deltax, so the same inverse
	// We compute key in the positive and negative way from the center of the group

        // Check starting point
        CHECK_PUB_SEARCH_MODE_MX(baseOffset + static_cast<uint32_t>(GRP_SIZE / 2));

        ModNeg256(pyn, py);

        for (i = 0; i < HSIZE; i++) {

                // P = StartPoint + i*G
                Load256(px, sx);
                Load256(py, sy);
                uint64_t gxVal[4];
                uint64_t gyVal[4];
                LoadGeneratorPoint(gxVal, Gx + 4 * i);
                LoadGeneratorPoint(gyVal, Gy + 4 * i);
                ModSub256(dy, gyVal, py);

                _ModMult(_s, dy, dx[i]);                 //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
                _ModSqr(_p2, _s);                        // _p2 = pow2(s)

                ModSub256(px, _p2, px);
                ModSub256(px, gxVal);               // px = pow2(s) - p1.x - p2.x;

                ModSub256(py, gxVal, px);
                _ModMult(py, _s);                        // py = - s*(ret.x-p2.x)
                ModSub256(py, gyVal);               // py = - p2.y - s*(ret.x-p2.x);

                CHECK_PUB_SEARCH_MODE_MX(baseOffset + static_cast<uint32_t>(GRP_SIZE / 2) + (i + 1u));

                // P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
                Load256(px, sx);
                ModSub256(dy, pyn, gyVal);

                _ModMult(_s, dy, dx[i]);                //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
                _ModSqr(_p2, _s);                       // _p = pow2(s)

                ModSub256(px, _p2, px);
                ModSub256(px, gxVal);              // px = pow2(s) - p1.x - p2.x;

                ModSub256(py, px, gxVal);
                _ModMult(py, _s);                       // py = s*(ret.x-p2.x)
                ModSub256(py, gyVal, py);          // py = - p2.y - s*(ret.x-p2.x);

                CHECK_PUB_SEARCH_MODE_MX(baseOffset + (static_cast<uint32_t>(GRP_SIZE / 2) - (i + 1u)));

        }

        // First point (startP - (GRP_SZIE/2)*G)
        Load256(px, sx);
        Load256(py, sy);
        uint64_t boundaryGx[4];
        uint64_t boundaryGy[4];
        LoadGeneratorPoint(boundaryGx, Gx + 4 * i);
        LoadGeneratorPoint(boundaryGy, Gy + 4 * i);
        ModNeg256(dy, boundaryGy);
        ModSub256(dy, py);

        _ModMult(_s, dy, dx[i]);            //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
        _ModSqr(_p2, _s);                   // _p = pow2(s)

        ModSub256(px, _p2, px);
        ModSub256(px, boundaryGx);         // px = pow2(s) - p1.x - p2.x;

        ModSub256(py, px, boundaryGx);
        _ModMult(py, _s);                  // py = s*(ret.x-p2.x)
        ModSub256(py, boundaryGy, py);     // py = - p2.y - s*(ret.x-p2.x);

        CHECK_PUB_SEARCH_MODE_MX(baseOffset);

        i++;

        // Next start point (startP + GRP_SIZE*G)
        Load256(px, sx);
        Load256(py, sy);
        ModSub256(dy, twoGy, py);

        _ModMult(_s, dy, dx[i]);          //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
        _ModSqr(_p2, _s);                 // _p2 = pow2(s)

        ModSub256(px, _p2, px);
        ModSub256(px, twoGx);             // px = pow2(s) - p1.x - p2.x;

        ModSub256(py, twoGx, px);
        _ModMult(py, _s);                 // py = - s*(ret.x-p2.x)
        ModSub256(py, twoGy);             // py = - p2.y - s*(ret.x-p2.x);

	// Update starting point
	__syncthreads();
	Store256A(startx, px);
	Store256A(starty, py);

}

// -----------------------------------------------------------------------------------------

#define CHECK_PUB_SEARCH_MODE_SX(offset) CheckPubSEARCH_MODE_SX(mode, px, py, offset, xpoint, maxFound, out)

__device__ void ComputeKeysSEARCH_MODE_SX(uint32_t mode, uint64_t* startx, uint64_t* starty,
        uint32_t* xpoint, uint32_t maxFound, uint32_t* out, uint32_t baseOffset)
{

	uint64_t dx[GRP_SIZE / 2 + 1][4];
	uint64_t px[4];
	uint64_t py[4];
        uint64_t pyn[4];
        uint64_t sx[4];
        uint64_t sy[4];
        uint64_t dy[4];
        uint64_t _s[4];
        uint64_t _p2[4];
        uint64_t twoGx[4];
        uint64_t twoGy[4];

        LoadGeneratorPoint(twoGx, _2Gnx);
        LoadGeneratorPoint(twoGy, _2Gny);

        // Load starting key
	__syncthreads();
	Load256A(sx, startx);
	Load256A(sy, starty);
	Load256(px, sx);
	Load256(py, sy);

	// Fill group with delta x
	uint32_t i;
        for (i = 0; i < HSIZE; i++) {
                uint64_t gxVal[4];
                LoadGeneratorPoint(gxVal, Gx + 4 * i);
                ModSub256(dx[i], gxVal, sx);
        }
        uint64_t centerGx[4];
        LoadGeneratorPoint(centerGx, Gx + 4 * i);
        ModSub256(dx[i], centerGx, sx);      // For the first point
        ModSub256(dx[i + 1], twoGx, sx);       // For the next center point

	// Compute modular inverse
	_ModInvGrouped(dx);

	// We use the fact that P + i*G and P - i*G has the same deltax, so the same inverse
	// We compute key in the positive and negative way from the center of the group

        // Check starting point
        CHECK_PUB_SEARCH_MODE_SX(baseOffset + static_cast<uint32_t>(GRP_SIZE / 2));

	ModNeg256(pyn, py);

        for (i = 0; i < HSIZE; i++) {

                // P = StartPoint + i*G
                Load256(px, sx);
                Load256(py, sy);
                uint64_t gxVal[4];
                uint64_t gyVal[4];
                LoadGeneratorPoint(gxVal, Gx + 4 * i);
                LoadGeneratorPoint(gyVal, Gy + 4 * i);
                ModSub256(dy, gyVal, py);

                _ModMult(_s, dy, dx[i]);           //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
                _ModSqr(_p2, _s);                  // _p2 = pow2(s)

                ModSub256(px, _p2, px);
                ModSub256(px, gxVal);         // px = pow2(s) - p1.x - p2.x;

                ModSub256(py, gxVal, px);
                _ModMult(py, _s);                  // py = - s*(ret.x-p2.x)
                ModSub256(py, gyVal);         // py = - p2.y - s*(ret.x-p2.x);

                CHECK_PUB_SEARCH_MODE_SX(baseOffset + static_cast<uint32_t>(GRP_SIZE / 2) + (i + 1u));

                // P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
                Load256(px, sx);
                ModSub256(dy, pyn, gyVal);

                _ModMult(_s, dy, dx[i]);            //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
                _ModSqr(_p2, _s);                   // _p = pow2(s)

                ModSub256(px, _p2, px);
                ModSub256(px, gxVal);         // px = pow2(s) - p1.x - p2.x;

                ModSub256(py, px, gxVal);
                _ModMult(py, _s);                  // py = s*(ret.x-p2.x)
                ModSub256(py, gyVal, py);     // py = - p2.y - s*(ret.x-p2.x);

                CHECK_PUB_SEARCH_MODE_SX(baseOffset + (static_cast<uint32_t>(GRP_SIZE / 2) - (i + 1u)));

        }

        // First point (startP - (GRP_SZIE/2)*G)
        Load256(px, sx);
        Load256(py, sy);
        uint64_t boundaryGx[4];
        uint64_t boundaryGy[4];
        LoadGeneratorPoint(boundaryGx, Gx + 4 * i);
        LoadGeneratorPoint(boundaryGy, Gy + 4 * i);
        ModNeg256(dy, boundaryGy);
        ModSub256(dy, py);

        _ModMult(_s, dy, dx[i]);           //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
        _ModSqr(_p2, _s);                  // _p = pow2(s)

        ModSub256(px, _p2, px);
        ModSub256(px, boundaryGx);         // px = pow2(s) - p1.x - p2.x;

        ModSub256(py, px, boundaryGx);
        _ModMult(py, _s);                  // py = s*(ret.x-p2.x)
        ModSub256(py, boundaryGy, py);     // py = - p2.y - s*(ret.x-p2.x);

        CHECK_PUB_SEARCH_MODE_SX(baseOffset);

	i++;

        // Next start point (startP + GRP_SIZE*G)
        Load256(px, sx);
        Load256(py, sy);
        ModSub256(dy, twoGy, py);

        _ModMult(_s, dy, dx[i]);           //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
        _ModSqr(_p2, _s);                  // _p2 = pow2(s)

        ModSub256(px, _p2, px);
        ModSub256(px, twoGx);              // px = pow2(s) - p1.x - p2.x;

        ModSub256(py, twoGx, px);
        _ModMult(py, _s);                  // py = - s*(ret.x-p2.x)
        ModSub256(py, twoGy);              // py = - p2.y - s*(ret.x-p2.x);

	// Update starting point
	__syncthreads();
	Store256A(startx, px);
	Store256A(starty, py);

}

// ------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------


__device__ __noinline__ void CheckPointSEARCH_ETH_MODE_MA(uint32_t* _h, uint32_t offset,
        uint8_t* bloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t bloomReciprocal,
        uint32_t bloomMask, uint32_t bloomIsPowerOfTwo, uint32_t maxFound, uint32_t* out)
{
        uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (BloomCheck(_h, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, 20, bloomReciprocal, bloomMask, bloomIsPowerOfTwo) > 0) {
                uint32_t pos = atomicAdd(out, 1);
                if (pos < maxFound) {
                        out[pos * ITEM_SIZE_A32 + 1] = tid;
                        out[pos * ITEM_SIZE_A32 + 2] = offset & 0x7FFFFFFF;
                        out[pos * ITEM_SIZE_A32 + 3] = _h[0];
			out[pos * ITEM_SIZE_A32 + 4] = _h[1];
			out[pos * ITEM_SIZE_A32 + 5] = _h[2];
			out[pos * ITEM_SIZE_A32 + 6] = _h[3];
			out[pos * ITEM_SIZE_A32 + 7] = _h[4];
		}
	}
}


#define CHECK_POINT_SEARCH_ETH_MODE_MA(_h,offset)  CheckPointSEARCH_ETH_MODE_MA(_h,offset,bloomLookUp,BLOOM_BITS,BLOOM_HASHES,bloomReciprocal,bloomMask,bloomIsPowerOfTwo,maxFound,out)

__device__ __noinline__ void CheckHashCompSEARCH_ETH_MODE_MA(uint64_t* px, uint64_t* py, uint32_t offset,
        uint8_t* bloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t bloomReciprocal,
        uint32_t bloomMask, uint32_t bloomIsPowerOfTwo, uint32_t maxFound, uint32_t* out)
{
        uint32_t h[5];
        _GetHashKeccak160(px, py, h);
        CHECK_POINT_SEARCH_ETH_MODE_MA(h, offset);
}


__device__ __noinline__ void CheckHashSEARCH_ETH_MODE_MA(uint64_t* px, uint64_t* py, uint32_t offset,
        uint8_t* bloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t bloomReciprocal,
        uint32_t bloomMask, uint32_t bloomIsPowerOfTwo, uint32_t maxFound, uint32_t* out)
{
        CheckHashCompSEARCH_ETH_MODE_MA(px, py, offset, bloomLookUp, BLOOM_BITS, BLOOM_HASHES,
                bloomReciprocal, bloomMask, bloomIsPowerOfTwo, maxFound, out);

}

#define CHECK_HASH_SEARCH_ETH_MODE_MA(offset) CheckHashSEARCH_ETH_MODE_MA(px, py, offset, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, bloomReciprocal, bloomMask, bloomIsPowerOfTwo, maxFound, out)

__device__ void ComputeKeysSEARCH_ETH_MODE_MA(uint64_t* startx, uint64_t* starty,
        uint8_t* bloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t bloomReciprocal,
        uint32_t bloomMask, uint32_t bloomIsPowerOfTwo, uint32_t maxFound, uint32_t* out, uint32_t baseOffset)
{

	uint64_t dx[GRP_SIZE / 2 + 1][4];
	uint64_t px[4];
	uint64_t py[4];
        uint64_t pyn[4];
        uint64_t sx[4];
        uint64_t sy[4];
        uint64_t dy[4];
        uint64_t _s[4];
        uint64_t _p2[4];
        uint64_t twoGx[4];
        uint64_t twoGy[4];

        LoadGeneratorPoint(twoGx, _2Gnx);
        LoadGeneratorPoint(twoGy, _2Gny);

        // Load starting key
	__syncthreads();
	Load256A(sx, startx);
	Load256A(sy, starty);
	Load256(px, sx);
	Load256(py, sy);

	// Fill group with delta x
	uint32_t i;
        for (i = 0; i < HSIZE; i++) {
                uint64_t gxVal[4];
                LoadGeneratorPoint(gxVal, Gx + 4 * i);
                ModSub256(dx[i], gxVal, sx);
        }
        uint64_t centerGx[4];
        LoadGeneratorPoint(centerGx, Gx + 4 * i);
        ModSub256(dx[i], centerGx, sx);   // For the first point
        ModSub256(dx[i + 1], twoGx, sx); // For the next center point

	// Compute modular inverse
	_ModInvGrouped(dx);

	// We use the fact that P + i*G and P - i*G has the same deltax, so the same inverse
	// We compute key in the positive and negative way from the center of the group

        // Check starting point
        CHECK_HASH_SEARCH_ETH_MODE_MA(baseOffset + static_cast<uint32_t>(GRP_SIZE / 2));

	ModNeg256(pyn, py);

        for (i = 0; i < HSIZE; i++) {

                // P = StartPoint + i*G
                Load256(px, sx);
                Load256(py, sy);
                uint64_t gxVal[4];
                uint64_t gyVal[4];
                LoadGeneratorPoint(gxVal, Gx + 4 * i);
                LoadGeneratorPoint(gyVal, Gy + 4 * i);
                ModSub256(dy, gyVal, py);

                _ModMult(_s, dy, dx[i]);                 //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
                _ModSqr(_p2, _s);                        // _p2 = pow2(s)

                ModSub256(px, _p2, px);
                ModSub256(px, gxVal);               // px = pow2(s) - p1.x - p2.x;

                ModSub256(py, gxVal, px);
                _ModMult(py, _s);                        // py = - s*(ret.x-p2.x)
                ModSub256(py, gyVal);               // py = - p2.y - s*(ret.x-p2.x);

                CHECK_HASH_SEARCH_ETH_MODE_MA(baseOffset + static_cast<uint32_t>(GRP_SIZE / 2) + (i + 1u));

                // P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
                Load256(px, sx);
                ModSub256(dy, pyn, gyVal);

                _ModMult(_s, dy, dx[i]);                //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
                _ModSqr(_p2, _s);                       // _p = pow2(s)

                ModSub256(px, _p2, px);
                ModSub256(px, gxVal);              // px = pow2(s) - p1.x - p2.x;

                ModSub256(py, px, gxVal);
                _ModMult(py, _s);                       // py = s*(ret.x-p2.x)
                ModSub256(py, gyVal, py);          // py = - p2.y - s*(ret.x-p2.x);

                CHECK_HASH_SEARCH_ETH_MODE_MA(baseOffset + (static_cast<uint32_t>(GRP_SIZE / 2) - (i + 1u)));

        }

        // First point (startP - (GRP_SZIE/2)*G)
        Load256(px, sx);
        Load256(py, sy);
        uint64_t boundaryGx[4];
        uint64_t boundaryGy[4];
        LoadGeneratorPoint(boundaryGx, Gx + 4 * i);
        LoadGeneratorPoint(boundaryGy, Gy + 4 * i);
        ModNeg256(dy, boundaryGy);
        ModSub256(dy, py);

        _ModMult(_s, dy, dx[i]);                  //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
        _ModSqr(_p2, _s);                         // _p = pow2(s)

        ModSub256(px, _p2, px);
        ModSub256(px, boundaryGx);                // px = pow2(s) - p1.x - p2.x;

        ModSub256(py, px, boundaryGx);
        _ModMult(py, _s);                         // py = s*(ret.x-p2.x)
        ModSub256(py, boundaryGy, py);            // py = - p2.y - s*(ret.x-p2.x);

        CHECK_HASH_SEARCH_ETH_MODE_MA(baseOffset);

	i++;

        // Next start point (startP + GRP_SIZE*G)
        Load256(px, sx);
        Load256(py, sy);
        ModSub256(dy, twoGy, py);

        _ModMult(_s, dy, dx[i]);              //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
        _ModSqr(_p2, _s);                     // _p2 = pow2(s)

        ModSub256(px, _p2, px);
        ModSub256(px, twoGx);                 // px = pow2(s) - p1.x - p2.x;

        ModSub256(py, twoGx, px);
        _ModMult(py, _s);                     // py = - s*(ret.x-p2.x)
        ModSub256(py, twoGy);                 // py = - p2.y - s*(ret.x-p2.x);


	// Update starting point
	__syncthreads();
	Store256A(startx, px);
	Store256A(starty, py);

}



__device__ __noinline__ void CheckPointSEARCH_MODE_SA(uint32_t* _h, uint32_t offset,
        uint32_t* hash, uint32_t maxFound, uint32_t* out)
{
        uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (MatchHash(_h, hash)) {
                uint32_t pos = atomicAdd(out, 1);
                if (pos < maxFound) {
                        out[pos * ITEM_SIZE_A32 + 1] = tid;
                        out[pos * ITEM_SIZE_A32 + 2] = offset & 0x7FFFFFFF;
                        out[pos * ITEM_SIZE_A32 + 3] = _h[0];
			out[pos * ITEM_SIZE_A32 + 4] = _h[1];
			out[pos * ITEM_SIZE_A32 + 5] = _h[2];
			out[pos * ITEM_SIZE_A32 + 6] = _h[3];
			out[pos * ITEM_SIZE_A32 + 7] = _h[4];
		}
	}
}

#define CHECK_POINT_SEARCH_ETH_MODE_SA(_h,offset)  CheckPointSEARCH_MODE_SA(_h,offset,hash,maxFound,out)

__device__ __noinline__ void CheckHashCompSEARCH_ETH_MODE_SA(uint64_t* px, uint64_t* py, uint32_t offset,
        uint32_t* hash, uint32_t maxFound, uint32_t* out)
{
        uint32_t h[5];
        _GetHashKeccak160(px, py, h);
        CHECK_POINT_SEARCH_ETH_MODE_SA(h, offset);
}

__device__ __noinline__ void CheckHashSEARCH_ETH_MODE_SA(uint64_t* px, uint64_t* py, uint32_t offset,
        uint32_t* hash, uint32_t maxFound, uint32_t* out)
{
        CheckHashCompSEARCH_ETH_MODE_SA(px, py, offset, hash, maxFound, out);

}
#define CHECK_HASH_SEARCH_ETH_MODE_SA(offset) CheckHashSEARCH_ETH_MODE_SA(px, py, offset, hash, maxFound, out)

__device__ void ComputeKeysSEARCH_ETH_MODE_SA(uint64_t* startx, uint64_t* starty,
        uint32_t* hash, uint32_t maxFound, uint32_t* out, uint32_t baseOffset)
{

	uint64_t dx[GRP_SIZE / 2 + 1][4];
	uint64_t px[4];
	uint64_t py[4];
        uint64_t pyn[4];
        uint64_t sx[4];
        uint64_t sy[4];
        uint64_t dy[4];
        uint64_t _s[4];
        uint64_t _p2[4];
        uint64_t twoGx[4];
        uint64_t twoGy[4];

        LoadGeneratorPoint(twoGx, _2Gnx);
        LoadGeneratorPoint(twoGy, _2Gny);

        // Load starting key
	__syncthreads();
	Load256A(sx, startx);
	Load256A(sy, starty);
	Load256(px, sx);
	Load256(py, sy);

	// Fill group with delta x
	uint32_t i;
        for (i = 0; i < HSIZE; i++) {
                uint64_t gxVal[4];
                LoadGeneratorPoint(gxVal, Gx + 4 * i);
                ModSub256(dx[i], gxVal, sx);
        }
        uint64_t centerGx[4];
        LoadGeneratorPoint(centerGx, Gx + 4 * i);
        ModSub256(dx[i], centerGx, sx);   // For the first point
        ModSub256(dx[i + 1], twoGx, sx); // For the next center point

	// Compute modular inverse
	_ModInvGrouped(dx);

	// We use the fact that P + i*G and P - i*G has the same deltax, so the same inverse
	// We compute key in the positive and negative way from the center of the group

        // Check starting point
        CHECK_HASH_SEARCH_ETH_MODE_SA(baseOffset + static_cast<uint32_t>(GRP_SIZE / 2));

	ModNeg256(pyn, py);

        for (i = 0; i < HSIZE; i++) {

                // P = StartPoint + i*G
                Load256(px, sx);
                Load256(py, sy);
                uint64_t gxVal[4];
                uint64_t gyVal[4];
                LoadGeneratorPoint(gxVal, Gx + 4 * i);
                LoadGeneratorPoint(gyVal, Gy + 4 * i);
                ModSub256(dy, gyVal, py);

                _ModMult(_s, dy, dx[i]);             //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
                _ModSqr(_p2, _s);                    // _p2 = pow2(s)

                ModSub256(px, _p2, px);
                ModSub256(px, gxVal);           // px = pow2(s) - p1.x - p2.x;

                ModSub256(py, gxVal, px);
                _ModMult(py, _s);                    // py = - s*(ret.x-p2.x)
                ModSub256(py, gyVal);           // py = - p2.y - s*(ret.x-p2.x);

                CHECK_HASH_SEARCH_ETH_MODE_SA(baseOffset + static_cast<uint32_t>(GRP_SIZE / 2) + (i + 1u));

                // P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
                Load256(px, sx);
                ModSub256(dy, pyn, gyVal);

                _ModMult(_s, dy, dx[i]);            //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
                _ModSqr(_p2, _s);                   // _p = pow2(s)

                ModSub256(px, _p2, px);
                ModSub256(px, gxVal);          // px = pow2(s) - p1.x - p2.x;

                ModSub256(py, px, gxVal);
                _ModMult(py, _s);                   // py = s*(ret.x-p2.x)
                ModSub256(py, gyVal, py);      // py = - p2.y - s*(ret.x-p2.x);

                CHECK_HASH_SEARCH_ETH_MODE_SA(baseOffset + (static_cast<uint32_t>(GRP_SIZE / 2) - (i + 1u)));

        }

        // First point (startP - (GRP_SZIE/2)*G)
        Load256(px, sx);
        Load256(py, sy);
        uint64_t boundaryGx[4];
        uint64_t boundaryGy[4];
        LoadGeneratorPoint(boundaryGx, Gx + 4 * i);
        LoadGeneratorPoint(boundaryGy, Gy + 4 * i);
        ModNeg256(dy, boundaryGy);
        ModSub256(dy, py);

        _ModMult(_s, dy, dx[i]);              //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
        _ModSqr(_p2, _s);                     // _p = pow2(s)

        ModSub256(px, _p2, px);
        ModSub256(px, boundaryGx);            // px = pow2(s) - p1.x - p2.x;

        ModSub256(py, px, boundaryGx);
        _ModMult(py, _s);                     // py = s*(ret.x-p2.x)
        ModSub256(py, boundaryGy, py);        // py = - p2.y - s*(ret.x-p2.x);

        CHECK_HASH_SEARCH_ETH_MODE_SA(baseOffset);

	i++;

        // Next start point (startP + GRP_SIZE*G)
        Load256(px, sx);
        Load256(py, sy);
        ModSub256(dy, twoGy, py);

        _ModMult(_s, dy, dx[i]);             //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
        _ModSqr(_p2, _s);                    // _p2 = pow2(s)

        ModSub256(px, _p2, px);
        ModSub256(px, twoGx);                // px = pow2(s) - p1.x - p2.x;

        ModSub256(py, twoGx, px);
        _ModMult(py, _s);                    // py = - s*(ret.x-p2.x)
        ModSub256(py, twoGy);                // py = - p2.y - s*(ret.x-p2.x);

	// Update starting point
	__syncthreads();
	Store256A(startx, px);
	Store256A(starty, py);

}

#undef GPU_LDG
