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

#include "GPUEngine.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdint.h>
#include <string>
#include <tuple>
#include <utility>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "../hash/ripemd160.h"
#include "../hash/sha256.h"
#include "../Timer.h"

#include "GPUHash.h"
#include "GPUMath.h"
#include "GPUCompute.h"
#include "GPUBase58.h"
#include "CudaCompat.h"

void KH_InitCudaOptimizations(cudaStream_t stream, size_t table_words);
void KH_LaunchWithDomain(const void* kernel, dim3 grid, dim3 block, void** args, size_t shm, cudaStream_t stream);

__global__ void compute_keys_comp_mode_ma(uint32_t mode, uint8_t* bloomLookUp, uint64_t BLOOM_BITS,
        uint8_t BLOOM_HASHES, uint64_t bloomReciprocal, uint32_t bloomMask, uint32_t bloomIsPowerOfTwo,
        uint64_t* keys, uint32_t maxFound, uint32_t* found, int stepMultiplier);

__global__ void compute_keys_comp_mode_sa(uint32_t mode, const uint32_t* __restrict__ hash160, uint64_t* keys,
        uint32_t maxFound, uint32_t* found, int stepMultiplier);

__global__ void compute_keys_comp_mode_mx(uint32_t mode, uint8_t* bloomLookUp, uint64_t BLOOM_BITS,
        uint8_t BLOOM_HASHES, uint64_t bloomReciprocal, uint32_t bloomMask, uint32_t bloomIsPowerOfTwo,
        uint64_t* keys, uint32_t maxFound, uint32_t* found, int stepMultiplier);

__global__ void compute_keys_comp_mode_sx(uint32_t mode, uint32_t* xpoint, uint64_t* keys, uint32_t maxFound,
        uint32_t* found, int stepMultiplier);

namespace {

inline void CheckCuda(cudaError_t result, const char* expr, const char* file, int line)
{
        if (result != cudaSuccess) {
                std::fprintf(stderr, "CUDA failure %s at %s:%d: %s (%s)\n",
                        expr, file, line, cudaGetErrorName(result), cudaGetErrorString(result));
                std::abort();
        }
}

inline uint64_t ComputeFastModReciprocal(uint64_t modulus)
{
        if (modulus <= 1) {
                return 0;
        }
#if defined(_MSC_VER)
        unsigned __int64 remainder = 0;
        return _udiv128(1ULL, 0ULL, static_cast<unsigned __int64>(modulus), &remainder);
#else
        const unsigned __int128 numerator = static_cast<unsigned __int128>(1) << 64;
        return static_cast<uint64_t>(numerator / modulus);
#endif
}

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
int RecommendOccupancyBlockSize()
{
        static int cachedBlockSize = -1;
        if (cachedBlockSize != -1) {
                return cachedBlockSize;
        }

        int minGridSize = 0;
        int blockSize = 0;
        const cudaError_t status = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                reinterpret_cast<const void*>(&compute_keys_comp_mode_ma), 0, 0);
        if (status == cudaSuccess) {
                cachedBlockSize = blockSize;
                return cachedBlockSize;
        }
        if (status == cudaErrorNotSupported || status == cudaErrorInvalidValue) {
                cudaGetLastError();
                cachedBlockSize = 0;
                return cachedBlockSize;
        }
        CheckCuda(status, "cudaOccupancyMaxPotentialBlockSize", __FILE__, __LINE__);
        cachedBlockSize = 0;
        return cachedBlockSize;
}
#else
int RecommendOccupancyBlockSize()
{
        return 0;
}
#endif

} // namespace

#define CUDA_CHECK(call) ::CheckCuda((call), #call, __FILE__, __LINE__)

namespace {

struct SmToCores {
        int sm;
        int cores;
};

constexpr std::array<SmToCores, 20> kSmToCores = { {
        {0x20, 32},  // Fermi Generation (SM 2.0) GF100 class
        {0x21, 48},  // Fermi Generation (SM 2.1) GF10x class
        {0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        {0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        {0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        {0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        {0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        {0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
        {0x53, 128}, // Maxwell Generation (SM 5.3)
        {0x60, 64},  // Pascal Generation (SM 6.0) GP100 class
        {0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
        {0x62, 128}, // Pascal Generation (SM 6.2) GP10B class
        {0x70, 64},  // Volta Generation (SM 7.0) GV100 class
        {0x72, 64},  // Volta Generation (SM 7.2)
        {0x75, 64},  // Turing Generation (SM 7.5) TU10x class
        {0x80, 64},  // Ampere Generation (SM 8.0) GA100 class
        {0x86, 128}, // Ampere Generation (SM 8.6) GA10x class
        {0x87, 128}, // Ampere Generation (SM 8.7) GA10B class
        {0x89, 128}, // Ada Generation   (SM 8.9)
        {0x90, 128}, // Hopper Generation (SM 9.0)
} };

struct DeviceCapabilityInfo
{
        int warpSize = 0;
        int maxThreadsPerMultiprocessor = 0;
        int maxBlocksPerMultiprocessor = 0;
        int memoryPoolsSupported = 0;
        int cooperativeMultiDeviceLaunch = 0;
        int clusterLaunch = 0;
        int memSyncDomainCount = 1;
};

int GetAttributeOrDefault(cudaDeviceAttr attr, int deviceId, int defaultValue = 0)
{
        int value = defaultValue;
        const cudaError_t status = cudaDeviceGetAttribute(&value, attr, deviceId);
        if (status == cudaSuccess) {
                return value;
        }
        if (status == cudaErrorInvalidValue
#if defined(cudaErrorNotSupported)
            || status == cudaErrorNotSupported
#endif
        ) {
                // Clear any sticky error state before returning the default.
                cudaGetLastError();
                return defaultValue;
        }
        CUDA_CHECK(status);
        return value;
}

DeviceCapabilityInfo QueryDeviceCapabilityInfo(int deviceId)
{
        DeviceCapabilityInfo info;
        CUDA_CHECK(cudaDeviceGetAttribute(&info.warpSize, cudaDevAttrWarpSize, deviceId));
        CUDA_CHECK(cudaDeviceGetAttribute(&info.maxThreadsPerMultiprocessor, cudaDevAttrMaxThreadsPerMultiProcessor, deviceId));
        info.maxBlocksPerMultiprocessor = GetAttributeOrDefault(cudaDevAttrMaxBlocksPerMultiprocessor, deviceId);
        info.memoryPoolsSupported = GetAttributeOrDefault(cudaDevAttrMemoryPoolsSupported, deviceId);
#ifdef cudaDevAttrCooperativeMultiDeviceLaunch
        info.cooperativeMultiDeviceLaunch = GetAttributeOrDefault(cudaDevAttrCooperativeMultiDeviceLaunch, deviceId);
#endif
#ifdef cudaDevAttrClusterLaunch
        info.clusterLaunch = GetAttributeOrDefault(cudaDevAttrClusterLaunch, deviceId);
#endif
#ifdef cudaDevAttrMemSyncDomainCount
        info.memSyncDomainCount = GetAttributeOrDefault(cudaDevAttrMemSyncDomainCount, deviceId, 1);
#endif
        return info;
}

struct AsyncAllocatorConfig
{
        bool enabled = false;
        cudaMemPool_t pool = nullptr;
};

AsyncAllocatorConfig ConfigureAsyncAllocatorForDevice(int deviceId)
{
        AsyncAllocatorConfig config;
#if CUDART_VERSION >= 11020
        cudaMemPool_t pool = nullptr;
        const cudaError_t status = cudaDeviceGetDefaultMemPool(&pool, deviceId);
        if (status == cudaSuccess) {
                config.enabled = true;
                config.pool = pool;
#ifdef cudaMemPoolAttrReleaseThreshold
                unsigned long long threshold = std::numeric_limits<unsigned long long>::max();
                const cudaError_t attrStatus = cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
                if (attrStatus == cudaErrorNotSupported || attrStatus == cudaErrorInvalidValue) {
                        cudaGetLastError();
                }
                else {
                        CUDA_CHECK(attrStatus);
                }
#endif
        }
        else if (status == cudaErrorNotSupported || status == cudaErrorInvalidValue) {
                cudaGetLastError();
        }
        else {
                CUDA_CHECK(status);
        }
#else
        (void)deviceId;
#endif
        return config;
}

} // namespace

// ---------------------------------------------------------------------------------------

// mode multiple addresses
KH_LAUNCH_BOUNDS
__global__ void compute_keys_mode_ma(uint32_t mode, uint8_t* bloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES,
        uint64_t bloomReciprocal, uint32_t bloomMask, uint32_t bloomIsPowerOfTwo,
        uint64_t* keys, uint32_t maxFound, uint32_t* found, int stepMultiplier)
{

        int xPtr = (blockIdx.x * blockDim.x) * 8;
        int yPtr = xPtr + 4 * blockDim.x;
        for (int iteration = 0; iteration < stepMultiplier; ++iteration) {
                const uint32_t baseOffset = static_cast<uint32_t>(iteration) * static_cast<uint32_t>(GRP_SIZE);
                ComputeKeysSEARCH_MODE_MA(mode, keys + xPtr, keys + yPtr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES,
                        bloomReciprocal, bloomMask, bloomIsPowerOfTwo, maxFound, found, baseOffset);
        }

}

KH_LAUNCH_BOUNDS
__global__ void compute_keys_comp_mode_ma(uint32_t mode, uint8_t* bloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES,
        uint64_t bloomReciprocal, uint32_t bloomMask, uint32_t bloomIsPowerOfTwo,
        uint64_t* keys, uint32_t maxFound, uint32_t* found, int stepMultiplier)
{

        int xPtr = (blockIdx.x * blockDim.x) * 8;
        int yPtr = xPtr + 4 * blockDim.x;
        for (int iteration = 0; iteration < stepMultiplier; ++iteration) {
                const uint32_t baseOffset = static_cast<uint32_t>(iteration) * static_cast<uint32_t>(GRP_SIZE);
                ComputeKeysSEARCH_MODE_MA(mode, keys + xPtr, keys + yPtr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES,
                        bloomReciprocal, bloomMask, bloomIsPowerOfTwo, maxFound, found, baseOffset);
        }

}

// mode single address
KH_LAUNCH_BOUNDS
__global__ void compute_keys_mode_sa(uint32_t mode, const uint32_t* __restrict__ hash160, uint64_t* keys, uint32_t maxFound, uint32_t* found,
        int stepMultiplier)
{

        int xPtr = (blockIdx.x * blockDim.x) * 8;
        int yPtr = xPtr + 4 * blockDim.x;
        __shared__ uint32_t sharedHash160[5];
        if (threadIdx.x < 5) {
                sharedHash160[threadIdx.x] = hash160[threadIdx.x];
        }
        __syncthreads();
        for (int iteration = 0; iteration < stepMultiplier; ++iteration) {
                const uint32_t baseOffset = static_cast<uint32_t>(iteration) * static_cast<uint32_t>(GRP_SIZE);
                ComputeKeysSEARCH_MODE_SA(mode, keys + xPtr, keys + yPtr, sharedHash160, maxFound, found, baseOffset);
        }

}

KH_LAUNCH_BOUNDS
__global__ void compute_keys_comp_mode_sa(uint32_t mode, const uint32_t* __restrict__ hash160, uint64_t* keys, uint32_t maxFound, uint32_t* found,
        int stepMultiplier)
{

        int xPtr = (blockIdx.x * blockDim.x) * 8;
        int yPtr = xPtr + 4 * blockDim.x;
        __shared__ uint32_t sharedHash160[5];
        if (threadIdx.x < 5) {
                sharedHash160[threadIdx.x] = hash160[threadIdx.x];
        }
        __syncthreads();
        for (int iteration = 0; iteration < stepMultiplier; ++iteration) {
                const uint32_t baseOffset = static_cast<uint32_t>(iteration) * static_cast<uint32_t>(GRP_SIZE);
                ComputeKeysSEARCH_MODE_SA(mode, keys + xPtr, keys + yPtr, sharedHash160, maxFound, found, baseOffset);
        }

}

// mode multiple x points
KH_LAUNCH_BOUNDS
__global__ void compute_keys_comp_mode_mx(uint32_t mode, uint8_t* bloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES,
        uint64_t bloomReciprocal, uint32_t bloomMask, uint32_t bloomIsPowerOfTwo, uint64_t* keys,
        uint32_t maxFound, uint32_t* found, int stepMultiplier)
{

        int xPtr = (blockIdx.x * blockDim.x) * 8;
        int yPtr = xPtr + 4 * blockDim.x;
        for (int iteration = 0; iteration < stepMultiplier; ++iteration) {
                const uint32_t baseOffset = static_cast<uint32_t>(iteration) * static_cast<uint32_t>(GRP_SIZE);
                ComputeKeysSEARCH_MODE_MX(mode, keys + xPtr, keys + yPtr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES,
                        bloomReciprocal, bloomMask, bloomIsPowerOfTwo, maxFound, found, baseOffset);
        }

}

// mode single x point
KH_LAUNCH_BOUNDS
__global__ void compute_keys_comp_mode_sx(uint32_t mode, uint32_t* xpoint, uint64_t* keys, uint32_t maxFound, uint32_t* found,
        int stepMultiplier)
{

        int xPtr = (blockIdx.x * blockDim.x) * 8;
        int yPtr = xPtr + 4 * blockDim.x;
        for (int iteration = 0; iteration < stepMultiplier; ++iteration) {
                const uint32_t baseOffset = static_cast<uint32_t>(iteration) * static_cast<uint32_t>(GRP_SIZE);
                ComputeKeysSEARCH_MODE_SX(mode, keys + xPtr, keys + yPtr, xpoint, maxFound, found, baseOffset);
        }

}

// ---------------------------------------------------------------------------------------
// ethereum

KH_LAUNCH_BOUNDS
__global__ void compute_keys_mode_eth_ma(uint8_t* bloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES,
        uint64_t bloomReciprocal, uint32_t bloomMask, uint32_t bloomIsPowerOfTwo, uint64_t* keys,
        uint32_t maxFound, uint32_t* found, int stepMultiplier)
{

        int xPtr = (blockIdx.x * blockDim.x) * 8;
        int yPtr = xPtr + 4 * blockDim.x;
        for (int iteration = 0; iteration < stepMultiplier; ++iteration) {
                const uint32_t baseOffset = static_cast<uint32_t>(iteration) * static_cast<uint32_t>(GRP_SIZE);
                ComputeKeysSEARCH_ETH_MODE_MA(keys + xPtr, keys + yPtr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES,
                        bloomReciprocal, bloomMask, bloomIsPowerOfTwo, maxFound, found, baseOffset);
        }

}

KH_LAUNCH_BOUNDS
__global__ void compute_keys_mode_eth_sa(const uint32_t* __restrict__ hash, uint64_t* keys, uint32_t maxFound, uint32_t* found,
        int stepMultiplier)
{

        int xPtr = (blockIdx.x * blockDim.x) * 8;
        int yPtr = xPtr + 4 * blockDim.x;
        __shared__ uint32_t sharedHash[5];
        if (threadIdx.x < 5) {
                sharedHash[threadIdx.x] = hash[threadIdx.x];
        }
        __syncthreads();
        for (int iteration = 0; iteration < stepMultiplier; ++iteration) {
                const uint32_t baseOffset = static_cast<uint32_t>(iteration) * static_cast<uint32_t>(GRP_SIZE);
                ComputeKeysSEARCH_ETH_MODE_SA(keys + xPtr, keys + yPtr, sharedHash, maxFound, found, baseOffset);
        }

}

// ---------------------------------------------------------------------------------------

template <typename KernelFunc, typename... Args>
bool GPUEngine::LaunchKeyKernel(KernelFunc kernel, dim3 gridDim, dim3 blockDim, Args&&... args)
{
        auto argsTuple = std::make_tuple(std::forward<Args>(args)...);

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12000)
        if (clusterLaunchActive_) {
                const dim3 clusterDim = QueryClusterDimension(kernel, gridDim, blockDim);
                if (clusterDim.x > 0U) {
                        cudaLaunchConfig_t config{};
                        config.gridDim = gridDim;
                        config.blockDim = blockDim;
                        config.dynamicSmemBytes = 0;
                        config.stream = stream_;

                        cudaLaunchAttribute attribute{};
                        attribute.id = cudaLaunchAttributeClusterDimension;
                        attribute.val.clusterDim.x = clusterDim.x;
                        attribute.val.clusterDim.y = clusterDim.y;
                        attribute.val.clusterDim.z = clusterDim.z;
                        config.attrs = &attribute;
                        config.numAttrs = 1;

                        const cudaError_t status = std::apply([&](auto&... tupleArgs) {
                                return cudaLaunchKernelEx(&config, kernel, tupleArgs...);
                        }, argsTuple);
                        if (status == cudaSuccess) {
                                return true;
                        }
                        if (status == cudaErrorNotSupported || status == cudaErrorInvalidValue) {
                                cudaGetLastError();
                                clusterLaunchActive_ = false;
                        }
                        else {
                                CheckCuda(status, "cudaLaunchKernelEx", __FILE__, __LINE__);
                                return true;
                        }
                }
        }
#endif

        std::array<void*, sizeof...(Args)> argPointers{};
        std::size_t index = 0;
        std::apply([&](auto&... tupleArgs) {
                ((argPointers[index++] = static_cast<void*>(&tupleArgs)), ...);
        }, argsTuple);

        const void* kernelPtr = reinterpret_cast<const void*>(kernel);
        KH_LaunchWithDomain(kernelPtr, gridDim, blockDim, argPointers.data(), 0, stream_);
        return true;
}

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12000)
template <typename KernelFunc>
dim3 GPUEngine::QueryClusterDimension(KernelFunc kernel, dim3 gridDim, dim3 blockDim)
{
        if (!clusterLaunchActive_) {
                return dim3(0U, 0U, 0U);
        }

        struct ClusterCache {
                bool computed = false;
                dim3 value{0U, 0U, 0U};
        };

        static ClusterCache cache;
        if (cache.computed) {
                if (cache.value.x > 0U && (gridDim.x % cache.value.x) != 0U) {
                        return dim3(0U, 0U, 0U);
                }
                return cache.value;
        }

        cudaLaunchConfig_t config{};
        config.gridDim = gridDim;
        config.blockDim = blockDim;
        config.dynamicSmemBytes = 0;

        int clusterSize = 0;
        const cudaError_t status = cudaOccupancyMaxPotentialClusterSize(&clusterSize,
                reinterpret_cast<const void*>(kernel), &config);
        if (status == cudaErrorNotSupported || status == cudaErrorInvalidValue) {
                cudaGetLastError();
                clusterLaunchActive_ = false;
                cache.computed = true;
                cache.value = dim3(0U, 0U, 0U);
                return cache.value;
        }
        CheckCuda(status, "cudaOccupancyMaxPotentialClusterSize", __FILE__, __LINE__);

        unsigned int recommended = static_cast<unsigned int>(clusterSize);
        if (recommended > 8U) {
                recommended = 8U;
        }
        if (recommended > gridDim.x) {
                recommended = gridDim.x;
        }
        while (recommended > 1U && (gridDim.x % recommended) != 0U) {
                --recommended;
        }

        if (recommended > 1U) {
                cache.value = dim3(recommended, 1U, 1U);
        }
        else {
                cache.value = dim3(0U, 0U, 0U);
        }
        cache.computed = true;
        return cache.value;
}
#endif

// ---------------------------------------------------------------------------------------

using namespace std;

int _ConvertSMVer2Cores(int major, int minor)
{

        const int sm = (major << 4) + minor;
        for (const auto& entry : kSmToCores) {
                if (entry.sm == sm) {
                        return entry.cores;
                }
        }

        const auto& fallback = kSmToCores.back();
        std::fprintf(stderr,
                "Warning: SM %d.%d is not explicitly supported, defaulting to %d cores/SM\n",
                major, minor, fallback.cores);
        return fallback.cores;
}

// ----------------------------------------------------------------------------

GPUEngine::GPUEngine(Secp256K1* secp, int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound,
        int searchMode, int compMode, int coinType, int64_t BLOOM_SIZE, uint64_t BLOOM_BITS,
        uint8_t BLOOM_HASHES, const uint8_t* BLOOM_DATA, uint8_t* DATA, uint64_t TOTAL_COUNT, bool rKey,
        int stepMultiplier)
{

        // Initialise CUDA
        this->nbThreadPerGroup = nbThreadPerGroup;
        this->searchMode = searchMode;
        this->compMode = compMode;
        this->coinType = coinType;
        this->rKey = rKey;
        this->stepMultiplier = std::max(1, stepMultiplier);

        this->BLOOM_SIZE = BLOOM_SIZE;
        this->BLOOM_BITS = BLOOM_BITS;
        this->BLOOM_HASHES = BLOOM_HASHES;
        this->DATA = DATA;
        this->TOTAL_COUNT = TOTAL_COUNT;

        bloomFastModReciprocal_ = 0;
        bloomMask_ = 0;
        bloomIsPowerOfTwo_ = 0;
        if (BLOOM_BITS > 0) {
                if ((BLOOM_BITS & (BLOOM_BITS - 1)) == 0) {
                        bloomIsPowerOfTwo_ = 1;
                        bloomMask_ = static_cast<uint32_t>(BLOOM_BITS - 1);
                }
                else {
                        bloomFastModReciprocal_ = ComputeFastModReciprocal(BLOOM_BITS);
                }
        }

        initialised = false;

        try {
                cuda_compat::EnsureRuntimeSupportsCuda13();
        }
        catch (const std::exception& ex) {
                std::fprintf(stderr, "GPUEngine: %s\n", ex.what());
                return;
        }

        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

        // This function call returns 0 if there are no CUDA capable devices.
        if (deviceCount == 0) {
                printf("GPUEngine: There are no available device(s) that support CUDA\n");
                return;
        }

        CUDA_CHECK(cudaSetDevice(gpuId));

        ConfigureAsyncAllocator(gpuId);

        cudaDeviceProp deviceProp{};
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, gpuId));
        const DeviceCapabilityInfo capability = QueryDeviceCapabilityInfo(gpuId);
        const int warpSize = capability.warpSize > 0 ? capability.warpSize : deviceProp.warpSize;
        warpSize_ = warpSize;

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12000)
        clusterLaunchSupported_ = capability.clusterLaunch != 0;
        clusterLaunchActive_ = clusterLaunchSupported_;
        if (clusterLaunchSupported_) {
                std::fprintf(stderr,
                        "GPUEngine: thread block clusters supported, attempting cudaLaunchKernelEx launches.\n");
        }
#else
        clusterLaunchSupported_ = false;
        clusterLaunchActive_ = false;
#endif

        if (nbThreadPerGroup <= 0) {
                const int recommendedBlockSize = RecommendOccupancyBlockSize();
                if (recommendedBlockSize > 0) {
                        nbThreadPerGroup = recommendedBlockSize;
                        std::fprintf(stderr,
                                "GPUEngine: auto-selecting %d threads per group based on CUDA 13 occupancy guidance.\n",
                                nbThreadPerGroup);
                }
        }

        if (nbThreadPerGroup <= 0) {
                nbThreadPerGroup = (warpSize > 0) ? warpSize : deviceProp.maxThreadsPerBlock;
                std::fprintf(stderr,
                        "GPUEngine: defaulting to %d threads per group due to invalid configuration.\n",
                        nbThreadPerGroup);
        }

        if (nbThreadPerGroup > deviceProp.maxThreadsPerBlock) {
                std::fprintf(stderr,
                        "GPUEngine: requested %d threads per group but device limit is %d. Clamping to %d for CUDA 13 compliance.\n",
                        nbThreadPerGroup, deviceProp.maxThreadsPerBlock, deviceProp.maxThreadsPerBlock);
                nbThreadPerGroup = deviceProp.maxThreadsPerBlock;
        }

        if (warpSize > 0 && nbThreadPerGroup % warpSize != 0) {
                const int adjusted = ((nbThreadPerGroup + warpSize - 1) / warpSize) * warpSize;
                std::fprintf(stderr,
                        "GPUEngine: adjusting threads per group from %d to %d to align with warp size %d.\n",
                        nbThreadPerGroup, adjusted, warpSize);
                nbThreadPerGroup = adjusted;
        }

        if (nbThreadGroup == -1) {
                nbThreadGroup = deviceProp.multiProcessorCount * 8;
        }
        const int maxRecommendedGroups = capability.maxBlocksPerMultiprocessor * deviceProp.multiProcessorCount;
        if (maxRecommendedGroups > 0 && nbThreadGroup > maxRecommendedGroups) {
                std::fprintf(stderr,
                        "GPUEngine: limiting thread groups from %d to %d based on SM capabilities for CUDA 13 compliance.\n",
                        nbThreadGroup, maxRecommendedGroups);
                nbThreadGroup = maxRecommendedGroups;
        }

        this->nbThreadPerGroup = nbThreadPerGroup;
        this->nbThread = nbThreadGroup * nbThreadPerGroup;
        this->activeThreadCount = this->nbThread;
        this->maxFound = maxFound;
        this->outputSize = (maxFound * ITEM_SIZE_A + 4);
        if (this->searchMode == (int)SEARCH_MODE_MX) {
                this->outputSize = (maxFound * ITEM_SIZE_X + 4);
        }

        char tmp[512];
        std::snprintf(tmp, sizeof(tmp), "GPU #%d %s (%dx%d cores, warp %d) Grid(%dx%d)",
                gpuId, deviceProp.name, deviceProp.multiProcessorCount,
                _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
                warpSize,
                this->nbThread / this->nbThreadPerGroup,
                this->nbThreadPerGroup);
        deviceName = std::string(tmp);

        // Prefer L1 (We do not use __shared__ at all)
        CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

        const size_t stackSize = 49152;
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, stackSize));

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        streamCreated_ = true;
        CUDA_CHECK(cudaEventCreateWithFlags(&syncEvent_, cudaEventDisableTiming));
        eventCreated_ = true;

        const size_t keyBufferSize = static_cast<size_t>(nbThread) * 32u * 2u;
        AllocateDeviceBuffer(reinterpret_cast<void**>(&inputKey), keyBufferSize);
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&inputKeyPinned), keyBufferSize,
                cudaHostAllocWriteCombined | cudaHostAllocMapped));

        AllocateDeviceBuffer(reinterpret_cast<void**>(&outputBuffer), outputSize);
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&outputBufferPinned), outputSize,
                cudaHostAllocWriteCombined | cudaHostAllocMapped));

        if (BLOOM_SIZE > 0) {
                const size_t bloomBytes = static_cast<size_t>(BLOOM_SIZE);
                AllocateDeviceBuffer(reinterpret_cast<void**>(&inputBloomLookUp), bloomBytes);
                CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&inputBloomLookUpPinned), bloomBytes,
                        cudaHostAllocWriteCombined | cudaHostAllocMapped));
                std::memcpy(inputBloomLookUpPinned, BLOOM_DATA, bloomBytes);
                CUDA_CHECK(cudaMemcpyAsync(inputBloomLookUp, inputBloomLookUpPinned, bloomBytes,
                        cudaMemcpyHostToDevice, stream_));
                CUDA_CHECK(cudaStreamSynchronize(stream_));
                CUDA_CHECK(cudaFreeHost(inputBloomLookUpPinned));
                inputBloomLookUpPinned = nullptr;
        }

        // generator table
        InitGenratorTable(secp);

        CUDA_CHECK(cudaPeekAtLastError());

        compMode = SEARCH_COMPRESSED;
        initialised = true;

}

// ----------------------------------------------------------------------------

GPUEngine::GPUEngine(Secp256K1* secp, int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound,
        int searchMode, int compMode, int coinType, const uint32_t* hashORxpoint, bool rKey,
        int stepMultiplier)
{

        // Initialise CUDA
        this->nbThreadPerGroup = nbThreadPerGroup;
        this->searchMode = searchMode;
        this->compMode = compMode;
        this->coinType = coinType;
        this->rKey = rKey;
        this->stepMultiplier = std::max(1, stepMultiplier);

        bloomFastModReciprocal_ = 0;
        bloomMask_ = 0;
        bloomIsPowerOfTwo_ = 0;

        initialised = false;

        try {
                cuda_compat::EnsureRuntimeSupportsCuda13();
        }
        catch (const std::exception& ex) {
                std::fprintf(stderr, "GPUEngine: %s\n", ex.what());
                return;
        }

        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

        // This function call returns 0 if there are no CUDA capable devices.
        if (deviceCount == 0) {
                printf("GPUEngine: There are no available device(s) that support CUDA\n");
                return;
        }

        CUDA_CHECK(cudaSetDevice(gpuId));

        ConfigureAsyncAllocator(gpuId);

        cudaDeviceProp deviceProp{};
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, gpuId));
        const DeviceCapabilityInfo capability = QueryDeviceCapabilityInfo(gpuId);
        const int warpSize = capability.warpSize > 0 ? capability.warpSize : deviceProp.warpSize;
        warpSize_ = warpSize;

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12000)
        clusterLaunchSupported_ = capability.clusterLaunch != 0;
        clusterLaunchActive_ = clusterLaunchSupported_;
        if (clusterLaunchSupported_) {
                std::fprintf(stderr,
                        "GPUEngine: thread block clusters supported, attempting cudaLaunchKernelEx launches.\n");
        }
#else
        clusterLaunchSupported_ = false;
        clusterLaunchActive_ = false;
#endif

        if (nbThreadPerGroup <= 0) {
                const int recommendedBlockSize = RecommendOccupancyBlockSize();
                if (recommendedBlockSize > 0) {
                        nbThreadPerGroup = recommendedBlockSize;
                        std::fprintf(stderr,
                                "GPUEngine: auto-selecting %d threads per group based on CUDA 13 occupancy guidance.\n",
                                nbThreadPerGroup);
                }
        }

        if (nbThreadPerGroup <= 0) {
                nbThreadPerGroup = (warpSize > 0) ? warpSize : deviceProp.maxThreadsPerBlock;
                std::fprintf(stderr,
                        "GPUEngine: defaulting to %d threads per group due to invalid configuration.\n",
                        nbThreadPerGroup);
        }

        if (nbThreadPerGroup > deviceProp.maxThreadsPerBlock) {
                std::fprintf(stderr,
                        "GPUEngine: requested %d threads per group but device limit is %d. Clamping to %d for CUDA 13 compliance.\n",
                        nbThreadPerGroup, deviceProp.maxThreadsPerBlock, deviceProp.maxThreadsPerBlock);
                nbThreadPerGroup = deviceProp.maxThreadsPerBlock;
        }

        if (warpSize > 0 && nbThreadPerGroup % warpSize != 0) {
                const int adjusted = ((nbThreadPerGroup + warpSize - 1) / warpSize) * warpSize;
                std::fprintf(stderr,
                        "GPUEngine: adjusting threads per group from %d to %d to align with warp size %d.\n",
                        nbThreadPerGroup, adjusted, warpSize);
                nbThreadPerGroup = adjusted;
        }

        if (nbThreadGroup == -1) {
                nbThreadGroup = deviceProp.multiProcessorCount * 8;
        }
        const int maxRecommendedGroups = capability.maxBlocksPerMultiprocessor * deviceProp.multiProcessorCount;
        if (maxRecommendedGroups > 0 && nbThreadGroup > maxRecommendedGroups) {
                std::fprintf(stderr,
                        "GPUEngine: limiting thread groups from %d to %d based on SM capabilities for CUDA 13 compliance.\n",
                        nbThreadGroup, maxRecommendedGroups);
                nbThreadGroup = maxRecommendedGroups;
        }

        this->nbThreadPerGroup = nbThreadPerGroup;
        this->nbThread = nbThreadGroup * nbThreadPerGroup;
        this->activeThreadCount = this->nbThread;
        this->maxFound = maxFound;
        this->outputSize = (maxFound * ITEM_SIZE_A + 4);
        if (this->searchMode == (int)SEARCH_MODE_SX) {
                this->outputSize = (maxFound * ITEM_SIZE_X + 4);
        }

        char tmp[512];
        std::snprintf(tmp, sizeof(tmp), "GPU #%d %s (%dx%d cores, warp %d) Grid(%dx%d)",
                gpuId, deviceProp.name, deviceProp.multiProcessorCount,
                _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
                warpSize,
                this->nbThread / this->nbThreadPerGroup,
                this->nbThreadPerGroup);
        deviceName = std::string(tmp);

        // Prefer L1 (We do not use __shared__ at all)
        CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

        const size_t stackSize = 49152;
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, stackSize));

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        streamCreated_ = true;
        CUDA_CHECK(cudaEventCreateWithFlags(&syncEvent_, cudaEventDisableTiming));
        eventCreated_ = true;

        // Allocate memory
        const size_t keyBufferSize = static_cast<size_t>(nbThread) * 32u * 2u;
        AllocateDeviceBuffer(reinterpret_cast<void**>(&inputKey), keyBufferSize);
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&inputKeyPinned), keyBufferSize,
                cudaHostAllocWriteCombined | cudaHostAllocMapped));

        AllocateDeviceBuffer(reinterpret_cast<void**>(&outputBuffer), outputSize);
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&outputBufferPinned), outputSize,
                cudaHostAllocWriteCombined | cudaHostAllocMapped));

        int K_SIZE = 5;
        if (this->searchMode == (int)SEARCH_MODE_SX) {
                K_SIZE = 8;
        }

        AllocateDeviceBuffer(reinterpret_cast<void**>(&inputHashORxpoint), K_SIZE * sizeof(uint32_t));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&inputHashORxpointPinned), K_SIZE * sizeof(uint32_t),
                cudaHostAllocWriteCombined | cudaHostAllocMapped));

        std::memcpy(inputHashORxpointPinned, hashORxpoint, K_SIZE * sizeof(uint32_t));

        CUDA_CHECK(cudaMemcpyAsync(inputHashORxpoint, inputHashORxpointPinned, K_SIZE * sizeof(uint32_t),
                cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        CUDA_CHECK(cudaFreeHost(inputHashORxpointPinned));
        inputHashORxpointPinned = nullptr;

        // generator table
        InitGenratorTable(secp);


        CUDA_CHECK(cudaPeekAtLastError());

        compMode = SEARCH_COMPRESSED;
        initialised = true;

}

// ----------------------------------------------------------------------------

void GPUEngine::ConfigureAsyncAllocator(int deviceId)
{
        const AsyncAllocatorConfig config = ConfigureAsyncAllocatorForDevice(deviceId);
        useAsyncAlloc_ = config.enabled;
        memPool_ = config.pool;
        asyncFallbackNotified_ = false;
        if (useAsyncAlloc_) {
                std::fprintf(stderr, "GPUEngine: enabling cudaMallocAsync with default memory pool.\n");
        }
}

bool GPUEngine::AllocateDeviceBuffer(void** ptr, size_t size)
{
        if (ptr == nullptr || size == 0) {
                return true;
        }

#if CUDART_VERSION >= 11020
        if (useAsyncAlloc_) {
                const cudaError_t status = cudaMallocAsync(ptr, size, stream_);
                if (status == cudaSuccess) {
                        return true;
                }
                if (status == cudaErrorNotSupported || status == cudaErrorInvalidValue) {
                        cudaGetLastError();
                        useAsyncAlloc_ = false;
                        if (!asyncFallbackNotified_) {
                                std::fprintf(stderr, "GPUEngine: cudaMallocAsync unsupported, falling back to cudaMalloc.\n");
                                asyncFallbackNotified_ = true;
                        }
                }
                else {
                        CUDA_CHECK(status);
                        return true;
                }
        }
#else
        useAsyncAlloc_ = false;
        if (!asyncFallbackNotified_) {
                std::fprintf(stderr, "GPUEngine: CUDA toolkit lacks cudaMallocAsync support, using cudaMalloc instead.\n");
                asyncFallbackNotified_ = true;
        }
#endif

        CUDA_CHECK(cudaMalloc(ptr, size));
        return true;
}

void GPUEngine::FreeDeviceBuffer(void** ptr)
{
        if (ptr == nullptr || *ptr == nullptr) {
                return;
        }

#if CUDART_VERSION >= 11020
        if (useAsyncAlloc_) {
                const cudaError_t status = cudaFreeAsync(*ptr, stream_);
                if (status == cudaSuccess) {
                        *ptr = nullptr;
                        return;
                }
                if (status == cudaErrorNotSupported || status == cudaErrorInvalidValue) {
                        cudaGetLastError();
                        useAsyncAlloc_ = false;
                        if (!asyncFallbackNotified_) {
                                std::fprintf(stderr, "GPUEngine: cudaFreeAsync unsupported, falling back to cudaFree.\n");
                                asyncFallbackNotified_ = true;
                        }
                }
                else {
                        CUDA_CHECK(status);
                        *ptr = nullptr;
                        return;
                }
        }
#else
        useAsyncAlloc_ = false;
        if (!asyncFallbackNotified_) {
                std::fprintf(stderr, "GPUEngine: CUDA toolkit lacks cudaFreeAsync support, using cudaFree instead.\n");
                asyncFallbackNotified_ = true;
        }
#endif

        CUDA_CHECK(cudaFree(*ptr));
        *ptr = nullptr;
}

void GPUEngine::SynchronizeStreamIfNeeded()
{
        if (streamCreated_ && stream_ != nullptr) {
                CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
}

// ----------------------------------------------------------------------------

void GPUEngine::InitGenratorTable(Secp256K1* secp)
{

        // generator table
	uint64_t* _2GnxPinned = nullptr;
	uint64_t* _2GnyPinned = nullptr;

	uint64_t* GxPinned = nullptr;
	uint64_t* GyPinned = nullptr;

	const uint64_t size = static_cast<uint64_t>(GRP_SIZE);
	constexpr int nbDigit = 4;
	const size_t limbBytes = static_cast<size_t>(nbDigit) * sizeof(uint64_t);

        AllocateDeviceBuffer(reinterpret_cast<void**>(&__2Gnx), limbBytes);
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&_2GnxPinned), limbBytes,
                cudaHostAllocWriteCombined | cudaHostAllocMapped));

        AllocateDeviceBuffer(reinterpret_cast<void**>(&__2Gny), limbBytes);
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&_2GnyPinned), limbBytes,
                cudaHostAllocWriteCombined | cudaHostAllocMapped));

        const size_t tableSize = static_cast<size_t>(size / 2) * limbBytes;
        const size_t tableWords = tableSize / sizeof(uint64_t);
        AllocateDeviceBuffer(reinterpret_cast<void**>(&_Gx), tableSize);
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&GxPinned), tableSize,
                cudaHostAllocWriteCombined | cudaHostAllocMapped));

        AllocateDeviceBuffer(reinterpret_cast<void**>(&_Gy), tableSize);
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&GyPinned), tableSize,
                cudaHostAllocWriteCombined | cudaHostAllocMapped));

	std::vector<Point> Gn(size);
	Point g = secp->G;
	Gn[0] = g;
	g = secp->DoubleDirect(g);
	Gn[1] = g;
	for (uint64_t i = 2; i < size; ++i) {
		g = secp->AddDirect(g, secp->G);
		Gn[i] = g;
	}
	// _2Gn = CPU_GRP_SIZE*G
	const Point _2Gn = secp->DoubleDirect(Gn[size / 2 - 1]);

	for (int i = 0; i < nbDigit; ++i) {
		_2GnxPinned[i] = _2Gn.x.bits64[i];
		_2GnyPinned[i] = _2Gn.y.bits64[i];
	}
	for (uint64_t i = 0; i < size / 2; ++i) {
		for (int j = 0; j < nbDigit; ++j) {
			GxPinned[i * nbDigit + j] = Gn[i].x.bits64[j];
			GyPinned[i * nbDigit + j] = Gn[i].y.bits64[j];
		}
	}

	CUDA_CHECK(cudaMemcpyAsync(__2Gnx, _2GnxPinned, limbBytes, cudaMemcpyHostToDevice, stream_));
	CUDA_CHECK(cudaMemcpyAsync(__2Gny, _2GnyPinned, limbBytes, cudaMemcpyHostToDevice, stream_));
	CUDA_CHECK(cudaMemcpyAsync(_Gx, GxPinned, tableSize, cudaMemcpyHostToDevice, stream_));
	CUDA_CHECK(cudaMemcpyAsync(_Gy, GyPinned, tableSize, cudaMemcpyHostToDevice, stream_));

	CUDA_CHECK(cudaMemcpyToSymbolAsync(_2Gnx, &__2Gnx, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice, stream_));
	CUDA_CHECK(cudaMemcpyToSymbolAsync(_2Gny, &__2Gny, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice, stream_));
	CUDA_CHECK(cudaMemcpyToSymbolAsync(Gx, &_Gx, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice, stream_));
	CUDA_CHECK(cudaMemcpyToSymbolAsync(Gy, &_Gy, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice, stream_));

        CUDA_CHECK(cudaStreamSynchronize(stream_));

        KH_InitCudaOptimizations(stream_, tableWords);

        CUDA_CHECK(cudaFreeHost(_2GnxPinned));
	CUDA_CHECK(cudaFreeHost(_2GnyPinned));
	CUDA_CHECK(cudaFreeHost(GxPinned));
	CUDA_CHECK(cudaFreeHost(GyPinned));

}

// ----------------------------------------------------------------------------

int GPUEngine::GetGroupSize()
{
        return GRP_SIZE * stepMultiplier;
}

// ----------------------------------------------------------------------------

uint64_t GPUEngine::GetStepSize() const
{
        return static_cast<uint64_t>(GRP_SIZE) * static_cast<uint64_t>(stepMultiplier);
}

// ----------------------------------------------------------------------------

int GPUEngine::GetStepMultiplier() const
{
        return stepMultiplier;
}

// ----------------------------------------------------------------------------

int GPUEngine::GetThreadsPerGroup()
{
        return nbThreadPerGroup;
}

// ----------------------------------------------------------------------------

int GPUEngine::GetCompiledGroupSize()
{
        return GRP_SIZE;
}

// ----------------------------------------------------------------------------

void GPUEngine::PrintCudaInfo()
{
        const char* sComputeMode[] = {
                "Multiple host threads",
                "Only one host thread",
                "No host thread",
                "Multiple process threads",
                "Unknown",
                nullptr
        };

        try {
                cuda_compat::EnsureRuntimeSupportsCuda13();
        }
        catch (const std::exception& ex) {
                std::fprintf(stderr, "GPUEngine: %s\n", ex.what());
                return;
        }

        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

        // This function call returns 0 if there are no CUDA capable devices.
        if (deviceCount == 0) {
                printf("GPUEngine: There are no available device(s) that support CUDA\n");
                return;
        }

        int runtimeVersion = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
        int driverVersion = 0;
        CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
        printf("CUDA Runtime %s, Driver %s\n",
                cuda_compat::FormatCudaVersion(runtimeVersion).c_str(),
                cuda_compat::FormatCudaVersion(driverVersion).c_str());

        for (int i = 0; i < deviceCount; i++) {
                CUDA_CHECK(cudaSetDevice(i));
                cudaDeviceProp deviceProp;
                CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, i));
                const DeviceCapabilityInfo capability = QueryDeviceCapabilityInfo(i);
                int computeModeIndex = 4;
                int computeModeValue = -1;
                if (cudaDeviceGetAttribute(&computeModeValue, cudaDevAttrComputeMode, i) == cudaSuccess) {
                        if (computeModeValue >= 0 && computeModeValue < 4) {
                                computeModeIndex = computeModeValue;
                        }
                }

                printf("GPU #%d %s (%dx%d cores) (Cap %d.%d) (%.1f MB) (%s)\n",
                        i, deviceProp.name, deviceProp.multiProcessorCount,
                        _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
                        deviceProp.major, deviceProp.minor, (double)deviceProp.totalGlobalMem / 1048576.0,
                        sComputeMode[computeModeIndex]);
                printf("    Warp size: %d, Max threads/SM: %d, Max blocks/SM: %d\n",
                        capability.warpSize,
                        capability.maxThreadsPerMultiprocessor,
                        capability.maxBlocksPerMultiprocessor);
                printf("    Memory pools: %s, Coop multi-device: %s, Cluster launch: %s, Mem sync domains: %d\n",
                        capability.memoryPoolsSupported ? "yes" : "no",
                        capability.cooperativeMultiDeviceLaunch ? "yes" : "no",
                        capability.clusterLaunch ? "yes" : "no",
                        capability.memSyncDomainCount);
        }
}

// ----------------------------------------------------------------------------

GPUEngine::~GPUEngine()
{
        SynchronizeStreamIfNeeded();

        if (inputKey != nullptr) {
                FreeDeviceBuffer(reinterpret_cast<void**>(&inputKey));
        }

        if (inputBloomLookUp != nullptr) {
                FreeDeviceBuffer(reinterpret_cast<void**>(&inputBloomLookUp));
        }

        if (inputHashORxpoint != nullptr) {
                FreeDeviceBuffer(reinterpret_cast<void**>(&inputHashORxpoint));
        }

        if (inputBloomLookUpPinned != nullptr) {
                CUDA_CHECK(cudaFreeHost(inputBloomLookUpPinned));
                inputBloomLookUpPinned = nullptr;
        }

        if (inputHashORxpointPinned != nullptr) {
                CUDA_CHECK(cudaFreeHost(inputHashORxpointPinned));
                inputHashORxpointPinned = nullptr;
        }

        if (outputBufferPinned != nullptr) {
                CUDA_CHECK(cudaFreeHost(outputBufferPinned));
                outputBufferPinned = nullptr;
        }

        if (outputBuffer != nullptr) {
                FreeDeviceBuffer(reinterpret_cast<void**>(&outputBuffer));
        }

        if (__2Gnx != nullptr) {
                FreeDeviceBuffer(reinterpret_cast<void**>(&__2Gnx));
        }

        if (__2Gny != nullptr) {
                FreeDeviceBuffer(reinterpret_cast<void**>(&__2Gny));
        }

        if (_Gx != nullptr) {
                FreeDeviceBuffer(reinterpret_cast<void**>(&_Gx));
        }

        if (_Gy != nullptr) {
                FreeDeviceBuffer(reinterpret_cast<void**>(&_Gy));
        }

        if (rKey && inputKeyPinned != nullptr) {
                CUDA_CHECK(cudaFreeHost(inputKeyPinned));
                inputKeyPinned = nullptr;
        }

#if CUDART_VERSION >= 11020
        if (useAsyncAlloc_ && memPool_ != nullptr) {
                const cudaError_t trimStatus = cudaMemPoolTrimTo(memPool_, 0);
                if (trimStatus == cudaErrorNotSupported || trimStatus == cudaErrorInvalidValue) {
                        cudaGetLastError();
                }
                else {
                        CUDA_CHECK(trimStatus);
                }
        }
#endif

        if (eventCreated_) {
                CUDA_CHECK(cudaEventDestroy(syncEvent_));
                syncEvent_ = nullptr;
                eventCreated_ = false;
        }

        if (streamCreated_) {
                CUDA_CHECK(cudaStreamDestroy(stream_));
                stream_ = nullptr;
                streamCreated_ = false;
        }
}

// ----------------------------------------------------------------------------

int GPUEngine::GetNbThread()
{
	return nbThread;
}

// ----------------------------------------------------------------------------

bool GPUEngine::callKernelSEARCH_MODE_MA()
{

        if (!initialised) {
                return false;
        }

        if (activeThreadCount <= 0) {
                return false;
        }

        if (nbThreadPerGroup <= 0) {
                return false;
        }

        // Reset nbFound
        CUDA_CHECK(cudaMemsetAsync(outputBuffer, 0, sizeof(uint32_t), stream_));

        const dim3 gridDim(static_cast<unsigned>(activeThreadCount / nbThreadPerGroup));
        const dim3 blockDim(static_cast<unsigned>(nbThreadPerGroup));

        // Call the kernel (Perform STEP_SIZE keys per thread)
        if (coinType == COIN_BTC) {
                if (compMode == SEARCH_COMPRESSED) {
                        LaunchKeyKernel(compute_keys_comp_mode_ma, gridDim, blockDim,
                                compMode, inputBloomLookUp, BLOOM_BITS, BLOOM_HASHES, bloomFastModReciprocal_,
                                bloomMask_, bloomIsPowerOfTwo_, inputKey, maxFound, outputBuffer, stepMultiplier);
                }
                else {
                        LaunchKeyKernel(compute_keys_mode_ma, gridDim, blockDim,
                                compMode, inputBloomLookUp, BLOOM_BITS, BLOOM_HASHES, bloomFastModReciprocal_,
                                bloomMask_, bloomIsPowerOfTwo_, inputKey, maxFound, outputBuffer, stepMultiplier);
                }
        }
        else {
                LaunchKeyKernel(compute_keys_mode_eth_ma, gridDim, blockDim,
                        inputBloomLookUp, BLOOM_BITS, BLOOM_HASHES, bloomFastModReciprocal_,
                        bloomMask_, bloomIsPowerOfTwo_, inputKey, maxFound, outputBuffer, stepMultiplier);
        }

        CUDA_CHECK(cudaPeekAtLastError());
        return true;

}

// ----------------------------------------------------------------------------

bool GPUEngine::callKernelSEARCH_MODE_MX()
{

        if (!initialised) {
                return false;
        }

        if (activeThreadCount <= 0) {
                return false;
        }

        if (nbThreadPerGroup <= 0) {
                return false;
        }

        // Reset nbFound
        CUDA_CHECK(cudaMemsetAsync(outputBuffer, 0, sizeof(uint32_t), stream_));

        const dim3 gridDim(static_cast<unsigned>(activeThreadCount / nbThreadPerGroup));
        const dim3 blockDim(static_cast<unsigned>(nbThreadPerGroup));

        // Call the kernel (Perform STEP_SIZE keys per thread)
        if (compMode == SEARCH_COMPRESSED) {
                LaunchKeyKernel(compute_keys_comp_mode_mx, gridDim, blockDim,
                        compMode, inputBloomLookUp, BLOOM_BITS, BLOOM_HASHES, bloomFastModReciprocal_,
                        bloomMask_, bloomIsPowerOfTwo_, inputKey, maxFound, outputBuffer, stepMultiplier);
        }
        else {
                printf("GPUEngine: PubKeys search doesn't support uncompressed\n");
                return false;
        }

        CUDA_CHECK(cudaPeekAtLastError());
        return true;
}

// ----------------------------------------------------------------------------

bool GPUEngine::callKernelSEARCH_MODE_SA()
{

        if (!initialised) {
                return false;
        }

        if (activeThreadCount <= 0) {
                return false;
        }

        if (nbThreadPerGroup <= 0) {
                return false;
        }

        // Reset nbFound
        CUDA_CHECK(cudaMemsetAsync(outputBuffer, 0, sizeof(uint32_t), stream_));

        const dim3 gridDim(static_cast<unsigned>(activeThreadCount / nbThreadPerGroup));
        const dim3 blockDim(static_cast<unsigned>(nbThreadPerGroup));

        // Call the kernel (Perform STEP_SIZE keys per thread)
        if (coinType == COIN_BTC) {
                if (compMode == SEARCH_COMPRESSED) {
                        LaunchKeyKernel(compute_keys_comp_mode_sa, gridDim, blockDim,
                                compMode, inputHashORxpoint, inputKey, maxFound, outputBuffer, stepMultiplier);
                }
                else {
                        LaunchKeyKernel(compute_keys_mode_sa, gridDim, blockDim,
                                compMode, inputHashORxpoint, inputKey, maxFound, outputBuffer, stepMultiplier);
                }
        }
        else {
                LaunchKeyKernel(compute_keys_mode_eth_sa, gridDim, blockDim,
                        inputHashORxpoint, inputKey, maxFound, outputBuffer, stepMultiplier);
        }

        CUDA_CHECK(cudaPeekAtLastError());
        return true;

}

// ----------------------------------------------------------------------------

bool GPUEngine::callKernelSEARCH_MODE_SX()
{

        // Reset nbFound
        if (!initialised) {
                return false;
        }

        if (activeThreadCount <= 0) {
                return false;
        }

        if (nbThreadPerGroup <= 0) {
                return false;
        }

        CUDA_CHECK(cudaMemsetAsync(outputBuffer, 0, sizeof(uint32_t), stream_));

        const dim3 gridDim(static_cast<unsigned>(activeThreadCount / nbThreadPerGroup));
        const dim3 blockDim(static_cast<unsigned>(nbThreadPerGroup));

        // Call the kernel (Perform STEP_SIZE keys per thread)
        if (compMode == SEARCH_COMPRESSED) {
                LaunchKeyKernel(compute_keys_comp_mode_sx, gridDim, blockDim,
                        compMode, inputHashORxpoint, inputKey, maxFound, outputBuffer, stepMultiplier);
        }
        else {
                printf("GPUEngine: PubKeys search doesn't support uncompressed\n");
                return false;
        }

        CUDA_CHECK(cudaPeekAtLastError());
        return true;
}

// ----------------------------------------------------------------------------

void GPUEngine::waitForStream(bool spinWait)
{
        if (!streamCreated_) {
                return;
        }

        if (spinWait || !eventCreated_) {
                CUDA_CHECK(cudaStreamSynchronize(stream_));
                return;
        }

        CUDA_CHECK(cudaEventRecord(syncEvent_, stream_));
        CUDA_CHECK(cudaEventSynchronize(syncEvent_));
}

// ----------------------------------------------------------------------------

bool GPUEngine::SetKeys(Point* p, int activeThreadCountOverride)
{
        // Sets the starting keys for each thread
        // p must contains nbThread public keys
        if (!initialised) {
                return false;
        }

        int requestedThreadCount = activeThreadCountOverride;
        if (requestedThreadCount <= 0 || requestedThreadCount > nbThread) {
                requestedThreadCount = nbThread;
        }
        if (requestedThreadCount <= 0) {
                return false;
        }

        activeThreadCount = requestedThreadCount;

        if (nbThreadPerGroup <= 0) {
                return false;
        }

        if ((activeThreadCount % nbThreadPerGroup) != 0) {
                const int groups = (activeThreadCount + nbThreadPerGroup - 1) / nbThreadPerGroup;
                activeThreadCount = groups * nbThreadPerGroup;
                if (activeThreadCount > nbThread) {
                        activeThreadCount = nbThread;
                }
        }

        if (warpSize_ > 0 && (activeThreadCount % warpSize_) != 0) {
                const int warps = (activeThreadCount + warpSize_ - 1) / warpSize_;
                int padded = warps * warpSize_;
                if (padded > nbThread) {
                        const int trimmed = nbThread - (nbThread % warpSize_);
                        if (trimmed > 0) {
                                padded = trimmed;
                        }
                        else {
                                padded = nbThread;
                        }
                }
                if (padded > 0) {
                        activeThreadCount = padded;
                }
        }

        const int sourceCount = std::min(requestedThreadCount, activeThreadCount);
        const int safeSourceCount = std::max(sourceCount, 1);
        const int maxSourceIndex = safeSourceCount - 1;

        for (int i = 0; i < activeThreadCount; i += nbThreadPerGroup) {
                for (int j = 0; j < nbThreadPerGroup && (i + j) < activeThreadCount; j++) {
                        const int logicalIndex = i + j;
                        const int sourceIndex = logicalIndex <= maxSourceIndex ? logicalIndex : maxSourceIndex;

                        inputKeyPinned[8 * i + j + 0 * nbThreadPerGroup] = p[sourceIndex].x.bits64[0];
                        inputKeyPinned[8 * i + j + 1 * nbThreadPerGroup] = p[sourceIndex].x.bits64[1];
                        inputKeyPinned[8 * i + j + 2 * nbThreadPerGroup] = p[sourceIndex].x.bits64[2];
                        inputKeyPinned[8 * i + j + 3 * nbThreadPerGroup] = p[sourceIndex].x.bits64[3];

                        inputKeyPinned[8 * i + j + 4 * nbThreadPerGroup] = p[sourceIndex].y.bits64[0];
                        inputKeyPinned[8 * i + j + 5 * nbThreadPerGroup] = p[sourceIndex].y.bits64[1];
                        inputKeyPinned[8 * i + j + 6 * nbThreadPerGroup] = p[sourceIndex].y.bits64[2];
                        inputKeyPinned[8 * i + j + 7 * nbThreadPerGroup] = p[sourceIndex].y.bits64[3];

                }
        }

        // Fill device memory
        const size_t keyBufferSize = static_cast<size_t>(activeThreadCount) * 32u * 2u;
        CUDA_CHECK(cudaMemcpyAsync(inputKey, inputKeyPinned, keyBufferSize, cudaMemcpyHostToDevice, stream_));
        waitForStream(true);

        if (!rKey) {
                // We do not need the input pinned memory anymore
                CUDA_CHECK(cudaFreeHost(inputKeyPinned));
                inputKeyPinned = nullptr;
	}

	switch (searchMode) {
	case (int)SEARCH_MODE_MA:
		return callKernelSEARCH_MODE_MA();
		break;
	case (int)SEARCH_MODE_SA:
		return callKernelSEARCH_MODE_SA();
		break;
	case (int)SEARCH_MODE_MX:
		return callKernelSEARCH_MODE_MX();
		break;
	case (int)SEARCH_MODE_SX:
		return callKernelSEARCH_MODE_SX();
		break;
	default:
		return false;
		break;
	}
}

// ----------------------------------------------------------------------------

bool GPUEngine::LaunchSEARCH_MODE_MA(std::vector<ITEM>& dataFound, bool spinWait, bool queueNextBatch)
{

        if (!initialised) {
                return false;
        }

        dataFound.clear();

        CUDA_CHECK(cudaMemcpyAsync(outputBufferPinned, outputBuffer, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_));
        waitForStream(spinWait);

        // Look for data found
        uint32_t nbFound = outputBufferPinned[0];
        if (nbFound > maxFound) {
                nbFound = maxFound;
        }

        const size_t payloadBytes = static_cast<size_t>(nbFound) * ITEM_SIZE_A + sizeof(uint32_t);
        if (payloadBytes > sizeof(uint32_t)) {
                CUDA_CHECK(cudaMemcpyAsync(outputBufferPinned, outputBuffer, payloadBytes, cudaMemcpyDeviceToHost, stream_));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        if (nbFound > 0) {
                dataFound.reserve(nbFound);
        }

        for (uint32_t i = 0; i < nbFound; i++) {

                uint32_t* itemPtr = outputBufferPinned + (i * ITEM_SIZE_A32 + 1);
                uint8_t* hash = (uint8_t*)(itemPtr + 2);
                if (CheckBinary(hash, 20) > 0) {

                        const uint32_t packed = itemPtr[1];
                        ITEM it;
                        it.thId = itemPtr[0];
                        it.mode = (packed & 0x80000000u) != 0u;
                        it.incr = (packed & 0x7FFFFFFFu);
                        it.hash = (uint8_t*)(itemPtr + 2);
                        dataFound.push_back(it);
                }
        }
        if (!queueNextBatch) {
                return true;
        }
        return callKernelSEARCH_MODE_MA();
}

// ----------------------------------------------------------------------------

bool GPUEngine::LaunchSEARCH_MODE_SA(std::vector<ITEM>& dataFound, bool spinWait, bool queueNextBatch)
{

        if (!initialised) {
                return false;
        }

        dataFound.clear();

        CUDA_CHECK(cudaMemcpyAsync(outputBufferPinned, outputBuffer, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_));
        waitForStream(spinWait);

        // Look for data found
        uint32_t nbFound = outputBufferPinned[0];
        if (nbFound > maxFound) {
                nbFound = maxFound;
        }

        const size_t payloadBytes = static_cast<size_t>(nbFound) * ITEM_SIZE_A + sizeof(uint32_t);
        if (payloadBytes > sizeof(uint32_t)) {
                CUDA_CHECK(cudaMemcpyAsync(outputBufferPinned, outputBuffer, payloadBytes, cudaMemcpyDeviceToHost, stream_));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        if (nbFound > 0) {
                dataFound.reserve(nbFound);
        }

        for (uint32_t i = 0; i < nbFound; i++) {
                uint32_t* itemPtr = outputBufferPinned + (i * ITEM_SIZE_A32 + 1);
                const uint32_t packed = itemPtr[1];
                ITEM it;
                it.thId = itemPtr[0];
                it.mode = (packed & 0x80000000u) != 0u;
                it.incr = (packed & 0x7FFFFFFFu);
                it.hash = (uint8_t*)(itemPtr + 2);
                dataFound.push_back(it);
        }
        if (!queueNextBatch) {
                return true;
        }
        return callKernelSEARCH_MODE_SA();
}

// ----------------------------------------------------------------------------

bool GPUEngine::LaunchSEARCH_MODE_MX(std::vector<ITEM>& dataFound, bool spinWait, bool queueNextBatch)
{

        if (!initialised) {
                return false;
        }

        dataFound.clear();

        CUDA_CHECK(cudaMemcpyAsync(outputBufferPinned, outputBuffer, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_));
        waitForStream(spinWait);

        // Look for data found
        uint32_t nbFound = outputBufferPinned[0];
        if (nbFound > maxFound) {
                nbFound = maxFound;
        }

        const size_t payloadBytes = static_cast<size_t>(nbFound) * ITEM_SIZE_X + sizeof(uint32_t);
        if (payloadBytes > sizeof(uint32_t)) {
                CUDA_CHECK(cudaMemcpyAsync(outputBufferPinned, outputBuffer, payloadBytes, cudaMemcpyDeviceToHost, stream_));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        if (nbFound > 0) {
                dataFound.reserve(nbFound);
        }

        for (uint32_t i = 0; i < nbFound; i++) {

                uint32_t* itemPtr = outputBufferPinned + (i * ITEM_SIZE_X32 + 1);
                uint8_t* pubkey = (uint8_t*)(itemPtr + 2);

                if (CheckBinary(pubkey, 32) > 0) {

                        const uint32_t packed = itemPtr[1];
                        ITEM it;
                        it.thId = itemPtr[0];
                        it.mode = (packed & 0x80000000u) != 0u;
                        it.incr = (packed & 0x7FFFFFFFu);
                        it.hash = (uint8_t*)(itemPtr + 2);
                        dataFound.push_back(it);
                }
        }
        if (!queueNextBatch) {
                return true;
        }
        return callKernelSEARCH_MODE_MX();
}

// ----------------------------------------------------------------------------

bool GPUEngine::LaunchSEARCH_MODE_SX(std::vector<ITEM>& dataFound, bool spinWait, bool queueNextBatch)
{

        if (!initialised) {
                return false;
        }

        dataFound.clear();

        CUDA_CHECK(cudaMemcpyAsync(outputBufferPinned, outputBuffer, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_));
        waitForStream(spinWait);

        // Look for data found
        uint32_t nbFound = outputBufferPinned[0];
        if (nbFound > maxFound) {
                nbFound = maxFound;
        }

        const size_t payloadBytes = static_cast<size_t>(nbFound) * ITEM_SIZE_X + sizeof(uint32_t);
        if (payloadBytes > sizeof(uint32_t)) {
                CUDA_CHECK(cudaMemcpyAsync(outputBufferPinned, outputBuffer, payloadBytes, cudaMemcpyDeviceToHost, stream_));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        if (nbFound > 0) {
                dataFound.reserve(nbFound);
        }

        for (uint32_t i = 0; i < nbFound; i++) {

                uint32_t* itemPtr = outputBufferPinned + (i * ITEM_SIZE_X32 + 1);
                uint8_t* pubkey = (uint8_t*)(itemPtr + 2);

                ITEM it;
                const uint32_t packed = itemPtr[1];
                it.thId = itemPtr[0];
                it.mode = (packed & 0x80000000u) != 0u;
                it.incr = (packed & 0x7FFFFFFFu);
                it.hash = (uint8_t*)(itemPtr + 2);
                dataFound.push_back(it);
        }
        if (!queueNextBatch) {
                return true;
        }
        return callKernelSEARCH_MODE_SX();
}

// ----------------------------------------------------------------------------

int GPUEngine::CheckBinary(const uint8_t* _x, int K_LENGTH)
{
	uint8_t* temp_read;
	uint64_t half, min, max, current; //, current_offset
	int64_t rcmp;
	int32_t r = 0;
	min = 0;
	current = 0;
	max = TOTAL_COUNT;
	half = TOTAL_COUNT;
	while (!r && half >= 1) {
		half = (max - min) / 2;
		temp_read = DATA + ((current + half) * K_LENGTH);
		rcmp = memcmp(_x, temp_read, K_LENGTH);
		if (rcmp == 0) {
			r = 1;  //Found!!
		}
		else {
			if (rcmp < 0) { //data < temp_read
				max = (max - half);
			}
			else { // data > temp_read
				min = (min + half);
			}
			current = min;
		}
	}
	return r;
}




