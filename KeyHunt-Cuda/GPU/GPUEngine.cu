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

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>

#include "../hash/ripemd160.h"
#include "../hash/sha256.h"
#include "../Timer.h"

#include "GPUBase58.h"
#include "GPUCompute.h"
#include "GPUHash.h"
#include "GPUMath.h"

namespace {

inline void CheckCuda(cudaError_t result, const char* expr, const char* file, int line)
{
        if (result != cudaSuccess) {
                std::fprintf(stderr, "CUDA failure %s at %s:%d: %s (%s)\n",
                        expr, file, line, cudaGetErrorName(result), cudaGetErrorString(result));
                std::abort();
        }
}

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

} // namespace

// ---------------------------------------------------------------------------------------

// mode multiple addresses
__global__ void compute_keys_mode_ma(uint32_t mode, uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES,
	uint64_t* keys, uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	ComputeKeysSEARCH_MODE_MA(mode, keys + xPtr, keys + yPtr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, found);

}

__global__ void compute_keys_comp_mode_ma(uint32_t mode, uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t* keys,
	uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	ComputeKeysSEARCH_MODE_MA(mode, keys + xPtr, keys + yPtr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, found);

}

// mode single address
__global__ void compute_keys_mode_sa(uint32_t mode, uint32_t* hash160, uint64_t* keys, uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	ComputeKeysSEARCH_MODE_SA(mode, keys + xPtr, keys + yPtr, hash160, maxFound, found);

}

__global__ void compute_keys_comp_mode_sa(uint32_t mode, uint32_t* hash160, uint64_t* keys, uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	ComputeKeysSEARCH_MODE_SA(mode, keys + xPtr, keys + yPtr, hash160, maxFound, found);

}

// mode multiple x points
__global__ void compute_keys_comp_mode_mx(uint32_t mode, uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t* keys,
	uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	ComputeKeysSEARCH_MODE_MX(mode, keys + xPtr, keys + yPtr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, found);

}

// mode single x point
__global__ void compute_keys_comp_mode_sx(uint32_t mode, uint32_t* xpoint, uint64_t* keys, uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	ComputeKeysSEARCH_MODE_SX(mode, keys + xPtr, keys + yPtr, xpoint, maxFound, found);

}

// ---------------------------------------------------------------------------------------
// ethereum

__global__ void compute_keys_mode_eth_ma(uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t* keys,
	uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	ComputeKeysSEARCH_ETH_MODE_MA(keys + xPtr, keys + yPtr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, found);

}

__global__ void compute_keys_mode_eth_sa(uint32_t* hash, uint64_t* keys, uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	ComputeKeysSEARCH_ETH_MODE_SA(keys + xPtr, keys + yPtr, hash, maxFound, found);

}

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
        uint8_t BLOOM_HASHES, const uint8_t* BLOOM_DATA, uint8_t* DATA, uint64_t TOTAL_COUNT, bool rKey)
{

        // Initialise CUDA
        this->nbThreadPerGroup = nbThreadPerGroup;
        this->searchMode = searchMode;
        this->compMode = compMode;
        this->coinType = coinType;
        this->rKey = rKey;

        this->BLOOM_SIZE = BLOOM_SIZE;
        this->BLOOM_BITS = BLOOM_BITS;
        this->BLOOM_HASHES = BLOOM_HASHES;
        this->DATA = DATA;
        this->TOTAL_COUNT = TOTAL_COUNT;

        initialised = false;

        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

        // This function call returns 0 if there are no CUDA capable devices.
        if (deviceCount == 0) {
                printf("GPUEngine: There are no available device(s) that support CUDA\n");
                return;
        }

        CUDA_CHECK(cudaSetDevice(gpuId));

        cudaDeviceProp deviceProp{};
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, gpuId));

        if (nbThreadGroup == -1) {
                nbThreadGroup = deviceProp.multiProcessorCount * 8;
        }

        this->nbThread = nbThreadGroup * nbThreadPerGroup;
        this->maxFound = maxFound;
        this->outputSize = (maxFound * ITEM_SIZE_A + 4);
        if (this->searchMode == (int)SEARCH_MODE_MX) {
                this->outputSize = (maxFound * ITEM_SIZE_X + 4);
        }

        char tmp[512];
        std::snprintf(tmp, sizeof(tmp), "GPU #%d %s (%dx%d cores) Grid(%dx%d)",
                gpuId, deviceProp.name, deviceProp.multiProcessorCount,
                _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
                nbThread / nbThreadPerGroup,
                nbThreadPerGroup);
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
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&inputKey), keyBufferSize));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&inputKeyPinned), keyBufferSize,
                cudaHostAllocWriteCombined | cudaHostAllocMapped));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&outputBuffer), outputSize));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&outputBufferPinned), outputSize,
                cudaHostAllocWriteCombined | cudaHostAllocMapped));

        if (BLOOM_SIZE > 0) {
                const size_t bloomBytes = static_cast<size_t>(BLOOM_SIZE);
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&inputBloomLookUp), bloomBytes));
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
        int searchMode, int compMode, int coinType, const uint32_t* hashORxpoint, bool rKey)
{

        // Initialise CUDA
        this->nbThreadPerGroup = nbThreadPerGroup;
        this->searchMode = searchMode;
        this->compMode = compMode;
        this->coinType = coinType;
        this->rKey = rKey;

        initialised = false;

        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

        // This function call returns 0 if there are no CUDA capable devices.
        if (deviceCount == 0) {
                printf("GPUEngine: There are no available device(s) that support CUDA\n");
                return;
        }

        CUDA_CHECK(cudaSetDevice(gpuId));

        cudaDeviceProp deviceProp{};
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, gpuId));

        if (nbThreadGroup == -1) {
                nbThreadGroup = deviceProp.multiProcessorCount * 8;
        }

        this->nbThread = nbThreadGroup * nbThreadPerGroup;
        this->maxFound = maxFound;
        this->outputSize = (maxFound * ITEM_SIZE_A + 4);
        if (this->searchMode == (int)SEARCH_MODE_SX) {
                this->outputSize = (maxFound * ITEM_SIZE_X + 4);
        }

        char tmp[512];
        std::snprintf(tmp, sizeof(tmp), "GPU #%d %s (%dx%d cores) Grid(%dx%d)",
                gpuId, deviceProp.name, deviceProp.multiProcessorCount,
                _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
                nbThread / nbThreadPerGroup,
                nbThreadPerGroup);
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
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&inputKey), keyBufferSize));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&inputKeyPinned), keyBufferSize,
                cudaHostAllocWriteCombined | cudaHostAllocMapped));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&outputBuffer), outputSize));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&outputBufferPinned), outputSize,
                cudaHostAllocWriteCombined | cudaHostAllocMapped));

        int K_SIZE = 5;
        if (this->searchMode == (int)SEARCH_MODE_SX) {
                K_SIZE = 8;
        }

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&inputHashORxpoint), K_SIZE * sizeof(uint32_t)));
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

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&__2Gnx), limbBytes));
	CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&_2GnxPinned), limbBytes,
	        cudaHostAllocWriteCombined | cudaHostAllocMapped));

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&__2Gny), limbBytes));
	CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&_2GnyPinned), limbBytes,
	        cudaHostAllocWriteCombined | cudaHostAllocMapped));

	const size_t tableSize = static_cast<size_t>(size / 2) * limbBytes;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_Gx), tableSize));
	CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&GxPinned), tableSize,
	        cudaHostAllocWriteCombined | cudaHostAllocMapped));

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_Gy), tableSize));
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

	CUDA_CHECK(cudaFreeHost(_2GnxPinned));
	CUDA_CHECK(cudaFreeHost(_2GnyPinned));
	CUDA_CHECK(cudaFreeHost(GxPinned));
	CUDA_CHECK(cudaFreeHost(GyPinned));

}

// ----------------------------------------------------------------------------

int GPUEngine::GetGroupSize()
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

	int deviceCount = 0;
	CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		printf("GPUEngine: There are no available device(s) that support CUDA\n");
		return;
	}

	for (int i = 0; i < deviceCount; i++) {
		CUDA_CHECK(cudaSetDevice(i));
		cudaDeviceProp deviceProp;
		CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, i));
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
        }
}

// ----------------------------------------------------------------------------

GPUEngine::~GPUEngine()
{
        if (streamCreated_) {
                CUDA_CHECK(cudaStreamDestroy(stream_));
                stream_ = nullptr;
                streamCreated_ = false;
        }

        if (eventCreated_) {
                CUDA_CHECK(cudaEventDestroy(syncEvent_));
                syncEvent_ = nullptr;
                eventCreated_ = false;
        }

        if (inputKey != nullptr) {
                CUDA_CHECK(cudaFree(inputKey));
                inputKey = nullptr;
        }

        if (inputBloomLookUp != nullptr) {
                CUDA_CHECK(cudaFree(inputBloomLookUp));
                inputBloomLookUp = nullptr;
        }

        if (inputHashORxpoint != nullptr) {
                CUDA_CHECK(cudaFree(inputHashORxpoint));
                inputHashORxpoint = nullptr;
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
                CUDA_CHECK(cudaFree(outputBuffer));
                outputBuffer = nullptr;
        }

        if (__2Gnx != nullptr) {
                CUDA_CHECK(cudaFree(__2Gnx));
                __2Gnx = nullptr;
        }

        if (__2Gny != nullptr) {
                CUDA_CHECK(cudaFree(__2Gny));
                __2Gny = nullptr;
        }

        if (_Gx != nullptr) {
                CUDA_CHECK(cudaFree(_Gx));
                _Gx = nullptr;
        }

        if (_Gy != nullptr) {
                CUDA_CHECK(cudaFree(_Gy));
                _Gy = nullptr;
        }

        if (rKey && inputKeyPinned != nullptr) {
                CUDA_CHECK(cudaFreeHost(inputKeyPinned));
                inputKeyPinned = nullptr;
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

        // Reset nbFound
        CUDA_CHECK(cudaMemsetAsync(outputBuffer, 0, sizeof(uint32_t), stream_));

        // Call the kernel (Perform STEP_SIZE keys per thread)
        if (coinType == COIN_BTC) {
                if (compMode == SEARCH_COMPRESSED) {
                        compute_keys_comp_mode_ma << < nbThread / nbThreadPerGroup, nbThreadPerGroup, 0, stream_ >> >
                                (compMode, inputBloomLookUp, BLOOM_BITS, BLOOM_HASHES, inputKey, maxFound, outputBuffer);
                }
                else {
                        compute_keys_mode_ma << < nbThread / nbThreadPerGroup, nbThreadPerGroup, 0, stream_ >> >
                                (compMode, inputBloomLookUp, BLOOM_BITS, BLOOM_HASHES, inputKey, maxFound, outputBuffer);
                }
        }
        else {
                compute_keys_mode_eth_ma << < nbThread / nbThreadPerGroup, nbThreadPerGroup, 0, stream_ >> >
                        (inputBloomLookUp, BLOOM_BITS, BLOOM_HASHES, inputKey, maxFound, outputBuffer);
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

        // Reset nbFound
        CUDA_CHECK(cudaMemsetAsync(outputBuffer, 0, sizeof(uint32_t), stream_));

        // Call the kernel (Perform STEP_SIZE keys per thread)
        if (compMode == SEARCH_COMPRESSED) {
                compute_keys_comp_mode_mx << < nbThread / nbThreadPerGroup, nbThreadPerGroup, 0, stream_ >> >
                        (compMode, inputBloomLookUp, BLOOM_BITS, BLOOM_HASHES, inputKey, maxFound, outputBuffer);
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

        // Reset nbFound
        CUDA_CHECK(cudaMemsetAsync(outputBuffer, 0, sizeof(uint32_t), stream_));

        // Call the kernel (Perform STEP_SIZE keys per thread)
        if (coinType == COIN_BTC) {
                if (compMode == SEARCH_COMPRESSED) {
                        compute_keys_comp_mode_sa << < nbThread / nbThreadPerGroup, nbThreadPerGroup, 0, stream_ >> >
                                (compMode, inputHashORxpoint, inputKey, maxFound, outputBuffer);
                }
                else {
                        compute_keys_mode_sa << < nbThread / nbThreadPerGroup, nbThreadPerGroup, 0, stream_ >> >
                                (compMode, inputHashORxpoint, inputKey, maxFound, outputBuffer);
                }
        }
        else {
                compute_keys_mode_eth_sa << < nbThread / nbThreadPerGroup, nbThreadPerGroup, 0, stream_ >> >
                        (inputHashORxpoint, inputKey, maxFound, outputBuffer);
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

        CUDA_CHECK(cudaMemsetAsync(outputBuffer, 0, sizeof(uint32_t), stream_));

        // Call the kernel (Perform STEP_SIZE keys per thread)
        if (compMode == SEARCH_COMPRESSED) {
                compute_keys_comp_mode_sx << < nbThread / nbThreadPerGroup, nbThreadPerGroup, 0, stream_ >> >
                        (compMode, inputHashORxpoint, inputKey, maxFound, outputBuffer);
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
        while (true) {
                const cudaError_t status = cudaEventQuery(syncEvent_);
                if (status == cudaSuccess) {
                        break;
                }
                if (status != cudaErrorNotReady) {
                        CUDA_CHECK(status);
                }
                Timer::SleepMillis(1);
        }
}

// ----------------------------------------------------------------------------

bool GPUEngine::SetKeys(Point* p)
{
        // Sets the starting keys for each thread
        // p must contains nbThread public keys
        if (!initialised) {
                return false;
        }

        for (int i = 0; i < nbThread; i += nbThreadPerGroup) {
                for (int j = 0; j < nbThreadPerGroup; j++) {

                        inputKeyPinned[8 * i + j + 0 * nbThreadPerGroup] = p[i + j].x.bits64[0];
			inputKeyPinned[8 * i + j + 1 * nbThreadPerGroup] = p[i + j].x.bits64[1];
			inputKeyPinned[8 * i + j + 2 * nbThreadPerGroup] = p[i + j].x.bits64[2];
			inputKeyPinned[8 * i + j + 3 * nbThreadPerGroup] = p[i + j].x.bits64[3];

			inputKeyPinned[8 * i + j + 4 * nbThreadPerGroup] = p[i + j].y.bits64[0];
			inputKeyPinned[8 * i + j + 5 * nbThreadPerGroup] = p[i + j].y.bits64[1];
			inputKeyPinned[8 * i + j + 6 * nbThreadPerGroup] = p[i + j].y.bits64[2];
			inputKeyPinned[8 * i + j + 7 * nbThreadPerGroup] = p[i + j].y.bits64[3];

                }
        }

        // Fill device memory
        const size_t keyBufferSize = static_cast<size_t>(nbThread) * 32u * 2u;
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

bool GPUEngine::LaunchSEARCH_MODE_MA(std::vector<ITEM>& dataFound, bool spinWait)
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

			ITEM it;
			it.thId = itemPtr[0];
			int16_t* ptr = (int16_t*)&(itemPtr[1]);
			//it.endo = ptr[0] & 0x7FFF;
			it.mode = (ptr[0] & 0x8000) != 0;
			it.incr = ptr[1];
			it.hash = (uint8_t*)(itemPtr + 2);
			dataFound.push_back(it);
		}
	}
	return callKernelSEARCH_MODE_MA();
}

// ----------------------------------------------------------------------------

bool GPUEngine::LaunchSEARCH_MODE_SA(std::vector<ITEM>& dataFound, bool spinWait)
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
                ITEM it;
                it.thId = itemPtr[0];
		int16_t* ptr = (int16_t*)&(itemPtr[1]);
		//it.endo = ptr[0] & 0x7FFF;
		it.mode = (ptr[0] & 0x8000) != 0;
		it.incr = ptr[1];
		it.hash = (uint8_t*)(itemPtr + 2);
		dataFound.push_back(it);
	}
	return callKernelSEARCH_MODE_SA();
}

// ----------------------------------------------------------------------------

bool GPUEngine::LaunchSEARCH_MODE_MX(std::vector<ITEM>& dataFound, bool spinWait)
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

			ITEM it;
			it.thId = itemPtr[0];
			int16_t* ptr = (int16_t*)&(itemPtr[1]);
			//it.endo = ptr[0] & 0x7FFF;
			it.mode = (ptr[0] & 0x8000) != 0;
			it.incr = ptr[1];
			it.hash = (uint8_t*)(itemPtr + 2);
			dataFound.push_back(it);
		}
	}
	return callKernelSEARCH_MODE_MX();
}

// ----------------------------------------------------------------------------

bool GPUEngine::LaunchSEARCH_MODE_SX(std::vector<ITEM>& dataFound, bool spinWait)
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
		it.thId = itemPtr[0];
		int16_t* ptr = (int16_t*)&(itemPtr[1]);
		//it.endo = ptr[0] & 0x7FFF;
		it.mode = (ptr[0] & 0x8000) != 0;
		it.incr = ptr[1];
		it.hash = (uint8_t*)(itemPtr + 2);
		dataFound.push_back(it);
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




