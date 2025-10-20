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

#ifndef GPUENGINEH
#define GPUENGINEH

#include <cuda_runtime_api.h>
#include <string>
#include <vector>
#include "../SECP256k1.h"

#ifdef __cplusplus
int RecommendOccupancyBlockSizeForDevice(int deviceId);
#endif

#define SEARCH_COMPRESSED 0
#define SEARCH_UNCOMPRESSED 1
#define SEARCH_BOTH 2

// operating mode
#define SEARCH_MODE_MA 1	// multiple addresses
#define SEARCH_MODE_SA 2	// single address
#define SEARCH_MODE_MX 3	// multiple xpoints
#define SEARCH_MODE_SX 4	// single xpoint

#define COIN_BTC 1
#define COIN_ETH 2

// Number of key per thread (must be a multiple of GRP_SIZE) per kernel call
#define STEP_SIZE (1024*2)

// Number of thread per block
#define ITEM_SIZE_A 28
#define ITEM_SIZE_A32 (ITEM_SIZE_A/4)

#define ITEM_SIZE_X 40
#define ITEM_SIZE_X32 (ITEM_SIZE_X/4)

typedef struct {
        uint32_t thId;
        uint32_t incr;
        uint8_t* hash;
        bool mode;
} ITEM;

class GPUEngine
{

public:

        GPUEngine(Secp256K1* secp, int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound,
                int searchMode, int compMode, int coinType, int64_t BLOOM_SIZE, uint64_t BLOOM_BITS,
                uint8_t BLOOM_HASHES, const uint8_t* BLOOM_DATA, uint8_t* DATA, uint64_t TOTAL_COUNT, bool rKey,
                int stepMultiplier);

        GPUEngine(Secp256K1* secp, int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound,
                int searchMode, int compMode, int coinType, const uint32_t* hashORxpoint, bool rKey,
                int stepMultiplier);

	~GPUEngine();

        bool SetKeys(Point* p, int activeThreadCount = -1);

        bool LaunchSEARCH_MODE_MA(std::vector<ITEM>& dataFound, bool spinWait = false, bool queueNextBatch = true);
        bool LaunchSEARCH_MODE_SA(std::vector<ITEM>& dataFound, bool spinWait = false, bool queueNextBatch = true);
        bool LaunchSEARCH_MODE_MX(std::vector<ITEM>& dataFound, bool spinWait = false, bool queueNextBatch = true);
        bool LaunchSEARCH_MODE_SX(std::vector<ITEM>& dataFound, bool spinWait = false, bool queueNextBatch = true);

        int GetNbThread();
        int GetGroupSize();
        uint64_t GetStepSize() const;
        int GetStepMultiplier() const;
        int GetThreadsPerGroup();
        static int GetCompiledGroupSize();

	//bool Check(Secp256K1 *secp);
	std::string deviceName;

	static void PrintCudaInfo();
	static void GenerateCode(Secp256K1* secp, int size);

private:
        void InitGenratorTable(Secp256K1* secp);

        void ConfigureAsyncAllocator(int deviceId);
        bool AllocateDeviceBuffer(void** ptr, size_t size);
        void FreeDeviceBuffer(void** ptr);
        void SynchronizeStreamIfNeeded();

        bool callKernelSEARCH_MODE_MA();
        bool callKernelSEARCH_MODE_SA();
        bool callKernelSEARCH_MODE_MX();
        bool callKernelSEARCH_MODE_SX();

        void waitForStream(bool spinWait);

        int CheckBinary(const uint8_t* x, int K_LENGTH);

        template <typename KernelFunc, typename... Args>
        bool LaunchKeyKernel(KernelFunc kernel, dim3 gridDim, dim3 blockDim, Args&&... args);

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12000)
        template <typename KernelFunc>
        dim3 QueryClusterDimension(KernelFunc kernel, dim3 gridDim, dim3 blockDim);
#endif

        int nbThread = 0;
        int nbThreadPerGroup = 0;
        int activeThreadCount = 0;
        int stepMultiplier = 1;
        int warpSize_ = 0;

        uint32_t* inputHashORxpoint = nullptr;
        uint32_t* inputHashORxpointPinned = nullptr;

        //uint8_t *bloomLookUp;
        uint8_t* inputBloomLookUp = nullptr;
        uint8_t* inputBloomLookUpPinned = nullptr;

        uint64_t bloomFastModReciprocal_ = 0;
        uint32_t bloomMask_ = 0;
        uint32_t bloomIsPowerOfTwo_ = 0;

        uint64_t* inputKey = nullptr;
        uint64_t* inputKeyPinned = nullptr;

        uint32_t* outputBuffer = nullptr;
        uint32_t* outputBufferPinned = nullptr;

        uint64_t* __2Gnx = nullptr;
        uint64_t* __2Gny = nullptr;

        uint64_t* _Gx = nullptr;
        uint64_t* _Gy = nullptr;

        bool initialised = false;
        uint32_t compMode = 0;
        uint32_t searchMode = 0;
        uint32_t coinType = 0;
        bool littleEndian = false;

        bool rKey = false;
        uint32_t maxFound = 0;
        uint32_t outputSize = 0;

        int64_t BLOOM_SIZE = 0;
        uint64_t BLOOM_BITS = 0;
        uint8_t BLOOM_HASHES = 0;

        uint8_t* DATA = nullptr;
        uint64_t TOTAL_COUNT = 0;

        cudaStream_t stream_ = nullptr;
        cudaEvent_t syncEvent_ = nullptr;
        bool streamCreated_ = false;
        bool eventCreated_ = false;

        bool useAsyncAlloc_ = false;
        cudaMemPool_t memPool_ = nullptr;
        bool asyncFallbackNotified_ = false;

        bool clusterLaunchSupported_ = false;
        bool clusterLaunchActive_ = false;
        bool cooperativeLaunchSupported_ = false;
        bool cooperativeLaunchActive_ = false;

};

#endif // GPUENGINEH
