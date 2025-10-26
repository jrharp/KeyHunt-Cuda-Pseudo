#pragma once

#include <cstdint>

struct DeviceKernelParams
{
        uint64_t bloomBits = 0;
        uint64_t bloomReciprocal = 0;
        uint64_t* keyBuffer = nullptr;
        uint32_t* foundBuffer = nullptr;
        uint8_t* bloomLookUp = nullptr;
        uint32_t* hashOrXpoint = nullptr;
        uint32_t mode = 0;
        uint32_t maxFound = 0;
        int32_t stepMultiplier = 1;
        uint32_t bloomMask = 0;
        uint32_t bloomIsPowerOfTwo = 0;
        uint8_t bloomHashes = 0;
        uint8_t reserved[7] = {0};
};

