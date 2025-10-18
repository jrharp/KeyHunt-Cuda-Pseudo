#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <cstdio>

namespace cuda_compat {

inline std::string FormatCudaVersion(int version)
{
        const int major = version / 1000;
        const int minor = (version % 1000) / 10;
        const int patch = version % 10;

        std::ostringstream oss;
        oss << major << "." << minor;
        if (patch != 0) {
                oss << "." << patch;
        }
        return oss.str();
}

inline void ThrowCudaError(const char* function, cudaError_t status)
{
        std::ostringstream oss;
        oss << function << " failed with error " << cudaGetErrorName(status)
            << ": " << cudaGetErrorString(status);
        throw std::runtime_error(oss.str());
}

inline void EnsureRuntimeSupportsCuda13()
{
        int rt = 0;
        const cudaError_t status = cudaRuntimeGetVersion(&rt);
        if (status != cudaSuccess) {
                ThrowCudaError("cudaRuntimeGetVersion", status);
        }
        if (rt < 12000) {
                std::fprintf(stderr, "Warning: CUDA runtime version %d < 12000; some optimizations are disabled.\n", rt);
        }
        // You may warm up hot kernels by calling cudaFuncGetAttributes on them here.
}

} // namespace cuda_compat

