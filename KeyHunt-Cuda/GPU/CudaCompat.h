#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>
#include <string>

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
#if CUDA_VERSION < 13000
#error "CUDA 13.0 or newer headers are required to build this project."
#endif
        int runtimeVersion = 0;
        const cudaError_t runtimeStatus = cudaRuntimeGetVersion(&runtimeVersion);
        if (runtimeStatus != cudaSuccess) {
                ThrowCudaError("cudaRuntimeGetVersion", runtimeStatus);
        }
        if (runtimeVersion < 13000) {
                std::ostringstream oss;
                oss << "Detected CUDA runtime version " << FormatCudaVersion(runtimeVersion)
                    << " but CUDA 13.0 or newer is required.";
                throw std::runtime_error(oss.str());
        }

        int driverVersion = 0;
        const cudaError_t driverStatus = cudaDriverGetVersion(&driverVersion);
        if (driverStatus != cudaSuccess) {
                ThrowCudaError("cudaDriverGetVersion", driverStatus);
        }
        if (driverVersion < 13000) {
                std::ostringstream oss;
                oss << "Detected CUDA driver version " << FormatCudaVersion(driverVersion)
                    << " but CUDA 13.0 or newer is required.";
                throw std::runtime_error(oss.str());
        }
}

} // namespace cuda_compat

