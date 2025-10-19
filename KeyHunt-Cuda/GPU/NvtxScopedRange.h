#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>

#if defined(__has_include)
#  if __has_include(<nvtx3/nvToolsExt.h>)
#    include <nvtx3/nvToolsExt.h>
#    define KEYHUNT_HAS_NVTX 1
#  elif __has_include(<nvToolsExt.h>)
#    include <nvToolsExt.h>
#    define KEYHUNT_HAS_NVTX 1
#  endif
#endif

#ifndef KEYHUNT_HAS_NVTX
#  define KEYHUNT_HAS_NVTX 0
#endif

#if KEYHUNT_HAS_NVTX && defined(__has_include)
#  if __has_include(<nvtx3/nvToolsExtCuda.h>)
#    include <nvtx3/nvToolsExtCuda.h>
#    define KEYHUNT_HAS_NVTX_CUDA 1
#  elif __has_include(<nvToolsExtCuda.h>)
#    include <nvToolsExtCuda.h>
#    define KEYHUNT_HAS_NVTX_CUDA 1
#  endif
#endif

#ifndef KEYHUNT_HAS_NVTX_CUDA
#  define KEYHUNT_HAS_NVTX_CUDA 0
#endif

namespace keyhunt::profiling {

namespace detail {

inline constexpr std::uint32_t Fnv1aHash(const char* text) noexcept
{
        std::uint32_t hash = 0x811C9DC5u;
        if (text == nullptr) {
                return hash;
        }
        while (*text != '\0') {
                hash ^= static_cast<std::uint8_t>(*text++);
                hash *= 0x01000193u;
        }
        return hash;
}

inline constexpr std::uint32_t NormalizeColor(std::uint32_t hash) noexcept
{
        constexpr std::uint32_t alpha = 0xFFu << 24;
        const std::uint32_t rgb = hash & 0x00FFFFFFu;
        return (rgb == 0u) ? (alpha | 0x00FF00u) : (alpha | rgb);
}

} // namespace detail

class NvtxScopedRange
{
public:
        explicit NvtxScopedRange(const char* name) noexcept
#if KEYHUNT_HAS_NVTX
                : rangeId_(0)
#endif
        {
#if KEYHUNT_HAS_NVTX
                if (name == nullptr) {
                        name = "Kernel";
                }
                nvtxEventAttributes_t attributes{};
                attributes.version = NVTX_VERSION;
                attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
                attributes.colorType = NVTX_COLOR_ARGB;
                attributes.color = detail::NormalizeColor(detail::Fnv1aHash(name));
                attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
                attributes.message.ascii = name;
                rangeId_ = nvtxRangeStartEx(&attributes);
#else
                (void)name;
#endif
        }

        NvtxScopedRange(const NvtxScopedRange&) = delete;
        NvtxScopedRange& operator=(const NvtxScopedRange&) = delete;

        NvtxScopedRange(NvtxScopedRange&& other) noexcept = delete;
        NvtxScopedRange& operator=(NvtxScopedRange&& other) noexcept = delete;

        ~NvtxScopedRange()
        {
#if KEYHUNT_HAS_NVTX
                if (rangeId_ != 0) {
                        nvtxRangeEnd(rangeId_);
                }
#endif
        }

private:
#if KEYHUNT_HAS_NVTX
        nvtxRangeId_t rangeId_;
#endif
};

inline void NameCudaStream(cudaStream_t stream, const char* name) noexcept
{
#if KEYHUNT_HAS_NVTX && KEYHUNT_HAS_NVTX_CUDA
        if (stream != nullptr && name != nullptr) {
#  if defined(NVTX_VERSION) && (NVTX_VERSION >= 0x030000)
                nvtxNameCuStreamA(reinterpret_cast<CUstream>(stream), name);
#  elif defined(NVTX_VERSION)
                nvtxNameCudaStreamA(stream, name);
#  endif
        }
#else
        (void)stream;
        (void)name;
#endif
}

inline constexpr bool IsAvailable() noexcept
{
        return KEYHUNT_HAS_NVTX != 0;
}

} // namespace keyhunt::profiling

