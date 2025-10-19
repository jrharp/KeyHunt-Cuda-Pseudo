#pragma once

#include <cstdint>

#if defined(USE_NVTX)
#include <nvToolsExt.h>
#endif

namespace gpu_instrumentation
{
#if defined(USE_NVTX)

class ScopedNvtxRange
{
public:
        explicit ScopedNvtxRange(const char* name)
        {
                nvtxEventAttributes_t attributes{};
                attributes.version = NVTX_VERSION;
                attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
                attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
                attributes.message.ascii = name;
                nvtxRangePushEx(&attributes);
        }

        ScopedNvtxRange(const char* name, std::uint32_t argbColor)
        {
                nvtxEventAttributes_t attributes{};
                attributes.version = NVTX_VERSION;
                attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
                attributes.colorType = NVTX_COLOR_ARGB;
                attributes.color = argbColor;
                attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
                attributes.message.ascii = name;
                nvtxRangePushEx(&attributes);
        }

        ScopedNvtxRange(const ScopedNvtxRange&) = delete;
        ScopedNvtxRange& operator=(const ScopedNvtxRange&) = delete;

        ~ScopedNvtxRange()
        {
                nvtxRangePop();
        }
};

inline void Mark(const char* message)
{
        nvtxMarkA(message);
}

#else

class ScopedNvtxRange
{
public:
        explicit ScopedNvtxRange(const char*) {}
        ScopedNvtxRange(const char*, std::uint32_t) {}
};

inline void Mark(const char*) {}

#endif
} // namespace gpu_instrumentation

#if defined(USE_NVTX)
#define NVTX_RANGE(name) gpu_instrumentation::ScopedNvtxRange nvtxRange##__LINE__{name}
#define NVTX_RANGE_COLOR(name, color) gpu_instrumentation::ScopedNvtxRange nvtxRange##__LINE__{name, color}
#else
#define NVTX_RANGE(name) (void)0
#define NVTX_RANGE_COLOR(name, color) (void)0
#endif

