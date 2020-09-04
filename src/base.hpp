#pragma once

#include <string>

#if defined(_MSC_VER)
#define TINY_INLINE inline
#else
#define TINY_INLINE __attribute__((always_inline)) inline
#endif

typedef void (*SubmitProfileTiming)(const std::string& profileName);
