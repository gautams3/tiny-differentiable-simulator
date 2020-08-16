#pragma once

#define TINY_INLINE  __attribute__ ((always_inline)) inline

typedef void (*SubmitProfileTiming)(const std::string& profileName);
