// Copyright (C) 2026 AI4BayesCode contributors. GPL-3.0-or-later. See LICENSE at the repository root.
// math_constants.hpp
//
//  The POSIX math constants M_PI, M_LN2, ... are not part of ISO C++. glibc
//  and macOS define them anyway, but MinGW-w64 (Windows: Rtools, MSYS2) hides
//  them when compiling with -std=c++17 instead of -std=gnu++17. The library
//  and its examples use them, so define each one that is missing, spelled
//  exactly as MinGW-w64's <math.h> spells it, so that a later definition from
//  the system header is identical and harmless.
// ============================================================================
#ifndef AI4BAYESCODE_MATH_CONSTANTS_HPP
#define AI4BAYESCODE_MATH_CONSTANTS_HPP

#include <cmath>

#ifndef M_E
#define M_E 2.7182818284590452354
#endif
#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634074
#endif
#ifndef M_LOG10E
#define M_LOG10E 0.43429448190325182765
#endif
#ifndef M_LN2
#define M_LN2 0.69314718055994530942
#endif
#ifndef M_LN10
#define M_LN10 2.30258509299404568402
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif
#ifndef M_PI_4
#define M_PI_4 0.78539816339744830962
#endif
#ifndef M_1_PI
#define M_1_PI 0.31830988618379067154
#endif
#ifndef M_2_PI
#define M_2_PI 0.63661977236758134308
#endif
#ifndef M_2_SQRTPI
#define M_2_SQRTPI 1.12837916709551257390
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

#endif  // AI4BAYESCODE_MATH_CONSTANTS_HPP
