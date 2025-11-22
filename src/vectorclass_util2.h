#ifndef VECTORCLASS_UTIL2_H
#define VECTORCLASS_UTIL2_H

#include "simd_types.h"

// For this ARM/SIMDe port we collapse the 2-wide helper types onto the 4-wide
// SIMD types. Call sites use only the lower lanes.
using Vec2f = Vec4f;
using Vec2ui = Vec4ui;

#endif  // VECTORCLASS_UTIL2_H
