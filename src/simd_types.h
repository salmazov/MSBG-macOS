/******************************************************************************
 * SIMD helper layer for ARM/NEON via SIMDe.
 *
 * This is a starting point to replace the existing vectorclass + x86 intrinsics
 * usage. It intentionally keeps the surface small; add helpers as you port
 * individual call sites.
 ******************************************************************************/
#ifndef SIMDTYPES_H
#define SIMDTYPES_H

#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>
#include <simde/x86/sse4.1.h>
#include <simde/x86/sse2.h>

// Basic aliases for readability
using v4f = simde__m128;
using v4d = simde__m128d;
using v4i = simde__m128i;
using v8f = simde__m256;
using v8i = simde__m256i;

// Construction helpers
inline v4f v4f_set1(float x) { return simde_mm_set1_ps(x); }
inline v8f v8f_set1(float x) { return simde_mm256_set1_ps(x); }
inline v4i v4i_set1(int32_t x) { return simde_mm_set1_epi32(x); }
inline v8i v8i_set1(int32_t x) { return simde_mm256_set1_epi32(x); }

inline v4f v4f_loadu(const float* p) { return simde_mm_loadu_ps(p); }
inline void v4f_storeu(float* p, v4f v) { simde_mm_storeu_ps(p, v); }
inline v8f v8f_loadu(const float* p) { return simde_mm256_loadu_ps(p); }
inline void v8f_storeu(float* p, v8f v) { simde_mm256_storeu_ps(p, v); }

// Arithmetic
inline v4f v4f_add(v4f a, v4f b) { return simde_mm_add_ps(a, b); }
inline v4f v4f_sub(v4f a, v4f b) { return simde_mm_sub_ps(a, b); }
inline v4f v4f_mul(v4f a, v4f b) { return simde_mm_mul_ps(a, b); }
inline v4f v4f_fmadd(v4f a, v4f b, v4f c) { return simde_mm_fmadd_ps(a, b, c); }
inline v4f v4f_min(v4f a, v4f b) { return simde_mm_min_ps(a, b); }
inline v4f v4f_max(v4f a, v4f b) { return simde_mm_max_ps(a, b); }

inline v8f v8f_add(v8f a, v8f b) { return simde_mm256_add_ps(a, b); }
inline v8f v8f_sub(v8f a, v8f b) { return simde_mm256_sub_ps(a, b); }
inline v8f v8f_mul(v8f a, v8f b) { return simde_mm256_mul_ps(a, b); }
inline v8f v8f_fmadd(v8f a, v8f b, v8f c) { return simde_mm256_fmadd_ps(a, b, c); }
inline v8f v8f_min(v8f a, v8f b) { return simde_mm256_min_ps(a, b); }
inline v8f v8f_max(v8f a, v8f b) { return simde_mm256_max_ps(a, b); }

// Comparisons / masks
inline v4i v4f_cmp_gt(v4f a, v4f b) { return simde_mm_castps_si128(simde_mm_cmpgt_ps(a, b)); }
inline v4i v4f_cmp_lt(v4f a, v4f b) { return simde_mm_castps_si128(simde_mm_cmplt_ps(a, b)); }
inline int v4f_movemask(v4f a) { return simde_mm_movemask_ps(a); }

// Horizontal sum (float)
inline float v4f_hsum(v4f v) {
  v = simde_mm_hadd_ps(v, v);
  v = simde_mm_hadd_ps(v, v);
  return simde_mm_cvtss_f32(v);
}

inline float v8f_hsum(v8f v) {
  v4f lo = simde_mm256_castps256_ps128(v);
  v4f hi = simde_mm256_extractf128_ps(v, 1);
  return v4f_hsum(simde_mm_add_ps(lo, hi));
}

// Utility: zero out lane >= pos for v4f
template<int pos>
inline v4f v4f_zero_to(v4f v) {
  static_assert(pos >= 0 && pos <= 3, "pos out of range");
  const int mask = (pos == 0) ? 0x0 : (pos == 1) ? 0x1 : (pos == 2) ? 0x3 : 0x7;
  return simde_mm_blend_ps(simde_mm_setzero_ps(), v, mask);
}

#endif // SIMDTYPES_H
