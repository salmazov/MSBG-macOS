/******************************************************************************
 *
 * Minimal SIMD compatibility shim:
 * - On arm64, pull in SIMDe's AVX2 implementation and expose native-like
 *   intrinsic names via SIMDE_ENABLE_NATIVE_ALIASES.
 * - On x86/x64, fall back to the platform intrinsics.
 *
 * This keeps existing _mm/_mm256 code compiling on Apple Silicon.
 *
 ******************************************************************************/

#ifndef SIMD_COMPAT_H
#define SIMD_COMPAT_H

#if defined(__aarch64__)
  #ifndef SIMDE_ENABLE_NATIVE_ALIASES
    #define SIMDE_ENABLE_NATIVE_ALIASES
  #endif
  #include <simde/x86/avx2.h>
#else
  #include <immintrin.h>
#endif

#endif // SIMD_COMPAT_H
