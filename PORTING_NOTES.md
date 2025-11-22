# macOS ARM Porting Notes

## Overall goal
Port the MSBG codebase from x86/vectorclass/TBB dependencies to native macOS on Apple Silicon (M4 Pro) using SIMDe for SIMD operations, OpenMP for threading, and Homebrew JPEG/PNG libraries.

## Work completed so far
- All sources now compile against SIMDe-backed wrappers instead of vectorclass; includes point to `simd_types.h` / `vectorclass_util.h`.
- Implemented SIMDe wrappers for `Vec4f/Vec4i/Vec8f/Vec8i/Vec4d/Vec8us/Vec4uq` with load/store (aligned/partial), arithmetic, comparisons, masks/select, gathers/scatters, horizontal ops, conversions, and helpers (`round_to_int`, `approx_rsqrt`, `vany/vall`, maskstore, etc.).
- Simplified `src/vectorclass_util2.h` to 2-wide aliases and added SIMD utility helpers (`vfget_*`, shuffles, blend8f, vmask4, etc.).
- `fastmath.*` RNG and math helpers ported to the new vector layer; scalar fallbacks retained.
- Added SIMDe/TBB vendored copies under `external/`, plus `src/simd_types.h` shim.
- Build system tweaks: fixed TMPDIR on macOS, corrected linker invocation order, and retained Homebrew include/lib paths for libpng/libjpeg.
- Resolved modulation/cross-product path in `pnoise3` and other SIMDe mismatches; `../mk` now succeeds from `build/` on Apple Silicon.

## Current status
- `../mk` (run inside `build/`) builds `libmsbg.a` and `msbg_demo` on macOS ARM with gcc-15/OpenMP and SIMDe. No remaining link errors.
- Warnings: strict-aliasing warnings around partial loads/mask stores in `vectorclass_util.h`; currently benign but worth a later cleanup.
- Runtime verification still pending (no numerical validation yet).

## Next steps
1) Sanity-check runtime: run `./msbg_demo` with representative inputs; compare against x86 output if available.
2) Address strict-alias warnings by reworking partial load/maskstore helpers to avoid type-punning stacks.
3) Consider trimming vendored SIMDe/TBB files if not needed beyond what is included.
4) Optional: replace remaining legacy macros (e.g., vstream/vmaskstore) with safer wrappers once behavior is validated.

## How to build
From `build/`: `../mk`
Requires Homebrew libs (`libpng`, `libjpeg`) and gcc-15 with OpenMP. Output: `libmsbg.a`, `msbg_demo`.
