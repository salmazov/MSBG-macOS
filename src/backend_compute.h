/******************************************************************************
 * Backend dispatch for compute kernels (PDE smoothing).
 *
 * GPU path is currently a stub that:
 *   - Packs active blocks and computes a checksum for validation
 *   - Logs the packed block count and checksum
 *   - Falls back to CPU implementation (applyChannelPdeFast)
 *
 * GPU implementation (implemented in metal_renderer.mm):
 *   - Packs active blocks with 1-voxel halo regions from neighbor blocks
 *   - Converts uint16_t to float for GPU processing
 *   - Uploads to Metal buffers and dispatches mean curvature compute kernel
 *   - Processes multiple iterations with halo reconstruction between iterations
 *   - Reads back results and converts float back to uint16_t
 *   - Falls back to CPU if Metal unavailable or on error
 *
 * Status: Phase 3 (GPU multigrid smoother) - fully implemented and ready for testing.
 ******************************************************************************/
#pragma once

#include <vector>
#include <cstdint>
#include <cstring>

#include "msbg_demo.h"
#include "msbg.h"
#include "metal_renderer_stub.h"

namespace Backend {

namespace detail {

// FNV-1a 64-bit over raw bytes.
inline uint64_t fnv1a64(const uint8_t* data, size_t count) {
  constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
  constexpr uint64_t kFnvPrime = 1099511628211ULL;
  uint64_t hash = kFnvOffset;
  for (size_t i = 0; i < count; ++i) {
    hash ^= static_cast<uint64_t>(data[i]);
    hash *= kFnvPrime;
  }
  return hash;
}

template <typename T>
struct PackedPdeBlocks {
  int blockSize = 0;
  int voxelsPerBlock = 0;
  std::vector<int> blockIds;
  std::vector<T> voxels;
  uint64_t checksum = 0;
};

template <typename T>
PackedPdeBlocks<T> packBlocks(MSBG::MultiresSparseGrid* msbg,
                              int chan,
                              const std::vector<int>* activeBlocks) {
  PackedPdeBlocks<T> packed;

  // For now, use level 0 channel.
  SBG::SparseGrid<T>* sg = nullptr;
  if constexpr (std::is_same<T, float>::value) {
    sg = msbg->getFloatChannel(chan, 0, 0);
  } else if constexpr (std::is_same<T, uint16_t>::value) {
    sg = msbg->getUint16Channel(chan, 0, 0);
  } else if constexpr (std::is_same<T, uint8_t>::value) {
    sg = msbg->getUint8Channel(chan, 0, 0);
  }
  if (!sg) {
    TRCP(("GPU PDE stub: channel %d not found; skipping pack.\n", chan));
    return packed;
  }

  packed.blockSize = sg->bsx();
  packed.voxelsPerBlock = static_cast<int>(sg->nVoxelsInBlock());
  packed.blockIds.reserve(activeBlocks ? activeBlocks->size() : 0);

  if (!activeBlocks) {
    return packed;
  }

  for (int bid : *activeBlocks) {
    T* data = sg->getBlockDataPtr(bid);
    if (!data) {
      continue;
    }
    packed.blockIds.push_back(bid);
    packed.voxels.insert(packed.voxels.end(), data,
                         data + packed.voxelsPerBlock);
  }

  packed.checksum = fnv1a64(reinterpret_cast<const uint8_t*>(packed.voxels.data()),
                            packed.voxels.size() * sizeof(T));
  return packed;
}

}  // namespace detail

template <typename RenderDensityT>
void applyMeanCurvaturePde(int testCase,
                           MSBG::MultiresSparseGrid* msbg,
                           int chan,
                           std::vector<int>* activeBlocks,
                           BackendType backend,
                           int nIter,
                           float dt) {
  if (backend == BACKEND_GPU) {
    // Try GPU path first (currently only implemented for uint16_t)
    bool gpuSuccess = false;
    if constexpr (std::is_same<RenderDensityT, uint16_t>::value) {
      uint64_t gpuChecksum = 0;
      gpuSuccess = Backend::applyMeanCurvaturePdeMetalUint16(
          msbg, chan, activeBlocks, nIter, dt, &gpuChecksum);
      if (gpuSuccess) {
        TRCP(("GPU PDE: completed %d iterations, checksum=0x%016llx\n",
              nIter, static_cast<unsigned long long>(gpuChecksum)));
        return;
      }
    }
    
    // Fallback: pack blocks for validation/logging
    auto packed = detail::packBlocks<RenderDensityT>(msbg, chan, activeBlocks);
    TRCP(("GPU PDE: packed %zu blocks (bs=%d, vox/block=%d), checksum=0x%016llx; falling back to CPU.\n",
          packed.blockIds.size(), packed.blockSize, packed.voxelsPerBlock,
          static_cast<unsigned long long>(packed.checksum)));
  }

  // CPU path (or GPU fallback)
  msbg->applyChannelPdeFast<RenderDensityT>(
      chan, CH_NULL, CH_NULL, activeBlocks,
      -(PDE_MEAN_CURVATURE + OPT_8_COLOR_SCHEME), TRUE,
      0, 0, 0,
      nIter,
      1.0f,
      dt,
      0, NULL, 0, 0, 0, 0, 0, 0, 0,
      0);
}

}  // namespace Backend
