/******************************************************************************
 * Metal renderer entry point (currently prepares data + falls back to CPU).
 *
 * This file keeps all Metal-specific plumbing isolated so the rest of the
 * codebase can stay pure C++. The runtime probe avoids hard-linking against
 * Metal when not present.
 ******************************************************************************/

#include "metal_renderer_stub.h"
#include "gpu_block_packing.h"

#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using Backend::MetalAvailability;

constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

uint64_t fnv1a64(const uint16_t* data, size_t count) {
  uint64_t hash = kFnvOffset;
  for (size_t i = 0; i < count; ++i) {
    hash ^= static_cast<uint64_t>(data[i]);
    hash *= kFnvPrime;
  }
  return hash;
}

MetalAvailability probeMetalAvailability() {
  MetalAvailability availability;

  void* handle = dlopen("/System/Library/Frameworks/Metal.framework/Metal",
                        RTLD_LAZY);
  if (!handle) {
    availability.reason = "Metal.framework not found";
    return availability;
  }

  availability.runtimePresent = true;

  using CreateDeviceFn = void* (*)();
  auto createDevice = reinterpret_cast<CreateDeviceFn>(
      dlsym(handle, "MTLCreateSystemDefaultDevice"));
  if (!createDevice) {
    availability.reason = "MTLCreateSystemDefaultDevice symbol missing";
    dlclose(handle);
    return availability;
  }

  void* device = createDevice();
  if (!device) {
    availability.reason = "MTLCreateSystemDefaultDevice returned null";
    dlclose(handle);
    return availability;
  }

  availability.devicePresent = true;
  availability.reason = "Metal device detected";
  dlclose(handle);
  return availability;
}

}  // namespace

namespace Backend {

MetalAvailability queryMetalAvailability() {
  static const MetalAvailability cached = probeMetalAvailability();
  return cached;
}

bool isMetalAvailable() {
  const MetalAvailability availability = queryMetalAvailability();
  return availability.runtimePresent && availability.devicePresent;
}

bool renderSceneMetal(int /*testCase*/, MSBG::MultiresSparseGrid* msbg,
                      int chan, const std::vector<int>* /*activeBlocks*/,
                      PnlPanel* /*pnl*/, uint64_t* checksumOut) {
  const MetalAvailability availability = queryMetalAvailability();
  if (!availability.runtimePresent || !availability.devicePresent) {
    TRCP(("GPU renderer unavailable (%s); using CPU path.\n",
          availability.reason.c_str()));
    if (checksumOut) *checksumOut = 0;
    return false;
  }

  SBG::SparseGrid<uint16_t>* sg = msbg->getUint16Channel(chan, 0);
  if (!sg) {
    TRCP(("GPU renderer: missing uint16 density channel; using CPU path.\n"));
    if (checksumOut) *checksumOut = 0;
    return false;
  }

  PackedBlockVolume packed = packActiveBlocksForGpu(sg);

  if (packed.blocks.empty()) {
    TRCP(("GPU renderer: no active blocks to upload; using CPU path.\n"));
    return false;
  }

  const uint64_t voxelHash = fnv1a64(packed.voxels.data(), packed.voxels.size());
  const double packedMB =
      (packed.voxels.size() * sizeof(uint16_t)) / (1024.0 * 1024.0);

  TRCP(("GPU renderer stub: packed %zu blocks (%dx%dx%d vox each), bbox [%d,%d,%d]-[%d,%d,%d], %.2f MB buffer, checksum=0x%016llx\n",
        packed.blocks.size(), packed.blockSize, packed.blockSize, packed.blockSize,
        packed.bounds.minX, packed.bounds.minY, packed.bounds.minZ,
        packed.bounds.maxX, packed.bounds.maxY, packed.bounds.maxZ,
        packedMB, static_cast<unsigned long long>(voxelHash)));

  TRCP(("GPU renderer (Metal) kernels not implemented yet; falling back to CPU render.\n"));
  if (checksumOut) *checksumOut = voxelHash;
  return false;
}

bool applyMeanCurvaturePdeMetalUint16(MSBG::MultiresSparseGrid* msbg,
                                      int chan,
                                      const std::vector<int>* activeBlocks,
                                      int nIter,
                                      float dt,
                                      uint64_t* checksumOut) {
  const MetalAvailability availability = queryMetalAvailability();
  if (!availability.runtimePresent || !availability.devicePresent) {
    TRCP(("GPU PDE unavailable (%s); using CPU path.\n",
          availability.reason.c_str()));
    if (checksumOut) *checksumOut = 0;
    return false;
  }

  // Stub: pack blocks and log checksum, then fall back to CPU
  SBG::SparseGrid<uint16_t>* sg = msbg->getUint16Channel(chan, 0);
  if (!sg) {
    TRCP(("GPU PDE: missing uint16 channel; using CPU path.\n"));
    if (checksumOut) *checksumOut = 0;
    return false;
  }

  // Use the packing function from backend_compute.h detail namespace
  // For now, just log and fall back
  TRCP(("GPU PDE (Metal) kernels not implemented yet; falling back to CPU.\n"));
  if (checksumOut) *checksumOut = 0;
  return false;
}

}  // namespace Backend
