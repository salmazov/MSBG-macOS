/******************************************************************************
 * Shared helpers for packing sparse grid blocks for GPU upload paths.
 *
 * Both the Metal renderer and stub use this to marshal dense buffers while
 * keeping all SparseGrid access in regular C++ translation units.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include "msbg.h"

struct PackedBlockVolume {
  struct BlockView {
    int bx = 0;
    int by = 0;
    int bz = 0;
    size_t offset = 0;  // offset into voxels vector
  };

  struct BBox {
    int minX = std::numeric_limits<int>::max();
    int minY = std::numeric_limits<int>::max();
    int minZ = std::numeric_limits<int>::max();
    int maxX = std::numeric_limits<int>::min();
    int maxY = std::numeric_limits<int>::min();
    int maxZ = std::numeric_limits<int>::min();
  };

  int blockSize = 0;
  int voxelsPerBlock = 0;
  int nbx = 0;
  int nby = 0;
  int nbz = 0;
  uint16_t emptyValue = 0;
  BBox bounds;

  std::vector<BlockView> blocks;
  std::vector<uint16_t> voxels;
};

// Packs active SparseGrid blocks into a contiguous dense buffer.
// When activeBlocks is provided, only the listed blocks are scanned.
// Otherwise this walks all blocks (slower, for fallback callers).
PackedBlockVolume packActiveBlocksForGpu(
    SBG::SparseGrid<uint16_t>* sg,
    const std::vector<int>* activeBlocks = nullptr);

