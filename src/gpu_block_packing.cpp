/******************************************************************************
 * Sparse grid block packing shared helper.
 ******************************************************************************/

#include "gpu_block_packing.h"

#include <algorithm>
#include <cstddef>

#include "common_headers.h"

PackedBlockVolume packActiveBlocksForGpu(
    SBG::SparseGrid<uint16_t>* sg,
    const std::vector<int>* activeBlocks) {
  PackedBlockVolume packed;
  if (!sg || !sg->hasData()) {
    TRCP(("GPU renderer: sparse grid missing data; skipping GPU path.\n"));
    return packed;
  }
  if (sg->isDenseGrid()) {
    TRCP(("GPU renderer: dense grid layout not supported for GPU path; skipping.\n"));
    return packed;
  }
  if (sg->nBlocks() <= 0) {
    TRCP(("GPU renderer: no blocks available; skipping GPU path.\n"));
    return packed;
  }

  packed.blockSize = sg->bsx();
  packed.voxelsPerBlock = static_cast<int>(sg->nVoxelsInBlock());
  packed.nbx = sg->nbx();
  packed.nby = sg->nby();
  packed.nbz = sg->nbz();
  packed.emptyValue = sg->getEmptyValue();

  const LongInt nBlocks = sg->nBlocks();
  const bool useActiveList = activeBlocks && !activeBlocks->empty();
  const size_t reserveSize =
      useActiveList ? activeBlocks->size() : std::min<LongInt>(nBlocks, 1024);
  packed.blocks.reserve(reserveSize);
  packed.voxels.reserve(reserveSize * packed.voxelsPerBlock);

  auto maybePackBid = [&](int bid) {
    if (bid < 0 || bid >= nBlocks) {
      TRCP(("GPU renderer: skipping invalid block id %d (nBlocks=%lld)\n",
            bid, static_cast<long long>(nBlocks)));
      return;
    }
    if (!sg->isValueBlock(bid) || sg->isEmptyBlock(bid)) {
      return;
    }

    uint16_t* blockData = sg->getBlockDataPtr(bid);
    if (!blockData) {
      return;
    }

    const Vec4i bpos = sg->getBlockCoordsById(bid);
    PackedBlockVolume::BlockView view{
        vget_x(bpos), vget_y(bpos), vget_z(bpos), packed.voxels.size()};

    packed.blocks.push_back(view);
    packed.voxels.insert(packed.voxels.end(), blockData,
                         blockData + packed.voxelsPerBlock);

    packed.bounds.minX = std::min(packed.bounds.minX, view.bx);
    packed.bounds.minY = std::min(packed.bounds.minY, view.by);
    packed.bounds.minZ = std::min(packed.bounds.minZ, view.bz);

    packed.bounds.maxX = std::max(packed.bounds.maxX, view.bx);
    packed.bounds.maxY = std::max(packed.bounds.maxY, view.by);
    packed.bounds.maxZ = std::max(packed.bounds.maxZ, view.bz);
  };

  if (useActiveList) {
    for (int bid : *activeBlocks) {
      maybePackBid(bid);
    }
  } else {
    for (LongInt bidLong = 0; bidLong < nBlocks; ++bidLong) {
      maybePackBid(static_cast<int>(bidLong));
    }
  }

  if (packed.blocks.empty()) {
    packed.bounds = {};
  }

  TRCP(("GPU renderer: packActiveBlocks grid=%dx%dx%d nBlocks=%lld blockSize=%d (packed %zu blocks)\n",
        packed.nbx, packed.nby, packed.nbz,
        static_cast<long long>(nBlocks), packed.blockSize,
        packed.blocks.size()));

  return packed;
}

