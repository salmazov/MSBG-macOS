/******************************************************************************
 * Placeholder for a future Metal renderer.
 * Returns false to indicate no GPU render performed.
 ******************************************************************************/
#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include "msbg.h"
#include "render.h"
#include "msbg_demo.h"

namespace Backend {

struct MetalAvailability {
  bool runtimePresent = false;
  bool devicePresent = false;
  std::string reason;
};

// Probe for Metal/MetalKit availability at runtime. Result is cached.
MetalAvailability queryMetalAvailability();
bool isMetalAvailable();

// Returns true if GPU rendering was performed, false otherwise.
bool renderSceneMetal(int testCase, MSBG::MultiresSparseGrid* msbg, int chan,
                      const std::vector<int>* activeBlocks,
                      PnlPanel* pnl, uint64_t* checksumOut = nullptr);

// Returns true if GPU PDE smoothing was performed, false otherwise.
// Currently a stub that falls back to CPU.
// This is called from backend_compute.h template wrapper.
bool applyMeanCurvaturePdeMetalUint16(MSBG::MultiresSparseGrid* msbg,
                                       int chan,
                                       const std::vector<int>* activeBlocks,
                                       int nIter,
                                       float dt,
                                       uint64_t* checksumOut = nullptr);

}  // namespace Backend
