/******************************************************************************
 *
 * Backend dispatch for rendering and future GPU kernels.
 *
 ******************************************************************************/
#pragma once

#include <vector>

#include "msbg_demo.h"

namespace Backend {

void renderScene(int testCase, MSBG::MultiresSparseGrid* msbg, int chan,
                 const std::vector<int>* activeBlocks,
                 PnlPanel* pnl, BackendType backend,
                 bool validateAgainstCpu = false);

}  // namespace Backend
