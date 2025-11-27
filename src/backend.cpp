/******************************************************************************
 *
 * Backend dispatch for rendering (CPU now, GPU stubbed).
 *
 ******************************************************************************/

#include <memory>
#include <cstdint>

#include "msbg.h"
#include "render.h"
#include "msbg_demo.h"
#include "backend.h"
#include "metal_renderer_stub.h"

namespace Backend {

static uint64_t bitmapChecksum(const BmpBitmap* bmp) {
  if (!bmp || !bmp->data) {
    return 0;
  }

  constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
  constexpr uint64_t kFnvPrime = 1099511628211ULL;

  uint64_t hash = kFnvOffset;
  const size_t nPix = static_cast<size_t>(bmp->sx) *
                      static_cast<size_t>(bmp->sy);
  for (size_t i = 0; i < nPix; ++i) {
    hash ^= static_cast<uint32_t>(bmp->data[i]);
    hash *= kFnvPrime;
  }
  return hash;
}

static uint64_t render_scene_cpu(int testCase, MSBG::MultiresSparseGrid* msbg,
                             int chan, PnlPanel* pnl,
                             bool showPanel = true,
                             bool saveImage = true) {
  using namespace MSBG;
  using namespace SBG;

  SBG::SparseGrid<uint16_t>* sg = msbg->getUint16Channel(chan, 0);

  UtTimer tm;
  int sx = camRes[0], sy = camRes[1];

  TRCP(("Rendering image %dx%d (backend=cpu).\n", sx, sy));

  RDR::RaymarchRenderer renderer(sx, sy, sg);

#if 1

#if 1
  renderer.setCamera({camPos[0], camPos[1], camPos[2]},
                     {camLookAt[0], camLookAt[1], camLookAt[2]}, camZoom);

  renderer.setSunLight({camLight[0], camLight[1], camLight[2]});
  renderer.setSurfaceColor({0.8f, 0.6f, 0.4f});
#else

  renderer.setCamera({.5, .8, .8}, {.5, .5, .5}, 1.f);

  renderer.setSunLight({.5, 5, 5});
  renderer.setSurfaceColor({0.8f, 0.6f, 0.4f});
#endif

#else
  renderer.setCamera({.5, .5, .65}, {.5, .5, .5}, 0.8f);
  renderer.setSunLight({.5, 5, 5});
  renderer.setSurfaceColor({0.8f, 0.6f, 0.4f});
#endif

  TIMER_START(&tm);

  BmpBitmap* B = renderer.render();

  TIMER_STOP(&tm);
  TRCP(("CPU (rendering) %.2f sec, %.0f pixels/sec)\n",
        (double)TIMER_DIFF_MS(&tm) / 1000.,
        ((double)sx * sy) / (double)(TIMER_DIFF_MS(&tm) / 1000.0)));

  if (showPanel && pnl) {
    PnlShowBitmap2(pnl, B);
  }

  pnl->totalTime = 0;
  if (saveImage) {
    msbg->saveVisualizationBitmap(B, pnl->title);
    TRCP(("Output images saved to '%s/'.\n", VIS_OUTPUT_DIR));
  }

  const uint64_t checksum = bitmapChecksum(B);
  TRCP(("Render checksum (CPU backend): 0x%016llx\n",
        static_cast<unsigned long long>(checksum)));

  BmpDeleteBitmap(&B);
  return checksum;
}

void renderScene(int testCase, MSBG::MultiresSparseGrid* msbg, int chan,
                 const std::vector<int>* activeBlocks,
                 PnlPanel* pnl, BackendType backend,
                 bool validateAgainstCpu) {
  if (backend == BACKEND_GPU) {
    uint64_t gpuChecksum = 0;
    if (renderSceneMetal(testCase, msbg, chan, activeBlocks, pnl,
                         &gpuChecksum)) {
      if (validateAgainstCpu) {
        TRCP(("GPU renderer produced checksum 0x%016llx; validating against CPU...\n",
              static_cast<unsigned long long>(gpuChecksum)));
        const uint64_t cpuChecksum = render_scene_cpu(testCase, msbg, chan, pnl,
                                                      /*showPanel=*/false,
                                                      /*saveImage=*/false);
        if (cpuChecksum == gpuChecksum) {
          TRCP(("GPU vs CPU render checksum MATCH (0x%016llx).\n",
                static_cast<unsigned long long>(gpuChecksum)));
        } else {
          TRCP(("GPU vs CPU render checksum MISMATCH (GPU=0x%016llx CPU=0x%016llx).\n",
                static_cast<unsigned long long>(gpuChecksum),
                static_cast<unsigned long long>(cpuChecksum)));
        }
      }
      return;
    }
    if (validateAgainstCpu) {
      TRCP(("GPU validation requested but GPU render unavailable; falling back to CPU only.\n"));
    }
    // fall back if GPU path not available
  }
  render_scene_cpu(testCase, msbg, chan, pnl);
}

}  // namespace Backend
