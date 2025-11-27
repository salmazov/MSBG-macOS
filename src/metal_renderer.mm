/******************************************************************************
 * Metal renderer implementation (raymarching scalar volume).
 *
 * This translation unit is Objective-C++ and only built when METAL=1 is set
 * in the make invocation. The C++ code stays unchanged for the default build.
 ******************************************************************************/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <simd/simd.h>
#undef MIN
#undef MAX

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "common_headers.h"
#include "bitmap.h"
#include "metal_renderer_stub.h"
#include "gpu_block_packing.h"
#include "msbg.h"
#include "render.h"

static_assert(BMP_RGB != 0, "BMP_RGB constant should be non-zero");

namespace {

using Backend::MetalAvailability;

constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

static uint64_t bitmapChecksum(const BmpBitmap* bmp) {
  if (!bmp || !bmp->data) {
    return 0;
  }
  uint64_t hash = kFnvOffset;
  const size_t nPix = static_cast<size_t>(bmp->sx) *
                      static_cast<size_t>(bmp->sy);
  for (size_t i = 0; i < nPix; ++i) {
    hash ^= static_cast<uint32_t>(bmp->data[i]);
    hash *= kFnvPrime;
  }
  return hash;
}

uint64_t fnv1a64(const uint8_t* data, size_t count) {
  uint64_t hash = kFnvOffset;
  for (size_t i = 0; i < count; ++i) {
    hash ^= static_cast<uint64_t>(data[i]);
    hash *= kFnvPrime;
  }
  return hash;
}

struct DenseVolume {
  std::vector<uint16_t> voxels;
  int dimX = 0, dimY = 0, dimZ = 0;
  int blockSize = 0;
  Vec4i originBlocks{0};
};

struct MetalParams {
  simd_float3 camPos;
  simd_float3 camLookAt;
  simd_float3 lightPos;
  float focalLen;
  simd_float3 volumeOrigin;
  float resolution;
  simd_float3 volumeSize;
  float maxDistance;
  float isoLevel;
  float stepScale;
  simd_float3 surfaceColor;
  uint32_t width;
  uint32_t height;
};

DenseVolume buildDenseVolume(const PackedBlockVolume& packed) {
  DenseVolume dense;
  if (packed.blocks.empty()) {
    return dense;
  }

  dense.blockSize = packed.blockSize;
  dense.originBlocks = Vec4i(packed.bounds.minX, packed.bounds.minY,
                             packed.bounds.minZ, 0);

  const int spanX = packed.bounds.maxX - packed.bounds.minX + 1;
  const int spanY = packed.bounds.maxY - packed.bounds.minY + 1;
  const int spanZ = packed.bounds.maxZ - packed.bounds.minZ + 1;

  dense.dimX = spanX * packed.blockSize;
  dense.dimY = spanY * packed.blockSize;
  dense.dimZ = spanZ * packed.blockSize;

  const size_t totalVoxels =
      static_cast<size_t>(dense.dimX) * dense.dimY * dense.dimZ;
  dense.voxels.assign(totalVoxels, packed.emptyValue);

  const size_t planeStride = static_cast<size_t>(dense.dimX) * dense.dimY;

  for (const auto& block : packed.blocks) {
    const int ox = (block.bx - packed.bounds.minX) * packed.blockSize;
    const int oy = (block.by - packed.bounds.minY) * packed.blockSize;
    const int oz = (block.bz - packed.bounds.minZ) * packed.blockSize;
    const uint16_t* src = packed.voxels.data() + block.offset;

    for (int z = 0; z < packed.blockSize; ++z) {
      const size_t zOff = static_cast<size_t>(z + oz) * planeStride;
      for (int y = 0; y < packed.blockSize; ++y) {
        const size_t base = zOff + static_cast<size_t>(y + oy) * dense.dimX + ox;
        const uint16_t* srcRow =
            src + static_cast<size_t>(z) * packed.blockSize * packed.blockSize +
            static_cast<size_t>(y) * packed.blockSize;
        std::copy(srcRow, srcRow + packed.blockSize,
                  dense.voxels.begin() + base);
      }
    }
  }

  return dense;
}

id<MTLTexture> createVolumeTexture(id<MTLDevice> device,
                                   const DenseVolume& dense) {
  if (!device || dense.voxels.empty() || dense.dimX <= 0 || dense.dimY <= 0 ||
      dense.dimZ <= 0) {
    return nil;
  }

  MTLTextureDescriptor* desc = [MTLTextureDescriptor new];
  desc.textureType = MTLTextureType3D;
  desc.pixelFormat = MTLPixelFormatR32Float;
  desc.width = dense.dimX;
  desc.height = dense.dimY;
  desc.depth = dense.dimZ;
  desc.mipmapLevelCount = 1;
  desc.usage = MTLTextureUsageShaderRead;
  desc.storageMode = MTLStorageModeShared;

  id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
  if (!texture) {
    return nil;
  }

  constexpr float kInvMaxUint16 = 1.0f / 65535.0f;
  std::vector<float> voxels(dense.voxels.size());
  for (size_t i = 0; i < dense.voxels.size(); ++i) {
    voxels[i] = static_cast<float>(dense.voxels[i]) * kInvMaxUint16;
  }

  const MTLRegion region =
      MTLRegionMake3D(0, 0, 0, dense.dimX, dense.dimY, dense.dimZ);
  const NSUInteger bytesPerRow =
      static_cast<NSUInteger>(dense.dimX) * sizeof(float);
  const NSUInteger bytesPerImage =
      static_cast<NSUInteger>(dense.dimY) * bytesPerRow;
  [texture replaceRegion:region
            mipmapLevel:0
                 slice:0
             withBytes:voxels.data()
           bytesPerRow:bytesPerRow
         bytesPerImage:bytesPerImage];

  return texture;
}

id<MTLTexture> createOutputTexture(id<MTLDevice> device, int width,
                                   int height) {
  if (!device || width <= 0 || height <= 0) {
    return nil;
  }

  MTLTextureDescriptor* desc = [MTLTextureDescriptor new];
  desc.textureType = MTLTextureType2D;
  desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
  desc.width = static_cast<NSUInteger>(width);
  desc.height = static_cast<NSUInteger>(height);
  desc.mipmapLevelCount = 1;
  desc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
  desc.storageMode = MTLStorageModeShared;
  return [device newTextureWithDescriptor:desc];
}

MetalParams makeMetalParams(const DenseVolume& dense,
                            SBG::SparseGrid<uint16_t>* sg, int width,
                            int height) {
  MetalParams params{};
  params.camPos = {camPos[0], camPos[1], camPos[2]};
  params.camLookAt = {camLookAt[0], camLookAt[1], camLookAt[2]};
  params.lightPos = {camLight[0], camLight[1], camLight[2]};
  params.focalLen = camZoom;

  const int originBlockX = vget_x(dense.originBlocks);
  const int originBlockY = vget_y(dense.originBlocks);
  const int originBlockZ = vget_z(dense.originBlocks);
  params.volumeOrigin = {
      static_cast<float>(originBlockX * dense.blockSize),
      static_cast<float>(originBlockY * dense.blockSize),
      static_cast<float>(originBlockZ * dense.blockSize),
  };
  params.volumeSize = {static_cast<float>(dense.dimX),
                       static_cast<float>(dense.dimY),
                       static_cast<float>(dense.dimZ)};

  const float resolution = static_cast<float>(sg->sxyzMax());
  params.resolution = resolution;
  params.maxDistance = 2.0f;
  params.isoLevel = 0.5f;
  params.stepScale = (resolution > 0.0f) ? (1.0f / resolution) : 0.001f;
  params.surfaceColor = {0.8f, 0.6f, 0.4f};
  params.width = static_cast<uint32_t>(width);
  params.height = static_cast<uint32_t>(height);
  return params;
}

BmpBitmap* bitmapFromTexture(id<MTLTexture> texture, int width, int height) {
  if (!texture || width <= 0 || height <= 0) {
    return nullptr;
  }

  const NSUInteger bytesPerPixel = 4;
  const NSUInteger bytesPerRow =
      static_cast<NSUInteger>(width) * bytesPerPixel;
  std::vector<uint8_t> pixels(static_cast<size_t>(bytesPerRow) * height, 0);
  const MTLRegion region = MTLRegionMake2D(0, 0, width, height);
  [texture getBytes:pixels.data()
       bytesPerRow:bytesPerRow
        fromRegion:region
       mipmapLevel:0];
  size_t nonZeroBytes = 0;
  for (uint8_t byte : pixels) {
    if (byte != 0) {
      ++nonZeroBytes;
    }
  }
  TRCP(("GPU renderer: bitmap readback bytes=%zu nonZero=%zu\n",
        pixels.size(), nonZeroBytes));

  BmpBitmap* bmpOut = BmpNewBitmap(width, height, BMP_RGB);
  if (!bmpOut) {
    TRCP(("GPU renderer: BmpNewBitmap failed (%dx%d).\n", width, height));
    return nullptr;
  }
  int* rgbData = BmpGetRGBChannel(bmpOut, BMP_CLEAR);
  if (!rgbData) {
    TRCP(("GPU renderer: bitmap missing RGB channel after allocation (opt=0x%x).\n",
          bmpOut->optOrig));
    BmpDeleteBitmap(&bmpOut);
    return nullptr;
  }

  for (int y = 0; y < height; ++y) {
    const uint8_t* row = pixels.data() + static_cast<size_t>(y) * bytesPerRow;
    for (int x = 0; x < width; ++x) {
      const uint8_t* px = row + static_cast<size_t>(x) * bytesPerPixel;
      rgbData[static_cast<size_t>(y) * bmpOut->sx + x] =
          BMP_MKRGB(px[0], px[1], px[2]);
    }
  }
  if (!pixels.empty()) {
    const uint64_t rawHash = fnv1a64(pixels.data(), pixels.size());
    TRCP(("GPU renderer: first pixel after copy=0x%08x rawByteHash=0x%016llx\n",
          rgbData[0], static_cast<unsigned long long>(rawHash)));
  }

  return bmpOut;
}

constexpr const char* kMetalShaderSrc = R"(
#pragma language(metal4.0)
#include <metal_stdlib>
using namespace metal;

struct Params {
    float3 camPos;
    float3 camLookAt;
    float3 lightPos;
    float focalLen;
    float3 volumeOrigin;
    float resolution;
    float3 volumeSize;
    float maxDistance;
    float isoLevel;
    float stepScale;
    float3 surfaceColor;
    uint  width;
    uint  height;
};

struct RaymarchArgs {
    texture3d<float, access::sample> volume [[id(0)]];
    texture2d<float, access::write> outTex [[id(1)]];
    constant Params& params [[id(2)]];
};

inline float sampleDensity(texture3d<float, access::sample> volume,
                           sampler s,
                           float3 posGrid,
                           constant Params& params) {
    float3 rel = (posGrid - params.volumeOrigin) / params.volumeSize;
    if (any(rel < 0.0) || any(rel > 1.0)) return 0.0;
    return volume.sample(s, rel).r;
}

kernel void raymarch(constant RaymarchArgs& args [[buffer(0)]],
                     uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.params.width || gid.y >= args.params.height) return;

#if 0  // Toggle to verify render target wiring.
    const float r = float(gid.x) / float(args.params.width);
    const float g = float(gid.y) / float(args.params.height);
    args.outTex.write(float4(r, g, 0.0f, 1.0f), gid);
    return;
#endif

    const float aspect = float(args.params.width) / float(args.params.height);
    const float2 ndc = float2(((float(gid.x) + 0.5f) / float(args.params.width) - 0.5f) * 2.0f * aspect,
                              (0.5f - (float(gid.y) + 0.5f) / float(args.params.height)) * 2.0f);

    float3 forward = normalize(args.params.camLookAt - args.params.camPos);
    float3 right = normalize(cross(forward, float3(0.0f, 1.0f, 0.0f)));
    float3 up = cross(right, forward);
    float3 rayDir = normalize(forward * args.params.focalLen + right * ndc.x + up * ndc.y);

    constexpr sampler volumeSampler(address::clamp_to_edge, filter::linear);

    float stepSize = args.params.stepScale;
    float maxDist = args.params.maxDistance;

    float traveled = 0.0f;
    float prev = sampleDensity(args.volume, volumeSampler, args.params.camPos * args.params.resolution, args.params);
    float3 prevPos = args.params.camPos;
    bool hit = false;
    float3 hitPos = float3(0.0f);

    const int maxSteps = int(maxDist / max(stepSize, 1e-5));
    for (int i = 0; i < maxSteps; ++i) {
        traveled += stepSize;
        float3 pos = args.params.camPos + rayDir * traveled;
        float d = sampleDensity(args.volume, volumeSampler, pos * args.params.resolution, args.params);
        if (prev < args.params.isoLevel && d >= args.params.isoLevel) {
            float denom = max(d - prev, 1e-5f);
            float t = (args.params.isoLevel - prev) / denom;
            hitPos = mix(prevPos, pos, t);
            hit = true;
            break;
        }
        prev = d;
        prevPos = pos;
    }

    float4 out = float4(0.0f, 0.0f, 0.0f, 1.0f);
    if (hit) {
        float eps = stepSize * args.params.resolution;
        float3 g;
        float3 offs = float3(eps, 0, 0);
        g.x = sampleDensity(args.volume, volumeSampler, (hitPos + offs) * args.params.resolution, args.params) -
              sampleDensity(args.volume, volumeSampler, (hitPos - offs) * args.params.resolution, args.params);
        offs = float3(0, eps, 0);
        g.y = sampleDensity(args.volume, volumeSampler, (hitPos + offs) * args.params.resolution, args.params) -
              sampleDensity(args.volume, volumeSampler, (hitPos - offs) * args.params.resolution, args.params);
        offs = float3(0, 0, eps);
        g.z = sampleDensity(args.volume, volumeSampler, (hitPos + offs) * args.params.resolution, args.params) -
              sampleDensity(args.volume, volumeSampler, (hitPos - offs) * args.params.resolution, args.params);

        float3 normal = normalize(g);
        float3 lightDir = normalize(args.params.lightPos - hitPos);
        float diff = max(dot(normal, lightDir), 0.0f);

        float3 c = clamp(args.params.surfaceColor * diff, 0.0f, 1.0f);
        out = float4(c, 1.0f);
    }
    args.outTex.write(out, gid);
}
)";

}  // namespace

namespace Backend {

MetalAvailability queryMetalAvailability() {
  MetalAvailability availability;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      availability.reason = "MTLCreateSystemDefaultDevice returned nil";
      return availability;
    }
    availability.runtimePresent = true;
    availability.devicePresent = true;
    availability.reason = "Metal device detected";
  }
  return availability;
}

bool isMetalAvailable() {
  const MetalAvailability availability = queryMetalAvailability();
  return availability.runtimePresent && availability.devicePresent;
}

bool renderSceneMetal(int testCase, MSBG::MultiresSparseGrid* msbg,
                      int chan, const std::vector<int>* activeBlocks,
                      PnlPanel* pnl, uint64_t* checksumOut) {
  (void)testCase;
  @autoreleasepool {
    const MetalAvailability availability = queryMetalAvailability();
    if (!availability.runtimePresent || !availability.devicePresent) {
      TRCP(("GPU renderer unavailable (%s); using CPU path.\n",
            availability.reason.c_str()));
      if (checksumOut) {
        *checksumOut = 0;
      }
      return false;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      TRCP(("GPU renderer: failed to create Metal device.\n"));
      if (checksumOut) {
        *checksumOut = 0;
      }
      return false;
    }

    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    if (!commandQueue) {
      TRCP(("GPU renderer: failed to create command queue.\n"));
      if (checksumOut) {
        *checksumOut = 0;
      }
      return false;
    }

    if (!msbg) {
      TRCP(("GPU renderer: null msbg pointer; using CPU path.\n"));
      if (checksumOut) {
        *checksumOut = 0;
      }
      return false;
    }

    msbg->prepareDataAccess(chan);
    SBG::SparseGrid<uint16_t>* sg = msbg->getUint16Channel(chan, 0, 0);
    if (!sg) {
      TRCP(("GPU renderer: missing uint16 density channel; using CPU path.\n"));
      if (checksumOut) {
        *checksumOut = 0;
      }
      return false;
    }

    // Ensure sparse grid is prepared for data access
    sg->prepareDataAccess();
    
    if (!sg->hasData()) {
      TRCP(("GPU renderer: sparse grid has no data; using CPU path.\n"));
      if (checksumOut) {
        *checksumOut = 0;
      }
      return false;
    }

    const size_t numActiveBlocks = activeBlocks ? activeBlocks->size() : 0;
    TRCP(("GPU renderer: active block list ptr=%p count=%zu\n",
          static_cast<const void*>(activeBlocks), numActiveBlocks));
    PackedBlockVolume packed = packActiveBlocksForGpu(sg, activeBlocks);
    if (packed.blocks.empty() || packed.voxels.empty() || packed.blockSize <= 0) {
      TRCP(("GPU renderer: no active blocks available for Metal upload; using CPU path.\n"));
      if (checksumOut) {
        *checksumOut = 0;
      }
      return false;
    }

    DenseVolume dense = buildDenseVolume(packed);
    if (dense.voxels.empty() || dense.dimX <= 0 || dense.dimY <= 0 ||
        dense.dimZ <= 0) {
      TRCP(("GPU renderer: failed to build dense voxel buffer; using CPU path.\n"));
      if (checksumOut) {
        *checksumOut = 0;
      }
      return false;
    }

    const uint64_t voxelHash = fnv1a64(
        reinterpret_cast<const uint8_t*>(dense.voxels.data()),
        dense.voxels.size() * sizeof(uint16_t));
    const double packedMB =
        (dense.voxels.size() * sizeof(uint16_t)) / (1024.0 * 1024.0);

    const auto [minIt, maxIt] =
        std::minmax_element(dense.voxels.begin(), dense.voxels.end());
    const float minNorm =
        dense.voxels.empty()
            ? 0.0f
            : static_cast<float>(*minIt) * (1.0f / 65535.0f);
    const float maxNorm =
        dense.voxels.empty()
            ? 0.0f
            : static_cast<float>(*maxIt) * (1.0f / 65535.0f);

    TRCP(("GPU renderer: packed %zu blocks (%d^3 vox each), dense volume %dx%dx%d, %.2f MB upload, hash=0x%016llx, density range=[%g,%g]\n",
          packed.blocks.size(), packed.blockSize, dense.dimX, dense.dimY,
          dense.dimZ, packedMB, static_cast<unsigned long long>(voxelHash),
          minNorm, maxNorm));

    NSError* error = nil;
    NSString* shaderSource = [NSString stringWithUTF8String:kMetalShaderSrc];
    if (!shaderSource) {
      TRCP(("GPU renderer: failed to create Metal shader string.\n"));
      if (checksumOut) {
        *checksumOut = 0;
      }
      return false;
    }
    id<MTLLibrary> library =
        [device newLibraryWithSource:shaderSource options:nil error:&error];
    if (!library) {
      const char* errMsg = error ? [[error localizedDescription] UTF8String]
                                 : "unknown";
      TRCP(("GPU renderer: failed to compile Metal shader (%s).\n",
            errMsg ? errMsg : "unknown"));
      if (checksumOut) {
        *checksumOut = 0;
      }
      return false;
    }
    id<MTLFunction> function = [library newFunctionWithName:@"raymarch"];
    if (!function) {
      TRCP(("GPU renderer: Metal function 'raymarch' missing.\n"));
      if (checksumOut) {
        *checksumOut = 0;
      }
      return false;
    }

    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
      const char* errMsg = error ? [[error localizedDescription] UTF8String]
                                 : "unknown";
      TRCP(("GPU renderer: failed to create compute pipeline (%s).\n",
            errMsg ? errMsg : "unknown"));
      if (checksumOut) {
        *checksumOut = 0;
      }
      return false;
    }

    id<MTLArgumentEncoder> raymarchArgsEncoder =
        [function newArgumentEncoderWithBufferIndex:0];
    if (!raymarchArgsEncoder) {
      TRCP(("GPU renderer: failed to create argument encoder for raymarch kernel.\n"));
      if (checksumOut) {
        *checksumOut = 0;
      }
      return false;
    }

    const int sx = camRes[0];
    const int sy = camRes[1];
    id<MTLTexture> volumeTexture = createVolumeTexture(device, dense);
    if (!volumeTexture) {
      TRCP(("GPU renderer: failed to create/upload volume texture.\n"));
      if (checksumOut) {
        *checksumOut = 0;
      }
      return false;
    }
    id<MTLTexture> outputTexture = createOutputTexture(device, sx, sy);
    if (!outputTexture) {
      TRCP(("GPU renderer: failed to create output texture (%dx%d).\n", sx,
            sy));
      if (checksumOut) {
        *checksumOut = 0;
      }
      return false;
    }

    const NSUInteger raymarchArgLength = raymarchArgsEncoder.encodedLength;
    id<MTLBuffer> raymarchArgBuffer =
        [device newBufferWithLength:raymarchArgLength
                             options:MTLResourceStorageModeShared];
    if (!raymarchArgBuffer) {
      TRCP(("GPU renderer: failed to allocate argument buffer for raymarch kernel.\n"));
      if (checksumOut) {
        *checksumOut = 0;
      }
      return false;
    }
    [raymarchArgBuffer setLabel:@"raymarch_args"];
    [raymarchArgsEncoder setArgumentBuffer:raymarchArgBuffer offset:0];
    [raymarchArgsEncoder setTexture:volumeTexture atIndex:0];
    [raymarchArgsEncoder setTexture:outputTexture atIndex:1];
    MetalParams params = makeMetalParams(dense, sg, sx, sy);
    id<MTLBuffer> paramsBuffer =
        [device newBufferWithBytes:&params
                             length:sizeof(params)
                            options:MTLResourceStorageModeShared];
    if (!paramsBuffer) {
      TRCP(("GPU renderer: failed to allocate parameter buffer.\n"));
      if (checksumOut) {
        *checksumOut = 0;
      }
      return false;
    }
    [raymarchArgsEncoder setBuffer:paramsBuffer offset:0 atIndex:2];
    auto sampleDensePoint = [&](const simd_float3& worldPos) -> float {
      const simd_float3 posGrid = worldPos * params.resolution;
      const simd_float3 rel =
          (posGrid - params.volumeOrigin) / params.volumeSize;
      if (rel.x < 0.0f || rel.y < 0.0f || rel.z < 0.0f || rel.x > 1.0f ||
          rel.y > 1.0f || rel.z > 1.0f) {
        return 0.0f;
      }
      const float gx = rel.x * static_cast<float>(dense.dimX - 1);
      const float gy = rel.y * static_cast<float>(dense.dimY - 1);
      const float gz = rel.z * static_cast<float>(dense.dimZ - 1);
      const int ix = std::max(0, std::min(dense.dimX - 1,
                                          static_cast<int>(std::round(gx))));
      const int iy = std::max(0, std::min(dense.dimY - 1,
                                          static_cast<int>(std::round(gy))));
      const int iz = std::max(0, std::min(dense.dimZ - 1,
                                          static_cast<int>(std::round(gz))));
      const size_t idx = static_cast<size_t>(iz) * dense.dimX * dense.dimY +
                         static_cast<size_t>(iy) * dense.dimX + ix;
      return static_cast<float>(dense.voxels[idx]) * (1.0f / 65535.0f);
    };
    auto debugRaymarch = [&]() -> std::pair<bool, float> {
      const float aspect = static_cast<float>(params.width) /
                           static_cast<float>(params.height);
      const float ndcX = 0.0f;
      const float ndcY = 0.0f;
      simd_float3 forward =
          simd_normalize(params.camLookAt - params.camPos);
      simd_float3 right = simd_normalize(
          simd_cross(forward, simd_make_float3(0.0f, 1.0f, 0.0f)));
      simd_float3 up = simd_cross(right, forward);
      simd_float3 rayDir =
          simd_normalize(forward * params.focalLen + right * ndcX +
                         up * ndcY);
      const float stepSize = params.stepScale;
      const float maxDist = params.maxDistance;
      float traveled = 0.0f;
      float prev = sampleDensePoint(params.camPos);
      simd_float3 prevPos = params.camPos;
      bool hit = false;
      float maxDensity = prev;
      const int maxSteps = static_cast<int>(
          maxDist / std::max(stepSize, 1e-5f));
      for (int i = 0; i < maxSteps; ++i) {
        traveled += stepSize;
        const simd_float3 pos = params.camPos + rayDir * traveled;
        const float d = sampleDensePoint(pos);
        maxDensity = std::max(maxDensity, d);
        if (prev < params.isoLevel && d >= params.isoLevel) {
          hit = true;
          break;
        }
        prev = d;
        prevPos = pos;
      }
      return {hit, maxDensity};
    };
    const auto [debugHit, debugMaxDensity] = debugRaymarch();
    TRCP(("GPU renderer debug raymarch: hit=%d maxDensity=%g\n",
          debugHit ? 1 : 0, debugMaxDensity));
    id<MTLResidencySet> rendererResidency = nil;
    if (@available(macOS 15.0, *)) {
      MTLResidencySetDescriptor* desc = [MTLResidencySetDescriptor new];
      desc.label = @"msbg_renderer_residency";
      desc.initialCapacity = 4;
      NSError* residencyError = nil;
      rendererResidency = [device newResidencySetWithDescriptor:desc
                                                         error:&residencyError];
      if (!rendererResidency && residencyError) {
        const char* errMsg = [[residencyError localizedDescription] UTF8String];
        TRCP(("GPU renderer: failed to allocate residency set (%s).\n",
              errMsg ? errMsg : "unknown"));
      }
      if (rendererResidency) {
        NSArray<id<MTLAllocation>>* allocations = @[
          volumeTexture, outputTexture, paramsBuffer, raymarchArgBuffer
        ];
        for (id<MTLAllocation> allocation in allocations) {
          if (allocation) {
            [rendererResidency addAllocation:allocation];
          }
        }
        [rendererResidency commit];
        [rendererResidency requestResidency];
      }
    }

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) {
      TRCP(("GPU renderer: failed to create compute encoder.\n"));
      if (checksumOut) {
        *checksumOut = 0;
      }
      return false;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:raymarchArgBuffer offset:0 atIndex:0];
    if (@available(macOS 11.0, *)) {
      [encoder useResource:volumeTexture usage:MTLResourceUsageRead];
      [encoder useResource:outputTexture usage:MTLResourceUsageWrite];
      [encoder useResource:paramsBuffer usage:MTLResourceUsageRead];
    }

    const NSUInteger tgWidth = 8;
    const NSUInteger tgHeight = 8;
    const MTLSize threadsPerGroup = MTLSizeMake(tgWidth, tgHeight, 1);
    const MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(sx),
                                         static_cast<NSUInteger>(sy), 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];
    
    // Metal 4: Add memory barrier for proper synchronization semantics
    // Note: Using MTLBarrierScopeBuffers for buffer synchronization
    if (@available(macOS 13.0, *)) {
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    [encoder endEncoding];

    if (rendererResidency) {
      if (@available(macOS 15.0, *)) {
        [commandBuffer useResidencySet:rendererResidency];
      }
    }

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) {
      NSError* cmdError = commandBuffer.error;
      const char* errMsg =
          cmdError ? [[cmdError localizedDescription] UTF8String] : "unknown";
      TRCP(("GPU renderer: Metal command buffer failed (%s).\n",
            errMsg ? errMsg : "unknown"));
      if (checksumOut) {
        *checksumOut = 0;
      }
      return false;
    }
    if (rendererResidency) {
      if (@available(macOS 15.0, *)) {
        [rendererResidency endResidency];
      }
    }

    BmpBitmap* bmpOut = bitmapFromTexture(outputTexture, sx, sy);
    if (!bmpOut) {
      TRCP(("GPU renderer: failed to read back GPU image.\n"));
      if (checksumOut) {
        *checksumOut = 0;
      }
      return false;
    }

    const uint64_t checksum = bitmapChecksum(bmpOut);
    const uint64_t debugBitmapHash =
        fnv1a64(reinterpret_cast<const uint8_t*>(bmpOut->data),
                static_cast<size_t>(bmpOut->sx) * bmpOut->sy * sizeof(int));
    if (checksumOut) {
      *checksumOut = checksum;
    }
    TRCP(("GPU renderer checksum (Metal): 0x%016llx (rawBitmap=0x%016llx)\n",
          static_cast<unsigned long long>(checksum),
          static_cast<unsigned long long>(debugBitmapHash)));

    const bool panelHasTitle = pnl && pnl->title[0] != '\0';
    std::string panelTitle = panelHasTitle ? pnl->title : "metal_gpu";
    if (pnl) {
      PnlShowBitmap2(pnl, bmpOut);
      pnl->totalTime = 0;
    }
    char* titlePtr = panelTitle.empty() ? const_cast<char*>("metal_gpu")
                                        : panelTitle.data();
    msbg->saveVisualizationBitmap(bmpOut, titlePtr);
    BmpDeleteBitmap(&bmpOut);
    return true;
  }

  return false;
}

// Metal compute shader for mean curvature PDE smoothing
constexpr const char* kMetalPdeShaderSrc = R"(
#pragma language(metal4.0)
#include <metal_stdlib>
using namespace metal;

struct PdeParams {
    uint blockSize;      // Block size (e.g., 16)
    uint haloSize;       // Halo size (1 for mean curvature)
    uint blockStride;   // Stride for block indexing
    float dt;            // Time step
    float lambda;        // Constraint weight (0 for now)
    float densThr;      // Density threshold
};

struct PdeResources {
    device float* blocksIn [[id(0)]];
    device float* blocksOut [[id(1)]];
    constant PdeParams& params [[id(2)]];
};

// Mean curvature computation kernel
// Processes one block at a time with halo region
kernel void meanCurvaturePde(
    constant PdeResources& res [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint bsx = res.params.blockSize;
    const uint hsx = bsx + 2 * res.params.haloSize;  // Halo block size
    const uint hsx2 = hsx * hsx;
    
    // Check bounds
    if (gid.x >= bsx || gid.y >= bsx || gid.z >= bsx) return;
    
    // Offset into the halo block (account for halo padding)
    const uint x = gid.x + res.params.haloSize;
    const uint y = gid.y + res.params.haloSize;
    const uint z = gid.z + res.params.haloSize;
    
    // Index into halo block
    const uint idx = x + y * hsx + z * hsx2;
    
    // Read center and neighbors
    float f0 = res.blocksIn[idx];
    float f1 = res.blocksIn[(x-1) + y * hsx + z * hsx2];
    float f2 = res.blocksIn[(x+1) + y * hsx + z * hsx2];
    float f3 = res.blocksIn[x + (y-1) * hsx + z * hsx2];
    float f4 = res.blocksIn[x + (y+1) * hsx + z * hsx2];
    float f5 = res.blocksIn[x + y * hsx + (z-1) * hsx2];
    float f6 = res.blocksIn[x + y * hsx + (z+1) * hsx2];
    
    // Gradients (central differences)
    float fx = 0.5f * (f2 - f1);
    float fy = 0.5f * (f4 - f3);
    float fz = 0.5f * (f6 - f5);
    
    // Second derivatives
    float fxx = f1 + f2 - 2.0f * f0;
    float fyy = f3 + f4 - 2.0f * f0;
    float fzz = f5 + f6 - 2.0f * f0;
    
    // Mixed derivatives (needs additional neighbors)
    float fy_right = 0.5f * (res.blocksIn[(x+1) + (y+1) * hsx + z * hsx2] - 
                             res.blocksIn[(x+1) + (y-1) * hsx + z * hsx2]);
    float fy_left = 0.5f * (res.blocksIn[(x-1) + (y+1) * hsx + z * hsx2] - 
                            res.blocksIn[(x-1) + (y-1) * hsx + z * hsx2]);
    float fz_right = 0.5f * (res.blocksIn[(x+1) + y * hsx + (z+1) * hsx2] - 
                             res.blocksIn[(x+1) + y * hsx + (z-1) * hsx2]);
    float fz_left = 0.5f * (res.blocksIn[(x-1) + y * hsx + (z+1) * hsx2] - 
                            res.blocksIn[(x-1) + y * hsx + (z-1) * hsx2]);
    float fz_up = 0.5f * (res.blocksIn[x + (y+1) * hsx + (z+1) * hsx2] - 
                          res.blocksIn[x + (y+1) * hsx + (z-1) * hsx2]);
    float fz_down = 0.5f * (res.blocksIn[x + (y-1) * hsx + (z+1) * hsx2] - 
                            res.blocksIn[x + (y-1) * hsx + (z-1) * hsx2]);
    
    float fyx = 0.5f * (fy_right - fy_left);
    float fzx = 0.5f * (fz_right - fz_left);
    float fzy = 0.5f * (fz_up - fz_down);
    
    // Mean curvature H
    float gradMagSq = fx * fx + fy * fy + fz * fz;
    float H = 0.0f;
    
    if (gradMagSq > 1e-7f) {
        H = ((fy * fy + fz * fz) * fxx +
             (fx * fx + fz * fz) * fyy +
             (fx * fx + fy * fy) * fzz -
             2.0f * (fx * fy * fyx + fx * fz * fzx + fy * fz * fzy)) / gradMagSq;
    }
    
    // Update with time step
    float D = res.params.dt * H;
    D = clamp(D, -0.1f, 0.1f);  // Limiter for stability
    
    // Write output (only to interior, not halo)
    const uint outIdx = gid.x + gid.y * bsx + gid.z * bsx * bsx;
    res.blocksOut[outIdx] = f0 + D;
}
)";

bool applyMeanCurvaturePdeMetalUint16(MSBG::MultiresSparseGrid* msbg,
                                      int chan,
                                      const std::vector<int>* activeBlocks,
                                      int nIter,
                                      float dt,
                                      uint64_t* checksumOut) {
  @autoreleasepool {
    const MetalAvailability availability = queryMetalAvailability();
    if (!availability.runtimePresent || !availability.devicePresent) {
      TRCP(("GPU PDE unavailable (%s); using CPU path.\n",
            availability.reason.c_str()));
      if (checksumOut) *checksumOut = 0;
      return false;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      TRCP(("GPU PDE: failed to create Metal device.\n"));
      if (checksumOut) *checksumOut = 0;
      return false;
    }

    if (!msbg) {
      TRCP(("GPU PDE: null msbg pointer; using CPU path.\n"));
      if (checksumOut) *checksumOut = 0;
      return false;
    }

    msbg->prepareDataAccess(chan);
    SBG::SparseGrid<uint16_t>* sg = msbg->getUint16Channel(chan, 0, 0);
    if (!sg) {
      TRCP(("GPU PDE: missing uint16 channel; using CPU path.\n"));
      if (checksumOut) *checksumOut = 0;
      return false;
    }

    // Ensure sparse grid is prepared for data access
    sg->prepareDataAccess();
    
    if (!sg->hasData()) {
      TRCP(("GPU PDE: sparse grid has no data; using CPU path.\n"));
      if (checksumOut) *checksumOut = 0;
      return false;
    }

    if (!activeBlocks || activeBlocks->empty()) {
      TRCP(("GPU PDE: no active blocks; using CPU path.\n"));
      if (checksumOut) *checksumOut = 0;
      return false;
    }

    const int blockSize = sg->bsx();
    const int haloSize = 1;  // Mean curvature needs 1-voxel halo
    const int haloBlockSize = blockSize + 2 * haloSize;
    const int voxelsPerBlock = blockSize * blockSize * blockSize;
    const int voxelsPerHaloBlock = haloBlockSize * haloBlockSize * haloBlockSize;

    TRCP(("GPU PDE: preparing %zu blocks (bs=%d, halo=%d, halo_bs=%d)\n",
          activeBlocks->size(), blockSize, haloSize, haloBlockSize));

    // Pack blocks with halo regions
    std::vector<float> haloBlocksFloat;
    std::vector<int> validBlockIndices;
    haloBlocksFloat.reserve(activeBlocks->size() * voxelsPerHaloBlock);
    validBlockIndices.reserve(activeBlocks->size());

    const uint16_t emptyValue = sg->getEmptyValue();
    const float emptyValueFloat = static_cast<float>(emptyValue) / 65535.0f;
    const int bsxLog2 = sg->bsxLog2();
    const int bsxMask = blockSize - 1;
    const int nbx = sg->nbx();
    const int nbxy = sg->nbxy();
    const int sx = sg->sx();
    const int sy = sg->sy();
    const int sz = sg->sz();

    for (int bid : *activeBlocks) {
      // Strict bounds check
      if (bid < 0 || bid >= sg->nBlocks()) {
        TRCP(("GPU PDE: invalid block id %d (nBlocks=%lld)\n", 
              bid, static_cast<long long>(sg->nBlocks())));
        continue;
      }

      if (!sg->isValueBlock(bid) || sg->isEmptyBlock(bid)) {
        continue;
      }

      uint16_t* blockData = sg->getBlockDataPtr(bid);
      if (!blockData) {
        continue;
      }

      // Get block coordinates (bid is validated above)
      Vec4i bpos = sg->getBlockCoordsById(bid);
      int bx0 = vget_x(bpos);
      int by0 = vget_y(bpos);
      int bz0 = vget_z(bpos);

      Vec4i pos0 = Vec4i(bx0, by0, bz0, 0) * blockSize;
      Vec4i pos1 = pos0 - haloSize;
      Vec4i pos2 = pos0 + blockSize + haloSize - 1;

      int x1 = vget_x(pos1), x2 = vget_x(pos2);
      int y1 = vget_y(pos1), y2 = vget_y(pos2);
      int z1 = vget_z(pos1), z2 = vget_z(pos2);

      // Allocate halo block
      std::vector<float> haloBlock(voxelsPerHaloBlock, emptyValueFloat);
      size_t haloOffset = haloBlocksFloat.size();
      haloBlocksFloat.resize(haloOffset + voxelsPerHaloBlock, emptyValueFloat);

      // Fill halo block by iterating over the extended region
      for (int z = z1; z <= z2; ++z) {
        bool inRangeZ = z >= 0 && z < sz;
        int ibz = inRangeZ ? (z >> bsxLog2) * nbxy : -1;
        int ivz = inRangeZ ? (z & bsxMask) << (2 * bsxLog2) : 0;
        int ihz = (z - z1) * haloBlockSize * haloBlockSize;

        for (int y = y1; y <= y2; ++y) {
          bool inRangeZY = inRangeZ && y >= 0 && y < sy;
          int ibzy = inRangeZY ? ibz + (y >> bsxLog2) * nbx : -1;
          int ivzy = inRangeZY ? ivz + ((y & bsxMask) << bsxLog2) : 0;
          int ihzy = ihz + (y - y1) * haloBlockSize;

          for (int x = x1; x <= x2; ++x) {
            int ih = ihzy + (x - x1);
            float val = emptyValueFloat;

            if (inRangeZY && x >= 0 && x < sx) {
              int ib = ibzy + (x >> bsxLog2);
              int iv = ivzy + (x & bsxMask);

              if (ib >= 0 && ib < sg->nBlocks() && 
                  iv >= 0 && iv < voxelsPerBlock &&
                  sg->isValueBlock(ib)) {
                uint16_t* neighborData = sg->getBlockDataPtr(ib);
                if (neighborData) {
                  val = static_cast<float>(neighborData[iv]) / 65535.0f;
                }
              }
            }

            haloBlocksFloat[haloOffset + ih] = val;
          }
        }
      }

      validBlockIndices.push_back(bid);
    }

    if (validBlockIndices.empty()) {
      TRCP(("GPU PDE: no valid blocks to process; using CPU path.\n"));
      if (checksumOut) *checksumOut = 0;
      return false;
    }

    const size_t numBlocks = validBlockIndices.size();
    TRCP(("GPU PDE: packed %zu blocks with halos, %.2f MB\n",
          numBlocks, (haloBlocksFloat.size() * sizeof(float)) / (1024.0 * 1024.0)));

    // Create Metal buffers
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    if (!commandQueue) {
      TRCP(("GPU PDE: failed to create command queue.\n"));
      if (checksumOut) *checksumOut = 0;
      return false;
    }

    // Input and output buffers (ping-pong for iterations)
    id<MTLBuffer> bufferIn = [device newBufferWithBytes:haloBlocksFloat.data()
                                                  length:haloBlocksFloat.size() * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferOut = [device newBufferWithLength:numBlocks * voxelsPerBlock * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    if (!bufferIn || !bufferOut) {
      TRCP(("GPU PDE: failed to create Metal buffers.\n"));
      if (checksumOut) *checksumOut = 0;
      return false;
    }

    // Compile shader
    NSError* error = nil;
    NSString* shaderSource = [NSString stringWithUTF8String:kMetalPdeShaderSrc];
    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
    if (!library) {
      const char* errMsg = error ? [[error localizedDescription] UTF8String] : "unknown";
      TRCP(("GPU PDE: failed to compile Metal shader (%s).\n", errMsg ? errMsg : "unknown"));
      if (checksumOut) *checksumOut = 0;
      return false;
    }

    id<MTLFunction> function = [library newFunctionWithName:@"meanCurvaturePde"];
    if (!function) {
      TRCP(("GPU PDE: Metal function 'meanCurvaturePde' missing.\n"));
      if (checksumOut) *checksumOut = 0;
      return false;
    }

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
      const char* errMsg = error ? [[error localizedDescription] UTF8String] : "unknown";
      TRCP(("GPU PDE: failed to create compute pipeline (%s).\n", errMsg ? errMsg : "unknown"));
      if (checksumOut) *checksumOut = 0;
      return false;
    }

    id<MTLArgumentEncoder> pdeArgEncoder = [function newArgumentEncoderWithBufferIndex:0];
    if (!pdeArgEncoder) {
      TRCP(("GPU PDE: failed to create argument encoder.\n"));
      if (checksumOut) *checksumOut = 0;
      return false;
    }

    // Parameters
    struct PdeParams {
      uint32_t blockSize;
      uint32_t haloSize;
      uint32_t blockStride;
      float dt;
      float lambda;
      float densThr;
    } params;
    params.blockSize = static_cast<uint32_t>(blockSize);
    params.haloSize = static_cast<uint32_t>(haloSize);
    params.blockStride = static_cast<uint32_t>(voxelsPerHaloBlock);
    params.dt = dt;
    params.lambda = 0.0f;
    params.densThr = 0.0f;

    const NSUInteger pdeArgStride = pdeArgEncoder.encodedLength;
    id<MTLBuffer> pdeArgBuffer =
        [device newBufferWithLength:pdeArgStride * numBlocks
                             options:MTLResourceStorageModeShared];
    if (!pdeArgBuffer) {
      TRCP(("GPU PDE: failed to allocate argument buffer.\n"));
      if (checksumOut) *checksumOut = 0;
      return false;
    }
    [pdeArgBuffer setLabel:@"pde_args"];
    for (size_t i = 0; i < numBlocks; ++i) {
      const NSUInteger entryOffset = pdeArgStride * i;
      [pdeArgEncoder setArgumentBuffer:pdeArgBuffer offset:entryOffset];
      size_t inOffset = i * voxelsPerHaloBlock * sizeof(float);
      size_t outOffset = i * voxelsPerBlock * sizeof(float);
      [pdeArgEncoder setBuffer:bufferIn offset:inOffset atIndex:0];
      [pdeArgEncoder setBuffer:bufferOut offset:outOffset atIndex:1];
      void* paramsPtr = [pdeArgEncoder constantDataAtIndex:2];
      if (!paramsPtr) {
        TRCP(("GPU PDE: argument encoder missing params slot.\n"));
        if (checksumOut) *checksumOut = 0;
        return false;
      }
      std::memcpy(paramsPtr, &params, sizeof(params));
    }

    id<MTLResidencySet> pdeResidency = nil;
    if (@available(macOS 15.0, *)) {
      MTLResidencySetDescriptor* desc = [MTLResidencySetDescriptor new];
      desc.label = @"msbg_pde_residency";
      desc.initialCapacity = 4;
      NSError* residencyError = nil;
      pdeResidency = [device newResidencySetWithDescriptor:desc error:&residencyError];
      if (!pdeResidency && residencyError) {
        const char* errMsg = [[residencyError localizedDescription] UTF8String];
        TRCP(("GPU PDE: failed to allocate residency set (%s).\n",
              errMsg ? errMsg : "unknown"));
      }
      if (pdeResidency) {
        NSArray<id<MTLAllocation>>* allocations = @[
          bufferIn, bufferOut, pdeArgBuffer
        ];
        for (id<MTLAllocation> allocation in allocations) {
          if (allocation) {
            [pdeResidency addAllocation:allocation];
          }
        }
        [pdeResidency commit];
        [pdeResidency requestResidency];
      }
    }

    // Run iterations (for now, process one iteration at a time and reconstruct halos)
    // TODO: Optimize to process multiple iterations with proper halo reconstruction
    for (int iter = 0; iter < nIter; ++iter) {
      id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

      [encoder setComputePipelineState:pipeline];

      // Dispatch per block
      const MTLSize threadsPerGroup = MTLSizeMake(8, 8, 4);
      for (size_t i = 0; i < numBlocks; ++i) {
        const NSUInteger entryOffset = pdeArgStride * i;
        [encoder setBuffer:pdeArgBuffer offset:entryOffset atIndex:0];

        MTLSize gridSize = MTLSizeMake(blockSize, blockSize, blockSize);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];
      }

      // Metal 4: Add memory barrier for proper synchronization (allows better GPU pipelining)
      // Add barrier before endEncoding to ensure proper ordering between compute stages
      if (@available(macOS 13.0, *)) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
      [encoder endEncoding];
      if (pdeResidency) {
        if (@available(macOS 15.0, *)) {
          [commandBuffer useResidencySet:pdeResidency];
        }
      }
      [commandBuffer commit];
      
      // We still need to wait since we read back data between iterations
      // The barrier ensures proper GPU-side synchronization, allowing better pipelining
      [commandBuffer waitUntilCompleted];

      if (commandBuffer.error) {
        NSError* cmdError = commandBuffer.error;
        const char* errMsg = cmdError ? [[cmdError localizedDescription] UTF8String] : "unknown";
        TRCP(("GPU PDE: iteration %d failed (%s).\n", iter, errMsg ? errMsg : "unknown"));
        if (checksumOut) *checksumOut = 0;
        return false;
      }
      if (pdeResidency) {
        if (@available(macOS 15.0, *)) {
          [pdeResidency endResidency];
        }
      }

      // For subsequent iterations, update blocks and reconstruct halos
      if (iter < nIter - 1) {
        // Update block data from output
        float* outputData = static_cast<float*>(bufferOut.contents);
        for (size_t i = 0; i < numBlocks; ++i) {
          int bid = validBlockIndices[i];
          uint16_t* blockData = sg->getBlockDataPtr(bid);
          if (!blockData) continue;

          float* blockOut = outputData + i * voxelsPerBlock;
          for (int v = 0; v < voxelsPerBlock; ++v) {
            float f = blockOut[v];
            f = std::max(0.0f, std::min(1.0f, f));
            blockData[v] = static_cast<uint16_t>(roundf(f * 65535.0f));
          }
        }

        // Reconstruct halo blocks for next iteration
        float* inputData = static_cast<float*>(bufferIn.contents);
        for (size_t i = 0; i < numBlocks; ++i) {
          int bid = validBlockIndices[i];
          Vec4i bpos = sg->getBlockCoordsById(bid);
          int bx0 = vget_x(bpos);
          int by0 = vget_y(bpos);
          int bz0 = vget_z(bpos);

          Vec4i pos0 = Vec4i(bx0, by0, bz0, 0) * blockSize;
          Vec4i pos1 = pos0 - haloSize;
          Vec4i pos2 = pos0 + blockSize + haloSize - 1;

          int x1 = vget_x(pos1), x2 = vget_x(pos2);
          int y1 = vget_y(pos1), y2 = vget_y(pos2);
          int z1 = vget_z(pos1), z2 = vget_z(pos2);

          float* haloBlock = inputData + i * voxelsPerHaloBlock;

          // Reconstruct halo from updated blocks
          for (int z = z1; z <= z2; ++z) {
            bool inRangeZ = z >= 0 && z < sz;
            int ibz = inRangeZ ? (z >> bsxLog2) * nbxy : -1;
            int ivz = inRangeZ ? (z & bsxMask) << (2 * bsxLog2) : 0;
            int ihz = (z - z1) * haloBlockSize * haloBlockSize;

            for (int y = y1; y <= y2; ++y) {
              bool inRangeZY = inRangeZ && y >= 0 && y < sy;
              int ibzy = inRangeZY ? ibz + (y >> bsxLog2) * nbx : -1;
              int ivzy = inRangeZY ? ivz + ((y & bsxMask) << bsxLog2) : 0;
              int ihzy = ihz + (y - y1) * haloBlockSize;

              for (int x = x1; x <= x2; ++x) {
                int ih = ihzy + (x - x1);
                float val = emptyValueFloat;

                if (inRangeZY && x >= 0 && x < sx) {
                  int ib = ibzy + (x >> bsxLog2);
                  int iv = ivzy + (x & bsxMask);

                  if (ib >= 0 && ib < sg->nBlocks() && sg->isValueBlock(ib)) {
                    uint16_t* neighborData = sg->getBlockDataPtr(ib);
                    if (neighborData && iv >= 0 && iv < voxelsPerBlock) {
                      val = static_cast<float>(neighborData[iv]) / 65535.0f;
                    }
                  }
                }

                haloBlock[ih] = val;
              }
            }
          }
        }
      }
    }

    // Read back results
    float* outputData = static_cast<float*>(bufferOut.contents);
    uint64_t outputChecksum = 0;

    // Convert back to uint16_t and write to grid
    for (size_t i = 0; i < numBlocks; ++i) {
      int bid = validBlockIndices[i];
      
      // Validate bid before accessing
      if (bid < 0 || bid >= sg->nBlocks()) {
        TRCP(("GPU PDE: invalid bid %d in readback (nBlocks=%lld)\n",
              bid, static_cast<long long>(sg->nBlocks())));
        continue;
      }
      
      if (!sg->isValueBlock(bid)) {
        continue;
      }
      
      uint16_t* blockData = sg->getBlockDataPtr(bid);
      if (!blockData) continue;

      float* blockOut = outputData + i * voxelsPerBlock;
      for (int v = 0; v < voxelsPerBlock; ++v) {
        float f = blockOut[v];
        f = std::max(0.0f, std::min(1.0f, f));  // Clamp
        uint16_t val = static_cast<uint16_t>(roundf(f * 65535.0f));
        blockData[v] = val;
        outputChecksum ^= static_cast<uint64_t>(val);
        outputChecksum *= 1099511628211ULL;  // FNV prime
      }
    }

    if (checksumOut) *checksumOut = outputChecksum;
    TRCP(("GPU PDE: completed %d iterations on %zu blocks, checksum=0x%016llx\n",
          nIter, numBlocks, static_cast<unsigned long long>(outputChecksum)));
    return true;
  }
}

}  // namespace Backend
constexpr bool kRaymarchWriteGradient = false;
