// CUDA kernels for the batch image processor.
//
// This header is included only by translation units compiled by nvcc. All
// kernels are fully defined here so that no separate device-code link step
// (`-rdc=true`) is required. The per-pixel math intentionally mirrors
// cpu_reference.cpp one-to-one so `--verify` can compare the two.

#ifndef CUDA_BATCH_IMAGE_PROCESSOR_SRC_KERNELS_CUH_
#define CUDA_BATCH_IMAGE_PROCESSOR_SRC_KERNELS_CUH_

#include <cuda_runtime.h>

namespace imgproc {

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ int ClampInt(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// __float2int_rn rounds to nearest (ties to even), matching lrintf() in the
// CPU reference implementation.
__device__ __forceinline__ unsigned char SaturateU8(float v) {
  const int i = __float2int_rn(v);
  return static_cast<unsigned char>(ClampInt(i, 0, 255));
}

__device__ __forceinline__ float4 RgbaToFloat4(uchar4 p) {
  return make_float4(p.x, p.y, p.z, p.w);
}

__device__ __forceinline__ uchar4 SaturateFloat4ToUchar4(float4 v) {
  return make_uchar4(SaturateU8(v.x), SaturateU8(v.y), SaturateU8(v.z),
                     SaturateU8(v.w));
}

// ---------------------------------------------------------------------------
// Point operations
// ---------------------------------------------------------------------------

// Converts an RGBA image to 8-bit grayscale using BT.601 luma weights.
__global__ void RgbaToGrayKernel(const uchar4* __restrict__ input,
                                 unsigned char* __restrict__ gray, int width,
                                 int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const uchar4 p = input[y * width + x];
  gray[y * width + x] = static_cast<unsigned char>(0.299f * p.x + 0.587f * p.y +
                                                   0.114f * p.z + 0.5f);
}

// Inverts every color channel, leaving alpha untouched.
__global__ void InvertKernel(const uchar4* __restrict__ input,
                             uchar4* __restrict__ output, int width,
                             int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const uchar4 p = input[y * width + x];
  output[y * width + x] = make_uchar4(255 - p.x, 255 - p.y, 255 - p.z, p.w);
}

// Brightness / contrast: out = (in - 128) * contrast + 128 + delta.
__global__ void BrightenKernel(const uchar4* __restrict__ input,
                               uchar4* __restrict__ output, int width,
                               int height, int delta, float contrast) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const uchar4 p = input[y * width + x];
  output[y * width + x] =
      make_uchar4(SaturateU8((p.x - 128.0f) * contrast + 128.0f + delta),
                  SaturateU8((p.y - 128.0f) * contrast + 128.0f + delta),
                  SaturateU8((p.z - 128.0f) * contrast + 128.0f + delta), p.w);
}

// ---------------------------------------------------------------------------
// Separable Gaussian blur (shared-memory tiled)
// ---------------------------------------------------------------------------
// The blur is split into a horizontal and a vertical pass. Each pass stages a
// tile with halo regions in shared memory so every input byte is read from
// global memory only once. `tmp` is a float4 scratch buffer that holds the
// intermediate (horizontal) result to avoid rounding loss between passes.
//
// Shared-memory requirements (16-byte float4 elements):
//   horizontal: (blockDim.x + 2 * radius) * blockDim.y elements
//   vertical:   blockDim.x * (blockDim.y + 2 * radius) elements
// With the default 16x16 block and the maximum radius of 15 this stays below
// 12 KiB, comfortably inside the default 48 KiB per-block limit.

__global__ void GaussianBlurHKernel(const uchar4* __restrict__ input,
                                    float4* __restrict__ tmp, int width,
                                    int height,
                                    const float* __restrict__ weights,
                                    int radius) {
  extern __shared__ float4 tile[];
  const int tile_w = blockDim.x + 2 * radius;
  const int tile_h = blockDim.y;
  const int x0 = blockIdx.x * blockDim.x - radius;
  const int y0 = blockIdx.y * blockDim.y;
  const int block_threads = blockDim.x * blockDim.y;

  // Cooperatively load the tile (with halo) into shared memory. Border pixels
  // are clamped so image edges are treated as replicated.
  for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < tile_w * tile_h;
       i += block_threads) {
    const int ly = i / tile_w;
    const int lx = i - ly * tile_w;
    const int xx = ClampInt(x0 + lx, 0, width - 1);
    const int yy = ClampInt(y0 + ly, 0, height - 1);
    tile[i] = RgbaToFloat4(input[yy * width + xx]);
  }
  __syncthreads();

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = y0 + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const int base = threadIdx.y * tile_w + threadIdx.x;
  for (int k = 0; k <= 2 * radius; ++k) {
    const float w = weights[k];
    const float4 v = tile[base + k];
    sum.x += w * v.x;
    sum.y += w * v.y;
    sum.z += w * v.z;
    sum.w += w * v.w;
  }
  tmp[y * width + x] = sum;
}

__global__ void GaussianBlurVKernel(const float4* __restrict__ tmp,
                                    uchar4* __restrict__ output, int width,
                                    int height,
                                    const float* __restrict__ weights,
                                    int radius) {
  extern __shared__ float4 tile[];
  const int tile_w = blockDim.x;
  const int tile_h = blockDim.y + 2 * radius;
  const int x0 = blockIdx.x * blockDim.x;
  const int y0 = blockIdx.y * blockDim.y - radius;
  const int block_threads = blockDim.x * blockDim.y;

  for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < tile_w * tile_h;
       i += block_threads) {
    const int ly = i / tile_w;
    const int lx = i - ly * tile_w;
    const int xx = ClampInt(x0 + lx, 0, width - 1);
    const int yy = ClampInt(y0 + ly, 0, height - 1);
    tile[i] = tmp[yy * width + xx];
  }
  __syncthreads();

  const int x = x0 + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const int base = threadIdx.y * tile_w + threadIdx.x;
  for (int k = 0; k <= 2 * radius; ++k) {
    const float w = weights[k];
    const float4 v = tile[base + k * tile_w];
    sum.x += w * v.x;
    sum.y += w * v.y;
    sum.z += w * v.z;
    sum.w += w * v.w;
  }
  output[y * width + x] = SaturateFloat4ToUchar4(sum);
}

// ---------------------------------------------------------------------------
// Sobel edge detection
// ---------------------------------------------------------------------------

// 3x3 Sobel operator applied to the grayscale image; writes the gradient
// magnitude. Border pixels are handled by edge replication (clamping).
__global__ void SobelKernel(const unsigned char* __restrict__ gray,
                            unsigned char* __restrict__ magnitude, int width,
                            int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const int xm = ClampInt(x - 1, 0, width - 1);
  const int xp = ClampInt(x + 1, 0, width - 1);
  const int ym = ClampInt(y - 1, 0, height - 1);
  const int yp = ClampInt(y + 1, 0, height - 1);
  const unsigned char* row_m = gray + static_cast<size_t>(ym) * width;
  const unsigned char* row_0 = gray + static_cast<size_t>(y) * width;
  const unsigned char* row_p = gray + static_cast<size_t>(yp) * width;
  const int gx = -row_m[xm] - 2 * row_0[xm] - row_p[xm] + row_m[xp] +
                 2 * row_0[xp] + row_p[xp];
  const int gy = -row_m[xm] - 2 * row_m[x] - row_m[xp] + row_p[xm] +
                 2 * row_p[x] + row_p[xp];
  magnitude[static_cast<size_t>(y) * width + x] =
      SaturateU8(sqrtf(static_cast<float>(gx * gx + gy * gy)));
}

// ---------------------------------------------------------------------------
// Unsharp-mask sharpening
// ---------------------------------------------------------------------------

// out = in + amount * (in - blurred), per color channel.
__global__ void SharpenKernel(const uchar4* __restrict__ input,
                              const uchar4* __restrict__ blurred,
                              uchar4* __restrict__ output, int width,
                              int height, float amount) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const size_t idx = static_cast<size_t>(y) * width + x;
  const uchar4 p = input[idx];
  const uchar4 b = blurred[idx];
  output[idx] = make_uchar4(SaturateU8(p.x + amount * (p.x - b.x)),
                            SaturateU8(p.y + amount * (p.y - b.y)),
                            SaturateU8(p.z + amount * (p.z - b.z)), p.w);
}

// ---------------------------------------------------------------------------
// Histogram (gray-level statistics)
// ---------------------------------------------------------------------------

// Grid-stride loop accumulating a 256-bin histogram with global atomics.
__global__ void HistogramKernel(const unsigned char* __restrict__ gray,
                                int* __restrict__ histogram, int pixel_count) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < pixel_count;
       i += stride) {
    atomicAdd(&histogram[gray[i]], 1);
  }
}

// ---------------------------------------------------------------------------
// Histogram equalization
// ---------------------------------------------------------------------------

// Applies the 256-entry lookup table (built on the host from the gray
// histogram by BuildEqualizationLut) to the grayscale image. Because both the
// CPU reference and the GPU pipeline build the table with the exact same
// host code, this gather is expected to match the reference exactly.
__global__ void EqualizeKernel(const unsigned char* __restrict__ gray,
                               const unsigned char* __restrict__ lut,
                               unsigned char* __restrict__ output, int width,
                               int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const size_t idx = static_cast<size_t>(y) * width + x;
  output[idx] = lut[gray[idx]];
}

}  // namespace imgproc

#endif  // CUDA_BATCH_IMAGE_PROCESSOR_SRC_KERNELS_CUH_
