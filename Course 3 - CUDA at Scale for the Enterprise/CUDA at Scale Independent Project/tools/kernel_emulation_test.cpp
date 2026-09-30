// CPU emulation test for the CUDA kernels in src/kernels.cuh.
//
// The kernels' exact indexing and arithmetic - thread-to-pixel mapping,
// shared-memory tile layout, halo handling, border clamping - is transcribed
// into plain host loops and checked against the CPU reference
// implementation. This makes the kernel logic (in particular the tiled
// shared-memory Gaussian blur) testable on machines without an NVIDIA GPU;
// the kernels themselves still run only on the GPU.
//
// Build & run: make test-kernels

#include <cmath>
#include <cstdio>
#include <vector>

#include "common.h"
#include "cpu_reference.h"
#include "image_io.h"

namespace imgproc {

namespace {

// Mirrors of the device types used by the kernels.
struct U4 {
  unsigned char x, y, z, w;
};

struct F4 {
  float x, y, z, w;
};

int ClampIntHost(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Round-to-nearest-even + saturate, matching __float2int_rn() on the device
// (and SaturateU8() in cpu_reference.cpp).
unsigned char SaturateHost(float v) {
  const long i = std::lrintf(v);
  return static_cast<unsigned char>(i < 0 ? 0 : (i > 255 ? 255 : i));
}

// ---------------------------------------------------------------------------
// Transcription of RgbaToGrayKernel (fixed 16x16 blocks in the test below).
// ---------------------------------------------------------------------------

void EmulateGray(const unsigned char* input, unsigned char* gray, int width,
                 int height) {
  const int block = 16;
  for (int by = 0; by * block < height; ++by) {
    for (int bx = 0; bx * block < width; ++bx) {
      for (int ty = 0; ty < block; ++ty) {
        for (int tx = 0; tx < block; ++tx) {
          const int x = bx * block + tx;
          const int y = by * block + ty;
          if (x >= width || y >= height) {
            continue;  // Early return of the kernel.
          }
          U4 p;
          const size_t o = (static_cast<size_t>(y) * width + x) * 4;
          p.x = input[o + 0];
          p.y = input[o + 1];
          p.z = input[o + 2];
          p.w = input[o + 3];
          gray[y * width + x] = static_cast<unsigned char>(
              0.299f * p.x + 0.587f * p.y + 0.114f * p.z + 0.5f);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Transcription of GaussianBlurHKernel / GaussianBlurVKernel.
//
// The cooperative tile-load loop is written as a plain loop over the tile:
// the kernel's per-thread striding only distributes the same loads over the
// block, and every element is stored exactly once either way. The tile
// layout, halo geometry and accumulation order are transcribed exactly.
// ---------------------------------------------------------------------------

void EmulateBlurH(const unsigned char* input, F4* tmp, int width, int height,
                  const float* weights, int radius, int tile) {
  const int tile_w = tile + 2 * radius;
  const int tile_h = tile;
  std::vector<F4> sh(static_cast<size_t>(tile_w) * tile_h);
  for (int by = 0; by * tile < height; ++by) {
    for (int bx = 0; bx * tile < width; ++bx) {
      const int x0 = bx * tile - radius;
      const int y0 = by * tile;
      for (int i = 0; i < tile_w * tile_h; ++i) {
        const int ly = i / tile_w;
        const int lx = i - ly * tile_w;
        const int xx = ClampIntHost(x0 + lx, 0, width - 1);
        const int yy = ClampIntHost(y0 + ly, 0, height - 1);
        const size_t o = (static_cast<size_t>(yy) * width + xx) * 4;
        sh[i] = F4{
            static_cast<float>(input[o + 0]), static_cast<float>(input[o + 1]),
            static_cast<float>(input[o + 2]), static_cast<float>(input[o + 3])};
      }
      for (int ty = 0; ty < tile; ++ty) {
        for (int tx = 0; tx < tile; ++tx) {
          const int x = bx * tile + tx;
          const int y = y0 + ty;
          if (x >= width || y >= height) {
            continue;
          }
          F4 sum = F4{0.0f, 0.0f, 0.0f, 0.0f};
          const int base = ty * tile_w + tx;
          for (int k = 0; k <= 2 * radius; ++k) {
            const float w = weights[k];
            const F4& v = sh[base + k];
            sum.x += w * v.x;
            sum.y += w * v.y;
            sum.z += w * v.z;
            sum.w += w * v.w;
          }
          tmp[static_cast<size_t>(y) * width + x] = sum;
        }
      }
    }
  }
}

void EmulateBlurV(const F4* tmp, unsigned char* output, int width, int height,
                  const float* weights, int radius, int tile) {
  const int tile_w = tile;
  const int tile_h = tile + 2 * radius;
  std::vector<F4> sh(static_cast<size_t>(tile_w) * tile_h);
  for (int by = 0; by * tile < height; ++by) {
    for (int bx = 0; bx * tile < width; ++bx) {
      const int x0 = bx * tile;
      const int y0 = by * tile - radius;
      for (int i = 0; i < tile_w * tile_h; ++i) {
        const int ly = i / tile_w;
        const int lx = i - ly * tile_w;
        const int xx = ClampIntHost(x0 + lx, 0, width - 1);
        const int yy = ClampIntHost(y0 + ly, 0, height - 1);
        sh[i] = tmp[static_cast<size_t>(yy) * width + xx];
      }
      for (int ty = 0; ty < tile; ++ty) {
        for (int tx = 0; tx < tile; ++tx) {
          const int x = x0 + tx;
          const int y = by * tile + ty;
          if (x >= width || y >= height) {
            continue;
          }
          F4 sum = F4{0.0f, 0.0f, 0.0f, 0.0f};
          const int base = ty * tile_w + tx;
          for (int k = 0; k <= 2 * radius; ++k) {
            const float w = weights[k];
            const F4& v = sh[base + k * tile_w];
            sum.x += w * v.x;
            sum.y += w * v.y;
            sum.z += w * v.z;
            sum.w += w * v.w;
          }
          const size_t idx = (static_cast<size_t>(y) * width + x) * 4;
          output[idx + 0] = SaturateHost(sum.x);
          output[idx + 1] = SaturateHost(sum.y);
          output[idx + 2] = SaturateHost(sum.z);
          output[idx + 3] = SaturateHost(sum.w);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Transcription of SobelKernel (fixed 16x16 blocks in the test below).
// ---------------------------------------------------------------------------

void EmulateSobel(const unsigned char* gray, unsigned char* magnitude,
                  int width, int height) {
  const int block = 16;
  for (int by = 0; by * block < height; ++by) {
    for (int bx = 0; bx * block < width; ++bx) {
      for (int ty = 0; ty < block; ++ty) {
        for (int tx = 0; tx < block; ++tx) {
          const int x = bx * block + tx;
          const int y = by * block + ty;
          if (x >= width || y >= height) {
            continue;
          }
          const int xm = ClampIntHost(x - 1, 0, width - 1);
          const int xp = ClampIntHost(x + 1, 0, width - 1);
          const int ym = ClampIntHost(y - 1, 0, height - 1);
          const int yp = ClampIntHost(y + 1, 0, height - 1);
          const unsigned char* row_m = gray + static_cast<size_t>(ym) * width;
          const unsigned char* row_0 = gray + static_cast<size_t>(y) * width;
          const unsigned char* row_p = gray + static_cast<size_t>(yp) * width;
          const int gx = -row_m[xm] - 2 * row_0[xm] - row_p[xm] + row_m[xp] +
                         2 * row_0[xp] + row_p[xp];
          const int gy = -row_m[xm] - 2 * row_m[x] - row_m[xp] + row_p[xm] +
                         2 * row_p[x] + row_p[xp];
          magnitude[static_cast<size_t>(y) * width + x] =
              SaturateHost(std::sqrt(static_cast<float>(gx * gx + gy * gy)));
        }
      }
    }
  }
}

// Deterministic image content (LCG), covering every clamping path.
void FillDeterministic(RgbaImage* image, int width, int height, unsigned seed) {
  image->width = width;
  image->height = height;
  image->pixels.resize(static_cast<size_t>(width) * height * 4);
  unsigned state = seed * 2654435761u + 1u;
  for (size_t i = 0; i < image->pixels.size(); ++i) {
    state = state * 1664525u + 1013904223u;
    image->pixels[i] = static_cast<unsigned char>(state >> 24);
  }
}

int MaxDiff(const std::vector<unsigned char>& a,
            const std::vector<unsigned char>& b) {
  int max_diff = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const int d = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
    if (d > max_diff) {
      max_diff = d;
    }
  }
  return max_diff;
}

struct TestCase {
  int width;
  int height;
  float sigma;
  int tile;
};

}  // namespace

int RunKernelEmulationTests() {
  const TestCase cases[] = {
      {12, 9, 1.0f, 16},     // Smaller than one block.
      {67, 41, 0.5f, 8},     // Non-multiples of the tile size.
      {67, 41, 1.0f, 16},    // Non-multiples of the tile size.
      {128, 128, 1.0f, 16},  // Exact multiple of the tile size.
      {200, 60, 1.5f, 32},   // Wide image, large tile.
      {320, 240, 5.0f, 16},  // Sigma clamps the radius to 15.
      {256, 256, 2.0f, 16},  // Typical case.
      {513, 77, 3.0f, 8},    // Odd dimensions, small tile.
  };
  const int case_count = static_cast<int>(sizeof(cases) / sizeof(cases[0]));

  int failures = 0;
  std::printf("%-6s %-4s %-6s %-5s | %-22s %-22s %-22s\n", "w", "h", "sigma",
              "tile", "gray max|diff|", "blur max|diff|", "sobel max|diff|");
  for (int c = 0; c < case_count; ++c) {
    const TestCase& t = cases[c];
    RgbaImage image;
    FillDeterministic(&image, t.width, t.height, static_cast<unsigned>(c) + 1u);

    int radius = 0;
    const std::vector<float> weights = ComputeGaussianKernel(t.sigma, &radius);

    // Emulated kernels.
    std::vector<unsigned char> emu_gray(static_cast<size_t>(t.width) *
                                        t.height);
    EmulateGray(image.pixels.data(), emu_gray.data(), t.width, t.height);

    std::vector<F4> tmp(static_cast<size_t>(t.width) * t.height);
    std::vector<unsigned char> emu_blur(static_cast<size_t>(t.width) *
                                        t.height * 4);
    EmulateBlurH(image.pixels.data(), tmp.data(), t.width, t.height,
                 weights.data(), radius, t.tile);
    EmulateBlurV(tmp.data(), emu_blur.data(), t.width, t.height, weights.data(),
                 radius, t.tile);

    std::vector<unsigned char> emu_sobel(static_cast<size_t>(t.width) *
                                         t.height);
    EmulateSobel(emu_gray.data(), emu_sobel.data(), t.width, t.height);

    // CPU reference implementations.
    const std::vector<unsigned char> ref_gray = CpuGrayscale(image);
    const std::vector<unsigned char> ref_blur = CpuGaussianBlur(image, t.sigma);
    const std::vector<unsigned char> ref_sobel = CpuSobel(image);

    const int d_gray = MaxDiff(emu_gray, ref_gray);
    const int d_blur = MaxDiff(emu_blur, ref_blur);
    const int d_sobel = MaxDiff(emu_sobel, ref_sobel);
    // The transcription performs the identical float operations in the
    // identical order, so any difference above 0 is a transcription or
    // kernel bug (1 is tolerated for defensive float-contract slack).
    const bool pass = d_gray <= 1 && d_blur <= 1 && d_sobel <= 1;
    if (!pass) {
      ++failures;
    }
    std::printf("%-6d %-4d %-6.2f %-5d | %-22d %-22d %-22d  %s\n", t.width,
                t.height, t.sigma, t.tile, d_gray, d_blur, d_sobel,
                pass ? "PASS" : "FAIL");
  }
  std::printf("\n%s (%d test cases)\n",
              failures == 0 ? "ALL KERNEL EMULATION TESTS PASS" : "FAILURES",
              case_count);
  return failures == 0 ? 0 : 1;
}

}  // namespace imgproc

int main() { return imgproc::RunKernelEmulationTests(); }
