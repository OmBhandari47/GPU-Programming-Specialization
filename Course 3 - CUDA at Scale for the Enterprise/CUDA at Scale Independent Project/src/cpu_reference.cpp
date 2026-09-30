#include "cpu_reference.h"

#include <cmath>
#include <cstddef>

#include "common.h"

namespace imgproc {

namespace {

inline int ClampInt(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// lrintf() uses the current rounding mode (round-to-nearest-even by default),
// which matches __float2int_rn() used by the CUDA kernels in kernels.cuh.
inline unsigned char SaturateU8(float v) {
  const long i = std::lrintf(v);
  return static_cast<unsigned char>(i < 0 ? 0 : (i > 255 ? 255 : i));
}

}  // namespace

std::vector<unsigned char> CpuGrayscale(const RgbaImage& image) {
  const size_t n = image.PixelCount();
  std::vector<unsigned char> gray(n);
  const unsigned char* in = image.pixels.data();
  for (size_t i = 0; i < n; ++i) {
    const size_t o = i * 4;
    gray[i] = static_cast<unsigned char>(
        0.299f * in[o + 0] + 0.587f * in[o + 1] + 0.114f * in[o + 2] + 0.5f);
  }
  return gray;
}

std::vector<unsigned char> CpuInvert(const RgbaImage& image) {
  const size_t n = image.PixelCount();
  std::vector<unsigned char> out(n * 4);
  const unsigned char* in = image.pixels.data();
  for (size_t i = 0; i < n; ++i) {
    const size_t o = i * 4;
    out[o + 0] = static_cast<unsigned char>(255 - in[o + 0]);
    out[o + 1] = static_cast<unsigned char>(255 - in[o + 1]);
    out[o + 2] = static_cast<unsigned char>(255 - in[o + 2]);
    out[o + 3] = in[o + 3];  // Alpha is left untouched.
  }
  return out;
}

std::vector<unsigned char> CpuBrighten(const RgbaImage& image, int delta,
                                       float contrast) {
  const size_t n = image.PixelCount();
  std::vector<unsigned char> out(n * 4);
  const unsigned char* in = image.pixels.data();
  for (size_t i = 0; i < n; ++i) {
    const size_t o = i * 4;
    for (int c = 0; c < 3; ++c) {
      const float v = (static_cast<float>(in[o + c]) - 128.0f) * contrast +
                      128.0f + static_cast<float>(delta);
      out[o + c] = SaturateU8(v);
    }
    out[o + 3] = in[o + 3];
  }
  return out;
}

std::vector<unsigned char> CpuGaussianBlur(const RgbaImage& image,
                                           float sigma) {
  int radius = 0;
  const std::vector<float> weights = ComputeGaussianKernel(sigma, &radius);
  const int width = image.width;
  const int height = image.height;
  const size_t n = image.PixelCount();
  const unsigned char* in = image.pixels.data();

  // Horizontal pass: RGBA bytes -> float4 scratch buffer.
  std::vector<float> tmp(n * 4);
  for (int y = 0; y < height; ++y) {
    const size_t row = static_cast<size_t>(y) * width;
    for (int x = 0; x < width; ++x) {
      float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
      for (int k = -radius; k <= radius; ++k) {
        const int xx = ClampInt(x + k, 0, width - 1);
        const float kw = weights[static_cast<size_t>(k + radius)];
        const unsigned char* q = &in[(row + xx) * 4];
        s0 += kw * q[0];
        s1 += kw * q[1];
        s2 += kw * q[2];
        s3 += kw * q[3];
      }
      float* t = &tmp[(row + x) * 4];
      t[0] = s0;
      t[1] = s1;
      t[2] = s2;
      t[3] = s3;
    }
  }

  // Vertical pass: float4 scratch buffer -> RGBA bytes.
  std::vector<unsigned char> out(n * 4);
  for (int y = 0; y < height; ++y) {
    const size_t row = static_cast<size_t>(y) * width;
    for (int x = 0; x < width; ++x) {
      float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
      for (int k = -radius; k <= radius; ++k) {
        const int yy = ClampInt(y + k, 0, height - 1);
        const float kw = weights[static_cast<size_t>(k + radius)];
        const float* t = &tmp[(static_cast<size_t>(yy) * width + x) * 4];
        s0 += kw * t[0];
        s1 += kw * t[1];
        s2 += kw * t[2];
        s3 += kw * t[3];
      }
      unsigned char* o = &out[(row + x) * 4];
      o[0] = SaturateU8(s0);
      o[1] = SaturateU8(s1);
      o[2] = SaturateU8(s2);
      o[3] = SaturateU8(s3);
    }
  }
  return out;
}

std::vector<unsigned char> CpuSobel(const RgbaImage& image) {
  const int width = image.width;
  const int height = image.height;
  const std::vector<unsigned char> gray = CpuGrayscale(image);
  std::vector<unsigned char> out(gray.size());
  for (int y = 0; y < height; ++y) {
    const int ym = ClampInt(y - 1, 0, height - 1);
    const int yp = ClampInt(y + 1, 0, height - 1);
    const unsigned char* row_m = &gray[static_cast<size_t>(ym) * width];
    const unsigned char* row_0 = &gray[static_cast<size_t>(y) * width];
    const unsigned char* row_p = &gray[static_cast<size_t>(yp) * width];
    for (int x = 0; x < width; ++x) {
      const int xm = ClampInt(x - 1, 0, width - 1);
      const int xp = ClampInt(x + 1, 0, width - 1);
      const int gx = -row_m[xm] - 2 * row_0[xm] - row_p[xm] + row_m[xp] +
                     2 * row_0[xp] + row_p[xp];
      const int gy = -row_m[xm] - 2 * row_m[x] - row_m[xp] + row_p[xm] +
                     2 * row_p[x] + row_p[xp];
      out[static_cast<size_t>(y) * width + x] =
          SaturateU8(std::sqrt(static_cast<float>(gx * gx + gy * gy)));
    }
  }
  return out;
}

std::vector<unsigned char> CpuSharpen(const RgbaImage& image, float sigma,
                                      float amount) {
  const size_t n = image.PixelCount();
  const std::vector<unsigned char> blurred = CpuGaussianBlur(image, sigma);
  std::vector<unsigned char> out(n * 4);
  const unsigned char* in = image.pixels.data();
  for (size_t i = 0; i < n; ++i) {
    const size_t o = i * 4;
    for (int c = 0; c < 3; ++c) {
      const float v = static_cast<float>(in[o + c]) +
                      amount * (static_cast<float>(in[o + c]) -
                                static_cast<float>(blurred[o + c]));
      out[o + c] = SaturateU8(v);
    }
    out[o + 3] = in[o + 3];
  }
  return out;
}

}  // namespace imgproc
