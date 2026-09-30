// Host-side reference implementations of every pipeline operation. These are
// used (a) by `--verify` to validate the GPU results against the CPU and
// (b) by the `--cpu-only` debug mode. They are intentionally written with the
// exact same formulas and rounding rules as the CUDA kernels so that results
// match to within a small tolerance (small differences can still occur due to
// floating-point contraction / FMA ordering on the GPU).

#ifndef CUDA_BATCH_IMAGE_PROCESSOR_SRC_CPU_REFERENCE_H_
#define CUDA_BATCH_IMAGE_PROCESSOR_SRC_CPU_REFERENCE_H_

#include <vector>

#include "image_io.h"

namespace imgproc {

// All functions return a packed output buffer: one byte per pixel for the
// grayscale-producing operations and four bytes (RGBA) per pixel otherwise.
std::vector<unsigned char> CpuGrayscale(const RgbaImage& image);
std::vector<unsigned char> CpuInvert(const RgbaImage& image);
std::vector<unsigned char> CpuBrighten(const RgbaImage& image, int delta,
                                       float contrast);
std::vector<unsigned char> CpuGaussianBlur(const RgbaImage& image, float sigma);
std::vector<unsigned char> CpuSobel(const RgbaImage& image);
std::vector<unsigned char> CpuSharpen(const RgbaImage& image, float sigma,
                                      float amount);

}  // namespace imgproc

#endif  // CUDA_BATCH_IMAGE_PROCESSOR_SRC_CPU_REFERENCE_H_
