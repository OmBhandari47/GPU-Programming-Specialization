// Common types and helpers shared by the CPU reference implementation and the
// GPU pipeline. This header is host-only (no CUDA dependencies) so that every
// translation unit builds with a plain C++17 host compiler as well.

#ifndef CUDA_BATCH_IMAGE_PROCESSOR_SRC_COMMON_H_
#define CUDA_BATCH_IMAGE_PROCESSOR_SRC_COMMON_H_

#include <string>
#include <vector>

namespace imgproc {

// Image processing operations supported by the pipeline. Every operation reads
// an RGBA image and produces either an RGBA image or an 8-bit grayscale image.
enum class Operation {
  kGray,      // RGBA to 8-bit grayscale (ITU-R BT.601 luma).
  kInvert,    // Channel-wise color inversion.
  kBrighten,  // Brightness / contrast adjustment.
  kBlur,      // Separable Gaussian blur.
  kSobel,     // 3x3 Sobel gradient magnitude (grayscale output).
  kSharpen,   // Unsharp-mask sharpening (blur + blend).
};

// Pixel layout of an operation result.
enum class OutputFormat { kGrayscale, kRgba };

// Tunable parameters shared by the CPU reference and the GPU pipeline.
struct PipelineParams {
  float sigma = 1.0f;           // Gaussian sigma for blur / sharpen.
  int brightness_delta = 25;    // Constant added to every channel.
  float contrast = 1.0f;        // Contrast factor applied around 128.
  float sharpen_amount = 0.8f;  // Unsharp-mask strength.
  int tile = 16;                // CUDA block edge length (blur tiling).
  int device = 0;               // CUDA device index.
};

// A single processed image returned by the pipeline.
struct OpResult {
  Operation op = Operation::kGray;
  std::string op_name;
  OutputFormat format = OutputFormat::kGrayscale;
  int width = 0;
  int height = 0;
  std::vector<unsigned char> pixels;  // Host copy of the processed pixels.
  float device_ms = 0.0f;             // GPU kernel time for this operation.
};

// Gray-level statistics computed from a 256-bin histogram.
struct GrayStats {
  long long histogram[256] = {};
  int min_value = 0;
  int max_value = 255;
  double mean_value = 0.0;
};

// Returns a short, filesystem-safe name for `op` ("gray", "sobel", ...).
std::string OperationName(Operation op);

// Parses an operation name (case insensitive). Returns false if unknown.
bool ParseOperationName(const std::string& name, Operation* op);

// Computes a normalized 1-D Gaussian kernel for `sigma`. The kernel always has
// an odd length (2 * radius + 1) and sums to 1.0. The radius is clamped to 15
// so the shared-memory blur kernels always fit in the default 48 KiB limit.
std::vector<float> ComputeGaussianKernel(float sigma, int* radius);

// Converts a 256-bin integer histogram into gray-level statistics.
GrayStats StatsFromHistogram(const int* histogram);

}  // namespace imgproc

#endif  // CUDA_BATCH_IMAGE_PROCESSOR_SRC_COMMON_H_
