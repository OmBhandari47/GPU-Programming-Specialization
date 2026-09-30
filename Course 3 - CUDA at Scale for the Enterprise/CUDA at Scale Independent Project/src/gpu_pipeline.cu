// GPU pipeline implementation: device memory management, kernel launches and
// CUDA event timing. This is the only translation unit that talks to the CUDA
// runtime; it is compiled by nvcc and exposes a plain C++ interface
// (gpu_pipeline.h) to the rest of the program.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "gpu_pipeline.h"
#include "kernels.cuh"

namespace imgproc {

namespace {

struct DeviceWorkspace {
  uchar4* d_input = nullptr;         // Input image (RGBA).
  float4* d_scratch = nullptr;       // Blur intermediate (float4).
  uchar4* d_blur = nullptr;          // Blurred copy used by sharpen.
  uchar4* d_output = nullptr;        // RGBA operation output.
  unsigned char* d_gray = nullptr;   // Grayscale version of the input.
  unsigned char* d_edges = nullptr;  // Sobel magnitude output.
  int* d_histogram = nullptr;        // 256-bin gray histogram.
  float* d_weights = nullptr;        // 1-D Gaussian kernel (max 31 taps).
  size_t capacity_pixels = 0;
  cudaEvent_t event_start = nullptr;
  cudaEvent_t event_stop = nullptr;
};

DeviceWorkspace g_workspace;

bool Check(cudaError_t status, const char* what, std::string* error) {
  if (status == cudaSuccess) {
    return true;
  }
  *error = std::string(what) + " failed: " + cudaGetErrorString(status);
  return false;
}

// (Re)allocates the reusable device buffers so that images of up to
// `pixel_count` pixels fit. Existing buffers are freed first; in a batch run
// this reallocates only when a larger image than ever before is seen.
bool EnsureCapacity(size_t pixel_count, std::string* error) {
  if (pixel_count <= g_workspace.capacity_pixels) {
    return true;
  }
  DeviceWorkspace& ws = g_workspace;
  cudaFree(ws.d_input);
  cudaFree(ws.d_scratch);
  cudaFree(ws.d_blur);
  cudaFree(ws.d_output);
  cudaFree(ws.d_gray);
  cudaFree(ws.d_edges);
  const size_t rgba_bytes = pixel_count * sizeof(uchar4);
  if (!Check(cudaMalloc(&ws.d_input, rgba_bytes), "cudaMalloc(d_input)",
             error) ||
      !Check(cudaMalloc(&ws.d_scratch, rgba_bytes), "cudaMalloc(d_scratch)",
             error) ||
      !Check(cudaMalloc(&ws.d_blur, rgba_bytes), "cudaMalloc(d_blur)", error) ||
      !Check(cudaMalloc(&ws.d_output, rgba_bytes), "cudaMalloc(d_output)",
             error) ||
      !Check(cudaMalloc(&ws.d_gray, pixel_count), "cudaMalloc(d_gray)",
             error) ||
      !Check(cudaMalloc(&ws.d_edges, pixel_count), "cudaMalloc(d_edges)",
             error)) {
    return false;
  }
  if (ws.d_histogram == nullptr) {
    if (!Check(cudaMalloc(&ws.d_histogram, 256 * sizeof(int)),
               "cudaMalloc(d_histogram)", error)) {
      return false;
    }
  }
  if (ws.d_weights == nullptr) {
    if (!Check(cudaMalloc(&ws.d_weights, 31 * sizeof(float)),
               "cudaMalloc(d_weights)", error)) {
      return false;
    }
  }
  ws.capacity_pixels = pixel_count;
  return true;
}

}  // namespace

bool InitializeGpu(int device_index, GpuDeviceInfo* info, std::string* error) {
  int device_count = 0;
  if (!Check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", error)) {
    return false;
  }
  if (device_count <= 0) {
    *error = "no CUDA-capable device was found";
    return false;
  }
  if (device_index < 0 || device_index >= device_count) {
    *error = "CUDA device index " + std::to_string(device_index) +
             " is out of range (found " + std::to_string(device_count) +
             " devices)";
    return false;
  }
  cudaDeviceProp prop = {};
  if (!Check(cudaGetDeviceProperties(&prop, device_index),
             "cudaGetDeviceProperties", error)) {
    return false;
  }
  if (!Check(cudaSetDevice(device_index), "cudaSetDevice", error)) {
    return false;
  }
  info->name = prop.name;
  info->device = device_index;
  info->compute_major = prop.major;
  info->compute_minor = prop.minor;
  info->total_memory = prop.totalGlobalMem;
  info->shared_mem_per_block = static_cast<int>(prop.sharedMemPerBlock);
  if (!Check(cudaRuntimeGetVersion(&info->runtime_version),
             "cudaRuntimeGetVersion", error)) {
    return false;
  }
  if (!Check(cudaDriverGetVersion(&info->driver_version),
             "cudaDriverGetVersion", error)) {
    return false;
  }
  if (!Check(cudaEventCreate(&g_workspace.event_start), "cudaEventCreate",
             error)) {
    return false;
  }
  if (!Check(cudaEventCreate(&g_workspace.event_stop), "cudaEventCreate",
             error)) {
    return false;
  }
  return true;
}

bool ProcessImageGpu(const RgbaImage& image, const std::vector<Operation>& ops,
                     const PipelineParams& params,
                     std::vector<OpResult>* results, GrayStats* stats,
                     std::string* error) {
  results->clear();
  DeviceWorkspace& ws = g_workspace;
  const int width = image.width;
  const int height = image.height;
  const size_t pixel_count = image.PixelCount();
  if (width <= 0 || height <= 0) {
    *error = "invalid image dimensions";
    return false;
  }

  if (!EnsureCapacity(pixel_count, error)) {
    return false;
  }

  // Upload the input and the Gaussian weights.
  int radius = 0;
  const std::vector<float> weights =
      ComputeGaussianKernel(params.sigma, &radius);
  if (!Check(cudaMemcpy(ws.d_input, image.pixels.data(),
                        pixel_count * sizeof(uchar4), cudaMemcpyHostToDevice),
             "cudaMemcpy (H2D image)", error)) {
    return false;
  }
  if (!Check(cudaMemcpy(ws.d_weights, weights.data(),
                        weights.size() * sizeof(float), cudaMemcpyHostToDevice),
             "cudaMemcpy (H2D weights)", error)) {
    return false;
  }
  if (!Check(cudaMemset(ws.d_histogram, 0, 256 * sizeof(int)),
             "cudaMemset (histogram)", error)) {
    return false;
  }

  dim3 block(params.tile, params.tile);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  const size_t shared_h = (block.x + 2 * radius) * block.y * sizeof(float4);
  const size_t shared_v = block.x * (block.y + 2 * radius) * sizeof(float4);

  // Grayscale + histogram. The grayscale buffer is also the input of the
  // Sobel kernel, and the histogram feeds the per-image statistics.
  if (!Check(cudaEventRecord(ws.event_start), "cudaEventRecord", error)) {
    return false;
  }
  RgbaToGrayKernel<<<grid, block>>>(ws.d_input, ws.d_gray, width, height);
  {
    const int threads = 256;
    const int blocks = static_cast<int>(
        std::min<size_t>((pixel_count + threads - 1) / threads, 4096));
    HistogramKernel<<<blocks, threads>>>(ws.d_gray, ws.d_histogram,
                                         static_cast<int>(pixel_count));
  }
  if (!Check(cudaEventRecord(ws.event_stop), "cudaEventRecord", error)) {
    return false;
  }
  if (!Check(cudaEventSynchronize(ws.event_stop), "cudaEventSynchronize",
             error)) {
    return false;
  }
  if (!Check(cudaGetLastError(), "RgbaToGray/Histogram kernel launch", error)) {
    return false;
  }
  float gray_ms = 0.0f;
  if (!Check(cudaEventElapsedTime(&gray_ms, ws.event_start, ws.event_stop),
             "cudaEventElapsedTime", error)) {
    return false;
  }

  for (Operation op : ops) {
    OpResult result;
    result.op = op;
    result.op_name = OperationName(op);
    result.width = width;
    result.height = height;
    result.format = (op == Operation::kGray || op == Operation::kSobel)
                        ? OutputFormat::kGrayscale
                        : OutputFormat::kRgba;
    const unsigned char* src = nullptr;

    if (!Check(cudaEventRecord(ws.event_start), "cudaEventRecord", error)) {
      return false;
    }
    switch (op) {
      case Operation::kGray:
        src = ws.d_gray;  // Already computed (timed above).
        break;
      case Operation::kInvert:
        InvertKernel<<<grid, block>>>(ws.d_input, ws.d_output, width, height);
        src = reinterpret_cast<const unsigned char*>(ws.d_output);
        break;
      case Operation::kBrighten:
        BrightenKernel<<<grid, block>>>(ws.d_input, ws.d_output, width, height,
                                        params.brightness_delta,
                                        params.contrast);
        src = reinterpret_cast<const unsigned char*>(ws.d_output);
        break;
      case Operation::kBlur:
        GaussianBlurHKernel<<<grid, block, shared_h>>>(
            ws.d_input, ws.d_scratch, width, height, ws.d_weights, radius);
        GaussianBlurVKernel<<<grid, block, shared_v>>>(
            ws.d_scratch, ws.d_output, width, height, ws.d_weights, radius);
        src = reinterpret_cast<const unsigned char*>(ws.d_output);
        break;
      case Operation::kSobel:
        SobelKernel<<<grid, block>>>(ws.d_gray, ws.d_edges, width, height);
        src = ws.d_edges;
        break;
      case Operation::kSharpen:
        GaussianBlurHKernel<<<grid, block, shared_h>>>(
            ws.d_input, ws.d_scratch, width, height, ws.d_weights, radius);
        GaussianBlurVKernel<<<grid, block, shared_v>>>(
            ws.d_scratch, ws.d_blur, width, height, ws.d_weights, radius);
        SharpenKernel<<<grid, block>>>(ws.d_input, ws.d_blur, ws.d_output,
                                       width, height, params.sharpen_amount);
        src = reinterpret_cast<const unsigned char*>(ws.d_output);
        break;
    }
    if (!Check(cudaEventRecord(ws.event_stop), "cudaEventRecord", error)) {
      return false;
    }
    if (!Check(cudaEventSynchronize(ws.event_stop), "cudaEventSynchronize",
               error)) {
      return false;
    }
    if (!Check(cudaGetLastError(), "kernel launch", error)) {
      return false;
    }
    float ms = 0.0f;
    if (!Check(cudaEventElapsedTime(&ms, ws.event_start, ws.event_stop),
               "cudaEventElapsedTime", error)) {
      return false;
    }
    result.device_ms = (op == Operation::kGray) ? gray_ms : ms;

    const size_t bytes = (result.format == OutputFormat::kGrayscale)
                             ? pixel_count
                             : pixel_count * 4;
    result.pixels.resize(bytes);
    if (!Check(cudaMemcpy(result.pixels.data(), src, bytes,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy (D2H result)", error)) {
      return false;
    }
    results->push_back(std::move(result));
  }

  // Download the histogram and turn it into statistics on the host.
  int histogram[256] = {};
  if (!Check(cudaMemcpy(histogram, ws.d_histogram, sizeof(histogram),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy (D2H histogram)", error)) {
    return false;
  }
  *stats = StatsFromHistogram(histogram);
  return true;
}

}  // namespace imgproc
