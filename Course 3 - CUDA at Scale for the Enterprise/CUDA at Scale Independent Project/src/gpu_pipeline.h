// Public interface of the GPU pipeline. Deliberately free of CUDA types so
// that the rest of the program (main.cpp etc.) compiles with a host-only
// toolchain; only gpu_pipeline.cu / gpu_stub.cpp implement these functions.

#ifndef CUDA_BATCH_IMAGE_PROCESSOR_SRC_GPU_PIPELINE_H_
#define CUDA_BATCH_IMAGE_PROCESSOR_SRC_GPU_PIPELINE_H_

#include <string>
#include <vector>

#include "common.h"
#include "image_io.h"

namespace imgproc {

// Basic information about the selected CUDA device.
struct GpuDeviceInfo {
  std::string name;
  int device = 0;
  int compute_major = 0;
  int compute_minor = 0;
  size_t total_memory = 0;
  int shared_mem_per_block = 0;
  int runtime_version = 0;
  int driver_version = 0;
};

// Selects and initializes the CUDA device. Returns false and fills `error` on
// failure (e.g. no device present, bad index).
bool InitializeGpu(int device_index, GpuDeviceInfo* info, std::string* error);

// Runs every operation in `ops` over `image` on the GPU and appends one
// OpResult per operation to `results` (in the same order as `ops`). `stats`
// receives the gray-level statistics of the input image. Device buffers are
// reused across calls, so batching many images allocates at most once (when
// the largest image is first seen).
bool ProcessImageGpu(const RgbaImage& image, const std::vector<Operation>& ops,
                     const PipelineParams& params,
                     std::vector<OpResult>* results, GrayStats* stats,
                     std::string* error);

}  // namespace imgproc

#endif  // CUDA_BATCH_IMAGE_PROCESSOR_SRC_GPU_PIPELINE_H_
