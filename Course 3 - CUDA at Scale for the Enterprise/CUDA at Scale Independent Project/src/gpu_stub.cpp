// Stub implementation of the GPU pipeline used by the CUDA-free debug build
// (`make imgproc_cpu`). It exists so `main.cpp` links without a CUDA toolkit;
// the resulting binary can only run in --cpu-only mode.

#include "gpu_pipeline.h"

namespace imgproc {

bool InitializeGpu(int, GpuDeviceInfo*, std::string* error) {
  *error =
      "this binary was built WITHOUT CUDA support (imgproc_cpu debug build); "
      "rebuild with `make imgproc` (requires nvcc + an NVIDIA GPU)";
  return false;
}

bool ProcessImageGpu(const RgbaImage&, const std::vector<Operation>&,
                     const PipelineParams&, std::vector<OpResult>*, GrayStats*,
                     std::string* error) {
  *error =
      "this binary was built WITHOUT CUDA support (imgproc_cpu debug "
      "build); rebuild with `make imgproc` (requires nvcc + an NVIDIA "
      "GPU)";
  return false;
}

}  // namespace imgproc
