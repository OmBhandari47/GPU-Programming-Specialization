# GPU Programming Specialization

My coursework for the GPU Programming specialization on Coursera,
organized by course. Each course folder holds its notebooks, labs, and the
independent course project.

| Course | Folder | Status |
|---|---|---|
| 1. Introduction to Concurrent Programming with GPUs | [Course 1 - Introduction to Concurrent Programming with GPUs](Course%201%20-%20Introduction%20to%20Concurrent%20Programming%20with%20GPUs) | in progress |
| 2. Introduction to Parallel Programming with CUDA | [Course 2 - Introduction to Parallel Programming with CUDA](Course%202%20-%20Introduction%20to%20Parallel%20Programming%20with%20CUDA) | in progress |
| 3. CUDA at Scale for the Enterprise | [Course 3 - CUDA at Scale for the Enterprise](Course%203%20-%20CUDA%20at%20Scale%20for%20the%20Enterprise) | **independent project completed** |
| 4. CUDA Advanced Libraries | [Course 4 - CUDA Advanced Libraries](Course%204%20-%20CUDA%20Advanced%20Libraries) | in progress |

## Course 3 independent project

**[CUDA Batch Image Processor](Course%203%20-%20CUDA%20at%20Scale%20for%20the%20Enterprise/CUDA%20at%20Scale%20Independent%20Project)** —
a command-line tool that batch-processes hundreds of images on the GPU with
six hand-written CUDA kernels (grayscale, invert, brightness/contrast,
shared-memory tiled Gaussian blur, Sobel edge detection, unsharp-mask
sharpening) plus a 256-bin histogram kernel, with a CPU reference
implementation and `--verify` mode for GPU-vs-CPU validation. See the
project's own [README.md](Course%203%20-%20CUDA%20at%20Scale%20for%20the%20Enterprise/CUDA%20at%20Scale%20Independent%20Project/README.md)
for build and usage instructions.
