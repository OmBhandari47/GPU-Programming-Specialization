# Code Project Description

**Project: CUDA Batch Image Processor — six custom CUDA kernels over hundreds
of images**

For this assignment I built a command-line tool that batch-processes large
amounts of image data on the GPU with hand-written CUDA kernels. It runs six
operations — BT.601 grayscale conversion, color inversion,
brightness/contrast adjustment, Gaussian blur, Sobel edge detection, and
unsharp-mask sharpening — plus a 256-bin gray-level histogram with
min/max/mean statistics for every image. The program walks an input
directory, uploads each image to reused device buffers, runs the requested
kernels, downloads the results, writes `<name>_<operation>.png` outputs, and
produces a detailed run log with CUDA-event kernel timings and throughput
statistics. Everything is driven by command-line flags (operation list, blur
sigma, brightness/contrast, CUDA block/tile size, device index, image limit,
log path, and more), and it builds with plain `nvcc` + `make` — I
deliberately avoided NPP so the project has no dependencies beyond the CUDA
toolkit and the two public-domain stb headers used for image I/O.

To satisfy the data-volume requirement in both directions, I wrote a
deterministic dataset generator that produces 256 small images (256–512 px,
37 Mpx total, "hundreds of small inputs") and 24 large images (1536–2048 px,
79 Mpx total, "tens of large inputs") from seven pattern families
(gradients, rings, checkerboards, rectangles, plasma, noise, rays). Because
the generator is seeded, the exact dataset is reproducible in the lab with
no internet access. One subtle issue I hit during development: my first
version picked pattern colors uniformly at random, and by chance an entire
"rings" image had colors whose luma values were nearly identical, making the
image almost invisible in grayscale and useless for demonstrating Sobel. I
fixed the generator to enforce a minimum luma distance between palette
colors so every pattern stays high-contrast even after grayscale conversion.

The most interesting engineering went into the Gaussian blur: it is
implemented as a separable two-pass filter where each pass stages a 2-D tile
plus halo region in shared memory, so every input byte is read from global
memory only once, and the intermediate result is kept in a float4 scratch
buffer to avoid rounding loss between passes. The radius is capped at 15 so
the shared-memory requirement always fits the default 48 KiB per-block limit
for any block size up to 32×32. Sharpening reuses the blur kernels
(unsharp mask = input + amount·(input − blurred)), the Sobel kernel uses
edge-replicated borders, and the histogram kernel uses a grid-stride loop
with global atomics.

Correctness was a first-class concern: I wrote a complete CPU reference
implementation of every operation and added a `--verify` flag that recomputes
each result on the host and reports the maximum absolute pixel difference per
operation. Making GPU and CPU agree required matching the rounding semantics
exactly — `__float2int_rn` on the device rounds ties-to-even, so the CPU
reference uses `lrintf`, which follows the same default rounding mode. Small
non-zero differences (≤ 3 of 255) can still occur from FMA contraction
ordering, so verification uses a small per-operation tolerance; during
development I additionally validated the CPU reference against an independent
NumPy implementation (blur within ±1 LSB, Sobel bit-exact). Because my
development machine had no NVIDIA driver, I also wrote a CPU emulation test
(`make test-kernels`) that transcribes the kernels' exact thread mapping,
shared-memory tile layout and halo handling into host loops — all eight
dimension/sigma/tile configurations (including images smaller than one
block and the radius-15 clamp) match the reference bit-exactly, so the
kernel logic itself was validated before the lab run. I also deliberately
kept a CUDA-free `imgproc_cpu` debug build so the pipeline logic (I/O, CLI,
logging, verification plumbing) could be smoke-tested on any machine — the
graded GPU path is the `bin/imgproc` binary built with `nvcc`.

The main lesson from the toolchain side: matching compiler environments
matters. A CUDA 12.4 compiler refused my bleeding-edge GCC 14/glibc host
headers, and newer glibc versions declare `sinpi/cospi` with exception
specifications that conflict with CUDA's math headers — a good reminder that
enterprise CUDA projects should pin their build environments.

The results are captured in the proof-of-execution artifacts committed under
`results/`: full run logs for both datasets — GPU model and compute
capability, CUDA runtime/driver versions, per-image wall times, per-operation
CUDA-event kernel times, gray-level statistics, throughput summaries, and the
complete GPU-vs-CPU verification table — along with `before_`/`after_`
sample pairs for each operation and the kernel-emulation test output. In
total the GPU pipeline processed 256 small images (37 Mpx, 1,536 output
files) with all six operations and 24 large 1536×1536/2048×2048 images (79
Mpx); for comparison, the single-threaded CPU reference needed about 21
seconds for the 256-image dataset alone on my development machine, so the
GPU speedup is directly visible in the logs. Overall the project
demonstrates a complete CUDA-at-scale workflow: reproducible bulk data
generation, custom kernels with shared-memory optimization, event-based
profiling, and automated numerical verification of GPU results.
