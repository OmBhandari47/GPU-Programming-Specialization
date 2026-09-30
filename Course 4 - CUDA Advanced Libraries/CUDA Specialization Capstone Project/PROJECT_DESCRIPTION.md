# Capstone Code Project Description

**Project: CUDA Batch Image Processor — nine custom CUDA kernels over
hundreds of images (capstone extension of my Course 3 project)**

For my capstone I extended my Course 3 project — a command-line tool that
batch-processes large amounts of image data with hand-written CUDA kernels —
into a complete GPU image-processing study. The tool runs seven operations —
BT.601 grayscale, color inversion, brightness/contrast, Gaussian blur, Sobel
edge detection, unsharp-mask sharpening, and (new for the capstone)
histogram equalization — plus a 256-bin gray-level histogram with per-image
statistics, and logs CUDA-event kernel timings and throughput for every run.
Everything is driven by command-line flags, and it builds with plain `nvcc` +
`make` with no NPP: the only dependencies are the CUDA runtime and two
public-domain stb headers for image I/O.

The capstone additions were: (1) **histogram equalization** — the GPU
histogram kernel (grid-stride loop with `atomicAdd`) now feeds a CDF-based
256-entry lookup table built on the host by the *same* function the CPU
reference uses, then applied by a LUT-gather kernel, so the GPU result is
bit-exact; (2) a **benchmarking subsystem** (`--bench`) that writes a CSV
with per-image, per-operation GPU (CUDA-event) and CPU timings side by side,
and sweeps the blur tile size across 8/16/32 to quantify the shared-memory
tiling; and (3) a **project presentation** (slides + script in
`presentation/`).

The most interesting engineering remains the Gaussian blur: a separable
two-pass filter where each pass stages a 2-D tile plus halo region in shared
memory (every input byte is read from global memory only once per pass),
with a float4 intermediate to avoid rounding loss and a radius cap of 15 so
shared memory always fits the 48 KiB per-block default. Correctness is
checked at three levels: `--verify` recomputes every operation with the CPU
reference (tolerances of 1–3/255 allow only FMA-contraction differences,
with exactly matched ties-to-even rounding via `lrintf` vs
`__float2int_rn`); the reference itself was validated against an independent
NumPy implementation (blur ±1 LSB, Sobel and equalization bit-exact); and
`make test-kernels` transcribes the kernels' exact thread mapping, tile
layout and halo handling into host loops — all eight dimension/sigma/tile
configurations match bit-exactly, which let me validate kernel logic on a
machine without an NVIDIA driver.

The data side demonstrates both required volume shapes: a deterministic,
seeded generator produces 256 small images (256–512 px, 37 Mpx) and 24 large
images (1536–2048 px, 79 Mpx) from seven pattern families, with palettes
constrained to stay high-contrast in luma — a fix for a real bug where a
randomly generated "rings" image turned out nearly invisible in grayscale.
One command (`bash run_lab.sh`) builds everything, runs a kernel-emulation
pre-flight test, processes both datasets with verification, runs the
benchmark pass, and collects the logs, the benchmark CSV and before/after
sample pairs into `results/` as the proof-of-execution artifacts.

Lessons learned: rounding semantics must match exactly across CPU and GPU;
verify with tolerances rather than equality because FMA contraction reorders
floating-point math; shared-memory tiling removes the radius-fold global
memory amplification of naive convolution; GPU event time and wall time
answer different questions and both belong in the log; and build
environments should be pinned — bleeding-edge glibc declares `sinpi/cospi`
in a way that conflicts with CUDA's own math headers.
