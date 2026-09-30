# CUDA Batch Image Processor

A command-line tool that batch-processes **large numbers of images on the GPU
with custom CUDA kernels**: grayscale conversion, color inversion,
brightness/contrast adjustment, tiled shared-memory Gaussian blur, Sobel edge
detection, unsharp-mask sharpening, and 256-bin gray-level histograms.

The project demonstrates both data-volume shapes required by the assignment:

| Dataset | Images | Sizes | Total pixels |
|---|---|---|---|
| `data/small` ("hundreds of small inputs") | 256 | 256×256 … 512×512 | 37.0 Mpx |
| `data/large` ("tens of large inputs") | 24 | 1536×1536 / 2048×2048 | 78.6 Mpx |

All GPU work is done by **hand-written CUDA kernels** (`src/kernels.cuh`) —
no NPP dependency, so the project builds with nothing but `nvcc` and a C++17
host compiler. A CPU reference implementation (`src/cpu_reference.cpp`) is
used by `--verify` to validate every GPU result against the host version.

## Operations

| Operation | Description | Kernel design |
|---|---|---|
| `gray` | RGBA → 8-bit grayscale (BT.601 luma) | one thread per pixel |
| `invert` | channel-wise color inversion | one thread per pixel |
| `brighten` | `(v-128)*contrast + 128 + delta` per channel | one thread per pixel |
| `blur` | separable Gaussian, float4 intermediate | **shared-memory tiles with halo regions**, horizontal + vertical pass |
| `sobel` | 3×3 Sobel gradient magnitude | one thread per pixel, edge-replicated borders |
| `sharpen` | unsharp mask: `in + amount*(in - blur(in))` | reuses the blur kernels + a blend kernel |
| *(always)* | 256-bin gray histogram + min/max/mean | grid-stride loop with `atomicAdd` |

## Repository layout

```
├── Makefile                  build file (GPU, CPU-debug and tools targets)
├── run_lab.sh                one-shot end-to-end run for the CUDA lab
├── src/
│   ├── main.cpp              command line parsing + batch orchestration + logging
│   ├── common.h/.cpp         shared types, op names, Gaussian kernel, histogram stats
│   ├── image_io.h/.cpp       image load/save + directory listing (stb wrappers)
│   ├── cpu_reference.h/.cpp  CPU reference implementation of every operation
│   ├── gpu_pipeline.h        CUDA-free interface of the GPU pipeline
│   ├── gpu_pipeline.cu       device memory management, kernel launches, CUDA-event timing
│   ├── gpu_stub.cpp          stub so the CUDA-free debug build links
│   ├── kernels.cuh           all CUDA kernels (documented, self-contained)
│   └── third_party/          vendored stb_image / stb_image_write (public domain)
├── tools/                   dataset generator + CPU kernel-emulation test
├── presentation/           capstone slides (PPTX) + speaker notes / script
├── data/                     sample (16), small (256) and large (24) datasets
├── results/                  proof-of-execution artifacts (logs, before/after images)
└── PROJECT_DESCRIPTION.md    write-up for the course submission
```

## Requirements

- NVIDIA GPU + CUDA Toolkit **11.5 or newer** (`nvcc` in `PATH`; the default
  `-arch=native` needs ≥ 11.5, otherwise override `ARCH`, see below)
- GNU `make` and a C++17 host compiler (g++ ≥ 7 / clang ≥ 7)
- No other dependencies — image I/O is vendored in `src/third_party/`

## Building

```bash
make                # GPU binary bin/imgproc + dataset generator bin/generate_dataset
make cpu            # CUDA-free debug binary bin/imgproc_cpu (--cpu-only mode)
make tools          # dataset generator only
make test-kernels   # CPU emulation test of the CUDA kernels' exact indexing
make smoke          # test-kernels + 16 generated images through the CPU reference
make clean
```

Useful variables:

```bash
make ARCH=sm_70                   # target a specific architecture (sm_70/75/80/86/89/90)
make EXTRA_NVCCFLAGS="--allow-unsupported-compiler"   # bleeding-edge host compiler
```

Manual build without make (e.g. in a notebook cell):

```bash
nvcc -O3 -std=c++17 -arch=native -Isrc \
  src/gpu_pipeline.cu src/common.cpp src/image_io.cpp src/cpu_reference.cpp src/main.cpp \
  -o bin/imgproc
```

**Note:** the `imgproc_cpu` binary is a *debug aid only* — it runs the CPU
reference so the pipeline can be exercised on machines without a GPU. It does
**not** satisfy the assignment's GPU requirement; `bin/imgproc` (built with
`nvcc`, running the CUDA kernels) is the deliverable.

## Generating datasets

```bash
./bin/generate_dataset --out data/small --preset small --count 256 --seed 42
./bin/generate_dataset --out data/large --preset large --count 24  --seed 43
```

The generator is fully deterministic (seeded), needs no internet access, and
writes a `manifest.txt` describing every image. Seven pattern families are
cycled: gradient, rings, checkerboard, rectangles, plasma, noise, rays — with
palettes constrained to be high-contrast in *luma* so patterns remain visible
after grayscale conversion.

## Usage

```
imgproc -i <input_dir> [options]

  -i,  --input <dir>      input directory (required)
  -o,  --output <dir>     output directory (default: <input>_processed)
  -op, --operation <list> gray,invert,brighten,blur,sobel,sharpen,equalize
                          or all
  -s,  --sigma <float>    Gaussian sigma for blur/sharpen (default 1.0)
  -br, --brightness <int> brightness delta (default +25)
  -c,  --contrast <float> contrast factor (default 1.0)
  -a,  --amount <float>   sharpen amount (default 0.8)
  -t,  --tile <int>       CUDA block edge length 4..32 (default 16)
  -d,  --device <int>     CUDA device index (default 0)
  -n,  --limit <int>      process only the first N images
  -l,  --log <file>       log file (default <output>/run_log.txt)
  -b,  --bench            write <output>/benchmark.csv (per-image per-op
                          GPU/CPU timings) and sweep blur tile sizes 8/16/32
  -v,  --verify           verify GPU results against the CPU reference
  -q,  --quiet            do not mirror the log to stdout
```

Examples:

```bash
# hundreds of small images, all operations, GPU-vs-CPU verification
./bin/imgproc -i data/small -o output/small_gpu --verify

# tens of large images, a subset of operations, larger blur radius
./bin/imgproc -i data/large -o output/large_gpu -op gray,blur,equalize -s 1.5 --verify

# benchmark: per-operation CSV timings + blur tile-size sweep
./bin/imgproc -i data/small -o output/bench -op blur,equalize --bench

# quick 20-image test on device 1
./bin/imgproc -i data/small -o output/quick -n 20 -d 1
```

Outputs are written as `<original-stem>_<operation>.png`; grayscale-producing
operations write single-channel PNGs.

## Example log output

Every run writes a detailed log (also mirrored to stdout). Format (from the
CPU reference mode; in GPU mode each operation line additionally reports the
CUDA-event kernel time and `--verify` appends per-op differences):

```
================================================================================
CUDA Batch Image Processor - run log
Started : 2026-09-30 11:23:11
Host    : Linux 6.1.158+ (x86_64)
Input   : data/small
Output  : output/small_gpu
Params  : sigma=1.00 tile=16 brightness=+25 contrast=1.00 sharpen=0.80
Mode    : GPU - NVIDIA Tesla T4 (compute capability 7.5, 14980 MiB ...)
CUDA    : runtime 12.4, driver 12.4
Dataset : 256 images (png/jpg/jpeg/bmp/tga), sorted by name
--------------------------------------------------------------------------------
[  1/256] img_0000_gradient_256x256.png (256x256, 65536 px) wall=1.83 ms
    gray stats : min=91 max=212 mean=151.13
    gray      :    0.112 ms (GPU)  verify max|diff|=0
    invert    :    0.098 ms (GPU)  verify max|diff|=0
    ...
--------------------------------------------------------------------------------
Summary
    images processed : 256 (0 failed to load)
    input pixels    : 36962304 (37.0 Mpx)
    output files    : 1536 written to output/small_gpu
    total wall time : 8.212 s (includes --verify CPU reference computation)
    throughput      : 31.14 images/s | 4.50 Mpx/s
    GPU kernel time by operation:
        gray      :    0.029 s total (0.113 ms/image, 256 images)
        ...
Verification (GPU results vs CPU reference):
        gray      : 256 images, max|diff|=0 (tolerance 1) ... PASS
        ...
    ALL OPERATIONS PASS
================================================================================
```

*(The GPU numbers above illustrate the log format; the real logs produced by
the lab run are committed under `results/`.)*

## One-shot lab run

`run_lab.sh` builds everything, generates both datasets (skipped if already
present), runs the GPU pipeline with `--verify` on both, tees the logs into
`results/`, and copies before/after sample pairs into `results/samples/`:

```bash
bash run_lab.sh                 # full run
SKIP_LARGE=1 bash run_lab.sh    # skip the large-image run
```

### Suggested lab / publishing workflow

1. Push this repository to GitHub/GitLab (it already contains both datasets,
   so the lab needs no internet):
   ```bash
   git remote add origin <your-repo-url>
   git push -u origin main
   ```
2. In the CUDA lab, clone the repository (or upload the zip) and run
   `bash run_lab.sh`.
3. Commit the artifacts back and push, so the repository carries the proof of
   execution out of the lab:
   ```bash
   git add results/ && git commit -m "Add GPU run logs and before/after samples" && git push
   ```
   The processed `output/` directory is intentionally gitignored (it is
   large); the committed `results/` logs and samples are the evidence
   artifacts.

## Verification methodology

Correctness is checked at three levels:

1. **CPU reference vs. GPU (`--verify`)**: every operation is recomputed on
   the host and the maximum absolute pixel difference per operation is
   reported. The reference uses formulas identical to the kernels — including
   round-to-nearest-even semantics (`lrintf` on the host matches
   `__float2int_rn` on the device). Differences of 0–3 are expected due to
   floating-point contraction (FMA) reordering on the GPU; anything larger
   is reported as a failure.
2. **Independent implementation**: during development the CPU reference was
   validated against an independent NumPy implementation (blur within ±1
   LSB, Sobel bit-exact).
3. **Kernel indexing emulation (`make test-kernels`)**: the kernels' exact
   thread-to-pixel mapping, shared-memory tile layout, halo geometry and
   border clamping are transcribed into host loops and compared with the
   CPU reference over eight dimension/sigma/tile configurations (including
   images smaller than one block and the radius-15 clamp). This validates
   the kernel logic on machines without an NVIDIA GPU; all configurations
   match bit-exactly.

## Code style

The code follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html):
2-space indentation, 80-column limit, `PascalCase` functions, `snake_case`
variables, header guards named after the file path, no exceptions or
RTTI. `.clang-format` (based on `Google`) is included; run
`clang-format -i src/*.cpp src/*.h src/*.cuh tools/*.cpp` to reformat.

## Troubleshooting

- **`-arch=native` not supported** (CUDA < 11.5): `make ARCH=sm_XX`.
- **"unsupported GNU version"** from `nvcc`: your host compiler is newer than
  your CUDA toolkit. Either update the toolkit or
  `make EXTRA_NVCCFLAGS="--allow-unsupported-compiler"`.
- **No GPU visible**: check `nvidia-smi`; in containers make sure the GPU and
  the NVIDIA driver (including `libcuda.so`) are exposed.
- **`imgproc_cpu` prints "built WITHOUT CUDA support"**: expected — that
  binary is the CUDA-free debug build; use `bin/imgproc` (or pass
  `--cpu-only` explicitly for a CPU smoke test).

## License

MIT (see `LICENSE`). The vendored stb libraries are public domain / MIT
dual-licensed (see `src/third_party/README.md`).
