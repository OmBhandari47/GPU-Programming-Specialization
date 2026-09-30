# Capstone Presentation — Speaker Notes & Recording Guide

`capstone_slides.pptx` contains 12 slides (~8 minutes at the pace below).
Every slide also has its script in the PowerPoint notes pane, so you can
record directly from PowerPoint's presenter view.

## How to record and publish (30–40 minutes total)

1. **Fill the one placeholder first**: after `bash run_lab.sh` in the CUDA
   lab, open slide 9 and replace the `[fill from lab]` cells with the GPU
   numbers from `results/benchmark_small_*.csv` (or quote the
   "GPU kernel time by operation" block from the run log). Also swap the two
   demo output images on slide 3 for GPU-run samples if you like.
2. **Record** (any of these):
   - PowerPoint: *Slide Show → Record → From Beginning* (records narration +
     timings; then *File → Export → Create a Video*).
   - Zoom/Meet: share the slideshow and record.
   - OBS Studio (free): capture screen + mic.
3. Keep it **5–10 minutes** — the script below is ~8 minutes at a normal pace.
4. **Upload**: YouTube → *Upload video* → visibility **Unlisted** (anyone
   with the link can view; not searchable). Test the link in an incognito
   window. (Google Drive "Anyone with the link" also works.)
5. Paste the video URL into the *Project Presentation/Demonstration* field.

## Full script (with timings)

| # | Slide | Time | Script |
|---|---|---|---|
| 1 | Title | 0:00–0:30 | Hi, I'm Om Bhandari, and this is my capstone project for the GPU Programming specialization: a CUDA batch image processor. It processes hundreds of images with seven different image-processing operations, all implemented as hand-written CUDA kernels. In the next few minutes I'll show you the architecture, the two most interesting kernels, how I verify GPU correctness against a CPU reference, and how the whole thing runs at scale. |
| 2 | Why this project | 0:30–1:15 | Why did I build this? My goal was to learn to write real CUDA, not just call libraries. Every operation here is a hand-written kernel — no NPP anywhere. Image processing is a perfect domain: embarrassingly parallel, performance is easy to reason about, and correctness can be shown both visually and numerically. The project covers both required data shapes — hundreds of small inputs and tens of large inputs — and verification is a first-class feature. |
| 3 | What the tool does | 1:15–2:00 | The tool is a command-line program: point it at a directory and it runs seven operations on every image — grayscale, inversion, brightness/contrast, Gaussian blur, Sobel edges, unsharp-mask sharpening, and, new for the capstone, histogram equalization. Every image also gets a 256-bin histogram with statistics. It writes one PNG per operation plus a detailed log with CUDA-event kernel timings, and about twenty flags control everything. On the right you can see an input and three outputs. |
| 4 | Architecture | 2:00–2:50 | The architecture: main.cpp handles CLI, batching and logging. gpu_pipeline.cu is the only file that talks to the CUDA runtime. All nine kernels live in one self-contained header. Two design decisions to call out: the GPU pipeline exposes a CUDA-free interface, so everything else compiles with a plain C++ compiler — that also gives me a debug build without a GPU; and device buffers are allocated once and reused across the whole batch. |
| 5 | The nine kernels | 2:50–3:35 | A quick gallery: point operations map one thread to one pixel. The Gaussian blur is the most interesting — I'll zoom in next. Sobel is a 3×3 gradient with edge-replicated borders. Sharpening reuses the blur. The histogram kernel uses a grid-stride loop with atomics, and equalization is a table-lookup kernel driven by that histogram. Block size is a tunable, and all kernels handle arbitrary image dimensions. |
| 6 | Blur deep dive | 3:35–4:50 | The blur: a 2-D Gaussian is separable, so it's a horizontal then a vertical pass. Each block stages a tile plus halo in shared memory, so every input byte is read from global memory exactly once per pass. The intermediate is float4 to avoid rounding loss, and the radius is capped at 15 so shared memory always fits the 48-kilobyte default. Rather than guessing the block size, I benchmark it — bench mode sweeps tiles 8, 16, 32 into a CSV. |
| 7 | Equalization deep dive | 4:50–6:05 | New for the capstone: histogram equalization. The GPU computes the histogram with atomics; the bins are downloaded; the host builds the CDF and a 256-entry lookup table; the table goes back up and a gather kernel applies it. The LUT is built by the exact same host function as the CPU reference, so the GPU result is bit-exact — validated against an independent NumPy implementation with zero difference. The visual effect: on this low-contrast plasma pattern, the pixel standard deviation jumps from 16.5 to 73.6. |
| 8 | Verification | 6:05–7:00 | How do I know it's right? Three levels. One: a verify flag recomputes every operation on the CPU and reports the max pixel difference, with tolerances that allow only FMA-contraction differences. Two: the CPU reference itself was validated against NumPy. Three: because my dev machine has no NVIDIA driver, a test transcribes the kernels' exact indexing into host loops — all eight configurations match bit-exactly. The subtle part: the device rounds ties-to-even, so the host uses lrintf to match. |
| 9 | Benchmarks | 7:00–7:45 | The benchmarking subsystem: bench mode writes a CSV with GPU and CPU times side by side per image and operation, plus a blur tile-size sweep. The table shows my single-threaded CPU reference numbers — the GPU column comes from the lab run's benchmark CSV, which is committed with the results. |
| 10 | Running at scale | 7:45–8:20 | At scale: 256 small images — 37 megapixels — and 24 large images up to 2048×2048 — another 79 megapixels — all from a seeded, reproducible generator. One command, run_lab.sh, builds, pre-flights, processes both datasets with verification, benchmarks, and collects every artifact into results/, committed to the repo. |
| 11 | Lessons learned | 8:20–9:10 | Lessons: match rounding semantics exactly; verify with tolerances because FMA contraction reorders math; shared-memory tiling removes the radius-fold global-memory amplification; log both event time and wall time; pin your build environment — bleeding-edge glibc declares sinpi/cospi in a way that clashes with CUDA headers; and design for testability — the CUDA-free interface is what made the emulation test possible. |
| 12 | Demo & close | 9:10–9:45 | To close, the demo: one command processes the whole small dataset with verification — the log ends with "ALL OPERATIONS PASS" — and bench mode produces the CSV and tile sweep. Everything — code, datasets, logs, and these slides — is in my GitHub repository. Thank you for watching! |

*(If you prefer a tighter ~6-minute video, drop slides 5 and 10 and compress
the timings — the 5–10 minute window is satisfied either way.)*

## Optional on-camera live demo (instead of slide 12 screenshots)

1. `./bin/generate_dataset --out data/demo --count 8 --seed 1 --quiet`
2. `./bin/imgproc -i data/demo -o output/demo --verify` — show the
   verification table lines and `ALL OPERATIONS PASS`
3. Open a `before_`/`after_` pair side by side
4. `./bin/imgproc -i data/small -o output/bench -op blur --bench` then show
   `benchmark.csv` rows and the tile-sweep averages
