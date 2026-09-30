# Proof-of-execution artifacts

This directory holds the evidence that the code runs on a GPU with a large
amount of data. It is (re)populated by `run_lab.sh` on a CUDA machine:

- `run_log_small_<timestamp>.txt` - full log of the "hundreds of small
  images" run: environment info (GPU model, CUDA versions), per-image
  processing times, per-image gray-level statistics, per-operation GPU kernel
  timings, throughput summary, and the GPU-vs-CPU verification results.
- `run_log_large_<timestamp>.txt` - same for the "tens of large images" run
  (1536x1536 and 2048x2048 images).
- `samples/` - selected before/after image pairs, named
  `before_<stem>.png` (the generated input) and
  `after_<stem>_<operation>.png` (the GPU-processed output), so a reviewer
  can see the effect of each operation at a glance.

The processed output directory itself (`output/`) is usually too large to
commit in full; the logs plus the before/after samples are the evidence
artifacts. See README.md for details.
