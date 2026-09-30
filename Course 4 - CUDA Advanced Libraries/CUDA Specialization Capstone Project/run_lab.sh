#!/usr/bin/env bash
#
# One-shot end-to-end run for the CUDA lab environment:
#   1. builds the GPU binary and the dataset generator,
#   2. generates the datasets (256 small images, 24 large images),
#   3. runs the full GPU pipeline with --verify on both datasets,
#   4. runs a benchmark pass (per-op CSV timings + blur tile sweep),
#   5. collects proof-of-execution artifacts into results/.
#
# Usage:
#   bash run_lab.sh                  # full run (small + large datasets)
#   SKIP_LARGE=1 bash run_lab.sh     # skip the large-image run
#
set -euo pipefail
cd "$(dirname "$0")"

STAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="results"
SMALL_DATA="data/small"
LARGE_DATA="data/large"

echo "=============================================================="
echo " CUDA Batch Image Processor - lab run"
echo " $(date)"
echo "=============================================================="

echo
echo "== Environment =="
nvidia-smi || true
nvcc --version | tail -n 2 || true

echo
echo "== Building =="
make -j"$(nproc)" all

echo
echo "== Pre-flight: kernel-indexing emulation test (CPU) =="
make test-kernels 2>&1 | tee "$RESULTS_DIR/kernel_emulation_${STAMP}.txt"

echo
echo "== Generating datasets (deterministic, no internet needed) =="
if [ -z "$(ls -A "$SMALL_DATA" 2>/dev/null | grep -v '^manifest.txt$' || true)" ]; then
  ./bin/generate_dataset --out "$SMALL_DATA" --preset small --count 256 --seed 42
else
  echo "small dataset already present in $SMALL_DATA - skipping generation"
fi
if [ "${SKIP_LARGE:-0}" != "1" ]; then
  if [ -z "$(ls -A "$LARGE_DATA" 2>/dev/null | grep -v '^manifest.txt$' || true)" ]; then
    ./bin/generate_dataset --out "$LARGE_DATA" --preset large --count 24 --seed 43
  else
    echo "large dataset already present in $LARGE_DATA - skipping generation"
  fi
fi

echo
echo "== GPU run 1/2: hundreds of small images (all operations, --verify) =="
./bin/imgproc --input "$SMALL_DATA" --output output/small_gpu --verify \
  2>&1 | tee "$RESULTS_DIR/run_log_small_${STAMP}.txt"

if [ "${SKIP_LARGE:-0}" != "1" ]; then
  echo
  echo "== GPU run 2/2: tens of large images (gray, blur, sobel, --verify) =="
  ./bin/imgproc --input "$LARGE_DATA" --output output/large_gpu \
    --operation gray,blur,sobel --verify \
    2>&1 | tee "$RESULTS_DIR/run_log_large_${STAMP}.txt"
fi

echo
echo "== Collecting before/after samples =="
SAMPLE_DIR="$RESULTS_DIR/samples"
mkdir -p "$SAMPLE_DIR"
# Copy a few inputs and their processed outputs, named before_*/after_* so
# the pairing is obvious to a reviewer.
for path in $(ls "$SMALL_DATA"/*.png | sort | head -n 3); do
  stem="$(basename "$path" .png)"
  cp "$path" "$SAMPLE_DIR/before_${stem}.png"
  for op in gray invert brighten blur sobel sharpen; do
    if [ -f "output/small_gpu/${stem}_${op}.png" ]; then
      cp "output/small_gpu/${stem}_${op}.png" "$SAMPLE_DIR/after_${stem}_${op}.png"
    fi
  done
done

echo
echo "== Artifacts produced =="
ls -la "$RESULTS_DIR"
echo
ls -la "$SAMPLE_DIR"

echo
echo "Done. Proof-of-execution artifacts are in $RESULTS_DIR/."
echo "Next steps:"
echo "  git add results/ data/ && git commit -m 'Add lab run artifacts' && git push"
