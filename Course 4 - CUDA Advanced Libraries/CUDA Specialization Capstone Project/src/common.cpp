#include "common.h"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace imgproc {

std::string OperationName(Operation op) {
  switch (op) {
    case Operation::kGray:
      return "gray";
    case Operation::kInvert:
      return "invert";
    case Operation::kBrighten:
      return "brighten";
    case Operation::kBlur:
      return "blur";
    case Operation::kSobel:
      return "sobel";
    case Operation::kSharpen:
      return "sharpen";
    case Operation::kEqualize:
      return "equalize";
  }
  return "unknown";
}

bool ParseOperationName(const std::string& name, Operation* op) {
  std::string lower;
  lower.reserve(name.size());
  for (char c : name) {
    lower.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  if (lower == "gray" || lower == "grayscale" || lower == "greyscale") {
    *op = Operation::kGray;
    return true;
  }
  if (lower == "invert" || lower == "negative") {
    *op = Operation::kInvert;
    return true;
  }
  if (lower == "brighten" || lower == "brightness") {
    *op = Operation::kBrighten;
    return true;
  }
  if (lower == "blur" || lower == "gaussian") {
    *op = Operation::kBlur;
    return true;
  }
  if (lower == "sobel" || lower == "edges") {
    *op = Operation::kSobel;
    return true;
  }
  if (lower == "sharpen" || lower == "unsharp") {
    *op = Operation::kSharpen;
    return true;
  }
  if (lower == "equalize" || lower == "histeq") {
    *op = Operation::kEqualize;
    return true;
  }
  return false;
}

std::vector<float> ComputeGaussianKernel(float sigma, int* radius) {
  const int kMaxRadius = 15;
  int r = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
  r = std::min(r, kMaxRadius);
  std::vector<float> weights(static_cast<size_t>(2) * r + 1);
  double sum = 0.0;
  for (int i = -r; i <= r; ++i) {
    const double w =
        std::exp(-(static_cast<double>(i) * i) / (2.0 * sigma * sigma));
    weights[static_cast<size_t>(i + r)] = static_cast<float>(w);
    sum += w;
  }
  for (float& w : weights) {
    w = static_cast<float>(w / sum);
  }
  *radius = r;
  return weights;
}

void BuildEqualizationLut(const int* histogram, long long pixel_count,
                          unsigned char lut[256]) {
  // Standard CDF-based formulation: lut[v] = round(255 * (cdf[v] - cdf_min) /
  // (N - cdf_min)), where cdf_min is the CDF at the first non-empty bin. A
  // degenerate image (single gray level) falls back to the identity table.
  long long cdf[256] = {};
  long long running = 0;
  long long cdf_min = -1;
  for (int v = 0; v < 256; ++v) {
    running += histogram[v];
    cdf[v] = running;
    if (cdf_min < 0 && histogram[v] > 0) {
      cdf_min = running;
    }
  }
  const long long denominator = pixel_count - cdf_min;
  if (denominator <= 0) {
    for (int v = 0; v < 256; ++v) {
      lut[v] = static_cast<unsigned char>(v);
    }
    return;
  }
  for (int v = 0; v < 256; ++v) {
    const float scaled = 255.0f * static_cast<float>(cdf[v] - cdf_min) /
                         static_cast<float>(denominator);
    long value = std::lrintf(scaled);
    value = value < 0 ? 0 : (value > 255 ? 255 : value);
    lut[v] = static_cast<unsigned char>(value);
  }
}

GrayStats StatsFromHistogram(const int* histogram) {
  GrayStats stats;
  long long total = 0;
  long long weighted = 0;
  for (int i = 0; i < 256; ++i) {
    stats.histogram[i] = histogram[i];
    total += histogram[i];
    weighted += static_cast<long long>(i) * histogram[i];
  }
  stats.min_value = 0;
  stats.max_value = 255;
  for (int i = 0; i < 256; ++i) {
    if (histogram[i] > 0) {
      stats.min_value = i;
      break;
    }
  }
  for (int i = 255; i >= 0; --i) {
    if (histogram[i] > 0) {
      stats.max_value = i;
      break;
    }
  }
  stats.mean_value =
      total > 0 ? static_cast<double>(weighted) / static_cast<double>(total)
                : 0.0;
  return stats;
}

}  // namespace imgproc
