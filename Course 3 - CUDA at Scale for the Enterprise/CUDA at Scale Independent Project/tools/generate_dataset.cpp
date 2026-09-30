// Deterministic synthetic image dataset generator for the CUDA batch image
// processor. Produces hundreds of small images (default "small" preset) or
// tens of large images ("large" preset) with a variety of patterns, so the
// project can demonstrate both "hundreds of small inputs" and "tens of large
// inputs" without depending on an internet connection in the lab.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <system_error>
#include <vector>

#include "common.h"
#include "image_io.h"

namespace imgproc {

namespace {

namespace fs = std::filesystem;

constexpr float kPi = 3.14159265358979323846f;

int RandByte(std::mt19937* rng) {
  std::uniform_int_distribution<int> dist(0, 255);
  return dist(*rng);
}

int RandRange(std::mt19937* rng, int lo, int hi) {
  std::uniform_int_distribution<int> dist(lo, hi);
  return dist(*rng);
}

float Rand01(std::mt19937* rng) {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  return dist(*rng);
}

int ClampChannel(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

// Approximate BT.601 luma of a color, used to keep generated palettes
// high-contrast even after grayscale conversion (Sobel operates on luma).
int LumaOf(const int color[3]) {
  return (299 * color[0] + 587 * color[1] + 114 * color[2]) / 1000;
}

// Fills `colors` (a count x 3 array) with random colors whose luma values
// differ pairwise by at least `min_luma_distance`, so the resulting patterns
// stay high-contrast in grayscale. Re-rolls a limited number of times.
void GenerateContrastingColors(std::mt19937* rng, int count,
                               int min_luma_distance, int colors[][3]) {
  for (int attempt = 0; attempt < 64; ++attempt) {
    for (int i = 0; i < count; ++i) {
      for (int c = 0; c < 3; ++c) {
        colors[i][c] = RandByte(rng);
      }
    }
    bool contrast_ok = true;
    for (int i = 0; i < count && contrast_ok; ++i) {
      for (int j = i + 1; j < count; ++j) {
        if (std::abs(LumaOf(colors[i]) - LumaOf(colors[j])) <
            min_luma_distance) {
          contrast_ok = false;
          break;
        }
      }
    }
    if (contrast_ok) {
      return;
    }
  }
}

void SetPixel(RgbaImage* image, int x, int y, int r, int g, int b) {
  const size_t offset = (static_cast<size_t>(y) * image->width + x) * 4;
  image->pixels[offset + 0] = static_cast<unsigned char>(ClampChannel(r));
  image->pixels[offset + 1] = static_cast<unsigned char>(ClampChannel(g));
  image->pixels[offset + 2] = static_cast<unsigned char>(ClampChannel(b));
  image->pixels[offset + 3] = 255;
}

void DrawGradient(RgbaImage* image, std::mt19937* rng) {
  int palette[2][3];
  GenerateContrastingColors(rng, 2, 60, palette);
  const int r0 = palette[0][0], g0 = palette[0][1], b0 = palette[0][2];
  const int r1 = palette[1][0], g1 = palette[1][1], b1 = palette[1][2];
  const float fx = static_cast<float>(std::max(1, image->width - 1));
  const float fy = static_cast<float>(std::max(1, image->height - 1));
  for (int y = 0; y < image->height; ++y) {
    for (int x = 0; x < image->width; ++x) {
      const float t = 0.5f * (x / fx + y / fy);
      SetPixel(image, x, y, static_cast<int>(r0 + (r1 - r0) * t + 0.5f),
               static_cast<int>(g0 + (g1 - g0) * t + 0.5f),
               static_cast<int>(b0 + (b1 - b0) * t + 0.5f));
    }
  }
}

void DrawRings(RgbaImage* image, std::mt19937* rng) {
  const float cx = image->width * (0.30f + 0.40f * Rand01(rng));
  const float cy = image->height * (0.30f + 0.40f * Rand01(rng));
  const float spacing = 12.0f + 36.0f * Rand01(rng);
  int colors[3][3];
  GenerateContrastingColors(rng, 3, 50, colors);
  for (int y = 0; y < image->height; ++y) {
    for (int x = 0; x < image->width; ++x) {
      const float dx = static_cast<float>(x) - cx;
      const float dy = static_cast<float>(y) - cy;
      const int band = static_cast<int>(std::hypot(dx, dy) / spacing) % 3;
      SetPixel(image, x, y, colors[band][0], colors[band][1], colors[band][2]);
    }
  }
}

void DrawChecker(RgbaImage* image, std::mt19937* rng) {
  const int cell = 8 << (RandByte(rng) % 3);  // 8, 16 or 32 pixels.
  int palette[2][3];
  GenerateContrastingColors(rng, 2, 70, palette);
  const int r0 = palette[0][0], g0 = palette[0][1], b0 = palette[0][2];
  const int r1 = palette[1][0], g1 = palette[1][1], b1 = palette[1][2];
  for (int y = 0; y < image->height; ++y) {
    for (int x = 0; x < image->width; ++x) {
      const int band = ((x / cell) + (y / cell)) & 1;
      SetPixel(image, x, y, band == 0 ? r0 : r1, band == 0 ? g0 : g1,
               band == 0 ? b0 : b1);
    }
  }
}

void DrawRects(RgbaImage* image, std::mt19937* rng) {
  int base[3] = {RandByte(rng), RandByte(rng), RandByte(rng)};
  for (int y = 0; y < image->height; ++y) {
    for (int x = 0; x < image->width; ++x) {
      SetPixel(image, x, y, base[0], base[1], base[2]);
    }
  }
  const int count = RandRange(rng, 8, 24);
  const int w = image->width;
  const int h = image->height;
  for (int i = 0; i < count; ++i) {
    const int x0 = RandRange(rng, 0, w - 1);
    const int y0 = RandRange(rng, 0, h - 1);
    const int rw = RandRange(rng, 16, std::max(17, w / 4));
    const int rh = RandRange(rng, 16, std::max(17, h / 4));
    // Re-roll until the rectangle's luma differs from the background.
    int color[3];
    for (int attempt = 0; attempt < 32; ++attempt) {
      color[0] = RandByte(rng);
      color[1] = RandByte(rng);
      color[2] = RandByte(rng);
      if (std::abs(LumaOf(color) - LumaOf(base)) >= 50) {
        break;
      }
    }
    for (int y = y0; y < std::min(h, y0 + rh); ++y) {
      for (int x = x0; x < std::min(w, x0 + rw); ++x) {
        SetPixel(image, x, y, color[0], color[1], color[2]);
      }
    }
  }
}

void DrawPlasma(RgbaImage* image, std::mt19937* rng) {
  const float a = 0.004f + 0.020f * Rand01(rng);
  const float b = 0.004f + 0.020f * Rand01(rng);
  const float c = 0.002f + 0.010f * Rand01(rng);
  const float d = 0.010f + 0.050f * Rand01(rng);
  const float cx = image->width * 0.5f;
  const float cy = image->height * 0.5f;
  for (int y = 0; y < image->height; ++y) {
    for (int x = 0; x < image->width; ++x) {
      const float dx = static_cast<float>(x) - cx;
      const float dy = static_cast<float>(y) - cy;
      const float v = std::sin(x * a) + std::sin(y * b) +
                      std::sin((x + y) * c) + std::sin(std::hypot(dx, dy) * d);
      SetPixel(image, x, y, static_cast<int>(128 + 35.0f * v + 0.5f),
               static_cast<int>(128 + 20.0f * v + 0.5f),
               static_cast<int>(128 - 30.0f * v + 0.5f));
    }
  }
}

// Blocky uniform noise: 2x2 blocks compress far better in PNG than per-pixel
// noise while still stressing the kernels with high-frequency content.
void DrawNoise(RgbaImage* image, std::mt19937* rng) {
  constexpr int kBlock = 2;
  const int w = image->width;
  const int h = image->height;
  for (int by = 0; by < h; by += kBlock) {
    for (int bx = 0; bx < w; bx += kBlock) {
      const int r = RandByte(rng), g = RandByte(rng), b = RandByte(rng);
      for (int y = by; y < std::min(h, by + kBlock); ++y) {
        for (int x = bx; x < std::min(w, bx + kBlock); ++x) {
          SetPixel(image, x, y, r, g, b);
        }
      }
    }
  }
}

void DrawRays(RgbaImage* image, std::mt19937* rng) {
  const float cx = image->width * (0.40f + 0.20f * Rand01(rng));
  const float cy = image->height * (0.40f + 0.20f * Rand01(rng));
  const int bands = 6 + RandByte(rng) % 12;
  int palette[2][3];
  GenerateContrastingColors(rng, 2, 60, palette);
  for (int y = 0; y < image->height; ++y) {
    for (int x = 0; x < image->width; ++x) {
      const float angle = std::atan2(static_cast<float>(y) - cy,
                                     static_cast<float>(x) - cx) +
                          kPi;  // Map to [0, 2*pi).
      const int band = static_cast<int>(angle / (2.0f * kPi) * bands * 2) % 2;
      SetPixel(image, x, y, palette[band][0], palette[band][1],
               palette[band][2]);
    }
  }
}

struct Pattern {
  const char* name;
  void (*draw)(RgbaImage* image, std::mt19937* rng);
};

const Pattern kPatterns[] = {
    {"gradient", DrawGradient}, {"rings", DrawRings},
    {"checker", DrawChecker},   {"rects", DrawRects},
    {"plasma", DrawPlasma},     {"noise", DrawNoise},
    {"rays", DrawRays},
};
constexpr size_t kPatternCount = sizeof(kPatterns) / sizeof(kPatterns[0]);

const int kSmallSizes[] = {256, 320, 384, 512};
const int kLargeSizes[] = {1536, 2048};

struct GeneratorOptions {
  std::string out_dir;
  int count = 0;  // 0 = preset default.
  std::string preset = "small";
  unsigned seed = 42;
  bool quiet = false;
};

const char kUsageText[] =
    "generate_dataset - deterministic synthetic image dataset generator\n"
    "\n"
    "Usage:\n"
    "  generate_dataset --out <dir> [options]\n"
    "\n"
    "Options:\n"
    "  -o, --out <dir>       Output directory (required)\n"
    "  -n, --count <int>     Number of images (default: 256 small / 24 large)\n"
    "  -p, --preset <name>   'small' (256..512 px) or 'large' (1536..2048 px)\n"
    "  -s, --seed <int>      Random seed (default 42; runs are reproducible)\n"
    "  -q, --quiet           Only print the final summary\n"
    "  -h, --help            Show this help text\n";

void PrintUsage() { std::fputs(kUsageText, stdout); }

int RunGenerator(int argc, char** argv) {
  GeneratorOptions opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    std::string key = arg;
    std::string inline_value;
    bool has_inline = false;
    const size_t eq = arg.find('=');
    if (eq != std::string::npos) {
      key = arg.substr(0, eq);
      inline_value = arg.substr(eq + 1);
      has_inline = true;
    }
    const auto value = [&](std::string* out) {
      if (has_inline) {
        *out = inline_value;
        return true;
      }
      if (i + 1 >= argc) {
        std::fprintf(stderr, "error: missing value for %s\n", key.c_str());
        return false;
      }
      *out = argv[++i];
      return true;
    };
    std::string v;
    if (key == "-h" || key == "--help") {
      PrintUsage();
      return 0;
    } else if (key == "-o" || key == "--out") {
      if (!value(&v)) return 1;
      opts.out_dir = v;
    } else if (key == "-n" || key == "--count") {
      if (!value(&v)) return 1;
      opts.count = std::atoi(v.c_str());
      if (opts.count <= 0) {
        std::fprintf(stderr, "error: --count must be positive\n");
        return 1;
      }
    } else if (key == "-p" || key == "--preset") {
      if (!value(&v)) return 1;
      if (v != "small" && v != "large") {
        std::fprintf(stderr, "error: --preset must be 'small' or 'large'\n");
        return 1;
      }
      opts.preset = v;
    } else if (key == "-s" || key == "--seed") {
      if (!value(&v)) return 1;
      opts.seed = static_cast<unsigned>(std::atoi(v.c_str()));
    } else if (key == "-q" || key == "--quiet") {
      opts.quiet = true;
    } else {
      std::fprintf(stderr, "error: unknown argument '%s'\n", arg.c_str());
      PrintUsage();
      return 1;
    }
  }
  if (opts.out_dir.empty()) {
    std::fprintf(stderr, "error: --out is required\n");
    PrintUsage();
    return 1;
  }

  const bool large = (opts.preset == "large");
  const int* sizes = large ? kLargeSizes : kSmallSizes;
  const size_t size_count = large ? sizeof(kLargeSizes) / sizeof(int)
                                  : sizeof(kSmallSizes) / sizeof(int);
  const int default_count = large ? 24 : 256;
  const int count = opts.count > 0 ? opts.count : default_count;

  if (!EnsureDirectory(opts.out_dir)) {
    std::fprintf(stderr, "error: cannot create directory '%s'\n",
                 opts.out_dir.c_str());
    return 1;
  }
  std::ofstream manifest(opts.out_dir + "/manifest.txt");
  if (!manifest) {
    std::fprintf(stderr, "error: cannot write manifest in '%s'\n",
                 opts.out_dir.c_str());
    return 1;
  }

  long long total_pixels = 0;
  long long total_bytes = 0;
  for (int i = 0; i < count; ++i) {
    const Pattern& pattern = kPatterns[static_cast<size_t>(i) % kPatternCount];
    const int size = sizes[static_cast<size_t>(i) % size_count];
    RgbaImage image;
    image.width = size;
    image.height = size;
    image.pixels.assign(static_cast<size_t>(size) * size * 4, 255);

    std::mt19937 rng(opts.seed * 1000003u + static_cast<unsigned>(i));
    pattern.draw(&image, &rng);

    char name[128];
    std::snprintf(name, sizeof(name), "img_%04d_%s_%dx%d.png", i, pattern.name,
                  size, size);
    const std::string path = opts.out_dir + "/" + name;
    if (!SaveRgbaPng(path, image.pixels.data(), image.width, image.height)) {
      std::fprintf(stderr, "error: failed to write '%s'\n", path.c_str());
      return 1;
    }
    manifest << name << " " << size << " " << size << " " << pattern.name
             << "\n";
    total_pixels += static_cast<long long>(size) * size;
    std::error_code ec;
    const auto file_size = fs::file_size(path, ec);
    if (!ec) {
      total_bytes += static_cast<long long>(file_size);
    }
    if (!opts.quiet && ((i + 1) % 32 == 0 || i + 1 == count)) {
      std::printf("[%4d/%4d] %s\n", i + 1, count, name);
    }
  }

  std::printf("wrote %d images (%lld pixels, %.1f Mpx, %.1f MiB) to %s\n",
              count, total_pixels, static_cast<double>(total_pixels) / 1e6,
              static_cast<double>(total_bytes) / (1024.0 * 1024.0),
              opts.out_dir.c_str());
  return 0;
}

}  // namespace

}  // namespace imgproc

int main(int argc, char** argv) { return imgproc::RunGenerator(argc, argv); }
