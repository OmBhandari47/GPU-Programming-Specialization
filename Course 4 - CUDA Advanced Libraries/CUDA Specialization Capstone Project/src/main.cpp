// cuda-batch-image-processor: batch-processes a directory of images on the
// GPU with custom CUDA kernels, writing the processed images plus a detailed
// run log (timings, per-image statistics, optional CPU/GPU verification) and
// an optional per-operation benchmark CSV.

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/utsname.h>
#endif

#include "common.h"
#include "cpu_reference.h"
#include "gpu_pipeline.h"
#include "image_io.h"

namespace imgproc {

namespace fs = std::filesystem;

namespace {

const char kUsageText[] =
    "cuda-batch-image-processor\n"
    "Batch image processing on the GPU with custom CUDA kernels.\n"
    "\n"
    "Usage:\n"
    "  imgproc -i <input_dir> [options]\n"
    "\n"
    "Options:\n"
    "  -i,  --input <dir>      Directory with input images (required)\n"
    "  -o,  --output <dir>     Directory for processed images (default:\n"
    "                          <input>_processed)\n"
    "  -op, --operation <list> Comma separated ops or 'all' (default: all).\n"
    "                          Ops: gray,invert,brighten,blur,sobel,sharpen,\n"
    "                          equalize\n"
    "  -s,  --sigma <float>    Gaussian sigma for blur/sharpen (default 1.0,\n"
    "                          range 0.05..5.0)\n"
    "  -br, --brightness <int> Brightness delta for brighten (default 25)\n"
    "  -c,  --contrast <float> Contrast factor for brighten (default 1.0)\n"
    "  -a,  --amount <float>   Sharpen amount (default 0.8)\n"
    "  -t,  --tile <int>       CUDA block edge length 4..32 (default 16)\n"
    "  -d,  --device <int>     CUDA device index (default 0)\n"
    "  -n,  --limit <int>      Only process the first N images (default all)\n"
    "  -l,  --log <file>       Log file (default <output>/run_log.txt)\n"
    "  -b,  --bench            Write <output>/benchmark.csv with per-image\n"
    "                          per-operation GPU/CPU timings; in GPU mode\n"
    "                          also sweeps blur tile sizes 8/16/32\n"
    "  -v,  --verify           Verify GPU results against the CPU reference\n"
    "  -cpu, --cpu-only        Run the CPU reference only (debug aid; does\n"
    "                          NOT use the GPU and does not satisfy the\n"
    "                          assignment's GPU requirement)\n"
    "  -q,  --quiet            Do not mirror the log to stdout\n"
    "  -h,  --help             Show this help text\n"
    "\n"
    "Examples:\n"
    "  imgproc -i data/small -o output/small --verify\n"
    "  imgproc -i data/large -op gray,blur,equalize -s 1.5\n"
    "  imgproc -i data/small -o output/bench -op blur --bench\n"
    "  imgproc -i data/sample --cpu-only   # smoke test without a GPU\n";

struct Options {
  std::string input_dir;
  std::string output_dir;
  std::string log_path;
  std::vector<Operation> ops;  // Empty means "run all operations".
  PipelineParams params;
  bool verify = false;
  bool bench = false;
  bool cpu_only = false;
  bool quiet = false;
  int limit = 0;  // 0 = process every image.
};

void PrintUsage() { std::fputs(kUsageText, stdout); }

bool ParseIntValue(const std::string& text, int* out) {
  if (text.empty()) {
    return false;
  }
  char* end = nullptr;
  errno = 0;
  const long value = std::strtol(text.c_str(), &end, 10);
  if (errno != 0 || end == nullptr || *end != '\0' || value < INT_MIN ||
      value > INT_MAX) {
    return false;
  }
  *out = static_cast<int>(value);
  return true;
}

bool ParseFloatValue(const std::string& text, float* out) {
  if (text.empty()) {
    return false;
  }
  char* end = nullptr;
  errno = 0;
  const float value = std::strtof(text.c_str(), &end);
  if (errno != 0 || end == nullptr || *end != '\0') {
    return false;
  }
  *out = value;
  return true;
}

bool ParseOperationList(const std::string& value, std::vector<Operation>* ops,
                        std::string* error) {
  size_t start = 0;
  while (start < value.size()) {
    size_t end = value.find(',', start);
    if (end == std::string::npos) {
      end = value.size();
    }
    std::string token = value.substr(start, end - start);
    token.erase(0, token.find_first_not_of(" \t"));
    token.erase(token.find_last_not_of(" \t") + 1);
    if (token.empty()) {
      *error = "empty operation name in '" + value + "'";
      return false;
    }
    if (token == "all") {
      ops->clear();  // An empty list means "all operations".
      return true;
    }
    Operation op;
    if (!ParseOperationName(token, &op)) {
      *error = "unknown operation '" + token +
               "' (valid: gray,invert,brighten,blur,sobel,sharpen,equalize,"
               "all)";
      return false;
    }
    ops->push_back(op);
    start = end + 1;
  }
  return true;
}

bool ParseArgs(int argc, char** argv, Options* opts, std::string* error) {
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
        *error = "missing value for " + key;
        return false;
      }
      *out = argv[++i];
      return true;
    };
    const auto match = [&](const char* short_flag, const char* long_flag) {
      return key == short_flag || key == long_flag;
    };

    std::string v;
    if (match("-h", "--help")) {
      PrintUsage();
      std::exit(0);
    } else if (match("-i", "--input")) {
      if (!value(&v)) return false;
      opts->input_dir = v;
    } else if (match("-o", "--output")) {
      if (!value(&v)) return false;
      opts->output_dir = v;
    } else if (match("-op", "--operation")) {
      if (!value(&v)) return false;
      if (!ParseOperationList(v, &opts->ops, error)) return false;
    } else if (match("-s", "--sigma")) {
      if (!value(&v)) return false;
      float f = 0.0f;
      if (!ParseFloatValue(v, &f) || f < 0.05f || f > 5.0f) {
        *error = "sigma must be a number in [0.05, 5.0], got '" + v + "'";
        return false;
      }
      opts->params.sigma = f;
    } else if (match("-br", "--brightness")) {
      if (!value(&v)) return false;
      int n = 0;
      if (!ParseIntValue(v, &n) || n < -255 || n > 255) {
        *error =
            "brightness must be an integer in [-255, 255], got '" + v + "'";
        return false;
      }
      opts->params.brightness_delta = n;
    } else if (match("-c", "--contrast")) {
      if (!value(&v)) return false;
      float f = 0.0f;
      if (!ParseFloatValue(v, &f) || f < 0.1f || f > 4.0f) {
        *error = "contrast must be a number in [0.1, 4.0], got '" + v + "'";
        return false;
      }
      opts->params.contrast = f;
    } else if (match("-a", "--amount")) {
      if (!value(&v)) return false;
      float f = 0.0f;
      if (!ParseFloatValue(v, &f) || f < 0.0f || f > 4.0f) {
        *error = "amount must be a number in [0.0, 4.0], got '" + v + "'";
        return false;
      }
      opts->params.sharpen_amount = f;
    } else if (match("-t", "--tile")) {
      if (!value(&v)) return false;
      int n = 0;
      if (!ParseIntValue(v, &n) || n < 4 || n > 32) {
        *error = "tile must be an integer in [4, 32], got '" + v + "'";
        return false;
      }
      opts->params.tile = n;
    } else if (match("-d", "--device")) {
      if (!value(&v)) return false;
      int n = 0;
      if (!ParseIntValue(v, &n) || n < 0) {
        *error = "device must be a non-negative integer, got '" + v + "'";
        return false;
      }
      opts->params.device = n;
    } else if (match("-n", "--limit")) {
      if (!value(&v)) return false;
      int n = 0;
      if (!ParseIntValue(v, &n) || n < 0) {
        *error = "limit must be a non-negative integer, got '" + v + "'";
        return false;
      }
      opts->limit = n;
    } else if (match("-l", "--log")) {
      if (!value(&v)) return false;
      opts->log_path = v;
    } else if (match("-b", "--bench")) {
      opts->bench = true;
    } else if (match("-v", "--verify")) {
      opts->verify = true;
    } else if (match("-cpu", "--cpu-only")) {
      opts->cpu_only = true;
    } else if (match("-q", "--quiet")) {
      opts->quiet = true;
    } else {
      *error = "unknown argument '" + arg + "' (see --help)";
      return false;
    }
  }
  if (opts->input_dir.empty()) {
    *error = "an input directory is required (-i/--input)";
    return false;
  }
  if (opts->output_dir.empty()) {
    opts->output_dir = opts->input_dir + "_processed";
  }
  if (opts->log_path.empty()) {
    opts->log_path = opts->output_dir + "/run_log.txt";
  }
  return true;
}

// Writes every line to the log file and (unless --quiet) to stdout, so a
// single stream feeds both the console and the proof-of-execution artifact.
class Logger {
 public:
  Logger(std::ofstream* file, bool quiet) : file_(file), quiet_(quiet) {}

  void Line(const std::string& text) {
    *file_ << text << '\n';
    if (!quiet_) {
      std::cout << text << '\n';
    }
  }

  void Separator(char fill) { Line(std::string(80, fill)); }

 private:
  std::ofstream* file_;
  bool quiet_;
};

std::string TimestampNow() {
  const std::time_t now = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
  return buf;
}

std::string HostDescription() {
#if defined(__linux__)
  struct utsname info;
  if (uname(&info) == 0) {
    return std::string(info.sysname) + " " + info.release + " (" +
           info.machine + ")";
  }
#endif
  return "unknown";
}

std::string CudaVersionString(int version) {
  if (version <= 0) {
    return "n/a";
  }
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%d.%d", version / 1000,
                (version % 1000) / 10);
  return buf;
}

// Formats a duration in milliseconds without locale-dependent separators.
std::string FormatMs(double ms) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.3f", ms);
  return buf;
}

OutputFormat FormatOf(Operation op) {
  return (op == Operation::kGray || op == Operation::kSobel ||
          op == Operation::kEqualize)
             ? OutputFormat::kGrayscale
             : OutputFormat::kRgba;
}

// Allowed CPU-vs-GPU difference per operation. Small non-zero tolerances
// account for floating-point contraction (FMA) reordering on the GPU; the
// equalization lookup table is built by identical host code on both sides,
// so only a gray-value rounding difference can propagate (tolerance 1).
int ToleranceOf(Operation op) {
  switch (op) {
    case Operation::kGray:
    case Operation::kInvert:
    case Operation::kBrighten:
    case Operation::kEqualize:
      return 1;
    case Operation::kBlur:
    case Operation::kSobel:
      return 2;
    case Operation::kSharpen:
      return 3;
  }
  return 0;
}

std::vector<unsigned char> ComputeCpuOperation(Operation op,
                                               const RgbaImage& image,
                                               const PipelineParams& params) {
  switch (op) {
    case Operation::kGray:
      return CpuGrayscale(image);
    case Operation::kInvert:
      return CpuInvert(image);
    case Operation::kBrighten:
      return CpuBrighten(image, params.brightness_delta, params.contrast);
    case Operation::kBlur:
      return CpuGaussianBlur(image, params.sigma);
    case Operation::kSobel:
      return CpuSobel(image);
    case Operation::kSharpen:
      return CpuSharpen(image, params.sigma, params.sharpen_amount);
    case Operation::kEqualize:
      return CpuEqualize(image);
  }
  return {};
}

int MaxAbsDiff(const std::vector<unsigned char>& a,
               const std::vector<unsigned char>& b) {
  if (a.size() != b.size()) {
    return 999999;
  }
  int max_diff = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const int diff = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
    if (diff > max_diff) {
      max_diff = diff;
    }
  }
  return max_diff;
}

struct OpAggregate {
  double total_device_ms = 0.0;
  long long images = 0;
};

struct VerifyAggregate {
  int max_abs_diff = 0;
  long long images = 0;
  long long failures = 0;
};

}  // namespace

int RunMain(int argc, char** argv) {
  Options opts;
  std::string error;
  if (!ParseArgs(argc, argv, &opts, &error)) {
    std::cerr << "error: " << error << "\n\n";
    PrintUsage();
    return 1;
  }

  std::vector<Operation> ops = opts.ops;
  if (ops.empty()) {
    ops = {Operation::kGray,    Operation::kInvert, Operation::kBrighten,
           Operation::kBlur,    Operation::kSobel,  Operation::kSharpen,
           Operation::kEqualize};
  }

  // Fail fast on a bad input directory before creating anything.
  std::error_code ec_exists;
  if (!fs::exists(opts.input_dir, ec_exists)) {
    std::cerr << "error: input directory '" << opts.input_dir
              << "' does not exist\n";
    return 2;
  }

  // Refuse to write outputs into the input directory: the outputs are PNGs
  // too and would be picked up as inputs on the next run.
  std::error_code ec_in;
  std::error_code ec_out;
  const fs::path in_canon = fs::weakly_canonical(opts.input_dir, ec_in);
  const fs::path out_canon = fs::weakly_canonical(opts.output_dir, ec_out);
  if (!ec_in && !ec_out && in_canon == out_canon) {
    std::cerr << "error: the output directory must differ from the input "
                 "directory\n";
    return 2;
  }

  if (!EnsureDirectory(opts.output_dir)) {
    std::cerr << "error: cannot create output directory '" << opts.output_dir
              << "'\n";
    return 2;
  }
  std::ofstream log_file(opts.log_path);
  if (!log_file) {
    std::cerr << "error: cannot open log file '" << opts.log_path << "'\n";
    return 2;
  }
  std::ofstream bench_csv;
  if (opts.bench) {
    const std::string csv_path = opts.output_dir + "/benchmark.csv";
    bench_csv.open(csv_path);
    if (!bench_csv) {
      std::cerr << "error: cannot open benchmark file '" << csv_path << "'\n";
      return 2;
    }
    bench_csv << "image,width,height,operation,tile,gpu_ms,cpu_ms\n";
  }

  Logger logger(&log_file, opts.quiet);
  logger.Separator('=');
  logger.Line("CUDA Batch Image Processor - run log");
  logger.Line("Started : " + TimestampNow());
  logger.Line("Host    : " + HostDescription());
  logger.Line("Input   : " + opts.input_dir);
  logger.Line("Output  : " + opts.output_dir);
  {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "Params  : sigma=%.2f tile=%d brightness=%+d contrast=%.2f "
                  "sharpen=%.2f",
                  opts.params.sigma, opts.params.tile,
                  opts.params.brightness_delta, opts.params.contrast,
                  opts.params.sharpen_amount);
    logger.Line(buf);
  }
  if (opts.bench) {
    logger.Line("Benchmark : per-image per-operation timings -> " +
                opts.output_dir + "/benchmark.csv");
  }

  GpuDeviceInfo gpu_info;
  if (opts.cpu_only) {
    logger.Line(
        "Mode    : CPU reference (--cpu-only; debug aid only, no GPU used)");
  } else {
    if (!InitializeGpu(opts.params.device, &gpu_info, &error)) {
      logger.Line("ERROR   : GPU initialization failed: " + error);
      logger.Separator('=');
      std::cerr << "error: GPU initialization failed: " << error << "\n";
      return 2;
    }
    char buf[256];
    std::snprintf(
        buf, sizeof(buf),
        "Mode    : GPU - %s (compute capability %d.%d, %.0f MiB "
        "global memory, %d KiB shared memory per block)",
        gpu_info.name.c_str(), gpu_info.compute_major, gpu_info.compute_minor,
        static_cast<double>(gpu_info.total_memory) / (1024.0 * 1024.0),
        gpu_info.shared_mem_per_block / 1024);
    logger.Line(buf);
    std::snprintf(buf, sizeof(buf), "CUDA    : runtime %s, driver %s",
                  CudaVersionString(gpu_info.runtime_version).c_str(),
                  CudaVersionString(gpu_info.driver_version).c_str());
    logger.Line(buf);
  }

  std::vector<std::string> files = ListImageFiles(opts.input_dir);
  if (files.empty()) {
    logger.Line("ERROR   : no supported image files found in '" +
                opts.input_dir + "'");
    logger.Separator('=');
    std::cerr << "error: no supported image files found in '" << opts.input_dir
              << "'\n";
    return 2;
  }
  if (opts.limit > 0 && files.size() > static_cast<size_t>(opts.limit)) {
    files.resize(static_cast<size_t>(opts.limit));
  }
  {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "Dataset : %zu images (png/jpg/jpeg/bmp/tga), sorted by name",
                  files.size());
    logger.Line(buf);
  }
  logger.Separator('-');

  std::map<Operation, OpAggregate> op_totals;
  std::map<Operation, VerifyAggregate> verify_totals;
  long long total_pixels = 0;
  long long images_processed = 0;
  long long images_failed = 0;
  long long files_saved = 0;
  const auto wall_start = std::chrono::steady_clock::now();

  for (size_t idx = 0; idx < files.size(); ++idx) {
    const std::string& path = files[idx];
    RgbaImage image;
    if (!LoadImageRgba(path, &image, &error)) {
      logger.Line("WARNING : skipping '" + Basename(path) + "': " + error);
      ++images_failed;
      continue;
    }
    total_pixels += static_cast<long long>(image.PixelCount());

    std::vector<OpResult> results;
    GrayStats stats;
    std::map<Operation, double> cpu_ms;
    const auto t0 = std::chrono::steady_clock::now();
    if (opts.cpu_only) {
      for (Operation op : ops) {
        OpResult result;
        result.op = op;
        result.op_name = OperationName(op);
        result.width = image.width;
        result.height = image.height;
        result.format = FormatOf(op);
        const auto c0 = std::chrono::steady_clock::now();
        result.pixels = ComputeCpuOperation(op, image, opts.params);
        cpu_ms[op] = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - c0)
                         .count();
        result.device_ms = 0.0f;
        results.push_back(std::move(result));
      }
      std::vector<int> histogram(256, 0);
      const std::vector<unsigned char> gray = CpuGrayscale(image);
      for (unsigned char v : gray) {
        ++histogram[v];
      }
      stats = StatsFromHistogram(histogram.data());
    } else {
      if (!ProcessImageGpu(image, ops, opts.params, &results, &stats, &error)) {
        logger.Line("ERROR   : GPU processing failed on '" + Basename(path) +
                    "': " + error);
        logger.Separator('=');
        return 2;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    for (const OpResult& result : results) {
      const std::string out_path =
          opts.output_dir + "/" + Stem(path) + "_" + result.op_name + ".png";
      const bool ok = (result.format == OutputFormat::kGrayscale)
                          ? SaveGrayPng(out_path, result.pixels.data(),
                                        result.width, result.height)
                          : SaveRgbaPng(out_path, result.pixels.data(),
                                        result.width, result.height);
      if (ok) {
        ++files_saved;
      } else {
        logger.Line("WARNING : failed to write '" + out_path + "'");
      }
    }

    // CPU reference: used for --verify (correctness) and --bench (timings).
    std::map<Operation, int> diffs;
    if ((opts.verify || opts.bench) && !opts.cpu_only) {
      for (const OpResult& result : results) {
        const auto c0 = std::chrono::steady_clock::now();
        const std::vector<unsigned char> cpu =
            ComputeCpuOperation(result.op, image, opts.params);
        cpu_ms[result.op] = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - c0)
                                .count();
        if (opts.verify) {
          const int diff = MaxAbsDiff(cpu, result.pixels);
          diffs[result.op] = diff;
          VerifyAggregate& agg = verify_totals[result.op];
          agg.max_abs_diff = std::max(agg.max_abs_diff, diff);
          ++agg.images;
          if (diff > ToleranceOf(result.op)) {
            ++agg.failures;
          }
        }
      }
    }

    if (opts.bench) {
      for (const OpResult& result : results) {
        std::string gpu_cell;
        std::string cpu_cell;
        if (!opts.cpu_only) {
          gpu_cell = FormatMs(result.device_ms);
        }
        const auto it = cpu_ms.find(result.op);
        if (it != cpu_ms.end()) {
          cpu_cell = FormatMs(it->second);
        }
        bench_csv << Basename(path) << ',' << image.width << ',' << image.height
                  << ',' << result.op_name << ',' << opts.params.tile << ','
                  << gpu_cell << ',' << cpu_cell << '\n';
      }
    }

    for (const OpResult& result : results) {
      OpAggregate& agg = op_totals[result.op];
      agg.total_device_ms += result.device_ms;
      ++agg.images;
    }

    {
      char buf[512];
      std::snprintf(buf, sizeof(buf),
                    "[%3zu/%3zu] %s (%dx%d, %lld px) "
                    "wall=%.2f ms",
                    idx + 1, files.size(), Basename(path).c_str(), image.width,
                    image.height, static_cast<long long>(image.PixelCount()),
                    wall_ms);
      logger.Line(buf);
      std::snprintf(buf, sizeof(buf),
                    "    gray stats : min=%d max=%d mean=%.2f", stats.min_value,
                    stats.max_value, stats.mean_value);
      logger.Line(buf);
    }
    for (const OpResult& result : results) {
      char buf[256];
      if (opts.cpu_only) {
        const auto it = cpu_ms.find(result.op);
        std::snprintf(buf, sizeof(buf), "    %-9s : (cpu reference %8.2f ms)",
                      result.op_name.c_str(),
                      it != cpu_ms.end() ? it->second : 0.0);
      } else {
        std::snprintf(buf, sizeof(buf), "    %-9s : %8.3f ms (GPU)",
                      result.op_name.c_str(), result.device_ms);
      }
      std::string line = buf;
      if (diffs.count(result.op) != 0) {
        std::snprintf(buf, sizeof(buf), "  verify max|diff|=%d",
                      diffs[result.op]);
        line += buf;
      }
      logger.Line(line);
    }
    ++images_processed;
  }

  // Blur tile-size sweep (GPU mode only): reruns the blur over the first
  // images with block sizes 8x8 / 16x16 / 32x32 and appends the timings to
  // the benchmark CSV, demonstrating the effect of the shared-memory tiling.
  if (opts.bench && !opts.cpu_only &&
      std::find(ops.begin(), ops.end(), Operation::kBlur) != ops.end()) {
    const int kSweepTiles[] = {8, 16, 32};
    const size_t sweep_images = std::min(files.size(), static_cast<size_t>(16));
    logger.Line("Benchmark: blur tile-size sweep over " +
                std::to_string(sweep_images) + " images:");
    for (int tile : kSweepTiles) {
      double total_ms = 0.0;
      int done = 0;
      for (size_t idx = 0; idx < sweep_images; ++idx) {
        RgbaImage image;
        if (!LoadImageRgba(files[idx], &image, &error)) {
          continue;
        }
        PipelineParams sweep_params = opts.params;
        sweep_params.tile = tile;
        std::vector<OpResult> sweep_results;
        GrayStats sweep_stats;
        if (!ProcessImageGpu(image, {Operation::kBlur}, sweep_params,
                             &sweep_results, &sweep_stats, &error)) {
          logger.Line("ERROR   : tile sweep failed on '" +
                      Basename(files[idx]) + "': " + error);
          logger.Separator('=');
          return 2;
        }
        const double ms = sweep_results.front().device_ms;
        total_ms += ms;
        ++done;
        bench_csv << Basename(files[idx]) << ',' << image.width << ','
                  << image.height << ",blur," << tile << ',' << FormatMs(ms)
                  << ",\n";
      }
      if (done > 0) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "    tile %2dx%-2d : avg blur %8.3f ms/image", tile, tile,
                      total_ms / done);
        logger.Line(buf);
      }
    }
  }
  if (opts.bench && opts.cpu_only) {
    logger.Line("Benchmark: tile-size sweep skipped (requires GPU mode)");
  }

  const auto wall_end = std::chrono::steady_clock::now();
  const double wall_seconds =
      std::chrono::duration<double>(wall_end - wall_start).count();

  logger.Separator('-');
  logger.Line("Summary");
  {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "    images processed : %lld (%lld failed to load)",
                  images_processed, images_failed);
    logger.Line(buf);
    std::snprintf(buf, sizeof(buf), "    input pixels    : %lld (%.1f Mpx)",
                  total_pixels, static_cast<double>(total_pixels) / 1e6);
    logger.Line(buf);
    std::snprintf(buf, sizeof(buf), "    output files    : %lld written to %s",
                  files_saved, opts.output_dir.c_str());
    logger.Line(buf);
    std::snprintf(buf, sizeof(buf), "    total wall time : %.3f s%s",
                  wall_seconds,
                  (opts.verify && !opts.cpu_only)
                      ? " (includes --verify CPU reference computation)"
                      : "");
    logger.Line(buf);
    std::snprintf(
        buf, sizeof(buf), "    throughput      : %.2f images/s | %.2f Mpx/s",
        images_processed / std::max(wall_seconds, 1e-9),
        static_cast<double>(total_pixels) / std::max(wall_seconds, 1e-9) / 1e6);
    logger.Line(buf);
    if (opts.bench) {
      logger.Line("    benchmark csv   : " + opts.output_dir +
                  "/benchmark.csv");
    }
  }
  if (!opts.cpu_only) {
    logger.Line("    GPU kernel time by operation:");
    for (Operation op : ops) {
      const OpAggregate& agg = op_totals[op];
      char buf[256];
      std::snprintf(buf, sizeof(buf),
                    "        %-9s : %8.3f s total (%.3f ms/image, %lld "
                    "images)",
                    OperationName(op).c_str(), agg.total_device_ms / 1000.0,
                    agg.images > 0
                        ? agg.total_device_ms / static_cast<double>(agg.images)
                        : 0.0,
                    agg.images);
      logger.Line(buf);
    }
  }
  if (opts.verify && !opts.cpu_only) {
    logger.Line("Verification (GPU results vs CPU reference):");
    bool all_pass = true;
    for (Operation op : ops) {
      const VerifyAggregate& agg = verify_totals[op];
      const bool pass = agg.failures == 0;
      all_pass = all_pass && pass;
      char buf[256];
      std::snprintf(buf, sizeof(buf),
                    "        %-9s : %lld images, max|diff|=%d (tolerance %d) "
                    "... %s",
                    OperationName(op).c_str(), agg.images, agg.max_abs_diff,
                    ToleranceOf(op), pass ? "PASS" : "FAIL");
      logger.Line(buf);
    }
    logger.Line(all_pass ? "    ALL OPERATIONS PASS"
                         : "    VERIFICATION FAILURES DETECTED");
  }
  logger.Separator('=');
  {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "DONE - %lld output files in '%s', log saved to '%s'",
                  files_saved, opts.output_dir.c_str(), opts.log_path.c_str());
    logger.Line(buf);
  }
  log_file.flush();
  bench_csv.flush();
  return (images_processed == 0) ? 2 : 0;
}

}  // namespace imgproc

int main(int argc, char** argv) { return imgproc::RunMain(argc, argv); }
