#include "image_io.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <set>
#include <system_error>

// The stb libraries are written against a very permissive warning profile;
// silence the few warnings they trigger under -Wall -Wextra.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace imgproc {

namespace fs = std::filesystem;

bool LoadImageRgba(const std::string& path, RgbaImage* image,
                   std::string* error) {
  int width = 0;
  int height = 0;
  int channels_in_file = 0;
  unsigned char* data =
      stbi_load(path.c_str(), &width, &height, &channels_in_file, 4);
  if (data == nullptr) {
    *error = "stbi_load failed for '" + path + "': " +
             (stbi_failure_reason() != nullptr ? stbi_failure_reason()
                                               : "unknown error");
    return false;
  }
  image->width = width;
  image->height = height;
  const size_t byte_count = static_cast<size_t>(width) * height * 4;
  image->pixels.assign(data, data + byte_count);
  stbi_image_free(data);
  return true;
}

bool SaveGrayPng(const std::string& path, const unsigned char* gray, int width,
                 int height) {
  return stbi_write_png(path.c_str(), width, height, 1, gray, width) != 0;
}

bool SaveRgbaPng(const std::string& path, const unsigned char* rgba, int width,
                 int height) {
  const int stride_bytes = width * 4;
  return stbi_write_png(path.c_str(), width, height, 4, rgba, stride_bytes) !=
         0;
}

std::vector<std::string> ListImageFiles(const std::string& dir) {
  static const std::set<std::string> kExtensions = {"bmp", "jpeg", "jpg", "png",
                                                    "tga"};
  std::vector<std::string> files;
  std::error_code ec;
  fs::directory_iterator it(dir, ec);
  if (ec) {
    return files;
  }
  for (const fs::directory_entry& entry : it) {
    if (!entry.is_regular_file(ec)) {
      continue;
    }
    std::string ext = entry.path().extension().string();
    if (ext.empty()) {
      continue;
    }
    ext = ext.substr(1);  // Drop the leading dot.
    for (char& c : ext) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (kExtensions.count(ext) != 0) {
      files.push_back(entry.path().string());
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

bool EnsureDirectory(const std::string& dir) {
  std::error_code ec;
  if (fs::exists(dir, ec)) {
    return true;
  }
  fs::create_directories(dir, ec);
  return !ec;
}

std::string Stem(const std::string& path) {
  const size_t slash = path.find_last_of("/\\");
  const std::string base =
      (slash == std::string::npos) ? path : path.substr(slash + 1);
  const size_t dot = base.find_last_of('.');
  if (dot == std::string::npos || dot == 0) {
    return base;
  }
  return base.substr(0, dot);
}

std::string Basename(const std::string& path) {
  const size_t slash = path.find_last_of("/\\");
  return (slash == std::string::npos) ? path : path.substr(slash + 1);
}

}  // namespace imgproc
