// Image loading / saving built on the public-domain stb libraries, plus small
// filesystem helpers. All images are normalized to 8-bit RGBA in memory.

#ifndef CUDA_BATCH_IMAGE_PROCESSOR_SRC_IMAGE_IO_H_
#define CUDA_BATCH_IMAGE_PROCESSOR_SRC_IMAGE_IO_H_

#include <cstddef>
#include <string>
#include <vector>

namespace imgproc {

// A 4-channel, 8-bit-per-channel image stored in row-major order.
struct RgbaImage {
  int width = 0;
  int height = 0;
  std::vector<unsigned char> pixels;  // width * height * 4 bytes (RGBA).

  size_t PixelCount() const {
    return static_cast<size_t>(width) * static_cast<size_t>(height);
  }
  size_t ByteCount() const { return PixelCount() * 4; }
};

// Loads an image file (png/jpg/jpeg/bmp/tga), always converting to RGBA.
// Returns false and fills `error` on failure.
bool LoadImageRgba(const std::string& path, RgbaImage* image,
                   std::string* error);

// Saves an 8-bit grayscale buffer (one byte per pixel) as PNG.
bool SaveGrayPng(const std::string& path, const unsigned char* gray, int width,
                 int height);

// Saves an RGBA image as PNG.
bool SaveRgbaPng(const std::string& path, const unsigned char* rgba, int width,
                 int height);

// Returns the sorted list of supported image files directly inside `dir`.
std::vector<std::string> ListImageFiles(const std::string& dir);

// Creates `dir` (and any missing parents) if it does not exist yet.
bool EnsureDirectory(const std::string& dir);

// "/path/to/photo.png" -> "photo".
std::string Stem(const std::string& path);

// "/path/to/photo.png" -> "photo.png".
std::string Basename(const std::string& path);

}  // namespace imgproc

#endif  // CUDA_BATCH_IMAGE_PROCESSOR_SRC_IMAGE_IO_H_
