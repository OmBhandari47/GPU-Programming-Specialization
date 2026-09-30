# Vendored third-party code

This directory contains unmodified single-header image I/O libraries from the
[stb](https://github.com/nothings/stb) project (retrieved from the master
branch on 2026-09-30):

- `stb_image.h` v2.30 - public domain image loader (png/jpg/bmp/tga/...)
- `stb_image_write.h` v1.16 - public domain image writer (png/bmp/tga/...)

Both files are dual-licensed: public domain or MIT (see the license
comment at the top of each file). They are vendored so the project builds
with no external image-library dependency.
