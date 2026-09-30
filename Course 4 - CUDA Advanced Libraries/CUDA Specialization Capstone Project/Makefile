# CUDA Batch Image Processor - build file
#
# Targets:
#   make / make all   -> GPU binary (bin/imgproc) + dataset generator
#   make gpu          -> GPU binary only (requires nvcc and an NVIDIA GPU)
#   make cpu          -> CUDA-free debug binary (bin/imgproc_cpu, --cpu-only)
#   make tools        -> dataset generator only
#   make test-kernels -> CPU emulation test of the CUDA kernel indexing
#   make smoke        -> quick end-to-end test on 16 generated images (CPU)
#   make clean
#
# Variables (override on the command line, e.g. `make ARCH=sm_70`):
#   NVCC               nvcc executable (default: from PATH)
#   ARCH               GPU architecture: native (default, CUDA >= 11.5) or
#                      sm_70 / sm_75 / sm_80 / sm_86 / sm_89 / sm_90
#   EXTRA_NVCCFLAGS    extra nvcc flags (e.g. --allow-unsupported-compiler)
#   EXTRA_CXXFLAGS     extra host compiler flags

NVCC ?= nvcc
CXX ?= g++
ARCH ?= native

BIN_DIR := bin
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
NVCCFLAGS ?= -O3 -std=c++17 -Xcompiler -Wall,-Wextra -lineinfo
EXTRA_NVCCFLAGS ?=
EXTRA_CXXFLAGS ?=

COMMON_SRCS := src/common.cpp src/image_io.cpp src/cpu_reference.cpp
GPU_SRCS := src/gpu_pipeline.cu
STUB_SRC := src/gpu_stub.cpp
MAIN_SRC := src/main.cpp
GEN_SRC := tools/generate_dataset.cpp
HEADERS := src/common.h src/image_io.h src/cpu_reference.h \
           src/gpu_pipeline.h src/kernels.cuh

ifeq ($(strip $(ARCH)),native)
  ARCH_FLAGS := -arch=native
else
  ARCH_FLAGS := -gencode arch=compute_$(patsubst sm_%,%,$(ARCH)),code=$(ARCH)
endif

.PHONY: all gpu cpu tools test-kernels smoke clean

all: gpu tools

gpu: $(BIN_DIR)/imgproc

cpu: $(BIN_DIR)/imgproc_cpu

tools: $(BIN_DIR)/generate_dataset

test-kernels: $(BIN_DIR)/kernel_emulation_test
	$(BIN_DIR)/kernel_emulation_test

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/imgproc: $(GPU_SRCS) $(COMMON_SRCS) $(MAIN_SRC) $(HEADERS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) $(EXTRA_NVCCFLAGS) -Isrc \
	  $(GPU_SRCS) $(COMMON_SRCS) $(MAIN_SRC) -o $@

$(BIN_DIR)/imgproc_cpu: $(COMMON_SRCS) $(MAIN_SRC) $(STUB_SRC) $(HEADERS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(EXTRA_CXXFLAGS) -Isrc \
	  $(COMMON_SRCS) $(STUB_SRC) $(MAIN_SRC) -o $@

$(BIN_DIR)/generate_dataset: $(GEN_SRC) src/common.cpp src/image_io.cpp \
                             src/common.h src/image_io.h | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(EXTRA_CXXFLAGS) -Isrc \
	  $(GEN_SRC) src/common.cpp src/image_io.cpp -o $@

# Quick end-to-end check that needs no GPU: emulates the CUDA kernels' exact
# indexing on the CPU, generates 16 images and runs the CPU reference over
# them. Useful for verifying the build on any machine.
smoke: cpu tools test-kernels
	$(BIN_DIR)/generate_dataset --out data/smoke --count 16 --seed 7
	$(BIN_DIR)/imgproc_cpu --input data/smoke --output output/smoke_cpu --cpu-only

$(BIN_DIR)/kernel_emulation_test: tools/kernel_emulation_test.cpp \
                                 src/cpu_reference.cpp src/common.cpp \
                                 src/image_io.cpp src/cpu_reference.h \
                                 src/common.h src/image_io.h | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(EXTRA_CXXFLAGS) -Isrc \
	  tools/kernel_emulation_test.cpp src/cpu_reference.cpp src/common.cpp \
	  src/image_io.cpp -o $@

clean:
	rm -rf $(BIN_DIR)
