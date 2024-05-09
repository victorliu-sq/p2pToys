#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(ans)                                                        \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " " << file
              << " " << line << std::endl;
    exit(code);
  }
}

// Timing function
uint64_t getNanoSecond() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             now.time_since_epoch())
      .count();
}

__global__ void simpleKernel(int *data, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    data[tid] = data[tid] * 2;
  }
}

int main() {
  const int size = 100 * 1024 * 1024; // 100 million elements (approx 400 MB)
  const int num_elements = size / sizeof(int);

  // Kernel configuration
  const int threads_per_block = 256;
  const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;

  // Allocate device memory
  int *d_data;
  CUDA_CHECK(cudaMalloc(&d_data, size));

  // Allocate pageable host memory
  int *pageable_data = new int[num_elements];
  for (int i = 0; i < num_elements; ++i) {
    pageable_data[i] = i;
  }

  // Allocate pinned (page-locked) host memory
  int *pinned_data;
  CUDA_CHECK(cudaMallocHost(&pinned_data, size));
  for (int i = 0; i < num_elements; ++i) {
    pinned_data[i] = i;
  }

  // Copy time and kernel execution time with pageable memory
  uint64_t start_cpy_pageable = getNanoSecond();
  CUDA_CHECK(cudaMemcpy(d_data, pageable_data, size, cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();
  uint64_t end_cpy_pageable = getNanoSecond();
  std::cout << "Copy time (pageable): "
            << (end_cpy_pageable - start_cpy_pageable) / 1e6 << " ms"
            << std::endl;

  uint64_t start_kernel_pageable = getNanoSecond();
  simpleKernel<<<blocks, threads_per_block>>>(d_data, num_elements);
  cudaDeviceSynchronize();
  uint64_t end_kernel_pageable = getNanoSecond();
  std::cout << "Kernel execution time (pageable): "
            << (end_kernel_pageable - start_kernel_pageable) / 1e6 << " ms"
            << std::endl;

  // Copy time and kernel execution time with pinned memory
  uint64_t start_cpy_pinned = getNanoSecond();
  CUDA_CHECK(cudaMemcpy(d_data, pinned_data, size, cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();
  uint64_t end_cpy_pinned = getNanoSecond();
  std::cout << "Copy time (pinned): "
            << (end_cpy_pinned - start_cpy_pinned) / 1e6 << " ms" << std::endl;

  uint64_t start_kernel_pinned = getNanoSecond();
  simpleKernel<<<blocks, threads_per_block>>>(d_data, num_elements);
  cudaDeviceSynchronize();
  uint64_t end_kernel_pinned = getNanoSecond();
  std::cout << "Kernel execution time (pinned): "
            << (end_kernel_pinned - start_kernel_pinned) / 1e6 << " ms"
            << std::endl;

  // Cleanup
  delete[] pageable_data;
  CUDA_CHECK(cudaFreeHost(pinned_data));
  CUDA_CHECK(cudaFree(d_data));

  return 0;
}
