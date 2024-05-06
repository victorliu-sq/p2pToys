#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

__global__ void scanArrayUnifiedMemory(int *data, int numElements) {
  int sum = 0;
  // Scan from the start to the end
  for (int i = 0; i < numElements; i++) {
    sum += data[i];
  }
  data[0] = sum;
}

__global__ void scanArrayGlobalMemory(int *data, int numElements) {
  int sum = 0;
  // Scan from the start to the end
  for (int i = 0; i < numElements; i++) {
    sum += data[i];
  }
  data[0] = sum;
}

void scanArrayCPU(int *data, int numElements) {
  int sum = 0;
  // Scan from the start to the end
  for (int i = 0; i < numElements; i++) {
    sum += data[i];
  }
  data[0] = sum;
}

int main() {
  int numElements = 1000000;
  size_t size = numElements * sizeof(int);
  int *dataUnified, *dataGlobal;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&dataUnified, size);

  // Allocate Global Memory on the GPU
  cudaMalloc(&dataGlobal, size);

  // Initialize array
  for (int i = 0; i < numElements; i++) {
    dataUnified[i] = i;
  }

  cudaMemcpy(dataGlobal, dataUnified, size, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize(); // Wait for GPU to finish any outstanding work

  // GPU computation with Unified Memory
  auto startGPUUnified = std::chrono::high_resolution_clock::now();
  scanArrayUnifiedMemory<<<1, 1>>>(dataUnified, numElements);
  cudaDeviceSynchronize(); // Wait for GPU to finish
  auto endGPUUnified = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsedGPUUnified =
      endGPUUnified - startGPUUnified;
  std::cout << "GPU Unified Memory Elapsed time: "
            << elapsedGPUUnified.count() * 1e6 << " ms\n";

  // GPU computation with Global Memory
  auto startGPUGlobal = std::chrono::high_resolution_clock::now();
  scanArrayGlobalMemory<<<1, 1>>>(dataGlobal, numElements);
  cudaDeviceSynchronize(); // Wait for GPU to finish
  auto endGPUGlobal = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsedGPUGlobal =
      endGPUGlobal - startGPUGlobal;
  std::cout << "GPU Global Memory Elapsed time: "
            << elapsedGPUGlobal.count() * 1e6 << " ms\n";

  // CPU computation
  auto startCPU = std::chrono::high_resolution_clock::now();
  scanArrayCPU(dataUnified, numElements);
  auto endCPU = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsedCPU = endCPU - startCPU;
  std::cout << "CPU Elapsed time: " << elapsedCPU.count() * 1e6 << " ms\n";

  // Free memory
  cudaFree(dataUnified);
  cudaFree(dataGlobal);

  return 0;
}