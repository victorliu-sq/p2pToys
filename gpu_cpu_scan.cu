#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

// Size of the array
const int SIZE = 1000000;

__global__ void gpuSequentialScan(int *array) {
  int sum = 0;
  for (int i = 0; i < SIZE; ++i) {
    sum += array[i];
  }
  array[0] = sum;
}

void cpuSequentialScan(int *array) {
  int sum = 0;
  for (int i = 0; i < SIZE; ++i) {
    sum += array[i];
  }
  array[0] = sum;
}

int main() {
  // Allocate and initialize CPU array
  int *cpuArray = new int[SIZE];
  for (int i = 0; i < SIZE; ++i) {
    cpuArray[i] = i;
  }

  // Allocate and initialize GPU array
  int *gpuArray;
  cudaMalloc(&gpuArray, SIZE * sizeof(int));
  cudaMemcpy(gpuArray, cpuArray, SIZE * sizeof(int), cudaMemcpyHostToDevice);

  // CPU timing
  auto cpuStart = std::chrono::high_resolution_clock::now();
  cpuSequentialScan(cpuArray);
  auto cpuEnd = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cpuDuration = cpuEnd - cpuStart;
  std::cout << "CPU time: " << cpuDuration.count() << " seconds\n";

  // GPU timing
  auto gpuStart = std::chrono::high_resolution_clock::now();
  gpuSequentialScan<<<1, 1>>>(gpuArray);
  cudaDeviceSynchronize();
  auto gpuEnd = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> gpuDuration = gpuEnd - gpuStart;
  std::cout << "GPU time: " << gpuDuration.count() << " seconds\n";

  // Clean up
  delete[] cpuArray;
  cudaFree(gpuArray);

  return 0;
}
