#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

void testMemcpyPerformance(int numElements) {
  size_t bytes = numElements * sizeof(int);

  // Allocate host memory
  int *hostArray = new int[numElements];
  for (int i = 0; i < numElements; ++i) {
    hostArray[i] = i;
  }

  // Allocate device memory
  int *deviceArray;
  cudaMalloc(&deviceArray, bytes);

  // Timing cudaMemcpy Host to Device
  auto h2dStart = std::chrono::high_resolution_clock::now();
  cudaMemcpy(deviceArray, hostArray, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  auto h2dEnd = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> h2dDuration = h2dEnd - h2dStart;
  std::cout << "Size: " << numElements
            << " integers, Host to Device memcpy time: "
            << h2dDuration.count() * 1e6 << " milliseconds\n";

  // Timing cudaMemcpy Device to Host
  auto d2hStart = std::chrono::high_resolution_clock::now();
  cudaMemcpy(hostArray, deviceArray, bytes, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  auto d2hEnd = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> d2hDuration = d2hEnd - d2hStart;
  std::cout << "Size: " << numElements
            << " integers, Device to Host memcpy time: "
            << d2hDuration.count() * 1e6 << " milliseconds\n";

  // Clean up
  delete[] hostArray;
  cudaFree(deviceArray);
}

int main() {
  // Test various sizes
  for (int size = 10; size <= 1000000; size *= 10) {
    testMemcpyPerformance(size);
  }
  return 0;
}
