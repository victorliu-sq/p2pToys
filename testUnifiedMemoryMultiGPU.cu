#include <cuda_runtime.h>
#include <stdio.h>

__global__ void testKernel(int *data, int value) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx == 0) { // Each GPU writes its ID at the start of the array
    data[idx] = value;
  }
}

int main() {
  int numGPUs;
  cudaGetDeviceCount(&numGPUs);
  printf("Number of GPUs: %d\n", numGPUs);

  int *data;
  cudaMallocManaged(&data, sizeof(int) * numGPUs);

  // Launch kernel on each GPU
  for (int i = 0; i < numGPUs; ++i) {
    cudaSetDevice(i);
    testKernel<<<1, 1>>>(
        data + i, i); // Each GPU writes its ID to a distinct part of the array
    cudaDeviceSynchronize();
  }

  // Check data
  bool isCorrect = true;
  for (int i = 0; i < numGPUs; ++i) {
    if (data[i] != i) {
      printf("Error: data[%d] = %d, expected %d\n", i, data[i], i);
      isCorrect = false;
    }
  }

  if (isCorrect) {
    printf("Unified Memory is correctly accessible from all GPUs.\n");
  }

  // Clean up
  cudaFree(data);

  return 0;
}
