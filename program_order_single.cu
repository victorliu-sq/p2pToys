#include <cuda_runtime.h>
#include <stdio.h>

__global__ void writeKernel(int *data) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  data[idx] = idx;     // First write
  data[idx] = idx + 1; // Second write, supposed to overwrite the first
}

__global__ void readKernel(int *data, int *result) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  result[idx] = data[idx]; // Read the value written by the first kernel
}

int main() {
  const int numElements = 256;
  int *d_data, *d_result;
  int h_result[numElements];

  cudaMalloc(&d_data, numElements * sizeof(int));
  cudaMalloc(&d_result, numElements * sizeof(int));

  // Launch the first kernel to write data
  writeKernel<<<1, numElements>>>(d_data);
  // Ensure all writes are completed before starting the next kernel
  cudaDeviceSynchronize();

  // Launch the second kernel to read data
  readKernel<<<1, numElements>>>(d_data, d_result);
  // Ensure all reads are completed
  cudaDeviceSynchronize();

  // Copy results back to host
  cudaMemcpy(h_result, d_result, numElements * sizeof(int),
             cudaMemcpyDeviceToHost);

  // Verify and print the results
  bool valid = true;
  for (int i = 0; i < numElements; ++i) {
    if (h_result[i] != i + 1) {
      printf("Error at index %d: Expected %d, found %d\n", i, i + 1,
             h_result[i]);
      valid = false;
    }
  }

  if (valid) {
    printf("All values are correct.\n");
  }

  // Clean up
  cudaFree(d_data);
  cudaFree(d_result);
  return 0;
}
