#include <cuda_runtime.h>
#include <stdio.h>

__global__ void rawHazardKernel(int *data, int *errorFlag) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // Read after write
  data[idx] = idx;          // Write
  int readData = data[idx]; // Read

  // Check if the read data is not what was written
  if (readData != idx) {
    errorFlag[idx] = 1; // Set error flag
  } else {
    errorFlag[idx] = 0;
  }
}

__global__ void warHazardKernel(int *data, int *errorFlag) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // Write after read
  int readData = data[idx]; // Read
  data[idx] = readData + 1; // Write

  // Check if the data written is not the incremented read value
  if (data[idx] != readData + 1) {
    errorFlag[idx] = 1; // Set error flag
  } else {
    errorFlag[idx] = 0;
  }
}

__global__ void wawHazardKernel(int *data, int *errorFlag) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // Write after write
  data[idx] = idx;     // First write
  data[idx] = idx + 1; // Second write

  // Check if the data is not the last written value
  if (data[idx] != idx + 1) {
    errorFlag[idx] = 1; // Set error flag
  } else {
    errorFlag[idx] = 0;
  }
}

int main() {
  const int numElements = 2560;
  const int numThreadsPerBlock = 512;
  int *d_data, *d_errorFlag;

  cudaMalloc(&d_data, numElements * sizeof(int));
  cudaMalloc(&d_errorFlag, numElements * sizeof(int));

  // Initialize data
  cudaMemset(d_data, 0, numElements * sizeof(int));
  cudaMemset(d_errorFlag, 0, numElements * sizeof(int));

  // Launch kernels
  rawHazardKernel<<<numElements / numThreadsPerBlock, numThreadsPerBlock>>>(
      d_data, d_errorFlag);
  warHazardKernel<<<numElements / numThreadsPerBlock, numThreadsPerBlock>>>(
      d_data, d_errorFlag);
  wawHazardKernel<<<numElements / numThreadsPerBlock, numThreadsPerBlock>>>(
      d_data, d_errorFlag);

  // Copy error flag array back to host to check for errors
  int h_errorFlag[numElements];
  cudaMemcpy(h_errorFlag, d_errorFlag, numElements * sizeof(int),
             cudaMemcpyDeviceToHost);

  // Check error flags
  bool errorOccurred = false;
  for (int i = 0; i < numElements; i++) {
    if (h_errorFlag[i] != 0) {
      printf("Error in thread %d\n", i);
      errorOccurred = true;
    }
  }

  if (!errorOccurred) {
    printf("No hazards detected.\n");
  }

  // Clean up
  cudaFree(d_data);
  cudaFree(d_errorFlag);
  cudaDeviceSynchronize();
  return 0;
}
