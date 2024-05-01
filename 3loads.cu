#include <cuda_runtime.h>
#include <stdio.h>

// Kernel using Cache-All (CA) load
__global__ void load_ca(int *data, int *result, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int w_rank = 0;
  // Cache-All load
  asm("ld.ca.s32 %0, [%1];" : "=r"(w_rank) : "l"(&data[idx]));
  result[idx] = w_rank;
}

// Kernel using Cache-Global (CG) load
__global__ void load_cg(int *data, int *result) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int w_rank = 0;
  // Cache-Global load
  asm("ld.cg.s32 %0, [%1];" : "=r"(w_rank) : "l"(&data[idx]));
  result[idx] = w_rank;
}

// Kernel using Global load
__global__ void load_global(int *data, int *result) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int w_rank = 0;
  // Direct global memory load
  asm("ld.global.s32 %0, [%1];" : "=r"(w_rank) : "l"(&data[idx]));
  result[idx] = w_rank;
}

void measureKernelPerformance(int *d_data, int *d_result, int n,
                              void (*kernel)(int *, int *),
                              const char *kernelName) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;

  cudaEventRecord(start);
  kernel<<<1, 1>>>(d_data, d_result);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("%s kernel execution time: %.5f ms\n", kernelName, milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

int main() {
  const int n = 1024 * 1024; // Number of elements
  int *d_data, *d_result;

  // Allocate memory
  cudaMalloc(&d_data, n * sizeof(int));
  cudaMalloc(&d_result, n * sizeof(int));

  // Initialize data
  int *h_data = new int[n];
  for (int i = 0; i < n; ++i) {
    h_data[i] = i;
  }
  cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

  // Measure each kernel's performance
  measureKernelPerformance(d_data, d_result, n, load_ca, "Cache-All (CA)");
  measureKernelPerformance(d_data, d_result, n, load_cg, "Cache-Global (CG)");
  measureKernelPerformance(d_data, d_result, n, load_global, "Global");

  // Cleanup
  cudaFree(d_data);
  cudaFree(d_result);
  delete[] h_data;

  return 0;
}
