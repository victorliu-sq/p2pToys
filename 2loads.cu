#include <cuda_runtime.h>
#include <stdio.h>

// Kernel using Cache-All (CA) load
__global__ void load_ca(int *data, int *result, int n) {
  if (threadIdx.x == 0) { // Ensure only one thread executes the loop
    int w_rank = 0;
    for (int i = 0; i < n; i++) {
      asm("ld.ca.s32 %0, [%1];" : "=r"(w_rank) : "l"(&data[i]));
      result[i] = w_rank;
    }
  }
}

// Kernel using Cache-Global (CG) load
__global__ void load_cg(int *data, int *result, int n) {
  if (threadIdx.x == 0) {
    int w_rank = 0;
    for (int i = 0; i < n; i++) {
      asm("ld.cg.s32 %0, [%1];" : "=r"(w_rank) : "l"(&data[i]));
      result[i] = w_rank;
    }
  }
}

void measureKernelPerformance(int *d_data, int *d_result, int n,
                              void (*kernel)(int *, int *, int),
                              const char *kernelName) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;

  cudaEventRecord(start);
  kernel<<<1, 1>>>(d_data, d_result, n); // Launch with one block of one thread
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("%s kernel execution time: %.5f ms\n", kernelName, milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

int main() {
  // const int n = 1024 * 1024; // Number of elements
  const int n = 1000 * 1000; // Number of elements
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

  // Cleanup
  cudaFree(d_data);
  cudaFree(d_result);
  delete[] h_data;

  return 0;
}
