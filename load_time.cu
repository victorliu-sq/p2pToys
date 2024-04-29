#include <cuda_runtime.h>
#include <stdio.h>

// __global__ void memory_load_test(volatile int *global_data, int n) {
//   int local; // Adjust size based on maximum expected `n`
//   // int sum = 0;
//   for (int i = 0; i < n; i++) {
//     local = global_data[i]; // Load data from global to local memory
//     // sum += local;
//   }
// }

__global__ void memory_load_test(int *global_data, int *indices, int n,
                                 int *sum) {
  int local;
  *sum = 0;
  for (int i = 0; i < n; i++) {
    local = global_data[indices[i]]; // Load data from global memory at random
                                     // indices
    *sum += local;
  }
}

int main() {
  int *d_data, *d_indices, *d_sum;
  int *h_indices, sum;

  for (int n = 1024; n <= 1024 * 1024; n <<= 1) {
    size_t bytes = n * sizeof(int);

    // Allocate memory on host and device
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_indices, bytes);
    cudaMalloc(&d_sum, sizeof(int));
    int *h_data = (int *)malloc(bytes);
    h_indices = (int *)malloc(bytes);

    // Initialize data on host
    for (int i = 0; i < n; i++) {
      h_data[i] = i; // Sequential numbers as data
    }

    // Generate random indices on host
    for (int i = 0; i < n; i++) {
      h_indices[i] = rand() % n;
    }

    // Copy data from host to device
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, bytes, cudaMemcpyHostToDevice);

    // Setup timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel and measure time
    cudaEventRecord(start);
    memory_load_test<<<1, 1>>>(d_data, d_indices, n,
                               d_sum); // Using 1 block of 1 thread
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();

    // Copy back the sum to check correctness if necessary
    cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken to load %d integers randomly from global to local "
           "memory: %.3f ms\n",
           n, milliseconds);

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_indices);
    cudaFree(d_sum);
    free(h_data);
    free(h_indices);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  return 0;
}
