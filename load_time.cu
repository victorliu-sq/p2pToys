#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void DoubleLoad(volatile int *global_data, int *indices, int n,
                           int *d_sum) {
  int local;
  int sum = 0;
  int idx;
  for (int i = 0; i < n; i++) {
    idx = indices[i];
    local = global_data[idx]; // Random access to global_data based on
                              // shuffled indices
    sum += local;
  }
  *d_sum = sum;
}

__global__ void SingleLoad(volatile int *global_data, int *indices, int n,
                           int *d_sum) {
  int local;
  int sum = 0;
  int idx;

  for (int i = 0; i < n; i++) {
    idx = indices[i];
    local = global_data[idx]; // Random access to global_data based on
                              // shuffled indices
    // local = global_data[i]; // Linear access to global_data using indices
    sum += local;
  }
  *d_sum = sum;
}

int main() {
  int *d_data, *d_indices, *d_sum;
  int *h_indices, sum;

  for (int n = 1024; n <= 16 * 16 * 1024 * 1024; n <<= 1) {
    size_t bytes = n * sizeof(int);

    // Allocate memory on host and device
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_indices, bytes);
    cudaMalloc(&d_sum, sizeof(int));
    int *h_data = (int *)malloc(bytes);
    h_indices = (int *)malloc(bytes);

    // Initialize data on host
    for (int i = 0; i < n; i++) {
      h_data[i] = i;             // Sequential numbers as data
      h_indices[i] = rand() % n; // Random indices for accessing data
    }

    // Copy data from host to device
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, bytes, cudaMemcpyHostToDevice);

    // Setup timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel for random access and measure time
    cudaEventRecord(start);
    DoubleLoad<<<1, 1>>>(d_data, d_indices, n, d_sum);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Double Access: Time taken to load %d integers: %.3f ms\n", n,
           milliseconds);

    // Linear access
    for (int i = 0; i < n; i++) {
      h_indices[i] = i; // Random indices for accessing data
    }
    cudaMemcpy(d_indices, h_indices, bytes, cudaMemcpyHostToDevice);
    // Launch kernel for linear access and measure time
    cudaEventRecord(start);
    SingleLoad<<<1, 1>>>(d_data, d_indices, n, d_sum);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    // cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Single Access: Time taken to load %d integers: %.3f ms\n", n,
           milliseconds);
    printf("-------------------------------\n");

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
