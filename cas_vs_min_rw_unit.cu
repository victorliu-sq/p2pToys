#include <cuda_runtime.h>
#include <stdio.h>

// Kernel using atomicCAS to perform operations on different memory locations
__global__ void atomic_cas_kernel(int *data, int n) {
  if (threadIdx.x == 0) {
    for (int i = 0; i < n; i++) {
      int old_value = data[i];
      atomicCAS(&data[i], old_value,
                old_value + 1); // Swap 0 to 1 at each index
    }
  }
}

// Kernel using atomicMin to perform operations on different memory locations
__global__ void atomic_min_kernel(int *data, int n) {
  if (threadIdx.x == 0) {
    for (int i = 0; i < n; i++) {
      int old_value = data[i];
      atomicMax(&data[i], old_value + 1); // Set the minimum to 1 at each index
    }
  }
}

// Kernel for sequential read and write increment
__global__ void read_write_increment_kernel(int *data, int n) {
  if (threadIdx.x == 0) {
    // for (int i = 0; i < n; i++) {
    //   int value = data[i]; // Read data
    //   data[i] = value + 1; // Write data incremented by 1
    // }
    for (int i = 0; i < n; i++) {
      int val = data[i];
      // val += 1;
      // data[i] = val;
      // atomicAdd(&data[i], 1);
      data[i] += 1;
    }
  }
}

int main() {
  const int n = 1024 * 1024; // Number of operations
  int *d_data;
  cudaMalloc(&d_data, n * sizeof(int));
  cudaMemset(d_data, 0, n * sizeof(int));

  // Setup CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds;

  // Launch atomicCAS kernel with only one thread and measure time
  cudaEventRecord(start);
  atomic_cas_kernel<<<1, 1>>>(d_data, n); // One block, one thread
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time for atomicCAS on different memory locations with one thread: "
         "%.5f ms\n",
         milliseconds);

  // Reset data for the next test
  cudaMemset(d_data, 0, n * sizeof(int));

  // Launch atomicMin kernel with only one thread and measure time
  cudaEventRecord(start);
  atomic_min_kernel<<<1, 1>>>(d_data, n); // One block, one thread
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time for atomicMin on different memory locations with one thread: "
         "%.5f ms\n",
         milliseconds);

  // Reset data for the next test
  cudaMemset(d_data, 0, n * sizeof(int));

  // Launch read and write increment kernel with only one thread and measure
  // time
  cudaEventRecord(start);
  read_write_increment_kernel<<<1, 1>>>(d_data, n); // One block, one thread
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time for read and write increment on different memory locations with "
         "one thread: %.5f ms\n",
         milliseconds);

  // Cleanup
  cudaFree(d_data);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
