#include <cuda_runtime.h>
#include <stdio.h>

// Kernel using Cache-All (CA) load
__global__ void load_ca(int *data, int n) {
  int cur = 0;
  int sum = 0;
  for (int i = 0; i < n; i++) {
    // asm("ld.ca.s32 %0, [%1];" : "=r"(cur) : "l"(&data[i]));
    cur = __ldg(&data[i]);
    sum += cur;
    __syncwarp();
    // __threadfence_block();
  }
  data[0] = sum;
}

// Kernel using Cache-Global (CG) load
__global__ void load_cg(volatile int *data, int n) {
  int cur = 0;
  int sum = 0;
  for (int i = 0; i < n; i++) {
    asm("ld.cg.s32 %0, [%1];" : "=r"(cur) : "l"(&data[i]));
    sum += cur;
    __syncwarp();
    // __threadfence_block();
  }
  data[0] = sum;
}

__global__ void load_cv(volatile int *data, int n) {
  int cur = 0;
  int sum = 0;
  for (int i = 0; i < n; i++) {
    asm("ld.cv.s32 %0, [%1];" : "=r"(cur) : "l"(&data[i]));
    sum += cur;
    __syncwarp();
    // __threadfence_block();
  }
  data[0] = sum;
}

// Kernel using atomicCAS to perform operations on different memory locations
__global__ void atomic_cas_kernel(int *data, int n) {
  int sum = 0;
  for (int i = 0; i < n; i++) {
    sum += atomicCAS(&data[i], 0, 1); // Swap 0 to 1 at each index
    // __syncwarp();
    // __threadfence_block();
  }
  data[0] = sum;
}

// Kernel using atomicMin to perform operations on different memory locations
__global__ void atomic_min_kernel(int *data, int n) {
  int sum = 0;
  for (int i = 0; i < n; i++) {
    sum += atomicMin(&data[i], 1); // Set the minimum to 1 at each index
    __syncwarp();
    // __threadfence_block();
  }

  data[0] = sum;
}

// Kernel for sequential read and write increment
__global__ void read_write_increment_kernel(int *data, int n) {
  int val = 0;
  for (int i = 0; i < n; i++) {
    // data[i] = 1;
    // asm("st.cg.s32 [%0], %1;"
    //     : // No outputs
    //     : "l"(data + 0),
    //       "r"(val) // Inputs - "l" for a 64-bit address, "r"
    //                // for a 32-bit integer
    //     : "memory" // Tells the compiler that memory is being
    //                // modified
    // );
    asm("st.wt.s32 [%0], %1;"
        : // No outputs
        : "l"(data + 0),
          "r"(val) // Inputs - "l" for a 64-bit address, "r"
                   // for a 32-bit integer
        : "memory" // Tells the compiler that memory is being
                   // modified
    );
    __syncwarp();
    // __threadfence();
    val++;
  }
}

void measureKernelPerformance(int *d_data, int n, void (*kernel)(int *, int),
                              const char *kernelName) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;

  cudaEventRecord(start);
  kernel<<<1, 1>>>(d_data, n); // Launch with one block of one thread
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("%s kernel execution time: %.5f ms\n", kernelName, milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

void measureKernelPerformanceV(int *d_data, int n,
                               void (*kernel)(volatile int *, int),
                               const char *kernelName) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;

  cudaEventRecord(start);
  kernel<<<1, 1>>>(d_data, n); // Launch with one block of one thread
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("%s kernel execution time: %.5f ms\n", kernelName, milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

int main() {
  const int n = 1000 * 1000 * 10; // Number of elements
  int *d_data, *d_result;

  // Allocate memory
  cudaMalloc(&d_data, n * sizeof(int));

  // Initialize data
  int *h_data = new int[n];
  for (int i = 0; i < n; ++i) {
    h_data[i] = i;
  }
  cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

  // Measure each kernel's performance
  measureKernelPerformance(d_data, n, load_ca, "Cache-All (CA)");
  measureKernelPerformanceV(d_data, n, load_cg, "Cache-Global (CG)");
  measureKernelPerformanceV(d_data, n, load_cv, "Global (CV)");
  measureKernelPerformance(d_data, n, atomic_cas_kernel, "Atomic CAS");
  measureKernelPerformance(d_data, n, atomic_min_kernel, "Atomic Min");
  measureKernelPerformance(d_data, n, read_write_increment_kernel,
                           "Read-Write Increment");

  // Cleanup
  cudaFree(d_data);
  cudaFree(d_result);
  delete[] h_data;

  return 0;
}
