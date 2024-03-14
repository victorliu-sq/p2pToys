#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>

#define ARRAY_SIZE (1024 * 32)
// #define ARRAY_SIZE (20)
#define THREADS_PER_BLOCK 256 // Tune this for your specific GPU

__global__ void checkLessThanN(int *arr, int *result, int n) {
  __shared__ int partialResults[THREADS_PER_BLOCK];
  // if (blockDim.x * blockIdx.x + threadIdx.x == 0 && it % 10000 == 0) {
  //   printf("current iteration is %d\n", it);
  // printf("n-1th element is %d\n", arr[n - 1]);
  // }
  // int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int lessThanN = 1; // Use 1 for true, 0 for false
  for (int i = index; i < n; i += stride) {
    if (arr[i] >= n) {
      // printf("i: %d is arr[i]: %d\n", i, arr[i]);
      lessThanN = 0;
      break;
    }
  }

  partialResults[threadIdx.x] = lessThanN;

  __syncthreads();

  // Reduce within the block
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      partialResults[threadIdx.x] &= partialResults[threadIdx.x + s];
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAnd(result, partialResults[0]);
  }
  __syncthreads();
}

void launchKernel(int *d_arr, int *d_result, int n, int blocks,
                  int threadsPerBlock, cudaEvent_t start, cudaEvent_t stop) {
  int it = 1;
  float totalGpuTime = 0;
  while (*d_result == 0) {
    *d_result = 1;
    if (it % 1000 == 0) {
      std::cout << "Iteration: " << it << std::endl;
    }
    cudaEventRecord(start);
    checkLessThanN<<<blocks, threadsPerBlock>>>(d_arr, d_result, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    totalGpuTime += milliseconds;
    cudaDeviceSynchronize(); // Wait for the kernel to finish
    it++;
  }
  std::cout << "Kernel launched and completed." << std::endl;
  std::cout << "Total GPU Time: " << totalGpuTime << " ms\n";
  std::cout << "Amortized GPU Time per Kernel Launch: "
            << totalGpuTime / (it - 1) << " ms\n";
}

void updateArray(int *d_arr, int n) {
  // Simulate some work on the host by sleeping for 2 seconds
  std::this_thread::sleep_for(std::chrono::seconds(2));
  for (int i = 0; i < n; ++i) {
    d_arr[i] = i; // Example initialization
  }
  std::cout << "Set all elements less than n" << std::endl;
}

int main() {
  const int n = ARRAY_SIZE; // Example array size
  const int threadsPerBlock = 256;
  const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
  int *d_arr, *d_result;

  // Allocate device memory
  cudaMallocManaged(&d_arr,
                    n * sizeof(int)); // Using unified memory for simplicity
  cudaMallocManaged(&d_result, sizeof(int));

  // Initialize d_arr and d_result as needed...

  for (int i = 0; i < n; ++i) {
    d_arr[i] = n; // Example initialization
  }
  // d_arr[n - 1] = n;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Launch both operations in parallel threads
  std::thread kernelThread(launchKernel, d_arr, d_result, n, blocks,
                           threadsPerBlock, start, stop);
  std::thread updateThread(updateArray, d_arr, n);

  // Wait for both threads to complete
  kernelThread.join();
  updateThread.join();

  std::cout << "All elements < " << n << ": "
            << (*d_result == 1 ? "TRUE" : "FALSE") << std::endl;

  // Cleanup
  cudaFree(d_arr);
  cudaFree(d_result);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
