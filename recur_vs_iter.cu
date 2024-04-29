#include <cstdio>
#include <stdio.h>

__device__ long factorial_recursive(long n) {
  if (n <= 1)
    return 1;
  else
    return n + factorial_recursive(n - 1);
}

__device__ long factorial_iterative(long n) {
  long result = 0;
  for (long i = 1; i <= n; ++i) {
    result += i;
  }
  return result;
}

__global__ void compute_factorial_recursive(long n, long *result) {
  result[0] = factorial_recursive(n);
}

__global__ void compute_factorial_iterative(long n, long *result) {
  result[0] = factorial_iterative(n);
}

int main() {
  long n;
  long *result;
  cudaMallocManaged(&result, sizeof(long));

  cudaEvent_t start, stop;
  float timeRecursive, timeIterative;

  for (n = 10; n <= 1000000; n *= 10) {
    printf("current size is %d\n", n);
    // Initialize CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Measure recursive version
    cudaEventRecord(start);
    compute_factorial_recursive<<<1, 1>>>(n, result);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&timeRecursive, start, stop);
    printf("Recursive Factorial of %d is %d, Time: %f ms\n", n, result[0],
           timeRecursive);

    // Measure iterative version
    cudaEventRecord(start);
    compute_factorial_iterative<<<1, 1>>>(n, result);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&timeIterative, start, stop);
    printf("Iterative Factorial of %d is %d, Time: %f ms\n", n, result[0],
           timeIterative);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  cudaFree(result);

  return 0;
}
