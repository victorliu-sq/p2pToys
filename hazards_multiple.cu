#include <cuda_runtime.h>
#include <stdio.h>

__device__ unsigned int count = 0;
__shared__ bool isLastBlockDone;

// Device function to calculate the partial sum of the input array
__device__ float calculatePartialSum(const float *array, unsigned int N) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0;
  while (i < N) {
    sum += array[i];
    i += blockDim.x * gridDim.x;
  }
  sdata[tid] = sum;
  __syncthreads();

  // Reduction within the block
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  return (tid == 0) ? sdata[0] : 0;
}

// Device function to calculate the total sum from partial sums
__device__ float calculateTotalSum(volatile float *result) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  float sum = 0;
  if (tid < gridDim.x) {
    sum = result[tid];
  }
  sdata[tid] = sum;
  __syncthreads();

  // Reduction within the block
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  return (tid == 0) ? sdata[0] : 0;
}

// Kernel to compute the sum of an array using multiple blocks
__global__ void sumWFence(const float *array, unsigned int N,
                          volatile float *result) {
  // Each block sums a subset of the input array
  float partialSum = calculatePartialSum(array, N);

  if (threadIdx.x == 0) {
    // Store the partial sum to global memory
    result[blockIdx.x] = partialSum;
    __threadfence();

    // Increment the count and check if this is the last block
    unsigned int value = atomicInc(&count, gridDim.x);
    isLastBlockDone = (value == (gridDim.x - 1));
  }

  // Synchronize to ensure all threads read the correct isLastBlockDone value
  __syncthreads();

  if (isLastBlockDone) {
    // The last block sums the partial sums from all blocks
    float totalSum = calculateTotalSum(result);

    if (threadIdx.x == 0) {
      // Store the total sum back to the result and reset count
      result[0] = totalSum;
      count = 0;
    }
  }
}

// Kernel to compute the sum of an array using multiple blocks
__global__ void sumWOFence(const float *array, unsigned int N,
                           volatile float *result) {
  // Each block sums a subset of the input array
  float partialSum = calculatePartialSum(array, N);

  if (threadIdx.x == 0) {
    // Store the partial sum to global memory
    result[blockIdx.x] = partialSum;
    // __threadfence();

    // Increment the count and check if this is the last block
    unsigned int value = atomicInc(&count, gridDim.x);
    isLastBlockDone = (value == (gridDim.x - 1));
  }

  // Synchronize to ensure all threads read the correct isLastBlockDone value
  __syncthreads();

  if (isLastBlockDone) {
    // The last block sums the partial sums from all blocks
    float totalSum = calculateTotalSum(result);

    if (threadIdx.x == 0) {
      // Store the total sum back to the result and reset count
      result[0] = totalSum;
      count = 0;
    }
  }
}

int main() {
  const int N = 1024 * 256;  // Size of the array
  const int blockSize = 256; // Threads per block
  const int numBlocks = (N + blockSize - 1) / blockSize;

  float *array;
  float *d_array;
  float *d_result_with_fence;
  float *d_result_without_fence;
  float result_with_fence, result_without_fence;

  // Allocate host and device memory
  array = new float[N];
  for (int i = 0; i < N; i++) {
    array[i] = 1.0f; // Example data
  }

  cudaMalloc(&d_array, N * sizeof(float));
  cudaMalloc(&d_result_with_fence, numBlocks * sizeof(float));
  cudaMalloc(&d_result_without_fence, numBlocks * sizeof(float));
  cudaMemcpy(d_array, array, N * sizeof(float), cudaMemcpyHostToDevice);

  int it = 1;
  do {
    cudaMemset((void *)d_result_with_fence, 0, numBlocks * sizeof(float));
    cudaMemset((void *)d_result_without_fence, 0, numBlocks * sizeof(float));

    // Launch both kernels
    sumWFence<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(
        d_array, N, d_result_with_fence);
    sumWOFence<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(
        d_array, N, d_result_without_fence);

    // Copy results back to host
    cudaMemcpy(&result_with_fence, d_result_with_fence, sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&result_without_fence, d_result_without_fence, sizeof(float),
               cudaMemcpyDeviceToHost);
    printf("current iteration is %d, sumWFence: %f, sumWOFence: %f\n", it,
           result_with_fence, result_without_fence);
    it++;
  } while (result_with_fence == result_without_fence);

  printf("Discrepancy detected: With Fence = %f, Without Fence = %f\n",
         result_with_fence, result_without_fence);

  // Cleanup
  cudaFree(d_array);
  cudaFree((void *)d_result_with_fence);
  cudaFree((void *)d_result_without_fence);
  delete[] array;

  return 0;
}