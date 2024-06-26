#include <cuda_runtime.h>
#include <iostream>

__global__ void copyKernel(int *d_src, int *d_dst, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    d_dst[idx] = d_src[idx];
  }
}

__global__ void zeroCopyKernel(int *d_src, int *d_dst, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    d_dst[idx] = d_src[idx];
  }
}

void scenario1(int *h_list, int n) {
  int *d_list1, *d_list2;
  size_t size = n * sizeof(int);

  cudaMalloc((void **)&d_list1, size);
  cudaMalloc((void **)&d_list2, size);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // Copy list from host to device
  cudaMemcpy(d_list1, h_list, size, cudaMemcpyHostToDevice);

  // Launch kernel to copy list to another device memory
  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  copyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_list1, d_list2, n);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Scenario 1 time: " << milliseconds << " ms" << std::endl;

  cudaFree(d_list1);
  cudaFree(d_list2);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

void scenario2(int *h_list, int n) {
  int *d_list2;
  int *h_pinned_list;
  size_t size = n * sizeof(int);

  cudaHostAlloc((void **)&h_pinned_list, size, cudaHostAllocMapped);
  cudaHostGetDevicePointer((void **)&d_list2, h_pinned_list, 0);

  for (int i = 0; i < n; ++i) {
    h_pinned_list[i] = h_list[i];
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // Launch kernel to copy list to another device memory
  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  zeroCopyKernel<<<blocksPerGrid, threadsPerBlock>>>(
      d_list2, d_list2, n); // Self copy for demonstration

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Scenario 2 time: " << milliseconds << " ms" << std::endl;

  cudaFreeHost(h_pinned_list);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

int main() {
  int n = 1 << 20; // Example size
  int *h_list = new int[n];

  // Initialize the list
  for (int i = 0; i < n; ++i) {
    h_list[i] = i;
  }

  scenario1(h_list, n);
  scenario2(h_list, n);

  delete[] h_list;
  return 0;
}