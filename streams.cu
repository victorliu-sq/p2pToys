#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

#define N 100000000
#define THREADS_PER_BLOCK 256

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

__global__ void simpleKernel1(float *data) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    data[idx] += 1.0f;
  }
}

__global__ void simpleKernel2(float *data) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    data[idx] *= 2.0f;
  }
}

void singleStreamTransferAndCompute(float *h_data, float *d_data1,
                                    float *d_data2) {
  cudaStream_t stream;
  checkCudaError(cudaStreamCreate(&stream), "Creating stream");

  // Transfer data to GPU1
  checkCudaError(cudaSetDevice(0), "Setting device 0");
  checkCudaError(cudaMemcpyAsync(d_data1, h_data, N * sizeof(float),
                                 cudaMemcpyHostToDevice, stream),
                 "MemcpyAsync to GPU1");

  // Transfer data to GPU2
  checkCudaError(cudaSetDevice(1), "Setting device 1");
  checkCudaError(cudaMemcpyAsync(d_data2, h_data, N * sizeof(float),
                                 cudaMemcpyHostToDevice, stream),
                 "MemcpyAsync to GPU2");

  // Synchronize the stream
  checkCudaError(cudaStreamSynchronize(stream), "Stream synchronize");

  // Launch kernels on GPU1
  checkCudaError(cudaSetDevice(0), "Setting device 0");
  simpleKernel1<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                  THREADS_PER_BLOCK>>>(d_data1);
  checkCudaError(cudaStreamSynchronize(stream),
                 "Stream synchronize after kernel1 on GPU1");
  simpleKernel2<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                  THREADS_PER_BLOCK>>>(d_data1);
  checkCudaError(cudaStreamSynchronize(stream),
                 "Stream synchronize after kernel2 on GPU1");

  // Launch kernels on GPU2
  checkCudaError(cudaSetDevice(1), "Setting device 1");
  simpleKernel1<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                  THREADS_PER_BLOCK>>>(d_data2);
  checkCudaError(cudaStreamSynchronize(stream),
                 "Stream synchronize after kernel1 on GPU2");
  simpleKernel2<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                  THREADS_PER_BLOCK>>>(d_data2);
  checkCudaError(cudaStreamSynchronize(stream),
                 "Stream synchronize after kernel2 on GPU2");

  // Clean up
  checkCudaError(cudaStreamDestroy(stream), "Destroying stream");
}

void multiStreamTransferAndCompute(float *h_data, float *d_data1,
                                   float *d_data2) {
  cudaStream_t stream1, stream2;
  checkCudaError(cudaStreamCreate(&stream1), "Creating stream1");
  checkCudaError(cudaStreamCreate(&stream2), "Creating stream2");

  // Transfer data to GPU1 using stream1
  checkCudaError(cudaSetDevice(0), "Setting device 0");
  checkCudaError(cudaMemcpyAsync(d_data1, h_data, N * sizeof(float),
                                 cudaMemcpyHostToDevice, stream1),
                 "MemcpyAsync to GPU1");

  // Transfer data to GPU2 using stream2
  checkCudaError(cudaSetDevice(1), "Setting device 1");
  checkCudaError(cudaMemcpyAsync(d_data2, h_data, N * sizeof(float),
                                 cudaMemcpyHostToDevice, stream2),
                 "MemcpyAsync to GPU2");

  // Launch kernels on GPU1 using stream1
  checkCudaError(cudaSetDevice(0), "Setting device 0");
  simpleKernel1<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                  THREADS_PER_BLOCK, 0, stream1>>>(d_data1);
  simpleKernel2<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                  THREADS_PER_BLOCK, 0, stream1>>>(d_data1);

  // Launch kernels on GPU2 using stream2
  checkCudaError(cudaSetDevice(1), "Setting device 1");
  simpleKernel1<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                  THREADS_PER_BLOCK, 0, stream2>>>(d_data2);
  simpleKernel2<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                  THREADS_PER_BLOCK, 0, stream2>>>(d_data2);

  // Synchronize the streams
  checkCudaError(cudaStreamSynchronize(stream1), "Stream1 synchronize");
  checkCudaError(cudaStreamSynchronize(stream2), "Stream2 synchronize");

  // Clean up
  checkCudaError(cudaStreamDestroy(stream1), "Destroying stream1");
  checkCudaError(cudaStreamDestroy(stream2), "Destroying stream2");
}

int main() {
  float *h_data;
  float *d_data1, *d_data2;

  // Allocate pinned host memory
  checkCudaError(cudaMallocHost((void **)&h_data, N * sizeof(float)),
                 "Allocating pinned host memory");

  // Initialize host data
  for (int i = 0; i < N; i++) {
    h_data[i] = static_cast<float>(i);
  }

  // Allocate memory on GPU1
  checkCudaError(cudaSetDevice(0), "Setting device 0");
  checkCudaError(cudaMalloc((void **)&d_data1, N * sizeof(float)),
                 "Allocating d_data1");

  // Allocate memory on GPU2
  checkCudaError(cudaSetDevice(1), "Setting device 1");
  checkCudaError(cudaMalloc((void **)&d_data2, N * sizeof(float)),
                 "Allocating d_data2");

  // Measure time for single stream version
  auto start = std::chrono::high_resolution_clock::now();
  singleStreamTransferAndCompute(h_data, d_data1, d_data2);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Single stream time: " << elapsed.count() << " seconds"
            << std::endl;

  // Clean up device memory
  checkCudaError(cudaFree(d_data1), "Freeing d_data1");
  checkCudaError(cudaFree(d_data2), "Freeing d_data2");

  // Allocate memory again for multi-stream version
  checkCudaError(cudaSetDevice(0), "Setting device 0");
  checkCudaError(cudaMalloc((void **)&d_data1, N * sizeof(float)),
                 "Allocating d_data1 again");
  checkCudaError(cudaSetDevice(1), "Setting device 1");
  checkCudaError(cudaMalloc((void **)&d_data2, N * sizeof(float)),
                 "Allocating d_data2 again");

  // Measure time for multi-stream version
  start = std::chrono::high_resolution_clock::now();
  multiStreamTransferAndCompute(h_data, d_data1, d_data2);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "Multi-stream time: " << elapsed.count() << " seconds"
            << std::endl;

  // Clean up
  checkCudaError(cudaFree(d_data1), "Freeing d_data1");
  checkCudaError(cudaFree(d_data2), "Freeing d_data2");
  checkCudaError(cudaFreeHost(h_data), "Freeing pinned host memory");

  return 0;
}