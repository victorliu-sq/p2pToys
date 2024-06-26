#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#define N 15000 // Assuming N is the width and height of the matrix

__global__ void readMatrixRowByRow(int *matrix, int *indices, int *result,
                                   int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < n * n) {
    int row = tid / n;
    int col = tid % n;
    int dummy =
        indices[row * n + col]; // Dummy read to match the number of reads
    result[tid] = matrix[row * n + col];
  }
}

__global__ void readMatrixWithIndices(int *matrix, int *indices, int *result,
                                      int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < n * n) {
    int row = tid / n;
    int col = tid % n;
    int index = indices[row * n + col];
    result[tid] = matrix[row * n + index];
  }
}

__global__ void readMatrixColumnByColumn(int *matrix, int *result, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < n * n) {
    int row = tid % n;
    int col = tid / n;
    result[tid] = matrix[row * n + col];
  }
}

void initializeRandomIndices(int *indices, int n) {
  std::vector<int> temp(n);
  for (int i = 0; i < n; ++i) {
    temp[i] = i;
  }
  std::random_device rd;
  std::mt19937 g(rd());
  for (int row = 0; row < n; ++row) {
    std::shuffle(temp.begin(), temp.end(), g);
    for (int col = 0; col < n; ++col) {
      indices[row * n + col] = temp[col];
    }
  }
}

int main() {
  int size = N * N * sizeof(int);
  int *h_matrix = (int *)malloc(size);
  int *h_result = (int *)malloc(size);
  int *h_indices = (int *)malloc(size);

  // Initialize the matrix with some values
  for (int i = 0; i < N * N; ++i) {
    h_matrix[i] = i;
  }

  // Initialize the indices with random values
  initializeRandomIndices(h_indices, N);

  int *d_matrix, *d_result, *d_indices;
  cudaMalloc((void **)&d_matrix, size);
  cudaMalloc((void **)&d_result, size);
  cudaMalloc((void **)&d_indices, size);

  cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, h_indices, size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;

  // Warm-up kernel to avoid first launch overhead
  readMatrixRowByRow<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_indices,
                                                         d_result, N);
  cudaDeviceSynchronize();

  // Measure the time for the row-by-row kernel
  cudaEvent_t start, stop;
  float rowTime, indexTime, colTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  readMatrixRowByRow<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_indices,
                                                         d_result, N);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&rowTime, start, stop);

  cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);

  // Measure the time for the index-based kernel
  cudaEventRecord(start);
  readMatrixWithIndices<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_indices,
                                                            d_result, N);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&indexTime, start, stop);

  cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);

  // Measure the time for the column-by-column kernel
  cudaEventRecord(start);
  readMatrixColumnByColumn<<<blocksPerGrid, threadsPerBlock>>>(d_matrix,
                                                               d_result, N);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&colTime, start, stop);

  cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);

  // Output the times
  std::cout << "Time for row-by-row read: " << rowTime << " ms" << std::endl;
  std::cout << "Time for index-based read: " << indexTime << " ms" << std::endl;
  std::cout << "Time for column-by-column read: " << colTime << " ms"
            << std::endl;

  cudaFree(d_matrix);
  cudaFree(d_result);
  cudaFree(d_indices);
  free(h_matrix);
  free(h_result);
  free(h_indices);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}