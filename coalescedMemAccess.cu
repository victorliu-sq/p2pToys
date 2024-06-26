#include <cuda_runtime.h>
#include <iostream>

#define N 1024 // Assuming N is the width and height of the matrix

__global__ void readMatrixRowByRow(int *matrix, int *result, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < n * n) {
    // Coalesced memory access: each thread reads one element of the matrix row
    // by row
    int row = tid / n;
    int col = tid % n;
    result[tid] = matrix[row * n + col];
  }
}

__global__ void readMatrixColumnByColumn(int *matrix, int *result, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < n * n) {
    // Non-coalesced memory access: each thread reads one element of the matrix
    // column by column
    int row = tid % n;
    int col = tid / n;
    result[tid] = matrix[row * n + col];
  }
}

int main() {
  int size = N * N * sizeof(int);
  int *h_matrix = (int *)malloc(size);
  int *h_result = (int *)malloc(size);

  // Initialize the matrix with some values
  for (int i = 0; i < N * N; ++i) {
    h_matrix[i] = i;
  }

  int *d_matrix, *d_result;
  cudaMalloc((void **)&d_matrix, size);
  cudaMalloc((void **)&d_result, size);

  cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;

  // Measure the time for the row-by-row kernel
  cudaEvent_t start, stop;
  float rowTime, colTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  readMatrixRowByRow<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_result, N);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&rowTime, start, stop);

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
  std::cout << "Time for column-by-column read: " << colTime << " ms"
            << std::endl;

  cudaFree(d_matrix);
  cudaFree(d_result);
  free(h_matrix);
  free(h_result);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}