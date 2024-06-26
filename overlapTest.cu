#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

// Kernel to add a value to each element in the array
__global__ void addValue(int *data, int value, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] += value;
  }
}

void checkValues(int *data, int size, int expected_value, const char *msg) {
  for (int i = 0; i < size; i++) {
    if (data[i] != expected_value) {
      std::cerr << msg << ": Verification failed at index " << i << "!\n";
      return;
    }
  }
  std::cout << msg << ": Verification passed!\n";
}

void synchronousProcedure(int *h_data1, int *h_data2, int *d_data1,
                          int *d_data2, int size, int valueToAdd) {
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  auto start = std::chrono::high_resolution_clock::now();
  auto copy1_start = start;

  // Copy the first array to the device
  cudaMemcpy(d_data1, h_data1, size * sizeof(int), cudaMemcpyHostToDevice);

  auto copy1_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> copy1_time =
      copy1_end - copy1_start;

  auto copy2_start = copy1_end;

  // Copy the second array to the device
  cudaMemcpy(d_data2, h_data2, size * sizeof(int), cudaMemcpyHostToDevice);

  auto copy2_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> copy2_time =
      copy2_end - copy2_start;

  auto kernel_start = copy2_end;

  // Execute kernel only on the first array
  addValue<<<gridSize, blockSize>>>(d_data1, valueToAdd, size);
  cudaDeviceSynchronize();

  auto kernel_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> kernel_time =
      kernel_end - kernel_start;

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> total_time = end - start;

  // Copy the results back to the host
  cudaMemcpy(h_data1, d_data1, size * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_data2, d_data2, size * sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  checkValues(h_data1, size, 1 + valueToAdd, "Synchronous Procedure");

  std::cout << "Synchronous Procedure total time: " << total_time.count()
            << " ms\n";
  std::cout << "  Copy1 time: " << copy1_time.count() << " ms\n";
  std::cout << "  Copy2 time: " << copy2_time.count() << " ms\n";
  std::cout << "  Kernel time: " << kernel_time.count() << " ms\n";
}

void synchronousProcedureInsert(int *h_data1, int *h_data2, int *d_data1,
                                int *d_data2, int size, int valueToAdd) {
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  auto copy2_start = std::chrono::high_resolution_clock::now();

  // Copy the second array to the device
  cudaMemcpy(d_data2, h_data2, size * sizeof(int), cudaMemcpyHostToDevice);

  auto copy2_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> copy2_time =
      copy2_end - copy2_start;

  auto start = std::chrono::high_resolution_clock::now();

  // Copy the first array to the device
  cudaMemcpy(d_data1, h_data1, size * sizeof(int), cudaMemcpyHostToDevice);

  // Execute kernel only on the first array
  addValue<<<gridSize, blockSize>>>(d_data1, valueToAdd, size);
  cudaDeviceSynchronize();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> total_time = end - start;

  // Copy the results back to the host
  cudaMemcpy(h_data1, d_data1, size * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_data2, d_data2, size * sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  checkValues(h_data1, size, 1 + valueToAdd, "Synchronous Procedure Insert");

  std::cout << "Synchronous Procedure Insert copy2 time: " << copy2_time.count()
            << " ms\n";
  std::cout << "Synchronous Procedure Insert total time (copy1 + kernel): "
            << total_time.count() << " ms\n";
}

int main() {
  const int n = 2000000;    // Size of the array
  const int valueToAdd = 5; // Value to add to each element

  int *h_data1 = new int[n];
  int *h_data2 = new int[n];
  int *d_data1, *d_data2;

  // Initialize host data with 1s
  std::fill(h_data1, h_data1 + n, 1);
  std::fill(h_data2, h_data2 + n, 1);

  // Allocate device memory
  cudaMalloc(&d_data1, n * sizeof(int));
  cudaMalloc(&d_data2, n * sizeof(int));

  // Run the synchronous procedure
  synchronousProcedure(h_data1, h_data2, d_data1, d_data2, n, valueToAdd);

  // Reinitialize host data with 1s
  std::fill(h_data1, h_data1 + n, 1);
  std::fill(h_data2, h_data2 + n, 1);

  // Run the synchronous procedure with insert
  synchronousProcedureInsert(h_data1, h_data2, d_data1, d_data2, n, valueToAdd);

  // Cleanup
  delete[] h_data1;
  delete[] h_data2;
  cudaFree(d_data1);
  cudaFree(d_data2);

  return 0;
}