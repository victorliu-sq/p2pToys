#include <curand.h>
#include <curand_kernel.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void find_min_atomicCAS(int *array, int n, int *result) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < n) {
    int val = array[index];
    int old = *result;
    while (old > val && atomicCAS(result, old, val) != old) {
      old = *result;
    }
  }
}

__global__ void find_min_atomicMin(int *array, int n, int *result) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < n) {
    atomicMin(result, array[index]);
  }
}

__global__ void setup_kernel(curandState *state, unsigned long seed) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, id, 0, &state[id]);
}

__global__ void randomize_array(int *array, int n, curandState *state) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < n) {
    int pos = curand(&state[id]) % n;
    int tmp = array[id];
    array[id] = array[pos];
    array[pos] = tmp;
  }
}

int main() {
  int *d_array, *d_result;
  int h_result;
  int *h_array;
  // for (int n = 1024; n <= n * 1024; n *= 2) {
  int n = 1000000;
  printf("for size n: %d\n", n);
  h_array = (int *)malloc(n * sizeof(int));

  // Allocate device memory
  cudaMalloc(&d_array, n * sizeof(int));
  cudaMalloc(&d_result, sizeof(int));

  // Initialize array with some values and shuffle
  for (int i = 0; i < n; i++) {
    h_array[i] = n - i; // Example data
  }

  // Setup CURAND for randomizing array
  curandState *devStates;
  cudaMalloc(&devStates, n * sizeof(curandState));
  setup_kernel<<<(n + 255) / 256, 256>>>(devStates, time(NULL));
  cudaMemcpy(d_array, h_array, n * sizeof(int), cudaMemcpyHostToDevice);
  randomize_array<<<(n + 255) / 256, 256>>>(d_array, n, devStates);

  cudaEvent_t start, stop;
  float timeCAS, timeMin;

  // Initialize CUDA events
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Set initial result high
  h_result = INT_MAX;
  cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);

  // Measure atomicCAS version
  cudaEventRecord(start);
  find_min_atomicCAS<<<(n + 255) / 256, 256>>>(d_array, n, d_result);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timeCAS, start, stop);
  cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
  printf("Minimum (atomicCAS): %d, Time: %f ms\n", h_result, timeCAS);

  // Reset result for next kernel
  h_result = INT_MAX;
  cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);

  // Measure atomicMin version
  cudaEventRecord(start);
  find_min_atomicMin<<<(n + 255) / 256, 256>>>(d_array, n, d_result);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timeMin, start, stop);
  cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
  printf("Minimum (atomicMin): %d, Time: %f ms\n", h_result, timeMin);

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_array);
  cudaFree(d_result);
  cudaFree(devStates);
  free(h_array);

  printf("------------------------------------------\n");
  // }

  return 0;
}
