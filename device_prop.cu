#include <cuda_runtime.h>
#include <stdio.h>

int main() {
  int deviceCount = 0;
  cudaError_t error = cudaGetDeviceCount(&deviceCount);

  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    return -1;
  }

  for (int device = 0; device < deviceCount; device++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    printf("Device %d: %s\n", device, deviceProp.name);
    printf("  Total Global Memory: %.2f GB\n",
           (float)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));
    printf("  L2 Cache Size: %.2f MB\n",
           (float)deviceProp.l2CacheSize / (1024 * 1024));

    // L1 Cache Size is not directly exposed via device properties. You need to
    // infer it from the compute capability if necessary, as it can vary by
    // architecture.
    int l1CacheSizePerSM;
    if (deviceProp.major >= 7) { // For Volta and newer (simplified example)
      l1CacheSizePerSM =
          128 *
          1024; // Assuming 128KB L1 cache per Streaming Multiprocessor (SM)
    } else {
      l1CacheSizePerSM =
          48 * 1024; // For older architectures, a common size was 48KB per SM
    }
    printf("  Approx L1 Cache Size per SM: %d KB\n", l1CacheSizePerSM / 1024);
    printf("  Number of SMs: %d\n", deviceProp.multiProcessorCount);
    printf("  Total Approx L1 Cache Size: %.2f MB\n",
           (float)(l1CacheSizePerSM * deviceProp.multiProcessorCount) /
               (1024 * 1024));
  }

  return 0;
}
