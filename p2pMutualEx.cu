#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cout << "API call failure \"" #func "\" with " << rt << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      throw;                                                                   \
    }                                                                          \
  } while (0);


// GPU1 increments elements by 1
__global__ void incrementByOneKernel(int *data, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        data[idx] += 1;
    }
    if (idx == 0) { // Print from a single thread to avoid clutter
        printf("GPU1 incremented data.\n");
    }
}

// GPU2 increments elements by 2
__global__ void incrementByTwoKernel(int *data, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        data[idx] += 2;
    }
    if (idx == 0) { // Print from a single thread to avoid clutter
        printf("GPU2 incremented data.\n");
    }
}

int main() {
    int *d_gpu0Data, *d_gpu1Data;
    int size = 1024; // Example size
    size_t bytes = size * sizeof(int);

    // Initialize CUDA P2P
    int canAccessPeer01, canAccessPeer10;
    cudaDeviceCanAccessPeer(&canAccessPeer01, 0, 1);
    cudaDeviceCanAccessPeer(&canAccessPeer10, 1, 0);
    if (!(canAccessPeer01 && canAccessPeer10)) {
        std::cerr << "P2P access not supported between the GPUs.\n";
        return EXIT_FAILURE;
    }

    cudaSetDevice(0);
    CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));
    cudaMalloc(&d_gpu0Data, bytes);

    cudaSetDevice(1);
    CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0));
    cudaMalloc(&d_gpu1Data, bytes);

    // GPU1 increments data on GPU2 by 1
    cudaSetDevice(1); // Switch to GPU1's context to access GPU2's memory
    incrementByOneKernel<<<(size + 255) / 256, 256>>>(d_gpu1Data, size);

    CUDA_CHECK(cudaGetLastError());

    // GPU2 increments data on GPU1 by 2
    cudaSetDevice(0); // Switch to GPU2's context to access GPU1's memory
    incrementByTwoKernel<<<(size + 255) / 256, 256>>>(d_gpu0Data, size);

    CUDA_CHECK(cudaGetLastError());

    // Wait for GPUs to finish
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaSetDevice(1);
    cudaDeviceSynchronize();

    // Cleanup
    cudaSetDevice(0);
    cudaFree(d_gpu0Data);
    cudaDeviceDisablePeerAccess(1);

    cudaSetDevice(1);
    cudaFree(d_gpu1Data);
    cudaDeviceDisablePeerAccess(0);

    return 0;
}
