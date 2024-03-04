#include <cuda_runtime.h>
#include <iostream>

__global__ void incrementKernel(int *data, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        data[idx] += 1;
        printf("Data[%d] after increment: %d\n", idx, data[idx]);
    }
}

int main() {
    int *d_data_gpu1;
    int size = 512;
    size_t bytes = size * sizeof(int);

    int gpu1 = 0, gpu2 = 1;
    int canAccessPeer = 0;

    // Set device to GPU1 and allocate memory
    cudaSetDevice(gpu1);
    cudaMalloc(&d_data_gpu1, bytes);

    // Initialize data on GPU1 if necessary

    // Check if GPU2 can access GPU1 memory directly
    cudaDeviceCanAccessPeer(&canAccessPeer, gpu2, gpu1);

    if (canAccessPeer) {
        // Enable peer access from GPU2 to GPU1
        cudaSetDevice(gpu2);
        cudaDeviceEnablePeerAccess(gpu1, 0);

        // Now you can launch a kernel on GPU2 to access data on GPU1
        incrementKernel<<<(size + 255) / 256, 256>>>(d_data_gpu1, size);

        // Wait for GPU2 to finish processing
        cudaDeviceSynchronize();

        // Disable peer access if it's no longer needed
        cudaDeviceDisablePeerAccess(gpu1);
    } else {
        std::cerr << "Peer access not supported between GPU1 and GPU2." << std::endl;
    }

    // Cleanup
    cudaSetDevice(gpu1);
    cudaFree(d_data_gpu1);

    return 0;
}
