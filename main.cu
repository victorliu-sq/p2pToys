#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time()
#include <algorithm> // For std::swap
#include <vector>

__global__ void findMinValue(int* data, int* result, int startIdx, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int value = data[startIdx + idx];
        atomicMin(result, value);
    }
}

cudaError_t launchKernelOnGPU(int gpuId, int* data, int* result, int startIdx, int count) {
    cudaSetDevice(gpuId);
    int threadsPerBlock = 256;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    findMinValue<<<blocksPerGrid, threadsPerBlock>>>(data, result, startIdx, count);
    return cudaGetLastError(); // Return the last error from CUDA runtime
}

int main() {
    int dataSize = 60000;
    int* data;
    int* result;
    int initialValue = 100000;
    int segmentSize = 30000; // Each GPU works on 30000 elements

    cudaMallocManaged(&data, dataSize * sizeof(int));
    cudaMallocManaged(&result, sizeof(int));

    // Seed the random number generator to get different results each time
    srand(static_cast<unsigned int>(time(0)));

    // Initialize data array with random values
    for (int i = 0; i < dataSize; ++i) {
        data[i] = rand() % 60000 + 1; // Random integers between 1 and 100000
    }

    *result = initialValue;

    // Find the minimum value in the first half on the CPU
    int minFirstHalf = INT_MAX;
    for (int i = 0; i < dataSize / 2; ++i) {
        if (data[i] < minFirstHalf) {
            minFirstHalf = data[i];
        }
    }

    // Find the minimum value in the second half on the CPU
    int minSecondHalf = INT_MAX;
    for (int i = dataSize / 2; i < dataSize; ++i) {
        if (data[i] < minSecondHalf) {
            minSecondHalf = data[i];
        }
    }

    std::cout << "Minimum value in the first half (CPU): " << minFirstHalf << std::endl;
    std::cout << "Minimum value in the second half (CPU): " << minSecondHalf << std::endl;


    // Launch kernels on two GPUs simultaneously
    cudaError_t err;
    err = launchKernelOnGPU(0, data, result, 0, segmentSize); // First half of the data
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch kernel on GPU 0: " << cudaGetErrorString(err) << std::endl;
    }

    err = launchKernelOnGPU(1, data, result, segmentSize, segmentSize); // Second half
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch kernel on GPU 1: " << cudaGetErrorString(err) << std::endl;
    }



    // Wait for GPU 0 to finish
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    // Wait for GPU 1 to finish
    cudaSetDevice(1);
    cudaDeviceSynchronize();

    std::cout << "The minimum value found is: " << *result << std::endl;

    cudaFree(data);
    cudaFree(result);

    return 0;
}

