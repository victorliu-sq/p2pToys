#include <cuda_runtime.h>
#include <iostream>

__global__ void faultyUpdateWithDelay(int* counter) {
 	for (int i = 0; i < 1000000;i++);
    int oldValue, newValue;
    do {
        oldValue = *counter; // Read the current value of the counter
        newValue = oldValue; // Prepare the new value based on some condition
        if (oldValue % 2 == 0) {
            newValue += 3; // If the counter is even, add 3
        } else {
            newValue += 1; // If the counter is odd, just add 1
        }
        // Attempt to update the counter atomically
    } while (atomicCAS(counter, oldValue, newValue) != oldValue);
}

__global__ void faultyUpdate(int* counter) {
    int oldValue, newValue;
    do {
        oldValue = *counter; // Read the current value of the counter
        newValue = oldValue; // Prepare the new value based on some condition
        if (oldValue % 2 == 0) {
            newValue += 3; // If the counter is even, add 3
        } else {
            newValue += 1; // If the counter is odd, just add 1
        }
        // Attempt to update the counter atomically
    } while (atomicCAS(counter, oldValue, newValue) != oldValue);
}

int main() {
    int* counter;
    cudaMallocManaged(&counter, sizeof(int));
    *counter = 0; // Initialize counter

    int numBlocks = 10; // Assume we have 2 blocks for simplicity
    int threadsPerBlock = 1024; // Max number of threads per block

    // Launch the kernel on GPU 0
    cudaSetDevice(0);
    faultyUpdateWithDelay<<<numBlocks, threadsPerBlock>>>(counter);

    // Launch the same kernel on GPU 1 (assuming a multi-GPU setup)
    cudaSetDevice(1);
    faultyUpdate<<<numBlocks, threadsPerBlock>>>(counter);

    // Wait for both GPUs to finish
    cudaDeviceSynchronize();
    cudaSetDevice(0); // Switch back to GPU 0 and synchronize (just in case)
    cudaDeviceSynchronize();

    std::cout << "Final counter value: " << *counter << std::endl;

    cudaFree(counter);
    return 0;
}

