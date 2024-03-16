#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>

void checkCudaErrors(cudaError_t result) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result)
              << std::endl;
    exit(-1);
  }
}

void enableP2P(int gpu0, int gpu1) {
  int canAccessPeer;
  checkCudaErrors(cudaDeviceCanAccessPeer(&canAccessPeer, gpu0, gpu1));
  if (canAccessPeer) {
    checkCudaErrors(cudaSetDevice(gpu0));
    checkCudaErrors(cudaDeviceEnablePeerAccess(gpu1, 0));
    checkCudaErrors(cudaSetDevice(gpu1));
    checkCudaErrors(cudaDeviceEnablePeerAccess(gpu0, 0));
  } else {
    std::cerr << "P2P access is not supported between GPU " << gpu0
              << " and GPU " << gpu1 << std::endl;
    exit(-1);
  }
}

__global__ void detectChangeKernel(int *data, int expectedValue) {
  int it = 0;
  while (*data != expectedValue) {
    // Loop until the data changes to the expected value
    it++;
    if (it % 1000000 == 0) {
      printf("still loop\n");
    }
  }
  printf("Change detected in GPU1's memory by GPU1\n");
}

__global__ void updateValueKernel(int *data, int newValue) { *data = newValue; }

void launchDetectChangeKernel(int *dataGPU1, int expectedValue) {
  // Assuming cudaSetDevice(0) has already been called for this thread
  cudaSetDevice(0);
  detectChangeKernel<<<1, 1>>>(dataGPU1, expectedValue);
  cudaDeviceSynchronize(); // Wait for the kernel to finish
}

void updateValueAfterDelay(int *dataGPU1, int newValue, int delayInSeconds) {
  // Wait for specified seconds
  std::this_thread::sleep_for(std::chrono::seconds(delayInSeconds));

  cudaSetDevice(1);
  // Assuming cudaSetDevice(1) has been set for this thread if necessary
  updateValueKernel<<<1, 1>>>(dataGPU1, newValue);
  cudaDeviceSynchronize(); // Ensure the update is completed
  printf("Change has been made to in GPU1's memory by GPU1\n");
}

int main() {
  int *dataGPU0;
  const int expectedValue = 123;
  const int delayInSeconds = 2;

  enableP2P(0, 1);

  // Set device to GPU1 and allocate memory
  cudaSetDevice(0);
  cudaMalloc(&dataGPU0, sizeof(int));
  cudaMemset(dataGPU0, 0, sizeof(int));

  // Create a thread to launch the detectChangeKernel on GPU1
  std::thread detectThread(launchDetectChangeKernel, dataGPU0, expectedValue);

  // Create another thread that waits for 2 seconds then updates the value on
  // GPU1 from GPU2
  std::thread updateThread(updateValueAfterDelay, dataGPU0, expectedValue,
                           delayInSeconds);

  // Wait for both threads to finish
  detectThread.join();
  updateThread.join();

  // Cleanup
  cudaFree(dataGPU0);
  cudaSetDevice(0);
  cudaDeviceReset();
  cudaSetDevice(1);
  cudaDeviceReset();

  return 0;
}
