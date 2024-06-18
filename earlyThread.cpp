#include <chrono>
#include <future>
#include <iostream>
#include <thread>

// Function that simulates some work
void doWork(std::promise<int> resultPromise, int workDuration) {
  std::this_thread::sleep_for(std::chrono::seconds(workDuration));
  resultPromise.set_value(workDuration); // Set the result to the work duration
}

int main() {
  // Create promise and future to get the result from the first finished thread
  std::promise<int> promise1;
  std::promise<int> promise2;
  std::future<int> future1 = promise1.get_future();
  std::future<int> future2 = promise2.get_future();

  // Start two threads with different work durations
  std::thread t1(doWork, std::move(promise1), 6); // Work duration: 2 seconds
  std::thread t2(doWork, std::move(promise2), 5); // Work duration: 5 seconds

  // Wait for the first thread to finish
  std::future_status status1;
  std::future_status status2;
  int result = -1;

  // Wait for the first future to be ready
  bool done = false;
  while (!done) {
    status1 = future1.wait_for(std::chrono::milliseconds(1));
    status2 = future2.wait_for(std::chrono::milliseconds(1));
    if (status1 == std::future_status::ready ||
        status2 == std::future_status::ready) {
      done = true;
      if (status1 == std::future_status::ready) {
        result = future1.get();
        t2.detach(); // Abandon the second thread
      } else {
        result = future2.get();
        t1.detach(); // Abandon the first thread
      }
    }
  }

  std::cout << "First finished thread work duration: " << result << " seconds"
            << std::endl;

  // Join the thread that finished
  if (t1.joinable()) {
    std::cout << "Thread 1 is done" << std::endl;
    t1.join();
  }
  if (t2.joinable()) {
    std::cout << "Thread 2 is done" << std::endl;
    t2.join();
  }

  return 0;
}