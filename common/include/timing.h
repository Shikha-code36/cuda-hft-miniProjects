#ifndef TIMING_H
#define TIMING_H

#include <chrono>
#include <string>
#include <cuda_runtime.h>

class CpuTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool running;

public:
    CpuTimer() : running(false) {}

    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        running = true;
    }

    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
        running = false;
    }

    // Get elapsed time in nanoseconds
    int64_t elapsedNanoseconds() const {
        if (running) {
            auto current_time = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                current_time - start_time).count();
        } else {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time).count();
        }
    }

    // Get elapsed time in microseconds
    double elapsedMicroseconds() const {
        return elapsedNanoseconds() / 1000.0;
    }

    // Get elapsed time in milliseconds
    double elapsedMilliseconds() const {
        return elapsedNanoseconds() / 1000000.0;
    }

    // Get elapsed time in seconds
    double elapsedSeconds() const {
        return elapsedNanoseconds() / 1000000000.0;
    }
};

class GpuTimer {
private:
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    bool initialized;
    bool running;

public:
    GpuTimer() : initialized(false), running(false) {
        cudaError_t err = cudaEventCreate(&start_event);
        if (err != cudaSuccess) return;
        
        err = cudaEventCreate(&stop_event);
        if (err != cudaSuccess) {
            cudaEventDestroy(start_event);
            return;
        }
        
        initialized = true;
    }

    ~GpuTimer() {
        if (initialized) {
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);
        }
    }

    void start(cudaStream_t stream = 0) {
        if (!initialized) return;
        cudaEventRecord(start_event, stream);
        running = true;
    }

    void stop(cudaStream_t stream = 0) {
        if (!initialized || !running) return;
        cudaEventRecord(stop_event, stream);
        running = false;
    }

    // Get elapsed time in milliseconds
    float elapsedMilliseconds() {
        if (!initialized || running) return 0.0f;
        
        float elapsed_time;
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
        return elapsed_time;
    }

    // Get elapsed time in microseconds
    float elapsedMicroseconds() {
        return elapsedMilliseconds() * 1000.0f;
    }

    // Get elapsed time in seconds
    float elapsedSeconds() {
        return elapsedMilliseconds() / 1000.0f;
    }
};

// Helper class to automatically measure execution time in a scope
class ScopedTimer {
private:
    CpuTimer timer;
    std::string name;
    bool print_on_destruction;

public:
    ScopedTimer(const std::string& timer_name, bool auto_print = true) 
        : name(timer_name), print_on_destruction(auto_print) {
        timer.start();
    }

    ~ScopedTimer() {
        timer.stop();
        if (print_on_destruction) {
            printf("[%s] Elapsed time: %.3f ms\n", 
                   name.c_str(), timer.elapsedMilliseconds());
        }
    }

    double getElapsedMilliseconds() {
        timer.stop();
        return timer.elapsedMilliseconds();
    }
};

#endif // TIMING_H