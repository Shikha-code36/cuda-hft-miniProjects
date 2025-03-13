#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

// Error checking macro
#define CHECK_CUDA_ERROR(err) { checkCudaError(err, __FILE__, __LINE__); }

// Helper function for error checking
inline void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Initialize CUDA device
inline void initCudaDevice(int deviceId = 0) {
    int deviceCount;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        exit(EXIT_FAILURE);
    }
    
    if (deviceId < 0 || deviceId >= deviceCount) {
        fprintf(stderr, "Invalid device ID. Using device 0 instead.\n");
        deviceId = 0;
    }
    
    CHECK_CUDA_ERROR(cudaSetDevice(deviceId));
    
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, deviceId));
    
    printf("Using CUDA device %d: %s\n", deviceId, deviceProp.name);
    printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Total global memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Multiprocessors: %d\n", deviceProp.multiProcessorCount);
}

// Calculate optimal block size for a given kernel
inline dim3 getOptimalBlockSize(int numElements) {
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((numElements + blockSize.x - 1) / blockSize.x, 1, 1);
    return gridSize;
}

// Allocate pinned memory with error checking
template<typename T>
inline T* allocatePinnedMemory(size_t count) {
    T* ptr;
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&ptr, count * sizeof(T)));
    return ptr;
}

// Free pinned memory with error checking
template<typename T>
inline void freePinnedMemory(T* ptr) {
    if (ptr != nullptr) {
        CHECK_CUDA_ERROR(cudaFreeHost(ptr));
    }
}

// Allocate device memory with error checking
template<typename T>
inline T* allocateDeviceMemory(size_t count) {
    T* ptr;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&ptr, count * sizeof(T)));
    return ptr;
}

// Free device memory with error checking
template<typename T>
inline void freeDeviceMemory(T* ptr) {
    if (ptr != nullptr) {
        CHECK_CUDA_ERROR(cudaFree(ptr));
    }
}

// Create CUDA stream with error checking
inline cudaStream_t createCudaStream() {
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    return stream;
}

// Destroy CUDA stream with error checking
inline void destroyCudaStream(cudaStream_t stream) {
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

// Synchronize CUDA stream with error checking
inline void synchronizeCudaStream(cudaStream_t stream) {
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
}

// Copy data from host to device asynchronously
template<typename T>
inline void copyToDeviceAsync(T* dst, const T* src, size_t count, cudaStream_t stream) {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyHostToDevice, stream));
}

// Copy data from device to host asynchronously
template<typename T>
inline void copyToHostAsync(T* dst, const T* src, size_t count, cudaStream_t stream) {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
}

#endif // CUDA_UTILS_H