#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cuda_runtime.h>

#include "cuda_utils.h"
#include "timing.h"
#include "market_formats.h"

// Circular buffer size (64MB)
#define BUFFER_SIZE (1024 * 1024 * 64)

// Data batch size for GPU processing
#define BATCH_SIZE (1024 * 1024)

// Number of CUDA streams to use for processing
#define NUM_STREAMS 4

// Class for managing market data feed processing
class MarketDataParser {
private:
    // Circular buffer for incoming data
    struct CircularBuffer {
        char* data;
        size_t head;
        size_t tail;
        std::mutex mutex;
        std::condition_variable cv;
        bool shutdown;
    };

    CircularBuffer buffer;
    std::vector<cudaStream_t> streams;
    
    // Pinned memory buffers for host-device transfers
    char** h_batches;
    char** d_batches;
    int** h_results;
    int** d_results;
    
    std::atomic<bool> running;
    std::thread processing_thread;
    
    // Buffer management methods
    void initCircularBuffer() {
        buffer.data = (char*)malloc(BUFFER_SIZE);
        buffer.head = 0;
        buffer.tail = 0;
        buffer.shutdown = false;
    }
    
    bool addToCircularBuffer(const char* data, size_t size) {
        std::unique_lock<std::mutex> lock(buffer.mutex);
        
        size_t used = (buffer.head - buffer.tail) % BUFFER_SIZE;
        size_t available = BUFFER_SIZE - used - 1;
        
        if (size > available) {
            return false; // Buffer full
        }
        
        // Copy data to buffer
        for (size_t i = 0; i < size; i++) {
            buffer.data[(buffer.head + i) % BUFFER_SIZE] = data[i];
        }
        
        buffer.head = (buffer.head + size) % BUFFER_SIZE;
        buffer.cv.notify_one();
        return true;
    }
    
    size_t getFromCircularBuffer(char* data, size_t max_size) {
        std::unique_lock<std::mutex> lock(buffer.mutex);
        
        // Wait for data or shutdown
        while ((buffer.head == buffer.tail) && !buffer.shutdown) {
            buffer.cv.wait(lock);
        }
        
        if (buffer.shutdown && buffer.head == buffer.tail) {
            return 0;
        }
        
        size_t head = buffer.head;
        size_t tail = buffer.tail;
        
        // Calculate available data
        size_t available = (head >= tail) ? (head - tail) : (BUFFER_SIZE - tail + head);
        size_t to_copy = (available < max_size) ? available : max_size;
        
        // Copy data
        if (tail + to_copy <= BUFFER_SIZE) {
            // Contiguous data
            memcpy(data, buffer.data + tail, to_copy);
        } else {
            // Data wraps around buffer end
            size_t first_chunk = BUFFER_SIZE - tail;
            memcpy(data, buffer.data + tail, first_chunk);
            memcpy(data + first_chunk, buffer.data, to_copy - first_chunk);
        }
        
        buffer.tail = (buffer.tail + to_copy) % BUFFER_SIZE;
        return to_copy;
    }

    // CUDA kernel to process market data
    __global__ void processMarketDataKernel(char* data, size_t data_size, int* results) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= data_size) return;
        
        // Get message type
        char msg_type = data[idx];
        
        // Simple processing based on message type
        switch (msg_type) {
            case static_cast<char>(market::ItchMessageType::ADD_ORDER):
                results[idx] = 1;
                break;
            case static_cast<char>(market::ItchMessageType::ORDER_DELETE):
                results[idx] = 2;
                break;
            // Add processing for other message types as needed
            default:
                results[idx] = 0;
        }
    }

    // Processing thread function
    void processingLoop() {
        int current_stream = 0;
        
        while (running) {
            // Get data from circular buffer
            size_t bytes_read = getFromCircularBuffer(h_batches[current_stream], BATCH_SIZE);
            if (bytes_read == 0) {
                // No more data, possibly shutdown
                if (!running) break;
                continue;
            }
            
            // Measure processing time
            GpuTimer timer;
            timer.start(streams[current_stream]);
            
            // Copy data to GPU
            copyToDeviceAsync(d_batches[current_stream], h_batches[current_stream], 
                              bytes_read, streams[current_stream]);
            
            // Launch kernel
            int blockSize = 256;
            int numBlocks = (bytes_read + blockSize - 1) / blockSize;
            
            processMarketDataKernel<<<numBlocks, blockSize, 0, streams[current_stream]>>>(
                d_batches[current_stream], bytes_read, d_results[current_stream]);
            
            // Copy results back to CPU
            copyToHostAsync(h_results[current_stream], d_results[current_stream], 
                            bytes_read, streams[current_stream]);
            
            // Synchronize to measure latency accurately
            synchronizeCudaStream(streams[current_stream]);
            timer.stop(streams[current_stream]);
            
            // Report latency
            float latency_us = timer.elapsedMicroseconds();
            printf("Processed %zu bytes with latency: %.2f µs\n", bytes_read, latency_us);
            
            // Process results as needed
            // ...
            
            // Move to next stream (round-robin)
            current_stream = (current_stream + 1) % NUM_STREAMS;
        }
    }

public:
    MarketDataParser() : running(false) {
        // Initialize CUDA device
        initCudaDevice();
        
        // Initialize circular buffer
        initCircularBuffer();
        
        // Create CUDA streams
        streams.resize(NUM_STREAMS);
        for (int i = 0; i < NUM_STREAMS; i++) {
            streams[i] = createCudaStream();
        }
        
        // Allocate pinned memory for batches
        h_batches = new char*[NUM_STREAMS];
        d_batches = new char*[NUM_STREAMS];
        h_results = new int*[NUM_STREAMS];
        d_results = new int*[NUM_STREAMS];
        
        for (int i = 0; i < NUM_STREAMS; i++) {
            h_batches[i] = allocatePinnedMemory<char>(BATCH_SIZE);
            d_batches[i] = allocateDeviceMemory<char>(BATCH_SIZE);
            h_results[i] = allocatePinnedMemory<int>(BATCH_SIZE);
            d_results[i] = allocateDeviceMemory<int>(BATCH_SIZE);
        }
    }
    
    ~MarketDataParser() {
        stop();
        
        // Free resources
        for (int i = 0; i < NUM_STREAMS; i++) {
            freePinnedMemory(h_batches[i]);
            freeDeviceMemory(d_batches[i]);
            freePinnedMemory(h_results[i]);
            freeDeviceMemory(d_results[i]);
            destroyCudaStream(streams[i]);
        }
        
        delete[] h_batches;
        delete[] d_batches;
        delete[] h_results;
        delete[] d_results;
        
        free(buffer.data);
    }
    
    // Start the processing loop
    void start() {
        if (running) return;
        
        running = true;
        processing_thread = std::thread(&MarketDataParser::processingLoop, this);
    }
    
    // Stop the processing loop
    void stop() {
        if (!running) return;
        
        running = false;
        buffer.shutdown = true;
        buffer.cv.notify_all();
        
        if (processing_thread.joinable()) {
            processing_thread.join();
        }
    }
    
    // Feed data to the parser
    bool feedData(const char* data, size_t size) {
        return addToCircularBuffer(data, size);
    }
    
    // Feed data from a file
    bool feedDataFromFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return false;
        }
        
        const size_t chunk_size = 64 * 1024; // 64KB chunks
        std::vector<char> buffer(chunk_size);
        bool success = true;
        
        while (file.good() && success) {
            file.read(buffer.data(), chunk_size);
            std::streamsize bytes_read = file.gcount();
            
            if (bytes_read > 0) {
                success = feedData(buffer.data(), bytes_read);
            }
        }
        
        return success;
    }
};

// Main entry point for the market data parser application
int main(int argc, char* argv[]) {
    MarketDataParser parser;
    
    // Start the parser
    parser.start();
    
    // Check for input file
    if (argc > 1) {
        std::string filename = argv[1];
        std::cout << "Processing market data from file: " << filename << std::endl;
        parser.feedDataFromFile(filename);
    } else {
        std::cout << "No input file specified. Simulating market data..." << std::endl;
        
        // Simulate market data
        const size_t message_size = 64; // Example message size
        const size_t num_messages = 100000;
        std::vector<char> simulated_data(message_size * num_messages);
        
        // Fill with sample data
        for (size_t i = 0; i < num_messages; i++) {
            // Add message header
            market::MessageHeader* header = reinterpret_cast<market::MessageHeader*>(
                &simulated_data[i * message_size]);
            header->length = message_size;
            header->timestamp = i * 100; // 100ns between messages
            
            // Alternate between message types
            if (i % 3 == 0) {
                header->type = static_cast<char>(market::ItchMessageType::ADD_ORDER);
            } else if (i % 3 == 1) {
                header->type = static_cast<char>(market::ItchMessageType::ORDER_EXECUTED);
            } else {
                header->type = static_cast<char>(market::ItchMessageType::ORDER_DELETE);
            }
        }
        
        // Feed the simulated data
        parser.feedData(simulated_data.data(), simulated_data.size());
    }
    
    // Allow processing to complete
    std::cout << "Processing... Press Enter to stop." << std::endl;
    std::cin.get();
    
    // Stop the parser
    parser.stop();
    
    return 0;
}