#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cassert>

#include "cuda_utils.h"
#include "market_formats.h"

namespace itch {

// ITCH Protocol constants
constexpr int MAX_SYMBOLS = 8192;
constexpr int MAX_ORDERS = 1024 * 1024 * 8;  // 8 million orders

// ITCH Message type flags for CUDA processing
// Each flag represents a different message type
enum MessageFlags {
    FLAG_SYSTEM_EVENT = 1 << 0,
    FLAG_ADD_ORDER = 1 << 1,
    FLAG_ADD_ORDER_MPID = 1 << 2,
    FLAG_ORDER_EXECUTED = 1 << 3,
    FLAG_ORDER_CANCEL = 1 << 4,
    FLAG_ORDER_DELETE = 1 << 5,
    FLAG_ORDER_REPLACE = 1 << 6,
    FLAG_TRADE = 1 << 7,
    FLAG_CROSS_TRADE = 1 << 8,
    FLAG_BROKEN_TRADE = 1 << 9
};

// Order book entry
struct OrderBookEntry {
    uint64_t order_ref_num;
    uint32_t price;
    uint32_t shares;
    uint16_t stock_locate;
    char buy_sell_indicator;
    bool is_active;
};

// Symbol directory entry
struct SymbolEntry {
    char symbol[8];
    uint16_t stock_locate;
    char market_category;
    char financial_status_indicator;
    uint32_t round_lot_size;
    char round_lots_only;
    char issue_classification;
    char issue_subtype[2];
    char authenticity;
    char short_sale_threshold_indicator;
    char ipo_flag;
    char luld_reference_price_tier;
    char etp_flag;
    uint32_t etp_leverage_factor;
    char inverse_indicator;
};

// Host data structure for ITCH processing
struct ItchProcessor {
    // Order book - maps order reference numbers to order details
    std::vector<OrderBookEntry> order_book;
    
    // Symbol directory - maps stock locate IDs to symbol information
    std::vector<SymbolEntry> symbols;
    
    // Statistics counters
    uint64_t message_count;
    uint64_t add_order_count;
    uint64_t executed_order_count;
    uint64_t canceled_order_count;
    uint64_t deleted_order_count;
    
    // Initialize the processor
    ItchProcessor() {
        order_book.resize(MAX_ORDERS);
        symbols.resize(MAX_SYMBOLS);
        reset();
    }
    
    // Reset all state
    void reset() {
        message_count = 0;
        add_order_count = 0;
        executed_order_count = 0;
        canceled_order_count = 0;
        deleted_order_count = 0;
        
        // Clear order book
        for (auto& order : order_book) {
            order.is_active = false;
        }
    }
};

// Device data structure for ITCH processing
struct DeviceItchProcessor {
    OrderBookEntry* d_order_book;
    SymbolEntry* d_symbols;
    uint64_t* d_message_counters;  // Array of counters for different message types
    
    // Initialize device memory
    DeviceItchProcessor() {
        CHECK_CUDA_ERROR(cudaMalloc(&d_order_book, MAX_ORDERS * sizeof(OrderBookEntry)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_symbols, MAX_SYMBOLS * sizeof(SymbolEntry)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_message_counters, 10 * sizeof(uint64_t)));  // 10 counter types
        
        // Initialize counters to zero
        CHECK_CUDA_ERROR(cudaMemset(d_message_counters, 0, 10 * sizeof(uint64_t)));
    }
    
    // Free device memory
    ~DeviceItchProcessor() {
        cudaFree(d_order_book);
        cudaFree(d_symbols);
        cudaFree(d_message_counters);
    }
};

// Device functions for processing ITCH messages

// Process System Event Message
__device__ void processSystemEvent(const char* msg_data, size_t msg_size) {
    // Extract fields from the message
    const market::SystemEventMessage* msg = 
        reinterpret_cast<const market::SystemEventMessage*>(msg_data);
    
    // Process based on event code
    switch (msg->event_code) {
        case 'O': // Start of messages
            // Reset state for a new day
            break;
        case 'S': // Start of system hours
            break;
        case 'Q': // Start of market hours
            break;
        case 'M': // End of market hours 
            break;
        case 'E': // End of system hours
            break;
        case 'C': // End of messages
            break;
    }
}

// Process Add Order Message
__device__ void processAddOrder(const char* msg_data, size_t msg_size, 
                                OrderBookEntry* order_book) {
    // Extract fields from the message
    const market::AddOrderMessage* msg = 
        reinterpret_cast<const market::AddOrderMessage*>(msg_data);
    
    // Get order reference number (index into the order book)
    uint64_t order_ref = msg->order_reference_number;
    
    // Check for valid index
    if (order_ref < MAX_ORDERS) {
        // Update the order book
        order_book[order_ref].order_ref_num = order_ref;
        order_book[order_ref].price = msg->price;
        order_book[order_ref].shares = msg->shares;
        order_book[order_ref].stock_locate = msg->stock_locate;
        order_book[order_ref].buy_sell_indicator = msg->buy_sell_indicator;
        order_book[order_ref].is_active = true;
    }
}

// Process Order Executed Message
__device__ void processOrderExecuted(const char* msg_data, size_t msg_size, 
                                     OrderBookEntry* order_book) {
    // Extract fields from the message
    const market::OrderExecutedMessage* msg = 
        reinterpret_cast<const market::OrderExecutedMessage*>(msg_data);
    
    // Get order reference number
    uint64_t order_ref = msg->order_reference_number;
    
    // Check for valid index and active order
    if (order_ref < MAX_ORDERS && order_book[order_ref].is_active) {
        // Reduce shares in the order book
        if (order_book[order_ref].shares >= msg->executed_shares) {
            order_book[order_ref].shares -= msg->executed_shares;
            
            // If no shares left, mark as inactive
            if (order_book[order_ref].shares == 0) {
                order_book[order_ref].is_active = false;
            }
        }
    }
}

// Process Order Delete Message
__device__ void processOrderDelete(const char* msg_data, size_t msg_size, 
                                   OrderBookEntry* order_book) {
    // Extract fields from the message
    const market::OrderDeleteMessage* msg = 
        reinterpret_cast<const market::OrderDeleteMessage*>(msg_data);
    
    // Get order reference number
    uint64_t order_ref = msg->order_reference_number;
    
    // Check for valid index
    if (order_ref < MAX_ORDERS) {
        // Mark order as inactive
        order_book[order_ref].is_active = false;
    }
}

// Main kernel for processing ITCH messages
__global__ void processItchMessagesKernel(const char* data, size_t* message_offsets, 
                                         size_t num_messages, OrderBookEntry* order_book,
                                         SymbolEntry* symbols, uint64_t* counters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_messages) return;
    
    // Get message data and size
    size_t offset = message_offsets[idx];
    size_t msg_size = message_offsets[idx + 1] - offset;
    const char* msg_data = data + offset;
    
    // Get message type
    char msg_type = msg_data[0];
    
    // Process based on message type
    switch (msg_type) {
        case static_cast<char>(market::ItchMessageType::SYSTEM_EVENT):
            processSystemEvent(msg_data, msg_size);
            atomicAdd(&counters[0], 1);
            break;
            
        case static_cast<char>(market::ItchMessageType::ADD_ORDER):
            processAddOrder(msg_data, msg_size, order_book);
            atomicAdd(&counters[1], 1);
            break;
            
        case static_cast<char>(market::ItchMessageType::ADD_ORDER_WITH_MPID):
            processAddOrder(msg_data, msg_size, order_book);
            atomicAdd(&counters[2], 1);
            break;
            
        case static_cast<char>(market::ItchMessageType::ORDER_EXECUTED):
            processOrderExecuted(msg_data, msg_size, order_book);
            atomicAdd(&counters[3], 1);
            break;
            
        case static_cast<char>(market::ItchMessageType::ORDER_DELETE):
            processOrderDelete(msg_data, msg_size, order_book);
            atomicAdd(&counters[4], 1);
            break;
            
        // Other message types can be added here
    }
    
    // Increment total message counter
    atomicAdd(&counters[9], 1);
}

// Host functions for ITCH processing

// Parse ITCH message header and determine message length
market::ParseResult parseItchMessageLength(const char* data, size_t max_size) {
    market::ParseResult result = { false, "", 0 };
    
    if (max_size < 2) {
        result.error_message = "Message too short for length field";
        return result;
    }
    
    // First two bytes are the message length
    uint16_t length = *reinterpret_cast<const uint16_t*>(data);
    
    // Validate length
    if (length < 2 || length > max_size) {
        result.error_message = "Invalid message length";
        return result;
    }
    
    result.success = true;
    result.bytes_processed = length;
    return result;
}

// Identify message boundaries in an ITCH data stream
std::vector<size_t> identifyItchMessages(const char* data, size_t data_size) {
    std::vector<size_t> message_offsets;
    message_offsets.push_back(0);  // First message starts at offset 0
    
    size_t offset = 0;
    while (offset < data_size) {
        market::ParseResult result = parseItchMessageLength(data + offset, data_size - offset);
        
        if (!result.success) {
            // Could not parse message, break
            break;
        }
        
        // Move to next message
        offset += result.bytes_processed;
        
        if (offset < data_size) {
            message_offsets.push_back(offset);
        }
    }
    
    return message_offsets;
}

// Process ITCH messages in parallel on GPU
void processItchMessages(const char* data, size_t data_size, 
                         ItchProcessor& host_processor,
                         DeviceItchProcessor& device_processor,
                         cudaStream_t stream) {
    // Identify message boundaries
    std::vector<size_t> message_offsets = identifyItchMessages(data, data_size);
    
    if (message_offsets.empty()) {
        std::cerr << "No valid ITCH messages found" << std::endl;
        return;
    }
    
    // Add one more offset for the end of data
    message_offsets.push_back(data_size);
    
    size_t num_messages = message_offsets.size() - 1;
    std::cout << "Found " << num_messages << " ITCH messages" << std::endl;
    
    // Copy data to device
    char* d_data;
    size_t* d_offsets;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, data_size));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data, data, data_size, 
                                     cudaMemcpyHostToDevice, stream));
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_offsets, message_offsets.size() * sizeof(size_t)));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_offsets, message_offsets.data(), 
                                     message_offsets.size() * sizeof(size_t),
                                     cudaMemcpyHostToDevice, stream));
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (num_messages + blockSize - 1) / blockSize;
    
    processItchMessagesKernel<<<numBlocks, blockSize, 0, stream>>>(
        d_data, d_offsets, num_messages,
        device_processor.d_order_book,
        device_processor.d_symbols,
        device_processor.d_message_counters);
    
    // Check for kernel errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Copy counters back to host
    uint64_t h_counters[10];
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_counters, device_processor.d_message_counters, 
                                     10 * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    
    // Synchronize to ensure all operations are complete
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    
    // Update host statistics
    host_processor.message_count += h_counters[9];
    host_processor.add_order_count += h_counters[1] + h_counters[2];
    host_processor.executed_order_count += h_counters[3];
    host_processor.deleted_order_count += h_counters[4];
    host_processor.canceled_order_count += h_counters[5];
    
    // Clean up
    cudaFree(d_data);
    cudaFree(d_offsets);
}

// Process a batch of ITCH messages with timing
double processItchMessagesWithTiming(const char* data, size_t data_size,
                                   ItchProcessor& host_processor,
                                   DeviceItchProcessor& device_processor,
                                   cudaStream_t stream) {
    // Start timer
    GpuTimer timer;
    timer.start(stream);
    
    // Process messages
    processItchMessages(data, data_size, host_processor, device_processor, stream);
    
    // Stop timer
    timer.stop(stream);
    
    // Return elapsed time in microseconds
    return timer.elapsedMicroseconds();
}

// Main entry point for ITCH format parsing functions
class ItchFormatParser {
private:
    ItchProcessor host_processor;
    DeviceItchProcessor device_processor;
    cudaStream_t stream;
    
public:
    ItchFormatParser() {
        // Create CUDA stream
        stream = createCudaStream();
    }
    
    ~ItchFormatParser() {
        // Destroy CUDA stream
        destroyCudaStream(stream);
    }
    
    // Process a batch of ITCH data
    void processBatch(const char* data, size_t data_size) {
        processItchMessages(data, data_size, host_processor, device_processor, stream);
    }
    
    // Process a batch with timing information
    double processBatchWithTiming(const char* data, size_t data_size) {
        return processItchMessagesWithTiming(data, data_size, host_processor, device_processor, stream);
    }
    
    // Print statistics
    void printStatistics() const {
        std::cout << "ITCH Processing Statistics:" << std::endl;
        std::cout << "  Total Messages:  " << host_processor.message_count << std::endl;
        std::cout << "  Add Orders:      " << host_processor.add_order_count << std::endl;
        std::cout << "  Executed Orders: " << host_processor.executed_order_count << std::endl;
        std::cout << "  Deleted Orders:  " << host_processor.deleted_order_count << std::endl;
        std::cout << "  Canceled Orders: " << host_processor.canceled_order_count << std::endl;
    }
    
    // Reset all state
    void reset() {
        host_processor.reset();
        
        // Reset device counters
        CHECK_CUDA_ERROR(cudaMemset(device_processor.d_message_counters, 0, 10 * sizeof(uint64_t)));
    }
    
    // Get reference to host processor
    const ItchProcessor& getProcessor() const {
        return host_processor;
    }
    
    // Process ITCH file
    bool processFile(const std::string& filename) {
        // Open file
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return false;
        }
        
        // Get file size
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        // Read file into buffer
        std::vector<char> buffer(file_size);
        file.read(buffer.data(), file_size);
        
        if (!file) {
            std::cerr << "Failed to read file: " << filename << std::endl;
            return false;
        }
        
        // Process buffer
        double latency = processBatchWithTiming(buffer.data(), file_size);
        std::cout << "Processed file in " << latency << " microseconds" << std::endl;
        
        return true;
    }
};

} // namespace itch