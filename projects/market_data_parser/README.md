# CUDA Market Data Feed Parser

A high-performance market data parser for financial market data feeds, designed to achieve ultra-low latency (< 5μs) processing using CUDA GPU acceleration.

## Features

- Parses binary market data formats including NASDAQ ITCH 5.0 and NYSE PITCH
- Uses GPU parallelism to process market data messages concurrently
- Implements optimized memory management with pinned memory and zero-copy transfers
- Utilizes asynchronous CUDA streams for overlapping data transfer and computation
- Achieves sub-microsecond message processing latencies

## Architecture

The parser is designed with the following components:

1. **Circular Buffer**: Lock-free circular buffer for high-throughput data ingestion
2. **Batch Processing**: Efficient batching of market data messages for GPU processing
3. **Parallel Message Parsing**: Each CUDA thread processes a single market data message
4. **Multiple Streams**: Uses multiple CUDA streams for pipeline parallelism

## Performance

- Processing latency: < 5μs per message
- Throughput: Millions of messages per second
- Memory usage: Configurable buffer sizes for different workloads

## Building

### Prerequisites

- NVIDIA CUDA Toolkit 11.0 or later
- C++14 compliant compiler (GCC 7.5+, Visual Studio 2019+)
- CMake 3.18 or later

### Build Instructions

```bash
# Create a build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build the project
cmake --build .
```

## Usage

```bash
# Run with sample data
./bin/market_data_parser/market_data_parser sample_data.bin

# Run with simulated data
./bin/market_data_parser/market_data_parser
```

## File Structure

- `include/market_formats.h`: Defines market data message formats
- `src/market_data_parser.cu`: Main implementation of the market data parser
- `src/itch_format.cu`: NASDAQ ITCH protocol-specific parsing functions

