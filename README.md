# hawkZip: High-Performance Lossy Compression for Floating-Point Data

**Modified by Benjamin Harks, Likith Kadiyala, and Aaron Clinkenbeard**  
*University of Iowa CS:4700 Final Project*

## Overview

hawkZip is an optimized CPU-based error-bounded lossy compressor specifically designed for floating-point scientific data. This implementation focuses on maximizing **throughput** and **compression ratio** through advanced parallel processing and vectorization techniques.

### Key Features

- **Error-Bounded Compression**: Maintains user-specified accuracy guarantees
- **High-Performance Parallel Processing**: OpenMP-based multi-threading (up to 16 cores)
- **SIMD Vectorization**: AVX2 instruction set for enhanced performance
- **Adaptive Threading**: Dynamic thread scaling based on data size
- **Block-Based Processing**: 32-element blocks with fixed-length encoding
- **Comprehensive Quality Metrics**: Built-in PSNR and SSIM computation

## Technical Architecture

### Compression Algorithm
- **Quantization**: Value-range-based relative error bound conversion
- **Block Processing**: Data organized into 32-element blocks for parallel processing
- **Fixed-Length Encoding**: Adaptive bit-width encoding per block
- **Sign Compression**: Efficient sign bit storage using bit packing

### Performance Optimizations
- **Loop Unrolling**: Minimized instruction overhead
- **SIMD Vectorization**: AVX2 instructions for 8-way parallel float operations
- **Memory Optimization**: Reduced memory bandwidth through efficient data layouts
- **Thread-Level Parallelism**: OpenMP with work-stealing load balancing
- **LICM (Loop Invariant Code Motion)**: Optimized constant computations

## System Requirements

- **Operating System**: Linux (tested on modern distributions)
- **Compiler**: GCC with AVX2 support
- **Dependencies**: 
  - OpenMP library
  - AVX2-capable processor (Intel Haswell+ or AMD Excavator+)

### Installation

Install OpenMP support:
```bash
# Debian/Ubuntu
sudo apt-get install libgomp1

# CentOS/RHEL
sudo yum install libgomp

# macOS (if using GCC)
brew install gcc libomp
```

## Compilation

```bash
gcc hawkZip_main.c -O0 -o hawkZip -lm -fopenmp -mavx2
```

**Compilation Flags Explained:**
- `-O0`: Disabled optimization for debugging (use `-O3` for production)
- `-lm`: Math library linking
- `-fopenmp`: OpenMP support
- `-mavx2`: AVX2 vectorization support

## Usage

```bash
./hawkZip -i [srcFilePath] -e [errorBound] [-x cmpFilePath] [-o decFilePath]
```

### Command-Line Options

| Option | Type | Description |
|--------|------|-------------|
| `-i [srcFilePath]` | Required | Input file path (binary float32 format) |
| `-e [errorBound]` | Required | Relative error bound (e.g., 1e-4, 0.01) |
| `-x [cmpFilePath]` | Optional | Output path for compressed data |
| `-o [decFilePath]` | Optional | Output path for decompressed data |

### Example Usage

```bash
# Basic compression with error checking
./hawkZip -i data.f32 -e 1e-4

# Compression with file output
./hawkZip -i scientific_data.f32 -e 0.001 -x compressed.hawkzip

# Full pipeline: compress and decompress
./hawkZip -i original.f32 -e 1e-3 -x compressed.hawkzip -o reconstructed.f32

# High compression ratio (lower quality)
./hawkZip -i dataset.f32 -e 0.1 -x high_compression.hawkzip
```

## Performance Characteristics

### Throughput Reporting
The tool automatically reports:
- **Compression Ratio**: `(original_size) / (compressed_size)`
- **Compression Throughput**: GB/s processing rate
- **Decompression Throughput**: GB/s reconstruction rate

### Thread Scaling
- **Large datasets** (≥1M elements): Uses all available cores (max 16)
- **Small datasets** (<1M elements): Uses half the available cores for efficiency
- **Adaptive load balancing**: Work-stealing between threads

### Memory Requirements
- **Runtime Memory**: ~4x original data size during processing
- **Compressed Output**: Varies based on data characteristics and error bound
- **Typical Compression Ratios**: 2:1 to 10:1 depending on data and error tolerance

## File Format Support

### Input Format
- **Binary float32** (`.f32`) files
- **Little-endian** byte ordering (automatic detection)
- **Arbitrary data dimensions** (treated as 1D array)

### Output Format
- **Compressed data**: Custom binary format with metadata
- **Decompressed data**: IEEE 754 float32 binary format

## Quality Assurance

### Error Verification
- Automatic error bound verification with tolerance (1.1x error bound)
- Color-coded success/failure reporting
- Per-element error analysis available

### Quality Metrics
Built-in computation of:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **MSE** (Mean Square Error)
- **NRMSE** (Normalized Root Mean Square Error)

## Project Structure
