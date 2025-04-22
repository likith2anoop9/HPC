#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <emmintrin.h>
#include <omp.h>

// Use dynamic thread count instead of fixed number
#define MAX_THREADS 16

void hawkZip_compress_kernel(float* oriData, unsigned char* cmpData, int* absQuant, unsigned int* signFlag, int* fixedRate, unsigned int* threadOfs, size_t nbEle, size_t* cmpSize, float errorBound)
{
    // Determine optimal thread count based on available hardware and data size
    int num_threads = omp_get_max_threads();
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;
    
    // For very small datasets, reduce thread count to avoid overhead
    if (nbEle < 1024) num_threads = 1;
    else if (nbEle < 4096) num_threads = 2;
    else if (nbEle < 8192) num_threads = 4;
    
    // Calculate block size - try to align with cache line size (typically 64 bytes)
    const int block_size = 32; // Keep fixed block size for bit operations
    
    // Calculate chunks with better load balancing
    int chunk_size = (nbEle + num_threads - 1) / num_threads;
    // Round chunk size to multiple of block_size to avoid thread boundary issues
    chunk_size = ((chunk_size + block_size - 1) / block_size) * block_size;
    
    // Pre-compute constants outside the parallel region
    const float recip_precision = 0.5f/errorBound;
    
    // Initialize thread offset array
    #pragma omp parallel for
    for (int i = 0; i < num_threads; i++) {
        threadOfs[i] = 0;
    }
    
    // hawkZip parallel compression begin
    #pragma omp parallel num_threads(num_threads)
    {
        // Divide data chunk for each thread - with better load balancing
        int thread_id = omp_get_thread_num();
        int start = thread_id * chunk_size;
        int end = start + chunk_size;
        if (end > nbEle) end = nbEle;
        
        // Skip empty chunks (might happen with uneven distribution)
        if (start < nbEle) {
            int block_num = (end - start + block_size - 1) / block_size;
            int start_block = thread_id * ((chunk_size + block_size - 1) / block_size);
            unsigned int thread_ofs = 0;
            
            // Process data in cache-friendly manner 
            // Pre-compute sign flags and quantization in one pass
            for (int i = 0; i < block_num; i++) {
                int block_start = start + i * block_size;
                int block_end = (block_start + block_size) > end ? end : block_start + block_size;
                int curr_block = start_block + i;
                unsigned int sign_flag = 0;
                int max_quant = 0;
                
                // Process the block with SIMD-friendly pattern - unroll by 4 where possible
                int j = block_start;
                // Main loop - unrolled by 4
                for (; j <= block_end - 4; j += 4) {
                    // Prequantization for 4 elements at once
                    float data_recip0 = oriData[j] * recip_precision;
                    float data_recip1 = oriData[j+1] * recip_precision;
                    float data_recip2 = oriData[j+2] * recip_precision;
                    float data_recip3 = oriData[j+3] * recip_precision;
                    
                    int s0 = data_recip0 >= -0.5f ? 0 : 1;
                    int s1 = data_recip1 >= -0.5f ? 0 : 1;
                    int s2 = data_recip2 >= -0.5f ? 0 : 1;
                    int s3 = data_recip3 >= -0.5f ? 0 : 1;
                    
                    int curr_quant0 = (int)(data_recip0 + 0.5f) - s0;
                    int curr_quant1 = (int)(data_recip1 + 0.5f) - s1;
                    int curr_quant2 = (int)(data_recip2 + 0.5f) - s2;
                    int curr_quant3 = (int)(data_recip3 + 0.5f) - s3;
                    
                    // Get sign data for 4 elements
                    int sign_ofs0 = j % block_size;
                    int sign_ofs1 = (j+1) % block_size;
                    int sign_ofs2 = (j+2) % block_size;
                    int sign_ofs3 = (j+3) % block_size;
                    
                    sign_flag |= (curr_quant0 < 0) << (31 - sign_ofs0);
                    sign_flag |= (curr_quant1 < 0) << (31 - sign_ofs1);
                    sign_flag |= (curr_quant2 < 0) << (31 - sign_ofs2);
                    sign_flag |= (curr_quant3 < 0) << (31 - sign_ofs3);
                    
                    // Get absolute quantization code for 4 elements
                    int abs_quant0 = abs(curr_quant0);
                    int abs_quant1 = abs(curr_quant1);
                    int abs_quant2 = abs(curr_quant2);
                    int abs_quant3 = abs(curr_quant3);
                    
                    max_quant = max_quant > abs_quant0 ? max_quant : abs_quant0;
                    max_quant = max_quant > abs_quant1 ? max_quant : abs_quant1;
                    max_quant = max_quant > abs_quant2 ? max_quant : abs_quant2;
                    max_quant = max_quant > abs_quant3 ? max_quant : abs_quant3;
                    
                    absQuant[j] = abs_quant0;
                    absQuant[j+1] = abs_quant1;
                    absQuant[j+2] = abs_quant2;
                    absQuant[j+3] = abs_quant3;
                }
                
                // Handle remaining elements (less than 4)
                for (; j < block_end; j++) {
                    // Prequantization
                    float data_recip = oriData[j] * recip_precision;
                    int s = data_recip >= -0.5f ? 0 : 1;
                    int curr_quant = (int)(data_recip + 0.5f) - s;
                    
                    // Get sign data
                    int sign_ofs = j % block_size;
                    sign_flag |= (curr_quant < 0) << (31 - sign_ofs);
                    
                    // Get absolute quantization code
                    int abs_quant = abs(curr_quant);
                    max_quant = max_quant > abs_quant ? max_quant : abs_quant;
                    absQuant[j] = abs_quant;
                }
                
                // Record fixed-length encoding rate
                signFlag[curr_block] = sign_flag;
                int temp_fixed_rate = max_quant == 0 ? 0 : sizeof(int) * 8 - __builtin_clz(max_quant);
                fixedRate[curr_block] = temp_fixed_rate;
                cmpData[curr_block] = (unsigned char)temp_fixed_rate;
                
                // Calculate thread-local offset
                thread_ofs += temp_fixed_rate ? (block_size + temp_fixed_rate * block_size) / 8 : 0;
            }
            
            // Store thread offset for global prefix sum
            threadOfs[thread_id] = thread_ofs;
        }
        
        // Synchronize all threads before prefix sum
        #pragma omp barrier
        
        // Perform exclusive prefix sum - optimize with parallel reduction for large thread counts
        unsigned int global_ofs = 0;
        if (thread_id > 0) {
            for (int i = 0; i < thread_id; i++) {
                global_ofs += threadOfs[i];
            }
        }
        
        // Calculate byte offset for compressed data
        int block_num = (end - start + block_size - 1) / block_size;
        int start_block = thread_id * ((chunk_size + block_size - 1) / block_size);
        unsigned int cmp_byte_ofs = global_ofs + ((chunk_size + block_size - 1) / block_size) * num_threads;
        
        // Skip empty chunks
        if (start < nbEle) {
            // Second pass - encode the data
            for (int i = 0; i < block_num; i++) {
                int block_start = start + i * block_size;
                int block_end = (block_start + block_size) > end ? end : block_start + block_size;
                int curr_block = start_block + i;
                int temp_fixed_rate = fixedRate[curr_block];
                unsigned int sign_flag = signFlag[curr_block];
                
                // Create mask for zero/non-zero block
                unsigned int non_zero_mask = -((temp_fixed_rate != 0) ? 1 : 0);
                
                // Masked operations for sign flag - only applies when non_zero_mask is non-zero
                cmpData[cmp_byte_ofs] = (non_zero_mask & (0xff & (sign_flag >> 24)));
                cmpData[cmp_byte_ofs+1] = (non_zero_mask & (0xff & (sign_flag >> 16)));
                cmpData[cmp_byte_ofs+2] = (non_zero_mask & (0xff & (sign_flag >> 8)));
                cmpData[cmp_byte_ofs+3] = (non_zero_mask & (0xff & sign_flag));
                
                // Only advance offset if non-zero block
                cmp_byte_ofs += non_zero_mask & 4;
                
                // Process each bit position for non-zero blocks
                int j = 0;
                while (j < temp_fixed_rate & non_zero_mask) {
                    // Initialize bit buffers
                    unsigned char tmp_char0 = 0;
                    unsigned char tmp_char1 = 0;
                    unsigned char tmp_char2 = 0;
                    unsigned char tmp_char3 = 0;
                    int mask = 1 << j;
                    
                    // Group 1: 0-7 - fully unrolled
                    if (block_start < block_end) {
                        tmp_char0 |= (((absQuant[block_start] & mask) >> j) << 7);
                    }
                    if (block_start+1 < block_end) {
                        tmp_char0 |= (((absQuant[block_start+1] & mask) >> j) << 6);
                    }
                    if (block_start+2 < block_end) {
                        tmp_char0 |= (((absQuant[block_start+2] & mask) >> j) << 5);
                    }
                    if (block_start+3 < block_end) {
                        tmp_char0 |= (((absQuant[block_start+3] & mask) >> j) << 4);
                    }
                    if (block_start+4 < block_end) {
                        tmp_char0 |= (((absQuant[block_start+4] & mask) >> j) << 3);
                    }
                    if (block_start+5 < block_end) {
                        tmp_char0 |= (((absQuant[block_start+5] & mask) >> j) << 2);
                    }
                    if (block_start+6 < block_end) {
                        tmp_char0 |= (((absQuant[block_start+6] & mask) >> j) << 1);
                    }
                    if (block_start+7 < block_end) {
                        tmp_char0 |= (((absQuant[block_start+7] & mask) >> j) << 0);
                    }
                    
                    // Group 2: 8-15 - fully unrolled
                    if (block_start+8 < block_end) {
                        tmp_char1 |= (((absQuant[block_start+8] & mask) >> j) << 7);
                    }
                    if (block_start+9 < block_end) {
                        tmp_char1 |= (((absQuant[block_start+9] & mask) >> j) << 6);
                    }
                    if (block_start+10 < block_end) {
                        tmp_char1 |= (((absQuant[block_start+10] & mask) >> j) << 5);
                    }
                    if (block_start+11 < block_end) {
                        tmp_char1 |= (((absQuant[block_start+11] & mask) >> j) << 4);
                    }
                    if (block_start+12 < block_end) {
                        tmp_char1 |= (((absQuant[block_start+12] & mask) >> j) << 3);
                    }
                    if (block_start+13 < block_end) {
                        tmp_char1 |= (((absQuant[block_start+13] & mask) >> j) << 2);
                    }
                    if (block_start+14 < block_end) {
                        tmp_char1 |= (((absQuant[block_start+14] & mask) >> j) << 1);
                    }
                    if (block_start+15 < block_end) {
                        tmp_char1 |= (((absQuant[block_start+15] & mask) >> j) << 0);
                    }
                    
                    // Group 3: 16-23 - fully unrolled
                    if (block_start+16 < block_end) {
                        tmp_char2 |= (((absQuant[block_start+16] & mask) >> j) << 7);
                    }
                    if (block_start+17 < block_end) {
                        tmp_char2 |= (((absQuant[block_start+17] & mask) >> j) << 6);
                    }
                    if (block_start+18 < block_end) {
                        tmp_char2 |= (((absQuant[block_start+18] & mask) >> j) << 5);
                    }
                    if (block_start+19 < block_end) {
                        tmp_char2 |= (((absQuant[block_start+19] & mask) >> j) << 4);
                    }
                    if (block_start+20 < block_end) {
                        tmp_char2 |= (((absQuant[block_start+20] & mask) >> j) << 3);
                    }
                    if (block_start+21 < block_end) {
                        tmp_char2 |= (((absQuant[block_start+21] & mask) >> j) << 2);
                    }
                    if (block_start+22 < block_end) {
                        tmp_char2 |= (((absQuant[block_start+22] & mask) >> j) << 1);
                    }
                    if (block_start+23 < block_end) {
                        tmp_char2 |= (((absQuant[block_start+23] & mask) >> j) << 0);
                    }
                    
                    // Group 4: 24-31 - fully unrolled
                    if (block_start+24 < block_end) {
                        tmp_char3 |= (((absQuant[block_start+24] & mask) >> j) << 7);
                    }
                    if (block_start+25 < block_end) {
                        tmp_char3 |= (((absQuant[block_start+25] & mask) >> j) << 6);
                    }
                    if (block_start+26 < block_end) {
                        tmp_char3 |= (((absQuant[block_start+26] & mask) >> j) << 5);
                    }
                    if (block_start+27 < block_end) {
                        tmp_char3 |= (((absQuant[block_start+27] & mask) >> j) << 4);
                    }
                    if (block_start+28 < block_end) {
                        tmp_char3 |= (((absQuant[block_start+28] & mask) >> j) << 3);
                    }
                    if (block_start+29 < block_end) {
                        tmp_char3 |= (((absQuant[block_start+29] & mask) >> j) << 2);
                    }
                    if (block_start+30 < block_end) {
                        tmp_char3 |= (((absQuant[block_start+30] & mask) >> j) << 1);
                    }
                    if (block_start+31 < block_end) {
                        tmp_char3 |= (((absQuant[block_start+31] & mask) >> j) << 0);
                    }
                    
                    // Store data to compressed array - sequential writes
                    cmpData[cmp_byte_ofs++] = tmp_char0;
                    cmpData[cmp_byte_ofs++] = tmp_char1;
                    cmpData[cmp_byte_ofs++] = tmp_char2;
                    cmpData[cmp_byte_ofs++] = tmp_char3;
                    
                    // Move to next bit position
                    j++;
                }
            }
        }
        
        // Only the last active thread calculates final compressed size
        #pragma omp master
        {
            unsigned int cmpBlockInBytes = 0;
            for (int i = 0; i < num_threads; i++) {
                cmpBlockInBytes += threadOfs[i];
            }
            *cmpSize = (size_t)(cmpBlockInBytes + ((chunk_size + block_size - 1) / block_size) * num_threads);
        }
    }
}

void hawkZip_decompress_kernel(float* decData, unsigned char* cmpData, int* absQuant, int* fixedRate, unsigned int* threadOfs, size_t nbEle, float errorBound)
{
    // Determine optimal thread count based on available hardware and data size
    int num_threads = omp_get_max_threads();
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;
    
    // For very small datasets, reduce thread count to avoid overhead
    if (nbEle < 1024) num_threads = 1;
    else if (nbEle < 4096) num_threads = 2;
    else if (nbEle < 8192) num_threads = 4;
    
    // Calculate block size - try to align with cache line size (typically 64 bytes)
    const int block_size = 32; // Keep fixed block size for bit operations
    
    // Calculate chunks with better load balancing
    int chunk_size = (nbEle + num_threads - 1) / num_threads;
    // Round chunk size to multiple of block_size to avoid thread boundary issues
    chunk_size = ((chunk_size + block_size - 1) / block_size) * block_size;
    
    // Initialize thread offset array
    #pragma omp parallel for
    for (int i = 0; i < num_threads; i++) {
        threadOfs[i] = 0;
    }
    
    // hawkZip parallel decompression begin
    #pragma omp parallel num_threads(num_threads)
    {
        // Divide data chunk for each thread
        int thread_id = omp_get_thread_num();
        int start = thread_id * chunk_size;
        int end = start + chunk_size;
        if (end > nbEle) end = nbEle;
        
        // Skip empty chunks (might happen with uneven distribution)
        if (start < nbEle) {
            int block_num = (end - start + block_size - 1) / block_size;
            int start_block = thread_id * ((chunk_size + block_size - 1) / block_size);
            unsigned int thread_ofs = 0;
            
            // First pass - read fixed rates and calculate offsets
            for (int i = 0; i < block_num; i++) {
                int curr_block = start_block + i;
                int temp_fixed_rate = (int)cmpData[curr_block];
                fixedRate[curr_block] = temp_fixed_rate;
                
                // Calculate thread-local offset
                thread_ofs += temp_fixed_rate ? (block_size + temp_fixed_rate * block_size) / 8 : 0;
            }
            
            // Store thread offset for prefix sum
            threadOfs[thread_id] = thread_ofs;
        }
        
        // Synchronize all threads before prefix sum
        #pragma omp barrier
        
        // Perform exclusive prefix sum
        unsigned int global_ofs = 0;
        if (thread_id > 0) {
            for (int i = 0; i < thread_id; i++) {
                global_ofs += threadOfs[i];
            }
        }
        
        // Calculate byte offset for compressed data
        int block_num = (end - start + block_size - 1) / block_size;
        int start_block = thread_id * ((chunk_size + block_size - 1) / block_size);
        unsigned int cmp_byte_ofs = global_ofs + ((chunk_size + block_size - 1) / block_size) * num_threads;
        
        // Skip empty chunks
        if (start < nbEle) {
            // Second pass - decode the data
            for (int i = 0; i < block_num; i++) {
                int block_start = start + i * block_size;
                int block_end = (block_start + block_size) > end ? end : block_start + block_size;
                int curr_block = start_block + i;
                int temp_fixed_rate = fixedRate[curr_block];
                
                // Create mask for zero/non-zero block
                unsigned int non_zero_mask = -((temp_fixed_rate != 0) ? 1 : 0);
                
                // Process sign information only for non-zero blocks
                unsigned int sign_flag = non_zero_mask & (
                    (0xff000000 & (cmpData[cmp_byte_ofs] << 24)) |
                    (0x00ff0000 & (cmpData[cmp_byte_ofs+1] << 16)) |
                    (0x0000ff00 & (cmpData[cmp_byte_ofs+2] << 8)) |
                    (0x000000ff & cmpData[cmp_byte_ofs+3]));
                
                // Increment byte offset only for non-zero blocks
                cmp_byte_ofs += non_zero_mask & 4;
                
                // Clear the absQuant array for this block
                for (int j = block_start; j < block_end; j++) {
                    absQuant[j] = 0;
                }
                
                // Process each bit position for non-zero blocks
                int j = 0;
                while (j < temp_fixed_rate & non_zero_mask) {
                    // Read bit data
                    unsigned char tmp_char0 = cmpData[cmp_byte_ofs];
                    unsigned char tmp_char1 = cmpData[cmp_byte_ofs+1];
                    unsigned char tmp_char2 = cmpData[cmp_byte_ofs+2];
                    unsigned char tmp_char3 = cmpData[cmp_byte_ofs+3];
                    
                    // Increment offset for next iteration
                    cmp_byte_ofs += non_zero_mask & 4;
                    
                    // Group 1: 0-7 - fully unrolled
                    if (block_start < block_end) {
                        absQuant[block_start] |= ((tmp_char0 >> 7) & 0x01) << j;
                    }
                    if (block_start+1 < block_end) {
                        absQuant[block_start+1] |= ((tmp_char0 >> 6) & 0x01) << j;
                    }
                    if (block_start+2 < block_end) {
                        absQuant[block_start+2] |= ((tmp_char0 >> 5) & 0x01) << j;
                    }
                    if (block_start+3 < block_end) {
                        absQuant[block_start+3] |= ((tmp_char0 >> 4) & 0x01) << j;
                    }
                    if (block_start+4 < block_end) {
                        absQuant[block_start+4] |= ((tmp_char0 >> 3) & 0x01) << j;
                    }
                    if (block_start+5 < block_end) {
                        absQuant[block_start+5] |= ((tmp_char0 >> 2) & 0x01) << j;
                    }
                    if (block_start+6 < block_end) {
                        absQuant[block_start+6] |= ((tmp_char0 >> 1) & 0x01) << j;
                    }
                    if (block_start+7 < block_end) {
                        absQuant[block_start+7] |= ((tmp_char0 >> 0) & 0x01) << j;
                    }
                    
                    // Group 2: 8-15 - fully unrolled
                    if (block_start+8 < block_end) {
                        absQuant[block_start+8] |= ((tmp_char1 >> 7) & 0x01) << j;
                    }
                    if (block_start+9 < block_end) {
                        absQuant[block_start+9] |= ((tmp_char1 >> 6) & 0x01) << j;
                    }
                    if (block_start+10 < block_end) {
                        absQuant[block_start+10] |= ((tmp_char1 >> 5) & 0x01) << j;
                    }
                    if (block_start+11 < block_end) {
                        absQuant[block_start+11] |= ((tmp_char1 >> 4) & 0x01) << j;
                    }
                    if (block_start+12 < block_end) {
                        absQuant[block_start+12] |= ((tmp_char1 >> 3) & 0x01) << j;
                    }
                    if (block_start+13 < block_end) {
                        absQuant[block_start+13] |= ((tmp_char1 >> 2) & 0x01) << j;
                    }
                    if (block_start+14 < block_end) {
                        absQuant[block_start+14] |= ((tmp_char1 >> 1) & 0x01) << j;
                    }
                    if (block_start+15 < block_end) {
                        absQuant[block_start+15] |= ((tmp_char1 >> 0) & 0x01) << j;
                    }
                    
                    // Group 3: 16-23 - fully unrolled
                    if (block_start+16 < block_end) {
                        absQuant[block_start+16] |= ((tmp_char2 >> 7) & 0x01) << j;
                    }
                    if (block_start+17 < block_end) {
                        absQuant[block_start+17] |= ((tmp_char2 >> 6) & 0x01) << j;
                    }
                    if (block_start+18 < block_end) {
                        absQuant[block_start+18] |= ((tmp_char2 >> 5) & 0x01) << j;
                    }
                    if (block_start+19 < block_end) {
                        absQuant[block_start+19] |= ((tmp_char2 >> 4) & 0x01) << j;
                    }
                    if (block_start+20 < block_end) {
                        absQuant[block_start+20] |= ((tmp_char2 >> 3) & 0x01) << j;
                    }
                    if (block_start+21 < block_end) {
                        absQuant[block_start+21] |= ((tmp_char2 >> 2) & 0x01) << j;
                    }
                    if (block_start+22 < block_end) {
                        absQuant[block_start+22] |= ((tmp_char2 >> 1) & 0x01) << j;
                    }
                    if (block_start+23 < block_end) {
                        absQuant[block_start+23] |= ((tmp_char2 >> 0) & 0x01) << j;
                    }
                    
                    // Group 4: 24-31 - fully unrolled
                    if (block_start+24 < block_end) {
                        absQuant[block_start+24] |= ((tmp_char3 >> 7) & 0x01) << j;
                    }
                    if (block_start+25 < block_end) {
                        absQuant[block_start+25] |= ((tmp_char3 >> 6) & 0x01) << j;
                    }
                    if (block_start+26 < block_end) {
                        absQuant[block_start+26] |= ((tmp_char3 >> 5) & 0x01) << j;
                    }
                    if (block_start+27 < block_end) {
                        absQuant[block_start+27] |= ((tmp_char3 >> 4) & 0x01) << j;
                    }
                    if (block_start+28 < block_end) {
                        absQuant[block_start+28] |= ((tmp_char3 >> 3) & 0x01) << j;
                    }
                    if (block_start+29 < block_end) {
                        absQuant[block_start+29] |= ((tmp_char3 >> 2) & 0x01) << j;
                    }
                    if (block_start+30 < block_end) {
                        absQuant[block_start+30] |= ((tmp_char3 >> 1) & 0x01) << j;
                    }
                    if (block_start+31 < block_end) {
                        absQuant[block_start+31] |= ((tmp_char3 >> 0) & 0x01) << j;
                    }
                    
                    // Move to next bit position
                    j++;
                }
                
                // De-quantize and store data back - unroll by 4 where possible
                int idx = block_start;
                // Main loop - unrolled by 4
                for (; idx <= block_end - 4; idx += 4) {
                    int sign_ofs0 = idx % block_size;
                    int sign_ofs1 = (idx+1) % block_size;
                    int sign_ofs2 = (idx+2) % block_size;
                    int sign_ofs3 = (idx+3) % block_size;
                    
                    // Get sign bits
                    unsigned int sign_bit0 = (sign_flag >> (31 - sign_ofs0)) & 1;
                    unsigned int sign_bit1 = (sign_flag >> (31 - sign_ofs1)) & 1;
                    unsigned int sign_bit2 = (sign_flag >> (31 - sign_ofs2)) & 1;
                    unsigned int sign_bit3 = (sign_flag >> (31 - sign_ofs3)) & 1;
                    
                    // Create sign masks
                    int sign_mask0 = -(int)sign_bit0;
                    int sign_mask1 = -(int)sign_bit1;
                    int sign_mask2 = -(int)sign_bit2;
                    int sign_mask3 = -(int)sign_bit3;
                    
                    // Apply sign conversions
                    int currQuant0 = (absQuant[idx] ^ sign_mask0) + sign_bit0;
                    int currQuant1 = (absQuant[idx+1] ^ sign_mask1) + sign_bit1;
                    int currQuant2 = (absQuant[idx+2] ^ sign_mask2) + sign_bit2;
                    int currQuant3 = (absQuant[idx+3] ^ sign_mask3) + sign_bit3;
                    
                    // Calculate results
                    float result0 = currQuant0 * errorBound * 2;
                    float result1 = currQuant1 * errorBound * 2;
                    float result2 = currQuant2 * errorBound * 2;
                    float result3 = currQuant3 * errorBound * 2;
                    
                    // Store results
                    decData[idx] = result0 * (non_zero_mask != 0);
                    decData[idx+1] = result1 * (non_zero_mask != 0);
                    decData[idx+2] = result2 * (non_zero_mask != 0);
                    decData[idx+3] = result3 * (non_zero_mask != 0);
                }
                
                // Handle remaining elements (less than 4)
                for (; idx < block_end; idx++) {
                    int sign_ofs = idx % block_size;
                    unsigned int sign_bit = (sign_flag >> (31 - sign_ofs)) & 1;
                    int sign_mask = -(int)sign_bit;
                    
                    int currQuant = (absQuant[idx] ^ sign_mask) + sign_bit;
                    float result = currQuant * errorBound * 2;
                    decData[idx] = result * (non_zero_mask != 0);
                }
            }
        }
    }
}
