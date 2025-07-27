#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <omp.h>

// Dynamic thread count.
#define MAX_NUM_THREADS 16

void hawkZip_compress_kernel(float* oriData, unsigned char* cmpData, int* absQuant, unsigned int* signFlag, int* fixedRate, unsigned int* threadOfs, size_t nbEle, size_t* cmpSize, float errorBound)
{
    // Use all available cores, up to 16.
    int NUM_THREADS = omp_get_max_threads();
    if (NUM_THREADS > MAX_NUM_THREADS) NUM_THREADS = MAX_NUM_THREADS;
    // If the number of elements is small, use less threads.
    if (nbEle < 1000000) NUM_THREADS = (NUM_THREADS > 2) ? NUM_THREADS / 2 : 1;

    // Shared variables across threads.
    int chunk_size = (nbEle + NUM_THREADS - 1) / NUM_THREADS;
    omp_set_num_threads(NUM_THREADS);


    // LICM: Moved outside the parallel region.
    const float recip_precision = 0.5f/errorBound;
    const __m256 recip_vec = _mm256_set1_ps(recip_precision);
    const __m256 minus_half = _mm256_set1_ps(-0.5f);
    const __m256 plus_half = _mm256_set1_ps(0.5f);

    // HawkZip parallel compression begin.
    #pragma omp parallel
    {
        // Divide data for each thread.
        int thread_id = omp_get_thread_num();
        int start = thread_id * chunk_size;
        int end = start + chunk_size;
        if (end > nbEle) end = nbEle;
        int block_num = (chunk_size + 31) / 32;
        int start_block = thread_id * block_num;
        int block_start, block_end;
        int sign_ofs;
        unsigned int thread_ofs = 0;

        // Iterate all blocks in current thread.
        for(int i=0; i < block_num; i++)
        {
            // Block initialization.
            block_start = start + i*32;
            block_end = (block_start + 32) > end ? end : block_start + 32;
            float data_recip;
            int s;
            int curr_quant, max_quant=0;
            int curr_block = start_block + i;
            unsigned int sign_flag = 0;
            int temp_fixed_rate;

            // Prequantization, get absolute value for each data.
            int vectorized_end = block_end - ((block_end - block_start) % 8);

            for (int j=block_start; j < vectorized_end; j+=8) {
                // Load floats.
                __m256 data = _mm256_loadu_ps(&oriData[j]);
                // Scale data.
                __m256 scaled = _mm256_mul_ps(data, recip_vec);
                // s ensures accurate rounding if x is negative.
                __m256 cmp = _mm256_cmp_ps(scaled, minus_half, _CMP_LT_OS);
                __m256i s = _mm256_srli_epi32(_mm256_castps_si256(cmp), 31);
                // Round to int.
                __m256 rounded = _mm256_add_ps(scaled, plus_half);
                __m256i q = _mm256_sub_epi32(_mm256_cvttps_epi32(rounded), s);
                // Set sign.
                __m256i mask_neg = _mm256_cmpgt_epi32(_mm256_setzero_si256(), q);

                // Compute and store abs_q.
                __m256i abs_q = _mm256_abs_epi32(q);
                _mm256_storeu_si256((__m256i*)&absQuant[j], abs_q);

                // Temporary abs_q and sign values.
                int temp_abs[8];
                int temp_sign[8];
                _mm256_storeu_si256((__m256i*)temp_abs, abs_q);
                _mm256_storeu_si256((__m256i*)temp_sign, mask_neg);

                // Unrolled loop for setting signs.
                int sign_s0 = (j + 0) % 32;
                sign_flag |= (temp_sign[0] ? 1 : 0) << (31 - sign_s0);
                if (temp_abs[0] > max_quant) max_quant = temp_abs[0];

                int sign_s1 = (j + 1) % 32;
                sign_flag |= (temp_sign[1] ? 1 : 0) << (31 - sign_s1);
                if (temp_abs[1] > max_quant) max_quant = temp_abs[1];

                int sign_s2 = (j + 2) % 32;
                sign_flag |= (temp_sign[2] ? 1 : 0) << (31 - sign_s2);
                if (temp_abs[2] > max_quant) max_quant = temp_abs[2];

                int sign_s3 = (j + 3) % 32;
                sign_flag |= (temp_sign[3] ? 1 : 0) << (31 - sign_s3);
                if (temp_abs[3] > max_quant) max_quant = temp_abs[3];

                int sign_s4 = (j + 4) % 32;
                sign_flag |= (temp_sign[4] ? 1 : 0) << (31 - sign_s4);
                if (temp_abs[4] > max_quant) max_quant = temp_abs[4];

                int sign_s5 = (j + 5) % 32;
                sign_flag |= (temp_sign[5] ? 1 : 0) << (31 - sign_s5);
                if (temp_abs[5] > max_quant) max_quant = temp_abs[5];

                int sign_s6 = (j + 6) % 32;
                sign_flag |= (temp_sign[6] ? 1 : 0) << (31 - sign_s6);
                if (temp_abs[6] > max_quant) max_quant = temp_abs[6];

                int sign_s7 = (j + 7) % 32;
                sign_flag |= (temp_sign[7] ? 1 : 0) << (31 - sign_s7);
                if (temp_abs[7] > max_quant) max_quant = temp_abs[7];
            }

            // Scalar cleanup.
            for (int j=vectorized_end; j < block_end; j++) {
                float data_recip = oriData[j] * recip_precision;
                int s = data_recip >= -0.5f ? 0 : 1;
                int curr_quant = (int)(data_recip + 0.5f) - s;
                int sign_ofs = j % 32;
                sign_flag |= (curr_quant < 0) << (31 - sign_ofs);
                int abs_val = abs(curr_quant);
                absQuant[j] = abs_val;
                if (abs_val > max_quant) max_quant = abs_val;
            }


            // Record fixed-length encoding rate for each block.
            signFlag[curr_block] = sign_flag;
            temp_fixed_rate = max_quant==0 ? 0 : sizeof(int) * 8 - __builtin_clz(max_quant);
            fixedRate[curr_block] = temp_fixed_rate;
            cmpData[curr_block] = (unsigned char)temp_fixed_rate;

            // Inner thread prefix-sum.
            thread_ofs += temp_fixed_rate ? (32 + temp_fixed_rate*32) / 8 : 0;
        }

        // Store thread ofs to global varaible, used for later global prefix-sum.
        threadOfs[thread_id] = thread_ofs;
        #pragma omp barrier

        // Exclusive prefix-sum.
        unsigned int global_ofs = 0;
        for(int i=0; i < thread_id; i++) global_ofs += threadOfs[i];
        unsigned int cmp_byte_ofs = global_ofs + block_num * NUM_THREADS;

        // Fixed-length encoding and store data to compressed data.
        for(int i=0; i < block_num; i++)
        {
            // Block initialization.
            block_start = start + i*32;
            block_end = (block_start+32) > end ? end : block_start + 32;
            int curr_block = start_block + i;
            int temp_fixed_rate = fixedRate[curr_block];
            unsigned int sign_flag = signFlag[curr_block];

            // Operation for each block, if zero block then do nothing.
            if(temp_fixed_rate)
            {
                // Retrieve sign information for one block.
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag >> 24);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag >> 16);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag >> 8);
                cmpData[cmp_byte_ofs++] = 0xff & sign_flag;

                // Retrieve quant data for one block.
                unsigned char tmp_char0, tmp_char1, tmp_char2, tmp_char3;
                for(int j=0; j < temp_fixed_rate; j++)
                {
                    // Load 32 abs_q vals.
                    __m256i q0 = _mm256_loadu_si256((__m256i*)&absQuant[block_start + 0]);
                    __m256i q1 = _mm256_loadu_si256((__m256i*)&absQuant[block_start + 8]);
                    __m256i q2 = _mm256_loadu_si256((__m256i*)&absQuant[block_start + 16]);
                    __m256i q3 = _mm256_loadu_si256((__m256i*)&absQuant[block_start + 24]);

                    __m256i bitmask = _mm256_set1_epi32(1 << j);

                    // Left shift.
                    __m256i b0 = _mm256_srli_epi32(_mm256_and_si256(q0, bitmask), j);
                    __m256i b1 = _mm256_srli_epi32(_mm256_and_si256(q1, bitmask), j);
                    __m256i b2 = _mm256_srli_epi32(_mm256_and_si256(q2, bitmask), j);
                    __m256i b3 = _mm256_srli_epi32(_mm256_and_si256(q3, bitmask), j);
                    
                    // Scalar ls bit is put into bytes.
                    int* v0 = (int*)&b0;
                    int* v1 = (int*)&b1;
                    int* v2 = (int*)&b2;
                    int* v3 = (int*)&b3;

                    // Unrolled loop to put the lowest bit of 8 integers into one byte for every vector.
                    tmp_char0 =   ((v0[0] & 1) << 7) | ((v0[1] & 1) << 6)
                                | ((v0[2] & 1) << 5) | ((v0[3] & 1) << 4)
                                | ((v0[4] & 1) << 3) | ((v0[5] & 1) << 2)
                                | ((v0[6] & 1) << 1) | ((v0[7] & 1) << 0);

                    tmp_char1 =   ((v1[0] & 1) << 7) | ((v1[1] & 1) << 6)
                                | ((v1[2] & 1) << 5) | ((v1[3] & 1) << 4)
                                | ((v1[4] & 1) << 3) | ((v1[5] & 1) << 2)
                                | ((v1[6] & 1) << 1) | ((v1[7] & 1) << 0);

                    tmp_char2 =   ((v2[0] & 1) << 7) | ((v2[1] & 1) << 6)
                                | ((v2[2] & 1) << 5) | ((v2[3] & 1) << 4)
                                | ((v2[4] & 1) << 3) | ((v2[5] & 1) << 2)
                                | ((v2[6] & 1) << 1) | ((v2[7] & 1) << 0);

                    tmp_char3 =   ((v3[0] & 1) << 7) | ((v3[1] & 1) << 6)
                                | ((v3[2] & 1) << 5) | ((v3[3] & 1) << 4)
                                | ((v3[4] & 1) << 3) | ((v3[5] & 1) << 2)
                                | ((v3[6] & 1) << 1) | ((v3[7] & 1) << 0);


                    // Store data to compressed data array.
                    cmpData[cmp_byte_ofs++] = tmp_char0;
                    cmpData[cmp_byte_ofs++] = tmp_char1;
                    cmpData[cmp_byte_ofs++] = tmp_char2;
                    cmpData[cmp_byte_ofs++] = tmp_char3;
                }
            }
        }

        // Return the compression data length.
        if(thread_id == NUM_THREADS - 1)
        {
            unsigned int cmpBlockInBytes = 0;
            for(int i=0; i<=thread_id; i++) cmpBlockInBytes += threadOfs[i];
            *cmpSize = (size_t)(cmpBlockInBytes + block_num * NUM_THREADS);
        }
    }
}

void hawkZip_decompress_kernel(float* decData, unsigned char* cmpData, int* absQuant, int* fixedRate, unsigned int* threadOfs, size_t nbEle, float errorBound)
{
    // Use all available cores, up to 16.
    int NUM_THREADS = omp_get_max_threads();
    if (NUM_THREADS > MAX_NUM_THREADS) NUM_THREADS = MAX_NUM_THREADS;
    // If the number of elements is small, use less threads.
    if (nbEle < 1000000) NUM_THREADS = (NUM_THREADS > 2) ? NUM_THREADS / 2 : 1;

    // Shared variables across threads.
    int chunk_size = (nbEle + NUM_THREADS - 1) / NUM_THREADS;
    omp_set_num_threads(NUM_THREADS);


    // LICM: Moved outside the parallel region
    const float error_bound_multiplier = errorBound * 2; //
    const __m256 errorBound_vec = _mm256_set1_ps(errorBound * 2);

    // HawkZip parallel decompression begin.
    #pragma omp parallel
    {
        // Divide data for each thread.
        int thread_id = omp_get_thread_num();
        int start = thread_id * chunk_size;
        int end = start + chunk_size;
        if (end > nbEle) end = nbEle;
        int block_num = (chunk_size+31)/32;
        int block_start, block_end;
        int start_block = thread_id * block_num;
        unsigned int thread_ofs = 0;


        // Iterate all blocks in current thread.
        for(int i=0; i<block_num; i++)
        {
            // Retrieve fixed-rate for each block in the compressed data.
            int curr_block = start_block + i;
            int temp_fixed_rate = (int)cmpData[curr_block];
            fixedRate[curr_block] = temp_fixed_rate;

            // Inner thread prefix-sum.
            thread_ofs += temp_fixed_rate ? (32+temp_fixed_rate*32)/8 : 0;
        }

        // Store thread ofs to global varaible, used for later global prefix-sum.
        threadOfs[thread_id] = thread_ofs;
        #pragma omp barrier

        // Exclusive prefix-sum.
        unsigned int global_ofs = 0;
        for(int i=0; i < thread_id; i++) global_ofs += threadOfs[i];
        unsigned int cmp_byte_ofs = global_ofs + block_num * NUM_THREADS;

        // Restore decompressed data.
        for(int i=0; i < block_num; i++)
        {
            // Block initialization.
            block_start = start + i * 32;
            block_end = (block_start+32) > end ? end : block_start+32;
            int curr_block = start_block + i;
            int temp_fixed_rate = fixedRate[curr_block];
            unsigned int sign_flag = 0;
            int sign_ofs;

            // Operation for each block, if zero block then do nothing.
            if(temp_fixed_rate)
            {
                // Retrieve sign information for one block.
                sign_flag = (0xff000000 & (cmpData[cmp_byte_ofs++] << 24)) |
                            (0x00ff0000 & (cmpData[cmp_byte_ofs++] << 16)) |
                            (0x0000ff00 & (cmpData[cmp_byte_ofs++] << 8))  |
                            (0x000000ff & cmpData[cmp_byte_ofs++]);

                // Retrieve quant data for one block.
                unsigned char tmp_char0, tmp_char1, tmp_char2, tmp_char3;
                for(int j=0; j < temp_fixed_rate; j++)
                {
                    // Initialization.
                    tmp_char0 = cmpData[cmp_byte_ofs++];
                    tmp_char1 = cmpData[cmp_byte_ofs++];
                    tmp_char2 = cmpData[cmp_byte_ofs++];
                    tmp_char3 = cmpData[cmp_byte_ofs++];

                    // Unrolled loop to get ith bit in 0~31 abs quant from global memory.
                    absQuant[block_start]   |= ((tmp_char0 >> (7+block_start-block_start)) & 0x00000001) << j;
                    absQuant[block_start+1] |= ((tmp_char0 >> (7+block_start-(block_start+1))) & 0x00000001) << j;
                    absQuant[block_start+2] |= ((tmp_char0 >> (7+block_start-(block_start+2))) & 0x00000001) << j;
                    absQuant[block_start+3] |= ((tmp_char0 >> (7+block_start-(block_start+3))) & 0x00000001) << j;
                    absQuant[block_start+4] |= ((tmp_char0 >> (7+block_start-(block_start+4))) & 0x00000001) << j;
                    absQuant[block_start+5] |= ((tmp_char0 >> (7+block_start-(block_start+5))) & 0x00000001) << j;
                    absQuant[block_start+6] |= ((tmp_char0 >> (7+block_start-(block_start+6))) & 0x00000001) << j;
                    absQuant[block_start+7] |= ((tmp_char0 >> (7+block_start-(block_start+7))) & 0x00000001) << j;

                    absQuant[block_start+8]  |= ((tmp_char1 >> (15+block_start-(block_start+8))) & 0x00000001) << j;
                    absQuant[block_start+9]  |= ((tmp_char1 >> (15+block_start-(block_start+9))) & 0x00000001) << j;
                    absQuant[block_start+10] |= ((tmp_char1 >> (15+block_start-(block_start+10))) & 0x00000001) << j;
                    absQuant[block_start+11] |= ((tmp_char1 >> (15+block_start-(block_start+11))) & 0x00000001) << j;
                    absQuant[block_start+12] |= ((tmp_char1 >> (15+block_start-(block_start+12))) & 0x00000001) << j;
                    absQuant[block_start+13] |= ((tmp_char1 >> (15+block_start-(block_start+13))) & 0x00000001) << j;
                    absQuant[block_start+14] |= ((tmp_char1 >> (15+block_start-(block_start+14))) & 0x00000001) << j;
                    absQuant[block_start+15] |= ((tmp_char1 >> (15+block_start-(block_start+15))) & 0x00000001) << j;

                    absQuant[block_start+16] |= ((tmp_char2 >> (23+block_start-(block_start+16))) & 0x00000001) << j;
                    absQuant[block_start+17] |= ((tmp_char2 >> (23+block_start-(block_start+17))) & 0x00000001) << j;
                    absQuant[block_start+18] |= ((tmp_char2 >> (23+block_start-(block_start+18))) & 0x00000001) << j;
                    absQuant[block_start+19] |= ((tmp_char2 >> (23+block_start-(block_start+19))) & 0x00000001) << j;
                    absQuant[block_start+20] |= ((tmp_char2 >> (23+block_start-(block_start+20))) & 0x00000001) << j;
                    absQuant[block_start+21] |= ((tmp_char2 >> (23+block_start-(block_start+21))) & 0x00000001) << j;
                    absQuant[block_start+22] |= ((tmp_char2 >> (23+block_start-(block_start+22))) & 0x00000001) << j;
                    absQuant[block_start+23] |= ((tmp_char2 >> (23+block_start-(block_start+23))) & 0x00000001) << j;

                    absQuant[block_start+24] |= ((tmp_char3 >> (31+block_start-(block_start+24))) & 0x00000001) << j;
                    absQuant[block_start+25] |= ((tmp_char3 >> (31+block_start-(block_start+25))) & 0x00000001) << j;
                    absQuant[block_start+26] |= ((tmp_char3 >> (31+block_start-(block_start+26))) & 0x00000001) << j;
                    absQuant[block_start+27] |= ((tmp_char3 >> (31+block_start-(block_start+27))) & 0x00000001) << j;
                    absQuant[block_start+28] |= ((tmp_char3 >> (31+block_start-(block_start+28))) & 0x00000001) << j;
                    absQuant[block_start+29] |= ((tmp_char3 >> (31+block_start-(block_start+29))) & 0x00000001) << j;
                    absQuant[block_start+30] |= ((tmp_char3 >> (31+block_start-(block_start+30))) & 0x00000001) << j;
                    absQuant[block_start+31] |= ((tmp_char3 >> (31+block_start-(block_start+31))) & 0x00000001) << j;
                }

                // De-quantize and store data back to decompression data. (Vectorized)
                int vectorized_end = block_end - ((block_end - block_start) % 8);

                for(int i=block_start; i < vectorized_end; i+=8) {    // Process 8 elements at a time
                    // Fully unrolled sign application Element 0 to 7
                    sign_ofs = i % 32;
                    if(sign_flag & (1 << (31 - sign_ofs))) {
                        absQuant[i] = -absQuant[i];
                    }
                    sign_ofs = (i+1) % 32;
                    if(sign_flag & (1 << (31 - sign_ofs))) {
                        absQuant[i+1] = -absQuant[i+1];
                    }
                    sign_ofs = (i+2) % 32;
                    if(sign_flag & (1 << (31 - sign_ofs))) {
                        absQuant[i+2] = -absQuant[i+2];
                    }
                    sign_ofs = (i+3) % 32;
                    if(sign_flag & (1 << (31 - sign_ofs))) {
                        absQuant[i+3] = -absQuant[i+3];
                    }
                    sign_ofs = (i+4) % 32;
                    if(sign_flag & (1 << (31 - sign_ofs))) {
                        absQuant[i+4] = -absQuant[i+4];
                    }
                    sign_ofs = (i+5) % 32;
                    if(sign_flag & (1 << (31 - sign_ofs))) {
                        absQuant[i+5] = -absQuant[i+5];
                    }
                    sign_ofs = (i+6) % 32;
                    if(sign_flag & (1 << (31 - sign_ofs))) {
                        absQuant[i+6] = -absQuant[i+6];
                    }
                    sign_ofs = (i+7) % 32;
                    if(sign_flag & (1 << (31 - sign_ofs))) {
                        absQuant[i+7] = -absQuant[i+7];
                    }

                    // Load values with applied signs.
                    __m256i curr_quant_vec = _mm256_loadu_si256((__m256i*)&absQuant[i]);
                    // Convert to float.
                    __m256 float_values = _mm256_cvtepi32_ps(curr_quant_vec);
                    // Apply error bound.
                    float_values = _mm256_mul_ps(float_values, errorBound_vec);
                    // Store result.
                    _mm256_storeu_ps(&decData[i], float_values);
                }

                 // Scalar cleanup.
                for(int i=vectorized_end; i < block_end; i++) {
                    sign_ofs = i % 32;
                    if(sign_flag & (1 << (31 - sign_ofs)))
                        absQuant[i] = -absQuant[i];
                    else
                        absQuant[i] = absQuant[i];
                    decData[i] = absQuant[i] * errorBound * 2;
                }
            }
        }
    }
}
