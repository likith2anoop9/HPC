#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "hawkZip_compressor.h"

void hawkZip_compress(float* oriData, unsigned char* cmpData, size_t nbEle, size_t* cmpSize, float errorBound)
{
    // Variables used in compression kernel
    // Calculate max possible blocks needed based on max threads
    int max_threads = omp_get_max_threads();
    if (max_threads > MAX_THREADS) max_threads = MAX_THREADS;
    
    // Determine block count dynamically
    int blockNum = ((nbEle + max_threads - 1) / max_threads + 31) / 32 * max_threads;
    
    // Allocate memory for compression data structures
    int* absQuant = (int*)calloc(nbEle, sizeof(int));
    unsigned int* signFlag = (unsigned int*)calloc(blockNum, sizeof(unsigned int));
    int* fixedRate = (int*)calloc(blockNum, sizeof(int));
    unsigned int* threadOfs = (unsigned int*)calloc(max_threads, sizeof(unsigned int));
    
    // Initialize output buffer
    memset(cmpData, 0, sizeof(float)*nbEle);
    double timerCMP_start, timerCMP_end;

    // Compression kernel computation and time measurement.
    timerCMP_start = omp_get_wtime();
    hawkZip_compress_kernel(oriData, cmpData, absQuant, signFlag, fixedRate, threadOfs, nbEle, cmpSize, errorBound);
    timerCMP_end = omp_get_wtime();
    printf("hawkZip   compression ratio:      %f\n", (float)(sizeof(float)*nbEle) / (float)(sizeof(unsigned char)*(*cmpSize)));
    printf("hawkZip   compression throughput: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0/1024.0)/(timerCMP_end-timerCMP_start));

    // Reallocate memory consumption of cmpData.
    cmpData = (unsigned char*)realloc(cmpData, sizeof(unsigned char)*(*cmpSize));

    free(absQuant);
    free(signFlag);
    free(fixedRate);
    free(threadOfs);
}

void hawkZip_decompress(float* decData, unsigned char* cmpData, size_t nbEle, float errorBound)
{
    // Varaibles used in decompression kernel
    // Calculate max possible blocks needed based on max threads
    int max_threads = omp_get_max_threads();
    if (max_threads > MAX_THREADS) max_threads = MAX_THREADS;
    
    // Determine block count dynamically
    int blockNum = ((nbEle + max_threads - 1) / max_threads + 31) / 32 * max_threads;
    
    // Allocate memory for decompression data structures
    int* absQuant = (int*)calloc(nbEle, sizeof(int));
    int* fixedRate = (int*)calloc(blockNum, sizeof(int));
    unsigned int* threadOfs = (unsigned int*)calloc(max_threads, sizeof(unsigned int));
    
    // Initialize output buffer
    memset(decData, 0, sizeof(float)*nbEle);
    double timerDEC_start, timerDEC_end;

    // Decompression kernel computation and time measurement.
    timerDEC_start = omp_get_wtime();
    hawkZip_decompress_kernel(decData, cmpData, absQuant, fixedRate, threadOfs, nbEle, errorBound);
    timerDEC_end = omp_get_wtime();
    printf("hawkZip decompression throughput: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0/1024.0)/(timerDEC_end-timerDEC_start));

    free(absQuant);
    free(fixedRate);
    free(threadOfs);
}
