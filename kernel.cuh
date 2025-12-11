#ifndef KERNEL_H
#define KERNEL_H
#define TILE_WIDTH 16
__global__ void serialMultiply(int m, int n, int k, const float *A, const float *B, float *C);

__host__ void serialHostMultiply(int m, int n, int k, const float *A, const float *B, float *C);

__global__ void denseMultiply(int m, int n, int k, const float *A, const float *B, float *C);

__global__ void tiledMultiply(int m, int n, int k, const float *A, const float *B, float *C);

__global__ void coarsenedMultiply(int m, int n, int k, const float *A, const float *B, float *C);

#endif
