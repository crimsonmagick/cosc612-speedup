#ifndef KERNEL_H
#define KERNEL_H
#define TILE_WIDTH 32
__global__ void serialMultiply(int m, int n, int k, const float *A, const float *B, float *C);

__global__ void parallelMultiply(int m, int n, int k, const float *A, const float *B, float *C);
#endif
