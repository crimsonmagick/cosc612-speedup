#include <stdio.h>
#include "kernel.cuh"
#include <stdio.h>
#include "kernel.cuh"

__global__ void tiledMultiply(int m, int n, int k,
                                     const float *A, const float *B, float *C) {

    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identify the row and column of the C element to work on
    int cRow = by * TILE_WIDTH + ty;
    int cCol = bx * TILE_WIDTH + tx;

    float cValue = 0;
    for (int i = 0; i < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++i) {
        // Collaborative loading of A and B tiles into shared memory
        const int aCol = i * TILE_WIDTH + tx;
        if (cRow < m && aCol < k) {
            tileA[ty][tx] = A[cRow * k + aCol]; // 1 global read
        } else {
            tileA[ty][tx] = 0.0f; // 0 global read
        }
        const int bRow = i * TILE_WIDTH + ty;
        if (cCol < n && bRow < k) {
            tileB[ty][tx] = B[bRow * n + cCol]; // 1 global read
        } else {
            tileB[ty][tx] = 0.0f; // 0 global read
        }
        __syncthreads();

        for (int kk = 0; kk < TILE_WIDTH; ++kk) {
            cValue += tileA[ty][kk] * tileB[kk][tx];
        }

        __syncthreads();
    }

    if (cRow < m && cCol < n) {
        C[cRow * n + cCol] = cValue; // 1 global write
    }
}

__global__ void denseMultiply(int m, int n, int k, const float *A, const float *B, float *C) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    float acc = 0;
    for (int i = 0; i < k; i++) {
      acc += A[row * k + i] * B[i * n + col];
    }
    C[row * n + col] = acc;
  }
}

__global__ void serialMultiply(int m, int n, int k, const float *A, const float *B, float *C) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float sum = 0;
      for (unsigned int i = 0; i < k; ++i) {
        sum += A[row * k + i] * B[i * n + col];
      }
      C[row * n + col] = sum;
    }
  }
}

__host__ void serialHostMultiply(int m, int n, int k, const float *A, const float *B, float *C) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0;
            for (unsigned int i = 0; i < k; ++i) {
                sum += A[row * k + i] * B[i * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}


