/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include "kernel.cuh"
#include <stdio.h>
#include "kernel.cuh"

__global__ void tiledMultiply(int m, int n, int k,
                              const float *__restrict__ A,
                              const float *__restrict__ B,
                              float *__restrict__ C)
{
    // blockDim = (8,16) → each block computes 16x16 tile of C
    int tx = threadIdx.x;      // 0..7
    int ty = threadIdx.y;      // 0..15

    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global C row & two column indices this thread computes
    int cRow  = by * TILE_WIDTH + ty;
    int cCol0 = bx * TILE_WIDTH + tx * 2;
    int cCol1 = cCol0 + 1;

    // Shared tiles
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    float c0 = 0.0f;
    float c1 = 0.0f;

    int numTiles = (k + TILE_WIDTH - 1) / TILE_WIDTH;

    // iterate over A/B tiles along k dimension
    for (int t = 0; t < numTiles; t++)
    {
        int aCol0 = t * TILE_WIDTH + tx * 2;
        int aCol1 = aCol0 + 1;

        // Load A tile row-wise
        if (cRow < m && aCol0 < k)
            tileA[ty][tx * 2] = A[cRow * k + aCol0];
        else
            tileA[ty][tx * 2] = 0.0f;

        if (cRow < m && aCol1 < k)
            tileA[ty][tx * 2 + 1] = A[cRow * k + aCol1];
        else
            tileA[ty][tx * 2 + 1] = 0.0f;

        // Load B tile correctly:
        // row index along K (shared dimension)
        int bRow = t * TILE_WIDTH + ty;

        if (bRow < k && cCol0 < n)
            tileB[ty][tx * 2] = B[bRow * n + cCol0];
        else
            tileB[ty][tx * 2] = 0.0f;

        if (bRow < k && cCol1 < n)
            tileB[ty][tx * 2 + 1] = B[bRow * n + cCol1];
        else
            tileB[ty][tx * 2 + 1] = 0.0f;

        __syncthreads();

        // Multiply accumulate over TILE_WIDTH
#pragma unroll
        for (int kk = 0; kk < TILE_WIDTH; kk++)
        {
            float a = tileA[ty][kk];
            c0 += a * tileB[kk][tx * 2];
            c1 += a * tileB[kk][tx * 2 + 1];
        }

        __syncthreads();
    }

    // Write output if in bounds
    if (cRow < m && cCol0 < n) C[cRow * n + cCol0] = c0;
    if (cRow < m && cCol1 < n) C[cRow * n + cCol1] = c1;
}


__global__ void denseMultiply(int m, int n, int k, const float *A, const float *B, float *C) {
  /********************************************************************
   *
   * Compute C = A x B
   *   where A is a (m x k) matrix
   *   where B is a (k x n) matrix
   *   where C is a (m x n) matrix
   *
   ********************************************************************/

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
  /********************************************************************
   *
   * Compute C = A x B
   *   where A is a (m x k) matrix
   *   where B is a (k x n) matrix
   *   where C is a (m x n) matrix
   *
   ********************************************************************/
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


