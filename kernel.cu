/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include "kernel.cuh"
__global__ void tiledMultiply(int m, int n, int k,
                              const float *A, const float *B, float *C) {
  __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];  // Will store B transposed

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Identify the row and column of the C element to work on
  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  float cValue = 0.0f;

  // Loop over tiles
  for (int p = 0; p < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++p) {
    // Load tile from A (normal)
    int a_col = p * TILE_WIDTH + tx;
    if (row < m && a_col < k) {
      tileA[ty][tx] = A[row * k + a_col];
    } else {
      tileA[ty][tx] = 0.0f;
    }

    // Load tile from B and store it TRANSPOSED in shared memory
    // This is the key optimization!
    int b_row = p * TILE_WIDTH + ty;
    int b_col = bx * TILE_WIDTH + tx;  // Original column in B
    if (b_row < k && b_col < n) {
      // Store B transposed: tileB[tx][ty] = B[b_row * n + b_col]
      tileB[tx][ty] = B[b_row * n + b_col];
    } else {
      tileB[tx][ty] = 0.0f;
    }

    __syncthreads();

    // Compute with coalesced access pattern
    // Now tileB is transposed, so we access tileB[tx][i] which is coalesced
#pragma unroll
    for (int i = 0; i < TILE_WIDTH; ++i) {
      cValue += tileA[ty][i] * tileB[tx][i];  // Coalesced access!
    }

    __syncthreads();
  }

  if (row < m && col < n) {
    C[row * n + col] = cValue;
  }
}

__global__ void denseMultiply(int m, int n, int k, const float *A, const float *B, float *C) {
  /********************************************************************
   *
   * Compute C = A x B
   * where A is a (m x k) matrix
   * where B is a (k x n) matrix
   * where C is a (m x n) matrix
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