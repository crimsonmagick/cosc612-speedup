/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include "kernel.cuh"
#define TILE_WIDTH 16
__global__ void matrixMultiplyShared(int m, int n, int k,
                                     const float *A, const float *B, float *C) {
	/********************************************************************
 *
 * Compute C = A x B
 *   where A is a (m x k) matrix
 *   where B is a (k x n) matrix
 *   where C is a (m x n) matrix
 *
 ********************************************************************/

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

		for (int k = 0; k < TILE_WIDTH; ++k) {
			cValue += tileA[ty][k] * tileB[k][tx];
		}

		__syncthreads();
	}

	if (cRow < m && cCol < n) {
		C[cRow * n + cCol] = cValue; // 1 global write
	}
}

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float *C) {
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

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda,
                const float *B, int ldb, float beta, float *C, int ldc) {
	if ((transa != 'N') && (transa != 'n')) {
		printf("unsupported value of 'transa'\n");
		return;
	}

	if ((transb != 'N') && (transb != 'n')) {
		printf("unsupported value of 'transb'\n");
		return;
	}

	if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
		printf("unsupported value of alpha\n");
		return;
	}

	if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
		printf("unsupported value of beta\n");
		return;
	}

	// Initialize thread block and kernel grid dimensions ---------------------

	const unsigned int BLOCK_SIZE = TILE_WIDTH; // Use 16x16 thread blocks, same as tile size

	//INSERT CODE HERE to define thread blocks and layout

	const unsigned int GRID_X = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	const unsigned int GRID_Y = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;

	dim3 blockDims(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDims(GRID_X, GRID_Y);

	// Invoke CUDA kernel -----------------------------------------------------
	matrixMultiplyShared<<<gridDims, blockDims>>>(m, n, k, A, B, C);
}


