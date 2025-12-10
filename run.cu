#include <stdio.h>
#include <stdlib.h>
#include "kernel.cuh"
#include "run.h"
#include "support.h"


void run(const int n, Implementation implType) {
  Timer timer;
  cudaError_t cuda_ret;

  // Initialize host variables ----------------------------------------------

  // printf("\nSetting up the problem...");
  fflush(stdout);
  startTime(&timer);

  float *A_h, *B_h, *C_h;
  float *A_d, *B_d, *C_d;
  size_t A_sz, B_sz, C_sz;

  A_sz = n * n;
  B_sz = n * n;
  C_sz = n * n;

  A_h = (float *) malloc(sizeof(float) * A_sz);
  for (unsigned int i = 0; i < A_sz; i++) { A_h[i] = (rand() % 100) / 100.00; }

  B_h = (float *) malloc(sizeof(float) * B_sz);
  for (unsigned int i = 0; i < B_sz; i++) { B_h[i] = (rand() % 100) / 100.00; }

  C_h = (float *) malloc(sizeof(float) * C_sz);

  stopTime(&timer);
  startTime(&timer);

  cudaMalloc(&A_d, sizeof(float) * A_sz);
  cudaMalloc(&B_d, sizeof(float) * B_sz);
  cudaMalloc(&C_d, sizeof(float) * C_sz);

  CHECK_CUDA_RESULT(cudaDeviceSynchronize());
  stopTime(&timer);
  fflush(stdout);
  startTime(&timer);

  cudaMemcpy(A_d, A_h, sizeof(float) * A_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, sizeof(float) * B_sz, cudaMemcpyHostToDevice);

  CHECK_CUDA_RESULT(cudaDeviceSynchronize());
  stopTime(&timer);

  fflush(stdout);

  dim3 blockDims(TILE_WIDTH / 2, TILE_WIDTH); // (8,16) threads per block
  dim3 gridDims((n + TILE_WIDTH - 1) / TILE_WIDTH,
                (n + TILE_WIDTH - 1) / TILE_WIDTH);

  const unsigned int block_size_dense = 16;
  dim3 blockDimsDense(block_size_dense, block_size_dense);
  dim3 gridDimsDense((n + block_size_dense - 1) / block_size_dense,
                     (n + block_size_dense - 1) / block_size_dense);

  startTime(&timer);
  constexpr int sample_size = 1;
  for (int i = 0; i < sample_size; i++) {
    if (implType == dense) {
      denseMultiply<<<gridDimsDense, blockDimsDense>>>(n, n, n, A_d, B_d, C_d);
    } else if (implType == tiled) {
      tiledMultiply<<<gridDims, blockDims>>>(n, n, n, A_d, B_d, C_d);
    } else {
      serialMultiply<<<1, 1>>>(n, n, n, A_d, B_d, C_d);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Kernel launch failed at n=%d, iter=%d: %s\n",
              n, i, cudaGetErrorString(err));
      exit(1);
    }
    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess) {
      FATAL("Unable to launch kernel");
    }
  }
  stopTime(&timer);
  printf("%d,%lld\n", n, elapsedTime(timer) / sample_size);
  fflush(stdout);


  cudaMemcpy(C_h, C_d, sizeof(float) * C_sz, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  // verify(A_h, B_h, C_h, n, n, n);

  free(A_h);
  free(B_h);
  free(C_h);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}
