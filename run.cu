#include <stdio.h>
#include <stdlib.h>
#include "kernel.cuh"
#include "run.h"
#include "support.h"


void run(int n) {
  Timer timer;
  cudaError_t cuda_ret;

  // Initialize host variables ----------------------------------------------

  // printf("\nSetting up the problem...");
  fflush(stdout);
  startTime(&timer);

  float *A_h, *B_h, *C_h;
  float *A_d, *B_d, *C_d;
  size_t A_sz, B_sz, C_sz;
  unsigned matArow, matAcol;
  unsigned matBrow, matBcol;
  dim3 dim_grid, dim_block;

  A_sz = n * n;
  B_sz = n * n;
  C_sz = n * n;

  A_h = (float *) malloc(sizeof(float) * A_sz);
  for (unsigned int i = 0; i < A_sz; i++) { A_h[i] = (rand() % 100) / 100.00; }

  B_h = (float *) malloc(sizeof(float) * B_sz);
  for (unsigned int i = 0; i < B_sz; i++) { B_h[i] = (rand() % 100) / 100.00; }

  C_h = (float *) malloc(sizeof(float) * C_sz);

  stopTime(&timer);
  // printf("%f s\n", elapsedTime(timer));
  // printf("    A: %u x %u\n    B: %u x %u\n    C: %u x %u\n", matArow, matAcol,
  //        matBrow, matBcol, matArow, matBcol);

  // Allocate device variables ----------------------------------------------

  // printf("Allocating device variables...");
  // fflush(stdout);
  startTime(&timer);

  cudaMalloc(&A_d, sizeof(float) * A_sz);
  cudaMalloc(&B_d, sizeof(float) * B_sz);
  cudaMalloc(&C_d, sizeof(float) * C_sz);

  CHECK_CUDA_RESULT(cudaDeviceSynchronize());
  stopTime(&timer);

  // Copy host variables to device ------------------------------------------

  // printf("Copying data from host to device...");
  fflush(stdout);
  startTime(&timer);

  cudaMemcpy(A_d, A_h, sizeof(float) * A_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, sizeof(float) * B_sz, cudaMemcpyHostToDevice);

  CHECK_CUDA_RESULT(cudaDeviceSynchronize());
  stopTime(&timer);

  // Launch kernel using standard sgemm interface ---------------------------
  fflush(stdout);
  startTime(&timer);
  basicSgemm('N', 'N', matArow, matBcol, matBrow, 1.0f,
             A_d, matArow, B_d, matBrow, 0.0f, C_d, matBrow);

  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to launch kernel");
  stopTime(&timer);

  // Copy device variables from host ----------------------------------------

  // printf("Copying data from device to host...");
  // fflush(stdout);
  startTime(&timer);

  //INSERT CODE HERE
  cudaMemcpy(C_h, C_d, sizeof(float) * C_sz, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  stopTime(&timer);
  printf("%d,%f\n", matArow, elapsedTime(timer));
  fflush(stdout);

  // Free memory ------------------------------------------------------------

  free(A_h);
  free(B_h);
  free(C_h);

  //INSERT CODE HERE
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}
