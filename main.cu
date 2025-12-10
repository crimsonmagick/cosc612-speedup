/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "run.h"

extern __global__ void denseMultiply(int m, int n, int k, const float *A, const float *B, float *C);

int main(int argc, char *argv[]) {
  int max_n;
  std::string mode;
  bool is_dense;
  if (argc == 3) {
    max_n = atoi(argv[1]);
    mode = argv[2];
  } else {
    printf("\n    Invalid input parameters!"
      "\n    Usage: ./sgemm <m> <mode>           # All matrices are m x m"
      "\n");
    exit(0);
  }
  if (mode == "tiled") {
    is_dense = false;
  } else if (mode == "dense") {
    is_dense = true;
  } else {
    printf("\n    Invalid mode input parameter: %s!\n", mode.c_str());
    exit(0);
  }
  printf("n,time_ms\n");
  int n = 2;
  while (n <= max_n) {
    run(n, is_dense);
    n *= 2;
  }
  return 0;
}

