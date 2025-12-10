/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "run.h"

extern __global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float *C);

int main(int argc, char *argv[]) {
  int n;
  if (argc == 2) {
    n = atoi(argv[1]);
  } else {
    printf("\n    Invalid input parameters!"
      "\n    Usage: ./sgemm <m>            # All matrices are m x m"
      "\n");
    exit(0);
  }
  run(n);
  return 0;
}

