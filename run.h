#ifndef COSC612_SPEEDUP_RUN_H
#define COSC612_SPEEDUP_RUN_H

enum Implementation {
  serial, dense, tiled, coarse
};

void run(int n, Implementation implType);
#endif