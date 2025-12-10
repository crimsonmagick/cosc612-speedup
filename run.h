#ifndef COSC612_SPEEDUP_RUN_H
#define COSC612_SPEEDUP_RUN_H

enum Implementation {
  serial, dense, tiled
};

void run(int n, Implementation implType);
#endif