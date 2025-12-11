#!/bin/bash

N=2048
#echo "Serial"
./speedup $N serial | tee -a serial.csv

#echo "Dense"
./speedup $N dense | tee -a dense.csv

#echo "Tiled"
./speedup $N tiled | tee -a tiled.csv

