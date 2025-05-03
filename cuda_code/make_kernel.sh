#! /bin/bash

# TODO: 将来直接调用下面这个命令
nvcc kernel_file.cu -arch=sm_80 -Xptxas -O3 -Xcompiler -O3 -std=c++11
