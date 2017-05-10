#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


template<typename Type>
cudaError_t sumArray(Type *arr, int size, Type *out, int blockCount, int blockSize=32, bool arrayOnGPU=false, bool createNewStream=false);

template<typename Type>
__global__ void sumKernel(Type *arr, int size);

template<typename Type>
__global__ void binaryReduction(Type *arr, int size);