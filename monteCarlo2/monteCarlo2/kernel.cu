
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GPU_Sum.h"
#include <curand_kernel.h>
#include <curand.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <string.h>

#define TIMING


__global__ void monteCarlo(long long n, int* inCirc_d, curandState_t* states, unsigned int seed)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x*gridDim.x;
	double x, y;
	long long myTotal = 0;
	curand_init(seed+index, 0/*index*/, 0, &states[index]);
	for (long long i = index; i < n; i += stride) {
		x = curand_uniform_double(&states[index]);
		y = curand_uniform_double(&states[index]);
		myTotal += ((x*x + y*y) <= 1.0);
	}
	inCirc_d[index] += myTotal;
}

//Used to check if there are any errors launching the kernel
void CUDAErrorCheck()
{

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("CUDA error : %s (%d)\n", cudaGetErrorString(error), error);
		exit(0);
	}
}

int main()
{
	/*
	creates cuda rand states and mallocs space for them, each state is thread safe
	*/
	cudaSetDevice(0);
	cudaEvent_t gpuStart, gpuEnd;
	cudaEventCreate(&gpuStart);
	cudaEventCreate(&gpuEnd);
	printf("Size");
	for (int blocks = 1; blocks <= 128; blocks *= 2) {
		printf("\t%d", blocks);
	}
	printf("\n");
	for (long long tosses = 25000; tosses <= 10000000; tosses += 25000) {
		printf("%d", tosses);
		for (int blocks = 1; blocks <= 128; blocks *= 2) {
			long long N = tosses;
			int blockSize = 32;
			int numBlocks = blocks;// (int)(N + blockSize - 1) / blockSize;

			curandState_t* states;
			cudaMalloc((void**)&states, blockSize * numBlocks * sizeof(curandState_t));

			long long total = 0;
			int* inCirc_d;
			int* inCirc = (int*)malloc(blockSize * numBlocks * sizeof(int));


			cudaMalloc((void**)&inCirc_d, blockSize * numBlocks * sizeof(int));
			cudaMemset(inCirc_d, 0, blockSize * numBlocks * sizeof(int));
			//CUDAErrorCheck();

			cudaEventRecord(gpuStart);
			cudaEventSynchronize(gpuStart);
			monteCarlo << <numBlocks, blockSize >> > (N, inCirc_d, states, time(NULL));
			cudaEventRecord(gpuEnd);
			cudaEventSynchronize(gpuEnd);
			cudaDeviceSynchronize();

			float msUsed;
			cudaEventElapsedTime(&msUsed, gpuStart, gpuEnd);
			//printf("%d	%lld	%f\n", blocks, tosses, msUsed);
			printf("\t%f", msUsed);


			//CUDAErrorCheck();

			cudaMemcpy(inCirc, inCirc_d, blockSize * numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
			//CUDAErrorCheck();
			cudaDeviceSynchronize();

			//sumArray(inCirc, N, &total, numBlocks, blockSize, onGpu);

			/*for (long long i = 0; i < blockSize * numBlocks; i++) {
				total += inCirc[i];
				//printf("i = %lld and count = %d\n", i, inCirc[i]);
			}*/


			//double pi_estimate = 4 * total / (double)N;

			//printf("Total in Circle = %lld\nEstimate of Pi = %lf\n", total, pi_estimate);

			cudaFree(states);
			cudaFree(inCirc_d);
		}
		printf("\n");
	}
	cudaEventDestroy(gpuStart);
	cudaEventDestroy(gpuEnd);

	cudaDeviceReset();
	return 0;
}

