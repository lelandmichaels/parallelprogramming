#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <omp.h>
#include <cstdlib>

#define TIMING
#define NUM_THREADS 1
#define TIMING
#define MIN_SIZE 12500
#define SIZE_INCREMENT 2
#define MAX_SIZE 100000000
#define SAMPLE_SIZE 50

#define BLOCKS_MIN 1
#define BLOCKS_INCREMENT 2
#define BLOCKS_MAX 128

#ifdef TIMING
double avgCPUTimeMS, avgGPUTimeMS;
double cpuStartTime, cpuEndTime;
#endif // TIMING


// Kernel function to calculate trapizoidal sum
__host__
__device__
double f(double x) {
	return x*x;
}


cudaError_t trapezoidalMethod(double startValue, double endValue, int subdivisions, double *out, int blockCount, int blockSize = 32, bool createNewStream = false);

__global__ void trapKernel(double *arr, int n, double startValue, double h)
{
	double x_i;
	int stride = blockDim.x * gridDim.x;
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	double mySum = 0;
	for (int i = id + 1; i < n; i += stride) {
		x_i = startValue + i*h;
		mySum += f(x_i);
	}
	arr[id] = mySum;
}

template<typename Type>
__global__ void binaryReduction(Type *arr, int size)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	if (id < size) {
		arr[id] += arr[id + size];
	}
}


int main(void) {
#ifdef TIMING
	avgCPUTimeMS = 0;
	avgGPUTimeMS = 0;
#endif // TIMING
	int a = 1;
	int b = 2;
	double gpuSum, cpuSum = 0.0;
	int timesCorrect = 0, timesWrong = 0;
	printf("Size\tCPU");
	for (int blocks = BLOCKS_MIN; blocks <= BLOCKS_MAX; blocks *= BLOCKS_INCREMENT) {
		printf("\t%d", blocks);
	}
	printf("\n");
	for (int n = MIN_SIZE; n <= MAX_SIZE; n *= SIZE_INCREMENT) {
		double  h = (b - a) / (double)n;
		for (int i = 0; i < SAMPLE_SIZE; i++) {
#ifdef TIMING
			cpuStartTime = omp_get_wtime();
#endif // TIMING
			cpuSum = (f(a) + f(b)) / 2.0;
			for (int i = 1; i < n; i++) {
				cpuSum += f(a + h*i);
			}
			cpuSum *= h;
#ifdef TIMING
			cpuEndTime = omp_get_wtime();
			double timeUsed = 1000 * (cpuEndTime - cpuStartTime);
			avgCPUTimeMS += timeUsed;
#endif // TIMING
		}
		printf("%ld\t%lf", n, avgCPUTimeMS / SAMPLE_SIZE);
		for (int blocks = BLOCKS_MIN; blocks <= BLOCKS_MAX; blocks *= BLOCKS_INCREMENT) {
#ifdef TIMING
			avgGPUTimeMS = 0;
#endif // TIMING
			for (int sample = 0; sample < SAMPLE_SIZE; sample++) {
				cudaError_t trapezoidalLaunch = trapezoidalMethod(a, b, n, &gpuSum, blocks);
				if (trapezoidalLaunch == cudaSuccess) {
					if (abs(gpuSum - cpuSum) < 1.0 / n) {
						timesCorrect++;
					}
					else {
						timesWrong++;
					}
				}
				else {
					printf("There was an error runnning the operation.\n");
					printf("Error code: %d\n", trapezoidalLaunch);
				}
			}
#ifdef TIMING
			printf("\t%lf", avgGPUTimeMS / SAMPLE_SIZE);
#endif // TIMING
		}
		printf("\n");
	}
	//printf("GPU Implementation was correct %d times and incorrect %d times.\n", timesCorrect, timesWrong);
	return 0;
}


//cudaStatus = cudaStreamSynchronize(myStream);
cudaError_t trapezoidalMethod(double startValue, double endValue, int subdivisions, double *out, int blockCount, int blockSize, bool createNewStream) {
	cudaStream_t myStream = (cudaStream_t)(0);
	if (createNewStream) {
		cudaStreamCreateWithFlags(&myStream, cudaStreamDefault);
	}
#ifdef TIMING
	double totalGpuTimeUsed = 0;
	float gpuTimeUsed;
	cudaEvent_t startStep, endStep;
	cudaEventCreateWithFlags(&startStep, cudaEventDefault);
	cudaEventCreateWithFlags(&endStep, cudaEventDefault);
#endif // TIMING
	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	else {
		int size = blockCount*blockSize;
		double h = (endValue - startValue) / (double)subdivisions;
		//create array for sum reduction on gpu
		double *gpuArray;
		cudaMalloc((void **)&gpuArray, size * sizeof(double));
		cudaStatus = cudaStreamSynchronize(myStream);
		cudaMemsetAsync((void*)gpuArray, 0, size * sizeof(double), myStream);
		cudaStatus = cudaStreamSynchronize(myStream);
#ifdef TIMING
		cudaEventRecord(startStep, myStream);
#endif // TIMING
		trapKernel << <blockCount, blockSize, 0, myStream >> > (gpuArray, subdivisions, startValue, h);
#ifdef TIMING
		cudaEventRecord(endStep, myStream);
		cudaStatus = cudaStreamSynchronize(myStream);
		cudaEventElapsedTime(&gpuTimeUsed, startStep, endStep);
		totalGpuTimeUsed += gpuTimeUsed;
#endif // TIMING
		//keep reducing the problem size by two
		while (size > 1) {
			cudaStatus = cudaStreamSynchronize(myStream);
			size /= 2;
			if (size % 2 != 0 && size > 1) {
				size++;
			}
#ifdef TIMING
			cudaEventRecord(startStep, myStream);
			cudaStatus = cudaEventSynchronize(startStep);
#endif // TIMING
			// Launch a kernel on the GPU with one thread for each pair of elements.
			binaryReduction << <blockCount, blockSize, 0, myStream >> > (gpuArray, size);
#ifdef TIMING
			cudaEventRecord(endStep, myStream);
			cudaStatus = cudaEventSynchronize(endStep);
			cudaEventElapsedTime(&gpuTimeUsed, startStep, endStep);
			totalGpuTimeUsed += gpuTimeUsed;
#endif // TIMING
			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "sumArray Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
				break;
			}
		}
#ifdef TIMING
		avgGPUTimeMS += totalGpuTimeUsed;
#endif // TIMING
		if (cudaStatus == cudaSuccess) {
			cudaMemcpyAsync(out, gpuArray, sizeof(double), cudaMemcpyDeviceToHost, myStream);
			cudaStatus = cudaStreamSynchronize(myStream);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			}
			*out += (f(startValue) + f(endValue)) / 2.0;
			*out *= h;
		}
	}
	if (createNewStream) {
		cudaStreamDestroy(myStream);
	}
	return cudaStatus;
}