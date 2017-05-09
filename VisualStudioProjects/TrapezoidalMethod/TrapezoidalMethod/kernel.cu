#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <cstdlib>

#define TIMING

#ifdef TIMING
double avgCPUTime, avgGPUTime;
double cpuStartTime, cpuEndTime;
#endif // TIMING


// Kernel function to calculate trapizoidal sum
__host__
__device__
double f(double x) {
	return x*x;
}

__global__
void trap(int a, int n, double h, double* sum) {
	double x_i;
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x*gridDim.x;
	for (int i = id; i < n; i += stride) {
		x_i = a + i*h;
		sum[id] += f(x_i);
	}
}

cudaError_t sumArray(double *arr, int size, double *out, int threadsPerBlock, bool arrayAlreadyOnGPU);

__global__ void sumKernel(double *arr, int size)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < size) {
		arr[i] += arr[i + size];
		arr[i + size] = 0;
	}
}

cudaError_t trapezoidalMethod(double start, double end, int subdivisions, double *out, int blockCount, int blockSize) {
#ifdef TIMING
	double totalGpuTimeUsed = 0;
	float gpuTimeUsed;
	cudaEvent_t startStep, endStep;
	cudaEventCreateWithFlags(&startStep, cudaEventDefault);//cudaEventBlockingSync);
	cudaEventCreateWithFlags(&endStep, cudaEventDefault);// cudaEventBlockingSync);
#endif // TIMING
	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	else {
		cudaStream_t myStream;
		cudaStreamCreateWithFlags(&myStream, cudaStreamNonBlocking);
#ifdef TIMING
		cudaEventRecord(startStep, myStream);
		cudaStreamSynchronize(myStream);
#endif // TIMING
		// Launch a kernel on the GPU with one thread for each element.
		double  h = (end - start) / (double)subdivisions;
		double *gpuSum, cpuSum;
		cudaMalloc(&gpuSum, sizeof(double)*blockCount*blockSize);
		cudaStreamSynchronize(myStream);
		trap<<<blockCount, blockSize, 0, myStream >>>(start, subdivisions, h, gpuSum);
		cudaStreamSynchronize(myStream);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Trapezoidal launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		else {
			//reduce array to one value now
			sumArray(gpuSum, blockCount*blockSize, &cpuSum, blockSize, true);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "Reduction failed: %s\n", cudaGetErrorString(cudaStatus));
			}
			else {
				// cudaDeviceSynchronize waits for the kernel to finish, and returns
				// any errors encountered during the launch.
				cudaMemcpyAsync(&cpuSum, gpuSum, sizeof(double), cudaMemcpyDeviceToHost, myStream);
				cudaStatus = cudaStreamSynchronize(myStream);
				cudaFree(gpuSum);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Trapezoidal!\n", cudaStatus);
				}
				else {
					*out = (f(start) + f(end)) / 2.0 + cpuSum;
					*out *= h;
				}
			}
		}
#ifdef TIMING
		cudaEventRecord(endStep, myStream);
		cudaStreamSynchronize(myStream);
		cudaEventElapsedTime(&gpuTimeUsed, startStep, endStep);
		totalGpuTimeUsed += gpuTimeUsed;
		avgGPUTime += totalGpuTimeUsed;
#endif // TIMING
		cudaStreamDestroy(myStream);
	}
	return cudaStatus;
}

int main(void) {
#ifdef TIMING
	avgCPUTime = 0;
	avgGPUTime = 0;
#endif // TIMING
	int a = 1;
	int b = 2;
	int n = 100;
	double sum = 0.0;
	cudaError_t trapezoidalLaunch = trapezoidalMethod(a, b, n, &sum, 5, 256);
	if (trapezoidalLaunch == cudaSuccess) {
		printf("%lf\n", sum);
	}
	else {
		printf("There was an error runnning the operation.\n");
		printf("Error code: %d\n", trapezoidalLaunch);
	}
	return 0;
}


cudaError_t sumArray(double *arr, int size, double *out, int threadsPerBlock, bool arrayAlreadyOnGPU) {
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
		cudaStream_t myStream;
		cudaStreamCreateWithFlags(&myStream, cudaStreamNonBlocking);
		//adjust size so it is even
		int adjustedSize = size + size % 2;
		//determine number of thread blocks required
		int numThreadBlocks = numThreadBlocks = (adjustedSize + threadsPerBlock - 1) / threadsPerBlock;
		if (numThreadBlocks > 65535) {
			double part1, part2;
			sumArray(arr, size / 2, &part1, threadsPerBlock,arrayAlreadyOnGPU);
			sumArray(&arr[size / 2], size / 2, &part2, threadsPerBlock, arrayAlreadyOnGPU);
			cudaStatus = cudaGetLastError();
			if (cudaStatus == cudaSuccess) {
				*out = part1 + part2;
			}
			return cudaStatus;
		}
		double *gpuArray;
		if (arrayAlreadyOnGPU) {
			gpuArray = arr;
		}
		else {
			//create array for sum reduction on gpu
			cudaMalloc((void **)&gpuArray, adjustedSize * sizeof(double));
			cudaStatus = cudaStreamSynchronize(myStream);
			cudaMemsetAsync((void*)gpuArray, 0, adjustedSize * sizeof(double), myStream);
			cudaStatus = cudaStreamSynchronize(myStream);
			//copy input data to gpu
			cudaMemcpyAsync(gpuArray, arr, size * sizeof(double), cudaMemcpyHostToDevice, myStream);
			cudaStatus = cudaStreamSynchronize(myStream);
		}
		//keep reducing the problem size by two
		while (adjustedSize > 1) {
#ifdef TIMING
			cudaEventRecord(startStep, myStream);
			cudaStreamSynchronize(myStream);
			//cudaStreamWaitEvent(myStream, startStep, 0);
#endif // TIMING
			adjustedSize /= 2;
			if (adjustedSize % 2 != 0 && adjustedSize > 1) {
				adjustedSize++;
			}
			if (adjustedSize < threadsPerBlock) {
				threadsPerBlock = adjustedSize;
				if (threadsPerBlock % 32 != 0) {
					threadsPerBlock += 32 - threadsPerBlock % 32;
				}
			}
			numThreadBlocks = (adjustedSize + threadsPerBlock - 1) / threadsPerBlock;
			//printf("Adjusted size:%d\tThreadBlocks:%d\tThreadsPerBlock:%d\n", adjustedSize, numThreadBlocks,threadsPerBlock);
			// Launch a kernel on the GPU with one thread for each pair of elements.
			sumKernel << <numThreadBlocks, threadsPerBlock, 0, myStream >> > (gpuArray, adjustedSize);

			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "sumArray Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
				break;
			}
			else {
				// cudaDeviceSynchronize waits for the kernel to finish, and returns
				// any errors encountered during the launch.
				cudaStreamSynchronize(myStream);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
					break;
				}
				cudaMemcpyAsync(out, gpuArray, sizeof(long long), cudaMemcpyDeviceToHost, myStream);
			}
#ifdef TIMING
			cudaEventRecord(endStep, myStream);
			cudaStreamSynchronize(myStream);
			//cudaStreamWaitEvent(myStream,endStep,0);
			cudaEventElapsedTime(&gpuTimeUsed, startStep, endStep);
			totalGpuTimeUsed += gpuTimeUsed;
#endif // TIMING
		}
#ifdef TIMING
		avgGPUTime += totalGpuTimeUsed;
#endif // TIMING
		cudaStreamDestroy(myStream);
		if (!arrayAlreadyOnGPU) {
			cudaFree(gpuArray);
		}
	}
	return cudaStatus;
}