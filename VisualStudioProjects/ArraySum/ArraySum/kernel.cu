#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NUM_THREADS 8
#define TIMING
#define MIN_SIZE 100000000
#define SIZE_INCREMENT 250000
#define MAX_SIZE 100000000
#define SAMPLE_SIZE 1

#ifdef TIMING
double avgCPUTime, avgGPUTime;
double cpuStartTime, cpuEndTime;
#endif // TIMING

cudaError_t sumArray(long long *arr, int size, long long *out, int threadsPerBlock);

__global__ void sumKernel(long long *arr, int size)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < size) {
		arr[i] += arr[i + size];
		arr[i + size] = 0;
	}
}

int main()
{
	long long *a = (long long*)malloc(MAX_SIZE * sizeof(long long));
	for (int i = 0; i < MAX_SIZE; i++) {
		a[i] = rand();
	}
	int timesCorrect = 0, timesWrong = 0;
	printf("Size\tAvgCPUTime\tAvgGPUTime\tSamples:%d\n", SAMPLE_SIZE);
	for (int arraySize = MIN_SIZE; arraySize <= MAX_SIZE; arraySize += SIZE_INCREMENT) {
#ifdef TIMING
		avgCPUTime = 0;
		avgGPUTime = 0;
#endif // TIMING
#pragma omp parallel for num_threads(NUM_THREADS) \
reduction(+:avgGPUTime,avgCPUTime,timesCorrect,timesWrong) \
private(cpuStartTime,cpuEndTime)
		for (int sample = 0; sample < SAMPLE_SIZE; sample++) {
			long long sum = 0, cudaSum;
#ifdef TIMING
			cpuStartTime = omp_get_wtime();
#endif // TIMING
			for (int i = 0; i < arraySize; i++) {
				sum += a[i];
			}
#ifdef TIMING
			cpuEndTime = omp_get_wtime();
			double timeUsed = 1000 * (cpuEndTime - cpuStartTime);
			avgCPUTime += timeUsed;
#endif // TIMING
			// Add vectors in parallel.
			cudaError_t cudaStatus = sumArray(a, arraySize, &cudaSum, 256);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "CUDA Sum Array failed!");
				return 1;
			}

			if (cudaSum == sum) {
				timesCorrect++;
			}
			else {
				timesWrong++;
			}

			// cudaDeviceReset must be called before exiting in order for profiling and
			// tracing tools such as Nsight and Visual Profiler to show complete traces.
			cudaStatus = cudaDeviceReset();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceReset failed!");
				return 1;
			}
		}
#ifdef TIMING
		printf("%d\t%lf\t%lf\n", arraySize, avgCPUTime / SAMPLE_SIZE, avgGPUTime / SAMPLE_SIZE);
#endif // TIMING
	}
	printf("GPU Implementation was correct %d times and incorrect %d times.\n", timesCorrect, timesWrong);
	return 0;
}


cudaError_t sumArray(long long *arr, int size, long long *out, int threadsPerBlock) {
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
		//adjust size so it is even
		int adjustedSize = size + size % 2;
		//determine number of thread blocks required
		int numThreadBlocks = numThreadBlocks = (adjustedSize + threadsPerBlock - 1) / threadsPerBlock;
		if (numThreadBlocks > 65535) {
			long long part1, part2;
			sumArray(arr, size / 2, &part1, threadsPerBlock);
			sumArray(&arr[size / 2], size / 2, &part2, threadsPerBlock);
			cudaStatus = cudaGetLastError();
			if (cudaStatus == cudaSuccess) {
				*out = part1 + part2;
			}
			return cudaStatus;
		}
		//create array for sum reduction on gpu
		long long *gpuArray;
		cudaMalloc((void **)&gpuArray, adjustedSize * sizeof(long long));
		cudaStatus = cudaStreamSynchronize(myStream);
		cudaMemsetAsync((void*)gpuArray, 0, adjustedSize * sizeof(long long),myStream);
		cudaStatus = cudaStreamSynchronize(myStream);
		//copy input data to gpu
		cudaMemcpyAsync(gpuArray, arr, size * sizeof(long long), cudaMemcpyHostToDevice,myStream);
		cudaStatus = cudaStreamSynchronize(myStream);
		//keep reducing the problem size by two
		while (adjustedSize > 1) {
#ifdef TIMING
			cudaEventRecord(startStep,myStream);
			cudaStreamSynchronize(myStream);
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
			sumKernel << <numThreadBlocks, threadsPerBlock,0, myStream >> > (gpuArray, adjustedSize);

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
				cudaMemcpyAsync(out, gpuArray, sizeof(long long), cudaMemcpyDeviceToHost,myStream);
			}
#ifdef TIMING
			cudaEventRecord(endStep,myStream);
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
		cudaFree(gpuArray);
	}
	return cudaStatus;
}