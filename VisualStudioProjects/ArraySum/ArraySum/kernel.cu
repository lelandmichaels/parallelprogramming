#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define TIMING

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
	int arraySize = 1000000;
	long long *a = (long long*)malloc(arraySize * sizeof(long long));
	long long sum = 0;
#ifdef TIMING
	double startTime = omp_get_wtime(), endTime;
#endif // TIMING
	for (int i = 0; i < arraySize; i++) {
		a[i] = rand();
		sum += a[i];
	}
#ifdef TIMING
	endTime = omp_get_wtime();
	double timeUsed = endTime - startTime;
#endif // TIMING
	printf("Actual sum: %lld.\n", sum);
#ifdef TIMING
	printf("CPU Calculated in %lf ms.\n", timeUsed*1000);
#endif // TIMING
	// Add vectors in parallel.
	cudaError_t cudaStatus = sumArray(a, arraySize, &sum, 256);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA Sum Array failed!");
		return 1;
	}
	printf("Cuda sum: %lld.\n", sum);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}


cudaError_t sumArray(long long *arr, int size, long long *out, int threadsPerBlock) {
#ifdef TIMING
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	double totalGpuTimeUsed = 0;
	float gpuTimeUsed;
#endif // TIMING
	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	else {
		//determine number of thread blocks required
		int numThreadBlocks;
		//create array for sum reduction on gpu
		long long *gpuArray;
		//adjust size so it is even
		int adjustedSize = size + size % 2;
		cudaMalloc((void **)&gpuArray, adjustedSize * sizeof(long long));
		cudaStatus = cudaDeviceSynchronize();
		cudaMemset((void*)gpuArray, 0, adjustedSize * sizeof(long long));
		cudaStatus = cudaDeviceSynchronize();
		//copy input data to gpu
		cudaMemcpy(gpuArray, arr, size * sizeof(long long), cudaMemcpyHostToDevice);
		cudaStatus = cudaDeviceSynchronize();
		//keep reducing the problem size by two
		while (adjustedSize > 1) {
#ifdef TIMING
			cudaEventRecord(start);
			cudaEventSynchronize(start);
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
			sumKernel << <numThreadBlocks, threadsPerBlock >> > (gpuArray, adjustedSize);

			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "sumArray Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
				break;
			}
			else {
				// cudaDeviceSynchronize waits for the kernel to finish, and returns
				// any errors encountered during the launch.
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
					break;
				}
				cudaMemcpy(out, gpuArray, sizeof(long long), cudaMemcpyDeviceToHost);
			}
#ifdef TIMING
			cudaEventRecord(end);
			cudaEventSynchronize(end);
			cudaEventElapsedTime(&gpuTimeUsed, start, end);
			totalGpuTimeUsed += gpuTimeUsed;
#endif // TIMING
		}
#ifdef TIMING
		printf("GPU Calculated in %lf ms.\n", totalGpuTimeUsed);
#endif // TIMING

		cudaFree(gpuArray);
	}
	return cudaStatus;
}