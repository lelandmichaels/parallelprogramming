#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

cudaError_t sumArray(long long *arr, int size, long long *out, int threadsPerBlock);

__global__ void sumKernel(long long *arr, int size)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < size) {
		//printf("Adding arr[%d]=%d and arr[%d]=%d\n", i, arr[i], i + size, arr[i + size]);
		arr[i] += arr[i + size];
		arr[i + size] = 0;
	}
}

int main()
{
	const int arraySize = 3000;
	long long a[arraySize];// = { 1, 2, 3, 4, 5,15 };
	long long sum = 0;
	for (int i = 0; i < arraySize; i++) {
		a[i] = rand();
		sum += a[i];
		//printf("i:%d\tsum so far:%d\n", i, sum);
	}
	printf("Actual sum: %lld\n", sum);
	// Add vectors in parallel.
	cudaError_t cudaStatus = sumArray(a, arraySize, &sum, 128);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA Sum Array failed!");
		return 1;
	}

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
			adjustedSize /= 2;
			if (adjustedSize % 2 != 0 && adjustedSize>1) {
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
		}
		printf("PartialResult:%lld\n", *out);
		cudaFree(gpuArray);
	}
	return cudaStatus;
}