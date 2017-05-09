#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

cudaError_t sumArray(int *arr, unsigned long long size, long long *out, int threadsPerBlock);

__global__ void sumKernel(int *inArr, int *outArr)
{
	int i = threadIdx.x * 2;
	outArr[threadIdx.x] = inArr[i] + inArr[i + 1];
}

int main()
{
	const int arraySize = 100;
	int a[arraySize] = { 1, 2, 3, 4, 5 };
	long long sum = 0;
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


cudaError_t sumArray(int *arr, unsigned long long size, long long *out, int threadsPerBlock) {
	unsigned long long adjustedSize = size;
	if (adjustedSize % 32 != 0) {
		adjustedSize += 32 - (size % 32);
	}
	if (adjustedSize < threadsPerBlock) {
		threadsPerBlock = adjustedSize;
	}
	if (threadsPerBlock % 32 != 0) {
		threadsPerBlock = threadsPerBlock + 32 - (threadsPerBlock % 32);
	}
	printf("Running array sum with %d threads per block.\n", threadsPerBlock);
	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	else {
		//create array for sum reduction on gpu
		int *gpuArray, gpuTempArray;
		cudaMalloc((void **)&gpuArray, adjustedSize * sizeof(int));
		cudaMalloc((void **)&gpuTempArray, adjustedSize / 2 * sizeof(int));
		//initialize array to 0 on gpu
		cudaMemset((void*)gpuArray, 0, adjustedSize);
		cudaMemset((void*)gpuTempArray, 0, adjustedSize / 2);
		//copy input data to gpu
		cudaMemcpy(gpuArray, arr, size, cudaMemcpyHostToDevice);
		//keep reducing the problem size by two
		while (adjustedSize > 1) {
			// Launch a kernel on the GPU with one thread for each pair of elements.
			sumKernel << <1, size >> > (gpuArray, gpuTempArray);


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
				adjustedSize /= 2;
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
					break;
				}
			}
		}
	}
	return cudaStatus;
}