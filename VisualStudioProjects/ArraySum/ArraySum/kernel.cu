#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

cudaError_t sumArray(int *arr, int size, int *out, int threadsPerBlock);

__global__ void sumKernel(int *arr, int size)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < size) {
		arr[i] += arr[i+size];
	}
}

int main()
{
	const int arraySize = 20;
	int a[arraySize];// = { 1, 2, 3, 4, 5,15 };
	int sum = 0;
	for (int i = 0; i < arraySize; i++) {
		a[i] = i;
		sum += a[i];
	}
	printf("Actual sum: %d\n", sum);
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


cudaError_t sumArray(int *arr, int size, int *out, int threadsPerBlock) {

	printf("Running array sum with %d threads per block.\n", threadsPerBlock);
	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	else {
		int adjustedSize = size;
		if (adjustedSize % 32 != 0) {
			adjustedSize += 32 - (size % 32);
		}
		//determine number of thread blocks required
		int numThreadBlocks;
		//create array for sum reduction on gpu
		int *gpuArray;
		cudaMalloc((void **)&gpuArray, adjustedSize * sizeof(int));
		//initialize array to 0 on gpu
		cudaMemset((void*)gpuArray, 0, adjustedSize);
		//copy input data to gpu
		cudaMemcpy(gpuArray, arr, size, cudaMemcpyHostToDevice);
		//keep reducing the problem size by two
		while (adjustedSize > 1) {
			adjustedSize /= 2;
			numThreadBlocks = (adjustedSize + threadsPerBlock - 1) / threadsPerBlock;
			printf("Adjusted size:%d\tThreadBlocks:%d\n", adjustedSize, numThreadBlocks);
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
				cudaMemcpy(out, gpuArray, sizeof(int), cudaMemcpyDeviceToHost);
				printf("PartialResult:%d\n", *out);
			}
		}
		cudaFree(gpuArray);
	}
	return cudaStatus;
}