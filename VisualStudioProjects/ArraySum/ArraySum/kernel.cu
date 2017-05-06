#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

cudaError_t sumArray(int *arr, long long *out, unsigned long long size, int threadsPerBlock, int numBlocks);

__global__ void sumKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

int main()
{
	const int arraySize = 5;
	int a[arraySize] = { 1, 2, 3, 4, 5 };
	long long sum = 0;
	// Add vectors in parallel.
	cudaError_t cudaStatus = sumArray(a, &sum, arraySize);
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


cudaError_t sumArray(int *arr, long long *out, unsigned long long size, int threadsPerBlock, int numBlocks) {
	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	else {
		// Launch a kernel on the GPU with one thread for each element.
		sumKernel<<<1, size>>>(dev_c, dev_a, dev_b);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		else {
			// cudaDeviceSynchronize waits for the kernel to finish, and returns
			// any errors encountered during the launch.
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			}
		}
	}
	return cudaStatus;
}