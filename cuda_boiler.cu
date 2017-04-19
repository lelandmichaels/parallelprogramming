//Taken from https://devblogs.nvidia.com/parallelforall/even-easier-introduction-cuda/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>

// CUDA Kernal function to add the elements of two arrays on the GPU
__global__
void add(int n, double *x, double *y)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i+=stride)
		y[i] = x[i] + y[i];
}

int main(void)
{
	cudaEvent_t start, stop;
	int N = 1 << 24; // 1M elements
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	double *x; //= new float[N];
	double *y; //= new float[N];
	cudaEventRecord(start);
	cudaMallocManaged(&x, N * sizeof(double));
	cudaMallocManaged(&y, N * sizeof(double));

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}
	int threadBlockSize = 128;
	int numThreadBlocks = (N+threadBlockSize-1)/threadBlockSize;
	// Run kernel on 1M elements on the CPU
	add<<<numThreadBlocks, threadBlockSize >>>(N, x, y);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// Check for errors (all values should be 3.0f)
	double maxError = 0.0f;
	for (int i = 0; i < N; i++)
	maxError = fmax(maxError, fabs(y[i] - 3.0f));
	std::cout << "max error: " << maxError << "\nTime used (ms): " << elapsedTime << std::endl;

	// Free memory
	//delete[] x;
	//delete[] y;
	cudaFree(x);
	cudaFree(y);

	return 0;
}
