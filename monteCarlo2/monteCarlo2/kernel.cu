
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GPU_Sum.h"
#include <curand_kernel.h>
#include <curand.h>
#include <stdlib.h>
#include <stdio.h>

#define TIMING

/*__global__ void monteCarlo(unsigned  int n, unsigned int* inCirc_d, curandState_t* states, unsigned int seed)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {

		double x, y;

		curand_init(seed, index, 0, &states[index]);
		//printf("%d\n", index);
		x = curand_uniform_double(&states[index]);
		y = curand_uniform_double(&states[index]);

		//printf("%.4f %.4f\n", x, y);

		if ((x*x + y*y) <= 1.0) inCirc_d[index] = 1;
		else inCirc_d = 0;

	}
}*/

__global__ void monteCarlo(unsigned  int n, unsigned int* inCirc_d, curandState_t* states, unsigned int seed)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x*gridDim.x;
	

	for(unsigned int i = index; i < n; i += stride) {

		double x, y;

		curand_init(seed, i, 0, &states[i]);
		//printf("%d\n", index);
		x = curand_uniform_double(&states[i]);
		y = curand_uniform_double(&states[i]);

		//printf("%.4f %.4f\n", x, y);

		if ((x*x + y*y) <= 1.0) inCirc_d[index] = 1;
		else inCirc_d = 0;

	}
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
	
	int N = 1000000;
	int blockSize = 512;
	int numBlocks = (N + blockSize - 1) / blockSize;
	bool onGpu = false;

	curandState_t* states;
	cudaMalloc((void**)&states, N * sizeof(curandState_t));

	unsigned int total = 0;
	unsigned int* inCirc_d;
	unsigned int* inCirc = (unsigned int*)malloc(N * sizeof(unsigned int));


	cudaMalloc((void**)&inCirc_d, N * sizeof(unsigned int));
	cudaMemset(inCirc_d, 0, N * sizeof(unsigned int));
	//CUDAErrorCheck();
	double cpu_estimate;

	monteCarlo << <numBlocks, blockSize >> > (N, inCirc_d, states, time(0));

	cudaDeviceSynchronize();
	//CUDAErrorCheck();

	cudaMemcpy(inCirc, inCirc_d, N * sizeof(int), cudaMemcpyDeviceToHost);
	//CUDAErrorCheck();
	cudaDeviceSynchronize();

	//sumArray(inCirc, N, &total, numBlocks, blockSize, onGpu);
	
	for (unsigned int i = 0; i < N; i++) {
		total += inCirc[i];
		//printf("i = %d and count = %d\n",i,inCirc[i]);
	}


	double pi_estimate = 4 * total / (double)N;

	printf("Total in Circle = %d\nEstimate of Pi = %.4f\n", total, pi_estimate);

	cudaFree(states);
	cudaFree(inCirc_d);


	return 0;
}

