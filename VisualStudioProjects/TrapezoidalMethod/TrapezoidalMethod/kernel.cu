#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <cstdlib>
// Kernel function to calculate trapizoidal sum
__host__
__device__
double f(double x) {
	return x*x;
}

__global__
void trap(int a, int n, double h, double* sum) {
	double x_i;
	for (int i = 1; i < n; i++) {
		x_i = a + i*h;
		*sum += f(x_i);
	}
}

cudaError_t trapezoidalMethod(double start, double end, int subdivisions, double *out) {
	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	else {
		// Launch a kernel on the GPU with one thread for each element.
		double  h = (end - start) / (double)subdivisions;
		double sum = 0.0;
		sum += (f(start) + f(end)) / 2.0;
		trap << <1, 1 >> > (start, subdivisions, h, &sum);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Trapezoidal launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		else {
			// cudaDeviceSynchronize waits for the kernel to finish, and returns
			// any errors encountered during the launch.
			cudaStatus = cudaDeviceSynchronize();
			sum *= h;
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Trapezoidal!\n", cudaStatus);
			}
		}
	}
	return cudaStatus;
}

int main(void) {

	int a = 1;
	int b = 2;
	int n = 100;
	double sum = 0.0;
	cudaError_t trapezoidalLaunch = trapezoidalMethod(a, b, n, &sum);
	if (trapezoidalLaunch == cudaSuccess) {
		printf("%lf\n", sum);
	}
	else {
		printf("There was an error runnning the operation.\n");
		printf("Error code: %d\n", trapezoidalLaunch);
	}
	return 0;
}
