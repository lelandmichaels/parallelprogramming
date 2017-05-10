#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <math.h>
#include <cstdlib>

#define TIMING
#define MIN_SIZE 250000
#define SIZE_INCREMENT 250000
#define MAX_SIZE 10000000
#define SAMPLE_SIZE 1

#ifdef TIMING
double avgCPUTime, avgGPUTime;
double cpuStartTime, cpuEndTime;
#endif // TIMING

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void writeheader(int N, int end) {
	FILE *fp;
	fp = fopen("outfile.pgm", "w");
	if (fp == NULL) {
		printf("sorry can't open outfile.pgm. Terminating.\n");
		exit(1);
	}
	else {
		// print a table header
		fprintf(fp, "%s\n%d %d\n%s\n", "P2", N, end, "255");
		fclose(fp);
	}
}

void writerow(int N, double rawdata[]) {
	FILE *fp;
	fp = fopen("outfile.pgm", "a");
	if (fp == NULL) {
		printf("sorry can't open outfile.pgm. Terminating.\n");
		exit(1);
	}
	else {
		for (int i = 0; i < N; i++) {
			int val = rawdata[i] * 127 + 127;
			fprintf(fp, "%d ", val);
		}
		fprintf(fp, "\n");
		fclose(fp);
	}
}



__host__
__device__
double f(int x, double *arr1, double *arr0) {
	return (double)(.01*(arr1[x - 1] - (2 * arr1[x]) + arr1[x + 1]) + 2 * arr1[x] - arr0[x]);
}


__global__
void wave(int n, double *arr0, double *arr1, double *arr2) {
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = gridDim.x*blockDim.x;
	for (int i = id; i < n; i += stride) {
	  if(i==n-1 || i==0){
	    arr2[i]=0;
	  }else{
	    arr2[i] = f(i, arr1, arr0);
	  }
	}
}

__global__
void initForWave(double startX, double endX, int n, double* arr0, double* arr1, double* arr2) {
	int stride = gridDim.x*blockDim.x;
	double x;
	for (int i = threadIdx.x + blockDim.x*blockIdx.x; i < n; i += stride) {
		x = startX + (double)i*1.0 / (n - 1);
		if (i == 0 || i == n - 1) {
			arr0[i] = 0;
			arr1[i] = 0;
		}
		else {
			arr0[i] = sin(M_PI*x);
			arr1[i] = sin(M_PI*x);
		}
	}
}



int main(void) {
	int output = 1;
	cudaDeviceReset();
	cudaEvent_t start, stop;
	int N = 10000; // 1M elements
<<<<<<< HEAD
	int steps = 1000;
=======
	int steps = 10;
>>>>>>> origin/master
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	double *arr0; //= new float[N];
	double *arr1; //= new float[N];
	double *arr2; //= new float[N];
	double *temp, *localArr0;
	cudaEventRecord(start);
	localArr0 = (double*)malloc(N * sizeof(double));
	cudaMalloc(&arr0, N * sizeof(double));
	cudaMalloc(&arr1, N * sizeof(double));
	cudaMalloc(&arr2, N * sizeof(double));
	if (output == 1) {
		writeheader(N, steps);
	}

	int threadBlockSize = 128;
	int numThreadBlocks = (N + threadBlockSize - 1) / threadBlockSize;
	cudaDeviceSynchronize();
	initForWave << <numThreadBlocks, threadBlockSize >> >(0.0, 1.0, N, arr0, arr1, arr2);

	if (output == 1) {
			cudaMemcpy((void*)localArr0, (void*)arr0, N * sizeof(double), cudaMemcpyDeviceToHost);
			writerow(N, localArr0);
		}

	// Run kernel on 1M elements on the CPU
	for (int i = 0; i < steps; i++) {
		wave<<<numThreadBlocks, threadBlockSize>>>(N, arr0, arr1, arr2);
		cudaDeviceSynchronize();
		temp = arr0;
		arr0 = arr1;
		arr1 = arr2;
		arr2 = temp;

		if (output == 1) {
			cudaMemcpy((void*)localArr0, (void*)arr0, N * sizeof(double), cudaMemcpyDeviceToHost);
			writerow(N, localArr0);
		}
	}
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	std::cout << "\nTime used (ms): " << elapsedTime << std::endl;
	
	free(localArr0);
	cudaFree(arr0);
	cudaFree(arr1);
	cudaFree(arr2);

	return 0;
}
