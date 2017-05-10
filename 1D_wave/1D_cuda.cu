#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <omp.h>
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
		for (int i=0; i<N; i++) {
			int val = rawdata[i]*127+127;
			fprintf(fp,"%d ",val);
		}
		fprintf(fp,"\n");
		fclose(fp);
	}
}



__host__
__device__
double f(int x, double* arr1, double* arr0) {
  return double ans = (.01*(arr1[x-1] - (2*arr1[x]) + arr1[x+1]) + 2* arr1[x] - arr0[x]);
}

__global__
void wave(int n, double *arr0, double* arr1, double* arr2) {
  for(int i = 0; i < n; i++){
    arr2[i] = f(i, arr1, arr0);
  }
}

int main(void) {


  	cudaEvent_t start, stop;
	int N = 100; // 1M elements
	int steps = 10;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	double *arr0; //= new float[N];
	double *arr1; //= new float[N];
	double *arr2; //= new float[N];
	double *temp;
	cudaEventRecord(start);
	cudaMallocManaged(&arr0, N * sizeof(double));
	cudaMallocManaged(&arr1, N * sizeof(double));
	cudaMallocManaged(&arr2, N * sizeof(double));

	arr1[0] = 0;
	arr1[N-1] = 0;
	arr0[0] = 0;
	arr0[N-1] = 0;

	// initialize arr1 and arr2 arrays
	for (int i = 1; i < N-2; i++) {
	  arr1[i] = sin(M_PI*i);
	  arr0[i] = sin(M_PI*i);
	}

	if(output ==1){
	  writeheader(N, steps);
	}
	
	int threadBlockSize = 128;
	int numThreadBlocks = (N+threadBlockSize-1)/threadBlockSize;
	
	
	// Run kernel on 1M elements on the CPU
	for(int i = 0; i < steps; i++){
	  wave<<<numThreadBlocks, threadBlockSize >>>(N, arr0, arr1, arr2);
	  
	  temp = arr0;
	  arr0 = arr1;
	  arr1 = arr2;
	  arr2 = temp;

	  if(output == 1){
	    writerow(N, arr0);
	  }
	}
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << "\nTime used (ms): " << elapsedTime << std::endl;
	
	// Free memory
	//delete[] x;
	//delete[] y;
	cudaFree(arr0);
	cudaFree(arr1);
	cudaFree(arr2);

	return 0;
}
