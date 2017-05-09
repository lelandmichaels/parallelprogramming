#include "GPU_Sum.h"

#define NUM_THREADS 8
#define TIMING
#define MIN_SIZE 12500
#define SIZE_INCREMENT 12500
#define MAX_SIZE 1000000
#define SAMPLE_SIZE 50

#ifdef TIMING
double avgCPUTime, avgGPUTime;
double cpuStartTime, cpuEndTime;
#endif // TIMING

template<typename Type>
cudaError_t sumArray(Type *arr, int size, Type *out, int blockCount, int threadsPerBlock, bool arrayOnGPU);

template<typename Type>
__global__ void sumKernel(Type *arr, int size)
{
	int stride = blockDim.x * gridDim.x;
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	long long temp = arr[id];
	for (int i = id + stride; i < size; i += stride) {
		temp += arr[i];
	}
	arr[id] = temp;
}

template<typename Type>
__global__ void binaryReduction(Type *arr, int size)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	if (id < size) {
		arr[id] += arr[id + size];
	}
}

int main()
{
	long long *a = (long long*)malloc(MAX_SIZE * sizeof(long long));
	for (int i = 0; i < MAX_SIZE; i++) {
		a[i] = i;// rand();
	}
	int timesCorrect = 0, timesWrong = 0;
	printf("Size\tAvgCPUTime\tAvgGPUTime\tSamples:%d\n", SAMPLE_SIZE);
	for (int arraySize = MIN_SIZE; arraySize <= MAX_SIZE; arraySize += SIZE_INCREMENT) {
#ifdef TIMING
		avgCPUTime = 0;
		avgGPUTime = 0;
#endif // TIMING
		long long sum = 0, cudaSum;
#pragma omp parallel for num_threads(NUM_THREADS) \
reduction(+:avgGPUTime,avgCPUTime,timesCorrect,timesWrong) \
private(cpuStartTime,cpuEndTime)
		for (int sample = 0; sample < SAMPLE_SIZE; sample++) {
			sum = 0;
#ifdef TIMING
			cpuStartTime = omp_get_wtime();
#endif // TIMING
			for (int i = 0; i < arraySize; i++) {
				sum += a[i];
			}
#ifdef TIMING
			cpuEndTime = omp_get_wtime();
			double timeUsed = 1000 * (cpuEndTime - cpuStartTime);
			avgCPUTime += timeUsed;
#endif // TIMING
			// Add vectors in parallel.
			cudaError_t cudaStatus = sumArray(a, arraySize, &cudaSum, 1, 256, false);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "CUDA Sum Array failed!");
				return 1;
			}

			if (cudaSum == sum) {
				timesCorrect++;
			}
			else {
				timesWrong++;
			}

			// cudaDeviceReset must be called before exiting in order for profiling and
			// tracing tools such as Nsight and Visual Profiler to show complete traces.
			cudaStatus = cudaDeviceReset();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceReset failed!");
				return 1;
			}
		}
#ifdef TIMING
		printf("%d\t%lf\t%lf\n", arraySize, avgCPUTime / SAMPLE_SIZE, avgGPUTime / SAMPLE_SIZE);
#endif // TIMING
	}
	printf("GPU Implementation was correct %d times and incorrect %d times.\n", timesCorrect, timesWrong);
	return 0;
}


template<typename Type>
cudaError_t sumArray(Type *arr, int size, Type *out, int blockCount, int blockSize, bool arrayOnGPU) {
#ifdef TIMING
	double totalGpuTimeUsed = 0;
	float gpuTimeUsed;
	cudaEvent_t startStep, endStep;
	cudaEventCreateWithFlags(&startStep, cudaEventDefault);
	cudaEventCreateWithFlags(&endStep, cudaEventDefault);
#endif // TIMING
	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	else {
		cudaStream_t myStream;
		cudaStreamCreateWithFlags(&myStream, cudaStreamNonBlocking);
		//adjust size so it is even
		int adjustedSize = size + size % 2;
		//determine number of thread blocks required
		//create array for sum reduction on gpu
		Type *gpuArray;
		if (arrayOnGPU) {
			gpuArray = arr;
		}
		else {
			cudaMalloc((void **)&gpuArray, adjustedSize * sizeof(Type));
			cudaStatus = cudaStreamSynchronize(myStream);
			cudaMemsetAsync((void*)gpuArray, 0, adjustedSize * sizeof(Type), myStream);
			cudaStatus = cudaStreamSynchronize(myStream);
			//copy input data to gpu
			cudaMemcpyAsync(gpuArray, arr, size * sizeof(Type), cudaMemcpyHostToDevice, myStream);
			cudaStatus = cudaStreamSynchronize(myStream);
		}
		sumKernel << <blockCount, blockSize, 0, myStream >> > (gpuArray, adjustedSize);
		adjustedSize = blockCount*blockSize;
		//keep reducing the problem size by two
		while (adjustedSize > 1) {
			cudaStatus = cudaStreamSynchronize(myStream);
#ifdef TIMING
			cudaEventRecord(startStep, myStream);
#endif // TIMING
			adjustedSize /= 2;
			if (adjustedSize % 2 != 0 && adjustedSize > 1) {
				adjustedSize++;
			}
			// Launch a kernel on the GPU with one thread for each pair of elements.
			binaryReduction << <blockCount, blockSize, 0, myStream >> > (gpuArray, adjustedSize);

			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "sumArray Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
				break;
			}
#ifdef TIMING
			cudaEventRecord(endStep, myStream);
			cudaStatus = cudaStreamSynchronize(myStream);
			cudaEventElapsedTime(&gpuTimeUsed, startStep, endStep);
			totalGpuTimeUsed += gpuTimeUsed;
#endif // TIMING
		}
#ifdef TIMING
		avgGPUTime += totalGpuTimeUsed;
#endif // TIMING
		if (cudaStatus == cudaSuccess) {
			cudaMemcpyAsync(out, gpuArray, sizeof(Type), cudaMemcpyDeviceToHost, myStream);
			cudaStatus = cudaStreamSynchronize(myStream);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			}
		}
		cudaStreamDestroy(myStream);
		if (arrayOnGPU) {
			cudaFree(gpuArray);
		}
	}
	return cudaStatus;
}