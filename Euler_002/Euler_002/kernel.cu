
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

cudaError_t findMultipleWithCuda(int *a, unsigned int size);
int findarraySize(int fibmaxval);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void findMultipleKernel(int *a)
{
	int i = blockIdx.x;
	if (a[i] % 2 != 0)
	{
		a[i] = 0;
	}
}

int main()
{
	int arraySize;
	int fibmaxval = 4000000;

	arraySize = findarraySize(fibmaxval);
	const int fibArraySize = 32;

	int *a;
	int b[fibArraySize];
	a = (int *) malloc(sizeof(int)*arraySize);
	// Fill array with ints
	a[0] = 1; 
	b[0] = 1;
	a[1] = 2; 
	b[1] = 2;
	for (int i = 2; i < arraySize; i++)
	{
		a[i] = a[i-1] + a[i-2];
		b[i] = b[i-1] + b[i-2];
		printf("Fill b %d %d\n", i, b[i]); fflush(stdout);
		printf("Fill a %d %d\n", i, a[i]); fflush(stdout);
	}

	/*const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };*/

	//// add vectors in parallel.
	//cudaerror_t cudastatus = addwithcuda(c, a, b, arraysize);
	//if (cudastatus != cudasuccess) {
	//    fprintf(stderr, "addwithcuda failed!");
	//    return 1;
	//}

	// Add vectors in parallel.
	//printf("Pre {%d,%d,%d,%d,%d}\n",
	//	b[0], b[1], b[2], b[3], b[22]); fflush(stdout);

	cudaError_t cudaStatus = findMultipleWithCuda(a, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("Post {%d,%d,%d,%d,%d}\n",
		a[0], a[1], a[2], a[3], a[4]); fflush(stdout);

	int sum = 0;
//	for (auto& num : a)
	for (int i = 0; i < arraySize; i++)
	{
		sum += a[i];
	}

	printf("Sum %d", sum);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

int findarraySize(int fibmaxval)
{
	int indx = 2;
	int prev1 = 2;
	int prev2 = 1;
	int currentval=0;

	while (currentval < fibmaxval)
	{
		indx += 1;
		currentval = prev1 + prev2;
		prev2 = prev1;
		prev1 = currentval;
		printf("%d,%d\n", indx, currentval);
	}
	return indx-1;
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t findMultipleWithCuda(int *a, unsigned int size)
{
	int *dev_a = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	findMultipleKernel << <size, 1 >> >(dev_a);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_a);

	return cudaStatus;
}
