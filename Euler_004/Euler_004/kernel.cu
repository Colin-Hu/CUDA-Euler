
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define THREADSPERBLOCK 1024

cudaError_t findMultipleWithCuda(int *a, unsigned int size);
cudaError_t fillWithCuda(int *a, int size, int numvalues);
cudaError_t checkWithCuda(int *a, int size);
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

__global__ void fillKernel(int *a)
{
	int i = blockIdx.x;
	int j = blockIdx.y;
	int idx = j * 900 + i;

	a[idx] = (i+100)*(j+100);
}

__global__ void checkKernel(int *a, int size)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	int idx = i*THREADSPERBLOCK + j;
	if (idx < size)
	{
		int myval = a[idx];

		int digit0 = myval % 10;
		int digit1 = (myval / 10) % 10;
		int digit2 = (myval / 100) % 10;
		int digit3 = (myval / 1000) % 10;
		int digit4 = (myval / 10000) % 10;
		int digit5 = (myval / 100000) % 10;

		if (digit5 > 0)
		{
			if (digit0 == digit5 && digit1 == digit4 && digit2 == digit3)
			{
			}
			else
			{
				a[idx] = 0;
			}
		}
		else if (digit4 > 0)
		{
			if (digit0 == digit4 && digit1 == digit3)
			{
			}
			else
			{
				a[idx] = 0;
			}
		}
		else
		{
			printf("Warning: Thread %d Value %d Not 5 or 6 digit numbet", idx, myval);
		}
	}
}

int main()
{
	int arraySize;
	int fibmaxval = 4000000;

	int *products; //Products of 2 three digit numbers
	const int numvalues = 900;
	const int numproducts = numvalues * numvalues; //100 - 999
	products = (int *)malloc(sizeof(int)*numproducts); 
	
	cudaError_t cudaStatus = fillWithCuda(products, numproducts, numvalues);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	printf("First couple %d %d\n",products[0],products[1]);

	cudaStatus = checkWithCuda(products, numproducts);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	printf("First couple %d %d\n", products[0], products[1]);

	int maxpalindrome = 0;

	for(int i = 0; i < numproducts; i++)
	{
		if (maxpalindrome < products[i])
		{
			maxpalindrome = products[i];
			printf("Max palindrome %d\n", maxpalindrome);
		}
	}


	//cudaStatus = findMultipleWithCuda(a, arraySize);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "addWithCuda failed!");
	//	return 1;
	//}

	//printf("Post {%d,%d,%d,%d,%d}\n",
	//	a[0], a[1], a[2], a[3], a[4]); fflush(stdout);

	//int sum = 0;
	////	for (auto& num : a)
	//for (int i = 0; i < arraySize; i++)
	//{
	//	sum += a[i];
	//}

	//printf("Sum %d", sum);

	//// cudaDeviceReset must be called before exiting in order for profiling and
	//// tracing tools such as Nsight and Visual Profiler to show complete traces.
	//cudaStatus = cudaDeviceReset();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaDeviceReset failed!");
	//	return 1;
	//}

	return 0;
}

int findarraySize(int fibmaxval)
{
	int indx = 2;
	int prev1 = 2;
	int prev2 = 1;
	int currentval = 0;

	while (currentval < fibmaxval)
	{
		indx += 1;
		currentval = prev1 + prev2;
		prev2 = prev1;
		prev1 = currentval;
		printf("%d,%d\n", indx, currentval);
	}
	return indx - 1;
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

// Helper function for computing array
cudaError_t fillWithCuda(int *a, int size, int numvalues)
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

	const dim3 blockSize(numvalues, numvalues, 1);
	// Launch a kernel on the GPU with one thread for each element.
	fillKernel << <blockSize, 1 >> >(dev_a);

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

// Helper function for using CUDA to check for condition
cudaError_t checkWithCuda(int *a, int size)
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
	printf("Block Size %d\n", size); fflush(stdout);
	int numBlocks = size / THREADSPERBLOCK + 1;
	checkKernel << < numBlocks , THREADSPERBLOCK >> >(dev_a, size);

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