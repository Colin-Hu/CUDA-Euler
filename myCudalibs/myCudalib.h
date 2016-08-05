#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>

#define LIBTHREADSPERBLOCK 1024

__global__ void libcheckKernel(int *a, int size)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	int idx = i*LIBTHREADSPERBLOCK + j;
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

__global__ void replaceKernel(int *a, int size)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	int idx = i*LIBTHREADSPERBLOCK + j;
	if (idx < size)
	{
		if (a[idx] > 0)
		{
			a[idx] = 1;
		}
	}

}

__global__ void libsumKernel(int *a, int *b, int size)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	int idx = i*LIBTHREADSPERBLOCK + j;
	if ((idx-1) < size/2){
		b[idx] = a[2*idx]+a[2*idx+1];
	}
	else if (size % 2 == 1 && idx == (size/2)){
		printf("I am thread %d working on size %d moving %d\n",idx,size,a[size-1]);
		b[idx] = a[size-1];
	}
}

void test()
{
	printf("Testlib\n");
}

// Helper function for using CUDA to check for condition
cudaError_t checkWithCudalib(int *a, int size)
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
	int numBlocks = size / LIBTHREADSPERBLOCK + 1;
	libcheckKernel << < numBlocks, LIBTHREADSPERBLOCK >> >(dev_a, size);

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
cudaError_t replaceWithCudalib(int *a, int size)
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
	int numBlocks = size / LIBTHREADSPERBLOCK + 1;
	replaceKernel << < numBlocks, LIBTHREADSPERBLOCK >> >(dev_a, size);

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
int sumWithmyCuda(int *inarray, int size, int *outarray)
{
	int *dev_in = 0;
	int *dev_out = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	// Allocate GPU buffers for input array.
	cudaStatus = cudaMalloc((void**)&dev_in, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	int outsize = ceil(float(size)/2);
	printf("insize %d outsize %d\n", size, outsize);
	// Allocate GPU buffers for input array.
	cudaStatus = cudaMalloc((void**)&dev_out, outsize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_in, inarray, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	int numBlocks = size / LIBTHREADSPERBLOCK + 1;
	libsumKernel << < numBlocks, LIBTHREADSPERBLOCK >> >(dev_in, dev_out, size);

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

	free(outarray);
	outarray = (int *)malloc(sizeof(int)*outsize);

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(outarray, dev_out, outsize * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int *finalresult;
	finalresult = (int *)malloc(sizeof(int) * 1);
	if (outsize > 1)
	{
		return sumWithmyCuda(outarray, outsize, finalresult);
	}
	else
	{
		return *outarray;
	}
Error:
	cudaFree(dev_in);

}

int myCuda_sum(int *a, int size)
{
	cudaError_t cudaStatus;
	int *copy;
	int *finalsum;

	copy = (int *)malloc(sizeof(int)*size);
	copy = a;

	finalsum = (int *)malloc(sizeof(int) * 1);
	finalsum = 0;
	int finalval = 0;
	cudaStatus = replaceWithCudalib(copy, size);
	finalval = sumWithmyCuda(copy, size, finalsum);
	printf("Finalval %d\n", finalval);

	//int finalsum = copy[0];
	//free(copy);

	//return finalsum;
	return 0;
}