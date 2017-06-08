#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <windows.h>

#define LIBTHREADSPERBLOCK 1024
#define TESTTHREADSPERBLOCK 1024

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
		b[idx] = a[size-1];
	}
}

__global__ void blocksortKernel(int *a, int size)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	int idx = i*TESTTHREADSPERBLOCK + j;
	__shared__ int swapped;
	__shared__ int swapcount;
	int temp;
	// Even swap
	swapped = 1;
	//swapcount = 0;
	while (swapped == 1){
		swapped = 0;
		temp = a[2 * idx + 1];
		__syncthreads();
		if (idx < size / 2){
			//printf("sortKertel: Even: %d checking %d %d values %d %d\n", idx, 2 * idx, 2 * idx + 1, a[2 * idx], a[2 * idx + 1]);
			if (a[2 * idx] > a[2 * idx + 1]){
				a[2 * idx + 1] = a[2 * idx];
				a[2 * idx] = temp;
				swapped = 1;
				//swapcount += 1;
				//printf("sertKernel: moving %d values %d %d\n", idx, a[2 * idx], a[2 * idx + 1]);
			}
		}
		__syncthreads();
		temp = a[2 * idx + 2];
		__syncthreads();
		if (idx < size / 2){
			// Odd swap
			//printf("sortKertel: Odd: %d checking %d %d values %d %d\n", idx, 2 * idx + 1, 2 * idx + 2, a[2 * idx + 1], a[2 * idx + 2]);
			if (a[2 * idx + 1] > a[2 * idx + 2]){
				a[2 * idx + 2] = a[2 * idx + 1];
				a[2 * idx + 1] = temp;
				swapped = 1;
				//swapcount += 1;
				//printf("sertKernel: moving %d values %d %d\n", idx, a[2 * idx + 1], a[2 * idx + 2]);
			}
		}
		__syncthreads();
	}
	//if (j == 0){ printf("Block %d swaps: %d \n", i, swapcount); }
}

__global__ void reverseMergeSortKernel(int *a, int size, int numblocks, int blocksize)
{
	int idx = blockIdx.x;
	int myblock1start = blocksize*(idx*2);
	int myblock2start = blocksize*(idx*2+1);
	int myblock1end = blocksize*(idx*2+1)-1;
	int myblock2end = blocksize*(idx*2+2)-1;
	if (myblock2end + 1 > size){
		int myblock2end = size - 1;
	}
	int outsize = myblock2end - myblock1start + 1;
	int iter1 = myblock1end;
	int iter2 = myblock2end;
	int *outtemp = new int[outsize];

	printf("MergeSortKernel: idx %d Blocksizes %d Block 1 range: %d %d %d %d Block 2 range: %d %d %d %d\n", idx, blocksize, myblock1start, myblock1end, a[myblock1start],a[myblock1end],myblock2start, myblock2end, a[myblock2start],a[myblock2end]);

	for (int outiter = outsize-1; outiter >= 0; outiter--){
		if (a[iter1] > a[iter2])
		{
			outtemp[outiter] = a[iter1];
			iter1--;
		}
		else if (a[iter1] <= a[iter2])
		{
			outtemp[outiter] = a[iter2];
			iter2--;
		}
		//if (outiter > 2000){ printf("Value %d: %d\n", outiter, outtemp[outiter]); }
	}
	for (int outiter = outsize - 1; outiter >= 0; outiter--){
		a[myblock1start+outiter] = outtemp[outiter];
	}
	delete[] outtemp;
}

int reverseMergeSortCPU(int *a, int size, int numblocks, int blocksize, int iblock)
{
	int idx = iblock;
	int myblock1start = blocksize*(idx * 2);
	int myblock2start = blocksize*(idx * 2 + 1);
	int myblock1end = blocksize*(idx * 2 + 1) - 1;
	int myblock2end = blocksize*(idx * 2 + 2) - 1;
	if (myblock2end + 1 > size){
		myblock2end = size - 1;
	}
	if (myblock2start > size - 1){
		return 0;
	}
	int outsize = myblock2end - myblock1start + 1;
	int iter1 = myblock1end;
	int iter2 = myblock2end;
	int *outtemp;
	outtemp = (int *)malloc(outsize * sizeof(int));

	printf("MergeSortCPU: idx %d Blocksizes %d Block 1 range: %d %d Block 2 range: %d %d Total Size: %d\n", idx, blocksize, myblock1start, myblock1end, myblock2start, myblock2end,size); fflush(stdout);

	for (int outiter = outsize - 1; outiter >= 0; outiter--){
		//printf("Iterators %d %d %d\n", iter1, iter2, outiter); fflush(stdout);
		//printf("Values %d %d\n", a[iter1], a[iter2]); fflush(stdout);
		if (a[iter1] > a[iter2])
		{
			outtemp[outiter] = a[iter1];
			iter1--;
		}
		else if (a[iter1] <= a[iter2])
		{
			outtemp[outiter] = a[iter2];
			iter2--;
		}
		//printf("Value %d %d Iterators %d %d\n", outiter, outtemp[outiter], iter1, iter2); fflush(stdout);
		//Sleep(1000);
	}

	for (int outiter = outsize - 1; outiter >= 0; outiter--){
		a[myblock1start + outiter] = outtemp[outiter];
	}
	free(outtemp);

	return 0;
}

int reverseMergeBlocks(int *dev_a, int size, int numblocks, int blocksize)
{
	cudaError_t cudaStatus;
	//Input of blocks recursively calls self until max number of blocks
	if (numblocks < 2){
		return 0;
	}
	int outblocks = (numblocks + 1) / 2;

	// Return non-finished and allow for CPU sorting due to inefficiency at large blocksize
	if (blocksize < 10000){
		printf("Calling reverse MergeSort: %d %d %d\n", numblocks, blocksize, outblocks); fflush(stdout);
		reverseMergeSortKernel << < outblocks, 1 >> >(dev_a, size, numblocks, blocksize);
	}
	else{
		return blocksize;
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "reverseMergeBlock: addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "reverseMergeBlock: cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}

	return reverseMergeBlocks(dev_a, size, outblocks, blocksize * 2);
}

int reverseMergeBlocksCPU(int *a, int size, int numblocks, int blocksize)
{
	if (numblocks < 2){
		return 0;
	}

	int outblocks = (numblocks + 1) / 2;

	for (int iblock = 0; iblock < outblocks; iblock++){
		reverseMergeSortCPU(a, size, numblocks, blocksize, iblock);
	}

	return reverseMergeBlocksCPU(a, size, outblocks, blocksize * 2);
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
		fprintf(stderr, "Library checkwithcuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
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

// Helper function for using CUDA to sum array
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

// Helper function for using CUDA to sort
cudaError_t sortWithCudalib(int *a, int size)
{
	int *dev_a = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "sort: cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "sort: cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "sort: cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	printf("Array Size %d\n", size); fflush(stdout);
	int numBlocks = size / (2*TESTTHREADSPERBLOCK)+1;
	printf("BubbleSort Blocks %d\n", numBlocks); fflush(stdout);
	blocksortKernel << < numBlocks, TESTTHREADSPERBLOCK >> >(dev_a, size);
	numBlocks = (size / TESTTHREADSPERBLOCK) + 1;
	int needCPUsort = reverseMergeBlocks(dev_a, size, numBlocks, TESTTHREADSPERBLOCK);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "sort: addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "sort: cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "sort: cudaMemcpy failed!");
		goto Error;
	}

	if (needCPUsort){
		printf("Calling CPUsort %d\n",needCPUsort);
		reverseMergeBlocksCPU(a, size, (size / (needCPUsort)) + 1, needCPUsort);
	}
	//printf("sort: final values %d %d\n", a[size-2], a[size-1]);

Error:
	cudaFree(dev_a);

	return cudaStatus;
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

int sortarray(int *array, int size)
{
	cudaError_t cudaStatus;
	cudaStatus = sortWithCudalib(array, size);
	return 0;
}