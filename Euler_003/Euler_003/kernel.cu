
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//NVIDIA GTX 670 1024 threads/block, 2.1B blocks	
#define THREADSPERBLOCK 1024


cudaError_t isPrimeWithCuda(long long int *a, long int *primes, int sizearray, int numprimes);
cudaError_t whichPrimeWithCuda(long long int *a, long int *primes, int sizearray, int numprimes);

long long int* filltestarray(long int largeprime);
long int* addtolist(long int* oldlist, long long int newprime, int plength);

__global__ void fillArrayKernel(long long int *a, int sizearray)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	long int idx = i * THREADSPERBLOCK + j;

	if (idx < sizearray)
	{
		a[idx] = a[0] + idx;
	}
}
__global__ void isPrimeKernel(long long int *a, long int *primes, int sizearray, int numprimes)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	int primeidx = blockIdx.y;
	long int idx = i* THREADSPERBLOCK + j;

	long long int value;

	if (idx < sizearray)
	{
		value = a[idx];
		if (value % primes[primeidx] == 0)
		{
			a[idx] = 0;
			return;
		}
	}
}
__global__ void whichPrimeKernel(long long int *a, long int *primes, int sizearray, int numprimes)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	long int idx = i * 1024 + j;

	if (idx < numprimes)
	{
		//a[idx] = a[0];
		//printf("Kernel: %d Checking: %lld %d", idx, a[0], primes[idx]);
		if (a[0] % primes[idx] == 0)
		{
			printf("Kernel: %d Prime Factor: %d\n",idx, primes[idx]);
			primes[idx] *= -1;
			return;
		}
	}
}

int main()
{
	int sqrtnum;
	double inputnum = 600851475143;
	long long int inputint = 600851475143;

	double squareroot = std::sqrt(inputnum);
	sqrtnum = floor(squareroot);
	printf("Sqrt %d %f\n", sqrtnum, squareroot); fflush(stdout);

	long int *primelist;
	primelist = (long int *)malloc(sizeof(long int));
	primelist[0] = 2;

	int plength = 1;

	long long int *testarray;

	long int maxprime = primelist[plength-1];
	bool done = false;
	while (!done)
	{
		int arraylength = maxprime * (maxprime - 1);
		printf("Arraylength: %d maxprime %d %d\n", arraylength, maxprime, primelist[plength-1]); fflush(stdout);
		testarray = filltestarray(maxprime);

		printf("Array: %d\n", testarray[0]); fflush(stdout);

		//Reduces computations to minimal needed, since square arrays get big fast
		if ((maxprime * maxprime) > sqrtnum)
		{
			arraylength = (sqrtnum - maxprime);
			done = true;
			printf("All done here %d %d\n", maxprime, sqrtnum); fflush(stdout);
		}

		printf("Calling isPrime: %d %d %d\n", arraylength, plength, sqrtnum); fflush(stdout);
		cudaError_t cudaStatus = isPrimeWithCuda(testarray, primelist, arraylength, plength);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}

		for (int i = 0; i < arraylength; i++)
		{
			if (testarray[i] != 0)
			{
				primelist = addtolist(primelist, testarray[i], plength);
				plength += 1;
				//printf("Add %d to total primes %d\n", primelist[plength-1], plength); fflush(stdout);
			}
		}

		free(testarray);
		maxprime = primelist[plength-1];
		printf("Max prime %d %d\n", primelist[plength-1], plength); fflush(stdout);
	}

	testarray = (long long int *)malloc(sizeof(long long int));
	int arraylength = 1;
	printf("Final Run arraylength %d %lld\n", plength, inputint); fflush(stdout);
	testarray[0] = inputint;
	printf("Testarray %lld\n", testarray[0]); fflush(stdout);
	cudaError_t cudaStatus = whichPrimeWithCuda(testarray, primelist, arraylength, plength);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	long int biggest = 0;
	for (int i = 0; i < plength; i++)
	{
		if (primelist[i] < 0)
		{
			biggest = primelist[i];
		}
	}
	printf("Biggest prime %d", biggest);
	free(testarray);
	return 0;
}

long long int* filltestarray(long int largeprime)
{
	long long int *array;
	long int maxval = largeprime * largeprime - largeprime;

	printf("Filltestarray maxval %d largeprime %d\n", maxval, largeprime);
	array = (long long  int *)malloc(sizeof(long long int)*maxval);
	array[0] = largeprime+1;
	return array;
}

long int* addtolist(long int* oldlist, long long int newprime, int plength)
{
	oldlist = (long int *)realloc(oldlist, sizeof(long int)*(plength+1));
	oldlist[plength] = newprime;
	return oldlist;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t isPrimeWithCuda(long long int *a, long int *primes, int sizearray, int numprimes)
{
	long long int *dev_a = 0;
	long int *dev_primes = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_a, sizearray * sizeof(long long int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_primes, numprimes * sizeof(long int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, sizeof(long long int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_primes, primes, numprimes*sizeof(long int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int numblocks = (sizearray / THREADSPERBLOCK)+1;
	printf("Cuda blocks %d\n", numblocks); fflush(stdout);
	printf("Sizearray %d numprimes %d\n", sizearray, numprimes); fflush(stdout);

	// First fills the array since only the first index was initialized
	fillArrayKernel << <numblocks, THREADSPERBLOCK >> >(dev_a, sizearray);
	// Launch a kernel on the GPU with one thread for each element.
	const dim3 blockSize(numblocks, numprimes, 1);
	isPrimeKernel << <blockSize, THREADSPERBLOCK >> >(dev_a, dev_primes, sizearray, numprimes);

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
	cudaStatus = cudaMemcpy(a, dev_a, sizearray * sizeof(long long int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_a);
	cudaFree(dev_primes);

	return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t whichPrimeWithCuda(long long int *a, long int *primes, int sizearray, int numprimes)
{
	long long int *dev_a = 0;
	long int *dev_primes = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_a, sizearray * sizeof(long long int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_primes, numprimes * sizeof(long int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, sizeof(long long int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_primes, primes, numprimes*sizeof(long int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int numblocks = (numprimes / 1024) + 1;
	printf("Cuda blocks %d\n", numblocks); fflush(stdout);
	printf("Sizearray %d numprimes %d\n", sizearray, numprimes); fflush(stdout);

	// Launch a kernel on the GPU with one thread for each element.
	whichPrimeKernel << <numblocks, 1024 >> >(dev_a, dev_primes, sizearray, numprimes);

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
	cudaStatus = cudaMemcpy(a, dev_a, sizearray * sizeof(long long int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(primes, dev_primes, numprimes* sizeof(long int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_a);
	cudaFree(dev_primes);

	return cudaStatus;
}
