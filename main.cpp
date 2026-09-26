#include<iostream>
#include<cuda_runtime.h>

#define ARRAYSIZE 1024 * 1024 * 10

// Macro for checking CUDA API errors
#define HANDLE_ERROR(call) \
do \
{ \
	cudaError_t error = (call); \
	if (error != cudaSuccess) \
	{ \
		cout << "CUDA error: " << cudaGetErrorString(error) << endl; \
		exit(EXIT_FAILURE); \
	} \
} while (0)

using std::cout;
using std::endl;

float cuda_malloc_test(int size, bool up);
float cuda_host_alloc_test(int size, bool up);


int main(void)
{
	float TimeRequired;

	// Total data transferred = size of one array * 100 transfers
	double transferMB = (double)ARRAYSIZE * sizeof(int) * 100 / (1024.0 * 1024.0);
	double MBPerSecond;

	TimeRequired = cuda_malloc_test(ARRAYSIZE, true);

	MBPerSecond = transferMB / (TimeRequired / 1000.0);

	cout << "Time using malloc (paged memory) from host to device : " << TimeRequired << " ms" << endl;
	cout << "Transfer speed : " << MBPerSecond << " MB/s" << endl;

	TimeRequired = cuda_malloc_test(ARRAYSIZE, false);

	MBPerSecond = transferMB / (TimeRequired / 1000.0);

	cout << "Time using malloc (paged memory) from device to host : " << TimeRequired << " ms" << endl;
	cout << "Transfer speed : " << MBPerSecond << " MB/s" << endl;

	TimeRequired = cuda_host_alloc_test(ARRAYSIZE, true);

	MBPerSecond = transferMB / (TimeRequired / 1000.0);

	cout << "Time using cudaHostAlloc() (paged-locked memory) from host to device : " << TimeRequired << " ms" << endl;
	cout << "Transfer speed : " << MBPerSecond << " MB/s" << endl;

	TimeRequired = cuda_host_alloc_test(ARRAYSIZE, false);

	MBPerSecond = transferMB / (TimeRequired / 1000.0);

	cout << "Time using cudaHostAlloc() (paged-locked memory) from device to host : " << TimeRequired << " ms" << endl;
	cout << "Transfer speed : " << MBPerSecond << " MB/s" << endl;

	getchar();

	return(0);
}


float cuda_malloc_test(int size, bool up)
{
	size_t arraySizeInBytes = size * sizeof(int);
	int* hostArray = nullptr;
	int* deviceArray = nullptr;
	cudaEvent_t start, stop;
	float elaspedTime;

	// Allocate pageable host memory
	hostArray = (int*)malloc(arraySizeInBytes);

	if (hostArray == nullptr)
	{
		cout << "failed to allocate memory for host array using malloc (paged memory) !" << endl;
		exit(EXIT_FAILURE);
	}

	HANDLE_ERROR(cudaMalloc((void**)&deviceArray, arraySizeInBytes));

	// Fill host array
	for (int i = 0; i < size; i++)
	{
		hostArray[i] = 1;
	}

	HANDLE_ERROR(cudaMemcpy(deviceArray, hostArray, arraySizeInBytes, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	HANDLE_ERROR(cudaEventRecord(start, 0));

	for (int i = 0; i < 100; i++)
	{
		if (up)
		{
			HANDLE_ERROR(cudaMemcpy((void*)deviceArray, (void*)hostArray, arraySizeInBytes, cudaMemcpyHostToDevice));
		}
		else
		{
			HANDLE_ERROR(cudaMemcpy((void*)hostArray, (void*)deviceArray, arraySizeInBytes, cudaMemcpyDeviceToHost));
		}
	}

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elaspedTime, start, stop));

	free(hostArray);
	cudaFree(deviceArray);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return(elaspedTime);
}


float cuda_host_alloc_test(int size, bool up)
{
	size_t arraySizeInBytes = size * sizeof(int);
	int* hostArray = nullptr;
	int* deviceArray = nullptr;
	cudaEvent_t start, stop;
	float elaspedTime;

	// Allocate pinned/page-locked host memory
	HANDLE_ERROR(cudaHostAlloc((void**)&hostArray, arraySizeInBytes, cudaHostAllocDefault));

	HANDLE_ERROR(cudaMalloc((void**)&deviceArray, arraySizeInBytes));

	for (int i = 0; i < size; i++)
	{
		hostArray[i] = 1;
	}

	HANDLE_ERROR(cudaMemcpy(deviceArray, hostArray, arraySizeInBytes, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	HANDLE_ERROR(cudaEventRecord(start, 0));

	for (int i = 0; i < 100; i++)
	{
		if (up)
		{
			HANDLE_ERROR(cudaMemcpy((void*)deviceArray, (void*)hostArray, arraySizeInBytes, cudaMemcpyHostToDevice));
		}
		else
		{
			HANDLE_ERROR(cudaMemcpy((void*)hostArray, (void*)deviceArray, arraySizeInBytes, cudaMemcpyDeviceToHost));
		}
	}

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elaspedTime, start, stop));

	cudaFreeHost(hostArray);
	cudaFree(deviceArray);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return(elaspedTime);
}