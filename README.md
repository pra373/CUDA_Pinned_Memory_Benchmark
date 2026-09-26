# CUDA Pinned Memory Benchmark

A simple CUDA benchmark to compare **pageable host memory** and **pinned (page-locked) host memory** for CPU ↔ GPU data transfers.

The project measures transfer time and effective transfer speed for:

* Host → Device
* Device → Host

Each test performs **100 memory transfers** of a **40 MiB** array.

## Memory Types

### Pageable Memory

Allocated using:

```cpp
malloc()
```

### Pinned Memory

Allocated using:

```cpp
cudaHostAlloc()
```

Pinned memory is page-locked, meaning the operating system cannot page that memory out while it is pinned. This allows CUDA to perform host ↔ device transfers more efficiently.

## Results

Tested on an **NVIDIA RTX 3060 Laptop GPU**.

| Transfer      | Memory   |       Time |       Speed |
| ------------- | -------- | ---------: | ----------: |
| Host → Device | Pageable | 366.528 ms | 10,913 MB/s |
| Device → Host | Pageable | 350.928 ms | 11,398 MB/s |
| Host → Device | Pinned   | 326.156 ms | 12,264 MB/s |
| Device → Host | Pinned   | 318.682 ms | 12,552 MB/s |

The benchmark shows that pinned host memory provides **higher transfer throughput and lower transfer time** compared with pageable host memory.

## Technologies

* C++
* CUDA 12.5
* CUDA Runtime API
* NVIDIA RTX 3060 Laptop GPU
* Visual Studio 2022

## Main CUDA APIs Used

```cpp
cudaMalloc()
cudaHostAlloc()
cudaMemcpy()
cudaEventCreate()
cudaEventRecord()
cudaEventElapsedTime()
cudaFree()
cudaFreeHost()
```

## Purpose

This project was built while learning about **pageable and pinned host memory** in CUDA and demonstrates their practical effect on CPU ↔ GPU memory transfer performance.
