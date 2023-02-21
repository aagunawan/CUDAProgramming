%%cu
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <functional>

using std::cout;
using std::vector;

__global__ void sum3Arr(const int* arr1, const int* arr2, const int* arr3, int* arr4){
    int blockOffset = blockIdx.x*blockDim.x;
    int gid = blockOffset + threadIdx.x;

    arr4[gid] = arr1[gid] + arr2[gid] + arr3[gid];
}

int main(){
    int N = 1 << 11;
    cudaError error;

    vector<int> h_A(N*N);
    vector<int> h_B(N*N);
    vector<int> h_C(N*N);
    vector<int> h_D(N*N);
    vector<int> h_(N*N);

    for (int i = 0; i < N*N; i++){
        h_A[i] = i;
        h_B[i] = i;
        h_C[i] = i;
    }

    clock_t cpu_start, cpu_end;
    cpu_start = clock();
    for (int i = 0; i < N*N; i++){
        h_[i] = h_A[i] + h_B[i] + h_C[i];
    }
    cpu_end = clock();
    printf("Sum array CPU: %4.6f\n", (double)((double)(cpu_end - cpu_start)/CLOCKS_PER_SEC));

    int *d_a, *d_b, *d_c, *d_d;
    size_t byte_size = N*N*sizeof(int);
    error = cudaMalloc((int**)&d_a, byte_size);
    if (error != cudaSuccess){
        fprintf(stderr,"Error: %s \n", cudaGetErrorString(error));
    }

    error = cudaMalloc((int**)&d_b, byte_size);
    if (error != cudaSuccess){
        fprintf(stderr,"Error: %s \n", cudaGetErrorString(error));
    }
    error = cudaMalloc((int**)&d_c, byte_size);
    if (error != cudaSuccess){
        fprintf(stderr,"Error: %s \n", cudaGetErrorString(error));
    }
    error = cudaMalloc((int**)&d_d, byte_size);
    if (error != cudaSuccess){
        fprintf(stderr,"Error: %s \n", cudaGetErrorString(error));
    }

    clock_t htod_start, htod_end;
    htod_start = clock();
    cudaMemcpy(d_a, h_A.data(), byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_B.data(), byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_C.data(), byte_size, cudaMemcpyHostToDevice);
    htod_end = clock();
    
    dim3 block(64);
    dim3 grid((N*N)/block.x);

    clock_t gpu_start, gpu_end;
    gpu_start = clock();
    sum3Arr<<<grid,block>>>(d_a, d_b, d_c, d_d);
    cudaDeviceSynchronize();
    gpu_end = clock();

    clock_t dtoh_start, dtoh_end;
    dtoh_start = clock();
    cudaMemcpy(h_D.data(), d_d, byte_size, cudaMemcpyDeviceToHost);
    dtoh_end = clock();
    // for (int i = 0; i < 128; i++){
    //     cout << h_D[i] << std::endl;
    // }

    printf("htod: %4.6f\n", (double)((double)(htod_end - htod_start)/CLOCKS_PER_SEC));
    printf("gpu: %4.6f\n", (double)((double)(gpu_end - gpu_start)/CLOCKS_PER_SEC));
    printf("dtoh: %4.6f\n", (double)((double)(dtoh_end - dtoh_start)/CLOCKS_PER_SEC));
    

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    cudaDeviceReset();
    return 0;
}