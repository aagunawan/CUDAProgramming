#include <iostream>
#include <cassert>
#include <algorithm>

__global__ void vectorAdd (int *a, int *b, int *c, int N){
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < N) c[tid] = a[tid] + b[tid];
}

int main(){
    const int N = 1 << 16;
    size_t bytes = sizeof(int)*N;

    // unified memory
    int *a, *b, *c;

    // allocation memory
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Initialize vectors
    for (int i = 0; i < N; i++){
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    int BLOCK_SIZE = 1 << 10;

    int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    vectorAdd<<<BLOCK_SIZE,GRID_SIZE>>>(a,b,c,N);

    cudaDeviceSynchronize();                              

    cudaFree(a);
    cudaFree(b);X
    cudaFree(c);

    std::cout << "Successful\n";

    return 0;

}