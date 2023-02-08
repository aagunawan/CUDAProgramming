#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <functional>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

__global__ void matrixMul(const int *a, const int *b, int *c, int N){
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate over row, and down column
    c[row * N + col] = 0;
    for (int k = 0; k < N; k++){
        c[row * N + col] += a[row * N + k]*b[k * N + col]; 
    }
}

int main(){
    // num threads
    int N = 1 << 10;

    size_t bytes = N * N * sizeof(int);

    // initialize vectors in CPU host
    vector<int> h_A(N*N);
    vector<int> h_B(N*N);
    vector<int> h_C(N*N);

    // initializes matrices
    generate(h_A.begin(), h_A.end(), []() { return rand() % 100; });
    generate(h_B.begin(), h_B.end(), []() { return rand() % 100; });

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // copy date from cpu to gpu
    cudaMemcpy(d_a, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_B.data(), bytes, cudaMemcpyHostToDevice);

    int THREADS = 32;

    int BLOCKS = N/THREADS;

    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);
    
    cudaMemcpy(h_C.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;



}