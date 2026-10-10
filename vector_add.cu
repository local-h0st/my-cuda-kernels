#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


__global__ void vecAdd(const float* A, const float* B, const float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        C[idx] = A[idx] + B[idx]
    }
}

int main(void){
    int N = 1 << 20;
    size_t size = N * sizeof(float);

    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size -1) / block_size;
    vecAdd<<grid_size, block_size>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // verify
    for (int i = 0; i < N; i++) {
        if (fabsf(h_C[i] - h_A[i] - h_B[i]) > 1e-5f) {
            printf("verify failed at index %d.\n", i);
            break;
        }
    }
    printf("verification passed.\n");

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;


}