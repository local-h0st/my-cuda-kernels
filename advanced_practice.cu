#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


__global__ void vecDotProduct(const float* A, const float* B, int N, float* result) {
    extern __shared__ float smem[]; // blockSize

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    smem[threadIdx.x] = (idx < N) ? (A[idx] * B[idx]) : 0.0f;

    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step = step / 2) {
        if (threadIdx.x < step) {
            smem[threadIdx.x] += smem[threadIdx.x + step];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(result, smem[0]);
    }

}


__global__ void saxpy(float alpha, const float* X, const float* Y, int N, float* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        result[idx] = alpha * X[idx] + Y[idx];
    }
}


int main(void) {
    // init
    int N = 1 << 20;

    float alpha = 2.0f;

    size_t size = N * sizeof(float);
    float* host_A = (float*)malloc(size);
    float* host_B = (float*)malloc(size);
    float host_vec_dot_product_result = 0.0f;
    float* host_saxpy_result = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        host_A[i] = (float)i;
        host_B[i] = (float)(i+1);
    }

    float *device_A, *device_B, *device_vec_dot_product_result, *device_saxpy_result;
    cudaMalloc(&device_A, size);
    cudaMalloc(&device_B, size);
    cudaMalloc(&device_vec_dot_product_result, sizeof(float));
    cudaMemset(&device_vec_dot_product_result, 0, sizeof(float));
    cudaMalloc(&device_saxpy_result, size);

    // transfer
    cudaMemcpy(device_A, host_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, size, cudaMemcpyHostToDevice);

    // kernel
    int blockSize = 256;
    int gridSize = (N + blockSize -1) / blockSize;
    vecDotProduct<<<gridSize, blockSize, blockSize * sizeof(float)>>>(device_A, device_B, N, device_vec_dot_product_result);
    saxpy<<<gridSize, blockSize, blockSize>>>(alpha, device_A, device_B, N, device_saxpy_result);
    cudaMemcpy(host_vec_dot_product_result, device_vec_dot_product_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_saxpy_result, device_saxpy_result, size, cudaMemcpyDeviceToHost);

    // verify
    bool passed = false;
    for (int i = 0; i < N; i++) {
        if (fabsf(host_saxpy_result[i] - alpha * host_A[i] - host_B[i]) > 1e-5f) {
            printf("verify failed at index %d.\n", i);
            break;
        }
    }
    if (passed) {
        printf("verification passed.\n");
    }
    printf("host_vec_dot_product_result: %d\n", host_vec_dot_product_result);

    
    // recycle memory
    free(host_A);
    free(host_B);
    free(host_saxpy_result);
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_saxpy_result);


}