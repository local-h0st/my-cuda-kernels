#include <cstdio>

__global__ void helloKernel() {
    printf("'hello' from (block_idx=%d, thread_idx=%d)\n", blockIdx.x, threadIdx.x);
}

int main(){
    printf("launching kernel ...\n");
    helloKernel<<<2, 4>>>();
    cudaDeviceSynchronize();
    printf("Done\n");
    return 0;
}