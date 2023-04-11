#include "latencyCuda.h"


void init(int local_gpu,int* d_data){
    cudaSetDevice(local_gpu);
    cudaMalloc(&d_data,10*sizeof(int));
    int a[10] = {9,7,5,5,7,8,9,0,1,1};
    cudaMemcpy(d_data,a, 10*sizeof(int), cudaMemcpyHostToDevice);
}

void cudaDTH(void* dst,void* src){
    cudaMemcpy(dst,src, 4, cudaMemcpyDeviceToHost);
}
void cudaHTD(void* dst,void* src){
    cudaMemcpy(dst,src, 4, cudaMemcpyHostToDevice);
}