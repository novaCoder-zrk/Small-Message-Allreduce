#include <cuda_runtime_api.h>
void init(int local_gpu,int* d_data);
void cudaDTH(void* dst,void* src);
void cudaHTD(void* dst,void* src);
