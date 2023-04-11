#include <stdio.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <pthread.h>

#include <ctime>
#include <ratio>
#include <chrono>

using namespace std;
using namespace std::chrono;
#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
}




vector<int *>buffer_send;
vector<int *>buffer_recv;
vector<cudaEvent_t> start;
vector<cudaEvent_t> stop;
vector<float> latency;

void myCudaMemCpy(void *buffer_recv,int rank2,void*buffer_send,int rank1, size_t buffer_size)
{
    // 主机创建一个块内存
    char* h_data = (char*)malloc(buffer_size);
    cudaMemcpy( h_data, buffer_send, buffer_size, cudaMemcpyDeviceToHost);

    cudaMemcpy(buffer_recv, h_data, buffer_size, cudaMemcpyHostToDevice);
    free(h_data);
}




void* deviceMemCpy(void * arg)
{
    int rank1 = *(int*)arg;
    int rank2 = *((int*)arg+1);
    int repeat = *((int*)arg +2);
    int index = *((int*)arg +3);

    cudaSetDevice(rank1);
    cudaCheckError();

    //cudaDeviceSynchronize();
    //cudaEventRecord(start[index]);
    
    for(int r=0;r <repeat;r++){
      myCudaMemCpy(buffer_recv[index],rank2,buffer_send[index],rank1,1);

    }
    //cudaEventRecord(stop[index]);
    //cudaCheckError();
    //cudaDeviceSynchronize();
    // float time_ms;
    // cudaEventElapsedTime(&time_ms,start[index],stop[index]);
    // latency[index] = time_ms*1e3/repeat;
    return NULL;
}


void latencyAll(vector<int> rank1,vector<int>rank2)
{
    int repeat = 100;
    buffer_send = vector<int *>(rank1.size());
    buffer_recv = vector<int *>(rank1.size());

    start = vector<cudaEvent_t>(rank1.size());
    stop = vector<cudaEvent_t>(rank1.size());
    latency = vector<float>(rank1.size());
    // 
    for(int i = 0;i < rank1.size();i++){
        cudaSetDevice(rank1[i]);
        cudaMalloc(&buffer_send[i],10*sizeof(int));
        cudaEventCreate(&start[i]);
        cudaEventCreate(&stop[i]);
        cudaCheckError();
    }

    for(int i = 0;i < rank2.size();i++){
        cudaSetDevice(rank2[i]);
        cudaMalloc(&buffer_recv[i],10*sizeof(int));
        cudaCheckError();
    }

    vector<pthread_t> threads(rank1.size());
    struct timespec start, end;

    // 获取起始时刻
    clock_gettime(CLOCK_MONOTONIC, &start);
   
    for(int i = 0;i < rank1.size();i++){
        int arg[] = {rank1[i], rank2[i], repeat,i};
        pthread_create(&threads[i],NULL, deviceMemCpy, arg);
    }

    for (int i = 0; i < rank1.size(); i++) {
        pthread_join(threads[i], NULL);
    }
     clock_gettime(CLOCK_MONOTONIC, &end);

    // 计算耗时并以us为单位输出
    double elapsed_us = duration_cast<microseconds>(duration<double>(end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1000000000.)).count();
    // float max_latency = -1;
    // for(int i = 0;i < rank1.size();i++)
    // {
    //     max_latency = max(max_latency, latency[i]);
    // }
    float latency = elapsed_us / repeat;
    
    printf("\n latency %f \n",latency);

}
void outputLatencyMatrix(int numGPUs)
{
    int repeat=100;
    vector<int *> buffers(numGPUs);
    vector<cudaEvent_t> start(numGPUs);
    vector<cudaEvent_t> stop(numGPUs);

    for (int d=0; d<numGPUs; d++)
    {
        cudaSetDevice(d);
        cudaMalloc(&buffers[d],1);
        cudaCheckError();
        cudaEventCreate(&start[d]);
        cudaCheckError();
        cudaEventCreate(&stop[d]);
        cudaCheckError();
    }

    vector<double> latencyMatrix(numGPUs*numGPUs);

    for (int i=0; i<numGPUs; i++)
    {
        cudaSetDevice(i);

        for (int j=0; j<numGPUs; j++)
        {

            cudaDeviceSynchronize();
            cudaCheckError();
            cudaEventRecord(start[i]);

            for (int r=0; r<repeat; r++)
            {
                myCudaMemCpy(buffers[i],i,buffers[j],j,1);
            }

            cudaEventRecord(stop[i]);
            cudaDeviceSynchronize();
            cudaCheckError();

            float time_ms;
            cudaEventElapsedTime(&time_ms,start[i],stop[i]);

            latencyMatrix[i*numGPUs+j]=time_ms*1e3/repeat;
        }
    }

    printf("   D\\D");

    for (int j=0; j<numGPUs; j++)
    {
        printf("%6d ", j);
    }

    printf("\n");

    for (int i=0; i<numGPUs; i++)
    {
        printf("%6d ",i);

        for (int j=0; j<numGPUs; j++)
        {
            printf("%6.02f ", latencyMatrix[i*numGPUs+j]);
        }

        printf("\n");
    }

    for (int d=0; d<numGPUs; d++)
    {
        cudaSetDevice(d);
        cudaFree(buffers[d]);
        cudaCheckError();
        cudaEventDestroy(start[d]);
        cudaCheckError();
        cudaEventDestroy(stop[d]);
        cudaCheckError();
    }
}



int main(int argc,char *argv[])
{
    vector<int> ranks1;
    vector<int> ranks2;
    for(int i = 1;i< argc;i += 2){
        int dev;
        dev = *argv[i] - '0';
        ranks1.push_back(dev);
        dev = *argv[i+1] - '0';
        ranks2.push_back(dev);
    }
    for(int i = 0;i < ranks1.size();i++){
        printf("%d %d\t",ranks1[i],ranks2[i]);
    }
   
    latencyAll(ranks1,ranks2);
    //outputLatencyMatrix(8);

}