#include <stdio.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <pthread.h>
using namespace std;

#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
}

void outputLatencyMatrix(int numGPUs)
{
    int repeat=10000;
    vector<int *> buffers(numGPUs);
    vector<cudaEvent_t> start(numGPUs);
    vector<cudaEvent_t> stop(numGPUs);

    for (int d=0; d<numGPUs; d++)
    {
        cudaSetDevice(d);
        cudaMalloc(&buffers[d],8);
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
                cudaMemcpyPeerAsync(buffers[i],i,buffers[j],j,1);
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

vector<int *> buffers(8);
vector<cudaStream_t> stream(8);
int* rootBuffer;

void* p2pMemCpy(void* arg) 
{
    
    int root = *(int*)arg;
    int d = *((int*)arg+1);
    int rank = *((int*)arg + 2);
    int repeat = *((int*)arg +3);
    cudaSetDevice(rank);
    for(int r=0;r <repeat;r++){
        cudaMemcpyPeerAsync(rootBuffer+d,root,buffers[d],rank,2,stream[d]);
    }
    return NULL;
}

void outputLatencyRoot(int root,vector<int> ranks,int numGPUs)
{
    int repeat=10000;


    cudaEvent_t start;
    cudaEvent_t stop;
    // root
    
    cudaSetDevice(root);
    cudaMalloc(&rootBuffer,32);
    cudaEventCreate(&start);
    cudaCheckError();
    cudaEventCreate(&stop);
    cudaCheckError();
    for(int d= 0;d < numGPUs;d++){
        cudaStreamCreate(&stream[d]);
        cudaCheckError();
    }

    for (int d=0; d<numGPUs; d++)
    {
        cudaSetDevice(ranks[d]);
        cudaMalloc(&buffers[d],8);
        cudaCheckError();
        // cudaEventCreate(&start[d]);
        // cudaCheckError();
        // cudaEventCreate(&stop[d]);
        // cudaCheckError();
    }

    pthread_t threads[8];
    
     

    cudaSetDevice(root);
    cudaDeviceSynchronize();
    cudaCheckError();
    cudaEventRecord(start);
    cudaCheckError();
    for(int d = 0;d < numGPUs;d++){
        int arg[] = {root, d, ranks[d], repeat};
        pthread_create(&threads[d],NULL,p2pMemCpy,arg);
    }

    for (int i = 0; i < numGPUs; i++) {
        pthread_join(threads[i], NULL);
    }
    cudaSetDevice(root);
    cudaCheckError();
    cudaEventRecord(stop);
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();

    float time_ms;
    cudaEventElapsedTime(&time_ms,start,stop);
    double latency = time_ms*1e3/repeat;
    printf("\n latency %6.02f ",latency);

}

int main(int argc,char *argv[])
{
    vector<int> ranks;
    for(int i = 1;i< argc;i++){
        int dev = *argv[i] - '0';
        ranks.push_back(dev);
    }
    //outputLatencyMatrix(ndev);

    
    outputLatencyRoot(0,ranks,ranks.size());
}