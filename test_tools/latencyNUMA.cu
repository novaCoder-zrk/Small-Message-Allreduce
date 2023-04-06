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




vector<cudaEvent_t> start(8);
vector<cudaEvent_t> stop(8);
cudaStream_t stream[8][2];
vector<int *> buffers(8);

float latency[8];
void* binaryMemCpy(void * arg)
{
    int rank1 = *(int*)arg;
    int rank2 = *((int*)arg+1);
    int repeat = *((int*)arg +2);
    cudaSetDevice(rank1);
    

    cudaDeviceSynchronize();
    cudaEventRecord(start[rank1]);
    
    for(int r=0;r <repeat;r++){
       cudaMemcpyPeerAsync(buffers[rank2],rank2,buffers[rank1]+1,rank1,1);
       cudaMemcpyPeerAsync(buffers[rank1],rank1,buffers[rank2]+1,rank2,1);
        //cudaMemcpyPeerAsync(buffers[rank1],rank1,buffers[rank2],rank2,1,stream[rank1][0]);
        //cudaMemcpyPeerAsync(buffers[rank2],rank2,buffers[rank1],rank1,1,stream[rank1][1]);
    }
    cudaEventRecord(stop[rank1]);
    cudaCheckError();
    cudaDeviceSynchronize();
    float time_ms;
    cudaEventElapsedTime(&time_ms,start[rank1],stop[rank1]);
    latency[rank1] = time_ms*1e3/repeat;
    return NULL;
}

void outputBiranyLatencyAll(vector<int> ranks1,vector<int> ranks2)
{
    int repeat=10000;
    
    for(int d= 0;d < 8;d++){
        cudaSetDevice(d);
        cudaStreamCreate(&stream[d][0]);
        cudaStreamCreate(&stream[d][1]);
        cudaEventCreate(&start[d]);
        cudaEventCreate(&stop[d]);
        cudaCheckError();
    }

    for (int d=0; d< 8 ; d++)
    {
        cudaSetDevice(d);
        cudaMalloc(&buffers[d],8);
        cudaCheckError();
    }

    pthread_t threads[8];
    
    cudaCheckError();
    for(int i = 0;i < ranks1.size();i++){
        int arg[] = {ranks1[i], ranks2[i], repeat};
        pthread_create(&threads[i],NULL, binaryMemCpy, arg);
    }

    for (int i = 0; i < ranks1.size(); i++) {
        pthread_join(threads[i], NULL);
    }
    float max_latency = -1;
    for(int i = 0;i < ranks1.size();i++)
    {
        max_latency = max(max_latency, latency[ranks1[i]]);
    }

    
    printf("\n latency %6.02f ",max_latency);

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
    //outputLatencyMatrix(ndev);
    outputBiranyLatencyAll(ranks1,ranks2);
    //outputLatencyRoot(0,ranks,ranks.size());
    //outputBinaryLatency(0,1);
}