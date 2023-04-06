#include <stdio.h>
#include <cuda_runtime_api.h>
#include "nvmlwrap.h"

#define CHECK_CUDA_ERROR(call) \
do { \
  cudaError_t result = call; \
  if (result != cudaSuccess) { \
    fprintf(stderr, "%s:%d: CUDA error (%d): %s.\n", __FILE__, __LINE__, static_cast<int>(result), cudaGetErrorString(result)); \
     \
  } \
} while(0)

void enableP2P(int numGPUs)
{
    for (int i=0; i<numGPUs; i++)
    {
        cudaSetDevice(i);

        for (int j=0; j<numGPUs; j++)
        {
            int access;
            cudaDeviceCanAccessPeer(&access,i,j);

            if (access)
            {
                cudaDeviceEnablePeerAccess(j,0);
               
            }
        }
    }
}
int main()
{
    int access;
    cudaSetDevice(0);
    for(int i = 1;i < 8;i++){
        cudaDeviceCanAccessPeer(&access,0,i);
        printf("%s \n",access ? "access" : "not access");

        CHECK_CUDA_ERROR(cudaDeviceEnablePeerAccess(i,0));
    }

}