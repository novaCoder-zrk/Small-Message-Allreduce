#include <iostream>
#include <mpi.h>
#include <sys/utsname.h>
#include "latencyCuda.h"
#include <ctime>
#include <ratio>
#include <chrono>
#include <stdlib.h>
using namespace std;
using namespace std::chrono;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // 初始化MPI环境
    int data;
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // 获取进程编号
    MPI_Comm_size(MPI_COMM_WORLD, &size); // 获取进程总数
    int gpu1 = atoi(argv[1+2*rank]);
    int gpu2 = atoi(argv[1+2*rank+1]);



    int* d_data[4];

    char buffer[256];

  
    init(gpu1,d_data[0]);
  
    init(gpu2,d_data[1]);

  


  struct timespec start, end;
  double elapsed_us = 0;   
  //MPI_Barrier(MPI_COMM_WORLD); 
  clock_gettime(CLOCK_MONOTONIC, &start); 
  int repeat = 1;
  for(int i = 0; i< repeat;i++){
    
      cudaDTH(buffer,d_data[0]);
    cudaHTD(d_data[1],buffer);
    }
    
  
  //MPI_Barrier(MPI_COMM_WORLD); 
  clock_gettime(CLOCK_MONOTONIC, &end);
  elapsed_us = duration_cast<microseconds>(duration<double>(end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1000000000.)).count();
  elapsed_us = elapsed_us / repeat;
  


    std::cout <<" local_gpu: "<<local_gpu<<" latency: "<<elapsed_us<< std::endl;
    MPI_Finalize(); // 结束MPI环境

  return 0;
}