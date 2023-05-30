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
  int local_gpu = atoi(argv[1+rank]);


  int* d_data[4];

  char buffer[256];

  if(local_gpu<4){
    for(int i = 0;i<=3;i++){
      init((local_gpu+i)%4,d_data[i]);
   }
  }else{
    for(int i = 0;i<=3;i++){
      init((local_gpu+i)%4+4,d_data[i]);
    }
  }
  


  struct timespec start, end;
  double elapsed_us = 0;   
  //MPI_Barrier(MPI_COMM_WORLD); 
  clock_gettime(CLOCK_MONOTONIC, &start); 
  int repeat = 1;
  for(int i = 0; i< repeat;i++){
    
      cudaDTH(buffer,d_data[0]);
      for(int i = 1;i <=3;i++)
        cudaHTD(d_data[i],buffer);
    }
    
  
  //MPI_Barrier(MPI_COMM_WORLD); 
  clock_gettime(CLOCK_MONOTONIC, &end);
  elapsed_us = duration_cast<microseconds>(duration<double>(end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1000000000.)).count();
  elapsed_us = elapsed_us / repeat;
  


    std::cout <<" local_gpu: "<<local_gpu<<" latency: "<<elapsed_us<< std::endl;
    MPI_Finalize(); // 结束MPI环境

  return 0;
}