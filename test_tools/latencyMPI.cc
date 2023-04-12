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

  // if (size != 2) { // 本例只适用于两个进程通信
  //   std::cerr << "This example requires exactly two processes." << std::endl;
  //   MPI_Abort(MPI_COMM_WORLD, 1);
  // }
  int* d_data;
  char buffer[256];
  int local_gpu = atoi(argv[1+rank]);
  init(local_gpu,d_data);

  struct timespec start, end;
  double elapsed_us = 0;   
  MPI_Barrier(MPI_COMM_WORLD); 
  clock_gettime(CLOCK_MONOTONIC, &start); 
  int repeat = 1;
  for(int i = 0; i< repeat;i++){
    if(rank%2 == 0){
      // send
      cudaDTH(buffer,d_data);
      MPI_Send(&data, 1, MPI_INT, (rank + 1) , 0, MPI_COMM_WORLD);
    }else{
      // recv
      MPI_Recv(&data, 1, MPI_INT, (rank-1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      cudaDTH(d_data,buffer);
    }
    
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  elapsed_us = duration_cast<microseconds>(duration<double>(end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1000000000.)).count();
  elapsed_us = elapsed_us / repeat;
  
  // for (int i = 0; i < size; i++) {
  //       if (rank == i % size) { // 每个进程依次发送数据
  //           data = i * i;
  //           MPI_Send(&data, 1, MPI_INT, (i + 1) % size, 0, MPI_COMM_WORLD);
  //           printf("Process %d sent %d to process %d\n", rank, data, (i + 1) % size);
  //       } else if (rank == (i + 1) % size) { // 每个进程依次接收数据
  //           MPI_Recv(&data, 1, MPI_INT, i % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //           printf("Process %d received %d from process %d\n", rank, data, i % size);
  //       }
  //       MPI_Barrier(MPI_COMM_WORLD); // 等待所有进程执行完以上代码
  //   }
   struct utsname uts;
    if (uname(&uts) == -1)
    {
        std::cerr << "Error: unable to get system information" << std::endl;
        return 1;
    }

    std::cout << "Hostname: " << uts.nodename <<" local_gpu: "<<local_gpu<<" latency: "<<elapsed_us<< std::endl;
    MPI_Finalize(); // 结束MPI环境

  return 0;
}