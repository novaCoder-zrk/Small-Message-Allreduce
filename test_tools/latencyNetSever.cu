#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <cuda_runtime_api.h>
#include <ctime>
#include <ratio>
#include <chrono>
using namespace std::chrono;
using namespace std;
void error(const char *msg) 
{    perror(msg);    
    exit(1);
}



int main(int argc, char *argv[]) 
{
    int* d_data;
    cudaMalloc(&d_data,10*sizeof(int));
    cudaSetDevice(0);
    
    int sockfd, newsockfd, portno;
    socklen_t clilen;    
    char buffer[256];    
    struct sockaddr_in serv_addr, cli_addr;    
    int n;    
    sockfd = socket(AF_INET, SOCK_STREAM, 0);    
    if (sockfd < 0)    
         error("ERROR opening socket");    
    bzero((char *) &serv_addr, sizeof(serv_addr));
    
    portno = atoi(argv[1]);
    serv_addr.sin_family = AF_INET;    
    serv_addr.sin_addr.s_addr = INADDR_ANY;    
    serv_addr.sin_port = htons(portno);    
    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0)         
        error("ERROR on binding");    listen(sockfd,5);    clilen = sizeof(cli_addr);    
    
    newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);    
    if (newsockfd < 0)    
         error("ERROR on accept");    
    bzero(buffer,256);

    n = read(newsockfd,buffer,255);    
    if (n < 0) 
        error("ERROR reading from socket");    
    printf("Here is the message: %s\n",buffer);   

    n = write(newsockfd,"I got your message",18);   
    
    if (n < 0) 
        error("ERROR writing to socket");    

    struct timespec start, end;

    
    // 获取起始时刻
     double elapsed_us = 0;

      
    clock_gettime(CLOCK_MONOTONIC, &start); 
    // 从主机发往GPU
    n = read(newsockfd,buffer,1);  
    cudaMemcpy(d_data,buffer, 1, cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_us = duration_cast<microseconds>(duration<double>(end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1000000000.)).count();



    float latency = elapsed_us;
    printf("latency: %f\n",latency);
    close(newsockfd);    
    close(sockfd);    
    return 0; 
}