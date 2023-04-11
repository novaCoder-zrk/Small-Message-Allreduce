#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 
#include <cuda_runtime_api.h>
#include <ctime>
#include <ratio>
#include <chrono>
using namespace std::chrono;
using namespace std;
void error(const char *msg) {
    perror(msg);
    exit(0);
}

int main(int argc, char *argv[]) {
    int sockfd, portno, n;
    struct sockaddr_in serv_addr;
    struct hostent *server;

    int* d_data;
    int a[10] = {9,7,5,5,7,8,9,0,1,1};
    cudaSetDevice(0);
    cudaMalloc(&d_data, 10*sizeof(int));
    cudaMemcpy(d_data,a, 10*sizeof(int), cudaMemcpyHostToDevice);


    char buffer[256];
    if (argc < 3) {
       fprintf(stderr,"usage %s hostname port\n", argv[0]);
       exit(0);
    }

    portno = atoi(argv[2]);
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) 
        error("ERROR opening socket");

    server = gethostbyname(argv[1]);
    if (server == NULL) {
        fprintf(stderr,"ERROR, no such host\n");
        exit(0);
    }

    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr, 
         (char *)&serv_addr.sin_addr.s_addr,
         server->h_length);
    serv_addr.sin_port = htons(portno);

    if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0) 
        error("ERROR connecting");

    //printf("Please enter the message: \n");
    bzero(buffer,256);
    //fgets(buffer,255,stdin);
    buffer[0] = 'A';
    n = write(sockfd,buffer,strlen(buffer));
    if (n < 0) 
         error("ERROR writing to socket");

    bzero(buffer,256);
    n = read(sockfd,buffer,255);
    if (n < 0) 
         error("ERROR reading from socket");

    printf("%s\n",buffer);

    
    struct timespec start, end;
    // 获取起始时刻
    clock_gettime(CLOCK_MONOTONIC, &start); 
    
     
    cudaMemcpy(buffer,d_data, 1, cudaMemcpyDeviceToHost);
        //printf("1\n");
    n = write(sockfd,buffer,1);  
        //printf("2\n");
        
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_us = duration_cast<microseconds>(duration<double>(end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1000000000.)).count();

    float latency = elapsed_us;
    close(sockfd);
    return 0;
}