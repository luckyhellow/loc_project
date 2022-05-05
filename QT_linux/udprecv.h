#ifndef UDPRECV_H
#define UDPRECV_H

#include <iostream>
#include <thread>
#include<stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <fstream>
#include <vector>


#define PORT 8888
#define BUF_SIZE 128

using namespace std;


class UDPrecv
{
    int fd[2];
    char buf1[BUF_SIZE] = {0};
    char buf2[BUF_SIZE] = {0};
    char buf[BUF_SIZE] = {0};
    double x = 0;
    double y = 0;

public:
    UDPrecv();
    void recvxy();
    vector<double> getxy(){
        recvxy();
        return vector<double>{x,y};
    }
};

#endif // UDPRECV_H
