#ifndef UDPRECV_H
#define UDPRECV_H

#include <iostream>
#include <thread>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <fstream>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <tools.h>

#define PORT 8888
#define BUF_SIZE 256

using namespace std;


class UDPrecv
{
    int st;
    int fd[2];
    char buflabel[BUF_SIZE] = {0};
    char buf1[BUF_SIZE] = {0};
    char buf2[BUF_SIZE] = {0};
    char buf[BUF_SIZE] = {0};
    char buflast[BUF_SIZE] = {0};
    double x = 0;
    double y = 0;
    bool re = true;
    string label;
    TIME time;

public:
    UDPrecv();
    void recvxy();
    Struct_XY getxy(){
        recvxy();
        return Struct_XY{label,time,x,y};
    }
};

#endif // UDPRECV_H
