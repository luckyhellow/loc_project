#ifndef UDPRECV_H
#define UDPRECV_H

#include <iostream>
#include <thread>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <WinSock2.h>
#include <sys/types.h>
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
    double x = 0;
    double y = 0;
    bool re = true;
    string label;
    TIME time;
    pthread_mutex_t mtx;

public:
    UDPrecv();
    static void* recvxy(void* args);
    Struct_XY getxy();
};

#endif // UDPRECV_H
