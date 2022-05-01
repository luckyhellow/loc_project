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


#define PORT 8888


using namespace std;


class UDPrecv
{
//    double * x = new double(0);
//    double * y = new double(0);
    char strx[128];
    char stry[128];
public:
    UDPrecv();
    double getx();
    double gety();
//    void udp(){UDP(x,y);}
};

#endif // UDPRECV_H
