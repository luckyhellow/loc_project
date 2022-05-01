#include "udprecv.h"
#include <signal.h>
UDPrecv::UDPrecv()
{
//    signal(SIGCHLD, SIG_IGN);
    if(fork()>0){
//        signal(SIGCHLD, SIG_IGN);
        return;
    }
    else{
        struct sockaddr_in serveraddr, clientaddr;
        char buf[128] = "11#11";
        char buf1[128] = {0};
        char buf2[128] = {0};
        /* socket文件描述符 */
        int sockfd;

        /* 建立udp socket */
        sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if(sockfd < 0)
        {
            perror("socket");
            exit(1);
        }

        /* 设置address */
        socklen_t len = sizeof(struct sockaddr_in);

        memset(&serveraddr, 0, sizeof(struct sockaddr_in));
        serveraddr.sin_family = AF_INET;
        serveraddr.sin_port = htons(PORT);
        /* INADDR_ANY表示不管是哪个网卡接收到数据，只要目的端口是SERV_PORT，就会被该应用程序接收到 */
        serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);


        /* 绑定socket */
        if(bind(sockfd, (struct sockaddr *)&serveraddr, sizeof(serveraddr)) < 0)
        {
            perror("bind error:");
            exit(1);
        }


        while (1)
        {
    //        if(fork()>0){
    //            //父进程
    //            break;
    //        }
            recvfrom(sockfd, buf, 128, 0, (struct sockaddr*)&clientaddr, &len);
            printf("client ip: %s, client port: %d\n", inet_ntoa(clientaddr.sin_addr), ntohs(clientaddr.sin_port));
            for(int i=0;i<128&&buf[i]!='\0';++i){
                if(buf[i]=='#'){
                    strncpy(buf1,buf,i);
                    buf[i] = '\0';
                    strncpy(buf2,buf+i+1,127-i); //0 1 2 3
                    break;
                }
            }
//            *x = strtod(buf1,NULL);
//            *y = strtod(buf2,NULL);
            ofstream outfile("xy.txt",ios::out);
            outfile<<strtod(buf1,NULL)<<"\n"<<strtod(buf2,NULL)<<endl;
            outfile.close();
        }

        close(sockfd);
    }
}

double UDPrecv::getx(){
    ifstream fin("xy.txt",ios::in);
    fin.getline(strx,128);
    fin.close();
    return strtod(strx,nullptr);
}


double UDPrecv::gety(){
    ifstream fin("xy.txt",ios::in);
    fin.getline(strx,128);
    fin.getline(stry,128);
    fin.close();
    return strtod(stry,nullptr);
}
