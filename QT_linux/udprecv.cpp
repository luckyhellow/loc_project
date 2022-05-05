#include "udprecv.h"
#include <signal.h>
UDPrecv::UDPrecv()
{
    pipe(fd);
    pid_t pid = fork();
    if(pid>0){
        //parent
        close(fd[1]);
        return;
    }
    else if(pid == 0){
        close(fd[0]);

        struct sockaddr_in serveraddr, clientaddr;

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
            recvfrom(sockfd, buf, 128, 0, (struct sockaddr*)&clientaddr, &len);
//            printf("client ip: %s, client port: %d\n", inet_ntoa(clientaddr.sin_addr), ntohs(clientaddr.sin_port));
            write(fd[1],buf,sizeof(buf));
        }

        close(sockfd);
    }
}

void UDPrecv::recvxy(){
    memset(buf,0,sizeof(buf));
    int ret = read(fd[0],buf,sizeof(buf));
    if(ret == 0){
        cout<<"get nothing!\n";
        return;
    }
    else if(ret == -1){
        cout<<"something error!!!";
        return;
    }
//    else if(ret > 0){
//        write(STDOUT_FILENO,buf,ret);
//        cout<<ret<<endl;
//        cout<<buf[0]<<endl;
//    }
    for(int i=0;i<128&&buf[i]!='\0';++i){
        if(buf[i]=='#'){
            strncpy(buf1,buf,i);
            buf[i] = '\0';
            strncpy(buf2,buf+i+1,127-i); //0 1 2 3
            break;
        }
    }
    x = strtod(buf1,NULL);
    y = strtod(buf2,NULL);
//    cout<<x<<endl;
//    cout<<y<<endl;
}
