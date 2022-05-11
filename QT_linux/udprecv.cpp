#include "udprecv.h"
#include <signal.h>
UDPrecv::UDPrecv()
{
    pipe(fd);
    pid_t pid = fork();
    if(pid>0){
        //parent process
        close(fd[1]);
        return;
    }
    else if(pid == 0){
        //sub process
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


        while (true){
            recvfrom(sockfd, buf, 128, 0, (struct sockaddr*)&clientaddr, &len);
//            printf("client ip: %s, client port: %d\n", inet_ntoa(clientaddr.sin_addr), ntohs(clientaddr.sin_port));
            write(fd[1],buf,BUF_SIZE);
        }
        close(sockfd);
    }
}

void UDPrecv::recvxy(){
    //setting pipe to none blocking
    int ret = fcntl(fd[0],F_GETFL);
    ret |=O_NONBLOCK;
    fcntl(fd[0],F_SETFL,ret);
    //recording whether there is something need to return
    re = true;
    while((ret = read(fd[0],buflast,BUF_SIZE))>0){
        memcpy(buf,buflast,BUF_SIZE);
        re = false;
    }

    if(re){//nothing need to update
        return;
    }
    //calculate xy by string
    //string x#y#
//modify the method to calculate xy to make sure we will get right answer
    for(int i=0;i<BUF_SIZE&&buf[i]!='\0';++i){
        if(buf[i]=='#'){
            strncpy(buf1,buf,i);
            buf1[i] = '\0';
            for(int j = i+1;j<BUF_SIZE&&buf[j]!='\0';++j){
                if(buf[j]=='#'){
                    buf2[j-1] = '\0';
                    strncpy(buf2,buf+i+1,j-i-1);
                }
            }
            break;
        }
    }
    //strtod: string to double
    x = strtod(buf1,NULL);
    y = strtod(buf2,NULL);
    if(x>800) cout<<buf<<endl;
}
