#include "udprecv.h"
#include <signal.h>
UDPrecv::UDPrecv(){
    //simple init
    time_t rawtime = 0;
    time.time_s = *localtime( &rawtime );
    time.time_ms = 0;

    if(pipe(fd)==-1){
        return;
    }
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
    int st = 0;
    for(int i=0;i<BUF_SIZE&&buf[i]!='\0';++i){
        if(buf[i]=='#'){
            //label
            strncpy(buflabel,buf,i);
            buflabel[i] = '\0';
            st = i+1;
            break;
        }
    }
    //get time
        time.time_s.tm_year = (buf[st]-'0')*1000+(buf[st+1]-'0')*100+(buf[st+2]-'0')*10+buf[st+3]-'0';
        st+=4;
        time.time_s.tm_mon = (buf[st]-'0')*10+buf[st+1]-'0';
        st+=2;
        time.time_s.tm_mday = (buf[st]-'0')*10+buf[st+1]-'0';
        st+=2;
        time.time_s.tm_hour = (buf[st]-'0')*10+buf[st+1]-'0';
        st+=2;
        time.time_s.tm_min = (buf[st]-'0')*10+buf[st+1]-'0';
        st+=2;
        time.time_s.tm_sec = (buf[st]-'0')*10+buf[st+1]-'0';
        st+=2;
        time.time_ms = (buf[st]-'0')*100+(buf[st+1]-'0')*10+(buf[st+2]-'0');
        st+=3;
        if(buf[st]!='#') {
//            cout<<"something wrong with time";
            return;
        }
        ++st;

    for(int i = st;i<BUF_SIZE&&buf[i]!='\0';++i){
        if(buf[i]=='#'){
            //value of x
            strncpy(buf1,buf+st,i-st);
            buf1[i-st] = '\0';
            st = i+1;
            break;
        }
    }
    for(int i = st;i<BUF_SIZE&&buf[i]!='\0';++i){
        if(buf[i]=='#'){
            //value of y
            strncpy(buf2,buf+st,i-st);
            buf2[i-st] = '\0';
            break;
        }
    }
    //strtod: string to double
    label = buflabel;
    x = strtod(buf1,NULL);
    y = strtod(buf2,NULL);
//    cout<<label<<" "<<x<<" "<<y<<endl;
}
