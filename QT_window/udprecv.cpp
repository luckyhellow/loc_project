#include "udprecv.h"
#include <signal.h>
UDPrecv::UDPrecv(){
    //simple init
    WSADATA ws;
    WSAStartup(MAKEWORD(2, 2), &ws);

    time_t rawtime = 0;
    time.time_s = *localtime( &rawtime );
    time.time_ms = 0;

    pthread_mutex_init(&mtx, NULL);

    pthread_t thread_id;
    int ret = pthread_create(&thread_id, NULL, recvxy, (void*)this);
    if (ret == 0)
    {
         pthread_detach(thread_id);
    }
    cout<<"creat thread!"<<endl;
}

void* UDPrecv::recvxy(void* args){
    UDPrecv* This = (UDPrecv*)args;
    struct sockaddr_in serveraddr, clientaddr;
    /* socket文件描述符 */
    int sockfd;
    char BUF[BUF_SIZE];

    /* 建立udp socket */
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if(sockfd < 0)
    {
        perror("socket");
        exit(1);
    }

    /* 设置address */
    int len = sizeof(struct sockaddr_in);

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
        recvfrom(sockfd, BUF, BUF_SIZE, 0, (struct sockaddr*)&clientaddr, &len);
//            printf("client ip: %s, client port: %d\n", inet_ntoa(clientaddr.sin_addr), ntohs(clientaddr.sin_port));
        //lock
        pthread_mutex_lock(&This->mtx);
        strncpy(This->buf,BUF,BUF_SIZE);
        //unlock
        pthread_mutex_unlock(&This->mtx);
    }
    close(sockfd);
}

Struct_XY UDPrecv::getxy(){
    //calculate xy by string
    //string x#y#
    //modify the method to calculate xy to make sure we will get right answer
    pthread_mutex_lock(&mtx);
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
            return Struct_XY{label,time,x,y};
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
    pthread_mutex_unlock(&mtx);
    label = buflabel;
    //strtod: string to double
    x = strtod(buf1,NULL);
    y = strtod(buf2,NULL);
//    cout<<label<<" "<<x<<" "<<y<<endl;
    return Struct_XY{label,time,x,y};
}
