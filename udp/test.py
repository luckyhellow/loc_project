import random
import socket
import time
BUFSIZE = 1024


client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
PORT = 8888


sleeptime = 0.01
date = ''
dis = '100#300'
recv = "start"#套接字编程控制 可以调节频率
count = 0

while recv!="end":
    count += 1
    x = random.randint(1,1000)
    y = random.randint(1,800)
    if count==3:
        dis = "label1#" + str(x)+"#"+str(y)+"#"
    elif count == 6:
        dis = "label2#" + str(x)+"#"+str(y)+"#"
        count = 0
    else:
        dis = "label3#" + str(x)+"#"+str(y)+"#"
    if dis!='':
        server_address = ("127.0.0.1", PORT)  # 接收方 服务器的ip地址和端口号
        client_socket.sendto(dis.encode(), server_address) #将msg内容发送给指定接收方
        print("send success",dis)
    time.sleep(sleeptime)
