import socket
import os
import time
from Algorithm.Process import CAL_Loc
import Algorithm.LA_class
BUFSIZE = 1024

PORT_recv = 7777
PORT_send = 8888
# IP_send = "192.168.3.169"
IP_send = '127.0.0.1'
ip_port_server = ('127.0.0.1', PORT_recv)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind(ip_port_server)
txt_path = "tdoa.txt"

data = ""
sleeptime = 0.1

bs_8 = []
bs_373 = []
tdoa_8 = []
tdoa_373 = []

pid = os.fork()

if pid == 0 :
    #sub process
    while True:
        data,server_addr = server_socket.recvfrom(BUFSIZE)
        if data!='':
            data_str = data.decode()
            print(data_str)
            print("writing into txt...wait a moment!")
            file = open(txt_path,'w+')
            file.write(data_str+"\n")
            file.close()

else :
    # for i in range(12):
        # print(i)
    while True:
        DATALIST = []
        with open(txt_path,"r") as f:
            for line in f:
                if(line!=""):
                    DATALIST.append(line.split("#"))


        if(DATALIST!=[]):
            Loc_xy,bs_8,bs_373,tdoa_8,tdoa_373 = CAL_Loc(DATALIST,bs_8,bs_373,tdoa_8,tdoa_373)

            print("calculate by function and send by udpconnect...")
            for i in Loc_xy:
                print("send to PORT:",PORT_send)
                print("send to IP:",IP_send)
                SendTo_QT_address = (IP_send, PORT_send)
                server_socket.sendto(i.encode(), SendTo_QT_address)
            time.sleep(sleeptime)