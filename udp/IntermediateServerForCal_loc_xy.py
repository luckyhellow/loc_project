import os
import socket
import sys
import time
from Algorithm.Process import CAL_Loc
import Algorithm.LA_class


BUFSIZE = 1024
PORT_recv = 7777
PORT_send = 8888
# IP_send = '172.27.159.203'
IP_send = '192.168.107.163'
# IP_send = '127.0.0.1'

sleeptime = 0.1


def Recv_tdoa_data(fd0, fd1):
    ip_port_recv = ('127.0.0.1', PORT_recv)
    recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_socket.bind(ip_port_recv)

    os.close(fd0)
    while True:
        data,server_addr = recv_socket.recvfrom(BUFSIZE)
        if data!='':
            # data_str = data.decode()
            #print(data_str)
            # print("Send message through Pipe...wait a moment!")
            try:
                os.write(fd1,data)
            except Exception as e:
                print(e)
        # time.sleep(sleeptime)



def Send_loc_xy(fd0, fd1):
    send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    bs_8 = []
    bs_373 = []
    tdoa_8 = []
    tdoa_373 = []

    os.close(fd1)
    while True:
        DATALIST = []
        # print("calculate by function and send by udpconnect...")
        try:
            line = os.read(fd0, 256)
            line = line.decode()
            if(line!=""):
                DATALIST.append(line.split("#"))
            #print( 'got   %s      at time : %s' % (line, time.time( )) )
            #print(DATALIST)
        except Exception as e:
            print(e)

        if(DATALIST!=[]):
            Loc_xy,tmp1,tmp2,tmp3,tmp4 = CAL_Loc(DATALIST,bs_8,bs_373,tdoa_8,tdoa_373)
            bs_8 = tmp1
            bs_373 = tmp2
            tdoa_8 = tmp3
            tdoa_373 = tmp4

            print("calculate by function and send by udpconnect...")
            for xy_tuple in Loc_xy:
                #print("send to PORT:",PORT_send)
                #print("send to IP:",IP_send)
                SendTo_QT_address = (IP_send, PORT_send)
                for i in xy_tuple:
                    print(i.encode())
                    send_socket.sendto(i.encode(), SendTo_QT_address)
                
        


if __name__ == '__main__':
    fd0 , fd1 = os.pipe()
    pid = os.fork()
    if pid:
        # father process
        Send_loc_xy(fd0, fd1)
    else:
        #sub process         
        Recv_tdoa_data(fd0, fd1)