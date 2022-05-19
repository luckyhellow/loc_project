import os
import socket
import sys
import time


BUFSIZE = 1024
PORT_recv = 7777
PORT_send = 8888
# IP_send = "192.168.3.169"
IP_send = '127.0.0.1'
ip_port_server = ('127.0.0.1', PORT_recv)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind(ip_port_server)

sleeptime = 0.1
bs_8 = []
bs_373 = []
tdoa_8 = []
tdoa_373 = []


def Recv_tdoa_data(fd0, fd1):
    os.close(fd0)
    while True:
        data,server_addr = server_socket.recvfrom(BUFSIZE)
        if data!='':
            # data_str = data.decode()
            #print(data_str)
            # print("Send message through Pipe...wait a moment!")
            try:
                #print ("child: writing...")
                os.write(fd1,data)
            except Exception as e:
                print(e)



def Send_loc_xy(fd0, fd1):
    os.close(fd1)
    while True:
        # print("calculate by function and send by udpconnect...")
        try:
            txt = os.read(fd0, 256)
            txt = txt.decode()
            print( 'got   %s      at time : %s' % (txt, time.time( )) )
        except Exception as e:
            print(e)



if __name__ == '__main__':
    fd0 , fd1 = os.pipe()
    pid = os.fork()
    if pid:
        # father process
        Send_loc_xy(fd0, fd1)
    else:
        #sub process         
        Recv_tdoa_data(fd0, fd1)