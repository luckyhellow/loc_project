import socket
import os
import time
BUFSIZE = 1024

PORT_recv = 7777
PORT_send = 8888
IP_send = "TODO"
ip_port_server = ('127.0.0.1', PORT_recv)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind(ip_port_server)

data = ""
sleeptime = 0.1

pid = os.fork()

if pid == 0 :
    while True:
        data,server_addr = server_socket.recvfrom(BUFSIZE)
        if data!='':
            print(data.decode())
            print("writing into txt...wait a moment!")

else :
    print("calculate by function and send by udpconnect...")
    print("send to PORT:",PORT_send)
    print("send to IP:",IP_send)
    time.sleep(sleeptime)