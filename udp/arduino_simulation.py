import numpy as np
import os
import pandas as pd
import socket

from time import sleep

BUFSIZE = 1024

#创建套节字
# local_address = ('127.0.0.1', 9997)
send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #创建udp套节字以发送数据
# client_socket.bind(local_address)

PORT = 7777

# target_address = ("192.168.3.180", PORT)  # 接收方的ip地址和端口号
target_address = ("127.0.0.1", PORT)  # 接收方的ip地址和端口号

sleeptime = 0.2

if __name__ == '__main__':
    base_path = "data/" #dirpath
    file = 'tdoa.xlsx'
    # xlsx
    if file[-4:]=="xlsx":
        df = pd.read_excel(io=base_path + file,index_col=None,header=None)
    # csv
    elif file[-3:]=="csv":
        df = pd.read_csv(base_path + file,index_col=None,header=None)
    while(True):
        for i in range(df.shape[0]):
            str_to_send = "#".join([str(j) for j in (df.iloc[i]) ])
            str_to_send = str_to_send[0:str_to_send.find("nan")] #截取字符串后边为“NaN”的部分
            # print(str_to_send)
            send_socket.sendto(str_to_send.encode(), target_address) #将msg内容发送给指定接收方
            print("send success content:",str_to_send)
            sleep(sleeptime)
