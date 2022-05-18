import numpy as np
import os
import pandas as pd
import socket
#import serial
import time
from Algorithm.LA_class import LocationAlgorithmAgent


Coordinate = {
                '842E':(2285.4,1480.4,212.4) , '840F':(1418.6,0,210),
                '8410':(186.1,0,210)         , '8415':(2584.6,10.7,240),
                '8423':(1708.4,625.9,207)    , '8429':(908.3,558,210),
                '8430':(1763,2434,210)       , '8428':(86,1465,206),
                '8432':(126,2960,210)        , '8412':(745,2400,210),
                '8437':(2580,2960,240)       , '841A':(1452,2960,210),
                '8400':(1371.8,154.4,210)    , '8404':(212.7,2994,210),
                '8406':(2343.2,1345.1,210)   , '840E':(2333.2,665,153.9),
                '8413':(2257.5,969,210)      , '841D':(2257.5,2139.6,210),
                '8425':(3167.9,0,210)        , '842B':(124.6,154.4,210),
                '8438':(2237.6,0,210)        , '8460':(2343.2,2119.3,155.3),
                '8494':(2257.5,1577.2,210)   , '84DC':(3097.6,680.6,210),
                '841F':(434.3,0,210)         , '44F2':(976.4,2536.6,210),
                '847F':(3113.3,2943.6,210)   , '84BF':(3110.3,2247.3,155)
    }

BS_ID_List = ['842E' , '840F' , '8410' , '8415' , '8423' , '8429' , '8430' , '8428' ,
              '8432' , '8412' , '8437' , '841A' , '8400' , '8404' , '8406' , '840E' ,
              '8413' , '841D' , '8425' , '842B' , '8438' , '8460' , '8494' , '84DC' ,
              '841F' , '44F2' , '847F' , '84BF' ]

def Get_raw_data():
    base_path = "./DATA/"
    #files = os.listdir(base_path)
    #for file in files:
    row_list = []
    file = 'raw.xlsx'
    df = pd.read_excel(io=base_path + file,index_col=None,header=None)
    ROWNUM = 100
    #ROWNUM = df.shape[0]
    for i in range (ROWNUM):
        if(df.iloc[i,1] == '$TAG'):
            row_list.append(i)
            print(df.iloc[i,:])
    df_csv = df.iloc[row_list,:]
    csv_file = "Process_raw.csv"
    df_csv.to_csv(base_path + csv_file , encoding="utf-8-sig", mode="a", header=False, index=False)
    #print(df_csv)   


def cal_loc(bs_8,bs_373,tdoa_8,tdoa_373):
    TRX = [bs_8,bs_373]
    TRD = [tdoa_8,tdoa_373]
    teste = 1e-2
    RetX = []
    RetY = []
    flag = 0   #flag表示有节
    for i in range(2):
        testx = np.array(TRX[i])
        testd = np.array(TRD[i][1:])
        LAA = LocationAlgorithmAgent( x=testx,d=testd,e=teste)
        try:
            Loc = LAA.loc_ls()
        except ValueError:
            #print("Something wrong")
            flag += (i+1)
            RetX.append(0)
            RetY.append(0)
        else:
            RetX.append(Loc[0])
            RetY.append(Loc[1])

    #return flag,RetX[0],RetY[0],RetX[1],RetY[1]
    return flag,RetX[0]/10,RetY[0]/5,RetX[1]/10,RetY[1]/5

def Get_TDOA_data():
    base_path = "./DATA/"
    file = 'new_tdoa.csv'
    #df = pd.read_excel(io=base_path + file,index_col=None,header=None)
    df = pd.read_csv(base_path + file,index_col=None,header=None)
    ROWNUM = df.shape[0]
    bs_8 = []
    bs_373 = []
    tdoa_8 = []
    tdoa_373 = []
    index = 0 
    tail = 0
    Str_Loc1 = ''
    Str_Loc2 = ''

    #ip_port_client = ('127.0.0.1', 9696)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #client_socket.bind(ip_port_client)
    PORT = 8888
    sleeptime = 0.5

    while index<ROWNUM:
        if(index+1<ROWNUM and df.iloc[index,0]==df.iloc[index+1,0]):
            tail = index+2
        else:
            tail = index+1
        for i in range(index,tail):
            bs_num = df.iloc[i,2]
            tmp1 = []
            tmp2 = []
            for j in range(bs_num):
                tmp1.append(list(Coordinate[str(df.iloc[i,3+j])]))
                tmp2.append(float(df.iloc[i,3+bs_num+j]))
            if(df.iloc[i,1]==8):
                bs_8 = tmp1
                tdoa_8 = tmp2
            if(df.iloc[i,1]==373):
                bs_373 = tmp1
                tdoa_373 = tmp2
        index = tail
        flag,x1,y1,x2,y2 = cal_loc(bs_8,bs_373,tdoa_8,tdoa_373)
        if(flag<2):
            Str_Loc2=str(x2)+'#'+str(y2)
        if(flag%2==0):
            Str_Loc1=str(x1)+'#'+str(y1)
        if(flag>0):
            print(flag)
        #Loc = [Str_Loc1,Str_Loc2]
        Loc = [Str_Loc1]
        '''for i in Loc:
            print(i)
        time.sleep(sleeptime)
        print('+++++++++++++++++++++++++++++++++++++++++++++++++')'''
        for i in Loc:
            if i!='':
                #server_address = ("127.0.0.1", PORT)  # 接收方 服务器的ip地址和端口号
                server_address = ("192.168.3.169", PORT)  # 接收方 服务器的ip地址和端口号
                client_socket.sendto(i.encode(), server_address) #将msg内容发送给指定接收方
                print(i)
                print('Send Sucess')
                time.sleep(sleeptime)
        #serial.close()    
        
    client_socket.close()



    #csv_file = "Process_tdoa.csv"
    #df_csv.to_csv(base_path + csv_file , encoding="utf-8-sig", mode="a", header=False, index=False)


def CAL_Loc(data_list,bs_8,bs_373,tdoa_8,tdoa_373):

    Len = len(data_list)
    index = 0 
    tail = 0
    Str_Loc1 = ''
    Str_Loc2 = ''
    ret = []

    while index<Len:
        if(index+1<Len and data_list[index][0]==data_list[index][0]):
            tail = index+2
        else:
            tail = index+1
        for i in range(index,tail):
            bs_num = int(data_list[index][2])
            fflag = 0
            for j in range(bs_num):
                if(data_list[i][3+j] not in BS_ID_List):
                    fflag = 1
                    break
            if(fflag == 1):
                continue
            tmp1 = []
            tmp2 = []
            for j in range(bs_num):
                tmp1.append(list(Coordinate[data_list[i][3+j]]))
                tmp2.append(float(data_list[i][3+bs_num+j]))
            if(data_list[i][1]=="8"):
                bs_8 = tmp1
                tdoa_8 = tmp2
            if(data_list[i][1]=="373"):
                bs_373 = tmp1
                tdoa_373 = tmp2
        index = tail
        flag,x1,y1,x2,y2 = cal_loc(bs_8,bs_373,tdoa_8,tdoa_373)
        if(flag<2):
            Str_Loc2=str(x2)+'#'+str(y2)+'#'
        if(flag%2==0):
            Str_Loc1=str(x1)+'#'+str(y1)+'#'
        if(flag>0):
            #print(flag)
            pass
        #ret.append((Str_Loc1,Str_Loc2))
        ret.append(Str_Loc1)
    
    return ret,bs_8,bs_373,tdoa_8,tdoa_373
        


if __name__ == '__main__':
    #Get_raw_data()
    #Get_TDOA_data()
    pass