import numpy as np
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
import socket
import serial 
import time

BUFSIZE = 1024

class LocationAlgorithmAgent(object):
    def __init__(self, x,d,z=0,e=1e-3,n_iter=1000, threshold=1e-6):
        self.x = x
        self.d = d
        self.z = z
        self.e = e
        self.n_iter = n_iter
        self.threshold = threshold


    def loc_ls(self):
        n,m = np.shape(self.x)[0] - 1, np.shape(self.x)[1]
        A = np.hstack( [np.tile(self.x[0,:2],(n,1))-self.x[1:,:2] , self.d.reshape((n,1))] )
        D = np.sum(self.x**2, axis=1)
        b = .5 * (self.d**2+D[0]-D[1:]).T + (self.x[1:,2]-self.x[0, 2]) * self.z
        A = matrix(A)
        b = matrix(b)
        sol = solvers.qp(A.T*A, -A.T*b)
        return np.array(sol['x'])[:2].reshape(-1)
    

    def loc_sdpi(self):
        n,m = np.shape(self.x)[0]-1 , np.shape(self.x)[1]
        Gy = []
        for i in range(m):
            Gy.append(np.zeros((n*2+m*2+5 , n*2+m*2+5)))
        Dict_key = ['Gys','Gds','Gdr','Gt','h']
        Array = dict.fromkeys( Dict_key, np.zeros((n*2+m*2+5 , n*2+m*2+5)) )
        for KEY in Dict_key:
            Array[KEY] = np.zeros((n*2+m*2+5 , n*2+m*2+5))
        
        for i in range(n):
            xi2 = 0
            for j in range(m):
                Gy[j][i,i] = 2 * self.x[i+1,j]
                Gy[j][i+n,i+n] = -2 * self.x[i+1,j]
                xi2 += self.x[i+1,j] ** 2
            tmp_value = [-1.,1.,1.,-1.,2*self.d[i],- 2 * self.d[i],-1.,-1.,xi2-self.d[i]**2,-xi2+self.d[i]**2]
            for k in range(5):
                Array[Dict_key[k]][i,i] = tmp_value[k*2]
                Array[Dict_key[k]][i+n,i+n] = tmp_value[k*2+1]

        for j in range(m):
            Gy[j][n*2+j,n*2+m] = -1.
            Gy[j][n*2+m,n*2+j] = -1.
            Array['h'][n*2+j,n*2+j] = 1
        Array['Gys'][n*2+m,n*2+m] = -1.
        Array['Gds'][n*2+m+2,n*2+m+2] = -1.
        Array['Gdr'][n*2+m+1,n*2+m+2] = -1.
        Array['Gdr'][n*2+m+2,n*2+m+1] = -1.
        Array['h'][n*2+m+1,n*2+m+1] = 1.

        G = []
        for j in range(m):
            G.append(Gy[j].reshape(-1, 1))
        for k in range(4):
            G.append(Array[Dict_key[k]].reshape(-1,1))

        c = np.zeros((m+4,1))
        c[m+3,0] = 1.
        A = np.zeros((2,m+4))
        x02 = 0
        for j in range(m):
            A[0,j] = 2 * self.x[0,j]
            x02 += self.x[0,j] ** 2
        A[0,m] , A[0,m+1] , A[1,2] = -1. ,  1. , 1
        b= np.zeros((2,1))
        b[0] , b[1] = x02 , self.z

        c = matrix(c)
        G = [matrix(np.hstack(G))]
        h = [matrix(Array['h'])]
        A = matrix(A)
        b = matrix(b)
        sol = solvers.sdp(c, Gs=G, hs=h, A=A, b=b)
        return np.array(sol['x'])[:2].reshape(-1)


    def loc_sdpc(self):
        n,m = np.shape(self.x)[0]-1 , np.shape(self.x)[1]
        ARRAY = {'Gyc':[],'Gu':[],'Gy':[]}
        NAME = ['Gu','Gy','Gyc']
        for i in range(m):
            for j in NAME:
                ARRAY[j].append(np.zeros((n*2+m*2+5,n*2+m*2+5)))
        Dict_key = ['Gys','Gds','Gdr','Gt','h']
        Array = dict.fromkeys( Dict_key, np.zeros((n*2+m*2+5 , n*2+m*2+5)) )
        for KEY in Dict_key:
            Array[KEY] = np.zeros((n*2+m*2+5 , n*2+m*2+5))

        for i in range(n):
            xi2 = 0
            for j in range(m):
                ARRAY['Gy'][j][i,i] = 2 * self.x[i+1,j]
                ARRAY['Gy'][j][i+n,i+n] = -2 * self.x[i+1,j]
                xi2 += self.x[i+1, j] ** 2
            tmp_value = [-1.,1.,1.,-1.,2*(self.d[i]-self.e),-2*(self.d[i]+self.e),xi2-(self.d[i]-self.e)**2,-xi2+(self.d[i]+self.e)**2]
            tmp_key = ['Gys','Gds','Gdr','h']
            for k in range(4):
                Array[tmp_key[k]][i,i] = tmp_value[k*2]
                Array[tmp_key[k]][i+n,i+n] = tmp_value[k*2+1]

        for j in range(m):
            tmp_index = [n*2+m+3+j , n*2+m*2+3 , n*2+j , n*2+m]
            for k in range(2):
                ARRAY[NAME[k]][j][ tmp_index[k*2],tmp_index[k*2+1] ] = -1.
                ARRAY[NAME[k]][j][ tmp_index[k*2+1],tmp_index[k*2] ] = -1.
            Array['Gt'][n*2+m+3+j,n*2+m+3+j] = -1.
            Array['h'][n*2+j,n*2+j] = 1
        Array['Gys'][n*2+m,n*2+m] = -1.
        Array['Gds'][n*2+m+2,n*2+m+2] = -1.
        Array['Gdr'][n*2+m+1,n*2+m+2] = -1.
        Array['Gdr'][n*2+m+2,n*2+m+1] = -1.
        Array['Gt'][n*2+m*2+3,n*2+m*2+3] = -1.
        Array['Gt'][n*2+m*2+4,n*2+m*2+4] = -1.
        Array['h'][n*2+m+1,n*2+m+1] = 1.

        G = []
        tmp_Name = ['Gyc','Gu','Gy']
        for i in tmp_Name:
            for j in range(m):
                G.append(ARRAY[i][j].reshape(-1, 1))
        for k in range(4):
            G.append(Array[Dict_key[k]].reshape(-1,1))

        c = np.zeros((m*3+4,1))
        c[m*3+3,0] = 1.
        A = np.zeros((m+2,m*3+4))
        x02 = 0
        for j in range(m):
            A[j+1,j] = A[j+1,m+j] = 1.
            A[0,m*2+j] = 2 * self.x[0,j]
            A[j+1,m*2+j] = -1.
            x02 += self.x[0,j] ** 2
        A[0,m*3] = -1.
        A[0,m*3+1] = 1.
        A[-1,2] = 1
        b = np.zeros((m+2,1))
        b[0] = x02
        b[-1] = self.z

        c = matrix(c)
        G = [matrix(np.hstack(G))]
        h = [matrix(Array['h'])]
        A = matrix(A)
        b = matrix(b)
        sol = solvers.sdp(c, Gs=G, hs=h, A=A, b=b)
        return np.array(sol['x'])[:2].reshape(-1)
    

    def loc_sdpr(self):
        n,m = np.shape(self.x)[0] - 1, np.shape(self.x)[1]
        s = np.sum(self.x ** 2, axis=1)
        vc1, vc2, ck1, ck2, ct1, ct2, mc1, mc2, mq, vs = [], [], [], [], [], [], [], [], [], []
        for i in range(n):
            ca = np.zeros((n + m, 1))
            ca[:m] = 2 * (self.x[0] - self.x[i + 1]).reshape((-1, 1))
            ca[m + i] = - 2 * self.d[i]
            cb = s[0] - s[i + 1] - self.d[i] ** 2
            cv = np.zeros((n + m, 1))
            cv[m + i] = 1
            vc1.append(ca + 2 * self.e * cv)
            vc2.append(ca - 2 * self.e * cv)
            ck1.append(self.e ** 2 + 2 * self.d[i] * self.e + cb)
            ck2.append(self.e ** 2 + 2 * self.d[i] * self.e - cb)
            ct1.append(self.e ** 2 - 2 * self.d[i] * self.e + cb)
            ct2.append(self.e ** 2 - 2 * self.d[i] * self.e - cb)
            mc1.append(vc1[i] * vc1[i].T)
            mc2.append(vc2[i] * vc2[i].T)
            mq.append(np.zeros((m + n, m + n)))
            mq[i][:m, :m] = np.eye(m)
            mq[i][m + i, m + i] = -1
            vs.append(np.zeros((m + n, 1)))
            vs[i][:m] = self.x[i + 1, :].reshape((-1, 1))

        c = np.zeros((int((n + m + 1) * (n + m) / 2) + n + m + n, 1))
        c[-n:] = np.ones((n, 1))
        A = np.zeros((1, int((n + m + 1) * (n + m) / 2) + n + m + n))
        A[0, 2] = 1
        b = np.zeros((1, 1))
        b[0] = self.z
        size = n * 6 + n + m + 1
        size += (m + 1) * n + n
        size = (size, size)
        h = np.zeros(size)
        Gy, Gz, Gt = [np.zeros(size) for _ in range(n + m)], [np.zeros(size) for _ in range((n + m) * (n + m))], [np.zeros(size) for _ in range(n)]
        for i in range(n):
            Gt[i][i + n * 0, i + n * 0] = - 1
            Gt[i][i + n * 1, i + n * 1] = - 1
            Gt[i][i + n * 2, i + n * 2] = - 1
            Gt[i][i + n * 3, i + n * 3] = - 1
            h[i + n * 0, i + n * 0] = 0 - ck2[i] ** 2
            h[i + n * 1, i + n * 1] = 0 - ct2[i] ** 2
            h[i + n * 2, i + n * 2] = 0 - ck1[i] ** 2
            h[i + n * 3, i + n * 3] = 0 - ct1[i] ** 2
            h[i + n * 4, i + n * 4] = 0 + s[i + 1]
            h[i + n * 5, i + n * 5] = 0 - s[i + 1]
            for j in range(n + m):
                Gy[j][i + n * 0, i + n * 0] = 0 + 2 * ck2[i] * vc2[i][j]
                Gy[j][i + n * 1, i + n * 1] = 0 + 2 * ct2[i] * vc1[i][j]
                Gy[j][i + n * 2, i + n * 2] = 0 - 2 * ck1[i] * vc1[i][j]
                Gy[j][i + n * 3, i + n * 3] = 0 - 2 * ct1[i] * vc2[i][j]
                Gy[j][i + n * 4, i + n * 4] = 0 + 2 * vs[i][j]
                Gy[j][i + n * 5, i + n * 5] = 0 - 2 * vs[i][j]
                for k in range(j):
                    Gz[j * (n + m) + k][i + n * 0, i + n * 0] = 0 + 2 * mc2[i][k, j]
                    Gz[j * (n + m) + k][i + n * 1, i + n * 1] = 0 + 2 * mc1[i][k, j]
                    Gz[j * (n + m) + k][i + n * 2, i + n * 2] = 0 + 2 * mc1[i][k, j]
                    Gz[j * (n + m) + k][i + n * 3, i + n * 3] = 0 + 2 * mc2[i][k, j]
                    Gz[j * (n + m) + k][i + n * 4, i + n * 4] = 0 - 2 * mq[i][k, j]
                    Gz[j * (n + m) + k][i + n * 5, i + n * 5] = 0 + 2 * mq[i][k, j]
                Gz[j * (n + m) + j][i + n * 0, i + n * 0] = 0 + mc2[i][j, j]
                Gz[j * (n + m) + j][i + n * 1, i + n * 1] = 0 + mc1[i][j, j]
                Gz[j * (n + m) + j][i + n * 2, i + n * 2] = 0 + mc1[i][j, j]
                Gz[j * (n + m) + j][i + n * 3, i + n * 3] = 0 + mc2[i][j, j]
                Gz[j * (n + m) + j][i + n * 4, i + n * 4] = 0 - mq[i][j, j]
                Gz[j * (n + m) + j][i + n * 5, i + n * 5] = 0 + mq[i][j, j]
            for j in range(m):
                Gy[j][n * 6 + n + m + 1 + i * (m + 1) + j, n * 6 + n + m + 1 + i * (m + 1) + m] = -1
                Gy[j][n * 6 + n + m + 1 + i * (m + 1) + m, n * 6 + n + m + 1 + i * (m + 1) + j] = -1
                Gy[m + i][n * 6 + n + m + 1 + i * (m + 1) + j, n * 6 + n + m + 1 + i * (m + 1) + j] = -1
                h[n * 6 + n + m + 1 + i * (m + 1) + j, n * 6 + n + m + 1 + i * (m + 1) + m] = - self.x[i + 1, j]
                h[n * 6 + n + m + 1 + i * (m + 1) + m, n * 6 + n + m + 1 + i * (m + 1) + j] = - self.x[i + 1, j]
            Gy[m + i][n * 6 + n + m + 1 + i * (m + 1) + m, n * 6 + n + m + 1 + i * (m + 1) + m] = -1
            Gy[m + i][n * 6 + n + m + 1 + n * (m + 1) + i, n * 6 + n + m + 1 + n * (m + 1) + i] = -1
        for j in range(n + m):
            Gy[j][n * 6 + j, n * 6 + n + m] = -1
            Gy[j][n * 6 + n + m, n * 6 + j] = -1
            for k in range(j):
                Gz[j * (n + m) + k][n * 6 + j, n * 6 + k] = -1
                Gz[j * (n + m) + k][n * 6 + k, n * 6 + j] = -1
            Gz[j * (n + m) + j][n * 6 + j, n * 6 + j] = -1
        h[n * 6 + n + m, n * 6 + n + m] = 1
        G = []
        for i in range(n + m):
            G.append(Gy[i].reshape((-1, 1)))
        for i in range(n + m):
            for j in range(i + 1):
                G.append(Gz[i * (n + m) + j].reshape((-1, 1)))
        for i in range(n):
            G.append(Gt[i].reshape((-1, 1)))
        c = matrix(c)
        G = [matrix(np.hstack(G))]
        h = [matrix(h)]
        A = matrix(A)
        b = matrix(b)
        sol = solvers.sdp(c, Gs=G, hs=h, A=A, b=b)
        return np.array(sol['x'])[:2].reshape(-1)


    def loc_rms(self):
        x = self.x
        tdoa = self.d
        h = self.z
        sigma = self.e
        n_iter = self.n_iter
        threshold = self.threshold
        n,m = np.shape(self.x)[0]-1 , np.shape(self.x)[1]
        Q = .5 * (np.ones((n, n)) + np.identity(n)) * sigma ** 2
        K = np.sum((x - np.array([0, 0, h])) ** 2, axis=1)
        E = np.reshape(.5 * (tdoa ** 2 + K[0] - K[1:]), (-1, 1))
        G = 0 - np.hstack([x[0, :2] - x[1:, :2], np.reshape(tdoa, (-1, 1))])
        Q_inv = np.linalg.inv(Q)
        Z_ = np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul(G.T, Q_inv), G)), G.T), Q_inv), E)
        r = np.sqrt(np.sum(np.hstack([(x[:, :2] - np.reshape(Z_[:2], (-1))) ** 2, np.reshape((x[:, 2] - h) ** 2, (-1, 1))]), axis=1))
        D = np.diag(r[1:])
        F = np.matmul(np.matmul(D, Q), D)
        F_inv = np.linalg.inv(F)
        Z = np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul(G.T, F_inv), G)), G.T), F_inv), E)
        coord = - np.reshape(Z[:2], (-1))
        for _ in range(n_iter):
            r = np.sqrt(np.sum(np.hstack([(x[:, :2] - coord) ** 2, np.reshape((x[:, 2] - h) ** 2, (-1, 1))]), axis=1))
            E = np.reshape(tdoa + r[0] - r[1:], (-1, 1))
            p = (x[:, :2] - coord) / np.reshape(r, (-1, 1))
            G = p[0, :] - p[1:, :]
            d = np.reshape(np.matmul(np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.matmul(G.T, Q_inv), G)), G.T), Q_inv), E), (-1))
            coord += d
            if np.sqrt(np.sum(d ** 2)) < threshold:
                return coord
        return None


def loc_rms(x, tdoa, h, sigma, n_iter, threshold):
    n, m = np.shape(x)[0] - 1, np.shape(x)[1]
    Q = .5 * (np.ones((n, n)) + np.identity(n)) * sigma ** 2
    K = np.sum((x - np.array([0, 0, h])) ** 2, axis=1)
    E = np.reshape(.5 * (tdoa ** 2 + K[0] - K[1:]), (-1, 1))
    G = 0 - np.hstack([x[0, :2] - x[1:, :2], np.reshape(tdoa, (-1, 1))])
    Q_inv = np.linalg.inv(Q)
    Z_ = np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul(G.T, Q_inv), G)), G.T), Q_inv), E)
    r = np.sqrt(np.sum(np.hstack([(x[:, :2] - np.reshape(Z_[:2], (-1))) ** 2, np.reshape((x[:, 2] - h) ** 2, (-1, 1))]), axis=1))
    D = np.diag(r[1:])
    F = np.matmul(np.matmul(D, Q), D)
    F_inv = np.linalg.inv(F)
    Z = np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul(G.T, F_inv), G)), G.T), F_inv), E)
    coord = - np.reshape(Z[:2], (-1))
    for _ in range(n_iter):
        r = np.sqrt(np.sum(np.hstack([(x[:, :2] - coord) ** 2, np.reshape((x[:, 2] - h) ** 2, (-1, 1))]), axis=1))
        E = np.reshape(tdoa + r[0] - r[1:], (-1, 1))
        p = (x[:, :2] - coord) / np.reshape(r, (-1, 1))
        G = p[0, :] - p[1:, :]
        d = np.reshape(np.matmul(np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.matmul(G.T, Q_inv), G)), G.T), Q_inv), E), (-1))
        coord += d
        if np.sqrt(np.sum(d ** 2)) < threshold:
            return coord
    return None

def loc_ls(x, d, z=0):
    n, m = np.shape(x)[0] - 1, np.shape(x)[1]
    A = np.hstack([np.tile(x[0, :2], (n, 1)) - x[1:, :2], d.reshape((n, 1))])
    D = np.sum(x ** 2, axis=1)
    b = .5 * (d ** 2 + D[0] - D[1:]).T + (x[1:, 2] - x[0, 2]) * z
    A = matrix(A)
    b = matrix(b)
    sol = solvers.qp(A.T*A, -A.T*b)
    return np.array(sol['x'])[:2].reshape(-1)

def loc_sdpi(x, d, z=0):
    n, m = np.shape(x)[0] - 1, np.shape(x)[1]
    c = np.zeros((m + 4, 1))
    c[m + 3, 0] = 1.
    A = np.zeros((2, m + 4))
    x02 = 0
    for j in range(m):
        A[0, j] = 2 * x[0, j]
        x02 += x[0, j] ** 2
    A[0, m] = - 1.
    A[0, m + 1] = 1.
    A[1, 2] = 1
    b= np.zeros((2, 1))
    b[0] = x02
    b[1] = z
    Gy = []
    for i in range(m):
        Gy.append(np.zeros((n * 2 + m * 2 + 5, n * 2 + m * 2 + 5)))
    Gys, Gds, Gdr, Gt, h = np.zeros((n * 2 + m * 2 + 5, n * 2 + m * 2 + 5)), np.zeros((n * 2 + m * 2 + 5, n * 2 + m * 2 + 5)), np.zeros((n * 2 + m * 2 + 5, n * 2 + m * 2 + 5)), np.zeros((n * 2 + m * 2 + 5, n * 2 + m * 2 + 5)), np.zeros((n * 2 + m * 2 + 5, n * 2 + m * 2 + 5))
    for i in range(n):
        xi2 = 0
        for j in range(m):
            Gy[j][i, i] = 2 * x[i + 1, j]
            Gy[j][i + n, i + n] = - 2 * x[i + 1, j]
            xi2 += x[i + 1, j] ** 2
        Gys[i, i] = -1.
        Gys[i + n, i + n] = 1.
        Gds[i, i] = 1.
        Gds[i + n, i + n] = -1.
        Gdr[i, i] = 2 * d[i]
        Gdr[i + n, i + n] = - 2 * d[i]
        Gt[i, i] = -1.
        Gt[i + n, i + n] = -1.
        h[i, i] = xi2 - d[i] ** 2
        h[i + n, i + n] = - xi2 + d[i] ** 2
    for j in range(m):
        Gy[j][n * 2 + j, n * 2 + m] = - 1.
        Gy[j][n * 2 + m, n * 2 + j] = - 1.
        h[n * 2 + j, n * 2 + j] = 1
    Gys[n * 2 + m, n * 2 + m] = - 1.
    Gds[n * 2 + m + 2, n * 2 + m + 2] = - 1.
    Gdr[n * 2 + m + 1, n * 2 + m + 2] = - 1.
    Gdr[n * 2 + m + 2, n * 2 + m + 1] = - 1.
    h[n * 2 + m + 1, n * 2 + m + 1] = 1.
    G = []
    for j in range(m):
        G.append(Gy[j].reshape(-1, 1))
    G.append(Gys.reshape(-1, 1))
    G.append(Gds.reshape(-1, 1))
    G.append(Gdr.reshape(-1, 1))
    G.append(Gt.reshape(-1, 1))
    c = matrix(c)
    G = [matrix(np.hstack(G))]
    h = [matrix(h)]
    A = matrix(A)
    b = matrix(b)
    sol = solvers.sdp(c, Gs=G, hs=h, A=A, b=b)
    return np.array(sol['x'])[:2].reshape(-1)

def loc_sdpc(x, d, e=1e-3, z=0):
    n, m = np.shape(x)[0] - 1, np.shape(x)[1]
    c = np.zeros((m * 3 + 4, 1))
    c[m * 3 + 3, 0] = 1.
    A = np.zeros((m + 2, m * 3 + 4))
    x02 = 0
    for j in range(m):
        A[j + 1, j] = 1.
        A[j + 1, m + j] = 1.
        A[0, m * 2 + j] = 2 * x[0, j]
        A[j + 1, m * 2 + j] = - 1.
        x02 += x[0, j] ** 2
    A[0, m * 3] = - 1.
    A[0, m * 3 + 1] = 1.
    A[-1, 2] = 1
    b = np.zeros((m + 2, 1))
    b[0] = x02
    b[-1] = z
    Gyc, Gu, Gy = [], [], []
    for i in range(m):
        Gyc.append(np.zeros((n * 2 + m * 2 + 5, n * 2 + m * 2 + 5)))
        Gu.append(np.zeros((n * 2 + m * 2 + 5, n * 2 + m * 2 + 5)))
        Gy.append(np.zeros((n * 2 + m * 2 + 5, n * 2 + m * 2 + 5)))
    Gys, Gds, Gdr, Gt, h = np.zeros((n * 2 + m * 2 + 5, n * 2 + m * 2 + 5)), np.zeros((n * 2 + m * 2 + 5, n * 2 + m * 2 + 5)), np.zeros((n * 2 + m * 2 + 5, n * 2 + m * 2 + 5)), np.zeros((n * 2 + m * 2 + 5, n * 2 + m * 2 + 5)), np.zeros((n * 2 + m * 2 + 5, n * 2 + m * 2 + 5))
    for i in range(n):
        xi2 = 0
        for j in range(m):
            Gy[j][i, i] = 2 * x[i + 1, j]
            Gy[j][i + n, i + n] = - 2 * x[i + 1, j]
            xi2 += x[i + 1, j] ** 2
        Gys[i, i] = -1.
        Gys[i + n, i + n] = 1.
        Gds[i, i] = 1.
        Gds[i + n, i + n] = -1.
        Gdr[i, i] = 2 * (d[i] - e)
        Gdr[i + n, i + n] = - 2 * (d[i] + e)
        h[i, i] = xi2 - (d[i] - e) ** 2
        h[i + n, i + n] = - xi2 + (d[i] + e) ** 2
    x02 = 0
    for j in range(m):
        Gu[j][n * 2 + m + 3 + j, n * 2 + m * 2 + 3] = - 1.
        Gu[j][n * 2 + m * 2 + 3, n * 2 + m + 3 + j] = - 1.
        Gy[j][n * 2 + j, n * 2 + m] = - 1.
        Gy[j][n * 2 + m, n * 2 + j] = - 1.
        Gt[n * 2 + m + 3 + j, n * 2 + m + 3 + j] = - 1.
        h[n * 2 + j, n * 2 + j] = 1
    Gys[n * 2 + m, n * 2 + m] = - 1.
    Gds[n * 2 + m + 2, n * 2 + m + 2] = - 1.
    Gdr[n * 2 + m + 1, n * 2 + m + 2] = - 1.
    Gdr[n * 2 + m + 2, n * 2 + m + 1] = - 1.
    Gt[n * 2 + m * 2 + 3, n * 2 + m * 2 + 3] = - 1.
    Gt[n * 2 + m * 2 + 4, n * 2 + m * 2 + 4] = - 1.
    h[n * 2 + m + 1, n * 2 + m + 1] = 1.
    G = []
    for j in range(m):
        G.append(Gyc[j].reshape(-1, 1))
    for j in range(m):
        G.append(Gu[j].reshape(-1, 1))
    for j in range(m):
        G.append(Gy[j].reshape(-1, 1))
    G.append(Gys.reshape(-1, 1))
    G.append(Gds.reshape(-1, 1))
    G.append(Gdr.reshape(-1, 1))
    G.append(Gt.reshape(-1, 1))
    c = matrix(c)
    G = [matrix(np.hstack(G))]
    h = [matrix(h)]
    A = matrix(A)
    b = matrix(b)
    sol = solvers.sdp(c, Gs=G, hs=h, A=A, b=b)
    return np.array(sol['x'])[:2].reshape(-1)

def loc_sdpr(x, d, e=1e-3, z=0):
    n, m = np.shape(x)[0] - 1, np.shape(x)[1]
    s = np.sum(x ** 2, axis=1)
    vc1, vc2, ck1, ck2, ct1, ct2, mc1, mc2, mq, vs = [], [], [], [], [], [], [], [], [], []
    for i in range(n):
        ca = np.zeros((n + m, 1))
        ca[:m] = 2 * (x[0] - x[i + 1]).reshape((-1, 1))
        ca[m + i] = - 2 * d[i]
        cb = s[0] - s[i + 1] - d[i] ** 2
        cv = np.zeros((n + m, 1))
        cv[m + i] = 1
        vc1.append(ca + 2 * e * cv)
        vc2.append(ca - 2 * e * cv)
        ck1.append(e ** 2 + 2 * d[i] * e + cb)
        ck2.append(e ** 2 + 2 * d[i] * e - cb)
        ct1.append(e ** 2 - 2 * d[i] * e + cb)
        ct2.append(e ** 2 - 2 * d[i] * e - cb)
        mc1.append(vc1[i] * vc1[i].T)
        mc2.append(vc2[i] * vc2[i].T)
        mq.append(np.zeros((m + n, m + n)))
        mq[i][:m, :m] = np.eye(m)
        mq[i][m + i, m + i] = -1
        vs.append(np.zeros((m + n, 1)))
        vs[i][:m] = x[i + 1, :].reshape((-1, 1))
    c = np.zeros((int((n + m + 1) * (n + m) / 2) + n + m + n, 1))
    c[-n:] = np.ones((n, 1))
    A = np.zeros((1, int((n + m + 1) * (n + m) / 2) + n + m + n))
    A[0, 2] = 1
    b = np.zeros((1, 1))
    b[0] = z
    size = n * 6 + n + m + 1
    size += (m + 1) * n + n
    size = (size, size)
    h = np.zeros(size)
    Gy, Gz, Gt = [np.zeros(size) for _ in range(n + m)], [np.zeros(size) for _ in range((n + m) * (n + m))], [np.zeros(size) for _ in range(n)]
    for i in range(n):
        Gt[i][i + n * 0, i + n * 0] = - 1
        Gt[i][i + n * 1, i + n * 1] = - 1
        Gt[i][i + n * 2, i + n * 2] = - 1
        Gt[i][i + n * 3, i + n * 3] = - 1
        h[i + n * 0, i + n * 0] = 0 - ck2[i] ** 2
        h[i + n * 1, i + n * 1] = 0 - ct2[i] ** 2
        h[i + n * 2, i + n * 2] = 0 - ck1[i] ** 2
        h[i + n * 3, i + n * 3] = 0 - ct1[i] ** 2
        h[i + n * 4, i + n * 4] = 0 + s[i + 1]
        h[i + n * 5, i + n * 5] = 0 - s[i + 1]
        for j in range(n + m):
            Gy[j][i + n * 0, i + n * 0] = 0 + 2 * ck2[i] * vc2[i][j]
            Gy[j][i + n * 1, i + n * 1] = 0 + 2 * ct2[i] * vc1[i][j]
            Gy[j][i + n * 2, i + n * 2] = 0 - 2 * ck1[i] * vc1[i][j]
            Gy[j][i + n * 3, i + n * 3] = 0 - 2 * ct1[i] * vc2[i][j]
            Gy[j][i + n * 4, i + n * 4] = 0 + 2 * vs[i][j]
            Gy[j][i + n * 5, i + n * 5] = 0 - 2 * vs[i][j]
            for k in range(j):
                Gz[j * (n + m) + k][i + n * 0, i + n * 0] = 0 + 2 * mc2[i][k, j]
                Gz[j * (n + m) + k][i + n * 1, i + n * 1] = 0 + 2 * mc1[i][k, j]
                Gz[j * (n + m) + k][i + n * 2, i + n * 2] = 0 + 2 * mc1[i][k, j]
                Gz[j * (n + m) + k][i + n * 3, i + n * 3] = 0 + 2 * mc2[i][k, j]
                Gz[j * (n + m) + k][i + n * 4, i + n * 4] = 0 - 2 * mq[i][k, j]
                Gz[j * (n + m) + k][i + n * 5, i + n * 5] = 0 + 2 * mq[i][k, j]
            Gz[j * (n + m) + j][i + n * 0, i + n * 0] = 0 + mc2[i][j, j]
            Gz[j * (n + m) + j][i + n * 1, i + n * 1] = 0 + mc1[i][j, j]
            Gz[j * (n + m) + j][i + n * 2, i + n * 2] = 0 + mc1[i][j, j]
            Gz[j * (n + m) + j][i + n * 3, i + n * 3] = 0 + mc2[i][j, j]
            Gz[j * (n + m) + j][i + n * 4, i + n * 4] = 0 - mq[i][j, j]
            Gz[j * (n + m) + j][i + n * 5, i + n * 5] = 0 + mq[i][j, j]
        for j in range(m):
            Gy[j][n * 6 + n + m + 1 + i * (m + 1) + j, n * 6 + n + m + 1 + i * (m + 1) + m] = -1
            Gy[j][n * 6 + n + m + 1 + i * (m + 1) + m, n * 6 + n + m + 1 + i * (m + 1) + j] = -1
            Gy[m + i][n * 6 + n + m + 1 + i * (m + 1) + j, n * 6 + n + m + 1 + i * (m + 1) + j] = -1
            h[n * 6 + n + m + 1 + i * (m + 1) + j, n * 6 + n + m + 1 + i * (m + 1) + m] = - x[i + 1, j]
            h[n * 6 + n + m + 1 + i * (m + 1) + m, n * 6 + n + m + 1 + i * (m + 1) + j] = - x[i + 1, j]
        Gy[m + i][n * 6 + n + m + 1 + i * (m + 1) + m, n * 6 + n + m + 1 + i * (m + 1) + m] = -1
        Gy[m + i][n * 6 + n + m + 1 + n * (m + 1) + i, n * 6 + n + m + 1 + n * (m + 1) + i] = -1
    for j in range(n + m):
        Gy[j][n * 6 + j, n * 6 + n + m] = -1
        Gy[j][n * 6 + n + m, n * 6 + j] = -1
        for k in range(j):
            Gz[j * (n + m) + k][n * 6 + j, n * 6 + k] = -1
            Gz[j * (n + m) + k][n * 6 + k, n * 6 + j] = -1
        Gz[j * (n + m) + j][n * 6 + j, n * 6 + j] = -1
    h[n * 6 + n + m, n * 6 + n + m] = 1
    G = []
    for i in range(n + m):
        G.append(Gy[i].reshape((-1, 1)))
    for i in range(n + m):
        for j in range(i + 1):
            G.append(Gz[i * (n + m) + j].reshape((-1, 1)))
    for i in range(n):
        G.append(Gt[i].reshape((-1, 1)))
    c = matrix(c)
    G = [matrix(np.hstack(G))]
    h = [matrix(h)]
    A = matrix(A)
    b = matrix(b)
    sol = solvers.sdp(c, Gs=G, hs=h, A=A, b=b)
    return np.array(sol['x'])[:2].reshape(-1)


def Print_Loc(i,X,D,E,Z):
    print(i)
    print(loc_ls(X, D, z=Z))
    print(loc_sdpi(X, D, z=Z))
    print(loc_sdpc(X, D, E, z=Z))
    print(loc_rms(X, D, Z, E, 1000, 1e-6))
    print(loc_sdpr(X, D, E, z=Z))

def Send_xy(str_xy):

    #client 发送端
    ip_port_client = ('127.0.0.1', 9696)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.bind(ip_port_client)
    PORT = 8888

    sleeptime = 1.0
    dis = str_xy
    recv = "start"#套接字编程控制 可以调节频率
    # ser = serial.Serial(serialPort, baudRate,timeout=1)#timeout为超时设置，必须要，并且注意！！！该函数每调用一次，将对Arduino程序进行一次刷新，reset

    while recv!="end":
        time.sleep(sleeptime)
        if dis!='':
            server_address = ("127.0.0.1", PORT)  # 接收方 服务器的ip地址和端口号
            client_socket.sendto(dis.encode(), server_address) #将msg内容发送给指定接收方
            print('Send Sucess')
    serial.close()    


if __name__ == '__main__':

    # testx为四个基站的
    testx = np.array([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [0., 1., 1.], [1., 0., 0.],[1., 0., 1.], [1., 1., 0.], [1., 1., 1.]])
    #TESTD = Get_tdoa()
    teste = 1e-2
    testd = np.array([1., 1., 1.41, 1., 1.41, 1.41, 1.73]) + np.random.randn(7) * 1e-4

    LAA1 = LocationAlgorithmAgent( x=testx,d=testd,e=teste)
    Loc_xy = LAA1.loc_ls()
    Send_xy(str(Loc_xy[0])+'#'+str(Loc_xy[1]))

    
    