# This Python file uses the following encoding: utf-8


import numpy as np
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
from filterpy.monte_carlo import residual_resample


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
    b = np.zeros((2, 1))
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


def crlb(x, y, d, e):
    n, m = np.shape(x)[0] - 1, np.shape(x)[1]
    Q = np.ones((n, n)) * .5 + np.eye(n) * .5
    Q = Q * e ** 2
    P = (np.tile(y.reshape((1, -1)), (n, 1)) - x[1:, :] / np.tile(d[1:].reshape((-1, 1)), (1, m))) - (np.tile(y.reshape((1, -1)), (n, 1)) - np.tile(x[0, :].reshape((1, -1)), (n, 1)) / d[0])
    J = np.linalg.inv(np.matmul(np.matmul(P.T, np.linalg.inv(Q)), P))
    return J


class ParticleFilter(object):
    def __init__(self, initial_coord, n_particle, cfg, sigma=1., initial_velo=0, velo_mul=1, h=None, min_h=0, max_h=200):
        self.n_particle = n_particle
        self.cfg = cfg
        self.sigma = sigma
        self.VELO_MUL = velo_mul
        self.h = h
        self.particle_coords = np.random.randn(self.n_particle, 2) + initial_coord[:2] if initial_coord.ndim == 1 else np.random.rand(self.n_particle, 2) * (initial_coord[:, 1] - initial_coord[:, 0]) + initial_coord[:, 0]
        self.particle_coords = np.hstack([self.particle_coords, np.random.rand(n_particle, 1) * (max_h - min_h) + min_h]) if h is None else np.hstack([self.particle_coords, np.ones((n_particle, 1)) * h])
        self.particle_velos = np.random.randn(self.n_particle) * self.VELO_MUL + initial_velo
        self.particle_heads = np.random.rand(self.n_particle) * np.pi * 2
        self.power = np.zeros(self.n_particle)

    def predict(self, t):
        self.particle_heads += np.random.randn(self.n_particle) * np.pi / 3.
        # self.particle_heads = np.random.rand(self.n_particle) * np.pi * 2.
        self.particle_velos += np.random.randn(self.n_particle) * t * self.VELO_MUL
        self.particle_coords[:, :2] += np.stack([np.cos(self.particle_heads) * self.particle_velos, np.sin(self.particle_heads) * self.particle_velos], axis=1) * t
        if self.h is None:
            self.particle_coords[:, 2] += np.random.randn(self.n_particle) * t * self.VELO_MUL / 100.
        return

    def update(self, target, mode):
        if mode == 'COORD':
            self.power += np.sum((self.particle_coords[:, :2] - target) ** 2, axis=1) / (2 * self.sigma ** 2)
        else:
            self.power += np.sum((self.cfg.tdoa(self.particle_coords) - target) ** 2, axis=1) / (2 * self.sigma ** 2)
        return

    def estimate(self):
        p = np.exp(np.min(self.power)  - self.power)
        p /= np.sum(p)
        idx = residual_resample(p)
        self.particle_coords, self.particle_velos, self.particle_heads = self.particle_coords[idx, :], self.particle_velos[idx], self.particle_heads[idx]
        self.power = np.zeros(self.n_particle)
        return np.mean(self.particle_coords, axis=0)


class ExtendedKalmanFilter():
    def __init__(self, initial_coord, h, cfg, Q, R):
        x = np.concatenate([initial_coord, np.zeros(2)])
        self.x = np.matrix(x.reshape((-1, 1)))
        self.h = h
        self.cfg = cfg
        self.P = np.matrix(Q)
        self.Q = np.matrix(Q)
        self.R = np.matrix(R)

    def predict(self, t):
        F = np.eye(4)
        F[0, 2] = t
        F[1, 3] = t
        F = np.matrix(F)
        v = np.random.multivariate_normal(np.zeros(4), self.Q)
        v = np.matrix(v.reshape((-1, 1)))
        self.x = F * self.x + v
        self.P = F * self.P * F.T + self.Q

    def update(self, target):
        coord = np.array([self.x[0, 0], self.x[1, 0], self.h])
        tdoa = self.cfg.tdoa(coord)[0]
        y = target - tdoa
        y = np.matrix(y.reshape(-1, 1))
        H = np.zeros((self.cfg.m - 1, 4))
        r = np.sqrt(np.sum((self.cfg.bs - coord) ** 2, axis=1))
        for i in range(self.cfg.m - 1):
            H[i, 0] = (self.x[0] - self.cfg.bs[i + 1, 0]) / r[i + 1] - (self.x[0] - self.cfg.bs[0, 0]) / r[0]
            H[i, 1] = (self.x[1] - self.cfg.bs[i + 1, 1]) / r[i + 1] - (self.x[1] - self.cfg.bs[0, 1]) / r[0]
        H = np.matrix(H)
        S = H * self.P * H.T + self.R
        K = self.P * H.T * S.I
        self.x = self.x + K * y
        I = np.matrix(np.eye(4))
        self.P = (I - K * H) * self.P

    def estimate(self):
        return self.x[:2]


class OutlierRubustKalmanFilter():
    def __init__(self, initial_coord, h, cfg, Q, R, s):
        coord = np.array([initial_coord[0], initial_coord[1], h])
        tdoa = cfg.tdoa(coord)[0]
        x = np.concatenate([tdoa, np.zeros(cfg.m * 2 - 2)])
        self.x = np.matrix(x.reshape((-1, 1)))
        self.h = h
        self.cfg = cfg
        P = np.zeros((cfg.m * 3 - 3, cfg.m * 3 - 3))
        P[cfg.m * 2 - 2:, cfg.m * 2 - 2:] = Q
        self.P = np.matrix(P)
        self.Q = np.matrix(Q)
        self.R = np.matrix(R)
        self.s = s
        H = np.hstack([np.eye(cfg.m - 1), np.zeros((cfg.m - 1, cfg.m * 2 - 2))])
        self.H = np.matrix(H)

    def predict(self, t):
        F = np.eye(self.cfg.m * 3 - 3)
        F[:self.cfg.m * 2 - 2, self.cfg.m - 1:] += np.eye(self.cfg.m * 2 - 2) * t
        F[:self.cfg.m - 1, self.cfg.m * 2 - 2:] += np.eye(self.cfg.m - 1) * t * t * .5
        F = np.matrix(F)
        self.x = F * self.x
        G = np.vstack([np.eye(self.cfg.m - 1) * t * t * .5, np.eye(self.cfg.m - 1) * t, np.eye(self.cfg.m - 1)])
        self.G = np.matrix(G)
        self.P = F * self.P * F.T + G * self.Q * G.T

    def update(self, target, threshold):
        y = np.matrix(target.reshape(-1, 1))
        T = np.matrix(self.R)
        I = np.matrix(np.eye(self.cfg.m * 3 - 3))
        d = y - self.H * self.x
        while np.sum(np.multiply(d, d)) > threshold:
            S = self.H * self.P * self.H.T + T
            K = self.P * self.H.T * S.I
            self.x = self.x + K * d
            self.P = (I - K * self.H) * self.P
            d = y - self.H * self.x
            T = (self.s * self.R + d * d.T + self.H * self.P * self.H.T) / (self.s + 1)

    def estimate(self):
        return np.array(self.H * self.x).reshape(-1), self.P[:self.cfg.m - 1, :self.cfg.m - 1]


class RecursivelyBoundedGridBasedFilter():
    def __init__(self, initial_coord, h, cfg, initial_grid, gap, v):
        self.cfg = cfg
        self.h = h
        self.gap = gap
        self.v = v
        self.l = initial_coord[0] - initial_grid * gap
        self.b = initial_coord[1] - initial_grid * gap
        self.w = np.ones((initial_grid * 2 + 1, initial_grid * 2 + 1))

    def predict(self, t):
        y, x = np.shape(self.w)
        m = int(self.v * t / self.gap + 1)
        self.l -= m * self.gap
        self.b -= m * self.gap
        w = np.zeros((x + m * 2, y + m * 2))
        for i in range(y):
            for j in range(x):
                w[i: i + m * 2 + 1, j: j + m * 2 + 1] += self.w[i, j]
        self.w = w

    def update(self, target, Q, threshold):
        y, x = np.shape(self.w)
        power = np.zeros((y, x))
        for i in range(y):
            for j in range(x):
                coord = np.array([self.l + j * self.gap, self.b + i * self.gap, self.h])
                tdoa = self.cfg.tdoa(coord)
                d = np.matrix((tdoa - target).reshape((- 1, 1)))
                power[i, j] = - .5 * d.T * Q.I * d
        power -= np.max(power)
        self.w = self.w * np.exp(power)
        w = self.w.reshape(-1) + 1e-10
        w /= np.sum(w)
        idx = np.argsort(w)[::-1]
        k, s, l, r, t, b = 0, 0, x, 0, 0, y
        while s < threshold:
            i = int(idx[k] / x)
            j = idx[k] - i * x
            if j < l:
                l = j
            if j > r:
                r = j
            if i > t:
                t = i
            if i < b:
                b = i
            s += w[idx[k]]
            k += 1
        self.l = self.l + l * self.gap
        self.b = self.b + b * self.gap
        self.w = self.w[b: t + 1, l: r + 1] + 1e-10

    def estimate(self):
        self.w /= np.sum(self.w)
        x = 0
        y = 0
        for i in range(np.shape(self.w)[0]):
            for j in range(np.shape(self.w)[1]):
                y += self.w[i, j] * (self.b + i * self.gap)
                x += self.w[i, j] * (self.l + j * self.gap)
        return np.array([x, y])


if __name__ == '__main__':
    x = np.array([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [0., 1., 1.], [1., 0., 0.],[1., 0., 1.], [1., 1., 0.], [1., 1., 1.]])
    # x = np.array([[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    e = 1e-2
    d = np.array([1., 1., 1.41, 1., 1.41, 1.41, 1.73]) + np.random.randn(7) * 1e-4
    # d = np.array([1.41, 1.41, 1.41]) + np.random.randn(3) * 1e-4
    y = np.array([0., 0., 0.])
    print(1)
    print(loc_ls(x, d, z=0))
    print(loc_sdpi(x, d, z=0))
    print(loc_sdpc(x, d, e, z=0))
    print(loc_rms(x, d, 0, e, 1000, 1e-6))
    print(loc_sdpr(x, d, e, z=0))
    # print(np.sqrt(np.trace(crlb(x, y, 1e-1))))
    d = np.array([-0.32, -0.32, -0.73, -0.32, -0.73, -0.73, -1.73]) + np.random.randn(7) * 1e-4
    y = np.array([1., 1., 1.])
    print(2)
    print(loc_ls(x, d, z=1))
    print(loc_sdpi(x, d, z=1))
    print(loc_sdpc(x, d, e, z=1))
    print(loc_rms(x, d, 1, e, 1000, 1e-6))
    print(loc_sdpr(x, d, e, z=1))
    # print(np.sqrt(np.trace(crlb(x, y, 1e-1))))
    d = np.array([0., 0., 0., 0., 0., 0., 0.]) + np.random.randn(7) * 1e-4
    y = np.array([.5, .5, .5])
    print(3)
    print(loc_ls(x, d, z=.5))
    print(loc_sdpi(x, d, z=.5))
    print(loc_sdpc(x, d, e, z=.5))
    print(loc_rms(x, d, .5, e, 1000, 1e-6))
    print(loc_sdpr(x, d, e, z=.5))
    # print(np.sqrt(np.trace(crlb(x, y, 1e-1))))
