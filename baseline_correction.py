# baseline_correction.py

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def baseline_als(y, lam=1e5, p=0.01, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def baseline_airpls(y, lambda_=1e5, porder=0.01, itermax=15):
    m = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(m, m-2))
    H = lambda_ * D.dot(D.transpose())
    w = np.ones(m)
    for i in range(itermax):
        W = sparse.spdiags(w, 0, m, m)
        Z = W + H
        z = spsolve(Z, w * y)
        d = y - z
        dn = d[d < 0]
        if len(dn) == 0:
            break
        m1 = np.mean(np.abs(dn))
        s1 = np.std(dn)
        w = np.where(d >= 0, 0, np.exp(- (d ** 2) / (2 * s1 ** 2)))
    return z
