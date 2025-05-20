# baseline_correction.py

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def baseline_als(y, lam=1e5, p=0.01, niter=10):
    """
    Asymmetric Least Squares Smoothing (baseline correction)

    Args:
        y (np.array): input signal (e.g. spectrum)
        lam (float): smoothness parameter (larger = smoother baseline)
        p (float): asymmetry parameter (0 < p < 1, controls penalty)
        niter (int): number of iterations (default: 10)

    Returns:
        np.array: estimated baseline
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L-2, L))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.T @ D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z
