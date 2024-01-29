"""Compute utilities with two-dimensional types (risk-aversion, risk)
    and (deductible, copay) contracts
"""

import numpy as np
from bs_python_utils.bsnputils import TwoArrays, npmaxabs

from .INSUR import nlLambda_j
from .values import S_fun


def compute_rents(
    y_second_best: np.ndarray,
    theta_mat: np.ndarray,
    s: float,
    loading: float,
    tol_fp: float = 1e-6,
) -> TwoArrays:
    """Computes the rents for each type using the iterative algorithm $T_{\Lambda}$ of Prop 2

    Args:
        y_second_best: the $(N,2)$-matrix of optimal contracts
        theta_mat: the $(N,2)$-matrix of types
        s: the dispersion of individual losses
        loading: the loading factor
        tol_fp: tolerance for fixed point

    Returns:
        U_vals: an $N$-vector of rents
        S: an $N$-vector of the values of the joint surplus
    """
    N = y_second_best.shape[0]
    y_second = np.concatenate((y_second_best[:, 0], y_second_best[:, 1]))
    Lambda_vals = nlLambda_j(y_second, theta_mat, s).reshape((N, N))
    S_vals = np.zeros(N)
    for i in range(N):
        sigma_vec = np.array([theta_mat[i, 0]])
        delta_vec = np.array([theta_mat[i, 1]])
        S_vals[i] = S_fun(
            np.array([y_second[i], y_second[N + i]]), sigma_vec, delta_vec, s, loading
        )[0, 0]
    U_vals = U_old = np.zeros(N)
    dU = np.inf
    it_max = N
    it = 0
    while dU > tol_fp and it < it_max:
        for i in range(N):
            U_vals[i] = np.max(Lambda_vals[i, :] + U_old)
        dU = npmaxabs(U_vals - U_old)
        U_old = U_vals
        it += 1

    return U_vals, S_vals
