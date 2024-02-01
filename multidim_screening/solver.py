"""Algorithm for multidimensional screening
"""

from datetime import timedelta
from math import sqrt
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, cast

import numpy as np
import scipy.sparse as sparse
from bs_python_utils.bs_opt import minimize_free
from bs_python_utils.bsnputils import TwoArrays, npmaxabs
from bs_python_utils.bsutils import bs_error_abort
from numba import njit, prange
from numba.typed import List

from .classes import ScreeningModel, ScreeningResults
from .specif import S_deriv, S_function, b_deriv, b_function
from .utils import L2_norm


def construct_D(N: int) -> tuple[sparse.csr_matrix, float]:
    """Constructs the matrix $D$ and the step size for the projection

    Args:
        N: number of types

    Returns:
        D: the $(N^2, N)$ matrix $D$
        gamma_proj: the step size
    """
    D = sparse.vstack(
        [
            -sparse.eye(N, dtype=np.float64)
            + sparse.coo_matrix(
                (
                    np.ones(N, dtype=np.float64),
                    (
                        np.arange(N, dtype=np.float64),
                        i * np.ones(N, dtype=int),
                    ),
                ),
                shape=(N, N),
            )
            for i in range(N)
        ],
        format="csr",
    )
    D = D[range(N * N)]
    _, s_proj, _ = sparse.linalg.svds(D, k=1, solver="arpack")  # norm_Lambda = s[0]
    gamma_proj = 1 / s_proj[0] ** 2
    return D, gamma_proj


@njit("float64[:,:,:](float64[:], float64[:, :], List)")
def JLambda(y: np.ndarray, theta_mat: np.ndarray, params: List) -> np.ndarray:
    """computes $\\Lambda^\\prime_{ij}(y) = b^\\prime_i(y_j)-b^\\prime_j(y_j)$

    Args:
        y: the contracts, an array of size $m N$
        theta_mat: the types, an `(N, m)` matrix
        params: the parameters of the model

    Returns:
        an $(m, N, N)$ array.
    """
    N, m = theta_mat.shape
    # we compute the (N, N) matrices db_i/dy_0(y_j) and db_i/dy_1(y_j)
    db_vals = b_deriv(y, theta_mat, params)
    J = np.zeros((2, N, N))
    for i in range(m):
        db_vals_i = db_vals[i, :, :]
        J[i, :, :] = db_vals_i - np.diag(db_vals_i)
    return J


def prox_H(
    z: np.ndarray, theta: np.ndarray, t: float, params: List
) -> np.ndarray | None:
    """Proximal operator of -t S_i at z;
        minimizes $-S_i(y) + 1/(2 t) \\lVert y-z \rVert^2$

    Args:
        z: an `m`-vector
        theta: type $i$'s characteristics, a $d$-vector
        t: the step
        params: the parameters of the model

    Returns:
        the minimizing $y$, an $m$-vector
    """
    d = theta.size

    def prox_obj_and_grad(
        y: np.ndarray, args: list, gr: bool = False
    ) -> float | tuple[float, np.ndarray]:
        dyz = y - z
        dist_yz2 = np.sum(dyz * dyz)
        theta_mat1 = np.empty((1, d))
        theta_mat1[0, :] = theta
        obj = -S_function(y, theta_mat1, params)[0, 0] + dist_yz2 / (2 * t)
        if gr:
            grad = -S_deriv(y, theta_mat1, params)[0, 0] + (y - z) / t
            return obj, grad
        else:
            return cast(float, obj)

    def prox_obj(y: np.ndarray, args: list) -> float:
        return cast(float, prox_obj_and_grad(y, args, gr=False))

    def prox_grad(y: np.ndarray, args: list) -> np.ndarray:
        return cast(tuple[float, np.ndarray], prox_obj_and_grad(y, args, gr=True))[1]

    # check_gradient_scalar_function(prox_obj_and_grad, z, args=[])

    mini = minimize_free(
        prox_obj,
        prox_grad,
        x_init=z,
        args=[],
    )

    if mini.success:
        y = mini.x
        return cast(np.ndarray, y)
    else:
        print(f"{mini.message}")
        bs_error_abort(f"Minimization did not converge: status {mini.status}")
        return None


def make_v(w: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """computes $v_{ij} = \\max(w_{ij}-\beta_i +\beta_j, 0)$

    Args:
        w: an `(N,N)` matrix
        beta: an `N`-vector

    Returns:
        v: an `(N,N)` matrix
    """
    dbeta = D_mul(beta)
    return cast(np.ndarray, np.clip(w - dbeta, 0.0, np.inf))


def D_mul(v: np.ndarray) -> np.ndarray:
    """computes $D v$

    Args:
        v: an `N`-vector

    Returns:
        an $N^d$-vector
    """
    N = v.size
    return np.add.outer(v, -v).reshape(N * N)


@njit("float64[:](float64[:, :])")
def D_star(v_mat: np.ndarray) -> Any:
    """computes $D^\ast v$

    Args:
        v_mat: an $(N, N)$-matrix

    Returns:
        an `N`-vector
    """
    return np.sum(v_mat, 1) - np.sum(v_mat, 0)


@njit("float64[:,:](float64[:], float64[:, :], List)", parallel=True)
def nlLambda(
    y: np.ndarray,
    theta_mat: np.ndarray,
    params: List,
) -> Any:
    """computes $\\Lambda_{ij}(y) = b_i(y_j)-b_j(y_j)$

    Args:
        y: the contracts, a vector of size $m N$ (`y_0` then `y_1` etc)
        theta_mat: the types, a matrix of size $(N, d)$
        params: the parameters of the model

    Returns:
        an $(N,N)$ matrix.
    """
    N = theta_mat.shape[0]
    b_vals = b_function(y, theta_mat, params)
    db = b_vals
    for j in prange(N):
        db[:, j] -= b_vals[j, j]
    return db


# @njit(parallel=True)
def prox_work_func(list_working: list) -> list:
    n_working = len(list_working)
    res: list = [None] * n_working
    for i in range(n_working):
        arg_i = list_working[i]
        res[i] = prox_H(*arg_i)
    return res


def prox_minusS(
    model: ScreeningModel,
    z: np.ndarray,
    tau: float,
    fix_top: bool = False,
    free_y: list | None = None,
) -> np.ndarray:
    """Proximal operator of -S(y) = sum_i f_i H(y_i, theta_i)

    Args:
        model: the model
        z: an `N`-vector
        tau: scale factor
        fix_top: True if first-best imposed at top
        free_y: a list of types for which we optimize over contracts

    Returns:
        the minimizing `y`, a $2 N$-vector
    """
    theta_mat = model.theta_mat
    N, d = theta_mat.shape
    params = model.params
    f = model.f
    y_first_best = model.FB_y
    list_args = [
        [
            np.array([z[k * N + i] for k in range(d)]),
            theta_mat[i, :],
            tau * f[i],
            params,
        ]
        for i in range(N)
    ]

    # by default no one is modeled
    y = np.concatenate((np.zeros(N), np.ones(N)))
    # these are the types we will be working with
    Nmax = N - 1 if fix_top else N
    working_i0 = [i for i in free_y if i < Nmax] if free_y else list(range(Nmax))

    list_working = [list_args[i] for i in working_i0]
    n_working = len(list_working)

    # print(f"{working_i0=}")

    if fix_top:
        # we fix the second-best at the first-best at the top
        for k in range(d):
            y[k * N + N - 1] = y_first_best[-1, k]
    res = prox_work_func(list_working)
    for i in range(n_working):
        res_i = res[i]
        i_working = working_i0[i]
        for k in range(d):
            y[i_working + k * N] = res_i[k]
    return y


def proj_K(
    model: ScreeningModel,
    w: np.ndarray,
    lamb: np.ndarray,
    gamma_proj: float,
    warmstart: bool = True,
    atol_proj: float = 1e-6,
    rtol_proj: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int] | None:
    """Projection of $w$ onto $K$ by Fast Projected Gradient

    Args:
        model: the model
        w: an `(N * N)` vector
        lamb: an `N` vector, the previous value
        warmstart: whether to start from the previous `lamb`
        gamma_proj: the step size
        atol_proj: absolute tolerance
        rtol_proj: relative tolerance
    Returns:
        the projection, the value of `lamb`, where the IR constraints bind,
        the number of iterations, and a status code
    """
    N = model.N
    eta = model.f

    it_max = 100_000
    lamb1 = lamb if warmstart else np.zeros(N)
    lamb_extra = lamb1
    converged: bool = False
    it = 0
    c_tol = rtol_proj * eta + atol_proj
    while it < it_max and not converged:
        lamb_old = lamb1
        Dw = np.clip(w - D_mul(lamb_extra), 0.0, None).reshape((N, N))
        lamb1 = np.clip(lamb_extra - gamma_proj * (eta - D_star(Dw)), 0.0, None)
        v = np.clip(w - D_mul(lamb1), 0.0, None)
        constraints = D_star(v.reshape((N, N))) - eta
        lamb_constraints = lamb1 @ constraints
        dvw = v - w
        dvw2 = np.sum(dvw * dvw)
        obj = 0.5 * dvw2 + lamb_constraints
        o_tol = rtol_proj * obj + atol_proj
        converged = cast(
            bool, np.all(constraints < c_tol) and abs(lamb_constraints) < o_tol
        )
        lamb_extra = lamb1 + (it - 1.0) / (it + 3.0) * (lamb1 - lamb_old)

    if converged:
        IR_binding = np.flatnonzero(
            (D_star(v.reshape((N, N))) - eta) < -(rtol_proj * eta + atol_proj)
        ).astype(int)
        return v, lamb1, IR_binding, it, converged
    else:
        bs_error_abort("failed to converge.")
        return None


def solve(
    model: ScreeningModel,
    stepratio: float = 1.0,
    scale: bool = True,
    warmstart: bool = True,
    t_acc: float = 1.0,
    log: bool = True,
    it_max: int = 100_000,
    tol_primal: float = 1e-6,
    tol_dual: float = 1e-6,
    fix_top: bool = True,
    free_y: list | None = None,
    mult_fac: float = 1.0,
):
    # initialization
    N = model.N
    N2 = N * N
    model.rescale_step(mult_fac)

    _, gamma_proj = construct_D(N)
    t_start = timer()
    it = 0
    criteria = np.array([False, False, False])
    rec_primal_residual = []
    rec_dual_residual = []
    rec_it_proj = []

    v = model.v0.reshape(N * N)
    y = model.y_init
    theta_mat = model.theta_mat
    d = theta_mat.shape[1]
    params = model.params

    # scaling of the tolerances
    if scale:
        tol_primal = sqrt(len(y)) * tol_primal
        tol_dual = sqrt(len(v)) * tol_dual

    tau = 1.0 / (sqrt(stepratio) * model.norm_Lambda) / mult_fac
    sig = sqrt(stepratio) / model.norm_Lambda / mult_fac

    prox_F = prox_minusS

    JLy = JLambda(y, theta_mat, params)  # this is (Lambda'(y))^*
    LTv = make_LTv(v.reshape((N, N)), JLy)

    # loop
    lamb = np.zeros(N)
    y_old = y.copy()
    v_old = v
    LTv_old = LTv
    while it < it_max and not criteria.all():
        it += 1
        # primal update
        # print("\t" * 10, f"first {y=}")
        y = prox_F(
            model,
            y_old - tau * LTv_old,
            tau,
            fix_top=fix_top,
            free_y=free_y,
        )
        # print(f" in proj {y[25]=}")
        # print("\t" * 10, f"change in norm {spla.norm(y - y_old) =}")
        # dual update
        y_bar = y + t_acc * (y - y_old)
        Ly_bar = nlLambda(y_bar, theta_mat, params).reshape(N2)
        proj_res = cast(
            tuple[np.ndarray, np.ndarray, np.ndarray, int, int],
            proj_K(
                model,
                v + sig * Ly_bar,
                lamb,
                gamma_proj,
                warmstart=warmstart,
            ),
        )
        v, lamb, IR_binding, n_it_proj, proj_converged = proj_res
        Ly = nlLambda(y_bar, theta_mat, params).reshape(N2)
        JLy = JLambda(y, theta_mat, params)
        LTv = make_LTv(v.reshape((N, N)), JLy)
        # record
        norm1 = L2_norm((y_old - y) * t_acc / tau - (LTv_old - LTv))
        rec_primal_residual.append(norm1)
        norm2 = L2_norm((v_old - v) / sig + (Ly_bar - Ly))
        rec_dual_residual.append(norm2)
        rec_it_proj.append(n_it_proj)
        # stopping criterion
        criteria = np.array(
            [
                rec_primal_residual[-1] < tol_primal,
                rec_dual_residual[-1] < tol_dual,
                proj_converged,
            ]
        )
        y_mat = np.column_stack([y[k * N : (k + 1)] for k in range(d)])
        if it % 100 == 0 and log:
            print("\n\ty is:")
            for i in range(N):
                print(f"{y[k*N+i]: >10.4f} " for k in range(d))
            np.savetxt(
                cast(Path, model.resdir) / "current_y.txt",
                np.round(y_mat, 4),
            )
            print(f"\n\t\t\t{criteria=} at {it=}\n")
            print(f"\t\t\t\tprimal: {rec_primal_residual[-1] / tol_primal: > 10.2f}")
            print(f"\t\t\t\tdual: {rec_dual_residual[-1] / tol_dual: > 10.2f}")
        y_old = y
        v_old = v
        LTv_old = LTv

    # end
    elapsed = timer() - t_start

    if log:
        print(
            f"convergence = {criteria.all()}, ",
            f" iterations = {it}, ",
            f"elapsed time = {str(timedelta(seconds=elapsed))}",
        )
        print(
            f"primal residual = {rec_primal_residual[-1] / tol_primal:.2e} tol, "
            f" dual residual = {rec_dual_residual[-1] / tol_dual:.2e} tol"
        )

    # evaluate the utilities
    U_second, S_second = compute_rents(y_mat, theta_mat, params, tol_fp=1e-6)

    with open(cast(Path, model.resdir) / ".txt", "a") as f:
        f.write(f"N = {N}, tau = {tau:.2e}, sig = {sig:.2e}, ")
        f.write(f"step ratio = {stepratio}, primal tol = {tol_primal:.2e}, ")
        f.write(
            f"dual tol = {tol_dual:.2e}\n",
        )
        f.writelines(
            ", ".join(
                [
                    key + f" = {value}"
                    for key, value in zip(model.params_names, model.params, strict=True)
                ]
            )
            + "\n"
        )
        f.write(f"convergence = {criteria.all()}, iterations = {it}, ")
        f.write(
            f"elapsed time = {timedelta(seconds=elapsed)}\n",
        )
        f.write(f"primal residual = {rec_primal_residual[-1] / tol_primal:.2e} tol, ")
        f.write(f" dual residual = {rec_dual_residual[-1] / tol_dual:.2e} tol\n")
        f.write("\n")

    v_mat = v.reshape((N, N))

    # the original version had IC binding where v_mat > 0; I added a tolerance
    v_tol = tol_dual
    IC_binding = np.argwhere(v_mat > v_tol).astype(int)

    model_results = ScreeningResults(
        model=model,
        SB_y=y_mat,
        v_mat=v_mat,
        IR_binds=IR_binding,
        IC_binds=IC_binding,
        rec_primal_residual=rec_primal_residual,
        rec_dual_residual=rec_dual_residual,
        rec_it_proj=rec_it_proj,
        it=it,
        elapsed=elapsed,
        info_rents=U_second,
        SB_surplus=S_second,
    )

    return model_results


@njit("float64[:](float64[:,:], float64[:,:,:])")
def make_LTv(v_mat: np.ndarray, JLy: np.ndarray) -> np.ndarray:
    """creates $(\\Lambda^\\prime(y))^\ast v$

    Args:
        v_mat: an `(N, N)` matrix
        JLy: an `(m,N,N)` array

    Returns:
        an $m N$-vector
    """
    m = JLy.shape[0]
    return np.concatenate([np.sum(JLy[i, :, :] * v_mat, 0) for i in range(m)])


def compute_rents(
    y_second_best: np.ndarray,
    theta_mat: np.ndarray,
    params: list,
    tol_fp: float = 1e-6,
) -> TwoArrays:
    """Computes the rents for each type using the iterative algorithm $T_{\\Lambda}$ of Prop 2

    Args:
        y_second_best: the $(N,m)$-matrix of optimal contracts
        theta_mat: the $(N,d)$-matrix of types
        params: the parameters of the model
        tol_fp: tolerance for fixed point

    Returns:
        U_vals: an $N$-vector of rents
        S: an $N$-vector of the values of the joint surplus
    """
    N = y_second_best.shape[0]
    d = theta_mat.shape[1]
    y_second = np.concatenate([y_second_best[:, i] for i in range(d)])
    Lambda_vals = nlLambda(y_second, theta_mat, params).reshape((N, N))
    S_vals = S_function(y_second_best, theta_mat, params)
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
