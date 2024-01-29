"""Algorithm for two-dimensional types (sigma=risk-aversion, delta=risk)
    and two-dimensional contracts  (y0=deductible, y1=proportional copay)
"""

import multiprocessing as mp
from dataclasses import dataclass, field
from datetime import timedelta
from math import sqrt
from timeit import default_timer as timer
from typing import Any, cast

import numpy as np
import pandas as pd
import scipy.linalg as spla
from bs_python_utils.bs_opt import minimize_free
from bs_python_utils.bsutils import bs_error_abort
from numba import float64, int32, njit, prange
from numba_stats import norm

from .utils import L2_norm, construct_D, mult_fac, n_procs
from .values import S_fun, b_fun, d0_S_fun, d1_S_fun, db_fun, val_D


@dataclass
class InsuranceModel:
    """Create a model for 2D insurance."""

    f: np.ndarray
    model_id: str
    theta_mat: np.ndarray
    param: dict
    resdir: str
    y_init: np.ndarray
    y_first_best_mat: np.ndarray

    s: float = field(init=False)
    loading: float = field(init=False)
    N: int = field(init=False)
    v0: np.ndarray = field(init=False)
    M: float = field(init=False)
    norm_Lambda: float = field(init=False)
    save_path: str = field(init=False)

    def __post_init__(self):
        self.s = self.param["s"]
        self.loading = self.param["loading"]
        self.N = self.f.size
        self.v0 = np.zeros((self.N, self.N), dtype=np.float64)
        JLy = JLambda_j(self.y_init, self.theta_mat, self.s)
        s0 = spla.svdvals(JLy[0, :, :])
        s1 = spla.svdvals(JLy[1, :, :])
        self.norm_Lambda = max(s0[0], s1[0])
        self.M = 2.0 * (self.N - 1) * mult_fac
        print(f"model id = {self.model_id}\n")
        self.save_path = f"{self.resdir}/{self.model_id}"


@dataclass
class InsuranceResults:
    """Simulation results."""

    model: InsuranceModel
    y_mat: np.ndarray
    v_mat: np.ndarray
    IR_binds: np.ndarray
    IC_binds: np.ndarray
    rec_primal_residual: list
    rec_dual_residual: list
    rec_it_proj: list
    it: int
    elapsed: float


# def JLambda(y: np.ndarray, insur: InsuranceModel) -> np.ndarray:
#     """computes $\Lambda^\prime_{ij}(y) = b^\prime_j(y_i)-b^\prime_i(y_i)$

#     Args:
#         y: the contracts, an array of size $2 N$
#         insur: the model

#     Returns:
#         a $(2, N, N)$ array.
#     """
#     theta0, theta1 = insur.theta_mat[0, :], insur.theta_mat[1, :]
#     N, s = insur.N, insur.s
#     # we compute the (N, N) matrices db_i/dy_0(y_j) and db_i/dy_1(y_j)
#     d0_b_vals = np.vstack(
#         [d0_b_fun(y, theta0[i], theta1[i], s) for i in range(N)], dtype=np.float64
#     )
#     d1_b_vals = np.vstack(
#         [d1_b_fun(y, theta0[i], theta1[i], s) for i in range(N)], dtype=np.float64
#     )
#     d0_b_ii = np.diag(d0_b_vals)
#     d1_b_ii = np.diag(d1_b_vals)
#     a0 = np.vstack(
#         [d0_b_vals[i, :] - d0_b_ii for i in range(N)],
#         dtype=np.float64,
#     ).T
#     a1 = np.vstack(
#         [d1_b_vals[i, :] - d1_b_ii for i in range(N)],
#         dtype=np.float64,
#     ).T
#     J = np.zeros(
#         (2, N, N),
#         dtype=np.float64,
#     )
#     J[0, :, :] = a0
#     J[1, :, :] = a1
#     return J


# @njit("float64[:,:,:](float64[:], float64[:, :], float64)")
def JLambda_j(y: np.ndarray, theta_mat: np.ndarray, s: float) -> np.ndarray:
    """computes $\Lambda^\prime_{ij}(y) = b^\prime_i(y_j)-b^\prime_j(y_j)$

    Args:
        y: the contracts, an array of size $2 N$
        theta_mat: the types, a `(N, 2)` matrix
        s: the dispersion of the losses

    Returns:
        a $(2, N, N)$ array.
    """
    theta0, theta1 = theta_mat[:, 0], theta_mat[:, 1]
    N = theta0.size
    # we compute the (N, N) matrices db_i/dy_0(y_j) and db_i/dy_1(y_j)
    db_vals = db_fun(y, theta0, theta1, s)
    d0_b_vals = db_vals[0, :, :]
    d1_b_vals = db_vals[1, :, :]
    J = np.zeros((2, N, N))
    J[0, :, :] = d0_b_vals - np.diag(d0_b_vals)
    J[1, :, :] = d1_b_vals - np.diag(d1_b_vals)
    return J


def prox_H(i, z, sigma, delta, t, s, loading, y_first_best_i) -> np.ndarray | None:
    """Proximal operator of -t S_i at z;
        minimizes $-S_i(y) + 1/(2 t) \lVert y-z \rVert^2$

    Args:
        i: the index of the type
        z: a 2-vector
        sigma, delta: type $i$'s risk-aversion and risk
        t: scale factor
        s: dispersion of individual losses
        loading: loading factor
        y_first_best_i: a 2-vector, the first-best contract for this type

    Returns:
        the minimizing $y$, a 2-vector
    """

    def prox_obj_and_grad(
        y: np.ndarray, args: list, gr: bool = False
    ) -> float | tuple[float, np.ndarray]:
        dyz = y - z
        dist_yz2 = np.sum(dyz * dyz)
        obj = -S_fun(y, np.array([sigma]), np.array([delta]), s, loading)[
            0, 0
        ] + dist_yz2 / (2 * t)
        if gr:
            grad = (
                -np.array(
                    [
                        d0_S_fun(y, np.array([sigma]), np.array([delta]), s, loading)[
                            0, 0
                        ],
                        d1_S_fun(y, np.array([sigma]), np.array([delta]), s, loading)[
                            0, 0
                        ],
                    ]
                )
                + (y - z) / t
            )
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
    """computes $v_{ij} = \max(w_{ij}-\beta_i +\beta_j, 0)$

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
        an $N^2$-vector
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


def first_best(
    theta_mat: np.ndarray, s: float, loading: float, resdir: str
) -> np.ndarray:
    """computes the first-best contracts for all types

    Args:
        theta_mat: the `(N, 2)` matrix of types
        s: the dispersion of the losses
        loading: the loading factor
        resdir: the directory where to save the results

    Returns:
        y_first_best_mat: the `(N, 2)` matrix of first-best contracts
    """
    sigmas, deltas = theta_mat[:, 0], theta_mat[:, 1]
    N = sigmas.size

    def minus_S_fun_loc(y, args):
        sigma, delta, s, loading = args
        sigma_vec = np.array([sigma])
        delta_vec = np.array([delta])
        # print(f"{S_fun(y, sigma_vec, delta_vec, s, loading)=}")
        return -S_fun(y, sigma_vec, delta_vec, s, loading)[0, 0]

    def minus_dS_fun_loc(y, args):
        sigma, delta, s, loading = args
        sigma_vec = np.array([sigma])
        delta_vec = np.array([delta])
        return -np.array(
            [
                d0_S_fun(y, sigma_vec, delta_vec, s, loading)[0, 0],
                d1_S_fun(y, sigma_vec, delta_vec, s, loading)[0, 0],
            ]
        )

    x0 = np.array([1.0, 0.3])
    f0 = minus_S_fun_loc(x0, args=[0.3, 2.0, s, loading])

    # def minus_SdS_fun_loc(y, args, gr=False):
    # if gr:
    #     return minus_S_fun_loc(y, args), minus_dS_fun_loc(y, args)
    # else:
    #     return minus_S_fun_loc(y, args)
    # check_gradient_scalar_function(minus_SdS_fun_loc, x0, args=[0.3, 2.0, s, loading])

    df_first_best = pd.DataFrame(
        {
            "sigma": sigmas,
            "delta": deltas,
            "Proba accident": norm.cdf(deltas / s, 0.0, 1.0),
        }
    )

    df_first_best["Expected positive loss"] = (
        s * norm.pdf(deltas / s, 0.0, 1.0) / df_first_best["Proba accident"] + deltas
    )

    y_first = np.empty((N, 2))
    actuarial_premium = np.empty(N)
    surplus = np.empty(N)
    for i in range(N):
        sigma, delta = sigmas[i], deltas[i]
        res_i = minimize_free(
            minus_S_fun_loc,
            minus_dS_fun_loc,
            x_init=x0,
            args=[sigma, delta, s, loading],
            # bounds=[(MIN_Y0, MAX_Y0), (MIN_Y1, MAX_Y1)],
        )
        y_first[i, :] = res_i.x
        y_i0 = y_first[i, 0]
        y_i1 = y_first[i, 1]

        if i % 10 == 0:
            print(f" Done {i=} types out of {N}")
            print(f"\ni={i}, sigma={sigma}, delta={delta}:")
            print(f"\t\t status {res_i.status}; at {y_i0: > 10.4f}, {y_i1: > 10.4f}\n")

        delta_vec = np.array([delta])
        actuarial_premium[i] = val_D(y_first[i, :], delta_vec, s)
        surplus[i] = -minus_S_fun_loc(y_first[i, :], args=[sigma, delta, s, loading])

    df_first_best["y_0"] = y_first[:, 0]
    df_first_best["y_1"] = y_first[:, 1]
    df_first_best["Surplus"] = surplus
    df_first_best["Actuarial premium"] = actuarial_premium

    # print(df_first_best)
    df_first_best.to_csv(f"{resdir}/insurance_first_best_{N}.csv")
    y_first_best_mat = np.column_stack((df_first_best.y_0, df_first_best.y_1))
    return cast(np.ndarray, y_first_best_mat)


# def nlLambda(
#     y: np.ndarray,
#     insur: InsuranceModel,
# ) -> np.ndarray:
#     """computes $\Lambda_{ij}(y) = b_i(y_j)-b_j(y_j)$

#     Args:
#         y: the contracts, an array of size $2 N$ (`y_0` then `y_1`)
#         insur: the model

#     Returns:
#         an $N^2$ vector.
#     """
#     N = insur.N
#     theta0, theta1 = insur.theta_mat[:, 0], insur.theta_mat[:, 1]
#     s = insur.s
#     b_vals = np.array(
#         [b_fun(y, theta0[i], theta1[i], s) for i in range(N)],
#         dtype=np.float64,
#     )
#     b_jj = np.diag(b_vals)
#     return np.vstack(
#         [b_vals[i, :] - b_jj for i in range(N)],
#         dtype=np.float64,
#     ).reshape(N * N)


@njit("float64[:,:](float64[:], float64[:, :], float64)", parallel=True)
def nlLambda_j(
    y: np.ndarray,
    theta_mat: np.ndarray,
    s: float,
) -> Any:
    """computes $\Lambda_{ij}(y) = b_i(y_j)-b_j(y_j)$

    Args:
        y: the contracts, a vector of size $2 N$ (`y_0` then `y_1`)
        theta_mat: the types, a matrix of size $(N, 2)$
        s: the dispersion of the losses

    Returns:
        an $(N,N)$ matrix.
    """
    N = theta_mat.shape[0]
    theta0, theta1 = theta_mat[:, 0], theta_mat[:, 1]
    b_vals = b_fun(y, theta0, theta1, s)
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
    insur: InsuranceModel,
    z: np.ndarray,
    tau: float,
    use_multiprocessing: bool = False,
    fix_top: bool = False,
    free_y: list | None = None,
) -> np.ndarray:
    """Proximal operator of -S(y) = sum_i f_i H(y_i, theta_i)

    Args:
        insur: the model
        z: an `N`-vector
        tau: scale factor
        use_multiprocessing: not used now
        fix_top: True if first-best imposed at top
        maybe_insured: a list of types that are not definitely not insured

    Returns:
        the minimizing `y`, a $2 N$-vector
    """
    N = insur.N
    theta0, theta1 = insur.theta_mat[:, 0], insur.theta_mat[:, 1]
    f, s, loading = (
        insur.f,
        insur.s,
        insur.loading,
    )
    y_first_best = insur.y_first_best_mat
    list_args = [
        [
            i,
            np.array([z[i], z[N + i]]),
            theta0[i],
            theta1[i],
            tau * f[i],
            s,
            loading,
            np.array(y_first_best[i, 0], y_first_best[i, 1]),
        ]
        for i in range(N)
    ]

    # by default no one is insured
    y = np.concatenate((np.zeros(N), np.ones(N)))
    # these are the types we will be working with
    Nmax = N - 1 if fix_top else N
    if free_y:
        working_i0 = [i for i in free_y if i < Nmax]
    else:
        working_i0 = list(range(Nmax))
    working_i1 = [N + i for i in working_i0]

    list_working = [list_args[i] for i in working_i0]
    n_working = len(list_working)

    # print(f"{working_i0=}")

    if fix_top:
        # we fix the second-best at the first-best at the top
        y[N - 1], y[-1] = y_first_best[-1, 0], y_first_best[-1, 1]

    if use_multiprocessing:
        pool = mp.Pool(n_procs)
        res = pool.map(prox_work_func, list_working, chunksize=20)
        pool.close()
        pool.join()
        y[working_i0] = [res[i][0] for i in range(n_working)]
        y[working_i1] = [res[i][1] for i in range(n_working)]
    else:
        # for i in working_i0:
        #     arg_i = list_args[i]
        #     res = prox_H(*arg_i)  # type: ignore
        #     if res is None:
        #         bs_error_abort(f"Proximal operator did not converge for {i=}")
        #     else:
        #         y[i], y[N + i] = res[0], res[1]
        res = prox_work_func(list_working)
        y[working_i0] = [res[i][0] for i in range(n_working)]
        y[working_i1] = [res[i][1] for i in range(n_working)]

    return y


def proj_K(
    insur: InsuranceModel,
    w: np.ndarray,
    lamb: np.ndarray,
    gamma_proj: float,
    warmstart: bool = True,
    atol_proj: float = 1e-6,
    rtol_proj: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int] | None:
    """Projection of $w$ onto $K$ by Fast Projected Gradient

    Args:
        insur: the model
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
    N = insur.N
    eta = insur.f

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
    insur: InsuranceModel,
    use_multiprocessing: bool = False,
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
):
    # initialization
    N = insur.N
    N2 = N * N
    _, gamma_proj = construct_D(N)
    t_start = timer()
    it = 0
    criteria = np.array([False, False, False])
    rec_primal_residual = []
    rec_dual_residual = []
    rec_it_proj = []

    v = insur.v0.reshape(N * N)
    y = insur.y_init
    theta_mat = insur.theta_mat
    s = insur.s

    # scaling of the tolerances
    if scale:
        tol_primal = sqrt(len(y)) * tol_primal
        tol_dual = sqrt(len(v)) * tol_dual

    tau = 1.0 / (sqrt(stepratio) * insur.norm_Lambda) / mult_fac
    sig = sqrt(stepratio) / insur.norm_Lambda / mult_fac

    prox_F = prox_minusS

    JLy = JLambda_j(y, theta_mat, s)  # this is (Lambda'(y))^*
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
            insur,
            y_old - tau * LTv_old,
            tau,
            use_multiprocessing=use_multiprocessing,
            fix_top=fix_top,
            free_y=free_y,
        )
        # print(f" in proj {y[25]=}")
        # print("\t" * 10, f"change in norm {spla.norm(y - y_old) =}")
        # dual update
        y_bar = y + t_acc * (y - y_old)
        Ly_bar = nlLambda_j(y_bar, theta_mat, s).reshape(N2)
        proj_res = cast(
            tuple[np.ndarray, np.ndarray, np.ndarray, int, int],
            proj_K(
                insur,
                v + sig * Ly_bar,
                lamb,
                gamma_proj,
                warmstart=warmstart,
            ),
        )
        v, lamb, IR_binding, n_it_proj, proj_converged = proj_res
        Ly = nlLambda_j(y_bar, theta_mat, s).reshape(N2)
        JLy = JLambda_j(y, theta_mat, s)
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
        if it % 100 == 0 and log:
            print("\n\ty is:")
            for i in range(N):
                print(f"{y[i]: >10.4f}, {y[N+i]: >10.4f}")
            np.savetxt(
                f"{insur.resdir}/current_y_{N}.txt",
                np.round(np.column_stack((y[:N], y[N:])), 4),
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

    with open(insur.save_path + ".txt", "a") as f:
        f.write(f"N = {N}, tau = {tau:.2e}, sig = {sig:.2e}, ")
        f.write(f"step ratio = {stepratio}, primal tol = {tol_primal:.2e}, ")
        f.write(
            f"dual tol = {tol_dual:.2e}\n",
        )
        f.writelines(
            ", ".join([key + f" = {value}" for key, value in insur.param.items()])
            + "\n"
        )
        f.write(f"convergence = {criteria.all()}, iterations = {it}, ")
        f.write(
            f"elapsed time = {str(timedelta(seconds=elapsed))}\n",
        )
        f.write(f"primal residual = {rec_primal_residual[-1] / tol_primal:.2e} tol, ")
        f.write(f" dual residual = {rec_dual_residual[-1] / tol_dual:.2e} tol\n")
        f.write("\n")

    y_mat = np.column_stack((y[:N], y[N:]))
    v_mat = v.reshape((N, N))

    # the original version had IC binding where v_mat > 0; I added a tolerance
    v_tol = tol_dual
    IC_binding = np.argwhere(v_mat > v_tol).astype(int)

    insur_results = InsuranceResults(
        model=insur,
        y_mat=y_mat,
        v_mat=v_mat,
        IR_binds=IR_binding,
        IC_binds=IC_binding,
        rec_primal_residual=rec_primal_residual,
        rec_dual_residual=rec_dual_residual,
        rec_it_proj=rec_it_proj,
        it=it,
        elapsed=elapsed,
    )

    return insur_results


@njit("float64[:](float64[:,:], float64[:,:,:])")
def make_LTv(v_mat: np.ndarray, JLy: np.ndarray) -> np.ndarray:
    """creates $(\Lambda^\prime(y))^\ast v$

    Args:
        v_mat: an `(N, N)` matrix
        JLy: a `(2,N,N)` array

    Returns:
        a $2 N$-vector
    """
    LTv0 = np.sum(JLy[0, :, :] * v_mat, 0)
    LTv1 = np.sum(JLy[1, :, :] * v_mat, 0)
    return np.concatenate((LTv0, LTv1))


def output_results(simres: InsuranceResults) -> None:
    """prints the optimal contracts, and saves a dataframe

    Args:
        simres: the `InsuranceResults`.
    """
    insur = simres.model
    df_output = pd.DataFrame(
        {
            "sigma": insur.theta_mat[:, 0],
            "delta": insur.theta_mat[:, 1],
            "y_0": simres.y_mat[:, 0],
            "y_1": simres.y_mat[:, 1],
            "FB_y_0": insur.y_first_best_mat[:, 0],
            "FB_y_1": insur.y_first_best_mat[:, 1],
        }
    )
    with pd.option_context(  # 'display.max_rows', None,
        "display.max_columns",
        None,
        "display.precision",
        3,
    ):
        print(df_output)

    np.savetxt(insur.save_path + "_IC_binds.txt", simres.IC_binds)
    np.savetxt(insur.save_path + "_IR_binds.txt", simres.IR_binds)
    np.savetxt(insur.save_path + "_v_mat.txt", simres.v_mat)

    df_param = pd.DataFrame.from_dict(insur.param.items())
    with pd.ExcelWriter(insur.save_path + ".xlsx") as writer:
        df_output.round(3).to_excel(writer, sheet_name="output")
        df_param.to_excel(writer, sheet_name="parameters")
    df_output.round(3).to_csv(insur.save_path + ".csv")
