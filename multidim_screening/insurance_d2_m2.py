from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from bs_python_utils.bs_opt import minimize_free
from bs_python_utils.bsutils import mkdir_if_needed
from numba import njit
from numba_stats import norm

from .classes import ScreeningModel, ScreeningResults
from .insurance_d2_m2_plots import (
    plot_best_contracts,
    plot_calibration,
    plot_constraints,
    plot_contract_models,
    plot_contract_riskavs,
    plot_copays,
    plot_second_best_contracts,
    plot_utilities,
    plot_y_range,
)
from .insurance_d2_m2_values import (
    S_penalties,
    d0_S_fun,
    d0_val_B,
    d0_val_C,
    d1_S_fun,
    d1_val_C,
    multiply_each_col,
    split_y,
    val_D,
    val_I,
)
from .utils import display_variable, plots_dir, results_dir


def create_model(model_name: str) -> ScreeningModel:
    """initializes the ScreeningModel object:
    fills in the dimensions, the numbers in each type, the characteristics of the types,
    the model parameters, and the directories.

    Args:
        model_name: the name of the model

    Returns:
        the ScreeningModel object
    """
    # size of grid for types in each dimension
    n0 = n1 = 4
    N = n0 * n1
    suffix = ""
    case = f"{model_name}_N{N}{suffix}"
    resdir = mkdir_if_needed(results_dir / case)
    plotdir = mkdir_if_needed(plots_dir / case)

    sigma_min = 0.2
    sigma_max = 0.5
    delta_min = -7.0
    delta_max = -3.0
    # risk-aversions and risk location parameter (unit=1,000 euros)
    sigmas, deltas = np.linspace(sigma_min, sigma_max, num=n0), np.linspace(
        delta_min, delta_max, num=n1
    )
    theta0, theta1 = np.meshgrid(sigmas, deltas)
    theta_mat = np.column_stack(
        (theta0.flatten(), theta1.flatten())
    )  # is a N x 2 matrix

    f = np.ones(N)  # weights of distribution

    # model parameters setting
    s = 4.0  # dispersion of individual losses
    loading = 0.25  # loading factor
    params: list = [s, loading]
    model_id = f"{model_name}_{case}"
    return ScreeningModel(
        f=f,
        model_id=model_id,
        theta_mat=theta_mat,
        params=params,
        params_names=["s", "loading"],
        resdir=resdir,
        plotdir=plotdir,
    )


@njit("float64[:,:](float64[:], float64[:, :], List)")
def b_fun(y, theta_mat, params):
    """evaluates the value of the coverage

    Args:
        y:  a $2 k$-vector of $k$ contracts
        theta_mat: a $(q,2)$-vector of characteristics of types
        params: the parameters of the model

    Returns:
        a $(q,k)$-matrix
    """
    sigmas, deltas = theta_mat[:, 0], theta_mat[:, 1]
    s = params[0]
    return multiply_each_col(
        np.log(val_I(np.array([0.0, 1.0]), sigmas, deltas, s))
        - np.log(val_I(y, sigmas, deltas, s)),
        1.0 / sigmas,
    )


@njit("float64[:,:,:](float64[:], float64[:, :], List)")
def db_fun(y, theta_mat, params):
    """calculates both derivatives of the coverage

    Args:
        y:  a $2 k$-vector of $k$ contracts
        theta_mat: a $(q,2)$-vector of characteristics of types
        params: the parameters of the model

    Returns:
        a $(2,q,k)$-array
    """
    y_0, _ = split_y(y)
    sigmas, deltas = theta_mat[:, 0], theta_mat[:, 1]
    s = params[0]
    denom_inv = 1.0 / multiply_each_col(val_I(y, sigmas, deltas, s), sigmas)
    derivatives_b = np.empty((2, sigmas.size, y_0.size))
    derivatives_b[0, :, :] = (
        -(d0_val_B(y, sigmas, deltas, s) + d0_val_C(y, sigmas, deltas, s)) * denom_inv
    )
    derivatives_b[1, :, :] = -d1_val_C(y, sigmas, deltas, s) * denom_inv
    return derivatives_b


@njit("float64[:,:](float64[:], float64[:, :], List)")
def S_fun(y, theta_mat, params):
    """evaluates the joint surplus

    Args:
        y:  a $2 k$-vector of $k$ contracts
        theta_mat: a $(q,2)$-vector of characteristics of types
        params: the parameters of the model

    Returns:
        a $(q,k)$-matrix
    """
    deltas = theta_mat[:, 1]
    s, loading = params
    return (
        b_fun(y, theta_mat, params)
        - (1.0 + loading) * val_D(y, deltas, s)
        - S_penalties(y)
    )


@njit("float64[:,:,:](float64[:], float64[:, :], List)")
def dS_fun(y, theta_mat, params):
    """calculates both derivatives of the surplus

    Args:
        y:  a $2 k$-vector of $k$ contracts
        theta_mat: a $(q,2)$-vector of characteristics of types
        params: the parameters of the model

    Returns:
        a $(2,q,k)$-array
    """
    sigmas, deltas = theta_mat[:, 0], theta_mat[:, 1]
    s, loading = params
    dS = np.empty((2, sigmas.size, y.size))
    dS[0, :, :] = d0_S_fun(y, sigmas, deltas, s, loading)
    dS[1, :, :] = d1_S_fun(y, sigmas, deltas, s, loading)
    return dS


def create_first_best(model: ScreeningModel) -> np.ndarray:
    """computes the first-best contracts for all types

    Args:
        model: the screening model

    Returns:
        y_first_best_mat: the `(N, m)` matrix of first-best contracts
    """
    theta_mat = model.theta_mat
    params = model.params
    s, loading = params
    N = model.N
    sigmas, deltas = theta_mat[:, 0], theta_mat[:, 1]

    def minus_S_fun_loc(y, args):
        theta_mat1 = args[0]
        return -S_fun(y, theta_mat1, params)[0, 0]

    def minus_dS_fun_loc(y, args):
        theta_mat1 = args[0]
        return -dS_fun(y, theta_mat1, params)[0, 0]

    df_first_best = pd.DataFrame(
        {
            "theta_0": sigmas,
            "theta_1": deltas,
            "Proba accident": norm.cdf(deltas / s, 0.0, 1.0),
        }
    )

    df_first_best["Expected positive loss"] = (
        s * norm.pdf(deltas / s, 0.0, 1.0) / df_first_best["Proba accident"] + deltas
    )

    x0 = np.array([2.0, 0.3])
    y_first = np.empty((N, 2))
    actuarial_premium = np.empty(N)
    FB_surplus = np.empty(N)
    for i in range(N):
        sigma, delta = sigmas[i], deltas[i]
        res_i = minimize_free(
            minus_S_fun_loc,
            minus_dS_fun_loc,
            x_init=x0,
            args=np.array([sigma, delta]),
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
        FB_surplus[i] = -minus_S_fun_loc(y_first[i, :], args=[sigma, delta, s, loading])

    df_first_best["y_0"] = y_first[:, 0]
    df_first_best["y_1"] = y_first[:, 1]
    df_first_best["FB surplus"] = FB_surplus
    df_first_best["Actuarial premium"] = actuarial_premium

    # print(df_first_best)
    df_first_best.to_csv(cast(Path, model.resdir) / f"first_best_{N}.csv")
    y_first_best_mat = np.column_stack((df_first_best.y_0, df_first_best.y_1))
    return cast(np.ndarray, y_first_best_mat)


def create_initial_contracts(
    start_from_first_best: bool,
    resdir: Path,
    N: int,
    y_first_best_mat: np.ndarray | None = None,
) -> tuple[np.ndarray, list]:
    """Initializes the contracts for the second best problem (MODEL-DEPENDENT)

    Args:
        start_from_first_best: whether to start from the first best
        resdir: directory where the results are stored
        N: number of types
        y_first_best_mat: the `(N, m)` matrix of first best contracts. Defaults to None.

    Returns:
        tuple[np.ndarray, list]: initial contracts (an `(N,m)` matrix) and a list of types for whom
            the contracts are to be determined.
    """
    if start_from_first_best:
        y_init = y_first_best_mat
        set_fixed_y: set[int] = set()
        set_not_insured: set[int] = set()
    else:
        y_init = np.loadtxt(f"{resdir}/current_y.txt")
        EPS = 0.001
        set_not_insured = {i for i in range(N) if y_init[i, 1] > 1.0 - EPS}
        # not_insured2 = {i for i in range(N) if theta_mat[i, 1] <= -6.0}
        # not_insured3 = {
        #     i
        #     for i in range(N)
        #     if theta_mat[i, 1] <= -5.0 and theta_mat[i, 0] <= 0.425
        # }
        # not_insured4 = {
        #     i
        #     for i in range(N)
        #     if theta_mat[i, 1] <= -4.0 and theta_mat[i, 0] <= 0.35
        # }
        # set_only_deductible = {i for i in range(N) if (start_from_current and y_init[i, 0] < EPS)}
        # set_fixed_y = set_not_insured.union(
        #         set_only_deductible
        #     )
        set_fixed_y = set_not_insured

    set_free_y = set(range(N)).difference(set_fixed_y)
    fixed_y = list(set_fixed_y)
    free_y = list(set_free_y)
    not_insured = list(set_not_insured)
    # only_deductible = list(set_only_deductible)

    print(f"{free_y=}")
    print(f"{fixed_y=}")

    rng = np.random.default_rng(645)

    MIN_Y0, MAX_Y0 = 0.3, np.inf
    MIN_Y1, MAX_Y1 = 0.0, np.inf
    y_init = cast(np.ndarray, y_init)
    yinit_0 = np.clip(y_init[:, 0] + rng.normal(0, 0.00000, N), MIN_Y0, MAX_Y0)
    yinit_1 = np.clip(y_init[:, 1] + rng.normal(0, 0.00000, N), MIN_Y1, MAX_Y1)
    yinit_0[not_insured] = 0.0
    yinit_1[not_insured] = 1.0

    y_init = cast(np.ndarray, np.concatenate((yinit_0, yinit_1)))

    return y_init, free_y


def plot_results(results: ScreeningResults) -> None:
    model = results.model
    N = model.N
    model_resdir = cast(Path, model.resdir)
    model_plotdir = cast(Path, model.plotdir)
    df_first_best = pd.read_csv(model_resdir / f"first_best_{N}.csv").rename(
        columns={
            "y_0": "Deductible",
            "y_1": "Copay",
            "theta_0": "Risk-aversion",
            "theta_1": "Risk location",
        }
    )

    # first plot the first best
    y_first_best = df_first_best[["Deductible", "Copay"]].values.round(3)
    theta_mat = df_first_best[["Risk-aversion", "Risk location"]].values

    plot_calibration(df_first_best, path=model_plotdir / f"calibration_{N}")

    y_second_best = results.SB_y.round(3)

    ## put FB and SB together
    df_first = pd.DataFrame(
        df_first_best, columns=["Risk-aversion", "Risk location", "Deductible", "Copay"]
    )
    df_first["Model"] = "First-best"
    df_second = pd.DataFrame(
        {
            "Deductible": y_second_best[:, 0],
            "Copay": y_second_best[:, 1],
        }
    )
    df_second["Model"] = "Second-best"
    df_first_and_second = pd.merge((df_first, df_second))

    display_variable(
        y_first_best[:, 0],
        theta_mat,
        cmap="viridis",
        cmap_label=r"First-best deductible $y_0$",
        path=model_plotdir / f"first_best_deduc_{N}",
    )

    plot_contract_models(
        df_first_and_second, "Deductible", path=model_plotdir / f"deducs_models_{N}"
    )

    plot_contract_models(
        df_first_and_second, "Copay", path=model_plotdir / f"copays_models_{N}"
    )

    plot_contract_riskavs(
        df_first_and_second,
        "Deductible",
        path=model_plotdir / f"deducs_riskavs_{N}",
    )

    plot_contract_riskavs(
        df_first_and_second, "Copay", path=model_plotdir / f"copays_riskavs_{N}"
    )

    plot_copays(df_second, path=model_plotdir / f"copays_{N}")

    plot_best_contracts(
        df_first_and_second,
        path=model_plotdir / f"optimal_contracts_{N}",
    )

    plot_y_range(df_first_and_second, path=model_plotdir / f"y_range_{N}")

    plot_second_best_contracts(
        theta_mat,
        y_second_best,
        title="Second-best contracts",
        cmap="viridis",
        path=model_plotdir / f"second_best_contracts_{N}",
    )

    IR_binds = (
        np.loadtxt(model_resdir / f"second_best_{N}_IR_binds.txt").astype(int).tolist()
    )

    IC_binds = (
        np.loadtxt(model_resdir / f"second_best_{N}_IC_binds.txt").astype(int).tolist()
    )

    plot_constraints(
        theta_mat, IR_binds, IC_binds, path=model_plotdir / f"constraints_{N}"
    )

    U_second, S_second = results.info_rents, results.SB_surplus
    S_first = df_first_best["Surplus"].values

    plot_utilities(
        theta_mat,
        S_first,
        S_second,
        U_second,
        path=model_plotdir / f"utilities_{N}",
    )
