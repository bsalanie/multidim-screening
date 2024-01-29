"""Example with two-dimensional types (sigma=risk-aversion, delta=risk)
    and two-dimensional contracts  (y0=deductible, y1=proportional copay)
"""

import multiprocessing as mp

import numpy as np
import pandas as pd
from bs_python_utils.bsutils import mkdir_if_needed, print_stars

from .INSUR import InsuranceModel, first_best, output_results, solve
from .utils import MAX_Y0, MAX_Y1, MIN_Y0, MIN_Y1, results_dir, use_multiprocessing

if __name__ == "__main__":
    print_stars("Running insurance_ex.py")
    suffix = "new"
    sigma_min = 0.2
    sigma_max = 0.5
    delta_min = -7.0
    delta_max = -3.0

    do_first_best = False
    do_solve = True

    start_from_first_best = False
    start_from_current = not start_from_first_best

    if use_multiprocessing:
        mp.freeze_support()
    # number of types in each dimension
    n0 = n1 = 20
    resdir = mkdir_if_needed(results_dir + f"/{n0}x{n1}new")

    # risk-aversions and risk location parameter (unit=1,000 euros)
    sigma, delta = np.linspace(sigma_min, sigma_max, num=n0), np.linspace(
        delta_min, delta_max, num=n1
    )
    theta0, theta1 = np.meshgrid(sigma, delta)
    theta_mat = np.column_stack(
        (theta0.flatten(), theta1.flatten())
    )  # is a N x 2 matrix
    N = theta_mat.shape[0]  # number of types N = n0 * n1

    f = np.ones(N)  # weights of distribution

    # model parameters setting
    s = 4.0  # dispersion of individual losses
    loading = 0.25  # loading factor
    param = {"s": s, "loading": loading}
    model_id = f"insurance_second_best_{N}"

    if do_first_best:
        # First let us look at the first best: we choose $y_i$ to maximize $S_i$ for each $i$.
        y_first_best_mat = first_best(theta_mat, s, loading, resdir)
    else:
        y_first_best_mat = pd.read_csv(f"{resdir}/insurance_first_best_{N}.csv")[
            ["y0", "y1"]
        ].values

    if do_solve:  # we solve for the second best
        rng = np.random.default_rng(1234)

        # initial values
        if start_from_first_best:
            y_init = y_first_best_mat
            set_fixed_y: set[int] = set()
            set_not_insured: set[int] = set()
        else:
            y_init = np.loadtxt(f"{resdir}/current_y_{N}.txt")
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

        yinit_0 = np.clip(y_init[:, 0] + rng.normal(0, 0.00000, N), MIN_Y0, MAX_Y0)
        yinit_1 = np.clip(y_init[:, 1] + rng.normal(0, 0.00000, N), MIN_Y1, MAX_Y1)
        yinit_0[not_insured] = 0.0
        yinit_1[not_insured] = 1.0
        # yinit_1[only_deductible] = 0.0

        y_init = np.concatenate((yinit_0, yinit_1))

        model = InsuranceModel(
            f=f,
            model_id=model_id,
            theta_mat=theta_mat,
            param=param,
            resdir=resdir,
            y_init=y_init,
            y_first_best_mat=y_first_best_mat,
        )

        simul_results = solve(
            model,
            use_multiprocessing=use_multiprocessing,
            warmstart=True,
            scale=True,
            it_max=1_000_000,
            stepratio=1.0,
            tol_primal=1e-4,
            tol_dual=1e-4,
            fix_top=True,
            free_y=free_y,
        )

        output_results(simul_results)
