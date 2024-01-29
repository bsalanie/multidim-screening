import numpy as np
import pandas as pd
from bs_python_utils.bsutils import bs_error_abort, mkdir_if_needed

from .plots import (
    display_variable,
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
from .surplus import compute_rents
from .utils import plots_dir, results_dir

for n in [20]:
    n0 = n1 = n
    N = n0 * n1
    f = np.ones(N)  # weights of distribution
    s = 4.0  # dispersion of individual losses
    loading = 0.25  # loading factor

    plotdir = mkdir_if_needed(plots_dir + f"/{n0}x{n1}new")
    resdir = results_dir + f"/{n0}x{n1}new"
    df_first_best = pd.read_csv(f"{resdir}/insurance_first_best_{N}.csv").rename(
        columns={
            "y0": "Deductible",
            "y1": "Copay",
            "sigma": "Risk-aversion",
            "delta": "Risk location",
        }
    )

    # first plot the first best
    y_first_best = df_first_best[["Deductible", "Copay"]].values
    theta_mat = df_first_best[["Risk-aversion", "Risk location"]].values

    plot_calibration(df_first_best, path=f"{plotdir}/insurance_calibration_{N}")

    df_second_best = pd.read_csv(f"{resdir}/insurance_second_best_{N}.csv").rename(
        columns={
            "y_0": "Deductible",
            "y_1": "Copay",
            "sigma": "Risk-aversion",
            "delta": "Risk location",
        }
    )
    y_second_best = df_second_best[["Deductible", "Copay"]].values

    # print(df_first_best.columns)
    # print(df_second_best.columns)

    #

    ## put FB and SB together
    df_first = pd.DataFrame(
        df_first_best, columns=["Risk-aversion", "Risk location", "Deductible", "Copay"]
    )
    df_first["Model"] = "First-best"
    df_second = pd.DataFrame(
        df_second_best,
        columns=["Risk-aversion", "Risk location", "Deductible", "Copay"],
    )
    df_second["Model"] = "Second-best"
    for col in ["Risk-aversion", "Risk location", "Deductible", "Copay"]:
        df_first[col] = df_first[col].round(3)
        df_second[col] = df_second[col].round(3)
    print(df_first["Risk-aversion"].unique())
    print(df_second["Risk-aversion"].unique())
    df_first_and_second = pd.concat((df_first, df_second))

    display_variable(
        y_first_best[:, 0],
        theta_mat,
        cmap="viridis",
        cmap_label=r"First-best deductible $y_0$",
        path=f"{plotdir}/insurance_first_best_deduc_{N}",
    )

    second_best_copay = np.round(y_second_best[:, 1], 3)
    # max_deduc = np.max(y_second_best[second_best_copay < 0.99, 0])
    # second_best_deduc = y_second_best[:, 0] if second_best_copay < 0.99 else max_deduc
    no_insurance = np.argwhere(second_best_copay > 0.99)

    plot_contract_models(
        df_first_and_second, "Deductible", path=f"{plotdir}/insurance_deducs_models_{N}"
    )

    plot_contract_models(
        df_first_and_second, "Copay", path=f"{plotdir}/insurance_copays_models_{N}"
    )

    plot_contract_riskavs(
        df_first_and_second,
        "Deductible",
        path=f"{plotdir}/insurance_deducs_riskavs_{N}",
    )

    plot_contract_riskavs(
        df_first_and_second, "Copay", path=f"{plotdir}/insurance_copays_riskavs_{N}"
    )

    plot_copays(df_second_best, path=f"{plotdir}/insurance_copays_{N}")

    plot_best_contracts(
        df_first_and_second,
        # title="Optimal contracts",
        path=f"{plotdir}/insurance_optimal_contracts_{N}",
    )

    plot_y_range(df_first_and_second, path=f"{plotdir}/insurance_y_range_{N}")

    plot_second_best_contracts(
        theta_mat,
        y_second_best,
        title="Second-best contracts",
        cmap="viridis",
        path=f"{plotdir}/insurance_second_best_contracts_{N}",
    )

    IR_binds = (
        np.loadtxt(resdir + f"/insurance_second_best_{N}_IR_binds.txt")
        .astype(int)
        .tolist()
    )
    print(IR_binds)

    IC_binds = (
        np.loadtxt(resdir + f"/insurance_second_best_{N}_IC_binds.txt")
        .astype(int)
        .tolist()
    )

    plot_constraints(
        theta_mat, IR_binds, IC_binds, path=f"{plotdir}/insurance_constraints_{N}"
    )

    # evaluate the utilities
    U_second, S_second = compute_rents(
        y_second_best, theta_mat, s, loading, tol_fp=1e-6
    )
    profits_second = S_second - U_second
    S_first = df_first_best["Surplus"].values

    plot_utilities(
        theta_mat,
        S_first,
        S_second,
        U_second,
        path=f"{plotdir}/insurance_utilities_{N}",
    )
