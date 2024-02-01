from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import scipy.linalg as spla
from bs_python_utils.bsutils import bs_error_abort, check_matrix, check_vector


@dataclass
class ScreeningModel:
    """Create a model."""

    f: np.ndarray  # the numbers of individuals in each type
    model_id: str
    theta_mat: np.ndarray  # the characteristics of the types
    params: list
    params_names: list
    resdir: Path
    plotdir: Path

    N: int = field(init=False)  # the number of types
    d: int = field(init=False)  # their dimension
    m: int = field(init=False)  # the dimension of the contracts
    v0: np.ndarray = field(init=False)
    y_init: np.ndarray = field(init=False)
    free_y: list = field(init=False)
    norm_Lambda: float = field(init=False)
    M: float = field(init=False)
    FB_y: np.ndarray = field(init=False)

    def __post_init__(self):
        self.N, self.d = check_matrix(self.theta_mat)
        N_f = check_vector(self.f)
        if N_f != self.N:
            bs_error_abort(
                f"Wrong number of rows of f: {N_f} but we have {self.N} types"
            )
        self.v0 = np.zeros((self.N, self.N), dtype=np.float64)

    def add_first_best(self, y_first_best: np.ndarray):
        self.FB_y = y_first_best

    def initialize(self, y_init: np.ndarray, free_y: list, JLy: np.ndarray):
        self.y_init = y_init
        self.free_y = free_y
        self.norm_Lambda = max([spla.svdvals(JLy[i:, :]) for i in range(self.d)])

    def rescale_step(self, mult_fac: float) -> None:
        self.M = 2.0 * (self.N - 1) * mult_fac


@dataclass
class ScreeningResults:
    """Simulation results."""

    model: ScreeningModel
    SB_y: np.ndarray
    v_mat: np.ndarray
    IR_binds: np.ndarray
    IC_binds: np.ndarray
    rec_primal_residual: list
    rec_dual_residual: list
    rec_it_proj: list
    it: int
    elapsed: float
    info_rents: np.ndarray
    SB_surplus: np.ndarray

    def output_results(self) -> None:
        """prints the optimal contracts, and saves a dataframe

        Args:
            self: the `Results`.
        """
        model = self.model
        theta_mat, y_mat = model.theta_mat, self.SB_y
        df_output = pd.DataFrame(
            {
                "theta_0": model.theta_mat[:, 0],
                "y_0": y_mat[:, 0],
                "FB_y_0": model.FB_y[:, 0],
            }
        )
        d, m = theta_mat.shape[1], y_mat.shape[1]
        for i in range(d):
            df_output[f"theta_{i}"] = model.theta_mat[:, i]
        for i in range(m):
            df_output[f"y_{i}"] = y_mat[:, i]
            df_output[f"FB_y_{i}"] = model.FB_y[:, i]
        df_output["SB_surplus"] = self.SB_surplus
        df_output["info_rents"] = self.info_rents
        with pd.option_context(  # 'display.max_rows', None,
            "display.max_columns",
            None,
            "display.precision",
            3,
        ):
            print(df_output)

        model_resdir = cast(Path, model.resdir)
        np.savetxt(model_resdir / "IC_binds.txt", self.IC_binds)
        np.savetxt(model_resdir / "IR_binds.txt", self.IR_binds)
        np.savetxt(model_resdir / "v_mat.txt", self.v_mat)
        df_params = pd.DataFrame()
        for k, v in zip(model.params_names, model.params, strict=True):
            df_params[k] = [v]
        df_params.to_csv(model_resdir / "params.csv")
