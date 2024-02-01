"""Example with two-dimensional types (sigma=risk-aversion, delta=risk)
    and two-dimensional contracts  (y0=deductible, y1=proportional copay)
"""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from bs_python_utils.bsutils import print_stars

from .solver import JLambda, solve
from .specif import first_best, initialize_contracts, model_name, plot, setup_model

if __name__ == "__main__":
    print_stars(f"Running model {model_name}")

    model = setup_model(model_name)

    do_first_best = True
    do_solve = True

    start_from_first_best = True
    start_from_current = not start_from_first_best

    if do_first_best:
        # First let us look at the first best: we choose $y_i$ to maximize $S_i$ for each $i$.
        y_first_best_mat, FB_surplus = first_best(model)
    else:
        y_first_best_mat = pd.read_csv(
            cast(Path, model.resdir) / f"first_best_{model.model_id}.csv"
        )[[f"y_{i}" for i in range(model.m)]].values

    model.add_first_best(y_first_best_mat)

    if do_solve:  # we solve for the second best
        rng = np.random.default_rng(1234)

        # initial values
        y_init, free_y = initialize_contracts(model, start_from_first_best)
        JLy = JLambda(y_init, free_y, model.theta_mat, model.params)
        model.initialize(y_init, free_y, JLy)

        results = solve(
            model,
            warmstart=True,
            scale=True,
            it_max=1_000_000,
            stepratio=1.0,
            tol_primal=1e-4,
            tol_dual=1e-4,
            fix_top=True,
            free_y=free_y,
        )

        results.output()

        plot(results)
