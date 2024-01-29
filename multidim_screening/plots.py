"""Plotting the first best deductible with two-dimensional types (risk-aversion, risk)
    and two-dimensional contracts  (deductible, proportional copay)
"""

from typing import Any, cast

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs_python_utils.bs_altair import _maybe_save

from .utils import drawArrow


def set_colors(list_vals: list[Any], interpolate: bool = False) -> alt.Scale:
    """sets the colors at values of a variable.

    Args:
        list_vals: the values of the variable
        interpolate: whether we interpolate

    Returns:
        the color scheme.
    """
    n_vals = len(list_vals)
    list_colors = [
        "red",
        "lightred",
        "orange",
        "yellow",
        "lightgreen",
        "green",
        "lightblue",
        "darkblue",
        "violet",
        "black",
    ]
    if n_vals == 3:
        list_colors = [list_colors[i] for i in [0, 5, 7]]
    elif n_vals == 4:
        list_colors = [list_colors[i] for i in [0, 3, 6, 9]]
    elif n_vals == 5:
        list_colors = [list_colors[i] for i in [0, 2, 4, 6, 8]]
    elif n_vals == 7:
        list_colors = [list_colors[i] for i in [0, 1, 2, 4, 6, 8, 9]]
    if interpolate:
        our_colors = alt.Scale(domain=list_vals, range=list_colors, interpolate="rgb")
    else:
        our_colors = alt.Scale(domain=list_vals, range=list_colors)

    return our_colors


def set_axis(variable: np.ndarray, margin: float = 0.05) -> tuple[float, float]:
    """sets the axis for a plot with a margin

    Args:
        variable: the values of the variable
        margin: the margin to add, a fraction of the range of the variable

    Returns:
        the min and max for the axis.
    """
    x_min, x_max = variable.min(), variable.max()
    scaled_diff = margin * (x_max - x_min)
    x_min -= scaled_diff
    x_max += scaled_diff
    return x_min, x_max


def plot_calibration(
    df_first_best: pd.DataFrame,
    title: str | None = None,
    path: str | None = None,
    **kwargs: dict | None,
) -> alt.Chart:
    dfm = pd.melt(
        df_first_best,
        id_vars=["Risk-aversion", "Risk location"],
        value_vars=[
            "Proba accident",
            "Expected positive loss",
            "Surplus",
            "Actuarial premium",
        ],
    )
    for var in ["Risk-aversion", "Risk location", "value"]:
        dfm[var] = dfm[var].round(2)
        risk_aversion_vals = sorted(dfm["Risk-aversion"].unique())

    our_colors = set_colors(risk_aversion_vals, interpolate=True)
    base = alt.Chart().encode(
        x=alt.X(
            "Risk location:Q",
            title="Risk location",
            scale=alt.Scale(domain=set_axis(dfm["Risk location"].values)),
        ),
        y=alt.Y("value:Q"),
        color=alt.Color("Risk-aversion:N", title="Risk-aversion", scale=our_colors),
        tooltip=["Risk-aversion", "Risk location", "value"],
    )
    ch_points = base.mark_point(filled=True, size=50)
    ch_lines = base.mark_line(strokeWidth=0.5)
    ch = (
        alt.layer(ch_points, ch_lines, data=dfm)
        .interactive()
        .facet(facet="variable:N", columns=2)
        .resolve_scale(y="independent")
    )
    if title:
        ch.properties(title=title)
    _maybe_save(ch, path)
    return cast(alt.Chart, ch)


def plot_utilities(
    theta_mat: np.ndarray,
    S_first: np.ndarray,
    S_second: np.ndarray,
    U_second: np.ndarray,
    path: str | None = None,
) -> alt.Chart:
    df2 = pd.DataFrame(theta_mat.round(3), columns=["Risk-aversion", "Risk location"])
    df2["First-best surplus"] = S_first
    df2["Second-best surplus"] = S_second
    df2["Lost surplus"] = S_first - S_second
    df2["Informational rent"] = U_second
    df2["Profit"] = S_second - U_second
    df2m = pd.melt(
        df2,
        id_vars=["Risk-aversion", "Risk location"],
        value_vars=["Informational rent", "Profit", "Lost surplus"],
    )
    # print(df2m)
    our_colors = alt.Scale(
        domain=["Informational rent", "Profit", "Lost surplus"],
        range=["blue", "green", "red"],
    )
    ch = (
        alt.Chart(df2m)
        .mark_bar()
        .encode(
            x=alt.X(
                "Risk-aversion:Q",
                scale=alt.Scale(domain=set_axis(df2["Risk-aversion"].values)),
            ),
            y=alt.Y("sum(value)"),
            color=alt.Color("variable", scale=our_colors),
            xOffset="variable:N",
        )
        .properties(width=150, height=120)
        .facet(facet="Risk location:N", columns=5)
    )
    _maybe_save(ch, path)


def plot_best_contracts(
    df_first_and_second: pd.DataFrame,
    title: str | None = None,
    path: str | None = None,
    **kwargs: dict | None,
) -> alt.Chart:
    """plots the optimal contracts for both first and second best  in the type space
    the size of a point is proportional to (1 minus the copay)
    the color of a point indicates the deductible

    Args:
        df_first_and_second: a dataframe
        title: the title of the plot
        path: the path to save the plot
        **kwargs: additional arguments to pass to the plot

    Returns:
          the two interactive scatterplots.
    """
    deduc = df_first_and_second.Deductible
    our_colors = set_colors(
        np.quantile(deduc, np.arange(10) / 10.0).tolist(), interpolate=True
    )
    ch = (
        alt.Chart(df_first_and_second)
        .mark_point(filled=True)
        .encode(
            x=alt.X(
                "Risk-aversion:Q",
                title="Risk-aversion",
                scale=alt.Scale(
                    domain=set_axis(df_first_and_second["Risk-aversion"].values)
                ),
            ),
            y=alt.Y(
                "Risk location:Q",
                title="Risk location",
                scale=alt.Scale(
                    domain=set_axis(df_first_and_second["Risk location"].values)
                ),
            ),
            color=alt.Color("Deductible:Q", scale=our_colors),
            size=alt.Size("Copay:Q", scale=alt.Scale(range=[500, 10])),
            tooltip=["Risk-aversion", "Risk location", "Deductible", "Copay"],
            facet=alt.Facet("Model:N", columns=2),
        )
        .interactive()
    )
    if title:
        ch = ch.properties(title=title)
    _maybe_save(ch, path)
    return cast(alt.Chart, ch)


def plot_second_best_contracts(
    theta_mat: np.ndarray,
    y_mat: np.ndarray,
    title: str | None = None,
    cmap=None,
    cmap_label: str | None = None,
    path: str | None = None,
    figsize: tuple[int, int] = (5, 5),
    **kwargs: dict | None,
) -> None:
    """the original scatterplot  of only the second-best contracts in the type space"""
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw=kwargs)
    _ = ax.set_xlabel(r"Risk-aversion $\sigma$")
    _ = ax.set_ylabel(r"Risk location $\delta$")
    _ = ax.set_title(title)
    scatter = ax.scatter(
        theta_mat[:, 0], theta_mat[:, 1], s=200 * (1.0 - y_mat[:, 1]), c=y_mat[:, 0]
    )
    _ = fig.colorbar(scatter, label=cmap_label)
    if path is not None:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)


def display_variable(
    variable: np.ndarray,
    theta_mat: np.ndarray,
    title: str | None = None,
    cmap=None,
    cmap_label: str | None = None,
    path: str | None = None,
    figsize: tuple[int, int] = (5, 5),
    **kwargs: dict | None,
) -> None:
    fig, ax = plt.subplots(
        1, 1, figsize=figsize, subplot_kw=kwargs
    )  # subplot_kw=dict(aspect='equal',)
    _ = ax.set_xlabel(r"Risk-aversion $\sigma$")
    _ = ax.set_ylabel(r"Risk location $\delta$")
    _ = ax.set_title(title)
    scatter = ax.scatter(theta_mat[:, 0], theta_mat[:, 1], c=variable, cmap=cmap)
    _ = fig.colorbar(scatter, label=cmap_label)

    if path is not None:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)


def plot_y_range(
    df_first_and_second: pd.DataFrame,
    figsize=(5, 5),
    s=20,
    title=None,
    path=None,
    **kwargs,
) -> None:
    """the supposed stingray: the optimal contracts for both first and second best in contract space"""
    first = df_first_and_second.query('Model == "First-best"')[["Deductible", "Copay"]]
    second = df_first_and_second.query('Model == "Second-best"')[
        ["Deductible", "Copay"]
    ]

    # discard y1 = 1
    second = second.query("Copay < 0.99")
    fig, ax = plt.subplots(
        1, 1, figsize=figsize, subplot_kw=kwargs
    )  # subplot_kw=dict(aspect='equal',)
    _ = ax.scatter(
        second.Deductible.values,
        second.Copay.values,
        color="tab:blue",
        alpha=0.5,
        s=s,
        label="Second-best",
    )
    _ = ax.scatter(
        first.Deductible.values,
        first.Copay.values,
        color="tab:pink",
        s=s,
        label="First-best",
    )
    _ = ax.set_xlabel("Deductible")
    _ = ax.set_ylabel("Copay")
    _ = ax.set_title(title)
    if path is not None:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)


def plot_contract_models(
    df_first_and_second: pd.DataFrame, varname: str, path=None, **kwargs
) -> alt.Chart:
    """plots a contract variable for both first and second best
    as a function of risk, with different colors by risk-aversion.

    Args:
        df_first_and_second: a dataframe
        varname: the contract variable
        path: the path to save the plot
        **kwargs: additional arguments to pass to the plot

    Returns:
        the two interactive scatterplots.
    """
    # discard y1 = 1
    df = df_first_and_second.query("Copay < 0.99")
    base = alt.Chart().encode(
        x=alt.X(
            "Risk location:Q",
            title="Risk location",
            scale=alt.Scale(domain=set_axis(df["Risk location"].values)),
        ),
        y=alt.Y(f"{varname}:Q"),
        color=alt.Color("Risk-aversion:N"),
        tooltip=["Risk-aversion", "Risk location", varname],
    )
    ch_points = base.mark_point(filled=True, size=50)
    ch_lines = base.mark_line(strokeWidth=0.5)
    ch = alt.layer(ch_points, ch_lines, data=df).interactive().facet(column="Model:N")
    _maybe_save(ch, path)


def plot_contract_riskavs(
    df_first_and_second: pd.DataFrame, varname: str, path=None, **kwargs
) -> alt.Chart:
    """plots the optimal value of a contract variable for both first and second best
    as a function of risk, with different colors by risk-aversion.

    Args:
        df_first_and_second: a dataframe
        varname: the contract variable
        path: the path to save the plot
        **kwargs: additional arguments to pass to the plot

    Returns:
        as many interactive scatterplots as values of the risk-aversion parameter.
    """
    # discard y1 = 1
    df = df_first_and_second.query("Copay < 0.99")
    base = alt.Chart().encode(
        x=alt.X(
            "Risk location:Q",
            scale=alt.Scale(domain=set_axis(df["Risk location"].values)),
        ),
        y=alt.Y(f"{varname}:Q"),
        color=alt.Color("Model:N"),
        tooltip=["Risk-aversion", "Risk location", varname],
    )
    ch_points = base.mark_point(filled=True, size=50)
    ch_lines = base.mark_line(strokeWidth=0.5)
    ch = (
        alt.layer(ch_points, ch_lines, data=df)
        .properties(width=150, height=120)
        .interactive()
        .facet(facet="Risk-aversion:N", columns=5)
    ).properties(title=f"Optimal {varname}")
    _maybe_save(ch, path)


def plot_copays(df_second: pd.DataFrame, path=None, **kwargs) -> alt.Chart:
    """plots the optimal copay for the second best.

    Args:
        df_second: a dataframe
        path: the path to save the plot
        **kwargs: additional arguments to pass to the plot

    Returns:
        the interactive scatterplot.
    """
    # discard y1 = 1
    Copay = df_second.Copay.values
    df = df_second[Copay < 0.99]
    rng = np.random.default_rng()
    # jiggle the points a bit
    df["Copay"] += rng.normal(0, 0.01, df.shape[0])
    base = alt.Chart(df).encode(
        x=alt.X(
            "Risk location:Q",
            title="Risk location",
            scale=alt.Scale(domain=set_axis(df["Risk location"].values)),
        ),
        y=alt.Y(
            "Copay:Q", title="Copay", scale=alt.Scale(domain=set_axis(df.Copay.values))
        ),
        color=alt.Color("Risk-aversion:N"),
        tooltip=["Risk-aversion", "Risk location", "Copay"],
    )
    ch_points = base.mark_point(filled=True, size=150)
    ch_lines = base.mark_line(strokeWidth=1)
    ch = (ch_points + ch_lines).interactive()
    _maybe_save(ch, path)


def plot_constraints(
    theta_mat: np.ndarray,
    IR_binds: list,
    IC_binds: list,
    figsize: tuple = (5, 5),
    s: float = 20,
    title: str | None = None,
    path: str | None = None,
    **kwargs,
) -> None:
    """the original scatterplot  of binding constraints.

    Args:
        theta_mat: the `(N,2)` matrix of type values
        IR_binds: the list of types for which  IR binds
        IC_binds: the list of pairs (i, j) for which i is indifferent between his contract and j's

    Returns:
        nothing. Just plots the constraints.
    """
    IC = "IC" if title else "IC binding"
    IR = "IR" if title else "IR binding"
    fig, ax = plt.subplots(
        1, 1, figsize=figsize, subplot_kw=kwargs
    )  # subplot_kw=dict(aspect='equal',)
    _ = ax.scatter(
        theta_mat[:, 0],
        theta_mat[:, 1],
        facecolors="w",
        edgecolors="k",
        s=s,
        zorder=2.5,
    )
    _ = ax.scatter([], [], marker=">", c="k", label=IC)
    _ = ax.scatter(
        theta_mat[:, 0][IR_binds],
        theta_mat[:, 1][IR_binds],
        label=IR,
        c="tab:green",
        s=s,
        zorder=2.5,
    )
    for i, j in IC_binds:
        # if not (i in IR_binds or j in IR_binds):
        _ = drawArrow(
            ax,
            theta_mat[i, 0],
            theta_mat[j, 0],
            theta_mat[i, 1],
            theta_mat[j, 1],
        )
    _ = ax.set_xlabel(r"$\sigma$")
    _ = ax.set_ylabel(r"$\delta$")

    if title is None:
        _ = ax.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            ncols=2,
            mode="expand",
            borderaxespad=0.0,
        )
    else:
        _ = ax.set_title(title)
        _ = ax.legend(bbox_to_anchor=(1.02, 1.0), loc="lower right")

    if path is not None:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
