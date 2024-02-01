"""the model-dependent functions and parameters are defined in  `model_name`.py
 we use `importlib` to import what is needed
"""

import importlib
from typing import cast

import numpy as np
from numba import njit
from numba.typed import List

from .classes import ScreeningModel, ScreeningResults

# we provide the name of the model
model_name = "insurance_d2_m2"


# DO NOT CHANGE BELOW THIS LINE
model_module = importlib.import_module(f".{model_name}", package="multidim_screening")


def setup_model(model_name: str) -> ScreeningModel:
    screening_model = model_module.create_model(model_name)
    return cast(ScreeningModel, screening_model)


def initialize_contracts(
    model: ScreeningModel,
    start_from_first_best: bool,
) -> tuple[np.ndarray, list]:
    """Initializes the contracts for the second best problem

    Args:
        model: the screening model
        start_from_first_best: whether to start from the first best

    Returns:
        tuple[np.ndarray, list]: initial contracts (an `(N,m)` matrix) and a list of types for whom
         we optimize contracts
    """
    return cast(
        tuple[np.ndarray, list],
        model_module.create_initial_contracts(model, start_from_first_best),
    )


def first_best(model: ScreeningModel) -> np.ndarray:
    """Returns the first best contracts and surpluses

    Args:
        model: the screening model

    Returns:
        the first best contracts, and the first-best surpluses
    """
    return cast(np.ndarray, model_module.create_first_best(model))


@njit("float64[:,:](float64[:], float64[:, :], List)")
def b_function(y: np.ndarray, theta_mat: np.ndarray, params: List) -> np.ndarray:
    """The b function

    Args:
        y:  a $2 k$-vector of $k$ contracts
        theta_mat: a $(q,2)$-vector of characteristics of types
        params: the parameters of the model

    Returns:
        a $(q,k)$-matrix
    """
    return cast(np.ndarray, model_module.b_fun(y, theta_mat, params))


@njit("float64[:,:, :](float64[:], float64[:, :], List)")
def b_deriv(y: np.ndarray, theta_mat: np.ndarray, params: tuple) -> np.ndarray:
    """The derivatives of the b function

    Args:
        y:  a $2 k$-vector of $k$ contracts
        theta_mat: a $(q,2)$-vector of characteristics of types
        params: the parameters of the model

    Returns:
        an $(m,q,k)$-matrix
    """
    return cast(np.ndarray, model_module.db_fun(y, theta_mat, params))


@njit("float64[:,:](float64[:], float64[:, :], List)")
def S_function(y: np.ndarray, theta_mat: np.ndarray, params: tuple) -> np.ndarray:
    """The S function

    Args:
        y:  a $2 k$-vector of $k$ contracts
        theta_mat: a $(q,2)$-vector of characteristics of types
        params: the parameters of the model

    Returns:
        a $(q,k)$-matrix
    """
    return cast(np.ndarray, model_module.S_fun(y, theta_mat, params))


@njit("float64[:,:, :](float64[:], float64[:, :], List)")
def S_deriv(y: np.ndarray, theta_mat: np.ndarray, params: tuple) -> np.ndarray:
    """The derivatives of the S function

    Args:
        y:  a $2 k$-vector of $k$ contracts
        theta_mat: a $(q,2)$-vector of characteristics of types
        params: the parameters of the model

    Returns:
        an $(m,q,k)$-matrix
    """
    return cast(np.ndarray, model_module.dS_fun(y, theta_mat, params))


def plot(results: ScreeningResults) -> None:
    """Plots the results

    Args:
        results: the results
    """
    model_module.plot_results(results)
