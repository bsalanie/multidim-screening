import multiprocessing as mp
from dataclasses import dataclass
from math import sqrt

import numpy as np
import scipy.sparse as sparse
from numba import float64, njit

examples_dir = "/Users/Bernard/Documents/Github/screening-algo/examples"
results_dir = f"{examples_dir}/results"
plots_dir = f"{examples_dir}//plots"


# MIN_Y0, MAX_Y0 = 0.0, 20.0
# MIN_Y1, MAX_Y1 = 0.2, 0.8
MIN_Y0, MAX_Y0 = 0.3, np.inf
MIN_Y1, MAX_Y1 = 0.0, np.inf

coeff_qpenalty_S0 = 0.00001  # coefficient of the quadratic penalty on S for y0 large
coeff_qpenalty_S0_0 = 1_000.0  # coefficient of the quadratic penalty on S for y0<0
coeff_qpenalty_S1_0 = 1_000.0  # coefficient of the quadratic penalty on S for y1<0
coeff_qpenalty_S1_1 = 1_000.0  # coefficient of the quadratic penalty on S for y1>1
coeff_qpenalty_S01_0 = (
    1_000.0  # coefficient of the quadratic penalty on S for y0 + y1 small
)

# multiprocessing does not provide a speedup at this stage
n_procs = 1
use_multiprocessing = False
if use_multiprocessing:
    # n_procs = mp.cpu_count() - 2
    n_procs = 4

mult_fac = 1.0  # (inverse) multiplier for the step sizes


@njit("float64(float64[:])")
def L2_norm(x: np.ndarray) -> float:
    return sqrt(np.sum(x * x))


@dataclass
class SimulParams:
    s: float
    loading: float
    id: str | None
    theta: np.ndarray
    f: np.ndarray
    resdir: str


def create_simuls_params(
    n0: int,
    n1: int,
    sigmaL: float,
    sigmaH: float,
    deltaL: float,
    deltaH: float,
    s: float,
    loading: float,
    resdir: str,
    with_id: bool = True,
) -> SimulParams:
    # risk-aversions and risk location parameter (unit=1,000 euros)
    sigmas, deltas = np.linspace(sigmaL, sigmaH, num=n0), np.linspace(
        deltaL, deltaH, num=n1
    )
    theta0, theta1 = np.meshgrid(sigmas, deltas)
    theta = np.stack((theta0.flatten(), theta1.flatten()))  # is a 2 x N matrix
    N = theta.shape[-1]  # number of types N = n0 * n1
    f = np.ones(N)  # weights of distribution
    ### Model parameters setting
    if with_id:
        id = f"INSUR_STRAIGHT_N{N}"  # + strftime("%m-%d_%H-%M")
    else:
        id = None
    params = SimulParams(s=s, loading=loading, id=id, theta=theta, f=f, resdir=resdir)
    return params


def drawArrow(ax, xA, xB, yA, yB, c="k", ls="-"):
    n = 50
    x = np.linspace(xA, xB, 2 * n + 1)
    y = np.linspace(yA, yB, 2 * n + 1)
    ax.plot(x, y, color=c, linestyle=ls)
    ax.annotate(
        "",
        xy=(x[n], y[n]),
        xytext=(x[n - 1], y[n - 1]),
        arrowprops=dict(arrowstyle="-|>", color=c),
        size=15,
        # zorder=2,
    )


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
