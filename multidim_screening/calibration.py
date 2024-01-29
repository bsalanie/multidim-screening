from itertools import product

import numpy as np
from scipy.stats import norm

from .values import val_I


def proba_accident(delta, s):
    return norm.cdf(delta / s, 0.0, 1.0)


def expected_positive_loss(delta, s):
    return s * norm.pdf(delta / s, 0.0, 1.0) / proba_accident(delta, s) + delta


def cost_non_insur(delta, s, sigma):
    y_no_insur = np.array([0.0, 1.0])
    return np.log(val_I(y_no_insur, sigma, delta, s))[0] / sigma


def value_deductible(deduc, delta, s, sigma):
    y = np.array([deduc, 0.0])
    return -np.log(val_I(y, sigma, delta, s))[0] / sigma + cost_non_insur(
        delta, s, sigma
    )


for delta, s, sig10, deduc1000 in product(range(-7, -2), [4], range(2, 6), [500, 1000]):
    sigma = sig10 / 10.0
    deduc = deduc1000 / 1_000.0
    print(f"For {delta=}, {s=}, {sigma=}:")
    print(f"   accident proba = {proba_accident(delta, s): 10.3f}")
    print(f"   expected positive loss = {expected_positive_loss(delta, s): 10.3f}")
    print(f"   cost of non-insurance = {cost_non_insur(delta, s, sigma): 10.3f}")
    print(
        f"   value of deductible {deduc}: "
        f" {value_deductible(deduc, delta, s, sigma): 10.3f}"
    )
    print(f"   cost of non-insurance = {cost_non_insur(delta, s, sigma): 10.3f}")
    print(
        f"   value of deductible {deduc}: "
        f" {value_deductible(deduc, delta, s, sigma): 10.3f}"
    )
