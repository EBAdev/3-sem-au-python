"""
Module to hold general market functions, that is functions that all assets have.
"""
import numpy as np
from typing import Union, List, Tuple
from math import exp
from ifpy.models.portfolio import Portfolio

Num = Union[int, float]

Mat = Union[
    List[List[float]],
    List[List[int]],
    Tuple[Tuple[float, ...], ...],
    Tuple[Tuple[int, ...], ...],
]

Vec = Union[
    List[float],
    List[int],
    Tuple[float, ...],
    Tuple[int, ...],
]


def discount_factor(r: Num, t: Num, T: Num, continous: bool = False):
    """
    Function to determine the discount factor of a future payment.
    #### LaTex formula
    * discrete \\frac{1}{(1+r)^{T-t}}
    * continous  e^{-r(T-t)}
    #### Parameters
    1. r:Num [required]
            * The interest rate on the payment
    2. t:Num[required]
            * The start time of the discount factor
    3. T:Num[required]
            * The ending time of the discount factor
    4. continous:bool = False
            * Boolean to control if market is continously compounded.

    """
    if continous:
        return exp(-r * (T - t))
    return 1 / (1 + r) ** (T - t)


def portfolio_beta(
    portfolio1: Portfolio,
    cov_mat: Mat,
    portfolio2: Portfolio,
    rounding: Union[int, None] = 4,
):
    """
    Function to calculate the beta between portfolio 1 with respects to portfolio two, i.e. if the beta of a asset with the market is to be calculated. The asset weight goes in portfolio 1 and the market weight goes in portfolio 2.
    #### Formula
    ß = ∂_{pf,M}/∂^2_M
    ##### LaTeX
    \\beta_{pf} = \\frac{\\mathrm{cov}(w_{pf},M)}{\\sigma_{M}^2}
    #### Paramters
    1. portfolio1 : iof.Portfolio
            * The first portolio instance, the one which beta is calculated
    2. cov_matrix : Mat object
            * The covariance matrix of the finanicial market.
    3. portfolio2 : iof.Portfolio
            * The second portolio instance, the one which the beta is calculated wrt.
    4. rounding: int or none
            * The rounding of the result.
    """
    cov_mat = portfolio1.cov_mat_check(cov_mat)
    cov_mat = portfolio2.cov_mat_check(cov_mat)

    pf_cov = portfolio1.covariance(cov_mat, portfolio2.w, None)
    pf2_var = portfolio2.variance(cov_mat, None)

    if rounding is None:
        return pf_cov / pf2_var
    else:
        return round(pf_cov / pf2_var, rounding)
