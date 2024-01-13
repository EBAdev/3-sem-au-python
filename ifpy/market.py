"""
Module to hold general market functions, that is functions that all assets have.
"""
import numpy as np
from typing import Union, List, Tuple
from math import exp, isclose, sqrt
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
        return exp(-1 * r * (T - t))
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


def expected_value(
    outcomes: list[int] | list[float] | list[float | int],
    probs: list[float],
    rounding: Union[int, None] = 4,
) -> float | int:
    """
    Function to calculate the expected value of a set of outcomes and probabilities
    #### Formula
    \\E[asset]=\\sum_{i=1}^4 \\pi_i\\cdot r_i
    #### Parameters
    1. outcomes: List[nums] [required]
            * List of all the possible outcomes
    2. probs: list[nums] [required]
            * List of the corresponding probabilities i.e. the idx of outcome i's prob. should be i.
    """
    if len(outcomes) != len(probs):
        raise ValueError("Outcome and probability should be the same length")
    if not isclose(np.sum(np.array(probs)), round(np.sum(np.array(probs)))):
        raise ValueError("Probabilities should sum to 1, they don't")
    if rounding is None:
        return float(np.sum(np.array(outcomes) * np.array(probs)))
    else:
        return round(float(np.sum(np.array(outcomes) * np.array(probs))), rounding)


def variance(
    outcomes: list[int] | list[float] | list[float | int],
    probs: list[float],
    rounding: Union[int, None] = 4,
) -> float | int:
    """
    Function to calculate the variance of a set of outcomes and probabilities
    #### Formula
     \\V[asset]=\\E[X^2]-\\E^2[X]
    #### Parameters
    1. outcomes: List[nums] [required]
            * List of all the possible outcomes
    2. probs: list[nums] [required]
            * List of the corresponding probabilities i.e. the idx of outcome i's prob. should be i.
    """
    expected = expected_value(outcomes, probs)
    squared_error = [(x - expected) ** 2 for x in outcomes]
    if rounding is None:
        return expected_value(squared_error, probs, None)
    else:
        return round(expected_value(squared_error, probs, None), rounding)


def standard_deviation(
    outcomes: list[int] | list[float] | list[float | int],
    probs: list[float],
    rounding: Union[int, None] = 4,
) -> float | int:
    """
    Function to calculate the variance of a set of outcomes and probabilities
    #### Formula
    \\sigma_{a_1} = \\sqrt{\\V[a_1]}
    #### Parameters
    1. outcomes: List[nums] [required]
            * List of all the possible outcomes
    2. probs: list[nums] [required]
            * List of the corresponding probabilities i.e. the idx of outcome i's prob. should be i.
    """
    if rounding is None:
        return sqrt(variance(outcomes, probs, None))
    else:
        return round(sqrt(variance(outcomes, probs, None)), rounding)


def covariance(
    x_outcomes: list[int] | list[float] | list[float | int],
    y_outcomes: list[int] | list[float] | list[float | int],
    probs: list[float],
    rounding: Union[int, None] = 4,
) -> float | int:
    """
    Function to calculate the covariance of a set of outcomes and probabilities.
    if cov(x,y) = 0 then the variables x,y are independent.
    #### Formula
    \\mathrm{cov}(a_1,a_2)=\\E[(a_1-\\E[a_1])(a_2-\\E[a_2])] = \\E[a_1\\cdot a_2]-\\E[a_1]\\cdot\\E[a_2]
    #### Parameters
    1. x_outcomes: List[nums] [required]
            * List of all the possible outcomes for X variable

    3. y_outcomes: List[nums] [required]
            * List of all the possible outcomes for Y variable, should be same length as x_outcomes.

    4. probs: list[nums] [required]
            * List of the corresponding probabilities
    """
    if len(x_outcomes) != len(y_outcomes):
        raise (
            ValueError(
                "The two variables does not have the same lenght in outcome lists."
            )
        )
    expected_xy = expected_value(
        [x * y_outcomes[i] for i, x in enumerate(x_outcomes)], probs, None
    )
    if rounding is None:
        return expected_xy - expected_value(x_outcomes, probs, None) * expected_value(
            y_outcomes, probs, None
        )
    else:
        return round(
            expected_xy
            - expected_value(x_outcomes, probs, None)
            * expected_value(y_outcomes, probs, None),
            rounding,
        )


def correlation(
    x_outcomes: list[int] | list[float] | list[float | int],
    y_outcomes: list[int] | list[float] | list[float | int],
    probs: list[float],
    rounding: Union[int, None] = 4,
) -> float | int:
    """
    Function to calculate the correlation coefficient of a set of outcomes and probabilities.
    if corr(x,y) = 0 then X,Y are uncorrelated,
    if corr(x,y) < 0 then X,Y are negatively correlated,
    if corr(x,y) > 0 then X,Y are positively correlated.
    #### Formula
     \\rho_{a_1,a_2}= \\frac{\\sigma_{a_1,a_2}}{\\sigma_{a_1}\\sigma_{a_2}}
    #### Parameters
    1. x_outcomes: List[nums] [required]
            * List of all the possible outcomes for X variable
    2. y_outcomes: List[nums] [required]
            * List of all the possible outcomes for Y variable, should be same length as x_outcomes.
    3. Probs: list[nums] [required]
            * List of the corresponding probabilities,
            i.e. the idx of outcome Xi's prob. should be Xi.
    """
    if rounding is None:
        return covariance(x_outcomes, y_outcomes, probs, None) / (
            standard_deviation(x_outcomes, probs, None)
            * standard_deviation(y_outcomes, probs, None)
        )
    else:
        return round(
            covariance(x_outcomes, y_outcomes, probs, None)
            / (
                standard_deviation(x_outcomes, probs, None)
                * standard_deviation(y_outcomes, probs, None)
            ),
            rounding,
        )
