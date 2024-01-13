from typing import Union, List, Tuple, Dict
from math import sqrt

import matplotlib.pyplot as plt

from scipy.stats import norm
import numpy as np

Num = Union[float, int]
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


class Portfolio:
    """
    General portfolio class
    """

    def __init__(self, weights: Vec) -> None:
        """
        Class to create a general portfolio with some weight vector.
        ### Parameters
        1. weights: Vec object
            * The portfolio weights in the financial market.
        """
        self.w = np.array(np.atleast_2d(weights)).T

    def cov_mat_check(self, cov_mat: Mat):
        cov_mat = np.array(cov_mat)
        if self.w.shape[0] != cov_mat.shape[0] or cov_mat.shape[0] != cov_mat.shape[1]:
            raise ValueError(
                "There was a error with the provided covariance matrix, please make sure that it is of dimension nxn, where n is number of assets."
            )
        return cov_mat

    def returns_check(self, ex_returns: Vec):
        ex_returns = np.array(np.atleast_2d(ex_returns)).T
        if ex_returns.shape[0] != self.w.shape[0]:
            raise ValueError(
                "There was a error with the provided covariance matrix, please make sure that it is a list of dimension 1xn, where n is number of assets."
            )
        return ex_returns

    def variance(self, cov_mat: Mat, rounding: Union[int, None] = 4):
        """
        Function to calculate a portfolio's variance
        #### Formula
        w^T * ∑ * w
        ##### LaTeX formula
        w^T \\cdot \\Sigma \\cdot w
        #### Parameters
        1. cov_matrix : Mat object
                * The covariance matrix of the finanicial market.
        2. rounding: int or None
                * The rounding of the result
        """
        cov_mat = self.cov_mat_check(cov_mat)
        if rounding is None:
            return (self.w.T @ cov_mat @ self.w)[0][0]
        return round((self.w.T @ cov_mat @ self.w)[0][0], rounding)

    def ex_return(
        self,
        ex_returns: Vec,
        cov_mat=None,
        rf_rate=None,
        rounding: Union[int, None] = 4,
    ):
        """
        Function to calculate the expected return of a general portfolio.
        #### LaTex
        \\bar{\\mathbf{r}}_{pf}=w_{pf}\\cdot \\bar{\\mathbf{r}}
        #### Parameters
        1. ex_returns: Vec object
                * The expected returns of the assets in the financial markets.
        """
        ex_returns = self.returns_check(ex_returns)
        if rounding is None:
            return (self.w.T @ ex_returns)[0][0]
        return round((self.w.T @ ex_returns)[0][0], rounding)

    def calc_a(self, cov_mat: np.ndarray, rounding: Union[int, None] = 4):
        """
        a = \\1^T \\Sigma^{-1}\\1
        """
        one_mat = np.atleast_2d(np.ones(cov_mat.shape[0])).T
        if rounding is None:
            return (one_mat.T @ np.linalg.inv(cov_mat) @ one_mat)[0][0]
        return round((one_mat.T @ np.linalg.inv(cov_mat) @ one_mat)[0][0], rounding)

    def calc_b(
        self,
        cov_mat: np.ndarray,
        ex_returns: np.ndarray,
        rounding: Union[int, None] = 4,
    ):
        """
        b = \\1^T \\Sigma^{-1}\\bar{\\mathbf{r}}
        """
        one_mat = np.atleast_2d(np.ones(cov_mat.shape[0])).T
        if rounding is None:
            return (one_mat.T @ np.linalg.inv(cov_mat) @ ex_returns)[0][0]
        return round((one_mat.T @ np.linalg.inv(cov_mat) @ ex_returns)[0][0], rounding)

    def calc_c(
        self,
        cov_mat: np.ndarray,
        ex_returns: np.ndarray,
        rounding: Union[int, None] = 4,
    ):
        """
        c = \\bar{\\mathbf{r}} \\Sigma^{-1}\\bar{\\mathbf{r}}
        """
        if rounding is None:
            return (ex_returns.T @ np.linalg.inv(cov_mat) @ ex_returns)[0][0]
        return round(
            (ex_returns.T @ np.linalg.inv(cov_mat) @ ex_returns)[0][0], rounding
        )

    def calc_d(
        self,
        cov_mat: np.ndarray,
        ex_returns: np.ndarray,
        rounding: Union[int, None] = 4,
    ):
        """
        d = a\\cdot c-b^2
        """
        a = self.calc_a(cov_mat, None)
        b = self.calc_b(cov_mat, ex_returns, None)
        c = self.calc_c(cov_mat, ex_returns, None)
        if rounding is None:
            return a * c - b**2
        return round(a * c - b**2, rounding)

    def get_abcd(
        self,
        cov_mat: Mat,
        ex_returns: Vec,
        rounding: Union[int, None] = 4,
    ):
        """
        Function to return a,b,c,d values for the market.
        
        #### Formulas:
        \\begin{aligned}
            & a=\\1^{\\top} \\Sigma^{-1} \\1>0                                       \\ \\\\
                
            & b=\\1^{\\top} \\Sigma^{-1} \\overline{\\mathbf{r}}                      \\ \\\\
                
            & c=\\overline{\\mathbf{r}}^{\\top} \\Sigma^{-1} \\overline{\\mathbf{r}}>0 \\ \\\\
                
            & d=a c-b^2>0
        \\end{aligned}
        
        """
        cov_mat = self.cov_mat_check(cov_mat)
        ex_r = self.returns_check(ex_returns)
        a = self.calc_a(cov_mat, rounding)
        b = self.calc_b(cov_mat, ex_r, rounding)
        c = self.calc_c(cov_mat, ex_r, rounding)
        d = self.calc_d(cov_mat, ex_r, rounding)
        return {"a": a, "b": b, "c": c, "d": d}

    def covariance(
        self, cov_mat: Mat, weights: np.ndarray, rounding: Union[int, None] = 4
    ):
        """
        Function to calculate the covariance of between the portifolio and another portfolio.
        #### Formula
        w_{pf_1}^T · ∑ · w_{pf_2}
        ##### LaTeX
        \\mathrm{cov}(w_{pf_1},w_{pf_2})=w_{pf_1}^{T}\\cdot\\Sigma\\cdot w_{pf_2}
        #### Parameters
        1. cov_matrix : Mat object.
                * The covariance matrix of the finanicial market.
        2. weights : np.NDarray
                * A weight vector of the other portfolio, should be a column vector.
        3. rounding: int or None
                * The rounding of the result.
        """
        cov_mat = self.cov_mat_check(cov_mat)
        if rounding is None:
            return (self.w.T @ cov_mat @ weights)[0][0]
        return round((self.w.T @ cov_mat @ weights)[0][0], rounding)

    def beta(
        self,
        cov_mat: Mat,
        ex_returns: Vec,
        rf_rate: float,
        rounding: Union[int, None] = 4,
    ):
        """
        Function to calculate the beta between the portfolio and the market portfolio, assuming that the market portfolio is the tangency portfolio. If this is not the case use function 'iof_2.portfolio_beta()' to use custom weights.
        #### Formula
        ß = ∂_{pf,M}/∂^2_M
        ##### LaTeX
        \\beta_{pf} = \\frac{\\mathrm{cov}(w_{pf},M)}{\\sigma_{M}^2}
        #### Paramters
        1. cov_matrix : Mat object
                * The covariance matrix of the finanicial market.
        2. ex_returns: Vec object
                * The expected returns of the assets in the financial markets.
        4. rf_rate: float or None
                * The risk free rate in the market, if it is present.
        4. rounding: int or none
                * The rounding of the result.
        """
        self.cov_mat_check(cov_mat)
        market_pf = TanPortfolio(cov_mat, ex_returns, rf_rate)
        pf_cov = self.covariance(cov_mat, market_pf.w, None)
        pf2_var = market_pf.variance(cov_mat, None)

        if rounding is None:
            return pf_cov / pf2_var
        return round(pf_cov / pf2_var, rounding)

    def prob_below(
        self,
        threshold,
        ex_returns,
        cov_mat,
        rf_rate=None,
        rounding: Union[int, None] = 4,
    ):
        """
        Function to return the probability of the portfolio return being below some threshold value. This assumes that the returns follow a multivariate normal distribution.
        #### LaTex formulas
        First the random variable that is the portfolio return is transformed into a standard normal random variable, using the transformation:

        z=\\frac{threshold-\\mu_{pf}}{\\sigma_{pf}}

        We then use the CDF of the standard normal distribution to find the probability that the portfolio return is less than or equal to z.

        \\P[\\bar{r}_{pf}\\leq z]


        """
        ex_return = self.ex_return(ex_returns, cov_mat, rf_rate, None)
        pf_sd = sqrt(self.variance(cov_mat, None))
        if rounding is None:
            return norm.cdf(threshold, loc=ex_return, scale=pf_sd)
        return round(norm.cdf(threshold, loc=ex_return, scale=pf_sd), rounding)

    def correlation_matrix(self, cov_mat: Mat):
        """
        Calculate standard deviations and correlation matrix from a given covariance matrix.

        Parameters:
        - cov_mat: Mat.

        Returns:
        - dict: Dictionary containing standard deviations and correlation matrix.
        """
        cov_mat = self.cov_mat_check(cov_mat)
        SD_vec = [sqrt(cov_mat[i][i]) for i in range(cov_mat.shape[0])]
        corr_mat = []
        for r in range(cov_mat.shape[0]):
            row = []
            for c in range(cov_mat.shape[1]):
                if r == c:
                    corr = 1
                else:
                    corr = cov_mat[r][c] / (SD_vec[r] * SD_vec[c])
                row.append(corr)
            corr_mat.append(row)
        return {"SD": np.array([SD_vec]).T, "correlation": np.array(corr_mat)}

    def sharpe_ratio(
        self,
        ex_returns: Vec,
        cov_mat=None,
        rf_rate=None,
        rounding: Union[int, None] = 4,
    ):
        """
        \\begin{equation*}
          S_{pf} = \\frac{\\bar{\\mathbf{r}}_{pf}-r_f}{\\sigma_{pf}}
        \\end{equation*}
        """
        ex_pf_r = self.ex_return(ex_returns, cov_mat, rf_rate, None)
        if isinstance(self, TanPortfolio):
            ex_pf_r = ex_pf_r["return"]
        if rounding is None:
            return (ex_pf_r - rf_rate) / sqrt(self.variance(cov_mat, None))
        return round(
            (ex_pf_r - rf_rate) / sqrt(self.variance(cov_mat, None)),
            rounding,
        )


class OptPortfolio(Portfolio):
    """
    Optimal portfolio in financial market given target expected return subclass
    """

    def __init__(
        self,
        ex_portfolio_return: Num,
        cov_mat: Mat,
        ex_returns: Vec,
        rf_rate: Union[float, None],
    ) -> None:
        """
        Class used to create the optimal portfolio in the market, with some expected portfolio return.
        i.e. the amount to be invested in each asset to optain a given expected return.

        portfolio is on the critical frontier and is the portfolio with lowest variance
        for a given return r.
        ### Weights Formula
        #### without rf rate:
            
            \\begin{equation*}      
                \\mathbf{w}^* =\\frac{a \\bar{r}_p-b}{d} \\Sigma^{-1} \\overline{\\mathbf{r}}+\\frac{c-\\bar{r}_p b}{d} \\Sigma^{-1} \\1
            \\end{equation*}
        where we have
            $$
                \\begin{aligned}
                    & a=\\1^{\\top} \\Sigma^{-1} \\1>0 \\\\
                    & b=\\1^{\\top} \\Sigma^{-1} \\overline{\\mathbf{r}}\\\\
                    & c=\\overline{\\mathbf{r}}^{\\top} \\Sigma^{-1} \\overline{\\mathbf{r}}>0 \\\\
                    & d=a c-b^2>0 
                \\end{aligned}
            $$
        
        #### With rf rate:

        \\mathbf{w}^*=\\frac{\\bar{r}_p^e}{\\left(\\overline{\\mathbf{r}}^e\\right)^{\\top} \\Sigma^{-1} \\overline{\\mathbf{r}}^e} \\Sigma^{-1} \\overline{\\mathbf{r}}^e,

        #### Parameters
        1. ex_portfolio_return: Num
                * The total (with risk free rate) expected portfolio return.
        2. cov_mat: Mat object
                * The covariance matrix of the finanicial market.
        3. ex_returns: Vec object
                * The expected returns of the assets in the financial markets.
        4. rf_rate: float or None
                * The risk free rate in the market, if it is present. NB. This changes the formula used.
        """
        self.ex_r = ex_portfolio_return
        cov_mat = np.array(cov_mat)
        ex_returns = np.array(np.atleast_2d(ex_returns)).T
        if cov_mat.shape[0] != cov_mat.shape[1]:
            raise ValueError(
                "There was a error with the provided covariance matrix, please make sure that it is of dimension nxn, where n is number of assets."
            )
        if ex_returns.shape[0] != cov_mat.shape[0]:
            raise ValueError(
                "The portfolio was provided more expected returns than there is assets in the covariance matrix. Make sure the expected returns are a 1xn list, where n is number of assets in covariance matrix."
            )
        if rf_rate is None:
            a = self.calc_a(cov_mat, None)
            b = self.calc_b(cov_mat, ex_returns, None)
            c = self.calc_c(cov_mat, ex_returns, None)
            d = self.calc_d(cov_mat, ex_returns, None)
            one_mat = np.atleast_2d(np.ones(cov_mat.shape[0])).T
            w = ((a * self.ex_r - b) / d) * np.linalg.inv(cov_mat) @ ex_returns + (
                (c - self.ex_r * b) / d
            ) * np.linalg.inv(cov_mat) @ one_mat
            self.w = w
        else:
            self.ex_ec_r = self.ex_r - rf_rate
            ex_ec_returns = ex_returns - rf_rate
            w = (
                (
                    self.ex_ec_r
                    / (ex_ec_returns.T @ np.linalg.inv(cov_mat) @ ex_ec_returns)
                )
                * np.linalg.inv(cov_mat)
                @ ex_ec_returns
            )
            self.w = w


class MVPortfolio(Portfolio):
    """
    Minimum Variance portfolio subclass
    """

    def __init__(self, cov_mat: Mat) -> None:
        """
        Class used to create the minimum variance portfolio
        i.e. the amount to be invested in each asset to optain a mininmum variance portfolio
        ##### Weights Formula
         w_{mvp} = \\frac{1}{a}\\cdot \\Sigma^{-1}\\cdot\\1^T,\\quad\\text{where }a=\\1^T\\Sigma^{-1}\\1
        #### Parameters
        1. cov_mat : Mat object
                * The covariance matrix of the finanicial market.
        """
        self.w = self.__calc_weights(cov_mat)

    def __calc_weights(self, cov_mat: Mat) -> np.ndarray:
        cov_mat = np.array(cov_mat)
        if cov_mat.shape[0] != cov_mat.shape[1]:
            raise ValueError(
                "There was a error with the provided covariance matrix, please make sure that it is of dimension nxn, where n is number of assets."
            )
        one_mat = np.atleast_2d(np.ones(cov_mat.shape[0])).T
        w = 1 / self.calc_a(cov_mat, None) * np.linalg.inv(cov_mat) @ one_mat
        return w

    def ex_return(
        self,
        ex_returns: Vec,
        cov_mat,
        rf_rate=None,
        rounding: Union[int, None] = 4,
    ):
        """
        Function to determine the MVP expected return
        i.e. the expected return of the mininmum variance portfolio in the financial market
        #### Formula
        b/a
        ##### LaTex equation
        \\bar r_{mvp}=\\frac{b}{a},\\quad\\text{where } b=\\1^T\\Sigma^{-1}\\bar{\\mathbf{r}}_{mvp}
        #### Parameters
        1. cov_mat : Mat object
                * The covariance matrix of the finanicial market.
        2. ex_returns : Vec object
                * The expected returns all the assets.
        3. rounding: int or None
                * The rounding of the result
        """
        r = self.returns_check(ex_returns)
        cov_mat = self.cov_mat_check(cov_mat)
        b = self.calc_b(cov_mat, r, None)
        a = self.calc_a(cov_mat, None)
        if rounding is None:
            return b / a
        else:
            return round(b / a, rounding)

    def variance(self, cov_mat, rounding: Union[int, None] = 4):
        """
        Function to calculate a minimum variance portfolio's variance
        #### Formula
        w^T * ∑ * w or (1/a)
        ##### LaTex equation
        \\sigma_{mvp}^2=\\frac{1}{a},\\quad\\text{where }a=\\1^T\\Sigma^{-1}\\1
        #### Parameters
        1. cov_mat : Mat object
                * The covariance matrix of the finanicial market.
        2. rounding: int or None
                * The rounding of the result
        """
        cov_mat = self.cov_mat_check(cov_mat)
        a = self.calc_a(cov_mat, None)
        if rounding is None:
            return 1 / a
        else:
            return round(1 / a, rounding)

    def prob_below(
        self, threshold, ex_returns, cov_mat, rf_rate=None, rounding: int | None = 4
    ):
        return super().prob_below(threshold, ex_returns, cov_mat, rf_rate, rounding)


class TanPortfolio(Portfolio):
    """
    Tangency/Market portfolio subclass.
    """

    def __init__(self, cov_mat: Mat, ex_returns: Vec, rf_rate: float) -> None:
        """
        Class to create the Tangency portfolio, and determine its weights.
        ##### LaTex weights formula
        w_{tan}= \\frac{\\Sigma^{-1} \\cdot \\bar{\\mathbf{r}}^e}{\\1^T\\cdot \\Sigma^{-1}\\cdot\\bar{\\mathbf{r}}^e}
        #### Parameters
        1. cov_mat : Mat object [required]
                * The covariance matrix of the finanicial market.
        2. ex_returns  : Vec object [required]
                * The expected return of each asset.
        3. rf_rate : float [required]
                * The return of a the risk free asset if it is present in the market.
        """
        self.w = self.__calc_weights(cov_mat, ex_returns, rf_rate)

    def __calc_weights(
        self, cov_mat: Mat, ex_returns: Vec, rf_rate: float
    ) -> np.ndarray:
        cov_mat = np.array(cov_mat)
        ex_returns = np.array(np.atleast_2d(ex_returns)).T
        if cov_mat.shape[0] != cov_mat.shape[1]:
            raise ValueError(
                "There was a error with the provided covariance matrix, please make sure that it is of dimension nxn, where n is number of assets."
            )
        if ex_returns.shape[0] != cov_mat.shape[0]:
            raise ValueError(
                "The portfolio was provided more expected returns than there is assets in the covariance matrix. Make sure the expected returns are a 1xn list, where n is number of assets in covariance matrix."
            )
        ex_ec_r = ex_returns - rf_rate
        one_mat = np.atleast_2d(np.ones(cov_mat.shape[0])).T
        w = (np.linalg.inv(cov_mat) @ ex_ec_r) / (
            one_mat.T @ np.linalg.inv(cov_mat) @ ex_ec_r
        )
        return w

    def ex_return(
        self, ex_returns: Vec, cov_mat: Mat, rf_rate: float, rounding: int | None = 4
    ) -> Dict:
        """
        Function to calculate the tangency portfolios expected excess return and expected return. Returns a dictionay with both values keyed at "excess return" and "return"
        #### Formula
        w_{tan}^T * r^e
        ##### LaTeX formula
        \\bar\\rr^e_{tan}= w_{tan}^T \\cdot \\bar\\rr^e
        #### Parameters
        1. ex_returns  : Vec object [required]
                * The expected return of each asset.
        2. cov_mat : Mat object [required]
                * The covariance matrix of the finanicial market.
        3. rf_rate : int | float [required]
                * The return of a the risk free asset if it is present in the market.
        4. rounding: int or None
                * The rounding of the result
        """
        ex_returns = self.returns_check(ex_returns)
        cov_mat = self.cov_mat_check(cov_mat)
        ex_ec_r = ex_returns - rf_rate
        one_mat = np.atleast_2d(np.ones(cov_mat.shape[0])).T
        pf_ex_ec_r = (
            (ex_ec_r.T @ np.linalg.inv(cov_mat) @ ex_ec_r)
            / (one_mat.T @ np.linalg.inv(cov_mat) @ ex_ec_r)
        )[0][0]
        if rounding is None:
            return {"excess return": pf_ex_ec_r, "return": pf_ex_ec_r + rf_rate}
        else:
            return {
                "excess return": round(pf_ex_ec_r, rounding),
                "return": round(pf_ex_ec_r + rf_rate, rounding),
            }

    def price_of_risk(
        self, ex_returns: Vec, cov_mat: Mat, rf_rate: float, rounding: int | None = 4
    ):
        """
        The slope of the CML is given by \\frac{\\bar{r_M}-r_f}{\\sigma_M} and is referred to as the price of risk as it tells you how much expected return you get for taking on risk in the form of \\sigma_p.
        """
        ex_ec_r = self.ex_return(ex_returns, cov_mat, rf_rate, None)["return"]
        pf_var = self.variance(cov_mat, None)

        if rounding is None:
            return (ex_ec_r - rf_rate) / sqrt(pf_var)
        else:
            return round((ex_ec_r - rf_rate) / sqrt(pf_var), rounding)

    def prob_below(
        self,
        threshold,
        ex_returns,
        cov_mat,
        rf_rate,
        rounding: Union[int, None] = 4,
    ):
        """
        Function to return the probability of the portfolio return being below some threshold value. This assumes that the returns follow a multivariate normal distribution.

        NOTE: we use total return currently for tangency portfolio.

        #### LaTex formulas
        First the random variable that is the portfolio return is transformed into a standard normal random variable, using the transformation:

        z=\\frac{threshold-\\mu_{pf}}{\\sigma_{pf}}

        We then use the CDF of the standard normal distribution to find the probability that the portfolio return is less than or equal to z.

        \\P[\\bar{r}_{pf}\\leq z]


        """
        ex_return = self.ex_return(ex_returns, cov_mat, rf_rate, None)["return"]
        pf_sd = sqrt(self.variance(cov_mat, None))
        standard_norm_val = (threshold - ex_return) / pf_sd
        if rounding is None:
            return norm.cdf(standard_norm_val)
        return round(norm.cdf(standard_norm_val), rounding)


def CML(sd: Num, rf_rate: Num, sd_market: Num, r_market: Num):
    return rf_rate + ((r_market - rf_rate) / sd_market) * sd


def CML_plot(
    cov_mat: Mat,
    ex_returns: Vec,
    rf_rate: float,
    portfolios: List[Portfolio] = None,
    portfolio_labels: List[str] = None,
    x_len=0.3,
):
    """
    Function to return a plot of the capital market line. If portfolios should be shown give them as a list, if labels of these should be shown give labels as a list also.
    """
    # evenly sampled time at 200ms intervals
    plt.rcParams.update({"text.usetex": True, "font.family": "Computer Modern"})
    sd = np.linspace(0, x_len)
    tan = TanPortfolio(cov_mat, ex_returns, rf_rate)

    plt.xlabel(
        r"Standard deviation, $\sigma$",
        fontsize=14,
    )
    plt.ylabel(
        r"Expected return, $\bar{r}$",
        fontsize=14,
    )
    plt.ylim(0, x_len / 2)
    plt.xlim(0, x_len)
    plt.title(r"Capital Market Line", fontsize=20)
    plt.plot(
        sd,
        CML(
            sd,
            rf_rate,
            sqrt(tan.variance(cov_mat, None)),
            tan.ex_return(ex_returns, cov_mat, rf_rate)["return"],
        ),
        color="black",
    )
    plt.axhline(y=rf_rate, color="blue", linestyle="--")
    plt.xticks([0] + [x_len / 10 * i for i in range(10)])
    plt.yticks([0] + [x_len / 10 * i for i in range(10)])
    plt.text(
        sd[10],
        rf_rate + 0.005,
        r"$r_f$",
        color="blue",
        fontsize=12,
        va="bottom",
    )
    plt.text(
        sd[10],
        CML(
            sd[15],
            rf_rate,
            sqrt(tan.variance(cov_mat, None)),
            tan.ex_return(ex_returns, cov_mat, rf_rate)["return"],
        ),
        r"$CML$",
        color="black",
        fontsize=12,
        va="bottom",
    )
    if portfolios is not None:
        pf_sd = [sqrt(pf.variance(cov_mat, None)) for pf in portfolios]
        pf_ex_r = [
            pf.ex_return(ex_returns, cov_mat, rf_rate, None)
            if isinstance(pf, TanPortfolio) is False
            else pf.ex_return(ex_returns, cov_mat, rf_rate, None)["return"]
            for pf in portfolios
        ]
        plt.scatter(pf_sd, pf_ex_r, color="red", label="Portfolios", marker="x")
        for i, (x, y) in enumerate(zip(pf_sd, pf_ex_r)):
            plt.text(
                x,
                y + 0.005,
                portfolio_labels[i],
                color="black",
                fontsize=10,
                ha="right",
                va="bottom",
            )

    plt.show()


def frontier(
    r: Num, cov_mat: Mat, ex_returns: Vec, rf_rate: Num = None, efficient: bool = True
):
    """
    calc frontier
    """
    ex_returns = np.array(np.atleast_2d(ex_returns)).T
    cov_mat = np.array(cov_mat)
    if rf_rate is None:
        pf = Portfolio([1] + [0 for i in range(len(ex_returns) - 1)])
        a = pf.calc_a(cov_mat, None)
        b = pf.calc_b(cov_mat, ex_returns, None)
        c = pf.calc_c(cov_mat, ex_returns, None)
        d = pf.calc_d(cov_mat, ex_returns, None)
        return sqrt((a * r**2 - 2 * b * r + c) / d)

    return sqrt(
        ((r - rf_rate) ** 2)
        / ((ex_returns - rf_rate).T @ np.linalg.inv(cov_mat) @ (ex_returns - rf_rate))
    )


def critical_frontier_plot(
    cov_mat: Mat,
    ex_returns: Vec,
    rf_rate: Union[float, None],
    portfolios: List[Portfolio] = None,
    portfolio_labels: List[str] = None,
    x_len=0.5,
    samples=1000,
):
    """
    Function to return a plot of the critical/efficient frontier. If risk-free rate is present do not leave as none since this will change the line to be the capital market line. If portfolios should be shown give them as a list, if labels of these should be shown give labels as a list also.
    """
    # evenly sampled time at 200ms intervals
    plt.rcParams.update({"text.usetex": True, "font.family": "Computer Modern"})

    rates = np.linspace(-x_len, x_len, num=int(x_len * samples))
    mvp = MVPortfolio(cov_mat)
    crit = [frontier(r, cov_mat, ex_returns, rf_rate, False) for r in rates]
    if rf_rate is None:
        eff = [
            c
            for i, c in enumerate(crit)
            if rates[i] >= mvp.ex_return(ex_returns, cov_mat, None, None)
        ]
        eff_r = [
            r for r in rates if r >= mvp.ex_return(ex_returns, cov_mat, None, None)
        ]
    else:
        eff = [c for i, c in enumerate(crit) if rates[i] >= rf_rate]
        eff_r = [r for r in rates if r >= rf_rate]

    plt.xlabel(
        r"Standard deviation, $\sigma$",
        fontsize=14,
    )
    plt.ylabel(
        r"Expected return, $\bar{r}$",
        fontsize=14,
    )
    plt.ylim(-x_len / 100, x_len / 100)
    plt.xlim(0, x_len)
    plt.title(r"Critical/Efficient frontier", fontsize=20)
    plt.plot(crit, rates, color="black", label="Critical frontier")
    plt.plot(eff, eff_r, color="red", label="Efficient frontier")
    plt.xticks([0] + [x_len / 10 * i for i in range(10)])
    plt.yticks(
        [-x_len / 10 * i for i in range(10)] + [0] + [x_len / 10 * i for i in range(10)]
    )
    plt.legend()

    if portfolios is not None:
        pf_sd = [sqrt(pf.variance(cov_mat, None)) for pf in portfolios]
        pf_ex_r = [
            pf.ex_return(ex_returns, cov_mat, rf_rate, None)
            if isinstance(pf, TanPortfolio) is False
            else pf.ex_return(ex_returns, cov_mat, rf_rate, None)["return"]
            for pf in portfolios
        ]
        plt.scatter(pf_sd, pf_ex_r, color="red", label="Portfolios", marker="x")
        for i, (x, y) in enumerate(zip(pf_sd, pf_ex_r)):
            plt.text(
                x,
                y,
                portfolio_labels[i],
                color="black",
                fontsize=10,
                ha="right",
                va="bottom",
            )

    plt.show()
