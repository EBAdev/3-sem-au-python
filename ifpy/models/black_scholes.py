from typing import Union, Tuple  # bruges til at typehinte
from math import log, exp, sqrt
from scipy import stats
from ifpy.Assets.option import Option, EuOption, AmOption

Num = Union[int, float]  # lav ny type så at S kan være enten heltal eller kommatal


class BlackScholes:
    """
    A class for working with the Black-Scholes model.
    """

    def __init__(self, r: Num, sigma: Num):
        """
        A class for working with the Black-Scholes model
        #### Parameters
        1. option:Option
                * A continous instance of the Option class
        2. r:Num
                * The risk free interest rate
        3. sigma
                * The sigma/volatility of the underlying asset.
        """
        self.r = r
        self.sigma = sigma

    def __opt_check(self, option: Option):
        """
        Func to raise value error if option is not continous.
        """
        if option.continous is False:
            raise ValueError(
                "The BlackScholes model can only be used for continous options, "
                + "please provide a continous option."
            )

    def d1(self, St: Num, K: Num, T: Num, t: Num = 0):
        """
        Function to calculate the d1 of the black scholes model
        #### Parameters
        1. St: Num
                * The price of the underlying asset at time t
        2. K: Num
                * The strike price of the option
        3. T: Num
                * The maturity of the option
        4. t: Num
                * The time at which the the option is to be priced
        #### LaTeX formula
        d_1=\\frac{\\log \\left(\\frac{S(t)}{K}\\right)+\\left(r+\\frac{\\sigma^2}{2}\\right)(T-t)}{\\sigma \\sqrt{T-t}}
        """
        return (log(St / K) + (self.r + self.sigma**2 / 2) * (T - t)) / (
            self.sigma * sqrt(T - t)
        )

    def d2(self, St: Num, K: Num, T: Num, t: Num = 0):
        """
        Function to calculate the d2 of the black scholes model
        #### Parameters
        1. St: Num
                * The price of the underlying asset at time t
        2. K: Num
                * The strike price of the option
        3. T: Num
                * The maturity of the option
        4. t: Num
                * The time at which the the option is to be priced
        #### LaTeX formula
        d_2=d_1-\\sigma \\sqrt{T-t}
        """
        return self.d1(St, K, T, t) - self.sigma * sqrt(T - t)

    def formula(self, option: Option, St: Num, t: Num = 0) -> Tuple[Num, Num, Num]:
        """
        Function to price a European Option in the black scholes model
        #### Parameters
        1. St: Num
                * The price of the underlying asset at time t
        3. t: Num
                * The time at which the the option is to be priced

        #### Return value
        A tuple of three values in the following order: (d_1, d_2, option price)
        
        #### LaTex import
        In the Black-Scholes model, the price of a European call and put option is given by
        
        $$
        \\begin{aligned}
        & C_t=S(t) \\Phi\\left(d_1\\right)-K e^{-r(T-t)} \\Phi\\left(d_2\\right) \\\\
        & P_t=K e^{-r(T-t)} \\Phi\\left(-d_2\\right)-S(t) \\Phi\\left(-d_1\\right)
        \\end{aligned}
        $$
        
        The coefficients $d_1$ and $d_2$ in the Black-Scholes formula are given by
        
        $$
        \\begin{aligned}
        & d_1=\\frac{\\log \\left(\\frac{S(t)}{K}\\right)+\\left(r+\\frac{\\sigma^2}{2}\\right)(T-t)}{\\sigma \\sqrt{T-t}} \\\\
        & d_2=d_1-\\sigma \\sqrt{T-t}
        \\end{aligned}
        $$
        
        and $\\Phi(x)$ is the cdf of the standard normal distribution:
        
        $$
        \\Phi(x)=\\frac{1}{\\sqrt{2 \\pi}} \\int_{-\\infty}^x e^{-\\frac{z^2}{2}} \\mathrm{~d} z
        $$
        """
        self.__opt_check(option)
        if isinstance(option, EuOption) is False or option.opt_type not in [
            "C",
            "P",
        ]:
            return ValueError(
                "The provided option type is not supported, please either provide a European put or call option."
            )
        if t > option.T:
            return ValueError(
                "The provided time t, at which the option is to be priced, is greater than the options maturity."
            )
        d1 = self.d1(St, option.K, option.T, t)
        d2 = self.d2(St, option.K, option.T, t)

        if option.opt_type == "P":
            return (
                d1,
                d2,
                option.K * exp(-self.r * (option.T - t)) * stats.norm.cdf(-d2)
                - St * stats.norm.cdf(-d1),
            )

        if option.opt_type == "C":
            return (
                d1,
                d2,
                St * stats.norm.cdf(d1)
                - option.K * exp(-self.r * (option.T - t)) * stats.norm.cdf(d2),
            )

    def greeks(
        self,
        option: Option,
        St: Num,
        t: Num,
    ):
        """
        Function to return the Greeks for a Eurpean call or put option.
        ### Parameters
        1. S_t: Num
                * The price of the underlying asset at time t
        2. t: Num
                * The time at which the greeks are calculated
        3. sigma: Num
                * The sigma/volatility of the underlying
        ### LaTeX formulas
        1. Call options:

        \\begin{aligned}
        \\Delta & =\\Phi\\left(d_1\\right) \\\\
        \\Gamma & =\\frac{\\varphi\\left(d_1\\right)}{S(t) \\sigma \\sqrt{T-t}} \\\\
        \\Theta & =-\\frac{\\sigma S(t) \\varphi\\left(d_1\\right)}{2 \\sqrt{T-t}}-r K e^{-r(T-t)} \\Phi\\left(d_2\\right) \\\\
        \\rho & =K(T-t) e^{-r(T-t)} \\Phi\\left(d_2\\right) \\\\
        \\mathcal{V} & =S(t) \\sqrt{T-t} \\varphi\\left(d_1\\right)
        \\end{aligned}

        2. Put options:

        \\begin{aligned}
        \\Delta & =-\\Phi\\left(-d_1\\right) \\\\
        \\Gamma & =\\frac{\\varphi\\left(-d_1\\right)}{S(t) \\sigma \\sqrt{T-t}} \\\\
        \\Theta & =-\\frac{\\sigma S(t) \\varphi\\left(d_1\\right)}{2 \\sqrt{T-t}}+r K e^{-r(T-t)} \\Phi\\left(-d_2\\right) \\\\
        \\rho & =-K(T-t) e^{-r(T-t)} \\Phi\\left(-d_2\\right) \\\\
        \\mathcal{V} & =S(t) \\sqrt{T-t} \\varphi\\left(d_1\\right)
        \\end{aligned}


        """
        self.__opt_check(option)
        if option.continous is False or isinstance(option, EuOption) is False:
            raise ValueError(
                "Greeks can only be calculated for continous European put or call options,"
                + "please provide a European continous put or call option."
            )
        d1 = self.d1(St, option.K, option.T, t)
        d2 = self.d2(St, option.K, option.T, t)

        if option.opt_type == "C":
            delta = stats.norm.cdf(d1)
            gamma = stats.norm.pdf(d1) / (St * self.sigma * sqrt(option.T - t))
            theta = -(self.sigma * St * stats.norm.pdf(d1)) / (
                2 * sqrt(option.T - t)
            ) - self.r * option.K * exp(-self.r * (option.T - t)) * stats.norm.cdf(d2)
            rho = (
                option.K
                * (option.T - t)
                * exp(-self.r * (option.T - t))
                * stats.norm.cdf(d2)
            )
            vega = St * sqrt(option.T - t) * stats.norm.pdf(d1)

        elif option.opt_type == "P":
            delta = -stats.norm.cdf(-d1)
            gamma = stats.norm.pdf(-d1) / (St * self.sigma * sqrt(option.T - t))
            theta = -(self.sigma * St * stats.norm.pdf(d1)) / (
                2 * sqrt(option.T - t)
            ) - self.r * option.K * exp(-self.r * (option.T - t)) * stats.norm.cdf(-d2)
            rho = (
                -option.K
                * (option.T - t)
                * exp(-self.r * (option.T - t))
                * stats.norm.cdf(-d2)
            )
            vega = St * sqrt(option.T - t) * stats.norm.pdf(d1)

        return {
            "Delta": delta,
            "Gamma": gamma,
            "Theta": theta,
            "Rho": rho,
            "Vega": vega,
        }
