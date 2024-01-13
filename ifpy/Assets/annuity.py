from typing import Union, List

from ifpy.Assets.asset import Asset

Num = Union[int, float]


class Annuity(Asset):
    def __init__(
        self,
        maturity: int,
        nominal: Num = 1,
        compounding_per_year: int = 1,
        continous: bool = False,
    ) -> None:
        super().__init__(continous)
        self.a_type = "B"
        self.nominal = nominal
        self.T = maturity
        self.m = compounding_per_year

    def price(self, annual_payment: Num, ytm: float, rounding: Union[int, None] = 4):
        """
        A(T) = \\frac{a}{\\lambda}\\bbr{1-\\frac{1}{(1+\\frac{\\lambda}{m})^{mT}}}
        """
        a = annual_payment
        if rounding is None:
            return (a / ytm) * (1 - 1 / (1 + ytm / self.m) ** (self.m * self.T))
        return round(
            (a / ytm) * (1 - 1 / (1 + ytm / self.m) ** (self.m * self.T)), rounding
        )

    def factor(self, coupon_rate: float, rounding: Union[int, None] = 4):
        """
        We want to determine the per period payments, of this annuity, which we will denote $a$. We have shown that the arbitrage free price of this annuity is given by:

        \\begin{equation*}
          1 = \\frac{a}{c}\\bbr{1-\\frac{1}{(1+\\frac{c}{m})^{mT}}}
        \\end{equation*}
        Where $a$ is the annual payment of the annuity  and $m$ is the amount of compounding per year. If we assume that $m=1$ the formula simplifies to:

        \\begin{equation*}
          1= \\frac{a}{c}\\bbr{1-\\frac{1}{(1+c)^{T}}} \\Rightarrow a=\\br{\\frac{1-(1+c)^{-T}}{c}}^{-1} =
        \\end{equation*}
        """
        if rounding is None:
            return (
                (1 - (1 + coupon_rate / self.m) ** (-self.T * self.m)) / coupon_rate
            ) ** -1
        return round(
            ((1 - (1 + coupon_rate / self.m) ** (-self.T * self.m)) / coupon_rate)
            ** -1,
            rounding,
        )

    def per_period_payment(self, coupon_rate, rounding: Union[int, None] = 4):
        """
        We want to determine the per period payments, of this annuity, which we will denote $a$. We have shown that the arbitrage free price of this annuity is given by:

        \\begin{equation*}
          1 = \\frac{a}{c}\\bbr{1-\\frac{1}{(1+\\frac{c}{m})^{mT}}}
        \\end{equation*}
        Where $a$ is the annual payment of the annuity  and $m$ is the amount of compounding per year. If we assume that $m=1$ the formula simplifies to:

        \\begin{equation*}
          1= \\frac{a}{c}\\bbr{1-\\frac{1}{(1+c)^{T}}} \\Rightarrow a=\\br{\\frac{1-(1+c)^{-T}}{c}}^{-1} =
        \\end{equation*}

        We then rescale this factor by the nominal of the annuity to get:
        """
        if rounding is None:
            return self.factor(coupon_rate, None) * self.nominal
        return round(self.factor(coupon_rate, None) * self.nominal, rounding)
