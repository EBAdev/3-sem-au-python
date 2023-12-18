from typing import Union
from ifpy.market import discount_factor
from ifpy.Assets.asset import Asset

Num = Union[int, float]


class Swap(Asset):
    def __init__(self, payment_periods: Num, continous: bool = False) -> None:
        """
        1. payment_periods:Num
              * The amount of payment periods until maturity
        """
        super().__init__(continous)
        self.n = payment_periods
        self.a_type = "D"


class CommoditySwap(Swap):
    """
    A general class for commodity swaps.

    In a commodity swap, the variable price is simply the spot price of the underlying commodity.
    * At each pre-specified period t_i, party A pays party B a fixed price X.
    * Party B pays the variable amount S(t_i) at each period, i = 1,...,n.
    * Thus party A is always able to buy the underlying asset at spot price, but pays only a fixed price to do so.
    """

    def fair_price(self, S0: Num, rf_rate: Num):
        """
        Function to determine the fair price of a commodity swap.
        #### LaTex formula
        X=\\frac{nS(0)}{\\sum_{i=1}^{n}d_{0,t_i}}
        #### Paramters
        2. s_0:Num [required]
                * The value of the underlying at time t=0
        3. rf_rate:Num [required]
                * The risk free interest rate
        """

        d_0_ti = [
            discount_factor(rf_rate, 0, i + 1, self.continous) for i in range(self.n)
        ]

        return self.n * S0 / sum(d_0_ti)


class InterestRateSwap(Swap):
    """
    A general class for interest rate swaps.

    In an interest rate swap, the underlying is a pre-specified interest rate. Typically the reference interest rate is the so called LIBOR rate.
    * A fictional notional H is chosen along with a reference interest rate r.
    * At each pre-specified period t_i , party A pays party B a fixed interest rate c based on this notional.
    """

    def __init__(
        self, payment_periods: Num, interest_rate: Num, continous: bool = False
    ) -> None:
        """
        1. payment_periods:Num
              * The amount of payment periods until maturity
        2. interest_rate:Num
              * the reference interest rate (LIBOR rate)
        """
        super().__init__(payment_periods, continous)
        self.r = interest_rate

    def fair_price(self):
        """
        Function to calculate the fair price of a interest rate swap.
        #### LaTex formula
        The fair fixed rate, c, at time 0 of an interest rate swap with periods t_0,..., t_n and reference interest rate r is given by

        $$
        c=\\frac{(1-d_{0,t_n})}{\\sum_{i=1}^{n}d_{0,t_i}}}
        $$
        """
        d_0_ti = [
            discount_factor(self.r, 0, i + 1, self.continous) for i in range(self.n)
        ]

        return (1 - discount_factor(self.r, 0, self.n, self.continous)) / sum(d_0_ti)
