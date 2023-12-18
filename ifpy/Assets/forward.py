from typing import List, Union
from ifpy.Assets.asset import Asset
from ifpy.market import discount_factor

Num = Union[int, float]


class Forward(Asset):
    """
    Genereral class for forward contracts.
    """

    def __init__(
        self, T: Num, cost_of_carry: List[Num] = None, continous: bool = False
    ) -> None:
        """
        1. T:Num
              * The maturity of the forward
        2. cost_of_carry:List[Num] = None
              * A ordered list of the (storage)costs incurred from holding the forward.
        """
        super().__init__(continous)
        self.a_type = "D"
        self.T = T
        self.carry_cost = cost_of_carry

    def price(self, St: Num, r: Num, t: Num = 0):
        """
        Function to return the price of a forward contract, assumes that there is no arbitrage and that interest rates are deterministic (it is possible to short the forward).
        #### Parameters
        1. St:Num
              * The price of the underlying at time t
        2. r:Num
              * The deterministic interest rate r of the underlying or in the market.
        3. t:Num=0
              * the time at which we want to price the forward
        #### LaTex formula
        Assume that a cost of carry $c_k$ is incurred at periods $k \\in\\{t, \\ldots, M\\}$ with $M<T$, then the forward price is:

        $$
        F_{t, T}=\\frac{S(t)}{d_{t, T}}+\\sum_{k=t}^M \\frac{c_k}{d_{k, T}}
        $$

        if there is zero cost of carry the sum is 0. If there is arbitrage in the market the RHS of the equality turns into an upper bound for $F_{t, T}$. ($\\leq$)
        """
        P = St / discount_factor(r, t, self.T, self.continous)
        if self.carry_cost is not None:
            disc_cf = [
                c / discount_factor(r, t + idx, self.T)
                for idx, c in enumerate(self.carry_cost)
            ]
            P += sum(disc_cf)
        return P
