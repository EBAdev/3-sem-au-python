from typing import Union
from ifpy.Assets.asset import Asset
from ifpy.Assets.forward import Forward
from ifpy.market import discount_factor

Num = Union[int, float]


class Future(Asset):
    """
    Genereral class for futures contracts.
    """

    def __init__(self, T: Num, continous: bool = False) -> None:
        """
        1. T:Num
              * The maturity of the future
        """
        super().__init__(continous)
        self.a_type = "D"
        self.T = T

    def price(self, St: Num, r: Num, t: Num = 0):
        """

        #### LaTex formula
        The price of a forward is given by:

        F_{t,T}=\\frac{\\mathbb{E}^{\\mathbb{Q}}[d_{t,T},S(T)]}{B_{t,T}} = \\mathbb{E}^{\\mathbb{Q}}[S(T)] + \\frac{\\text{Cov}(d_{t,T}, S(T))}{B_{t,T}}

        The price of a future is given by:

        G_{0,T} = \\mathbb{E}^{\\mathbb{Q}}[S(T)] = F_{0,T}

        Hence if interest rates are deterministic, the price of the future is equal to the price of the forward since the covariance term cancels.
        """
        F = Forward(self.T, continous=self.continous)
        return F.price(St, r, t)
