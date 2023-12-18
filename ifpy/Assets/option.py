"""
General module for option class
"""

from typing import Union
from ifpy.Assets.asset import Asset
from ifpy.market import discount_factor


Num = Union[int, float]  # lav ny type så at S kan være enten heltal eller kommatal


class Option(Asset):
    """
    General class to hold functions that apply to all options
    """

    def __init__(
        self,
        strike: Num,
        T: Num,
        opt_type: str = "C",
        continous: bool = False,
    ) -> None:
        """
        1. strike: Num
                    * The strike price of the option
        2. T:Num
                    * The time to maturity of the option
        3. opt_type: str = "C"
                * The option type, should be a string corresponding to either put or call.
                    * For put use: ["p", "P", "put", "Put"]
                    * For call use: ["c", "C", "call", "Call"]
        4. continous: bool = False
              * boolean to control if the option is in continous time.
        """
        ### known values based on cls
        super().__init__(continous)
        self.a_type = "D"
        self.K = strike
        self.T = T

        ### Provided values
        if opt_type in ["p", "P", "put", "Put"]:
            self.opt_type = "P"
        elif opt_type in ["c", "C", "call", "Call"]:
            self.opt_type = "C"
        else:
            raise ValueError(
                "The provided option type is currently not supported, please choose either a 'put' or 'call' option."
            )


class EuOption(Option):
    """
    Class to hold all functions specific to European options
    """

    def __init__(
        self,
        strike: Num,
        T: Num,
        opt_type: str = "C",
        continous: bool = False,
    ) -> None:
        super().__init__(strike, T, opt_type, continous)
        self.region = "E"

    def put_call_parity(
        self,
        price_t: Num,
        t: Num,
        T: Num,
        rf_rate: Num,
        St: Num,
    ):
        """
        Function to determine the price of a European call/put option, using put-call parity.
        #### LaTex formula
        C_t-P_t+d_{t,T}K=S(t)
        * Put option solved: P_t=C_t+d_{t,T}K-S(t)
        * Call option solved: C_t=S(t)+P_t-d_{t,T}K
        #### Parameters
        1. option_price_t:Num [required]
                * The price of the known option at time t
        2. t:Num [required]
                * The time at which the option is to be priced
        3. T:Num [required]
                * The time to maturity of the options
        4. rf_rate:Num [required]
                * The risk free interest rate
        5. K:Num [required]
                * The stike price of the options
        6. s_t:Num [required]
                * The price of the underlying at time t
        """
        disc = discount_factor(rf_rate, t, T)
        if self.opt_type == "C":
            return price_t + disc * self.K - St
        return St - disc * self.K + price_t


class AmOption(Option):
    """
    Class to hold all functions specific to American options
    """

    def __init__(
        self, strike: Num, T: Num, opt_type: str = "C", continous: bool = False
    ) -> None:
        super().__init__(strike, T, opt_type, continous)
        self.region = "A"
