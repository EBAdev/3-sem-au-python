"""
This module was created by Emil Beck Aagaard Kornelussen
Github: https://github.com/EBAdev

### General Assumptions
#### Determinisitc cash flow streams:
* Yearly Compound (m)
        - Generally we assume that the yearly compund of an interest rate is 1, 
        i.e. we compound once at the end of eack period
* Facevalue of bonds (F)
        - We assume that the face value of a bond is 100, 
        if this is not the case pass it as a variable or another value.

Use at own risk, all calculations were however somewhat tested.

Dependencies include but are not limited to:

* Math package 
* Numpy package
* Numpy financial functions
* Sympy
"""

from ifpy.market import *
from ifpy.utils import *

from ifpy.Assets.forward import Forward
from ifpy.Assets.annuity import Annuity
from ifpy.Assets.future import Future
from ifpy.Assets.option import AmOption, EuOption
from ifpy.Assets.swap import CommoditySwap, InterestRateSwap

from ifpy.models.black_scholes import BlackScholes
from ifpy.models.one_period_market import OnePeriodMarket
from ifpy.models.lattice import InterestLattice, BinomialLattice
from ifpy.models.portfolio import (
    Portfolio,
    OptPortfolio,
    MVPortfolio,
    TanPortfolio,
    CML_plot,
    critical_frontier_plot,
)
