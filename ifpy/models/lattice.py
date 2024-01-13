from typing import Union, List, Tuple
from scipy import special
from ifpy.utils import column_sum
from ifpy.Assets.option import Option, EuOption, AmOption
from ifpy.Assets.asset import Asset
from ifpy.Assets.annuity import Annuity
from ifpy.market import discount_factor
import numpy as np


Num = Union[int, float]
Mat = Union[
    List[List[float]],
    List[List[int]],
    Tuple[Tuple[float, ...], ...],
    Tuple[Tuple[int, ...], ...],
]


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    except TypeError:
        return False


class Lattice:
    def __init__(
        self, lattice: Mat, risk_neutral_prob: Union[Mat, None], continous=False
    ) -> None:
        self.check_lattice(
            lattice,
        )
        self.lat = lattice
        if is_float(risk_neutral_prob):
            risk_neutral_prob = self.create_rn_prop_lat(risk_neutral_prob)
            self.constant_prob_rn = True
        else:
            self.constant_prob_rn = False

        self.check_rn_prop_lattice(risk_neutral_prob)
        self.q = risk_neutral_prob
        self.continous = continous

    def check_lattice(
        self,
        lattice: Mat,
    ) -> Union[ValueError, None]:
        for idx, step in enumerate(lattice):
            if special.comb(len(step), 1) != len(step):
                raise ValueError(
                    "LatticeError: The provided lattice does not have enough entries to match the amount of periods, please provide a recombining lattice."
                )
            if idx == 0:
                continue

            _ = step.copy()
            _.sort()
            if _ != step:
                raise ValueError(
                    "LatticeError: Please make sure that each step in the provided lattice is sorted, such that the greatest value is the first entry."
                )
        return None

    def create_rn_prop_lat(self, q):
        rn_lat = []
        for i in range(len(self.lat)):
            num_probs = (i + 1) * 2
            col = []
            for j in range(num_probs):
                if j % 2 == 0:
                    col.append(1 - q)
                    col.append(q)
            rn_lat.append(col)
        return rn_lat

    def check_rn_prop_lattice(
        self, risk_neutral_prob_lat: Mat
    ) -> Union[ValueError, None]:
        for i, step in enumerate(risk_neutral_prob_lat):
            num_probs = (i + 1) * 2
            if num_probs != len(step):
                raise ValueError(
                    "LatticeError: The risk neutral lattice dimensions at time "
                    + str(i + 1)
                    + " does not have enough entries. It should have "
                    + str(num_probs)
                )
            for idx, q in enumerate(step):
                if idx % 2 == 0:  # if we are at an even node
                    if (q + step[idx + 1]) != 1:
                        raise ValueError(
                            "LatticeError: Some of the probabilities in the risk neutral lattice does not sum to 1 please check your input."
                        )
        if len(self.lat) != len(risk_neutral_prob_lat):
            raise ValueError(
                "LatticeError: The risk neutral probability lattice, dimension does not match the dimension of the provided lattice. PLease maske sure there is two probabilities for each node."
            )

        return None


class InterestLattice(Lattice):
    def state_prices(self, rounding: Union[int, None] = 4):
        """
        Function to determine the state prices of the interest rate lattice, using the forward formula. Returns the state prices of at each node in the lattice, in the same order as the latice input. i.e. the state price at the most upper node is the last entry in each list.

        ### LaTex
        \\\\begin{equation*}
            \psi_{1,0} = \frac{(1-q)}{1+s_{0,1}(0)}  \quad   
        
            \psi_{1,1} = \frac{(1-q)}{1+s_{0,1}(1)}
        \\\\end{equation*}
        
        \\\\begin{equation*}
            \\psi_{t, s}= \\begin{cases}
                \\frac{(1-q)}{1+s_{t-1, t}(s)} \\psi_{t-1, s}, & \\text { if } s = 0 \\\\

                \\frac{(1-q)}{1+s_{t-1, t}(s)} \\psi_{t-1, s}+\\frac{q}{1+s_{t-1, t}(s-1)} \\psi_{t-1, s-1}, & \\text { if } 0 < s < t \\\\

                \\frac{q}{1+s_{t-1, t}(s-1)} \\psi_{t-1, s-1}, & \\text { if } s = t
            \\end{cases}
        \\\\end{equation*}
        """
        psi_01 = self.q[0][0] / (1 + self.lat[0][0])
        psi_11 = self.q[0][1] / (1 + self.lat[0][0])
        if rounding is None:
            state_prices = [[1], [psi_01, psi_11]]
        else:
            state_prices = [[1], [round(psi_01, rounding), round(psi_11, rounding)]]
        for t in range(len(self.lat) + 1):
            if t <= 1:
                continue
            col = []
            num_states = t + 1
            for s in range(num_states):
                if s == 0:
                    rate = self.lat[t - 1][s]
                    q = self.q[t - 1][s]
                    psi = (q / (1 + rate)) * state_prices[t - 1][s]

                if 0 < s < t:
                    rate_up = self.lat[t - 1][s]
                    rate_down = self.lat[t - 1][s - 1]
                    q_down = self.q[t - 1][s]
                    q_up = self.q[t - 1][s + 1]
                    psi_up = state_prices[t - 1][s]
                    psi_down = state_prices[t - 1][s - 1]
                    psi = (q_up / (1 + rate_up)) * psi_up + (
                        q_down / (1 + rate_down)
                    ) * psi_down
                if s == t:
                    rate = self.lat[t - 1][s - 1]
                    q = self.q[t - 1][s + 1]
                    psi = (q / (1 + rate)) * state_prices[t - 1][s - 1]
                if rounding is None:
                    col.append(psi)
                else:
                    col.append(round(psi, rounding))
            state_prices.append(col)

        return state_prices

    def ZCB_prices(self, rounding: Union[int, None] = 4):
        """
        Function to sum all the state prices and thereby determine the ZCB prices.

        ### LaTex
        \\begin{equation*}
            B_{0,T} = \\sum_{s=0}^{T}\\psi_{T,s}\\quad \\text{when } t=T
        \\end{equation*}
        """
        psi = self.state_prices(None)
        ZCB = {}
        for idx, col in enumerate(psi):
            if idx == 0:
                continue
            csum = column_sum(col)
            if rounding is not None:
                csum = round(csum, rounding)

            ZCB.update({"B_0_" + str(idx): csum})
        return ZCB

    def term_structure(self, rounding: Union[int, None] = 4):
        """
        Function to return the implied term structure, knowing the ZCB prices.
        #### LaTex
        \\begin{equation*}
                s_{0,t}=B^{-\\frac{1}{t}}_{0,t}-1
        \\end{equation*}
        """
        ZCB = self.ZCB_prices(None)
        spot_rates = {}
        for i in range(len(ZCB)):
            B = ZCB["B_0_" + str(i + 1)]
            s = B ** (-1 / (i + 1)) - 1
            if rounding is not None:
                s = round(s, rounding)
            spot_rates.update({"s_0_" + str(i + 1): s})
        return spot_rates

    def rn_price(
        self, payments_per_period: List[Num], T: int, rounding: Union[int, None] = 4
    ):
        """
        Function determine time zero value of payments happening in the future, in the lattice.
        ### Latex
            V_{t-1}^s=\\frac{1}{1+s_{t-1,t}(s)}[qV_t^{s+1}+(1-q)V_{t}^s]
        #### Parameters
        1. payments_per_period:List[Num]
                * List of numbers corresponding to the payments happening at times 1 to T.
        2. T: int
                * The period in the lattice where the payments end.
        """
        lattice = self.lat.copy()
        lattice.reverse()
        payments_per_period.reverse()
        if len(payments_per_period) > T:
            raise ValueError("More payments were provided than the integer value of T.")
        if T > len(self.lat):
            raise ValueError(
                "Cannot discount payments at time T. The value is too big for the lattice."
            )
        if rounding is None:
            V = [[payments_per_period[0] for t in range(T + 1)]]
        else:
            V = [[round(payments_per_period[0], rounding) for t in range(T + 1)]]
        for i, step in enumerate(lattice):
            t = T - i - 1
            if len(step) - 1 > T or t < 0:
                continue

            col = []
            for s, rate in enumerate(self.lat[t]):
                q = self.q[t][s]
                v_st = V[T - t - 1][s]
                v_s1t = V[T - t - 1][s + 1]
                v = (1 / (1 + rate)) * (q * v_s1t + (1 - q) * v_st)
                if t > 0:
                    if rounding is None:
                        col.append(v + payments_per_period[i - 1])
                    else:
                        col.append(round(v + payments_per_period[i - 1], rounding))
                else:
                    if rounding is None:
                        col.append(v)
                    else:
                        col.append(round(v, rounding))
            V.append(col)
        V.reverse()
        return V

    def remaining_ann_debt(
        self,
        annuity: Annuity,
        coupon_rate: float,
        rounding: Union[int, None] = 4,
    ):
        Rt = [annuity.nominal]
        for i in range(annuity.T + 1):
            if i == 0:
                continue
            remaining = max(
                Rt[i - 1]
                - annuity.per_period_payment(coupon_rate, None)
                * discount_factor(
                    coupon_rate, 0, annuity.T - i + 1, continous=annuity.continous
                ),
                0,
            )
            if np.isclose(remaining, 0, atol=0.0001):
                remaining = 0
            if rounding is None:
                Rt.append(remaining)
            else:
                Rt.append(round(remaining, rounding))
        return Rt

    def rn_call_ann_price(
        self, annuity: Annuity, coupon_rate: float, rounding: Union[int, None] = 4
    ):
        """
        Function determine time zero value of payments happening in the future, in the lattice.
        ### Latex
            V_{t-1}^s=\\frac{1}{1+s_{t-1,t}(s)}[qV_t^{s+1}+(1-q)V_{t}^s]
        #### Parameters
        1. payments_per_period:List[Num]
                * List of numbers corresponding to the payments happening at times 1 to T.
        2. T: int
                * The period in the lattice where the payments end.
        """
        lattice = self.lat.copy()
        lattice.reverse()

        if rounding is None:
            V = [
                [
                    annuity.per_period_payment(coupon_rate, None)
                    for t in range(annuity.T + 1)
                ]
            ]
        else:
            V = [
                [
                    round(annuity.per_period_payment(coupon_rate, None), rounding)
                    for t in range(annuity.T + 1)
                ]
            ]

        for i, step in enumerate(lattice):
            t = annuity.T - i - 1
            if len(step) - 1 > annuity.T or t < 0:
                continue
            remaining_debt = self.remaining_ann_debt(annuity, coupon_rate, None)
            col = []
            for s, rate in enumerate(self.lat[t]):
                q = self.q[t][s]
                v_st = V[annuity.T - t - 1][s]
                v_s1t = V[annuity.T - t - 1][s + 1]
                v = (1 / (1 + rate)) * (q * v_s1t + (1 - q) * v_st)

                if t > 0:
                    if rounding is None:
                        col.append(
                            min(
                                v + annuity.per_period_payment(coupon_rate, None),
                                remaining_debt[t],
                            )
                        )
                    else:
                        col.append(
                            round(
                                min(
                                    v + annuity.per_period_payment(coupon_rate, None),
                                    remaining_debt[t],
                                ),
                                rounding,
                            )
                        )
                else:
                    if rounding is None:
                        col.append(
                            min(
                                v,
                                remaining_debt[t],
                            )
                        )
                    else:
                        col.append(
                            round(
                                min(
                                    v,
                                    remaining_debt[t],
                                ),
                                rounding,
                            )
                        )
            V.append(col)
        V.reverse()
        return V


class BinomialLattice(Lattice):
    def __init__(
        self,
        lattice: Mat,
        risk_free_rate: float,
        prob: Mat | None | float,
        risk_neutral_prob: Mat | None,
        continous=False,
    ) -> None:
        self.check_lattice(lattice)

        self.lat = lattice
        self.r = risk_free_rate
        self.continous = continous

        if prob is not None:
            if is_float(prob):
                self.prob = self.create_rn_prop_lat(prob)
                self.constant_prob = True
            else:
                self.constant_prob = False
            if risk_neutral_prob is None:
                risk_neutral_prob = (
                    self.calc_rn_prob()
                )  ## Assumes constant q then, with constant d and u

        if is_float(risk_neutral_prob):
            risk_neutral_prob = self.create_rn_prop_lat(risk_neutral_prob)
            self.constant_prob_rn = True
        else:
            self.constant_prob_rn = False

        self.check_rn_prop_lattice(risk_neutral_prob)
        self.q = risk_neutral_prob

    def calc_rn_prob(self) -> float:
        d = self.lat[1][0] / self.lat[0][0]
        u = self.lat[1][1] / self.lat[0][0]

        q = ((1 / discount_factor(self.r, 0, 1, self.continous)) - d) / (u - d)
        return q

    def price_option(self, opt: Option, rounding: int | None = 4):
        if opt.T > len(self.lat) - 1:
            raise ValueError(
                "The provided option has a higher maturity than the lattice"
            )
        if isinstance(opt, EuOption):
            disc = discount_factor(self.r, 0, opt.T, self.continous)
            to_sum = 0
            for k in range(opt.T + 1):
                num_paths = int(special.binom(opt.T, k))
                if self.constant_prob_rn is False:
                    print(
                        "Warning: The current implementation does not support variable risk neutral prob."
                    )
                payoff = max(self.lat[opt.T][k] - opt.K, 0)
                q = self.q[0][1]
                to_sum += (1 - q) ** (opt.T - k) * q ** (k) * num_paths * payoff
            if rounding is None:
                return disc * to_sum
            else:
                return round(disc * to_sum, rounding)
        elif isinstance(opt, AmOption):
            rlat = self.lat
            rlat.reverse()
            state_values = []
            eu_prices = {}
            for idx, col in enumerate(rlat):
                t = opt.T - idx
                state_val_col = []
                eu_price_col = []
                for s, element in enumerate(col):
                    if t == opt.T:
                        if opt.opt_type == "P":
                            P = max(opt.K - element, 0)
                        else:
                            P = max(element - opt.K, 0)
                        if rounding is None:
                            state_val_col.append(P)
                        else:
                            state_val_col.append(round(P, rounding))
                    elif t < opt.T:
                        disc = discount_factor(self.r, 0, 1)
                        q = self.q[0][1]
                        P_d = state_values[idx - 1][s]
                        P_u = state_values[idx - 1][s + 1]
                        P_s = disc * (q * P_u + (1 - q) * P_d)
                        if t != 0:
                            eu_prices.update({"P_" + str(t) + "_" + str(s): P_s})

                        if opt.opt_type == "P":
                            IV_s = max(P_s, opt.K - element)
                        else:
                            IV_s = max(P_s, element - opt.K)
                        if rounding is None:
                            state_val_col.append(IV_s)
                        else:
                            state_val_col.append(round(IV_s, rounding))

                state_values.append(state_val_col)
            state_values.reverse()
            return (state_values, eu_prices)
        else:
            raise ValueError(
                "The provided option is not currently supported, please provide another option"
            )
