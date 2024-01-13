from typing import Union, List, Tuple
from ifpy.utils import column_sum
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


class OnePeriodMarket:
    def __init__(
        self,
        value_matrix: Mat,
        price_vector: Vec = None,
        state_prices: Mat = None,
        probabilities: Vec = None,
    ):
        self.P = np.atleast_2d(np.array(price_vector)).T
        self.D = np.array(value_matrix)
        self.pi = np.atleast_2d(np.array(probabilities)).T
        self.psi = state_prices
        if state_prices is None and price_vector is not None:
            self.psi = self.calc_psi()
        elif price_vector is None and state_prices is not None:
            self.P = self.calc_p()
        elif state_prices is None and price_vector is None:
            raise ValueError(
                "To use the model, we must know either the price vector or the state prices, please provide one of those values"
            )

    def calc_psi(self):
        """
        Function to return the stateprices in the market, will raise error if they are not deterministic.
        """
        try:
            np.linalg.inv(self.D)
        except np.LinAlgError:
            print(
                "Since the value matrix is not invertible, we conclude that the market is not complete, to use the model this is a requirement."
            )
        else:
            return np.linalg.inv(self.D) @ self.P

    def calc_p(self):
        """
        Function to return the prices in the market, will raise error if they are not deterministic.
        """
        return self.D @ self.psi

    def arbitrage(self) -> bool:
        """
        The model is arbitrage free if the state prices are positive.
        """
        for s in self.psi:
            if s <= 0:  # if a stateprice is non-positive.
                return True
        return False

    def complete(self) -> bool:
        """
        The model is complete if it is arbitrage free and the state prices are unique.
        """
        try:
            np.linalg.inv(self.D)
        except np.LinAlgError:
            return False  # if value matrix is not invertible, the state prices are not unique.
        # the model is then complete if it is arbitrage free
        return not self.arbitrage()

    def risk_neutral_prob(self, rounding: Union[int, None] = 4):
        """
        Function to return the risk-netutral probabilities and the price of a risk-free asset assuming the model is arbitrage free.
        """
        if self.arbitrage() is True:
            raise ValueError(
                "The provided state prices for the model are not strictly positive, therefore the model is not arbitrage free.\n"
                + "This means that the risk-neutral probabilities are not deterministic. "
            )
        rf_a_p = column_sum(self.psi)
        rf_rate = 1 / rf_a_p - 1
        q = [p[0] / rf_a_p for p in self.psi]
        if rounding is None:
            return {"rf_asset_p": rf_a_p, "rf_rate": rf_rate, "q_s": q}
        else:
            return {
                "rf_asset_p": round(rf_a_p, rounding),
                "rf_rate": round(rf_rate, rounding),
                "q_s": np.round(q, rounding),
            }

    def expected_value(
        self,
        state_values: List[Num],
        probability_measure="Q",
        rounding: Union[int, None] = 4,
    ):
        """
        Function to return the expected value of a value stream under a probability measure of either P or Q.
        """
        if len(state_values) != self.D.shape[0]:
            raise ValueError(
                "The state_values does not match the probability measure, please provide a state value for each state."
            )
        if probability_measure == "P":
            if self.pi is None:
                raise ValueError(
                    "The model has cannot use the probability measure P, since the objective probabilities haven't been provieded."
                )
            ex_value = sum([v * self.pi[i][0] for i, v in enumerate(state_values)])
        else:
            q = self.risk_neutral_prob(None)["q_s"]
            ex_value = sum([d * q[i] for i, d in enumerate(state_values)])
        if rounding is None:
            return ex_value
        return round(ex_value, rounding)

    def asset_price(self, state_values: List[Num], rounding: Union[int, None] = 4):
        """
        Function to price a asset in the model, using the Q probability measure.
        """
        if len(state_values) != self.psi.shape[0]:
            raise ValueError(
                "The state values does not match the probability measure, please provide a state value for each state."
            )
        ex_value = self.expected_value(state_values=state_values, rounding=None)
        rf_rate = self.risk_neutral_prob(None)["rf_rate"]
        disc = 1 / (1 + rf_rate)
        if rounding is None:
            return disc * ex_value
        return round(disc * ex_value, rounding)
