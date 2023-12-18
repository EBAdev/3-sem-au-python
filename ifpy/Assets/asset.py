from typing import Union  # bruges til at typehinte

Num = Union[int, float]  # lav ny type så at S kan være enten heltal eller kommatal


class Asset:
    """
    General superclass for all assets.
    """

    def __init__(self, continous: bool = False) -> None:
        self.continous = continous
        self.a_type = None  # str to hold the type of asset. 'D' = derivative
