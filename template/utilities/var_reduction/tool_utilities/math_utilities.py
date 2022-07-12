import math as m
from math import log10, floor

import numpy as np


def round_sig(x, sig=2):
    if np.isnan(x):
        return "NaN"
    elif x == 0:
        return 0
    elif m.isinf(x):
        return float("inf")
    else:
        return round(x, sig-int(floor(log10(abs(x))))-1)
