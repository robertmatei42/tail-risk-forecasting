import pandas as pd
import numpy as np
from scipy.stats import chi2



def kupiec_test(hit: pd.Series, p: float = 0.01) -> dict:
    """Computes the Kupiec likelihood ratio test. Takes a boolean hit series as argument. """

    n = hit.size
    x = hit.sum()
    pi_hat = x / n

    l1 = (n - x) * np.log(1 - p) + x * np.log(p) if x > 0 else 0
    l2 = (n - x) * np.log(1 - pi_hat) + x * np.log(pi_hat) if x > 0 else 0 # Handle edge cases

    LR = -2 * (l1 - l2)
    p_value = chi2.sf(LR, df=1)

    return {'n': int(n), 
            'x': int(x),
            'rate': float(hit.mean()),
            'LR': float(LR),
            'p_value': float(p_value)}
