import pandas as pd
import numpy as np
from scipy.stats import chi2
from scipy.special import xlogy


def kupiec_test(hit: pd.Series, p: float = 0.01) -> dict:
    """Computes the Kupiec likelihood ratio test. Takes a boolean hit series as argument."""

    n = hit.size
    x = hit.sum()
    pi_hat = x / n

    l1 = xlogy(n - x, 1 - p) + xlogy(x, p)
    l2 = xlogy(n - x, 1 - pi_hat) + xlogy(x, pi_hat)

    LR = -2 * (l1 - l2)
    p_value = chi2.sf(LR, df=1)

    return {'n': int(n), 
            'x': int(x),
            'rate': float(hit.mean()),
            'LR': float(LR),
            'p_value': float(p_value)}


def christoffersen_independence(hit: pd.Series) -> dict:
    """Computes the Christoffersen Independence test. Takes a boolean hit series as argument."""

    h = hit.astype(int).to_numpy()
    prev = h[:-1]
    curr = h[1:]
    
    # Transitions
    n00 = ((prev == 0) & (curr == 0)).sum()
    n01 = ((prev == 0) & (curr == 1)).sum()
    n10 = ((prev == 1) & (curr == 0)).sum()
    n11 = ((prev == 1) & (curr == 1)).sum()

    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n01 + n10 + n11 + n00)

    log_L_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    log_L_alternative  = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)

    LR_ind = -2 * (log_L_null -  log_L_alternative)
    p_value = chi2.sf(LR_ind, df=1)

    return {'n00' : int(n00),
            'n01' : int(n01),
            'n10' : int(n10),
            'n11' : int(n11),
            'pi01' : float(pi01),
            'pi11' : float(pi11),
            'LR': float(LR_ind),
            'p_value' : float(p_value)}


def conditional_coverage(hit: pd.Series, p: float = 0.01) -> dict:
    """Returns the results of the Christoffersen Conditional Coverage test as a dictionary. Takes a boolean hit series as argument."""

    LR_unconditional = kupiec_test(hit, p)['LR']
    LR_independence = christoffersen_independence(hit)['LR']

    LR_conditional = LR_independence + LR_unconditional
    p_value = chi2.sf(LR_conditional, df = 2)

    return {'LR_unconditional' : LR_unconditional,
            'LR_independence' : LR_independence,
            'LR_conditional' : LR_conditional,
            'p_value' : float(p_value)}



