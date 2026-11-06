import pandas as pd
import numpy as np
from scipy.stats import chi2
from scipy.special import xlogy
from scipy import stats
from tabulate import tabulate

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



def backtest_summary(hit: pd.Series, model: str, p:float = 0.01)-> pd.DataFrame:
    """Returns a dataframe containing the violation rate, independence test, and conditional and unconditional coverage tests."""

    save_data = {}
    save_data['days'] = hit.index.size
    save_data['violations'] = hit.sum()
    save_data['expected'] = p * save_data['days']

    save_data['rate'] = save_data['violations'] / save_data['days']
    
    # Test Values
    save_data['LR_uc'], save_data['p_uc'] = kupiec_test(hit)['LR'], kupiec_test(hit)['p_value']
    save_data['LR_ind'], save_data['p_ind'] = christoffersen_independence(hit)['LR'], christoffersen_independence(hit)['p_value']
    save_data['LR_cc'], save_data['p_cc'] = conditional_coverage(hit)['LR_conditional'], conditional_coverage(hit)['p_value']

    return pd.DataFrame(data = save_data, index=[model])

def es_backtest_mf(losses: pd.Series, var: pd.Series, es: pd.Series, sigma: pd.Series) -> dict:
    """Returns the McNeil-Frey exceedance residual test for ES. """

    breach_days= (losses > var)
    n_hits = int(breach_days.sum())

    r = (losses[breach_days] - es[breach_days])/sigma[breach_days]
    t = r.mean() / (r.std(ddof=1) / np.sqrt(n_hits))

    return {'n_hits': n_hits,
            'mean_resid': r.mean(),
            't_stat': t,
            'p_value': 1 - stats.t.cdf(t, df = n_hits-1)}



def es_backtest_mf_boot(losses: pd.Series, var: pd.Series, es: pd.Series, sigma: pd.Series, n_boot: int = 10000, seed: int = 42) -> dict:
    """McNeil-Frey ES backtest with a bootstrap p-value."""

    rng = np.random.default_rng(seed)
    breach_days= (losses > var)  
    r = ((losses[breach_days] - es[breach_days])/sigma[breach_days]).values
    r0 = (r - r.mean())
    n = len(r)


    t_values = []
    t_obs = r.mean() / (r.std(ddof=1) / np.sqrt(n))

    for _ in range(n_boot):    

        r_boot = rng.choice(r0, n, replace=True)
        t = r_boot.mean() / (r_boot.std(ddof = 1)/np.sqrt(n))
        t_values.append(t)

    p_boot = np.mean(np.array(t_values) >= t_obs)

    return {'n_hits': n,
            'mean_resid': r.mean(),
            't_stat': t_obs,
            'p_boot': p_boot}


def e_values(losses: pd.Series, var: pd.Series, es: pd.Series,  p: float = 0.99) -> pd.Series:
    """Daily e-values for a (VaR, ES) forecast. (Wang, Wang and Ziegel)."""

    e = ((losses - var).clip(lower = 0)) / ((1 - p) * (es - var))

    return e