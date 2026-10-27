import pandas as pd
import numpy as np
from scipy.stats import norm

def hs_var_es(loss: pd.Series, alpha: float = 0.99, window: int = 250) -> pd.DataFrame:
    """Historical-simulation VaR and ES forecasts for the next day. Takes the loss Series as primary argument."""

    VaR = loss.rolling(window=window).quantile(alpha)
    es = loss.rolling(window=window).apply(lambda x: x[x >=np.quantile(x, alpha)].mean(), raw=True)

    out = pd.DataFrame({"var": VaR, "es": es}).shift(1)
    return out


def ewma_volatility(losses: pd.Series, lam: float = 0.94, init_window: int = 30) -> pd.Series:
    """EWMA (RiskMetrics) volatility forecast for each day."""

    variances = np.ones(losses.size)
    variances[init_window] = losses[:init_window].var()
    variances[:init_window] = np.nan

    for t in range(init_window + 1, len(variances)):
        variances[t] = lam * variances[t-1] + (1 - lam) * losses.values[t-1]**2

    return pd.Series(np.sqrt(variances), index=losses.index, name="ewma_vol")

def ewma_var_es(sigma: pd.Series, alpha: float = 0.99) -> pd.DataFrame:
    """Exponentially Weighted Moving Average Method for VaR and ES forecasts for the next day. Takes the loss Series as primary argument. 
    Returns a `pd.DataFrame` containing both measures.""" 

    z = norm.ppf(alpha) 
    VaR = z * sigma 
    ES = (sigma * norm.pdf(z)) / (1 - alpha)

    return pd.DataFrame({'VaR' : VaR, 'ES' : ES}, index = sigma.index)