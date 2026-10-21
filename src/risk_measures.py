import pandas as pd
import numpy as np


def hs_var_es(loss: pd.Series, alpha: float = 0.99, window: int = 250) -> pd.DataFrame:
    """Historical-simulation VaR and ES forecasts for the next day. Takes the loss Series as primary argument."""

    VaR = loss.rolling(window=window).quantile(alpha)
    es = loss.rolling(window=window).apply(lambda x: x[x >=np.quantile(x, alpha)].mean(), raw=True)

    out = pd.DataFrame({"var": VaR, "es": es}).shift(1)
    return out

