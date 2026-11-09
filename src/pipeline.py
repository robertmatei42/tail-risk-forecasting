from src.data import load_prices, compute_losses
from src.risk_measures import ewma_volatility, hs_var_es, ewma_var_es, fhs_var_es



def build_models() -> dict:
    """Returns the primary variables for all models in the notebooks, such as loss, sigma_ewma, fhs_main."""

    raw = load_prices('^GSPC')
    loss = compute_losses(raw)

    sigma_ewma = ewma_volatility(loss)
    sigma_hs = loss.rolling(250).std().shift(1)

    hs_main = hs_var_es(loss, window=250)
    ewma_main = ewma_var_es(sigma_ewma)
    fhs_main = fhs_var_es(loss, sigma_ewma, window=250)
    fhs_500 = fhs_var_es(loss, sigma_ewma, window=500)

    common_main = (hs_main['var'].dropna().index
               .intersection(ewma_main['VaR'].dropna().index)
               .intersection(fhs_main['VaR'].dropna().index))

    c2 = fhs_500['VaR'].dropna().index 

    return {'loss': loss, 
            'sigma_ewma': sigma_ewma, 
            'sigma_hs': sigma_hs,
            'hs_main': hs_main, 
            'ewma_main': ewma_main,
            'fhs_main': fhs_main, 
            'fhs_500': fhs_500,
            'common_main': common_main, 
            'c2': c2}