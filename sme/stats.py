"""
Module for generating statistical output from parameter estimates.
Produces a table with parameter estimates, standard errors, and 95% confidence intervals.
"""
import pandas as pd
import numpy as np

def generate_stats_table(params: np.ndarray, se: np.ndarray, alpha: float = 0.05) -> pd.DataFrame:
    """
    Generate an easy-to-read statistical table.
    
    Args:
        params: Estimated parameters as a NumPy array.
        se: Standard errors as a NumPy array.
        alpha: Significance level for CIs (default 0.05 for 95% CIs).
    
    Returns:
        A DataFrame with columns: Parameter, Std_Error, Lower_CI, Upper_CI.
    """
    z_value = 1.96  # for 95% confidence interval, can generalize via scipy.stats.norm.ppf(1 - alpha/2)
    lower_ci = params - z_value * se
    upper_ci = params + z_value * se
    data = {
        'Parameter': params,
        'Std_Error': se,
        'Lower_CI': lower_ci,
        'Upper_CI': upper_ci
    }
    return pd.DataFrame(data)