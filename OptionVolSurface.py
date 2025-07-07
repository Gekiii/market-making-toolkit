import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ----- FUNZIONI SVI ---------------------------------------------------
def svi_total_variance(k, a, b, rho, m, sigma):
    """Formula SVI: total variance w(k)."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

def objective(params, k, w):
    return np.mean((svi_total_variance(k, *params) - w) ** 2)

# ----- FIT ------------------------------------------------------------
def fit_svi(df: pd.DataFrame):
    """
    df colonne: log_moneyness, iv, ttm.
    Ritorna parametri (a, b, rho, m, sigma).
    """
    k = df["log_moneyness"].values
    w = (df["iv"] ** 2) * df["ttm"]
    x0     = [0.1, 0.1, 0.0, 0.0, 0.1]
    bounds = [(-1, 1), (0, 10), (-0.999, 0.999), (-2, 2), (1e-4, 1)]
    res = minimize(objective, x0, args=(k, w), bounds=bounds, method="L-BFGS-B")
    return res.x

# ----- PLOT -----------------------------------------------------------
def plot_slice(df: pd.DataFrame, params):
    k_grid = np.linspace(df["log_moneyness"].min(), df["log_moneyness"].max(), 100)
    w_fit  = svi_total_variance(k_grid, *params)
    iv_fit = np.sqrt(w_fit / df["ttm"].iloc[0])

    plt.scatter(df["log_moneyness"], df["iv"], label="Market IV", marker='x')
    plt.plot(k_grid, iv_fit, label="SVI fit")
    plt.xlabel("log-moneyness")
    plt.ylabel("Implied Vol")
    plt.title("SVI surface slice")
    plt.legend()
    plt.show()

# ----- CLI ------------------------------------------------------------
if __name__ == "__main__":
    import argparse, pathlib
    p = argparse.ArgumentParser(description="SVI surface fitter")
    p.add_argument("csv", help="CSV con colonne: strike, iv, ttm, log_moneyness")
    args = p.parse_args()

    data = pd.read_csv(args.csv)
    params = fit_svi(data)
    print("SVI params:", params)
    plot_slice(data, params)
