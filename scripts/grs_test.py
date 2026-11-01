
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import f


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data_processed"
TABLE_DIR = BASE_DIR / "results" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)


# Load datasets

Y = pd.read_csv(
    DATA_DIR / "size_bm_excess.csv",
    index_col=0
)

FF5 = pd.read_csv(
    DATA_DIR / "factors_ff5.csv",
    index_col=0
)

FF6 = pd.read_csv(
    DATA_DIR / "factors_ff6.csv",
    index_col=0
)


# Convert dates

Y.index = pd.PeriodIndex(Y.index.astype(str), freq="M")
FF5.index = pd.PeriodIndex(FF5.index.astype(str), freq="M")
FF6.index = pd.PeriodIndex(FF6.index.astype(str), freq="M")


def grs_test(Y, X, model_name):

    data = Y.join(X, how="inner").dropna()

    returns = data[Y.columns].to_numpy()

    factors = data[X.columns].to_numpy()

    T = returns.shape[0]
    N = returns.shape[1]
    K = factors.shape[1]

    if T <= N + K:
        raise ValueError("Insufficient observations.")

    # Include an intercept

    design = np.column_stack([
        np.ones(T),
        factors
    ])

    # Estimate all portfolio regressions together

    coefficients = np.linalg.lstsq(
        design,
        returns,
        rcond=None
    )[0]

    alpha = coefficients[0, :]

    residuals = returns - design @ coefficients

    # Estimate residual covariance

    sigma_e = (
        residuals.T @ residuals
    ) / (T - K - 1)

    # Estimate factor covariance

    sigma_f = np.cov(
        factors,
        rowvar=False,
        ddof=1
    )

    if np.linalg.matrix_rank(sigma_e) < N:
        raise ValueError("Residual covariance matrix is singular.")

    if np.linalg.matrix_rank(sigma_f) < K:
        raise ValueError("Factor covariance matrix is singular.")

    # Calculate the GRS components

    alpha_component = alpha @ np.linalg.solve(
        sigma_e,
        alpha
    )

    mean_factors = factors.mean(axis=0)

    factor_component = mean_factors @ np.linalg.solve(
        sigma_f,
        mean_factors
    )

    # Calculate GRS statistic

    statistic = (
        (T / N)
        * ((T - N - K) / (T - K - 1))
        * alpha_component
        / (1 + factor_component)
    )

    p_value = f.sf(
        statistic,
        N,
        T - N - K
    )

    result = {
        "Model": model_name,
        "Observations": T,
        "Portfolios": N,
        "Factors": K,
        "GRS Statistic": statistic,
        "Numerator df": N,
        "Denominator df": T - N - K,
        "P-value": p_value,
        "Reject at 5%": p_value < 0.05,
        "Average Absolute Alpha (%)":
            np.abs(alpha).mean() * 100,
        "RMS Alpha (%)":
            np.sqrt(np.mean(alpha ** 2)) * 100
    }

    print("\n", model_name)

    for name, value in result.items():
        print(name, ":", value)

    return result


# Run both models

results = []

results.append(
    grs_test(Y, FF5, "FF5")
)

results.append(
    grs_test(Y, FF6, "FF6")
)


# Save results

results = pd.DataFrame(results)

results.to_csv(
    TABLE_DIR / "grs_test_results.csv",
    index=False
)

print("\nFINAL GRS COMPARISON")

print(results.to_string(index=False))

print("\nGRS TESTING COMPLETED")
