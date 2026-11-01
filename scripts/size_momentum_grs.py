
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import f


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data_processed"
TABLE_DIR = BASE_DIR / "results" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)


# Load factor datasets

FF5 = pd.read_csv(
    DATA_DIR / "factors_ff5.csv",
    index_col=0
)

FF6 = pd.read_csv(
    DATA_DIR / "factors_ff6.csv",
    index_col=0
)


# Convert dates

FF5.index = pd.PeriodIndex(
    FF5.index.astype(str),
    freq="M"
)

FF6.index = pd.PeriodIndex(
    FF6.index.astype(str),
    freq="M"
)


FF5 = FF5[
    ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
]

FF6 = FF6[
    ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
]


# Load portfolio datasets

portfolio_files = {
    "Size-BM": "size_bm_excess.csv",
    "Size-Momentum": "size_momentum_excess.csv"
}


def grs_test(Y, X, dataset_name, model_name):

    if not Y.index.equals(X.index):
        raise ValueError("Portfolio and factor dates do not match.")

    if Y.isna().any().any() or X.isna().any().any():
        raise ValueError("Missing observations detected.")

    returns = Y.to_numpy()

    factors = X.to_numpy()

    T = returns.shape[0]
    N = returns.shape[1]
    K = factors.shape[1]

    if T <= N + K:
        raise ValueError("Insufficient observations.")

    # Joint OLS estimation

    design = np.column_stack([
        np.ones(T),
        factors
    ])

    coefficients = np.linalg.lstsq(
        design,
        returns,
        rcond=None
    )[0]

    residuals = returns - design @ coefficients

    alphas = coefficients[0, :]

    # Residual covariance matrix

    residual_cov = (
        residuals.T @ residuals
    ) / T

    # Factor covariance matrix

    factor_mean = factors.mean(axis=0)

    centered_factors = factors - factor_mean

    factor_cov = (
        centered_factors.T @ centered_factors
    ) / T

    # Check invertibility

    if np.linalg.matrix_rank(residual_cov) != N:
        raise ValueError("Residual covariance matrix is singular.")

    if np.linalg.matrix_rank(factor_cov) != K:
        raise ValueError("Factor covariance matrix is singular.")

    # Quadratic forms

    alpha_component = alphas @ np.linalg.solve(
        residual_cov,
        alphas
    )

    factor_component = factor_mean @ np.linalg.solve(
        factor_cov,
        factor_mean
    )

    # Classical GRS statistic

    statistic = (
        (T - N - K) / N
    ) * (
        alpha_component /
        (1 + factor_component)
    )

    numerator_df = N
    denominator_df = T - N - K

    p_value = f.sf(
        statistic,
        numerator_df,
        denominator_df
    )

    # Pricing error measures

    average_abs_alpha = np.mean(
        np.abs(alphas)
    ) * 100

    rms_alpha = np.sqrt(
        np.mean(alphas ** 2)
    ) * 100

    return {
        "Dataset": dataset_name,
        "Model": model_name,
        "Observations": T,
        "Portfolios": N,
        "Factors": K,
        "GRS Statistic": statistic,
        "Numerator df": numerator_df,
        "Denominator df": denominator_df,
        "P-value": p_value,
        "Reject at 5%": p_value < 0.05,
        "Average Absolute Alpha (%)": average_abs_alpha,
        "RMS Alpha (%)": rms_alpha
    }


# Run all four tests

results = []

for dataset_name, filename in portfolio_files.items():

    Y = pd.read_csv(
        DATA_DIR / filename,
        index_col=0
    )

    Y.index = pd.PeriodIndex(
        Y.index.astype(str),
        freq="M"
    )

    if len(Y) != 750:
        raise ValueError(
            f"Unexpected sample size for {dataset_name}."
        )

    if len(Y.columns) != 25:
        raise ValueError(
            f"Expected 25 portfolios for {dataset_name}."
        )

    if not Y.index.equals(FF5.index):
        raise ValueError(
            f"FF5 dates do not match {dataset_name}."
        )

    if not Y.index.equals(FF6.index):
        raise ValueError(
            f"FF6 dates do not match {dataset_name}."
        )

    print("\n" + "=" * 60)
    print("DATASET:", dataset_name)
    print("=" * 60)

    result_ff5 = grs_test(
        Y,
        FF5,
        dataset_name,
        "FF5"
    )

    result_ff6 = grs_test(
        Y,
        FF6,
        dataset_name,
        "FF6"
    )

    results.append(result_ff5)
    results.append(result_ff6)

    for result in [result_ff5, result_ff6]:

        print("\nMODEL:", result["Model"])

        print("Observations:", result["Observations"])
        print("Portfolios:", result["Portfolios"])
        print("Factors:", result["Factors"])

        print(
            "GRS Statistic:",
            round(result["GRS Statistic"], 6)
        )

        print(
            "Degrees of freedom:",
            result["Numerator df"],
            result["Denominator df"]
        )

        print(
            "P-value:",
            f"{result['P-value']:.10g}"
        )

        print(
            "Reject at 5%:",
            result["Reject at 5%"]
        )

        print(
            "Average Absolute Alpha (%):",
            round(result["Average Absolute Alpha (%)"], 6)
        )

        print(
            "RMS Alpha (%):",
            round(result["RMS Alpha (%)"], 6)
        )


# Save results

results = pd.DataFrame(results)

results.to_csv(
    TABLE_DIR / "grs_size_momentum_comparison.csv",
    index=False
)


print("\nFINAL GRS COMPARISON")

print(
    results.round(6).to_string(index=False)
)

print("\nGRS ANALYSIS COMPLETED")
