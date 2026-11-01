
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import f


# ==================================================
# 1. PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data_processed"

TABLE_DIR = BASE_DIR / "results" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD DATA
# ==================================================

def load_returns(filename):

    data = pd.read_csv(
        DATA_DIR / filename,
        index_col=0
    )

    data.index = pd.PeriodIndex(
        data.index.astype(str),
        freq="M"
    )

    data = data.apply(pd.to_numeric, errors="coerce")

    return data


def load_factors(filename):

    data = pd.read_csv(
        DATA_DIR / filename,
        index_col=0
    )

    data.index = pd.PeriodIndex(
        data.index.astype(str),
        freq="M"
    )

    data.columns = data.columns.str.strip()

    data = data.apply(pd.to_numeric, errors="coerce")

    return data


# ==================================================
# 3. GRS TEST FUNCTION
# ==================================================

def run_grs(returns, factors, sample_name, model_name):

    Y = returns.to_numpy()

    F = factors.to_numpy()

    T, N = Y.shape

    K = F.shape[1]

    print("\nSample:", sample_name)

    print("Model:", model_name)

    print("Observations:", T)

    print("Industries:", N)

    print("Factors:", K)


    # Add intercept to factor matrix

    X = np.column_stack([
        np.ones(T),
        F
    ])


    # Estimate all industry regressions simultaneously

    coefficients = np.linalg.lstsq(
        X,
        Y,
        rcond=None
    )[0]


    # Extract alphas

    alphas = coefficients[0, :]


    # Calculate residuals

    residuals = Y - X @ coefficients


    # Residual covariance matrix, divided by T

    sigma_e = residuals.T @ residuals / T


    # Factor mean and covariance matrix

    factor_mean = F.mean(axis=0)

    centered_factors = F - factor_mean

    sigma_f = (
        centered_factors.T @ centered_factors / T
    )


    # Calculate GRS components

    alpha_component = alphas @ np.linalg.solve(
        sigma_e,
        alphas
    )

    factor_component = factor_mean @ np.linalg.solve(
        sigma_f,
        factor_mean
    )


    # GRS statistic

    grs = (
        (T - N - K) / N
    ) * (
        alpha_component / (1 + factor_component)
    )


    # Degrees of freedom

    df1 = N

    df2 = T - N - K


    # P-value

    p_value = f.sf(
        grs,
        df1,
        df2
    )


    # Additional pricing error measures

    average_absolute_alpha = np.mean(
        np.abs(alphas)
    ) * 100

    rms_alpha = np.sqrt(
        np.mean(alphas ** 2)
    ) * 100


    print("GRS statistic:", round(grs, 6))

    print("Numerator degrees of freedom:", df1)

    print("Denominator degrees of freedom:", df2)

    print("P-value:", "{:.10g}".format(p_value))

    print(
        "Average absolute alpha (%):",
        round(average_absolute_alpha, 6)
    )

    print(
        "RMS alpha (%):",
        round(rms_alpha, 6)
    )


    if p_value < 0.05:

        print("Conclusion: Reject joint zero-alpha hypothesis.")

    else:

        print("Conclusion: Do not reject joint zero-alpha hypothesis.")


    return {

        "Sample": sample_name,

        "Model": model_name,

        "Observations": T,

        "Industries": N,

        "Factors": K,

        "GRS": grs,

        "DF1": df1,

        "DF2": df2,

        "P_value": p_value,

        "Average_absolute_alpha_pct": average_absolute_alpha,

        "RMS_alpha_pct": rms_alpha

    }


# ==================================================
# 4. SAMPLE AND MODEL SPECIFICATIONS
# ==================================================

samples = {

    "49_industry_main": "industry_49_excess.csv",

    "47_industry_extended": "industry_47_excess.csv"

}


models = {

    "FF5": "factors_ff5.csv",

    "FF6": "factors_ff6.csv"

}


# ==================================================
# 5. RUN ALL FOUR TESTS
# ==================================================

results = []


for sample_name, returns_file in samples.items():

    returns = load_returns(returns_file)

    for model_name, factor_file in models.items():

        factors = load_factors(factor_file)

        # Align observations

        common_dates = returns.index.intersection(
            factors.index
        )

        Y = returns.loc[common_dates]

        F = factors.loc[common_dates]

        if len(common_dates) != len(returns):

            raise ValueError(
                "Dates do not match between returns and factors."
            )

        if Y.isna().any().any() or F.isna().any().any():

            raise ValueError(
                "Missing observations detected."
            )


        result = run_grs(
            Y,
            F,
            sample_name,
            model_name
        )

        results.append(result)


# ==================================================
# 6. SAVE RESULTS
# ==================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    TABLE_DIR / "industry_grs_results.csv",
    index=False
)


print("\n" + "=" * 65)

print("INDUSTRY GRS TEST SUMMARY")

print("=" * 65)

print(
    results_df.to_string(index=False)
)

print("\nResults saved: industry_grs_results.csv")

print("\nGRS ANALYSIS COMPLETED")
