
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data_processed"
TABLE_DIR = BASE_DIR / "results" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)


# Settings

HAC_LAGS = [3, 6, 12]

FACTOR_SETS = {
    "FF5": [
        "Mkt-RF", "SMB", "HML", "RMW", "CMA"
    ],
    "FF6": [
        "Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"
    ]
}


# Load datasets

Y = pd.read_csv(
    DATA_DIR / "size_momentum_excess.csv",
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


# Validate data

if len(Y) != 750:
    raise ValueError("Expected 750 observations.")

if len(Y.columns) != 25:
    raise ValueError("Expected 25 portfolios.")

if Y.isna().any().any():
    raise ValueError("Missing portfolio observations.")

if not Y.index.equals(FF5.index):
    raise ValueError("FF5 dates do not match.")

if not Y.index.equals(FF6.index):
    raise ValueError("FF6 dates do not match.")


# Estimate first-stage betas

def estimate_betas(Y, factors):

    betas = pd.DataFrame(
        index=Y.columns,
        columns=factors.columns,
        dtype=float
    )

    X = sm.add_constant(
        factors,
        has_constant="add"
    )

    for portfolio in Y.columns:

        model = sm.OLS(
            Y[portfolio],
            X
        ).fit()

        betas.loc[portfolio] = (
            model.params[factors.columns]
        )

    return betas


# Estimate monthly second-stage regressions

def run_fama_macbeth(Y, factors, model_name):

    print("\n" + "=" * 60)
    print("MODEL:", model_name)
    print("=" * 60)

    betas = estimate_betas(Y, factors)

    X_cross = sm.add_constant(
        betas,
        has_constant="add"
    )

    monthly_results = []
    monthly_r2 = []

    for date in Y.index:

        y_cross = Y.loc[date].reindex(
            betas.index
        )

        model = sm.OLS(
            y_cross,
            X_cross
        ).fit()

        monthly_results.append(
            model.params
        )

        monthly_r2.append({
            "Date": date,
            "R2": model.rsquared
        })

    monthly_lambdas = pd.DataFrame(
        monthly_results,
        index=Y.index
    )

    monthly_lambdas.index.name = "Date"

    monthly_r2 = pd.DataFrame(
        monthly_r2
    ).set_index("Date")

    # Save estimated betas and monthly premiums

    betas.to_csv(
        TABLE_DIR / f"size_momentum_{model_name}_betas.csv"
    )

    monthly_lambdas.to_csv(
        TABLE_DIR / f"size_momentum_{model_name}_monthly_lambdas.csv"
    )

    monthly_r2.to_csv(
        TABLE_DIR / f"size_momentum_{model_name}_monthly_r2.csv"
    )

    print("\nESTIMATED BETAS")

    print(betas.round(4).to_string())

    print("\nAVERAGE MONTHLY RISK PREMIUMS (%)")

    print(
        (monthly_lambdas.mean() * 100).round(4)
    )

    print("\nAVERAGE MONTHLY CROSS-SECTIONAL R2")

    print(round(monthly_r2["R2"].mean(), 4))

    # HAC inference on average premiums

    inference_results = []

    for lag in HAC_LAGS:

        for coefficient in monthly_lambdas.columns:

            values = monthly_lambdas[coefficient]

            model = sm.OLS(
                values.to_numpy(),
                np.ones((len(values), 1))
            ).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": lag},
                use_t=True
            )

            inference_results.append({
                "Model": model_name,
                "HAC Lags": lag,
                "Coefficient": coefficient,
                "Average Premium (%)": model.params[0] * 100,
                "HAC SE (%)": model.bse[0] * 100,
                "t-stat": model.tvalues[0],
                "p-value": model.pvalues[0]
            })

    inference = pd.DataFrame(
        inference_results
    )

    print("\nHAC RISK PREMIUM INFERENCE")

    print(
        inference.round(5).to_string(index=False)
    )

    return betas, monthly_lambdas, monthly_r2, inference


# Run FF5

FF5_RESULTS = run_fama_macbeth(
    Y,
    FF5[FACTOR_SETS["FF5"]],
    "FF5"
)


# Run FF6

FF6_RESULTS = run_fama_macbeth(
    Y,
    FF6[FACTOR_SETS["FF6"]],
    "FF6"
)


# Combine summary results

summary = pd.concat([
    FF5_RESULTS[3],
    FF6_RESULTS[3]
], ignore_index=True)


summary.to_csv(
    TABLE_DIR / "size_momentum_fama_macbeth_comparison.csv",
    index=False
)


print("\nFAMA-MACBETH ANALYSIS COMPLETED")
