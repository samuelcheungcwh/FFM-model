
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data_processed"
TABLE_DIR = BASE_DIR / "results" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)


# Load the datasets

ff5 = pd.read_csv(
    DATA_DIR / "factors_ff5.csv",
    index_col=0
)

ff6 = pd.read_csv(
    DATA_DIR / "factors_ff6.csv",
    index_col=0
)

Y = pd.read_csv(
    DATA_DIR / "size_bm_excess.csv",
    index_col=0
)


# Convert dates to monthly periods

ff5.index = pd.PeriodIndex(
    ff5.index.astype(str),
    freq="M"
)

ff6.index = pd.PeriodIndex(
    ff6.index.astype(str),
    freq="M"
)

Y.index = pd.PeriodIndex(
    Y.index.astype(str),
    freq="M"
)


# Check the sample

if not ff5.index.equals(ff6.index):
    raise ValueError("FF5 and FF6 dates do not match.")

if not Y.index.equals(ff5.index):
    raise ValueError("Portfolio and factor dates do not match.")

if ff5.isna().any().any():
    raise ValueError("Missing values in FF5.")

if ff6.isna().any().any():
    raise ValueError("Missing values in FF6.")

if Y.isna().any().any():
    raise ValueError("Missing values in portfolios.")

if len(Y.columns) != 25:
    raise ValueError("Expected 25 test portfolios.")

if len(ff5) != 750:
    raise ValueError("Unexpected number of observations.")


# Estimate all portfolio regressions

def estimate_model(Y, X, model_name):

    results = []

    for portfolio in Y.columns:

        y = Y[portfolio]

        X_constant = sm.add_constant(
            X,
            has_constant="add"
        )

        model = sm.OLS(
            y,
            X_constant
        ).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 6},
            use_t=True
        )

        result = {}

        result["Portfolio"] = portfolio

        result["Alpha (%)"] = model.params["const"] * 100

        result["Alpha t-stat"] = model.tvalues["const"]

        result["Alpha p-value"] = model.pvalues["const"]

        result["Adjusted R2"] = model.rsquared_adj

        result["RMSE (%)"] = np.sqrt(
            np.mean(model.resid ** 2)
        ) * 100

        for factor in X.columns:

            result["Beta_" + factor] = model.params[factor]

            result["t_" + factor] = model.tvalues[factor]

            result["p_" + factor] = model.pvalues[factor]

        results.append(result)

    results = pd.DataFrame(results)

    # Correct the alpha tests for multiple comparisons

    reject, adjusted_p, _, _ = multipletests(
        results["Alpha p-value"],
        method="fdr_bh"
    )

    results["Alpha_FDR_p"] = adjusted_p

    results["Alpha_FDR_significant"] = reject

    if "MOM" in X.columns:

        reject, adjusted_p, _, _ = multipletests(
            results["p_MOM"],
            method="fdr_bh"
        )

        results["MOM_FDR_p"] = adjusted_p

        results["MOM_FDR_significant"] = reject

    output_file = (
        TABLE_DIR / (model_name + "_time_series_regressions.csv")
    )

    results.to_csv(output_file, index=False)

    print("\n", model_name)

    print(results[
        ["Portfolio", "Alpha (%)", "Alpha t-stat",
         "Adjusted R2", "RMSE (%)"]
    ].round(4))

    print("\nAverage absolute alpha:")

    print(results["Alpha (%)"].abs().mean())

    print("\nAverage adjusted R2:")

    print(results["Adjusted R2"].mean())

    print("\nNumber of significant alphas:")

    print((results["Alpha p-value"] < 0.05).sum())

    print("\nNumber of FDR-adjusted significant alphas:")

    print(results["Alpha_FDR_significant"].sum())

    return results


# Define the factor sets

factor_5 = [
    "Mkt-RF",
    "SMB",
    "HML",
    "RMW",
    "CMA"
]

factor_6 = factor_5 + ["MOM"]


# Run both models

results_ff5 = estimate_model(
    Y,
    ff5[factor_5],
    "FF5"
)

results_ff6 = estimate_model(
    Y,
    ff6[factor_6],
    "FF6"
)


# Compare the two models

comparison = pd.DataFrame({

    "Model": ["FF5", "FF6"],

    "Mean Absolute Alpha (%)": [
        results_ff5["Alpha (%)"].abs().mean(),
        results_ff6["Alpha (%)"].abs().mean()
    ],

    "RMS Alpha (%)": [
        np.sqrt(np.mean(results_ff5["Alpha (%)"] ** 2)),
        np.sqrt(np.mean(results_ff6["Alpha (%)"] ** 2))
    ],

    "Average Adjusted R2": [
        results_ff5["Adjusted R2"].mean(),
        results_ff6["Adjusted R2"].mean()
    ],

    "Average RMSE (%)": [
        results_ff5["RMSE (%)"].mean(),
        results_ff6["RMSE (%)"].mean()
    ],

    "Significant Alphas": [
        (results_ff5["Alpha p-value"] < 0.05).sum(),
        (results_ff6["Alpha p-value"] < 0.05).sum()
    ],

    "FDR Significant Alphas": [
        results_ff5["Alpha_FDR_significant"].sum(),
        results_ff6["Alpha_FDR_significant"].sum()
    ]

})


comparison.to_csv(
    TABLE_DIR / "ff5_ff6_model_comparison.csv",
    index=False
)


print("\nFINAL MODEL COMPARISON")

print(comparison.round(4))

print("\nREGRESSION ANALYSIS COMPLETED")
