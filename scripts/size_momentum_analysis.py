
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data_processed"
TABLE_DIR = BASE_DIR / "results" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

HAC_LAGS = 6


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

Y.index = pd.PeriodIndex(
    Y.index.astype(str), freq="M"
)

FF5.index = pd.PeriodIndex(
    FF5.index.astype(str), freq="M"
)

FF6.index = pd.PeriodIndex(
    FF6.index.astype(str), freq="M"
)


# Validate datasets

if not Y.index.equals(FF5.index):
    raise ValueError("Size-Momentum and FF5 dates do not match.")

if not Y.index.equals(FF6.index):
    raise ValueError("Size-Momentum and FF6 dates do not match.")

if len(Y) != 750:
    raise ValueError("Expected 750 observations.")

if len(Y.columns) != 25:
    raise ValueError("Expected 25 portfolios.")

if Y.isna().any().any():
    raise ValueError("Missing values in portfolio returns.")

if FF5.isna().any().any():
    raise ValueError("Missing values in FF5.")

if FF6.isna().any().any():
    raise ValueError("Missing values in FF6.")

FF5_COLUMNS = [
    "Mkt-RF",
    "SMB",
    "HML",
    "RMW",
    "CMA"
]

FF6_COLUMNS = [
    "Mkt-RF",
    "SMB",
    "HML",
    "RMW",
    "CMA",
    "MOM"
]

if not set(FF5_COLUMNS).issubset(FF5.columns):
    raise ValueError("Required FF5 factors are missing.")

if not set(FF6_COLUMNS).issubset(FF6.columns):
    raise ValueError("Required FF6 factors are missing.")

FF5 = FF5[FF5_COLUMNS]
FF6 = FF6[FF6_COLUMNS]


print("DATA VALIDATION")
print("Observations:", len(Y))
print("Portfolios:", len(Y.columns))
print("First month:", Y.index.min())
print("Last month:", Y.index.max())
print("HAC lags:", HAC_LAGS)


# Estimate all portfolio regressions

def estimate_model(Y, factors, model_name):

    results = []

    X = sm.add_constant(
        factors,
        has_constant="add"
    )

    for portfolio in Y.columns:

        y = Y[portfolio]

        model = sm.OLS(y, X).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": HAC_LAGS},
            use_t=True
        )

        alpha = model.params["const"]

        row = {
            "Portfolio": portfolio,
            "Alpha (%)": alpha * 100,
            "Alpha SE (%)": model.bse["const"] * 100,
            "Alpha t-stat": model.tvalues["const"],
            "Alpha p-value": model.pvalues["const"],
            "Adjusted R2": model.rsquared_adj,
            "R2": model.rsquared,
            "RMSE (%)": np.sqrt(
                np.mean(model.resid ** 2)
            ) * 100
        }

        for factor in factors.columns:

            row[factor + " Beta"] = model.params[factor]

        results.append(row)

    results = pd.DataFrame(results)

    # Benjamini-Hochberg FDR adjustment

    reject, adjusted_p, _, _ = multipletests(
        results["Alpha p-value"],
        alpha=0.05,
        method="fdr_bh"
    )

    results["FDR p-value"] = adjusted_p

    results["FDR Significant"] = reject

    # Display results

    print("\n" + "=" * 65)
    print("MODEL:", model_name)
    print("=" * 65)

    display_columns = [
        "Portfolio",
        "Alpha (%)",
        "Alpha t-stat",
        "Adjusted R2",
        "RMSE (%)"
    ]

    print(
        results[display_columns].round(4).to_string(
            index=False
        )
    )

    print("\nAverage absolute alpha (%):")
    print(results["Alpha (%)"].abs().mean())

    print("\nAverage RMS alpha (%):")
    print(
        np.sqrt(
            np.mean(results["Alpha (%)"] ** 2)
        )
    )

    print("\nAverage adjusted R2:")
    print(results["Adjusted R2"].mean())

    print("\nAverage RMSE (%):")
    print(results["RMSE (%)"].mean())

    print("\nSignificant alphas:",
          (results["Alpha p-value"] < 0.05).sum())

    print("FDR significant alphas:",
          results["FDR Significant"].sum())

    results.to_csv(
        TABLE_DIR / f"size_momentum_{model_name}_regressions.csv",
        index=False
    )

    return results


# Run FF5 and FF6

FF5_RESULTS = estimate_model(
    Y,
    FF5,
    "FF5"
)

FF6_RESULTS = estimate_model(
    Y,
    FF6,
    "FF6"
)


# Compare models portfolio by portfolio

comparison = FF5_RESULTS[
    ["Portfolio", "Alpha (%)", "Adjusted R2", "RMSE (%)"]
].merge(
    FF6_RESULTS[
        ["Portfolio", "Alpha (%)", "Adjusted R2", "RMSE (%)"]
    ],
    on="Portfolio",
    suffixes=("_FF5", "_FF6")
)

comparison["Abs Alpha Improvement (%)"] = (
    comparison["Alpha (%)_FF5"].abs()
    - comparison["Alpha (%)_FF6"].abs()
)

comparison["FF6 Lower Abs Alpha"] = (
    comparison["Abs Alpha Improvement (%)"] > 0
)

comparison["Adjusted R2 Difference"] = (
    comparison["Adjusted R2_FF6"]
    - comparison["Adjusted R2_FF5"]
)

comparison["RMSE Difference (%)"] = (
    comparison["RMSE (%)_FF5"]
    - comparison["RMSE (%)_FF6"]
)


# Overall model comparison

summary = pd.DataFrame({
    "Model": ["FF5", "FF6"],
    "Average Absolute Alpha (%)": [
        FF5_RESULTS["Alpha (%)"].abs().mean(),
        FF6_RESULTS["Alpha (%)"].abs().mean()
    ],
    "RMS Alpha (%)": [
        np.sqrt(np.mean(FF5_RESULTS["Alpha (%)"] ** 2)),
        np.sqrt(np.mean(FF6_RESULTS["Alpha (%)"] ** 2))
    ],
    "Average Adjusted R2": [
        FF5_RESULTS["Adjusted R2"].mean(),
        FF6_RESULTS["Adjusted R2"].mean()
    ],
    "Average RMSE (%)": [
        FF5_RESULTS["RMSE (%)"].mean(),
        FF6_RESULTS["RMSE (%)"].mean()
    ],
    "Significant Alphas": [
        (FF5_RESULTS["Alpha p-value"] < 0.05).sum(),
        (FF6_RESULTS["Alpha p-value"] < 0.05).sum()
    ],
    "FDR Significant Alphas": [
        FF5_RESULTS["FDR Significant"].sum(),
        FF6_RESULTS["FDR Significant"].sum()
    ]
})


comparison.to_csv(
    TABLE_DIR / "size_momentum_portfolio_comparison.csv",
    index=False
)

summary.to_csv(
    TABLE_DIR / "size_momentum_model_comparison.csv",
    index=False
)


print("\nFINAL SIZE-MOMENTUM MODEL COMPARISON")

print(summary.round(6).to_string(index=False))

print("\nPortfolio-level improvements")

print(
    "Portfolios with lower absolute alpha under FF6:",
    comparison["FF6 Lower Abs Alpha"].sum()
)

print(
    "Average absolute alpha reduction (%):",
    comparison["Abs Alpha Improvement (%)"].mean()
)

print("\nSIZE-MOMENTUM ANALYSIS COMPLETED")
