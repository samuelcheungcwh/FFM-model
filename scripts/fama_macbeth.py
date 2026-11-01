
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm


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


def fama_macbeth(Y, X, model_name):

    # Check dates and observations

    if not Y.index.equals(X.index):
        raise ValueError("Dates do not match.")

    if Y.isna().any().any() or X.isna().any().any():
        raise ValueError("Missing observations detected.")

    N = len(Y.columns)
    K = len(X.columns)

    # First pass: estimate portfolio betas

    beta_results = []

    for portfolio in Y.columns:

        regression = sm.OLS(
            Y[portfolio],
            sm.add_constant(X)
        ).fit()

        beta_results.append(
            regression.params[X.columns].values
        )

    betas = pd.DataFrame(
        beta_results,
        index=Y.columns,
        columns=X.columns
    )

    # Check cross-sectional identification

    design = sm.add_constant(
        betas,
        has_constant="add"
    )

    if np.linalg.matrix_rank(design.values) < K + 1:
        raise ValueError(
            "The second-pass beta matrix is not full rank."
        )

    # Second pass: monthly cross-sectional regressions

    monthly_premiums = []
    monthly_r2 = []

    for date in Y.index:

        returns = Y.loc[date].reindex(betas.index)

        regression = sm.OLS(
            returns,
            design
        ).fit()

        monthly_premiums.append(
            regression.params
        )

        monthly_r2.append(
            regression.rsquared
        )

    premiums = pd.DataFrame(
        monthly_premiums,
        index=Y.index
    )

    # Save monthly estimated premiums

    premiums.to_csv(
        TABLE_DIR / (
            model_name + "_monthly_risk_premiums.csv"
        )
    )

    # Calculate HAC inference for average premiums

    results = []

    for lag in [3, 6, 12]:

        for coefficient in premiums.columns:

            series = premiums[coefficient]

            mean_model = sm.OLS(
                series.values,
                np.ones((len(series), 1))
            ).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": lag},
                use_t=True
            )

            results.append({

                "Model": model_name,

                "HAC Lags": lag,

                "Coefficient": coefficient,

                "Average Premium (%)":
                    mean_model.params[0] * 100,

                "HAC SE (%)":
                    mean_model.bse[0] * 100,

                "t-stat":
                    mean_model.tvalues[0],

                "p-value":
                    mean_model.pvalues[0]

            })

    results = pd.DataFrame(results)

    results.to_csv(
        TABLE_DIR / (
            model_name + "_risk_premium_results.csv"
        ),
        index=False
    )

    print("\nMODEL:", model_name)

    print("\nESTIMATED BETAS")

    print(betas.round(4))

    print("\nAVERAGE MONTHLY RISK PREMIUMS")

    print(
        premiums.mean().mul(100).round(4)
    )

    print("\nAVERAGE CROSS-SECTIONAL R2")

    print(round(np.mean(monthly_r2), 4))

    print("\nHAC RISK PREMIUM INFERENCE")

    print(
        results.round(5).to_string(index=False)
    )

    return betas, premiums, results


# Run FF5 and FF6

betas_ff5, premiums_ff5, results_ff5 = fama_macbeth(
    Y,
    FF5,
    "FF5"
)

betas_ff6, premiums_ff6, results_ff6 = fama_macbeth(
    Y,
    FF6,
    "FF6"
)


# Save combined summary

combined = pd.concat(
    [results_ff5, results_ff6],
    ignore_index=True
)

combined.to_csv(
    TABLE_DIR / "fama_macbeth_comparison.csv",
    index=False
)

print("\nFAMA-MACBETH ANALYSIS COMPLETED")
