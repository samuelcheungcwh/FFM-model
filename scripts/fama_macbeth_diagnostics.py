
from pathlib import Path
import numpy as np
import pandas as pd
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


if not Y.index.equals(FF5.index):
    raise ValueError("FF5 dates do not match portfolio dates.")

if not Y.index.equals(FF6.index):
    raise ValueError("FF6 dates do not match portfolio dates.")


# Estimate first-stage betas

def estimate_betas(Y, factors):

    betas = pd.DataFrame(
        index=Y.columns,
        columns=factors.columns,
        dtype=float
    )

    X = sm.add_constant(factors, has_constant="add")

    for portfolio in Y.columns:

        model = sm.OLS(Y[portfolio], X).fit()

        betas.loc[portfolio] = model.params[factors.columns]

    return betas


# Analyse each model

def diagnose_model(Y, factors, model_name):

    print("\n" + "=" * 60)
    print("MODEL:", model_name)
    print("=" * 60)

    betas = estimate_betas(Y, factors)

    # Beta dispersion

    summary = pd.DataFrame({
        "Mean": betas.mean(),
        "Std": betas.std(),
        "Minimum": betas.min(),
        "Maximum": betas.max(),
        "Range": betas.max() - betas.min()
    })

    print("\nBETA DISPERSION")
    print(summary.round(5))

    summary.to_csv(
        TABLE_DIR / f"{model_name}_beta_diagnostics.csv"
    )

    # Beta correlations

    correlations = betas.corr()

    print("\nBETA CORRELATIONS")
    print(correlations.round(4))

    # Condition numbers

    design = sm.add_constant(betas, has_constant="add")

    raw_condition = np.linalg.cond(design.to_numpy())

    correlation_condition = np.linalg.cond(
        correlations.to_numpy()
    )

    print("\nDESIGN MATRIX DIAGNOSTICS")
    print("Matrix rank:", np.linalg.matrix_rank(design))
    print("Raw condition number:", round(raw_condition, 4))
    print("Beta correlation condition number:",
          round(correlation_condition, 4))

    # MOM beta explained by other betas

    if "MOM" in betas.columns:

        other_betas = betas.drop(columns=["MOM"])

        spanning_model = sm.OLS(
            betas["MOM"],
            sm.add_constant(other_betas)
        ).fit()

        print("\nMOM BETA IDENTIFICATION")
        print("MOM beta standard deviation:",
              round(betas["MOM"].std(), 6))

        print("MOM beta range:",
              round(betas["MOM"].max() - betas["MOM"].min(), 6))

        print("R2 from other betas:",
              round(spanning_model.rsquared, 4))

        print("Unique MOM beta variation:",
              round(1 - spanning_model.rsquared, 4))

    # Average portfolio returns

    average_returns = Y.mean(axis=0)

    average_returns = average_returns.reindex(betas.index)

    cross_section = sm.OLS(
        average_returns,
        design
    ).fit()

    print("\nAVERAGE-RETURN CROSS-SECTION")
    print(cross_section.params.round(6))

    # Verify Fama-MacBeth coefficient identity

    print("\nAVERAGE CROSS-SECTIONAL R2")
    print(round(cross_section.rsquared, 6))

    # Leave-one-portfolio-out sensitivity

    if "MOM" in betas.columns:

        loo_results = []

        for portfolio in betas.index:

            reduced_betas = betas.drop(index=portfolio)

            reduced_returns = average_returns.drop(index=portfolio)

            reduced_design = sm.add_constant(
                reduced_betas,
                has_constant="add"
            )

            loo_model = sm.OLS(
                reduced_returns,
                reduced_design
            ).fit()

            loo_results.append({
                "Excluded Portfolio": portfolio,
                "MOM Premium (%)": loo_model.params["MOM"] * 100
            })

        loo_results = pd.DataFrame(loo_results)

        loo_results.to_csv(
            TABLE_DIR / f"{model_name}_MOM_leave_one_out.csv",
            index=False
        )

        print("\nMOM LEAVE-ONE-PORTFOLIO-OUT SENSITIVITY")
        print("Minimum MOM premium (%):",
              round(loo_results["MOM Premium (%)"].min(), 4))

        print("Maximum MOM premium (%):",
              round(loo_results["MOM Premium (%)"].max(), 4))

        print("\nMost influential exclusions")

        print(
            loo_results.assign(
                Sensitivity=abs(
                    loo_results["MOM Premium (%)"]
                    - loo_results["MOM Premium (%)"].median()
                )
            ).sort_values(
                "Sensitivity", ascending=False
            ).head(5).round(4).to_string(index=False)
        )


diagnose_model(Y, FF5, "FF5")

diagnose_model(Y, FF6, "FF6")

print("\nFAMA-MACBETH DIAGNOSTICS COMPLETED")
