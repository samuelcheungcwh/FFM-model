
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.stats.multitest import multipletests


# ==================================================
# 1. PATHS AND SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data_processed"
TABLE_DIR = BASE_DIR / "results" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

HAC_LAGS = 6


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

    data.columns = data.columns.str.strip()

    data = data.sort_index()

    if data.index.has_duplicates:
        raise ValueError("Duplicate dates in " + filename)

    data = data.apply(pd.to_numeric, errors="coerce")

    if data.isna().any().any():
        raise ValueError("Missing returns in " + filename)

    return data


def load_factors(filename, required_factors):

    data = pd.read_csv(
        DATA_DIR / filename,
        index_col=0
    )

    data.index = pd.PeriodIndex(
        data.index.astype(str),
        freq="M"
    )

    data.columns = data.columns.str.strip()

    data = data.sort_index()

    if data.index.has_duplicates:
        raise ValueError("Duplicate dates in " + filename)

    # Match factor names without depending on capitalization.

    factor_lookup = {}

    for column in data.columns:
        factor_lookup[column.upper()] = column

    selected_columns = []

    for factor in required_factors:

        key = factor.upper()

        if key not in factor_lookup:
            raise ValueError(
                factor + " not found in " + filename
            )

        selected_columns.append(
            factor_lookup[key]
        )

    data = data[selected_columns].copy()

    data.columns = required_factors

    data = data.apply(pd.to_numeric, errors="coerce")

    if data.isna().any().any():
        raise ValueError("Missing factor observations in " + filename)

    return data


# ==================================================
# 3. MODEL SPECIFICATIONS
# ==================================================

FF5_FACTORS = [
    "Mkt-RF",
    "SMB",
    "HML",
    "RMW",
    "CMA"
]

FF6_FACTORS = [
    "Mkt-RF",
    "SMB",
    "HML",
    "RMW",
    "CMA",
    "MOM"
]


MODEL_FILES = {
    "FF5": {
        "file": "factors_ff5.csv",
        "factors": FF5_FACTORS
    },

    "FF6": {
        "file": "factors_ff6.csv",
        "factors": FF6_FACTORS
    }
}


SAMPLES = {
    "49_industry_main": "industry_49_excess.csv",
    "47_industry_extended": "industry_47_excess.csv"
}


# ==================================================
# 4. ESTIMATE TIME-SERIES REGRESSIONS
# ==================================================

portfolio_results = []

coefficient_results = []


for sample_name, returns_file in SAMPLES.items():

    print("\n" + "=" * 60)
    print("SAMPLE:", sample_name)
    print("=" * 60)

    returns = load_returns(returns_file)

    print("Observations:", len(returns))
    print("Industries:", len(returns.columns))
    print("First month:", returns.index.min())
    print("Last month:", returns.index.max())


    for model_name, specification in MODEL_FILES.items():

        factors = load_factors(
            specification["file"],
            specification["factors"]
        )

        # Align returns and factors by month.

        common_dates = returns.index.intersection(
            factors.index
        )

        y_data = returns.loc[common_dates]
        x_data = factors.loc[common_dates]

        if len(common_dates) != len(returns):
            raise ValueError(
                "Factor and return dates do not match for "
                + sample_name
                + " "
                + model_name
            )

        if y_data.isna().any().any():
            raise ValueError("Missing portfolio returns.")

        if x_data.isna().any().any():
            raise ValueError("Missing factor observations.")


        print("\nEstimating", model_name)

        alphas = []

        for portfolio in y_data.columns:

            y = y_data[portfolio]

            X = sm.add_constant(x_data)

            # OLS with Newey-West HAC standard errors.

            result = sm.OLS(y, X).fit(
                cov_type="HAC",
                cov_kwds={
                    "maxlags": HAC_LAGS
                }
            )

            alpha = result.params["const"]

            alpha_se = result.bse["const"]

            alpha_t = result.tvalues["const"]

            alpha_p = result.pvalues["const"]

            rmse = np.sqrt(
                np.mean(result.resid ** 2)
            )

            alphas.append(alpha)


            # Portfolio-level results

            portfolio_results.append({

                "Sample": sample_name,
                "Model": model_name,
                "Portfolio": portfolio,

                "Observations": int(result.nobs),

                "Alpha": alpha,
                "Alpha_pct": alpha * 100,

                "Alpha_HAC_SE": alpha_se,
                "Alpha_t": alpha_t,
                "Alpha_p": alpha_p,

                "R_squared": result.rsquared,
                "Adjusted_R_squared": result.rsquared_adj,

                "RMSE": rmse,
                "RMSE_pct": rmse * 100

            })


            # Save every estimated coefficient.

            for factor in result.params.index:

                coefficient_results.append({

                    "Sample": sample_name,
                    "Model": model_name,
                    "Portfolio": portfolio,

                    "Factor": factor,

                    "Coefficient": result.params[factor],
                    "HAC_SE": result.bse[factor],
                    "HAC_t": result.tvalues[factor],
                    "HAC_p": result.pvalues[factor]

                })


        # ==================================================
        # 5. MODEL-LEVEL SUMMARY
        # ==================================================

        alphas = np.array(alphas)

        print("Regressions completed:", len(alphas))

        print(
            "Average absolute alpha (%):",
            round(np.mean(np.abs(alphas)) * 100, 6)
        )

        print(
            "RMS alpha (%):",
            round(np.sqrt(np.mean(alphas ** 2)) * 100, 6)
        )


# ==================================================
# 6. ADJUST ALPHA P-VALUES FOR MULTIPLE TESTING
# ==================================================

portfolio_df = pd.DataFrame(portfolio_results)

portfolio_df["Alpha_FDR_p"] = np.nan

portfolio_df["Alpha_FDR_significant"] = False


for (sample, model), group in portfolio_df.groupby(
    ["Sample", "Model"]
):

    rejected, adjusted_p, _, _ = multipletests(
        group["Alpha_p"],
        alpha=0.05,
        method="fdr_bh"
    )

    portfolio_df.loc[
        group.index,
        "Alpha_FDR_p"
    ] = adjusted_p

    portfolio_df.loc[
        group.index,
        "Alpha_FDR_significant"
    ] = rejected


portfolio_df["Alpha_significant"] = (
    portfolio_df["Alpha_p"] < 0.05
)


# ==================================================
# 7. CREATE MODEL SUMMARY TABLE
# ==================================================

summary_rows = []

for (sample, model), group in portfolio_df.groupby(
    ["Sample", "Model"]
):

    summary_rows.append({

        "Sample": sample,
        "Model": model,

        "Industries": len(group),
        "Observations": int(group["Observations"].min()),

        "Average_alpha_pct": group["Alpha_pct"].mean(),

        "Average_absolute_alpha_pct": (
            group["Alpha_pct"].abs().mean()
        ),

        "RMS_alpha_pct": np.sqrt(
            np.mean(group["Alpha_pct"] ** 2)
        ),

        "Average_R_squared": group["R_squared"].mean(),

        "Average_adjusted_R_squared": (
            group["Adjusted_R_squared"].mean()
        ),

        "Average_RMSE_pct": group["RMSE_pct"].mean(),

        "Significant_alphas": int(
            group["Alpha_significant"].sum()
        ),

        "FDR_significant_alphas": int(
            group["Alpha_FDR_significant"].sum()
        )

    })


summary_df = pd.DataFrame(summary_rows)


# ==================================================
# 8. SAVE RESULTS
# ==================================================

portfolio_df.to_csv(
    TABLE_DIR / "industry_time_series_portfolio_results.csv",
    index=False
)

coefficient_df = pd.DataFrame(coefficient_results)

coefficient_df.to_csv(
    TABLE_DIR / "industry_time_series_coefficients.csv",
    index=False
)

summary_df.to_csv(
    TABLE_DIR / "industry_time_series_summary.csv",
    index=False
)


# ==================================================
# 9. DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 70)
print("INDUSTRY FF5 VS FF6 RESULTS")
print("=" * 70)

display_columns = [
    "Sample",
    "Model",
    "Industries",
    "Observations",
    "Average_absolute_alpha_pct",
    "RMS_alpha_pct",
    "Average_adjusted_R_squared",
    "Average_RMSE_pct",
    "Significant_alphas",
    "FDR_significant_alphas"
]

print(
    summary_df[display_columns].to_string(
        index=False
    )
)


print("\nRESULTS SAVED")

print("industry_time_series_portfolio_results.csv")
print("industry_time_series_coefficients.csv")
print("industry_time_series_summary.csv")

print("\nINDUSTRY ANALYSIS COMPLETED")
