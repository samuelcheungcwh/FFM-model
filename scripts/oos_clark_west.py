
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import norm


# ==================================================
# 1. SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TABLE_DIR = BASE_DIR / "results" / "tables"

INPUT_FILE = TABLE_DIR / "oos_monthly_forecasts.csv"

HAC_LAGS = [6, 12, 24]


# ==================================================
# 2. LOAD MONTHLY FORECASTS
# ==================================================

data = pd.read_csv(INPUT_FILE)

required_columns = [
    "Dataset",
    "Date",
    "Portfolio",
    "Actual_excess_return",
    "FF5_forecast",
    "FF6_forecast"
]

for column in required_columns:

    if column not in data.columns:

        raise ValueError(
            "Missing column: " + column
        )


# Convert dates and numerical columns.

data["Date"] = pd.PeriodIndex(
    data["Date"].astype(str),
    freq="M"
)

numeric_columns = [
    "Actual_excess_return",
    "FF5_forecast",
    "FF6_forecast"
]

for column in numeric_columns:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

if data[numeric_columns].isna().any().any():

    raise ValueError("Missing forecast or return observations.")

if not np.isfinite(
    data[numeric_columns].to_numpy()
).all():

    raise ValueError("Invalid numerical observations.")


# Check unique portfolio-month observations.

if data.duplicated(
    ["Dataset", "Date", "Portfolio"]
).any():

    raise ValueError("Duplicate portfolio-month observations.")


# ==================================================
# 3. RECONSTRUCT FORECAST ERRORS
# ==================================================

data["Error_FF5"] = (
    data["Actual_excess_return"]
    - data["FF5_forecast"]
)

data["Error_FF6"] = (
    data["Actual_excess_return"]
    - data["FF6_forecast"]
)


# ==================================================
# 4. CLARK-WEST ADJUSTMENT
# ==================================================

data["CW_adjusted_loss_difference"] = (

    data["Error_FF5"] ** 2

    - data["Error_FF6"] ** 2

    + (
        data["FF5_forecast"]
        - data["FF6_forecast"]
    ) ** 2

)


# Raw squared-error difference.

data["Raw_squared_loss_difference"] = (

    data["Error_FF5"] ** 2

    - data["Error_FF6"] ** 2

)


# Absolute-error difference.

data["Absolute_loss_difference"] = (

    data["Error_FF5"].abs()

    - data["Error_FF6"].abs()

)


# ==================================================
# 5. AGGREGATE PORTFOLIOS BY MONTH
# ==================================================

monthly = (
    data.groupby(
        ["Dataset", "Date"],
        as_index=False
    )
    .agg(

        CW_adjusted_difference=(
            "CW_adjusted_loss_difference",
            "mean"
        ),

        Raw_squared_difference=(
            "Raw_squared_loss_difference",
            "mean"
        ),

        Absolute_difference=(
            "Absolute_loss_difference",
            "mean"
        ),

        Number_of_portfolios=(
            "Portfolio",
            "nunique"
        )

    )
)


# Ensure constant portfolio count within each dataset.

for dataset, group in monthly.groupby("Dataset"):

    if group["Number_of_portfolios"].nunique() != 1:

        raise ValueError(
            "Portfolio count changes over time in " + dataset
        )


# ==================================================
# 6. HAC TEST FUNCTION
# ==================================================

def run_hac_test(series, lags):

    series = np.asarray(series, dtype=float)

    X = np.ones(
        (len(series), 1)
    )

    result = sm.OLS(
        series,
        X
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": lags
        },
        use_t=False
    )

    mean_difference = result.params[0]

    standard_error = result.bse[0]

    z_statistic = (
        mean_difference / standard_error
    )

    return {

        "Mean_difference": mean_difference,

        "HAC_SE": standard_error,

        "Z_statistic": z_statistic,

        "N_months": len(series)

    }


# ==================================================
# 7. RUN CLARK-WEST AND SECONDARY TESTS
# ==================================================

cw_results = []

secondary_results = []


for dataset_name, group in monthly.groupby("Dataset"):

    group = group.sort_values("Date")

    print("\n" + "=" * 70)

    print("DATASET:", dataset_name)

    print("Forecast months:", len(group))

    print(
        "Portfolio count:",
        group["Number_of_portfolios"].iloc[0]
    )

    print("=" * 70)

    for lag in HAC_LAGS:

        # --------------------------------------------------
        # CLARK-WEST TEST
        # --------------------------------------------------

        cw = run_hac_test(
            group["CW_adjusted_difference"],
            lag
        )

        # One-sided alternative:
        # FF6 has superior forecast accuracy.

        cw_p = norm.sf(
            cw["Z_statistic"]
        )

        # Scale squared returns into squared percentage points.

        cw_results.append({

            "Dataset": dataset_name,

            "HAC_lags": lag,

            "CW_adjusted_difference_pct_squared":
                cw["Mean_difference"] * 10000,

            "HAC_SE_pct_squared":
                cw["HAC_SE"] * 10000,

            "CW_z_statistic":
                cw["Z_statistic"],

            "CW_one_sided_p":
                cw_p,

            "N_months":
                cw["N_months"]

        })

        print("\nClark-West test")

        print("HAC lags:", lag)

        print(
            "Adjusted difference:",
            round(
                cw["Mean_difference"] * 10000,
                6
            )
        )

        print(
            "Z statistic:",
            round(cw["Z_statistic"], 4)
        )

        print(
            "One-sided p-value:",
            round(cw_p, 6)
        )


        # --------------------------------------------------
        # SECONDARY ABSOLUTE-ERROR TEST
        # --------------------------------------------------

        absolute = run_hac_test(
            group["Absolute_difference"],
            lag
        )

        # Two-sided test.

        absolute_p = 2 * norm.sf(
            abs(absolute["Z_statistic"])
        )

        secondary_results.append({

            "Dataset": dataset_name,

            "HAC_lags": lag,

            "Mean_absolute_loss_difference_pct":
                absolute["Mean_difference"] * 100,

            "HAC_SE_pct":
                absolute["HAC_SE"] * 100,

            "Z_statistic":
                absolute["Z_statistic"],

            "Two_sided_p":
                absolute_p,

            "N_months":
                absolute["N_months"]

        })


# ==================================================
# 8. CREATE RESULT TABLES
# ==================================================

cw_df = pd.DataFrame(cw_results)

secondary_df = pd.DataFrame(
    secondary_results
)


# ==================================================
# 9. SAVE RESULTS
# ==================================================

cw_df.to_csv(
    TABLE_DIR / "oos_clark_west_results.csv",
    index=False
)

secondary_df.to_csv(
    TABLE_DIR / "oos_absolute_error_results.csv",
    index=False
)


# ==================================================
# 10. DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 80)

print("CLARK-WEST NESTED FORECAST COMPARISON")

print("=" * 80)

print(
    cw_df.round(6).to_string(index=False)
)


print("\n" + "=" * 80)

print("SECONDARY ABSOLUTE-ERROR COMPARISON")

print("=" * 80)

print(
    secondary_df.round(6).to_string(index=False)
)


# ==================================================
# 11. COMPLETION
# ==================================================

print("\nRESULTS SAVED")

print("oos_clark_west_results.csv")

print("oos_absolute_error_results.csv")

print("\nCLARK-WEST EVALUATION COMPLETED")
