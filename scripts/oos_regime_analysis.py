
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import norm


# ==================================================
# 1. SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data_processed"

TABLE_DIR = BASE_DIR / "results" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = (
    TABLE_DIR /
    "oos_rolling_monthly_losses.csv"
)

CALIBRATION_END = "1989-06"

OOS_START = "1989-07"

VOLATILITY_WINDOW = 12

HAC_LAGS = 12


# ==================================================
# 2. LOAD MARKET FACTOR
# ==================================================

factors = pd.read_csv(
    DATA_DIR / "factors_ff5.csv",
    index_col=0
)

factors.index = pd.PeriodIndex(
    factors.index.astype(str),
    freq="M"
)

factors.columns = factors.columns.str.strip()

factors = factors.sort_index()

if "Mkt-RF" not in factors.columns:

    raise ValueError(
        "Mkt-RF factor was not found."
    )


market = pd.to_numeric(
    factors["Mkt-RF"],
    errors="coerce"
)

if market.isna().any():

    raise ValueError(
        "Missing market factor observations."
    )


# ==================================================
# 3. CALCULATE LAGGED MARKET VOLATILITY
# ==================================================

# At month t, this uses market returns from
# t-12 through t-1.

lagged_volatility = (
    market
    .rolling(VOLATILITY_WINDOW)
    .std()
    .shift(1)
)


# ==================================================
# 4. ESTABLISH REGIME THRESHOLD
# ==================================================

calibration_end = pd.Period(
    CALIBRATION_END,
    freq="M"
)

oos_start = pd.Period(
    OOS_START,
    freq="M"
)

calibration_volatility = (
    lagged_volatility.loc[:calibration_end]
    .dropna()
)

if len(calibration_volatility) == 0:

    raise ValueError(
        "No volatility observations in calibration period."
    )


# Fix the threshold before the out-of-sample period.

volatility_threshold = (
    calibration_volatility.median()
)


# ==================================================
# 5. CREATE MARKET REGIMES
# ==================================================

regime_data = pd.DataFrame({

    "Date": lagged_volatility.index.astype(str),

    "Lagged_12M_Volatility": (
        lagged_volatility.to_numpy()
    )

})

regime_data["High_Volatility"] = (

    regime_data["Lagged_12M_Volatility"]
    > volatility_threshold

).astype(int)


regime_data["Regime"] = np.where(

    regime_data["High_Volatility"] == 1,

    "High volatility",

    "Low volatility"

)


# Retain the out-of-sample period.

regime_data = regime_data[
    regime_data["Date"] >= str(oos_start)
].copy()


if regime_data["High_Volatility"].nunique() != 2:

    raise ValueError(
        "Both volatility regimes are not represented."
    )


print("\nMARKET REGIME CLASSIFICATION")

print("Calibration ends:", CALIBRATION_END)

print("First OOS month:", OOS_START)

print(
    "Median volatility threshold (%):",
    round(volatility_threshold * 100, 6)
)

print("\nREGIME COUNTS")

print(
    regime_data["Regime"].value_counts().to_string()
)


# ==================================================
# 6. LOAD ROLLING-WINDOW FORECAST LOSSES
# ==================================================

monthly = pd.read_csv(INPUT_FILE)

monthly["Date"] = pd.PeriodIndex(
    monthly["Date"].astype(str),
    freq="M"
)

monthly["Date"] = monthly["Date"].astype(str)

required_columns = [

    "Dataset",

    "Rolling_window",

    "Date",

    "CW_adjusted_difference",

    "Absolute_loss_difference"

]

for column in required_columns:

    if column not in monthly.columns:

        raise ValueError(
            "Missing column: " + column
        )


# Correct the label from the earlier script.
# This sample has 47 industries over 1969-2025,
# not the extended 1963-2025 historical period.

monthly["Dataset"] = monthly["Dataset"].replace({

    "Industry-47-Extended": "Industry-47-Matched"

})


# ==================================================
# 7. MERGE REGIMES WITH FORECAST RESULTS
# ==================================================

monthly = monthly.merge(
    regime_data,
    on="Date",
    how="inner",
    validate="many_to_one"
)

monthly = monthly.sort_values(
    ["Dataset", "Rolling_window", "Date"]
)

if monthly["Regime"].isna().any():

    raise ValueError(
        "Some forecast months could not be classified."
    )


# ==================================================
# 8. HAC REGIME REGRESSION
# ==================================================

def calculate_linear_combination(
    result,
    weights,
    alternative
):

    weights = np.asarray(
        weights,
        dtype=float
    )

    estimate = float(
        weights @ result.params
    )

    covariance = result.cov_params()

    standard_error = np.sqrt(
        weights @ covariance @ weights
    )

    z_statistic = (
        estimate / standard_error
    )

    if alternative == "positive":

        p_value = norm.sf(z_statistic)

    else:

        p_value = 2 * norm.sf(
            abs(z_statistic)
        )

    lower = estimate - 1.96 * standard_error

    upper = estimate + 1.96 * standard_error

    return {

        "Estimate": estimate,

        "HAC_SE": standard_error,

        "Z_statistic": z_statistic,

        "P_value": p_value,

        "CI_lower": lower,

        "CI_upper": upper

    }


def run_regime_test(
    group,
    outcome,
    scale,
    dataset_name,
    rolling_window
):

    # Ensure observations are chronological.

    group = group.sort_values("Date").copy()

    y = (
        group[outcome].to_numpy(dtype=float)
        * scale
    )

    high = group[
        "High_Volatility"
    ].to_numpy(dtype=float)

    # Regression:
    #
    # loss_difference_t =
    # alpha + beta * HighVol_t + error_t
    #
    # alpha       = average low-volatility performance
    # alpha+beta  = average high-volatility performance
    # beta        = high minus low difference

    X = np.column_stack([
        np.ones(len(group)),
        high
    ])

    result = sm.OLS(
        y,
        X
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": HAC_LAGS
        },
        use_t=False
    )

    # --------------------------------------------------
    # LOW-VOLATILITY PERFORMANCE
    # --------------------------------------------------

    low_result = calculate_linear_combination(
        result,
        [1, 0],
        "positive"
    )

    # --------------------------------------------------
    # HIGH-VOLATILITY PERFORMANCE
    # --------------------------------------------------

    high_result = calculate_linear_combination(
        result,
        [1, 1],
        "positive"
    )

    # --------------------------------------------------
    # HIGH MINUS LOW DIFFERENCE
    # --------------------------------------------------

    difference_result = calculate_linear_combination(
        result,
        [0, 1],
        "two-sided"
    )

    return {

        "Dataset": dataset_name,

        "Rolling_window": rolling_window,

        "Outcome": outcome,

        "Low_vol_months": int(
            (high == 0).sum()
        ),

        "High_vol_months": int(
            (high == 1).sum()
        ),

        "Low_vol_mean": low_result["Estimate"],

        "Low_vol_SE": low_result["HAC_SE"],

        "Low_vol_p_one_sided": low_result["P_value"],

        "High_vol_mean": high_result["Estimate"],

        "High_vol_SE": high_result["HAC_SE"],

        "High_vol_p_one_sided": high_result["P_value"],

        "High_minus_low": difference_result["Estimate"],

        "Difference_SE": difference_result["HAC_SE"],

        "Difference_Z": difference_result["Z_statistic"],

        "Difference_p_two_sided": difference_result["P_value"],

        "Difference_CI_lower": difference_result["CI_lower"],

        "Difference_CI_upper": difference_result["CI_upper"],

        "HAC_lags": HAC_LAGS

    }


# ==================================================
# 9. RUN REGIME TESTS
# ==================================================

results = []


for (dataset_name, rolling_window), group in monthly.groupby(
    ["Dataset", "Rolling_window"]
):

    print("\n" + "=" * 70)

    print("DATASET:", dataset_name)

    print("ROLLING WINDOW:", rolling_window)

    print("=" * 70)

    # Clark-West differences are in decimal squared-return units.
    # Convert to squared percentage-point units.

    cw_result = run_regime_test(

        group,

        "CW_adjusted_difference",

        10000,

        dataset_name,

        rolling_window

    )

    results.append(cw_result)

    # Absolute loss is a secondary outcome.
    # Convert decimal return differences to percentage points.

    absolute_result = run_regime_test(

        group,

        "Absolute_loss_difference",

        100,

        dataset_name,

        rolling_window

    )

    results.append(absolute_result)

    print(
        "Low-volatility CW mean:",
        round(cw_result["Low_vol_mean"], 6)
    )

    print(
        "High-volatility CW mean:",
        round(cw_result["High_vol_mean"], 6)
    )

    print(
        "High minus low CW difference:",
        round(cw_result["High_minus_low"], 6)
    )

    print(
        "Regime-difference p-value:",
        round(cw_result["Difference_p_two_sided"], 6)
    )


# ==================================================
# 10. SAVE RESULTS
# ==================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    TABLE_DIR / "oos_regime_results.csv",
    index=False
)


# ==================================================
# 11. DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 90)

print("OUT-OF-SAMPLE VOLATILITY REGIME RESULTS")

print("=" * 90)

print(
    results_df.round(6).to_string(index=False)
)


# ==================================================
# 12. PRIMARY RESULTS: 10-YEAR ROLLING WINDOW
# ==================================================

primary_results = results_df[
    results_df["Rolling_window"] == 120
]

print("\n" + "=" * 90)

print("PRIMARY REGIME RESULTS: 10-YEAR ROLLING WINDOW")

print("=" * 90)

print(
    primary_results.round(6).to_string(index=False)
)


# ==================================================
# 13. COMPLETION
# ==================================================

print("\nRESULTS SAVED")

print("oos_regime_results.csv")

print("Calibration threshold (%):",
      round(volatility_threshold * 100, 6))

print("\nOOS REGIME ANALYSIS COMPLETED")
