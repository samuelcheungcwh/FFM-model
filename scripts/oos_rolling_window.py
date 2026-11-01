
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

ROLLING_WINDOWS = [120, 180, 240]

HAC_LAGS = [6, 12, 24]

COMMON_START = "1969-07"

COMMON_END = "2025-12"

OOS_START = "1989-07"


# ==================================================
# 2. DATASET CONFIGURATION
# ==================================================

DATASETS = {

    "Size-BM": "size_bm_excess.csv",

    "Size-Momentum": "size_momentum_excess.csv",

    "Industry-49": "industry_49_excess.csv",

    "Industry-47-Extended": "industry_47_excess.csv"

}


FF5_FACTORS = [
    "Mkt-RF",
    "SMB",
    "HML",
    "RMW",
    "CMA"
]

FF6_FACTORS = FF5_FACTORS + ["MOM"]


# ==================================================
# 3. LOAD AND VALIDATE DATA
# ==================================================

def load_data(returns_file):

    returns = pd.read_csv(
        DATA_DIR / returns_file,
        index_col=0
    )

    ff5 = pd.read_csv(
        DATA_DIR / "factors_ff5.csv",
        index_col=0
    )

    ff6 = pd.read_csv(
        DATA_DIR / "factors_ff6.csv",
        index_col=0
    )

    # Convert dates.

    returns.index = pd.PeriodIndex(
        returns.index.astype(str),
        freq="M"
    )

    ff5.index = pd.PeriodIndex(
        ff5.index.astype(str),
        freq="M"
    )

    ff6.index = pd.PeriodIndex(
        ff6.index.astype(str),
        freq="M"
    )

    # Clean column names.

    returns.columns = returns.columns.str.strip()

    ff5.columns = ff5.columns.str.strip()

    ff6.columns = ff6.columns.str.strip()

    # Sort dates.

    returns = returns.sort_index()

    ff5 = ff5.sort_index()

    ff6 = ff6.sort_index()

    # Check duplicate dates.

    if returns.index.has_duplicates:
        raise ValueError("Duplicate return dates.")

    if ff5.index.has_duplicates:
        raise ValueError("Duplicate FF5 dates.")

    if ff6.index.has_duplicates:
        raise ValueError("Duplicate FF6 dates.")

    # Select required factors.

    for factor in FF5_FACTORS:

        if factor not in ff5.columns:
            raise ValueError("Missing FF5 factor: " + factor)

    for factor in FF6_FACTORS:

        if factor not in ff6.columns:
            raise ValueError("Missing FF6 factor: " + factor)

    ff5 = ff5[FF5_FACTORS].copy()

    ff6 = ff6[FF6_FACTORS].copy()

    # Align all datasets.

    common_dates = (
        returns.index
        .intersection(ff5.index)
        .intersection(ff6.index)
    )

    returns = returns.loc[common_dates]

    ff5 = ff5.loc[common_dates]

    ff6 = ff6.loc[common_dates]

    # Restrict all samples to July 1969-Dec 2025.
    # This includes the 47-industry dataset.

    start = pd.Period(COMMON_START, freq="M")

    end = pd.Period(COMMON_END, freq="M")

    selected = (
        (returns.index >= start)
        & (returns.index <= end)
    )

    returns = returns.loc[selected]

    ff5 = ff5.loc[selected]

    ff6 = ff6.loc[selected]

    # Confirm common FF5 factors are identical.

    if not np.allclose(
        ff5.to_numpy(),
        ff6[FF5_FACTORS].to_numpy(),
        rtol=0,
        atol=1e-10
    ):

        raise ValueError(
            "Common FF5 factors differ between files."
        )

    Y = returns.to_numpy(dtype=float)

    F5 = ff5.to_numpy(dtype=float)

    F6 = ff6.to_numpy(dtype=float)

    # Validate numerical values.

    if not np.isfinite(Y).all():
        raise ValueError("Invalid portfolio returns.")

    if not np.isfinite(F5).all():
        raise ValueError("Invalid FF5 factors.")

    if not np.isfinite(F6).all():
        raise ValueError("Invalid FF6 factors.")

    return Y, F5, F6, returns.index, returns.columns


# ==================================================
# 4. ROLLING-WINDOW FORECAST FUNCTION
# ==================================================

def rolling_forecast(Y, F, window, first_oos_index):

    T, N = Y.shape

    K = F.shape[1]

    predictions = []

    benchmarks = []

    # Every model begins on the same forecast date.

    for t in range(first_oos_index, T):

        # Define the historical estimation window.

        start = t - window

        Y_train = Y[start:t, :]

        F_train = F[start:t, :]

        # Confirm the estimation window is complete.

        if len(Y_train) != window:

            raise ValueError(
                "Incorrect rolling training window."
            )

        # --------------------------------------------------
        # FIRST PASS: ESTIMATE FACTOR BETAS
        # --------------------------------------------------

        X_train = np.column_stack([
            np.ones(window),
            F_train
        ])

        if np.linalg.matrix_rank(X_train) < K + 1:

            raise ValueError(
                "Rolling factor matrix is rank deficient."
            )

        coefficients = np.linalg.lstsq(
            X_train,
            Y_train,
            rcond=None
        )[0]

        betas = coefficients[1:, :]

        # --------------------------------------------------
        # SECOND PASS: ESTIMATE FACTOR RISK PREMIUMS
        # --------------------------------------------------

        risk_premiums = F_train.mean(axis=0)

        # --------------------------------------------------
        # NEXT-MONTH FORECAST
        # --------------------------------------------------

        forecast = (
            betas.T @ risk_premiums
        )

        predictions.append(forecast)

        # --------------------------------------------------
        # ROLLING HISTORICAL-MEAN BENCHMARK
        # --------------------------------------------------

        historical_mean = Y_train.mean(axis=0)

        benchmarks.append(historical_mean)

    predictions = np.array(predictions)

    benchmarks = np.array(benchmarks)

    actual = Y[first_oos_index:, :]

    return actual, predictions, benchmarks


# ==================================================
# 5. CALCULATE MODEL PERFORMANCE
# ==================================================

def calculate_metrics(actual, predictions, benchmark):

    errors = actual - predictions

    benchmark_errors = actual - benchmark

    model_sse = np.sum(errors ** 2)

    benchmark_sse = np.sum(
        benchmark_errors ** 2
    )

    # Root mean squared forecast error.

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    # Mean absolute forecast error.

    mae = np.mean(
        np.abs(errors)
    )

    # Out-of-sample R-squared versus rolling mean.

    oos_r2 = (
        1 - model_sse / benchmark_sse
    ) * 100

    return {

        "OOS_RMSE_pct": rmse * 100,

        "OOS_MAE_pct": mae * 100,

        "OOS_R2_pct": oos_r2

    }


# ==================================================
# 6. HAC LOSS-DIFFERENTIAL TEST
# ==================================================

def run_hac_test(difference, hac_lags):

    difference = np.asarray(
        difference,
        dtype=float
    )

    X = np.ones(
        (len(difference), 1)
    )

    result = sm.OLS(
        difference,
        X
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": hac_lags
        },
        use_t=False
    )

    return {

        "Mean_difference": result.params[0],

        "HAC_SE": result.bse[0],

        "Z_statistic": result.tvalues[0],

        "N_months": len(difference)

    }


# ==================================================
# 7. RUN ALL DATASETS
# ==================================================

summary_results = []

test_results = []

portfolio_results = []

forecast_results = []

monthly_results = []


for dataset_name, returns_file in DATASETS.items():

    # Load data.

    Y, F5, F6, dates, portfolios = load_data(
        returns_file
    )

    T, N = Y.shape

    # Locate the common first forecast month.

    oos_start = pd.Period(
        OOS_START,
        freq="M"
    )

    matching_dates = np.where(
        dates == oos_start
    )[0]

    if len(matching_dates) != 1:

        raise ValueError(
            "Could not locate common OOS start date."
        )

    first_oos_index = int(
        matching_dates[0]
    )

    print("\n" + "=" * 75)

    print("DATASET:", dataset_name)

    print("Observations:", T)

    print("Portfolios:", N)

    print("First month:", dates.min())

    print("Last month:", dates.max())

    print("First OOS month:", OOS_START)

    print("Common OOS observations:", T - first_oos_index)

    print("=" * 75)

    for window in ROLLING_WINDOWS:

        if window > first_oos_index:

            raise ValueError(
                "Rolling window exceeds available history."
            )

        print("\nROLLING WINDOW:", window, "MONTHS")

        # --------------------------------------------------
        # FF5 FORECAST
        # --------------------------------------------------

        actual, pred5, benchmark5 = rolling_forecast(
            Y,
            F5,
            window,
            first_oos_index
        )

        # --------------------------------------------------
        # FF6 FORECAST
        # --------------------------------------------------

        actual6, pred6, benchmark6 = rolling_forecast(
            Y,
            F6,
            window,
            first_oos_index
        )

        if not np.allclose(actual, actual6):
            raise ValueError("Actual returns differ.")

        if not np.allclose(benchmark5, benchmark6):
            raise ValueError("Rolling benchmarks differ.")

        # --------------------------------------------------
        # FORECAST ERRORS
        # --------------------------------------------------

        errors5 = actual - pred5

        errors6 = actual - pred6

        test_dates = dates[first_oos_index:]

        # --------------------------------------------------
        # MODEL PERFORMANCE
        # --------------------------------------------------

        metrics5 = calculate_metrics(
            actual,
            pred5,
            benchmark5
        )

        metrics6 = calculate_metrics(
            actual,
            pred6,
            benchmark5
        )

        benchmark_metrics = calculate_metrics(
            actual,
            benchmark5,
            benchmark5
        )

        models = {

            "FF5": metrics5,

            "FF6": metrics6,

            "RollingMean": benchmark_metrics

        }

        for model_name, metrics in models.items():

            summary_results.append({

                "Dataset": dataset_name,

                "Model": model_name,

                "Rolling_window": window,

                "Rolling_years": window / 12,

                "OOS_observations": len(test_dates),

                "Portfolios": N,

                "First_OOS_month": str(test_dates.min()),

                "Last_OOS_month": str(test_dates.max()),

                **metrics

            })

        # --------------------------------------------------
        # CLARK-WEST LOSS DIFFERENCE
        # --------------------------------------------------

        cw_difference = np.mean(

            errors5 ** 2

            - errors6 ** 2

            + (pred5 - pred6) ** 2,

            axis=1

        )

        # --------------------------------------------------
        # ABSOLUTE-ERROR DIFFERENCE
        # --------------------------------------------------

        absolute_difference = np.mean(

            np.abs(errors5)
            - np.abs(errors6),

            axis=1

        )

        # --------------------------------------------------
        # SAVE MONTHLY LOSS DIFFERENCES
        # --------------------------------------------------

        for t, date in enumerate(test_dates):

            monthly_results.append({

                "Dataset": dataset_name,

                "Rolling_window": window,

                "Date": str(date),

                "CW_adjusted_difference": cw_difference[t],

                "Absolute_loss_difference": absolute_difference[t]

            })

        # --------------------------------------------------
        # FORMAL FORECAST TESTS
        # --------------------------------------------------

        for lag in HAC_LAGS:

            # Clark-West: one-sided alternative that FF6 is better.

            cw = run_hac_test(
                cw_difference,
                lag
            )

            cw_p = norm.sf(
                cw["Z_statistic"]
            )

            test_results.append({

                "Dataset": dataset_name,

                "Rolling_window": window,

                "Test": "Clark-West",

                "HAC_lags": lag,

                "Mean_difference_pct_squared": (
                    cw["Mean_difference"] * 10000
                ),

                "HAC_SE_pct_squared": (
                    cw["HAC_SE"] * 10000
                ),

                "Z_statistic": cw["Z_statistic"],

                "P_value": cw_p,

                "Alternative": "FF6 superior",

                "OOS_observations": cw["N_months"]

            })

            # Absolute-error comparison: two-sided.

            absolute = run_hac_test(
                absolute_difference,
                lag
            )

            absolute_p = 2 * norm.sf(
                abs(absolute["Z_statistic"])
            )

            test_results.append({

                "Dataset": dataset_name,

                "Rolling_window": window,

                "Test": "Absolute_Error",

                "HAC_lags": lag,

                "Mean_difference_pct": (
                    absolute["Mean_difference"] * 100
                ),

                "HAC_SE_pct": (
                    absolute["HAC_SE"] * 100
                ),

                "Z_statistic": absolute["Z_statistic"],

                "P_value": absolute_p,

                "Alternative": "Difference from zero",

                "OOS_observations": absolute["N_months"]

            })

        # --------------------------------------------------
        # PORTFOLIO-LEVEL RESULTS
        # --------------------------------------------------

        for j, portfolio in enumerate(portfolios):

            error5 = errors5[:, j]

            error6 = errors6[:, j]

            benchmark_error = (
                actual[:, j] - benchmark5[:, j]
            )

            benchmark_sse = np.sum(
                benchmark_error ** 2
            )

            portfolio_results.append({

                "Dataset": dataset_name,

                "Rolling_window": window,

                "Portfolio": portfolio,

                "FF5_RMSE_pct": (
                    np.sqrt(np.mean(error5 ** 2)) * 100
                ),

                "FF6_RMSE_pct": (
                    np.sqrt(np.mean(error6 ** 2)) * 100
                ),

                "FF5_MAE_pct": (
                    np.mean(np.abs(error5)) * 100
                ),

                "FF6_MAE_pct": (
                    np.mean(np.abs(error6)) * 100
                ),

                "FF5_OOS_R2_pct": (
                    1 - np.sum(error5 ** 2) / benchmark_sse
                ) * 100,

                "FF6_OOS_R2_pct": (
                    1 - np.sum(error6 ** 2) / benchmark_sse
                ) * 100

            })

        # --------------------------------------------------
        # SAVE MONTHLY FORECASTS
        # --------------------------------------------------

        for t, date in enumerate(test_dates):

            for j, portfolio in enumerate(portfolios):

                forecast_results.append({

                    "Dataset": dataset_name,

                    "Rolling_window": window,

                    "Date": str(date),

                    "Portfolio": portfolio,

                    "Actual_excess_return": actual[t, j],

                    "Rolling_mean_forecast": benchmark5[t, j],

                    "FF5_forecast": pred5[t, j],

                    "FF6_forecast": pred6[t, j],

                    "FF5_error": errors5[t, j],

                    "FF6_error": errors6[t, j]

                })


# ==================================================
# 8. COMBINE RESULTS
# ==================================================

summary_df = pd.DataFrame(summary_results)

test_df = pd.DataFrame(test_results)

portfolio_df = pd.DataFrame(portfolio_results)

monthly_df = pd.DataFrame(monthly_results)

forecast_df = pd.DataFrame(forecast_results)


# ==================================================
# 9. SAVE RESULTS
# ==================================================

summary_df.to_csv(
    TABLE_DIR / "oos_rolling_summary.csv",
    index=False
)

test_df.to_csv(
    TABLE_DIR / "oos_rolling_tests.csv",
    index=False
)

portfolio_df.to_csv(
    TABLE_DIR / "oos_rolling_portfolio_metrics.csv",
    index=False
)

monthly_df.to_csv(
    TABLE_DIR / "oos_rolling_monthly_losses.csv",
    index=False
)

forecast_df.to_csv(
    TABLE_DIR / "oos_rolling_forecasts.csv",
    index=False
)


# ==================================================
# 10. DISPLAY MODEL PERFORMANCE
# ==================================================

print("\n" + "=" * 90)

print("ROLLING-WINDOW MODEL PERFORMANCE")

print("=" * 90)

print(
    summary_df.round(6).to_string(index=False)
)


# ==================================================
# 11. DISPLAY CLARK-WEST RESULTS
# ==================================================

print("\n" + "=" * 90)

print("ROLLING-WINDOW CLARK-WEST TESTS")

print("=" * 90)

cw_results = test_df[
    test_df["Test"] == "Clark-West"
]

print(
    cw_results.round(6).to_string(index=False)
)


# ==================================================
# 12. DISPLAY PRIMARY CLARK-WEST RESULTS
# ==================================================

print("\n" + "=" * 90)

print("PRIMARY ROLLING-WINDOW RESULTS: 12 HAC LAGS")

print("=" * 90)

primary = cw_results[
    cw_results["HAC_lags"] == 12
]

print(
    primary.round(6).to_string(index=False)
)


# ==================================================
# 13. COMPLETION
# ==================================================

print("\nFILES SAVED")

print("oos_rolling_summary.csv")

print("oos_rolling_tests.csv")

print("oos_rolling_portfolio_metrics.csv")

print("oos_rolling_monthly_losses.csv")

print("oos_rolling_forecasts.csv")

print("\nROLLING WINDOWS:", ROLLING_WINDOWS)

print("COMMON OOS PERIOD: JULY 1989 - DECEMBER 2025")

print("\nROLLING-WINDOW EVALUATION COMPLETED")
