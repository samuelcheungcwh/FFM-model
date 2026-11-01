
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

TRAINING_WINDOWS = [120, 180, 240]

HAC_LAGS = [6, 12, 24]


# ==================================================
# 2. DATASET CONFIGURATION
# ==================================================

DATASETS = {

    "Size-BM": {
        "file": "size_bm_excess.csv",
        "window": "matched"
    },

    "Size-Momentum": {
        "file": "size_momentum_excess.csv",
        "window": "matched"
    },

    "Industry-49": {
        "file": "industry_49_excess.csv",
        "window": "matched"
    },

    "Industry-47-Extended": {
        "file": "industry_47_excess.csv",
        "window": "full"
    }

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

def load_data(returns_file, window):

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

    # Check required factors.

    for factor in FF5_FACTORS:

        if factor not in ff5.columns:
            raise ValueError("Missing FF5 factor: " + factor)

    for factor in FF6_FACTORS:

        if factor not in ff6.columns:
            raise ValueError("Missing FF6 factor: " + factor)

    ff5 = ff5[FF5_FACTORS].copy()

    ff6 = ff6[FF6_FACTORS].copy()

    # Align dates.

    common_dates = (
        returns.index
        .intersection(ff5.index)
        .intersection(ff6.index)
    )

    returns = returns.loc[common_dates]

    ff5 = ff5.loc[common_dates]

    ff6 = ff6.loc[common_dates]

    # Restrict the main groups to the matched period.

    if window == "matched":

        start = pd.Period("1969-07", freq="M")

        end = pd.Period("2025-12", freq="M")

        selected = (
            (returns.index >= start)
            & (returns.index <= end)
        )

        returns = returns.loc[selected]

        ff5 = ff5.loc[selected]

        ff6 = ff6.loc[selected]

    # Check the common factors.

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

    # Check invalid observations.

    if not np.isfinite(Y).all():
        raise ValueError("Invalid portfolio returns.")

    if not np.isfinite(F5).all():
        raise ValueError("Invalid FF5 factors.")

    if not np.isfinite(F6).all():
        raise ValueError("Invalid FF6 factors.")

    return Y, F5, F6, returns.index, returns.columns


# ==================================================
# 4. EXPANDING-WINDOW FORECAST
# ==================================================

def expanding_forecast(Y, F, initial_train):

    T, N = Y.shape

    K = F.shape[1]

    if initial_train >= T:
        raise ValueError("Training period is too long.")

    predictions = []

    benchmarks = []

    # Forecast one month at a time.

    for t in range(initial_train, T):

        # All observations available before month t.

        Y_train = Y[:t, :]

        F_train = F[:t, :]

        # ------------------------------------------
        # FIRST PASS: ESTIMATE FACTOR BETAS
        # ------------------------------------------

        X_train = np.column_stack([
            np.ones(t),
            F_train
        ])

        if np.linalg.matrix_rank(X_train) < K + 1:

            raise ValueError(
                "Factor matrix is rank deficient."
            )

        coefficients = np.linalg.lstsq(
            X_train,
            Y_train,
            rcond=None
        )[0]

        betas = coefficients[1:, :]

        # ------------------------------------------
        # SECOND PASS: HISTORICAL FACTOR PREMIUMS
        # ------------------------------------------

        risk_premiums = F_train.mean(axis=0)

        # ------------------------------------------
        # FORECAST NEXT MONTH
        # ------------------------------------------

        predicted_returns = (
            betas.T @ risk_premiums
        )

        predictions.append(predicted_returns)

        # ------------------------------------------
        # HISTORICAL-MEAN BENCHMARK
        # ------------------------------------------

        historical_mean = Y_train.mean(axis=0)

        benchmarks.append(historical_mean)

    predictions = np.array(predictions)

    benchmarks = np.array(benchmarks)

    actual = Y[initial_train:, :]

    return actual, predictions, benchmarks


# ==================================================
# 5. CALCULATE FORECAST METRICS
# ==================================================

def calculate_metrics(actual, predictions, benchmark):

    errors = actual - predictions

    benchmark_errors = actual - benchmark

    model_sse = np.sum(errors ** 2)

    benchmark_sse = np.sum(
        benchmark_errors ** 2
    )

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    mae = np.mean(
        np.abs(errors)
    )

    # Out-of-sample R-squared against historical mean.

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
# 7. PORTFOLIO-LEVEL METRICS
# ==================================================

def calculate_portfolio_metrics(
    actual,
    predictions,
    benchmark,
    dataset_name,
    model_name,
    training_window,
    portfolios
):

    results = []

    for j, portfolio in enumerate(portfolios):

        y = actual[:, j]

        forecast = predictions[:, j]

        historical = benchmark[:, j]

        errors = y - forecast

        historical_errors = y - historical

        model_sse = np.sum(errors ** 2)

        benchmark_sse = np.sum(
            historical_errors ** 2
        )

        results.append({

            "Dataset": dataset_name,

            "Model": model_name,

            "Training_window": training_window,

            "Portfolio": portfolio,

            "RMSE_pct": (
                np.sqrt(np.mean(errors ** 2)) * 100
            ),

            "MAE_pct": (
                np.mean(np.abs(errors)) * 100
            ),

            "OOS_R2_pct": (
                1 - model_sse / benchmark_sse
            ) * 100

        })

    return results


# ==================================================
# 8. RUN ALL DATASETS AND TRAINING WINDOWS
# ==================================================

summary_results = []

test_results = []

portfolio_results = []

monthly_results = []

forecast_results = []


for dataset_name, specification in DATASETS.items():

    # Load dataset once.

    Y, F5, F6, dates, portfolios = load_data(
        specification["file"],
        specification["window"]
    )

    T, N = Y.shape

    print("\n" + "=" * 75)

    print("DATASET:", dataset_name)

    print("Observations:", T)

    print("Portfolios:", N)

    print("First month:", dates.min())

    print("Last month:", dates.max())

    print("=" * 75)

    for initial_train in TRAINING_WINDOWS:

        if initial_train >= T:

            raise ValueError(
                "Insufficient observations for "
                + dataset_name
            )

        print("\nTRAINING WINDOW:", initial_train)

        # ------------------------------------------
        # FF5 FORECAST
        # ------------------------------------------

        actual, pred5, benchmark5 = expanding_forecast(
            Y,
            F5,
            initial_train
        )

        # ------------------------------------------
        # FF6 FORECAST
        # ------------------------------------------

        actual6, pred6, benchmark6 = expanding_forecast(
            Y,
            F6,
            initial_train
        )

        if not np.allclose(actual, actual6):
            raise ValueError("Forecast actuals differ.")

        if not np.allclose(benchmark5, benchmark6):
            raise ValueError("Historical benchmarks differ.")

        # ------------------------------------------
        # FORECAST ERRORS
        # ------------------------------------------

        errors5 = actual - pred5

        errors6 = actual - pred6

        test_dates = dates[initial_train:]

        print("Training observations:", initial_train)

        print("OOS observations:", len(test_dates))

        print("First forecast:", test_dates.min())

        print("Last forecast:", test_dates.max())

        # ------------------------------------------
        # MODEL PERFORMANCE
        # ------------------------------------------

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

        model_metrics = {

            "FF5": metrics5,

            "FF6": metrics6,

            "HistoricalMean": benchmark_metrics

        }

        for model_name, metrics in model_metrics.items():

            summary_results.append({

                "Dataset": dataset_name,

                "Model": model_name,

                "Training_window": initial_train,

                "Training_years": initial_train / 12,

                "OOS_observations": len(test_dates),

                "Portfolios": N,

                "First_OOS_month": str(test_dates.min()),

                "Last_OOS_month": str(test_dates.max()),

                **metrics

            })

        # ------------------------------------------
        # MONTHLY LOSS DIFFERENCES
        # ------------------------------------------

        # Clark-West adjusted squared-loss difference.

        cw_difference = np.mean(

            errors5 ** 2

            - errors6 ** 2

            + (pred5 - pred6) ** 2,

            axis=1

        )

        # Absolute-error difference.

        absolute_difference = np.mean(

            np.abs(errors5)
            - np.abs(errors6),

            axis=1

        )

        for t, date in enumerate(test_dates):

            monthly_results.append({

                "Dataset": dataset_name,

                "Training_window": initial_train,

                "Date": str(date),

                "CW_adjusted_difference": cw_difference[t],

                "Absolute_loss_difference": absolute_difference[t]

            })

        # ------------------------------------------
        # CLARK-WEST AND ABSOLUTE-ERROR TESTS
        # ------------------------------------------

        for lag in HAC_LAGS:

            # Clark-West: one-sided alternative FF6 is better.

            cw = run_hac_test(
                cw_difference,
                lag
            )

            cw_p = norm.sf(
                cw["Z_statistic"]
            )

            test_results.append({

                "Dataset": dataset_name,

                "Training_window": initial_train,

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

            # Absolute error: two-sided alternative.

            absolute = run_hac_test(
                absolute_difference,
                lag
            )

            absolute_p = 2 * norm.sf(
                abs(absolute["Z_statistic"])
            )

            test_results.append({

                "Dataset": dataset_name,

                "Training_window": initial_train,

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

        # ------------------------------------------
        # PORTFOLIO-LEVEL RESULTS
        # ------------------------------------------

        portfolio_results.extend(
            calculate_portfolio_metrics(
                actual,
                pred5,
                benchmark5,
                dataset_name,
                "FF5",
                initial_train,
                portfolios
            )
        )

        portfolio_results.extend(
            calculate_portfolio_metrics(
                actual,
                pred6,
                benchmark5,
                dataset_name,
                "FF6",
                initial_train,
                portfolios
            )
        )

        # ------------------------------------------
        # SAVE MONTHLY FORECASTS
        # ------------------------------------------

        for t, date in enumerate(test_dates):

            for j, portfolio in enumerate(portfolios):

                forecast_results.append({

                    "Dataset": dataset_name,

                    "Training_window": initial_train,

                    "Date": str(date),

                    "Portfolio": portfolio,

                    "Actual": actual[t, j],

                    "HistoricalMean": benchmark5[t, j],

                    "FF5_forecast": pred5[t, j],

                    "FF6_forecast": pred6[t, j],

                    "FF5_error": errors5[t, j],

                    "FF6_error": errors6[t, j]

                })


# ==================================================
# 9. COMBINE RESULTS
# ==================================================

summary_df = pd.DataFrame(summary_results)

test_df = pd.DataFrame(test_results)

portfolio_df = pd.DataFrame(portfolio_results)

monthly_df = pd.DataFrame(monthly_results)

forecast_df = pd.DataFrame(forecast_results)


# ==================================================
# 10. SAVE RESULTS
# ==================================================

summary_df.to_csv(
    TABLE_DIR / "oos_window_sensitivity_summary.csv",
    index=False
)

test_df.to_csv(
    TABLE_DIR / "oos_window_sensitivity_tests.csv",
    index=False
)

portfolio_df.to_csv(
    TABLE_DIR / "oos_window_sensitivity_portfolios.csv",
    index=False
)

monthly_df.to_csv(
    TABLE_DIR / "oos_window_sensitivity_monthly.csv",
    index=False
)

forecast_df.to_csv(
    TABLE_DIR / "oos_window_sensitivity_forecasts.csv",
    index=False
)


# ==================================================
# 11. DISPLAY MODEL PERFORMANCE
# ==================================================

print("\n" + "=" * 90)

print("OUT-OF-SAMPLE TRAINING-WINDOW SENSITIVITY")

print("=" * 90)

print(
    summary_df.round(6).to_string(index=False)
)


# ==================================================
# 12. DISPLAY CLARK-WEST TESTS
# ==================================================

print("\n" + "=" * 90)

print("CLARK-WEST TESTS")

print("=" * 90)

cw_df = test_df[
    test_df["Test"] == "Clark-West"
]

print(
    cw_df.round(6).to_string(index=False)
)


# ==================================================
# 13. DISPLAY PRIMARY 12-LAG RESULTS
# ==================================================

print("\n" + "=" * 90)

print("PRIMARY CLARK-WEST RESULTS: 12 HAC LAGS")

print("=" * 90)

primary = cw_df[
    cw_df["HAC_lags"] == 12
]

print(
    primary.round(6).to_string(index=False)
)


# ==================================================
# 14. COMPLETION
# ==================================================

print("\nFILES SAVED")

print("oos_window_sensitivity_summary.csv")

print("oos_window_sensitivity_tests.csv")

print("oos_window_sensitivity_portfolios.csv")

print("oos_window_sensitivity_monthly.csv")

print("oos_window_sensitivity_forecasts.csv")

print("\nTRAINING WINDOWS:", TRAINING_WINDOWS)

print("HAC LAGS:", HAC_LAGS)

print("\nOOS WINDOW SENSITIVITY COMPLETED")
