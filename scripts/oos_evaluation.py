
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ==================================================
# 1. SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data_processed"

TABLE_DIR = BASE_DIR / "results" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_TRAIN = 120

DM_HAC_LAGS = [6, 12, 24]


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
# 3. LOAD DATA
# ==================================================

def load_data(returns_file, window):

    # Load portfolio returns.

    returns = pd.read_csv(
        DATA_DIR / returns_file,
        index_col=0
    )

    # Load factor data.

    ff5 = pd.read_csv(
        DATA_DIR / "factors_ff5.csv",
        index_col=0
    )

    ff6 = pd.read_csv(
        DATA_DIR / "factors_ff6.csv",
        index_col=0
    )

    # Convert dates to monthly periods.

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

    # Sort datasets.

    returns = returns.sort_index()

    ff5 = ff5.sort_index()

    ff6 = ff6.sort_index()

    # Check duplicate dates.

    if returns.index.has_duplicates:
        raise ValueError("Duplicate dates in returns.")

    if ff5.index.has_duplicates:
        raise ValueError("Duplicate dates in FF5.")

    if ff6.index.has_duplicates:
        raise ValueError("Duplicate dates in FF6.")

    # Check factor columns.

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

    # Use matched period for the three main groups.

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

    # Ensure common factors are identical.

    if not np.allclose(
        ff5.to_numpy(),
        ff6[FF5_FACTORS].to_numpy(),
        rtol=0,
        atol=1e-10
    ):

        raise ValueError(
            "Common FF5 factor observations differ."
        )

    # Convert to arrays.

    Y = returns.to_numpy(dtype=float)

    F5 = ff5.to_numpy(dtype=float)

    F6 = ff6.to_numpy(dtype=float)

    # Validate observations.

    if not np.isfinite(Y).all():
        raise ValueError("Invalid portfolio returns.")

    if not np.isfinite(F5).all():
        raise ValueError("Invalid FF5 observations.")

    if not np.isfinite(F6).all():
        raise ValueError("Invalid FF6 observations.")

    return Y, F5, F6, returns.index, returns.columns


# ==================================================
# 4. EXPANDING-WINDOW FORECAST
# ==================================================

def expanding_forecast(Y, F, initial_train):

    T, N = Y.shape

    K = F.shape[1]

    if initial_train >= T:
        raise ValueError("Initial training sample is too large.")

    # Store forecasts.

    predictions = []

    benchmarks = []

    # Start the first forecast after the initial
    # 120 months of training observations.

    for t in range(initial_train, T):

        # ------------------------------------------
        # TRAINING SAMPLE
        # ------------------------------------------

        Y_train = Y[:t, :]

        F_train = F[:t, :]

        # ------------------------------------------
        # FIRST STAGE: ESTIMATE PORTFOLIO BETAS
        # ------------------------------------------

        X_train = np.column_stack([
            np.ones(t),
            F_train
        ])

        if np.linalg.matrix_rank(X_train) < K + 1:

            raise ValueError(
                "Training factor matrix is rank deficient."
            )

        coefficients = np.linalg.lstsq(
            X_train,
            Y_train,
            rcond=None
        )[0]

        # Exclude estimated intercepts.
        # The pricing model imposes alpha = 0.

        betas = coefficients[1:, :]

        # ------------------------------------------
        # SECOND STAGE: ESTIMATE FACTOR RISK PREMIUMS
        # ------------------------------------------

        # Historical average factor returns.

        risk_premiums = F_train.mean(axis=0)

        # ------------------------------------------
        # NEXT-MONTH EXPECTED RETURN
        # ------------------------------------------

        predicted_returns = (
            betas.T @ risk_premiums
        )

        predictions.append(
            predicted_returns
        )

        # ------------------------------------------
        # HISTORICAL MEAN BENCHMARK
        # ------------------------------------------

        historical_mean = Y_train.mean(axis=0)

        benchmarks.append(
            historical_mean
        )

    predictions = np.array(predictions)

    benchmarks = np.array(benchmarks)

    actual = Y[initial_train:, :]

    return actual, predictions, benchmarks


# ==================================================
# 5. CALCULATE OUT-OF-SAMPLE METRICS
# ==================================================

def calculate_forecast_metrics(actual, predictions, benchmark):

    errors = actual - predictions

    benchmark_errors = actual - benchmark

    # Pooled squared and absolute errors.

    model_sse = np.sum(errors ** 2)

    benchmark_sse = np.sum(
        benchmark_errors ** 2
    )

    oos_rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    oos_mae = np.mean(
        np.abs(errors)
    )

    # Out-of-sample R-squared relative to
    # the expanding historical-average benchmark.

    oos_r_squared = (
        1 - model_sse / benchmark_sse
    ) * 100

    # Average pricing errors across portfolios.

    portfolio_mean_errors = errors.mean(axis=0)

    average_absolute_mean_error = (
        np.mean(
            np.abs(portfolio_mean_errors)
        ) * 100
    )

    rms_mean_error = (
        np.sqrt(
            np.mean(portfolio_mean_errors ** 2)
        ) * 100
    )

    return {

        "OOS_RMSE_pct": oos_rmse * 100,

        "OOS_MAE_pct": oos_mae * 100,

        "OOS_R2_pct": oos_r_squared,

        "Average_absolute_mean_error_pct":
            average_absolute_mean_error,

        "RMS_mean_error_pct":
            rms_mean_error

    }


# ==================================================
# 6. HAC LOSS-DIFFERENTIAL TEST
# ==================================================

def run_loss_test(errors5, errors6, loss_type, hac_lags):

    if loss_type == "Squared":

        # Positive difference means FF6 has
        # lower average squared forecast error.

        loss5 = np.mean(
            errors5 ** 2,
            axis=1
        )

        loss6 = np.mean(
            errors6 ** 2,
            axis=1
        )

        difference = loss5 - loss6

        # Convert decimal squared returns into
        # squared percentage-point units.

        scale = 10000

    elif loss_type == "Absolute":

        # Positive difference means FF6 has
        # lower average absolute forecast error.

        loss5 = np.mean(
            np.abs(errors5),
            axis=1
        )

        loss6 = np.mean(
            np.abs(errors6),
            axis=1
        )

        difference = loss5 - loss6

        scale = 100

    else:

        raise ValueError("Unknown loss type.")

    # Test whether mean loss difference equals zero.

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
        }
    )

    return {

        "Loss_type": loss_type,

        "HAC_lags": hac_lags,

        "Mean_loss_difference": (
            result.params[0] * scale
        ),

        "HAC_SE": (
            result.bse[0] * scale
        ),

        "Test_statistic": result.tvalues[0],

        "P_value": result.pvalues[0],

        "Observations": len(difference)

    }


# ==================================================
# 7. RUN ALL DATASETS
# ==================================================

summary_results = []

dm_results = []

portfolio_results = []

forecast_results = []


for dataset_name, specification in DATASETS.items():

    # Load data.

    Y, F5, F6, dates, portfolios = load_data(
        specification["file"],
        specification["window"]
    )

    T, N = Y.shape

    print("\n" + "=" * 70)

    print("DATASET:", dataset_name)

    print("Observations:", T)

    print("Portfolios:", N)

    print("Initial training observations:", INITIAL_TRAIN)

    print("First month:", dates.min())

    print("Last month:", dates.max())

    print("=" * 70)

    if T <= INITIAL_TRAIN:

        raise ValueError(
            "Not enough observations for out-of-sample testing."
        )

    # ------------------------------------------
    # FF5 EXPANDING FORECAST
    # ------------------------------------------

    actual, predictions5, benchmark5 = expanding_forecast(
        Y,
        F5,
        INITIAL_TRAIN
    )

    # ------------------------------------------
    # FF6 EXPANDING FORECAST
    # ------------------------------------------

    actual6, predictions6, benchmark6 = expanding_forecast(
        Y,
        F6,
        INITIAL_TRAIN
    )

    if not np.allclose(actual, actual6):
        raise ValueError("Actual returns do not match.")

    if not np.allclose(benchmark5, benchmark6):
        raise ValueError("Benchmark forecasts do not match.")

    # ------------------------------------------
    # FORECAST ERRORS
    # ------------------------------------------

    errors5 = actual - predictions5

    errors6 = actual - predictions6

    benchmark_errors = actual - benchmark5

    test_dates = dates[INITIAL_TRAIN:]

    print("Out-of-sample observations:", len(test_dates))

    print("First forecast:", test_dates.min())

    print("Last forecast:", test_dates.max())

    print("Forecasts completed.")

    # ------------------------------------------
    # MODEL PERFORMANCE
    # ------------------------------------------

    metrics5 = calculate_forecast_metrics(
        actual,
        predictions5,
        benchmark5
    )

    metrics6 = calculate_forecast_metrics(
        actual,
        predictions6,
        benchmark5
    )

    benchmark_metrics = calculate_forecast_metrics(
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

            "Training_observations": INITIAL_TRAIN,

            "OOS_observations": len(test_dates),

            "Portfolios": N,

            "First_OOS_month": str(test_dates.min()),

            "Last_OOS_month": str(test_dates.max()),

            **metrics

        })

    # ------------------------------------------
    # DIEBOLD-MARIANO-STYLE LOSS COMPARISON
    # ------------------------------------------

    for hac_lags in DM_HAC_LAGS:

        for loss_type in ["Squared", "Absolute"]:

            test_result = run_loss_test(
                errors5,
                errors6,
                loss_type,
                hac_lags
            )

            dm_results.append({

                "Dataset": dataset_name,

                **test_result

            })

    # ------------------------------------------
    # PORTFOLIO-LEVEL PERFORMANCE
    # ------------------------------------------

    for j, portfolio in enumerate(portfolios):

        portfolio_actual = actual[:, j]

        portfolio_forecast5 = predictions5[:, j]

        portfolio_forecast6 = predictions6[:, j]

        portfolio_benchmark = benchmark5[:, j]

        error5 = (
            portfolio_actual - portfolio_forecast5
        )

        error6 = (
            portfolio_actual - portfolio_forecast6
        )

        benchmark_error = (
            portfolio_actual - portfolio_benchmark
        )

        sse_benchmark = np.sum(
            benchmark_error ** 2
        )

        sse5 = np.sum(error5 ** 2)

        sse6 = np.sum(error6 ** 2)

        portfolio_results.append({

            "Dataset": dataset_name,

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
                1 - sse5 / sse_benchmark
            ) * 100,

            "FF6_OOS_R2_pct": (
                1 - sse6 / sse_benchmark
            ) * 100

        })

    # ------------------------------------------
    # SAVE MONTHLY FORECASTS
    # ------------------------------------------

    for t, date in enumerate(test_dates):

        for j, portfolio in enumerate(portfolios):

            forecast_results.append({

                "Dataset": dataset_name,

                "Date": str(date),

                "Portfolio": portfolio,

                "Actual_excess_return": actual[t, j],

                "Historical_mean_forecast": benchmark5[t, j],

                "FF5_forecast": predictions5[t, j],

                "FF6_forecast": predictions6[t, j],

                "FF5_error": errors5[t, j],

                "FF6_error": errors6[t, j],

                "Benchmark_error": benchmark_errors[t, j]

            })


# ==================================================
# 8. COMBINE RESULTS
# ==================================================

summary_df = pd.DataFrame(summary_results)

dm_df = pd.DataFrame(dm_results)

portfolio_df = pd.DataFrame(portfolio_results)

forecast_df = pd.DataFrame(forecast_results)


# ==================================================
# 9. SAVE RESULTS
# ==================================================

summary_df.to_csv(
    TABLE_DIR / "oos_forecast_summary.csv",
    index=False
)

dm_df.to_csv(
    TABLE_DIR / "oos_dm_tests.csv",
    index=False
)

portfolio_df.to_csv(
    TABLE_DIR / "oos_portfolio_metrics.csv",
    index=False
)

forecast_df.to_csv(
    TABLE_DIR / "oos_monthly_forecasts.csv",
    index=False
)


# ==================================================
# 10. DISPLAY MODEL PERFORMANCE
# ==================================================

print("\n" + "=" * 85)

print("OUT-OF-SAMPLE MODEL PERFORMANCE")

print("=" * 85)

print(
    summary_df.round(6).to_string(index=False)
)


# ==================================================
# 11. DISPLAY FF5 VS FF6 COMPARISON
# ==================================================

print("\n" + "=" * 85)

print("OUT-OF-SAMPLE LOSS-DIFFERENTIAL TESTS")

print("=" * 85)

print(
    dm_df.round(6).to_string(index=False)
)


# ==================================================
# 12. DISPLAY PRIMARY 12-LAG TEST
# ==================================================

print("\n" + "=" * 85)

print("PRIMARY FORECAST COMPARISON: 12 HAC LAGS")

print("=" * 85)

primary_dm = dm_df[
    dm_df["HAC_lags"] == 12
]

print(
    primary_dm.round(6).to_string(index=False)
)


# ==================================================
# 13. COMPLETION
# ==================================================

print("\nFILES SAVED")

print("oos_forecast_summary.csv")

print("oos_dm_tests.csv")

print("oos_portfolio_metrics.csv")

print("oos_monthly_forecasts.csv")

print("\nINITIAL TRAINING PERIOD:", INITIAL_TRAIN, "MONTHS")

print("\nOUT-OF-SAMPLE EVALUATION COMPLETED")
