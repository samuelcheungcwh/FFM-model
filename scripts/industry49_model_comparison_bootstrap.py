
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# 1. SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data_processed"

TABLE_DIR = BASE_DIR / "results" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

B = 999

BLOCK_LENGTHS = [6, 12, 24]

SEED = 1401

rng = np.random.default_rng(SEED)


# ==================================================
# 2. SAMPLE AND FACTOR SPECIFICATIONS
# ==================================================

SAMPLES = {

    "49_industry_main": "industry_49_excess.csv",

    "47_industry_extended": "industry_47_excess.csv"

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

    # Load industry excess returns.

    returns = pd.read_csv(
        DATA_DIR / returns_file,
        index_col=0
    )

    # Load FF5 and FF6 factors.

    ff5 = pd.read_csv(
        DATA_DIR / "factors_ff5.csv",
        index_col=0
    )

    ff6 = pd.read_csv(
        DATA_DIR / "factors_ff6.csv",
        index_col=0
    )

    # Convert dates into monthly periods.

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

    # Check for duplicate dates.

    if returns.index.has_duplicates:
        raise ValueError("Duplicate dates in industry returns.")

    if ff5.index.has_duplicates:
        raise ValueError("Duplicate dates in FF5 factors.")

    if ff6.index.has_duplicates:
        raise ValueError("Duplicate dates in FF6 factors.")

    # Select required factors.

    missing_ff5 = [
        factor for factor in FF5_FACTORS
        if factor not in ff5.columns
    ]

    missing_ff6 = [
        factor for factor in FF6_FACTORS
        if factor not in ff6.columns
    ]

    if missing_ff5:
        raise ValueError(
            "Missing FF5 factors: " + str(missing_ff5)
        )

    if missing_ff6:
        raise ValueError(
            "Missing FF6 factors: " + str(missing_ff6)
        )

    ff5 = ff5[FF5_FACTORS].copy()

    ff6 = ff6[FF6_FACTORS].copy()

    # Align all three datasets by month.

    common_dates = (
        returns.index
        .intersection(ff5.index)
        .intersection(ff6.index)
    )

    if len(common_dates) != len(returns):

        raise ValueError(
            "Return and factor samples do not match."
        )

    returns = returns.loc[common_dates]

    ff5 = ff5.loc[common_dates]

    ff6 = ff6.loc[common_dates]

    # Confirm that the five common factors are identical.

    if not np.allclose(
        ff5.to_numpy(),
        ff6[FF5_FACTORS].to_numpy(),
        rtol=0,
        atol=1e-10
    ):

        raise ValueError(
            "The common FF5 factors differ between the two files."
        )

    # Convert data into numerical arrays.

    Y = returns.to_numpy(dtype=float)

    F5 = ff5.to_numpy(dtype=float)

    F6 = ff6.to_numpy(dtype=float)

    # Check for missing or invalid observations.

    if not np.isfinite(Y).all():
        raise ValueError("Invalid industry returns detected.")

    if not np.isfinite(F5).all():
        raise ValueError("Invalid FF5 factors detected.")

    if not np.isfinite(F6).all():
        raise ValueError("Invalid FF6 factors detected.")

    return Y, F5, F6


# ==================================================
# 4. STATIONARY BLOCK BOOTSTRAP
# ==================================================

def stationary_indices(T, mean_block_length, rng):

    indices = np.empty(T, dtype=int)

    # Select the first observation randomly.

    indices[0] = rng.integers(0, T)

    # Probability of starting a new block.

    restart_probability = 1 / mean_block_length

    for t in range(1, T):

        if rng.random() < restart_probability:

            # Start a new block.

            indices[t] = rng.integers(0, T)

        else:

            # Continue the previous block.

            indices[t] = (
                indices[t - 1] + 1
            ) % T

    return indices


# ==================================================
# 5. ESTIMATE MODEL AND CALCULATE PRICING METRICS
# ==================================================

def calculate_metrics(Y, F):

    T, N = Y.shape

    K = F.shape[1]

    # Add intercept.

    X = np.column_stack([
        np.ones(T),
        F
    ])

    if np.linalg.matrix_rank(X) < K + 1:

        raise ValueError(
            "Factor design matrix is rank deficient."
        )

    # Estimate all industry regressions.

    coefficients = np.linalg.lstsq(
        X,
        Y,
        rcond=None
    )[0]

    # Extract industry alphas.

    alphas = coefficients[0, :]

    # Calculate fitted values and residuals.

    fitted = X @ coefficients

    residuals = Y - fitted

    # --------------------------------------------------
    # A. AVERAGE ABSOLUTE ALPHA
    # --------------------------------------------------

    average_absolute_alpha = (
        np.mean(np.abs(alphas)) * 100
    )

    # --------------------------------------------------
    # B. RMS ALPHA
    # --------------------------------------------------

    rms_alpha = (
        np.sqrt(np.mean(alphas ** 2)) * 100
    )

    # --------------------------------------------------
    # C. AVERAGE RMSE
    # --------------------------------------------------

    rmse = np.sqrt(
        np.mean(residuals ** 2, axis=0)
    )

    average_rmse = np.mean(rmse) * 100

    # --------------------------------------------------
    # D. AVERAGE ADJUSTED R-SQUARED
    # --------------------------------------------------

    sst = np.sum(
        (Y - Y.mean(axis=0)) ** 2,
        axis=0
    )

    ssr = np.sum(
        residuals ** 2,
        axis=0
    )

    r_squared = 1 - ssr / sst

    adjusted_r_squared = (
        1
        - (1 - r_squared)
        * (T - 1)
        / (T - K - 1)
    )

    average_adjusted_r_squared = (
        np.mean(adjusted_r_squared) * 100
    )

    return {

        "Average_absolute_alpha_pct":
            average_absolute_alpha,

        "RMS_alpha_pct":
            rms_alpha,

        "Average_RMSE_pct":
            average_rmse,

        "Average_adjusted_R2_pct":
            average_adjusted_r_squared

    }


# ==================================================
# 6. CALCULATE FF6 IMPROVEMENT OVER FF5
# ==================================================

def calculate_improvements(metrics5, metrics6):

    return {

        # Positive means FF6 has lower average absolute alpha.

        "Average_absolute_alpha_improvement":

            metrics5["Average_absolute_alpha_pct"]
            - metrics6["Average_absolute_alpha_pct"],


        # Positive means FF6 has lower RMS alpha.

        "RMS_alpha_improvement":

            metrics5["RMS_alpha_pct"]
            - metrics6["RMS_alpha_pct"],


        # Positive means FF6 has lower RMSE.

        "Average_RMSE_improvement":

            metrics5["Average_RMSE_pct"]
            - metrics6["Average_RMSE_pct"],


        # Positive means FF6 has higher adjusted R-squared.

        "Adjusted_R2_improvement":

            metrics6["Average_adjusted_R2_pct"]
            - metrics5["Average_adjusted_R2_pct"]

    }


# ==================================================
# 7. BOOTSTRAP MODEL COMPARISON
# ==================================================

def run_comparison(
    Y,
    F5,
    F6,
    sample_name,
    block_length
):

    T, N = Y.shape

    # Calculate original sample metrics.

    original5 = calculate_metrics(
        Y,
        F5
    )

    original6 = calculate_metrics(
        Y,
        F6
    )

    # Original FF6 improvements.

    original_improvements = calculate_improvements(
        original5,
        original6
    )

    # Storage for bootstrap estimates.

    bootstrap_results = {

        metric: np.empty(B)

        for metric in original_improvements

    }

    print(
        "\nSample:",
        sample_name,
        "| Block length:",
        block_length
    )

    # --------------------------------------------------
    # BOOTSTRAP REPETITIONS
    # --------------------------------------------------

    for b in range(B):

        # Generate common monthly indices.

        indices = stationary_indices(
            T,
            block_length,
            rng
        )

        # Resample returns and both factor sets
        # using exactly the same monthly indices.

        Y_boot = Y[indices, :]

        F5_boot = F5[indices, :]

        F6_boot = F6[indices, :]

        # Re-estimate FF5.

        metrics5 = calculate_metrics(
            Y_boot,
            F5_boot
        )

        # Re-estimate FF6.

        metrics6 = calculate_metrics(
            Y_boot,
            F6_boot
        )

        # Calculate paired improvement.

        improvements = calculate_improvements(
            metrics5,
            metrics6
        )

        # Store bootstrap results.

        for metric in bootstrap_results:

            bootstrap_results[metric][b] = (
                improvements[metric]
            )

    # ==================================================
    # 8. CONFIDENCE INTERVALS AND P-VALUES
    # ==================================================

    metric_source = {

        "Average_absolute_alpha_improvement":
            "Average_absolute_alpha_pct",

        "RMS_alpha_improvement":
            "RMS_alpha_pct",

        "Average_RMSE_improvement":
            "Average_RMSE_pct",

        "Adjusted_R2_improvement":
            "Average_adjusted_R2_pct"

    }

    rows = []

    for metric, bootstrap_values in bootstrap_results.items():

        # Original-sample improvement.

        estimate = original_improvements[metric]

        # Bootstrap percentile confidence interval.

        lower = np.quantile(
            bootstrap_values,
            0.025
        )

        upper = np.quantile(
            bootstrap_values,
            0.975
        )

        # Center bootstrap distribution under the null.

        centered = (
            bootstrap_values - estimate
        )

        # Two-sided centered bootstrap p-value.

        p_value = (
            1 + np.sum(
                np.abs(centered) >= abs(estimate)
            )
        ) / (B + 1)

        # Get the correct original model metric names.

        source_metric = metric_source[metric]

        # Store comparison results.

        rows.append({

            "Sample": sample_name,

            "Block_length": block_length,

            "Metric": metric,

            "FF5_estimate": original5[source_metric],

            "FF6_estimate": original6[source_metric],

            "FF6_improvement": estimate,

            "CI_95_lower": lower,

            "CI_95_upper": upper,

            "Bootstrap_p": p_value,

            "Bootstrap_repetitions": B

        })

    return rows


# ==================================================
# 9. RUN ALL SAMPLES AND BLOCK LENGTHS
# ==================================================

all_results = []

for sample_name, returns_file in SAMPLES.items():

    # Load the selected sample.

    Y, F5, F6 = load_data(
        returns_file
    )

    print("\n" + "=" * 65)

    print("SAMPLE:", sample_name)

    print("Observations:", Y.shape[0])

    print("Industries:", Y.shape[1])

    print("=" * 65)

    # Run bootstrap with three block lengths.

    for block_length in BLOCK_LENGTHS:

        rows = run_comparison(
            Y,
            F5,
            F6,
            sample_name,
            block_length
        )

        all_results.extend(rows)


# ==================================================
# 10. SAVE RESULTS
# ==================================================

results_df = pd.DataFrame(all_results)

output_file = (
    TABLE_DIR /
    "industry_ff5_ff6_bootstrap_comparison.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ==================================================
# 11. DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 75)

print("FF5 VS FF6 BOOTSTRAP COMPARISON")

print("=" * 75)

print(
    results_df.round(5).to_string(index=False)
)


# ==================================================
# 12. DISPLAY PRIMARY COMPARISON
# ==================================================

print("\n" + "=" * 75)

print("PRIMARY METRIC: AVERAGE ABSOLUTE ALPHA")

print("=" * 75)

primary_results = results_df[
    results_df["Metric"]
    == "Average_absolute_alpha_improvement"
]

print(
    primary_results.round(5).to_string(index=False)
)


# ==================================================
# 13. COMPLETION
# ==================================================

print("\nRESULTS SAVED")

print(output_file)

print("\nBOOTSTRAP REPETITIONS:", B)

print("BLOCK LENGTHS:", BLOCK_LENGTHS)

print("\nMODEL COMPARISON COMPLETED")
