
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

B = 4999

BLOCK_LENGTHS = [6, 12, 24]

SEED = 1402

rng = np.random.default_rng(SEED)

COMMON_START = "1969-07"
COMMON_END = "2025-12"


# ==================================================
# 2. DATASET CONFIGURATION
# ==================================================

DATASETS = {

    # Main matched-period comparisons.

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

    # Additional extended historical robustness sample.

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
# 3. METRIC SPECIFICATION
# ==================================================

METRIC_SOURCE = {

    "Average_absolute_alpha_improvement":
        "Average_absolute_alpha_pct",

    "RMS_alpha_improvement":
        "RMS_alpha_pct",

    "Average_RMSE_improvement":
        "Average_RMSE_pct",

    "Adjusted_R2_improvement":
        "Average_adjusted_R2_pct"

}


# ==================================================
# 4. LOAD AND VALIDATE DATA
# ==================================================

def load_data(returns_file, window):

    returns_path = DATA_DIR / returns_file

    if not returns_path.exists():

        available_files = [
            file.name
            for file in DATA_DIR.glob("*.csv")
        ]

        raise FileNotFoundError(
            "\nCould not find: "
            + returns_file
            + "\nAvailable processed CSV files:\n"
            + "\n".join(available_files)
            + "\n\nCheck the filename in DATASETS."
        )

    # Load returns and factor files.

    returns = pd.read_csv(
        returns_path,
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

    # Convert indices to monthly periods.

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

    # Sort all datasets.

    returns = returns.sort_index()

    ff5 = ff5.sort_index()

    ff6 = ff6.sort_index()

    # Check duplicate dates.

    if returns.index.has_duplicates:
        raise ValueError("Duplicate dates in portfolio returns.")

    if ff5.index.has_duplicates:
        raise ValueError("Duplicate dates in FF5 factors.")

    if ff6.index.has_duplicates:
        raise ValueError("Duplicate dates in FF6 factors.")

    # Check required factor columns.

    missing_ff5 = [
        factor for factor in FF5_FACTORS
        if factor not in ff5.columns
    ]

    missing_ff6 = [
        factor for factor in FF6_FACTORS
        if factor not in ff6.columns
    ]

    if missing_ff5 or missing_ff6:

        raise ValueError(
            "Required factor columns are missing."
        )

    ff5 = ff5[FF5_FACTORS].copy()

    ff6 = ff6[FF6_FACTORS].copy()

    # Find dates common to all three datasets.

    common_dates = (
        returns.index
        .intersection(ff5.index)
        .intersection(ff6.index)
    )

    returns = returns.loc[common_dates]

    ff5 = ff5.loc[common_dates]

    ff6 = ff6.loc[common_dates]

    # Restrict the three main test-asset groups
    # to the same estimation period.

    if window == "matched":

        start = pd.Period(COMMON_START, freq="M")

        end = pd.Period(COMMON_END, freq="M")

        selected_dates = (
            (returns.index >= start)
            & (returns.index <= end)
        )

        returns = returns.loc[selected_dates]

        ff5 = ff5.loc[selected_dates]

        ff6 = ff6.loc[selected_dates]

    # Confirm common factors are identical.

    if not np.allclose(
        ff5.to_numpy(),
        ff6[FF5_FACTORS].to_numpy(),
        rtol=0,
        atol=1e-10
    ):

        raise ValueError(
            "Common FF5 factors differ between factor files."
        )

    # Convert to numerical arrays.

    Y = returns.to_numpy(dtype=float)

    F5 = ff5.to_numpy(dtype=float)

    F6 = ff6.to_numpy(dtype=float)

    # Check for invalid observations.

    if not np.isfinite(Y).all():
        raise ValueError("Missing or invalid portfolio returns.")

    if not np.isfinite(F5).all():
        raise ValueError("Missing or invalid FF5 factors.")

    if not np.isfinite(F6).all():
        raise ValueError("Missing or invalid FF6 factors.")

    return Y, F5, F6, returns.index


# ==================================================
# 5. STATIONARY BLOCK BOOTSTRAP INDICES
# ==================================================

def stationary_indices(T, mean_block_length):

    indices = np.empty(T, dtype=int)

    # Choose the first observation randomly.

    indices[0] = rng.integers(0, T)

    # Probability of starting a new block.

    restart_probability = 1 / mean_block_length

    for t in range(1, T):

        if rng.random() < restart_probability:

            indices[t] = rng.integers(0, T)

        else:

            indices[t] = (
                indices[t - 1] + 1
            ) % T

    return indices


# ==================================================
# 6. CALCULATE MODEL PERFORMANCE
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
            "Factor matrix is rank deficient."
        )

    # Estimate all portfolio regressions.

    coefficients = np.linalg.lstsq(
        X,
        Y,
        rcond=None
    )[0]

    # Extract intercepts.

    alphas = coefficients[0, :]

    # Calculate residuals.

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
# 7. CALCULATE FF6 IMPROVEMENT
# ==================================================

def calculate_improvements(metrics5, metrics6):

    improvements = {

        # Positive means FF6 has lower pricing errors.

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

    return improvements


# ==================================================
# 8. RUN PAIRED BOOTSTRAP COMPARISON
# ==================================================

def run_comparison(
    Y,
    F5,
    F6,
    dataset_name,
    block_length
):

    T, N = Y.shape

    # Original sample estimates.

    original5 = calculate_metrics(
        Y,
        F5
    )

    original6 = calculate_metrics(
        Y,
        F6
    )

    original_improvements = calculate_improvements(
        original5,
        original6
    )

    # Store bootstrap estimates.

    bootstrap_results = {

        metric: np.empty(B)

        for metric in original_improvements

    }

    print(
        "\nDataset:",
        dataset_name,
        "| Block length:",
        block_length
    )

    # --------------------------------------------------
    # BOOTSTRAP LOOP
    # --------------------------------------------------

    for b in range(B):

        # Generate a common sample of monthly indices.

        indices = stationary_indices(
            T,
            block_length
        )

        # Resample returns and factors together.

        Y_boot = Y[indices, :]

        F5_boot = F5[indices, :]

        F6_boot = F6[indices, :]

        # Estimate FF5 and FF6.

        metrics5 = calculate_metrics(
            Y_boot,
            F5_boot
        )

        metrics6 = calculate_metrics(
            Y_boot,
            F6_boot
        )

        # Calculate relative improvement.

        improvements = calculate_improvements(
            metrics5,
            metrics6
        )

        for metric in bootstrap_results:

            bootstrap_results[metric][b] = (
                improvements[metric]
            )

        # Progress information.

        if (b + 1) % 1000 == 0:

            print(
                "Completed:",
                b + 1,
                "of",
                B
            )

    # ==================================================
    # 9. BASIC BOOTSTRAP CONFIDENCE INTERVALS
    # ==================================================

    rows = []

    for metric, bootstrap_values in bootstrap_results.items():

        estimate = original_improvements[metric]

        # Bootstrap quantiles.

        q025 = np.quantile(
            bootstrap_values,
            0.025
        )

        q975 = np.quantile(
            bootstrap_values,
            0.975
        )

        # Basic bootstrap confidence interval.
        #
        # This interval corresponds to the
        # centered bootstrap p-value below.

        lower = 2 * estimate - q975

        upper = 2 * estimate - q025

        # Bootstrap standard error.

        bootstrap_se = np.std(
            bootstrap_values,
            ddof=1
        )

        # Center bootstrap distribution under H0.

        centered = (
            bootstrap_values - estimate
        )

        # Two-sided centered bootstrap p-value.

        p_value = (
            1 + np.sum(
                np.abs(centered) >= abs(estimate)
            )
        ) / (B + 1)

        source_metric = METRIC_SOURCE[metric]

        rows.append({

            "Dataset": dataset_name,

            "Block_length": block_length,

            "Metric": metric,

            "FF5_estimate": original5[source_metric],

            "FF6_estimate": original6[source_metric],

            "FF6_improvement": estimate,

            "Bootstrap_SE": bootstrap_se,

            "CI_95_lower_basic": lower,

            "CI_95_upper_basic": upper,

            "Bootstrap_p": p_value,

            "Bootstrap_repetitions": B

        })

    return rows


# ==================================================
# 10. RUN ALL DATASETS
# ==================================================

all_results = []

for dataset_name, specification in DATASETS.items():

    returns_file = specification["file"]

    window = specification["window"]

    # Load the dataset.

    Y, F5, F6, dates = load_data(
        returns_file,
        window
    )

    T, N = Y.shape

    print("\n" + "=" * 70)

    print("DATASET:", dataset_name)

    print("File:", returns_file)

    print("Observations:", T)

    print("Portfolios:", N)

    print("First month:", dates.min())

    print("Last month:", dates.max())

    print("=" * 70)

    # Run all three bootstrap block lengths.

    for block_length in BLOCK_LENGTHS:

        rows = run_comparison(
            Y,
            F5,
            F6,
            dataset_name,
            block_length
        )

        all_results.extend(rows)


# ==================================================
# 11. COMBINE ALL RESULTS
# ==================================================

results_df = pd.DataFrame(all_results)


# ==================================================
# 12. CREATE PRIMARY METRIC TABLE
# ==================================================

primary_results = results_df[
    results_df["Metric"]
    == "Average_absolute_alpha_improvement"
].copy()


# ==================================================
# 13. CREATE CONSOLIDATED COMPARISON
# ==================================================

consolidated = primary_results[[
    "Dataset",
    "Block_length",
    "FF5_estimate",
    "FF6_estimate",
    "FF6_improvement",
    "Bootstrap_SE",
    "CI_95_lower_basic",
    "CI_95_upper_basic",
    "Bootstrap_p",
    "Bootstrap_repetitions"
]].copy()


consolidated = consolidated.sort_values(
    ["Dataset", "Block_length"]
)


# ==================================================
# 14. SAVE OUTPUT FILES
# ==================================================

results_df.to_csv(
    TABLE_DIR / "all_asset_bootstrap_results.csv",
    index=False
)

primary_results.to_csv(
    TABLE_DIR / "all_asset_primary_metric.csv",
    index=False
)

consolidated.to_csv(
    TABLE_DIR / "all_asset_consolidated_comparison.csv",
    index=False
)


# ==================================================
# 15. DISPLAY FULL RESULTS
# ==================================================

print("\n" + "=" * 85)

print("ALL ASSET-PRICING MODEL COMPARISON RESULTS")

print("=" * 85)

print(
    results_df.round(5).to_string(index=False)
)


# ==================================================
# 16. DISPLAY PRIMARY COMPARISON
# ==================================================

print("\n" + "=" * 85)

print("PRIMARY METRIC: AVERAGE ABSOLUTE ALPHA")

print("=" * 85)

print(
    consolidated.round(5).to_string(index=False)
)


# ==================================================
# 17. DISPLAY MAIN THREE-GROUP COMPARISON
# ==================================================

main_groups = consolidated[
    consolidated["Dataset"].isin([
        "Size-BM",
        "Size-Momentum",
        "Industry-49"
    ])
]

print("\n" + "=" * 85)

print("MAIN COMPARISON: COMMON SAMPLE JULY 1969 - DECEMBER 2025")

print("=" * 85)

print(
    main_groups.round(5).to_string(index=False)
)


# ==================================================
# 18. COMPLETION
# ==================================================

print("\nFILES SAVED")

print("all_asset_bootstrap_results.csv")

print("all_asset_primary_metric.csv")

print("all_asset_consolidated_comparison.csv")

print("\nBOOTSTRAP REPETITIONS:", B)

print("BLOCK LENGTHS:", BLOCK_LENGTHS)

print("\nALL ASSET COMPARISONS COMPLETED")
