
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

SEED = 1400

rng = np.random.default_rng(SEED)


# ==================================================
# 2. MODEL AND SAMPLE SPECIFICATIONS
# ==================================================

SAMPLES = {
    "49_industry_main": "industry_49_excess.csv",
    "47_industry_extended": "industry_47_excess.csv"
}

MODELS = {
    "FF5": {
        "file": "factors_ff5.csv",
        "factors": [
            "Mkt-RF", "SMB", "HML", "RMW", "CMA"
        ]
    },

    "FF6": {
        "file": "factors_ff6.csv",
        "factors": [
            "Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"
        ]
    }
}


# ==================================================
# 3. LOAD DATA
# ==================================================

def load_data(returns_file, factor_file, factor_names):

    returns = pd.read_csv(
        DATA_DIR / returns_file,
        index_col=0
    )

    factors = pd.read_csv(
        DATA_DIR / factor_file,
        index_col=0
    )

    returns.index = pd.PeriodIndex(
        returns.index.astype(str),
        freq="M"
    )

    factors.index = pd.PeriodIndex(
        factors.index.astype(str),
        freq="M"
    )

    returns.columns = returns.columns.str.strip()
    factors.columns = factors.columns.str.strip()

    returns = returns.sort_index()
    factors = factors.sort_index()

    factors = factors[factor_names]

    common_dates = returns.index.intersection(
        factors.index
    )

    if len(common_dates) != len(returns):

        raise ValueError(
            "Return and factor dates do not match."
        )

    returns = returns.loc[common_dates]
    factors = factors.loc[common_dates]

    Y = returns.to_numpy(dtype=float)
    F = factors.to_numpy(dtype=float)

    if not np.isfinite(Y).all():
        raise ValueError("Invalid industry returns.")

    if not np.isfinite(F).all():
        raise ValueError("Invalid factor observations.")

    return Y, F


# ==================================================
# 4. FAMA-MACBETH ESTIMATION
# ==================================================

def estimate_fmb(Y, F):

    T, N = Y.shape

    K = F.shape[1]

    # First pass: estimate betas for each industry.

    X_first = np.column_stack([
        np.ones(T),
        F
    ])

    coefficients = np.linalg.lstsq(
        X_first,
        Y,
        rcond=None
    )[0]

    betas = coefficients[1:, :].T

    # Second pass: estimate monthly risk premiums.

    X_second = np.column_stack([
        np.ones(N),
        betas
    ])

    if np.linalg.matrix_rank(X_second) < K + 1:

        raise ValueError(
            "Second-pass beta matrix is rank deficient."
        )

    # Each row of lambdas contains one month's
    # cross-sectional regression coefficients.

    inverse = np.linalg.pinv(X_second)

    monthly_lambdas = Y @ inverse.T

    average_lambdas = monthly_lambdas.mean(axis=0)

    return average_lambdas


# ==================================================
# 5. STATIONARY BLOCK BOOTSTRAP INDICES
# ==================================================

def stationary_indices(T, mean_block_length, rng):

    indices = np.empty(T, dtype=int)

    indices[0] = rng.integers(0, T)

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
# 6. RUN BOOTSTRAP
# ==================================================

def run_bootstrap(
    Y,
    F,
    sample_name,
    model_name,
    factor_names,
    block_length
):

    T, N = Y.shape

    K = F.shape[1]

    coefficient_names = ["const"] + factor_names

    # Original Fama-MacBeth estimate.

    point_estimate = estimate_fmb(Y, F)

    bootstrap_estimates = np.empty(
        (B, K + 1)
    )

    print(
        "\nRunning:",
        sample_name,
        model_name,
        "Block length:",
        block_length
    )

    for b in range(B):

        # Generate a common block-bootstrap sample.

        indices = stationary_indices(
            T,
            block_length,
            rng
        )

        Y_boot = Y[indices, :]

        F_boot = F[indices, :]

        # Re-estimate both Fama-MacBeth stages.

        bootstrap_estimates[b, :] = estimate_fmb(
            Y_boot,
            F_boot
        )

    # Bootstrap standard errors.

    bootstrap_se = np.std(
        bootstrap_estimates,
        axis=0,
        ddof=1
    )

    # Percentile confidence intervals.

    lower_ci = np.quantile(
        bootstrap_estimates,
        0.025,
        axis=0
    )

    upper_ci = np.quantile(
        bootstrap_estimates,
        0.975,
        axis=0
    )

    # Two-sided centered bootstrap p-values.
    #
    # Compare bootstrap deviations from the original
    # estimate with the distance of the estimate from zero.

    centered_deviations = (
        bootstrap_estimates - point_estimate
    )

    p_values = (
        1
        + np.sum(
            np.abs(centered_deviations)
            >= np.abs(point_estimate),
            axis=0
        )
    ) / (B + 1)

    results = []

    for j, coefficient in enumerate(coefficient_names):

        results.append({

            "Sample": sample_name,

            "Model": model_name,

            "Coefficient": coefficient,

            "Block_length": block_length,

            "Bootstrap_repetitions": B,

            "Estimate_pct": point_estimate[j] * 100,

            "Bootstrap_SE_pct": bootstrap_se[j] * 100,

            "CI_95_lower_pct": lower_ci[j] * 100,

            "CI_95_upper_pct": upper_ci[j] * 100,

            "Bootstrap_p": p_values[j]

        })

    return results


# ==================================================
# 7. RUN ALL SAMPLES, MODELS AND BLOCK LENGTHS
# ==================================================

all_results = []

for sample_name, returns_file in SAMPLES.items():

    for model_name, specification in MODELS.items():

        factor_names = specification["factors"]

        Y, F = load_data(
            returns_file,
            specification["file"],
            factor_names
        )

        print("\n" + "=" * 65)

        print("SAMPLE:", sample_name)

        print("MODEL:", model_name)

        print("Observations:", Y.shape[0])

        print("Industries:", Y.shape[1])

        print("=" * 65)

        for block_length in BLOCK_LENGTHS:

            results = run_bootstrap(
                Y,
                F,
                sample_name,
                model_name,
                factor_names,
                block_length
            )

            all_results.extend(results)


# ==================================================
# 8. SAVE RESULTS
# ==================================================

results_df = pd.DataFrame(all_results)

output_file = (
    TABLE_DIR /
    "industry_fama_macbeth_stationary_bootstrap.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ==================================================
# 9. DISPLAY MOMENTUM RESULTS
# ==================================================

momentum_results = results_df[
    results_df["Coefficient"] == "MOM"
]

print("\n" + "=" * 75)

print("INDUSTRY MOMENTUM BOOTSTRAP RESULTS")

print("=" * 75)

print(
    momentum_results.to_string(index=False)
)


print("\nALL BOOTSTRAP RESULTS SAVED")

print(output_file)

print("\nSTATIONARY BLOCK BOOTSTRAP COMPLETED")
