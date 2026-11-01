
from pathlib import Path

import numpy as np
import pandas as pd

from statsmodels.stats.multitest import multipletests


# ==================================================
# 1. SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data_processed"

TABLE_DIR = BASE_DIR / "results" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

B = 4999

BLOCK_LENGTHS = [6, 12, 24]

SEED = 1403

rng = np.random.default_rng(SEED)


# ==================================================
# 2. DATASET CONFIGURATION
# ==================================================

DATASETS = {

    "Size-BM": "size_bm_excess.csv",

    "Size-Momentum": "size_momentum_excess.csv",

    "Industry-49": "industry_49_excess.csv"

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

def load_data():

    datasets = {}

    # Load FF5 and FF6 factors.

    ff5 = pd.read_csv(
        DATA_DIR / "factors_ff5.csv",
        index_col=0
    )

    ff6 = pd.read_csv(
        DATA_DIR / "factors_ff6.csv",
        index_col=0
    )

    ff5.index = pd.PeriodIndex(
        ff5.index.astype(str),
        freq="M"
    )

    ff6.index = pd.PeriodIndex(
        ff6.index.astype(str),
        freq="M"
    )

    ff5.columns = ff5.columns.str.strip()

    ff6.columns = ff6.columns.str.strip()

    ff5 = ff5.sort_index()

    ff6 = ff6.sort_index()

    ff5 = ff5[FF5_FACTORS]

    ff6 = ff6[FF6_FACTORS]

    if ff5.index.has_duplicates or ff6.index.has_duplicates:
        raise ValueError("Duplicate factor dates.")

    common_factor_dates = ff5.index.intersection(
        ff6.index
    )

    ff5 = ff5.loc[common_factor_dates]

    ff6 = ff6.loc[common_factor_dates]

    if not np.allclose(
        ff5.to_numpy(),
        ff6[FF5_FACTORS].to_numpy(),
        rtol=0,
        atol=1e-10
    ):

        raise ValueError(
            "FF5 and FF6 common factors do not match."
        )

    # Restrict factors to common estimation period.

    start = pd.Period("1969-07", freq="M")

    end = pd.Period("2025-12", freq="M")

    selected = (
        (ff5.index >= start)
        & (ff5.index <= end)
    )

    ff5 = ff5.loc[selected]

    ff6 = ff6.loc[selected]

    # Load each test-asset group.

    for name, filename in DATASETS.items():

        returns = pd.read_csv(
            DATA_DIR / filename,
            index_col=0
        )

        returns.index = pd.PeriodIndex(
            returns.index.astype(str),
            freq="M"
        )

        returns.columns = returns.columns.str.strip()

        returns = returns.sort_index()

        if returns.index.has_duplicates:
            raise ValueError(
                "Duplicate dates in " + name
            )

        returns = returns.loc[
            (returns.index >= start)
            & (returns.index <= end)
        ]

        if not returns.index.equals(ff5.index):

            raise ValueError(
                "The dates do not match for " + name
            )

        Y = returns.to_numpy(dtype=float)

        if not np.isfinite(Y).all():

            raise ValueError(
                "Invalid returns in " + name
            )

        datasets[name] = Y

    F5 = ff5.to_numpy(dtype=float)

    F6 = ff6.to_numpy(dtype=float)

    if not np.isfinite(F5).all():
        raise ValueError("Invalid FF5 factors.")

    if not np.isfinite(F6).all():
        raise ValueError("Invalid FF6 factors.")

    print("\nCOMMON SAMPLE VALIDATION")

    print("First month:", ff5.index.min())

    print("Last month:", ff5.index.max())

    print("Observations:", len(ff5.index))

    for name, Y in datasets.items():

        print(name, "| Portfolios:", Y.shape[1])

    return datasets, F5, F6


# ==================================================
# 4. CALCULATE AVERAGE ABSOLUTE ALPHA
# ==================================================

def average_absolute_alpha(Y, F):

    T, N = Y.shape

    K = F.shape[1]

    X = np.column_stack([
        np.ones(T),
        F
    ])

    coefficients = np.linalg.lstsq(
        X,
        Y,
        rcond=None
    )[0]

    alphas = coefficients[0, :]

    return np.mean(np.abs(alphas)) * 100


# ==================================================
# 5. CALCULATE FF6 IMPROVEMENT
# ==================================================

def calculate_improvement(Y, F5, F6):

    alpha5 = average_absolute_alpha(
        Y,
        F5
    )

    alpha6 = average_absolute_alpha(
        Y,
        F6
    )

    return alpha5 - alpha6


# ==================================================
# 6. STATIONARY BLOCK BOOTSTRAP INDICES
# ==================================================

def stationary_indices(T, mean_block_length):

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
# 7. CALCULATE CONFIDENCE INTERVAL AND P-VALUE
# ==================================================

def bootstrap_summary(estimate, bootstrap_values):

    q025 = np.quantile(
        bootstrap_values,
        0.025
    )

    q975 = np.quantile(
        bootstrap_values,
        0.975
    )

    # Basic bootstrap confidence interval.

    lower = 2 * estimate - q975

    upper = 2 * estimate - q025

    bootstrap_se = np.std(
        bootstrap_values,
        ddof=1
    )

    # Center bootstrap distribution under the null.

    centered = bootstrap_values - estimate

    p_value = (
        1 + np.sum(
            np.abs(centered) >= abs(estimate)
        )
    ) / (len(bootstrap_values) + 1)

    return {

        "Estimate": estimate,

        "Bootstrap_SE": bootstrap_se,

        "CI_95_lower": lower,

        "CI_95_upper": upper,

        "Bootstrap_p": p_value

    }


# ==================================================
# 8. ORIGINAL SAMPLE IMPROVEMENTS
# ==================================================

def run_heterogeneity_bootstrap(
    datasets,
    F5,
    F6,
    block_length
):

    T = F5.shape[0]

    # Store original FF6 improvements.

    original_improvements = {}

    for name, Y in datasets.items():

        original_improvements[name] = (
            calculate_improvement(
                Y,
                F5,
                F6
            )
        )

    # Bootstrap storage.

    bootstrap_improvements = {

        name: np.empty(B)

        for name in datasets

    }

    pairwise_bootstrap = {

        "Size-Momentum minus Size-BM": np.empty(B),

        "Size-Momentum minus Industry-49": np.empty(B),

        "Industry-49 minus Size-BM": np.empty(B)

    }

    print(
        "\nBlock length:",
        block_length
    )

    # --------------------------------------------------
    # BOOTSTRAP LOOP
    # --------------------------------------------------

    for b in range(B):

        # Generate one common set of monthly indices.

        indices = stationary_indices(
            T,
            block_length
        )

        F5_boot = F5[indices, :]

        F6_boot = F6[indices, :]

        # Apply the exact same indices to all portfolios.

        for name, Y in datasets.items():

            Y_boot = Y[indices, :]

            bootstrap_improvements[name][b] = (
                calculate_improvement(
                    Y_boot,
                    F5_boot,
                    F6_boot
                )
            )

        # Calculate differences in improvements.

        size_bm = bootstrap_improvements[
            "Size-BM"
        ][b]

        size_momentum = bootstrap_improvements[
            "Size-Momentum"
        ][b]

        industry = bootstrap_improvements[
            "Industry-49"
        ][b]

        pairwise_bootstrap[
            "Size-Momentum minus Size-BM"
        ][b] = size_momentum - size_bm

        pairwise_bootstrap[
            "Size-Momentum minus Industry-49"
        ][b] = size_momentum - industry

        pairwise_bootstrap[
            "Industry-49 minus Size-BM"
        ][b] = industry - size_bm

        if (b + 1) % 1000 == 0:

            print(
                "Completed:",
                b + 1,
                "of",
                B
            )

    # ==================================================
    # 9. INDIVIDUAL GROUP IMPROVEMENTS
    # ==================================================

    group_results = []

    for name in datasets:

        estimate = original_improvements[name]

        result = bootstrap_summary(
            estimate,
            bootstrap_improvements[name]
        )

        result["Dataset"] = name

        result["Block_length"] = block_length

        group_results.append(result)

    # ==================================================
    # 10. PAIRWISE HETEROGENEITY TESTS
    # ==================================================

    pairwise_results = []

    for comparison, bootstrap_values in pairwise_bootstrap.items():

        group1, group2 = comparison.split(" minus ")

        estimate = (
            original_improvements[group1]
            - original_improvements[group2]
        )

        result = bootstrap_summary(
            estimate,
            bootstrap_values
        )

        result["Comparison"] = comparison

        result["Block_length"] = block_length

        pairwise_results.append(result)

    return group_results, pairwise_results


# ==================================================
# 11. RUN ALL BLOCK LENGTHS
# ==================================================

datasets, F5, F6 = load_data()

all_group_results = []

all_pairwise_results = []


for block_length in BLOCK_LENGTHS:

    print("\n" + "=" * 70)

    print(
        "CROSS-ASSET HETEROGENEITY TEST:",
        block_length,
        "MONTH BLOCKS"
    )

    print("=" * 70)

    group_results, pairwise_results = (
        run_heterogeneity_bootstrap(
            datasets,
            F5,
            F6,
            block_length
        )
    )

    all_group_results.extend(group_results)

    all_pairwise_results.extend(pairwise_results)


# ==================================================
# 12. MULTIPLE-TESTING ADJUSTMENT
# ==================================================

pairwise_df = pd.DataFrame(
    all_pairwise_results
)

pairwise_df["Holm_adjusted_p"] = np.nan

# Adjust the three planned pairwise comparisons
# separately for each block length.

for block_length, group in pairwise_df.groupby(
    "Block_length"
):

    _, adjusted_p, _, _ = multipletests(
        group["Bootstrap_p"],
        alpha=0.05,
        method="holm"
    )

    pairwise_df.loc[
        group.index,
        "Holm_adjusted_p"
    ] = adjusted_p


# ==================================================
# 13. SAVE RESULTS
# ==================================================

group_df = pd.DataFrame(
    all_group_results
)

group_df.to_csv(
    TABLE_DIR / "cross_asset_group_improvements.csv",
    index=False
)

pairwise_df.to_csv(
    TABLE_DIR / "cross_asset_heterogeneity_tests.csv",
    index=False
)


# ==================================================
# 14. DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 80)

print("INDIVIDUAL GROUP IMPROVEMENTS")

print("=" * 80)

print(
    group_df.round(5).to_string(index=False)
)


print("\n" + "=" * 80)

print("PAIRWISE CROSS-ASSET HETEROGENEITY TESTS")

print("=" * 80)

print(
    pairwise_df.round(5).to_string(index=False)
)


# ==================================================
# 15. MAIN 12-MONTH RESULTS
# ==================================================

print("\n" + "=" * 80)

print("PRIMARY HETEROGENEITY RESULTS: 12-MONTH BLOCKS")

print("=" * 80)

main_results = pairwise_df[
    pairwise_df["Block_length"] == 12
]

print(
    main_results.round(5).to_string(index=False)
)


# ==================================================
# 16. COMPLETION
# ==================================================

print("\nFILES SAVED")

print("cross_asset_group_improvements.csv")

print("cross_asset_heterogeneity_tests.csv")

print("\nBOOTSTRAP REPETITIONS:", B)

print("BLOCK LENGTHS:", BLOCK_LENGTHS)

print("\nCROSS-ASSET HETEROGENEITY ANALYSIS COMPLETED")
