
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ==================================================
# 1. PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data_processed"

TABLE_DIR = BASE_DIR / "results" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

HAC_LAGS = [3, 6, 12]


# ==================================================
# 2. LOAD DATA
# ==================================================

def load_data(returns_file):

    returns = pd.read_csv(
        DATA_DIR / returns_file,
        index_col=0
    )

    factors = pd.read_csv(
        DATA_DIR / "factors_ff6.csv",
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

    factors = factors[
        ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    ]

    common_dates = returns.index.intersection(
        factors.index
    )

    returns = returns.loc[common_dates]
    factors = factors.loc[common_dates]

    if returns.isna().any().any():
        raise ValueError("Missing industry returns.")

    if factors.isna().any().any():
        raise ValueError("Missing factor observations.")

    return returns, factors


# ==================================================
# 3. FIRST PASS: ESTIMATE INDUSTRY BETAS
# ==================================================

def estimate_betas(returns, factors):

    X = sm.add_constant(factors)

    factor_names = list(factors.columns)

    betas = []

    for industry in returns.columns:

        result = sm.OLS(
            returns[industry],
            X
        ).fit()

        betas.append(
            result.params[factor_names].to_numpy()
        )

    return np.array(betas)


# ==================================================
# 4. SECOND PASS: MONTHLY CROSS-SECTIONAL PREMIUMS
# ==================================================

def estimate_monthly_lambdas(
    returns,
    betas,
    excluded_index=None
):

    Y = returns.to_numpy()

    if excluded_index is None:

        keep = np.arange(Y.shape[1])

    else:

        keep = np.array([
            i for i in range(Y.shape[1])
            if i != excluded_index
        ])

    Y = Y[:, keep]

    beta_subset = betas[keep, :]

    X = np.column_stack([
        np.ones(len(keep)),
        beta_subset
    ])

    monthly_lambdas = []

    for t in range(len(Y)):

        result = np.linalg.lstsq(
            X,
            Y[t, :],
            rcond=None
        )[0]

        monthly_lambdas.append(result)

    return np.array(monthly_lambdas)


# ==================================================
# 5. HAC INFERENCE ON AVERAGE MOMENTUM PREMIUM
# ==================================================

def estimate_hac_premium(momentum_lambdas, lags):

    X = np.ones(
        (len(momentum_lambdas), 1)
    )

    result = sm.OLS(
        momentum_lambdas,
        X
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": lags
        }
    )

    return {

        "Premium_pct": result.params[0] * 100,

        "HAC_SE_pct": result.bse[0] * 100,

        "t_stat": result.tvalues[0],

        "p_value": result.pvalues[0],

        "HAC_lags": lags

    }


# ==================================================
# 6. RUN SENSITIVITY TESTS
# ==================================================

SAMPLES = {

    "49_industry_main": "industry_49_excess.csv",

    "47_industry_extended": "industry_47_excess.csv"

}


all_beta_results = []

all_hac_results = []

all_loo_results = []


for sample_name, returns_file in SAMPLES.items():

    print("\n" + "=" * 65)

    print("SAMPLE:", sample_name)

    print("=" * 65)

    returns, factors = load_data(returns_file)

    industries = list(returns.columns)

    T = len(returns)

    N = len(industries)

    print("Observations:", T)

    print("Industries:", N)


    # First pass

    betas = estimate_betas(
        returns,
        factors
    )

    factor_names = list(factors.columns)


    # Save individual industry MOM betas

    for i, industry in enumerate(industries):

        all_beta_results.append({

            "Sample": sample_name,

            "Industry": industry,

            "MOM_beta": betas[i, -1]

        })


    positive = int((betas[:, -1] > 0).sum())

    negative = int((betas[:, -1] < 0).sum())

    print("\nMOMENTUM BETA DISTRIBUTION")

    print("Positive MOM betas:", positive)

    print("Negative MOM betas:", negative)

    print(
        "MOM beta standard deviation:",
        round(np.std(betas[:, -1], ddof=1), 6)
    )


    # Baseline second pass

    baseline_lambdas = estimate_monthly_lambdas(
        returns,
        betas
    )

    baseline_mom = baseline_lambdas[:, -1]


    # HAC lag sensitivity

    print("\nHAC LAG SENSITIVITY")

    for lag in HAC_LAGS:

        result = estimate_hac_premium(
            baseline_mom,
            lag
        )

        result["Sample"] = sample_name

        all_hac_results.append(result)

        print(
            "Lag:", lag,
            "| Premium (%):", round(result["Premium_pct"], 6),
            "| SE:", round(result["HAC_SE_pct"], 6),
            "| t:", round(result["t_stat"], 4),
            "| p:", round(result["p_value"], 6)
        )


    # Leave-one-industry-out

    print("\nLEAVE-ONE-INDUSTRY-OUT ANALYSIS")

    for i, industry in enumerate(industries):

        loo_lambdas = estimate_monthly_lambdas(
            returns,
            betas,
            excluded_index=i
        )

        loo_mom = loo_lambdas[:, -1]

        result = estimate_hac_premium(
            loo_mom,
            6
        )

        all_loo_results.append({

            "Sample": sample_name,

            "Excluded_industry": industry,

            "Premium_pct": result["Premium_pct"],

            "HAC_SE_pct": result["HAC_SE_pct"],

            "t_stat": result["t_stat"],

            "p_value": result["p_value"]

        })


# ==================================================
# 7. CREATE RESULT TABLES
# ==================================================

beta_df = pd.DataFrame(all_beta_results)

hac_df = pd.DataFrame(all_hac_results)

loo_df = pd.DataFrame(all_loo_results)


# Sort industry momentum betas

beta_df = beta_df.sort_values(
    ["Sample", "MOM_beta"]
)


# ==================================================
# 8. SAVE RESULTS
# ==================================================

beta_df.to_csv(
    TABLE_DIR / "industry_mom_beta_by_industry.csv",
    index=False
)

hac_df.to_csv(
    TABLE_DIR / "industry_mom_hac_sensitivity.csv",
    index=False
)

loo_df.to_csv(
    TABLE_DIR / "industry_mom_leave_one_out.csv",
    index=False
)


# ==================================================
# 9. DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 65)

print("HAC SENSITIVITY SUMMARY")

print("=" * 65)

print(
    hac_df.to_string(index=False)
)


print("\n" + "=" * 65)

print("LEAVE-ONE-INDUSTRY-OUT SUMMARY")

print("=" * 65)


for sample_name, group in loo_df.groupby("Sample"):

    minimum = group.loc[
        group["Premium_pct"].idxmin()
    ]

    maximum = group.loc[
        group["Premium_pct"].idxmax()
    ]

    print("\nSample:", sample_name)

    print(
        "Minimum MOM premium:",
        round(minimum["Premium_pct"], 6),
        "| Excluded:",
        minimum["Excluded_industry"]
    )

    print(
        "Maximum MOM premium:",
        round(maximum["Premium_pct"], 6),
        "| Excluded:",
        maximum["Excluded_industry"]
    )

    print(
        "Average MOM premium:",
        round(group["Premium_pct"].mean(), 6)
    )

    print(
        "Number of positive premiums:",
        int((group["Premium_pct"] > 0).sum())
    )

    print(
        "Number of significant premiums:",
        int((group["p_value"] < 0.05).sum())
    )


print("\nRESULTS SAVED")

print("industry_mom_beta_by_industry.csv")

print("industry_mom_hac_sensitivity.csv")

print("industry_mom_leave_one_out.csv")

print("\nFAMA-MACBETH SENSITIVITY COMPLETED")
