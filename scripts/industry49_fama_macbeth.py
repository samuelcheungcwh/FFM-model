
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ==================================================
# 1. PATHS AND SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data_processed"
TABLE_DIR = BASE_DIR / "results" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

HAC_LAGS = 6


# ==================================================
# 2. DATA LOADING
# ==================================================

def load_returns(filename):

    data = pd.read_csv(
        DATA_DIR / filename,
        index_col=0
    )

    data.index = pd.PeriodIndex(
        data.index.astype(str),
        freq="M"
    )

    data.columns = data.columns.str.strip()

    data = data.sort_index()

    if data.index.has_duplicates:
        raise ValueError("Duplicate return dates.")

    data = data.apply(pd.to_numeric, errors="coerce")

    if data.isna().any().any():
        raise ValueError("Missing portfolio returns.")

    return data


def load_factors(filename, required_factors):

    data = pd.read_csv(
        DATA_DIR / filename,
        index_col=0
    )

    data.index = pd.PeriodIndex(
        data.index.astype(str),
        freq="M"
    )

    data.columns = data.columns.str.strip()

    data = data.sort_index()

    if data.index.has_duplicates:
        raise ValueError("Duplicate factor dates.")

    factor_lookup = {}

    for column in data.columns:
        factor_lookup[column.upper()] = column

    selected_columns = []

    for factor in required_factors:

        key = factor.upper()

        if key not in factor_lookup:
            raise ValueError(
                factor + " not found in " + filename
            )

        selected_columns.append(
            factor_lookup[key]
        )

    data = data[selected_columns].copy()

    data.columns = required_factors

    data = data.apply(pd.to_numeric, errors="coerce")

    if data.isna().any().any():
        raise ValueError("Missing factor observations.")

    return data


# ==================================================
# 3. MODEL AND SAMPLE SPECIFICATIONS
# ==================================================

FF5_FACTORS = [
    "Mkt-RF",
    "SMB",
    "HML",
    "RMW",
    "CMA"
]

FF6_FACTORS = [
    "Mkt-RF",
    "SMB",
    "HML",
    "RMW",
    "CMA",
    "MOM"
]


MODELS = {

    "FF5": {
        "file": "factors_ff5.csv",
        "factors": FF5_FACTORS
    },

    "FF6": {
        "file": "factors_ff6.csv",
        "factors": FF6_FACTORS
    }

}


SAMPLES = {

    "49_industry_main": "industry_49_excess.csv",

    "47_industry_extended": "industry_47_excess.csv"

}


# ==================================================
# 4. FAMA-MACBETH ESTIMATION FUNCTION
# ==================================================

def run_fama_macbeth(
    returns,
    factors,
    sample_name,
    model_name
):

    common_dates = returns.index.intersection(
        factors.index
    )

    Y = returns.loc[common_dates].copy()

    F = factors.loc[common_dates].copy()

    if len(common_dates) != len(returns):
        raise ValueError("Return and factor dates do not match.")

    T = len(Y)

    N = len(Y.columns)

    factor_names = list(F.columns)

    K = len(factor_names)

    print("\nSample:", sample_name)

    print("Model:", model_name)

    print("Observations:", T)

    print("Industries:", N)

    print("Factors:", K)


    # ==================================================
    # FIRST PASS: ESTIMATE INDUSTRY BETAS
    # ==================================================

    print("\nFIRST PASS: TIME-SERIES REGRESSIONS")

    X_first = sm.add_constant(F)

    beta_estimates = pd.DataFrame(
        index=Y.columns,
        columns=factor_names,
        dtype=float
    )

    for industry in Y.columns:

        result = sm.OLS(
            Y[industry],
            X_first
        ).fit()

        beta_estimates.loc[industry] = (
            result.params[factor_names]
        )


    print("First-pass regressions completed:", N)


    # ==================================================
    # BETA DISPERSION AND CONDITIONING
    # ==================================================

    beta_rows = []

    for factor in factor_names:

        beta_rows.append({

            "Sample": sample_name,

            "Model": model_name,

            "Factor": factor,

            "Beta_mean": beta_estimates[factor].mean(),

            "Beta_std": beta_estimates[factor].std(),

            "Beta_min": beta_estimates[factor].min(),

            "Beta_max": beta_estimates[factor].max(),

            "Beta_range": (
                beta_estimates[factor].max()
                - beta_estimates[factor].min()
            )

        })


    standardized_betas = (
        beta_estimates - beta_estimates.mean()
    ) / beta_estimates.std()

    design_matrix = sm.add_constant(
        standardized_betas
    ).to_numpy()

    design_rank = np.linalg.matrix_rank(
        design_matrix
    )

    condition_number = np.linalg.cond(
        design_matrix
    )


    # ==================================================
    # SECOND PASS: MONTHLY CROSS-SECTIONAL REGRESSIONS
    # ==================================================

    print("\nSECOND PASS: MONTHLY CROSS-SECTIONAL REGRESSIONS")

    X_second = sm.add_constant(
        beta_estimates
    )

    monthly_lambdas = []

    monthly_r_squared = []

    for date in Y.index:

        cross_section_returns = Y.loc[date]

        cross_section_result = sm.OLS(
            cross_section_returns,
            X_second
        ).fit()

        monthly_lambdas.append(
            cross_section_result.params
        )

        monthly_r_squared.append(
            cross_section_result.rsquared
        )


    lambda_df = pd.DataFrame(
        monthly_lambdas,
        index=Y.index
    )

    lambda_df.index.name = "Date"

    average_monthly_r2 = np.mean(
        monthly_r_squared
    )


    # ==================================================
    # 5. HAC INFERENCE FOR AVERAGE RISK PREMIUMS
    # ==================================================

    print("\nAVERAGE RISK PREMIUMS")

    premium_rows = []

    for coefficient in lambda_df.columns:

        monthly_series = lambda_df[coefficient]

        constant = np.ones(
            (len(monthly_series), 1)
        )

        mean_result = sm.OLS(
            monthly_series.to_numpy(),
            constant
        ).fit(
            cov_type="HAC",
            cov_kwds={
                "maxlags": HAC_LAGS
            }
        )

        mean_premium = mean_result.params[0]

        hac_se = mean_result.bse[0]

        t_stat = mean_result.tvalues[0]

        p_value = mean_result.pvalues[0]

        premium_rows.append({

            "Sample": sample_name,

            "Model": model_name,

            "Coefficient": coefficient,

            "Average_monthly_premium": mean_premium,

            "Average_monthly_premium_pct": (
                mean_premium * 100
            ),

            "HAC_SE": hac_se,

            "HAC_SE_pct": hac_se * 100,

            "HAC_t": t_stat,

            "HAC_p": p_value,

            "HAC_lags": HAC_LAGS

        })


    # ==================================================
    # 6. POOLED CROSS-SECTIONAL R-SQUARED
    # ==================================================

    average_returns = Y.mean(axis=0)

    pooled_result = sm.OLS(
        average_returns,
        X_second
    ).fit()


    print(
        "Average monthly cross-sectional R-squared:",
        round(average_monthly_r2, 6)
    )

    print(
        "Pooled average-return cross-sectional R-squared:",
        round(pooled_result.rsquared, 6)
    )

    print(
        "Standardized beta design rank:",
        design_rank,
        "of",
        K + 1
    )

    print(
        "Standardized beta design condition number:",
        round(condition_number, 6)
    )


    return {

        "premiums": premium_rows,

        "lambdas": lambda_df,

        "betas": beta_estimates,

        "beta_diagnostics": beta_rows,

        "r_squared": {

            "Sample": sample_name,

            "Model": model_name,

            "Average_monthly_R2": average_monthly_r2,

            "Pooled_average_return_R2": pooled_result.rsquared,

            "Beta_design_rank": design_rank,

            "Beta_design_columns": K + 1,

            "Beta_design_condition_number": condition_number

        }

    }


# ==================================================
# 7. RUN ALL SAMPLES AND MODELS
# ==================================================

all_premiums = []

all_lambdas = []

all_betas = []

all_beta_diagnostics = []

all_model_diagnostics = []


for sample_name, returns_file in SAMPLES.items():

    returns = load_returns(returns_file)

    for model_name, specification in MODELS.items():

        factors = load_factors(
            specification["file"],
            specification["factors"]
        )

        output = run_fama_macbeth(
            returns,
            factors,
            sample_name,
            model_name
        )

        all_premiums.extend(
            output["premiums"]
        )

        all_beta_diagnostics.extend(
            output["beta_diagnostics"]
        )

        all_model_diagnostics.append(
            output["r_squared"]
        )


        # Add sample and model identifiers to monthly lambdas

        lambda_data = output["lambdas"].copy()

        lambda_data.insert(
            0,
            "Model",
            model_name
        )

        lambda_data.insert(
            0,
            "Sample",
            sample_name
        )

        all_lambdas.append(lambda_data)


        # Save industry beta estimates

        beta_data = output["betas"].copy()

        beta_data.insert(
            0,
            "Model",
            model_name
        )

        beta_data.insert(
            0,
            "Sample",
            sample_name
        )

        all_betas.append(beta_data)


# ==================================================
# 8. COMBINE RESULTS
# ==================================================

premium_df = pd.DataFrame(all_premiums)

lambda_df = pd.concat(
    all_lambdas
)

beta_df = pd.concat(
    all_betas
)

beta_diagnostics_df = pd.DataFrame(
    all_beta_diagnostics
)

model_diagnostics_df = pd.DataFrame(
    all_model_diagnostics
)


# ==================================================
# 9. SAVE RESULTS
# ==================================================

premium_df.to_csv(
    TABLE_DIR / "industry_fama_macbeth_premiums.csv",
    index=False
)

lambda_df.to_csv(
    TABLE_DIR / "industry_fama_macbeth_monthly_lambdas.csv"
)

beta_df.to_csv(
    TABLE_DIR / "industry_fama_macbeth_betas.csv"
)

beta_diagnostics_df.to_csv(
    TABLE_DIR / "industry_fama_macbeth_beta_diagnostics.csv",
    index=False
)

model_diagnostics_df.to_csv(
    TABLE_DIR / "industry_fama_macbeth_model_diagnostics.csv",
    index=False
)


# ==================================================
# 10. DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 75)

print("INDUSTRY FAMA-MACBETH RESULTS")

print("=" * 75)

print("\nAVERAGE RISK PREMIUMS")

print(
    premium_df.to_string(index=False)
)


print("\nMODEL DIAGNOSTICS")

print(
    model_diagnostics_df.to_string(index=False)
)


print("\nRESULTS SAVED")

print("industry_fama_macbeth_premiums.csv")

print("industry_fama_macbeth_monthly_lambdas.csv")

print("industry_fama_macbeth_betas.csv")

print("industry_fama_macbeth_beta_diagnostics.csv")

print("industry_fama_macbeth_model_diagnostics.csv")

print("\nFAMA-MACBETH ANALYSIS COMPLETED")
