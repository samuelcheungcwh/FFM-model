
from pathlib import Path
import pandas as pd
import statsmodels.api as sm


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data_processed"
TABLE_DIR = BASE_DIR / "results" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)


# Load FF6 dataset

factors = pd.read_csv(
    DATA_DIR / "factors_ff6.csv",
    index_col=0
)

factors.index = pd.PeriodIndex(
    factors.index.astype(str),
    freq="M"
)


# Define the dependent and explanatory variables

y = factors["MOM"]

factor_columns = [
    "Mkt-RF",
    "SMB",
    "HML",
    "RMW",
    "CMA"
]

X = factors[factor_columns]

X = sm.add_constant(X)


# Run the spanning regression with different HAC lags

lag_values = [3, 6, 12]

results = []

for lag in lag_values:

    model = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": lag},
        use_t=True
    )

    alpha = model.params["const"]

    alpha_se = model.bse["const"]

    alpha_t = model.tvalues["const"]

    alpha_p = model.pvalues["const"]

    result = {
        "HAC Lags": lag,
        "Alpha (%)": alpha * 100,
        "Alpha SE (%)": alpha_se * 100,
        "Alpha t-stat": alpha_t,
        "Alpha p-value": alpha_p,
        "Reject H0 at 5%": alpha_p < 0.05,
        "Adjusted R2": model.rsquared_adj
    }

    results.append(result)

    print("\nHAC LAGS:", lag)

    print(model.summary())


# Save the regression results

results = pd.DataFrame(results)

results.to_csv(
    TABLE_DIR / "momentum_spanning_results.csv",
    index=False
)

print("\nMOMENTUM SPANNING RESULTS")

print(results.round(6).to_string(index=False))

print("\nMOMENTUM SPANNING TEST COMPLETED")
