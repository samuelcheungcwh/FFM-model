
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data_processed"


# Load cleaned industry returns

returns = pd.read_csv(
    DATA_DIR / "industry_49_returns.csv",
    index_col=0
)

returns.index = pd.PeriodIndex(
    returns.index.astype(str),
    freq="M"
)


# Load risk-free rate

rf_data = pd.read_csv(
    DATA_DIR / "risk_free_rate.csv",
    index_col=0
)

rf = rf_data.iloc[:, 0]

rf.index = pd.PeriodIndex(
    rf.index.astype(str),
    freq="M"
)

rf = pd.to_numeric(rf, errors="coerce")


# RF is already in decimal form.
# Do not divide it by 100 again.

if rf.abs().max() > 0.10:
    raise ValueError(
        "RF appears not to be in decimal form."
    )


# Align the risk-free rate with industry returns

rf = rf.reindex(returns.index)

if rf.isna().any():
    raise ValueError(
        "Missing risk-free observations."
    )


# --------------------------------------------------
# SAMPLE 1: ALL 49 INDUSTRIES, JULY 1969-DEC 2025
# --------------------------------------------------

industry_49_returns = returns.loc[
    "1969-07":"2025-12"
].copy()

if industry_49_returns.isna().any().any():
    raise ValueError(
        "Missing observations remain in the 49-industry sample."
    )

industry_49_rf = rf.loc[
    industry_49_returns.index
]

industry_49_excess = industry_49_returns.sub(
    industry_49_rf,
    axis=0
)


# --------------------------------------------------
# SAMPLE 2: 47 COMPLETE INDUSTRIES, JULY 1963-DEC 2025
# --------------------------------------------------

industry_47_returns = returns.drop(
    columns=["Hlth", "Softw"]
).copy()

if industry_47_returns.isna().any().any():
    raise ValueError(
        "Missing observations remain in the 47-industry sample."
    )

industry_47_excess = industry_47_returns.sub(
    rf,
    axis=0
)


# --------------------------------------------------
# SAVE DATASETS
# --------------------------------------------------

industry_49_returns.to_csv(
    DATA_DIR / "industry_49_returns_common.csv",
    index_label="Date"
)

industry_49_excess.to_csv(
    DATA_DIR / "industry_49_excess.csv",
    index_label="Date"
)

industry_47_returns.to_csv(
    DATA_DIR / "industry_47_returns_full.csv",
    index_label="Date"
)

industry_47_excess.to_csv(
    DATA_DIR / "industry_47_excess.csv",
    index_label="Date"
)


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

print("\nINDUSTRY SAMPLE PROCESSING")

print("\n49 INDUSTRY SAMPLE")

print("Observations:", len(industry_49_excess))
print("Industries:", len(industry_49_excess.columns))
print("First month:", industry_49_excess.index.min())
print("Last month:", industry_49_excess.index.max())
print("Missing values:", industry_49_excess.isna().sum().sum())


print("\n47 INDUSTRY SAMPLE")

print("Observations:", len(industry_47_excess))
print("Industries:", len(industry_47_excess.columns))
print("First month:", industry_47_excess.index.min())
print("Last month:", industry_47_excess.index.max())
print("Missing values:", industry_47_excess.isna().sum().sum())


print("\nFILES SAVED SUCCESSFULLY")
