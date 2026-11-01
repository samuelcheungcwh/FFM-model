
from pathlib import Path
import re
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]

CLEAN_DIR = BASE_DIR / "data_clean"
OUTPUT_DIR = BASE_DIR / "data_processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "1963-07"
END_DATE = "2025-12"


def read_monthly_file(filename):

    file_path = CLEAN_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    df = pd.read_csv(file_path)

    date_column = df.columns[0]
    dates = df.pop(date_column).astype(str).str.strip()

    periods = []

    for value in dates:

        if re.fullmatch(r"\d{6}", value):
            period = pd.Period(
                year=int(value[:4]),
                month=int(value[4:]),
                freq="M"
            )

        elif re.fullmatch(r"\d{4}-\d{2}", value):
            period = pd.Period(value, freq="M")

        else:
            period = pd.to_datetime(value).to_period("M")

        periods.append(period)

    df.index = pd.PeriodIndex(periods, freq="M")

    df.columns = df.columns.str.strip()

    df = df.apply(pd.to_numeric, errors="coerce")

    df = df.replace([-99.99, -999], np.nan)

    df = df.sort_index()

    if df.index.duplicated().any():
        raise ValueError("Duplicate monthly dates found in " + filename)

    return df


# Load the four cleaned datasets

ff5 = read_monthly_file(
    "ff5_monthly_factors.csv"
)

momentum = read_monthly_file(
    "momentum_monthly.csv"
)

size_bm = read_monthly_file(
    "size_bm_monthly_value_weighted.csv"
)

industry = read_monthly_file(
    "industry_monthly_value_weighted.csv"
)


# Standardize the momentum column name

mom_column = [
    column for column in momentum.columns
    if column.upper() == "MOM"
]

if len(mom_column) != 1:
    raise ValueError("Could not identify the MOM column.")

momentum = momentum.rename(
    columns={mom_column[0]: "MOM"}
)


# Check required FF5 columns

ff5_columns = [
    "Mkt-RF", "SMB", "HML", "RMW", "CMA"
]

required_columns = ff5_columns + ["RF"]

if not all(column in ff5.columns for column in required_columns):
    raise ValueError("FF5 columns do not match expectations.")

if size_bm.shape[1] != 25:
    raise ValueError("Expected 25 Size-BM portfolios.")

if industry.shape[1] != 10:
    raise ValueError("Expected 10 industry portfolios.")


# Combine FF5 and momentum

factors = ff5.join(
    momentum[["MOM"]],
    how="inner"
)


# Select the primary research period

factors = factors.loc[START_DATE:END_DATE]

size_bm = size_bm.loc[START_DATE:END_DATE]

industry = industry.loc[START_DATE:END_DATE]


# Find dates available in all four datasets

common_dates = factors.index.intersection(
    size_bm.index
)

common_dates = common_dates.intersection(
    industry.index
)

factors = factors.loc[common_dates]
size_bm = size_bm.loc[common_dates]
industry = industry.loc[common_dates]


# Construct excess portfolio returns

size_bm_excess = size_bm.sub(
    factors["RF"], axis=0
)

industry_excess = industry.sub(
    factors["RF"], axis=0
)


# Separate FF5 and FF6 explanatory variables

ff5_factors = factors[ff5_columns]

ff6_factors = factors[
    ff5_columns + ["MOM"]
]


# Save all processed datasets

factors.to_csv(
    OUTPUT_DIR / "all_factors_with_RF.csv",
    index_label="Date"
)

ff5_factors.to_csv(
    OUTPUT_DIR / "factors_ff5.csv",
    index_label="Date"
)

ff6_factors.to_csv(
    OUTPUT_DIR / "factors_ff6.csv",
    index_label="Date"
)

size_bm_excess.to_csv(
    OUTPUT_DIR / "size_bm_excess.csv",
    index_label="Date"
)

industry_excess.to_csv(
    OUTPUT_DIR / "industry_excess.csv",
    index_label="Date"
)


# Print validation results

datasets = {
    "FF5 factors": ff5_factors,
    "FF6 factors": ff6_factors,
    "Size-BM excess returns": size_bm_excess,
    "Industry excess returns": industry_excess
}

for name, df in datasets.items():

    print("\n", name)

    print("Observations:", len(df))
    print("Columns:", len(df.columns))
    print("Start:", df.index.min())
    print("End:", df.index.max())
    print("Missing values:", df.isna().sum().sum())


expected_dates = pd.period_range(
    START_DATE,
    END_DATE,
    freq="M"
)

missing_dates = expected_dates.difference(common_dates)

print("\nCOMMON SAMPLE")
print("Expected months:", len(expected_dates))
print("Available months:", len(common_dates))
print("Missing calendar months:", len(missing_dates))
print("Missing dates:", list(missing_dates))

print("\nDATA PREPARATION COMPLETED")
