
from pathlib import Path
import pandas as pd
import numpy as np
import re
from io import StringIO


BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data_raw"
DATA_DIR = BASE_DIR / "data_processed"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# Load original file

raw_file = RAW_DIR / "49_Industry_Portfolios.csv"

if not raw_file.exists():

    raise FileNotFoundError(
        "49_Industry_Portfolios.csv not found."
    )

content = raw_file.read_text(
    encoding="latin1"
)

lines = content.splitlines()

print("RAW FILE:", raw_file.name)


# Locate value-weighted monthly section

section_position = None

for i, line in enumerate(lines):

    if "Average Value Weighted Returns -- Monthly" in line:

        section_position = i
        break


if section_position is None:

    raise ValueError(
        "Value-weighted monthly section not found."
    )


# Locate portfolio header

header_position = None

for i in range(
    section_position + 1,
    min(section_position + 10, len(lines))
):

    line = lines[i]

    if (
        line.strip().startswith(",")
        and "Agric" in line
        and "Other" in line
    ):

        header_position = i
        break


if header_position is None:

    raise ValueError(
        "Could not find the 49-industry header."
    )


# Extract monthly observations only

table_lines = [lines[header_position]]

for line in lines[header_position + 1:]:

    first_value = line.split(",")[0].strip()

    if re.fullmatch(r"\d{6}", first_value):

        table_lines.append(line)

    elif len(table_lines) > 1:

        break


# Read extracted table

df = pd.read_csv(
    StringIO("\n".join(table_lines))
)

df.columns = [
    str(col).strip()
    for col in df.columns
]

df = df.rename(
    columns={df.columns[0]: "Date"}
)


# Identify industry portfolios

industry_columns = [
    col for col in df.columns
    if col != "Date"
]

if len(industry_columns) != 49:

    raise ValueError(
        f"Expected 49 industries, found {len(industry_columns)}."
    )

if len(set(industry_columns)) != 49:

    raise ValueError(
        "Duplicate industry names detected."
    )


# Convert dates

dates = df["Date"].astype(str).str.strip()

if not dates.str.fullmatch(r"\d{6}").all():

    raise ValueError("Unexpected date format.")

df["Date"] = pd.PeriodIndex(
    pd.to_datetime(dates, format="%Y%m"),
    freq="M"
)


# Convert returns to numeric

returns = df[industry_columns].apply(
    pd.to_numeric,
    errors="coerce"
)


# Replace missing-value codes

returns = returns.replace(
    [-99.99, -999],
    np.nan
)


# Convert percentage returns into decimals

returns = returns / 100


# Set monthly date index

returns.index = df["Date"]

returns.index.name = "Date"


# Check duplicate dates

if returns.index.duplicated().any():

    raise ValueError("Duplicate monthly dates detected.")


# Restrict sample to July 1963 - December 2025

sample_dates = pd.period_range(
    start="1963-07",
    end="2025-12",
    freq="M"
)

returns = returns.reindex(sample_dates)


# Save cleaned raw returns

returns.to_csv(
    DATA_DIR / "industry_49_returns.csv"
)


# Load existing risk-free rate

rf_file = DATA_DIR / "risk_free_rate.csv"

if not rf_file.exists():

    raise FileNotFoundError(
        "risk_free_rate.csv not found in data_processed."
    )

rf_data = pd.read_csv(
    rf_file,
    index_col=0
)

rf_data.index = pd.PeriodIndex(
    rf_data.index.astype(str),
    freq="M"
)

if "RF" not in rf_data.columns:

    raise ValueError("RF column not found.")

rf = rf_data["RF"].reindex(sample_dates)


# Validate risk-free rate

if rf.isna().any():

    raise ValueError(
        "Missing risk-free observations."
    )

if rf.abs().median() > 0.1:

    raise ValueError(
        "RF appears to be in percentage units. "
        "Check the original risk-free-rate processing."
    )


# Check portfolio data

print("\n49 INDUSTRY DATA CHECK")

print("Observations:", len(returns))
print("Industries:", len(returns.columns))
print("First month:", returns.index.min())
print("Last month:", returns.index.max())

print("\nMissing observations by industry:")

print(
    returns.isna().sum().to_string()
)

total_missing = returns.isna().sum().sum()

print("\nTotal missing values:", total_missing)


# Calculate excess returns only if complete

if total_missing > 0:

    print(
        "\nWARNING: Missing observations remain."
    )

    print(
        "Cleaned raw returns were saved, "
        "but excess returns were not created."
    )

else:

    # Subtract RF exactly once

    excess_returns = returns.sub(
        rf,
        axis=0
    )

    # Save excess returns

    excess_returns.to_csv(
        DATA_DIR / "industry_49_excess.csv"
    )

    print("\nEXCESS RETURNS CHECK")

    print(
        "Average risk-free rate (%):",
        round(rf.mean() * 100, 4)
    )

    print(
        "Average raw return (%):",
        round(returns.mean().mean() * 100, 4)
    )

    print(
        "Average excess return (%):",
        round(excess_returns.mean().mean() * 100, 4)
    )

    print(
        "Missing excess returns:",
        excess_returns.isna().sum().sum()
    )

    print("\nALL 49 INDUSTRIES VALIDATED.")

    print(
        "industry_49_excess.csv successfully created."
    )


print("\nINDUSTRY PROCESSING COMPLETED")
