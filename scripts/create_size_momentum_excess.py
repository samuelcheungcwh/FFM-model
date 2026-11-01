
from pathlib import Path
import pandas as pd
import numpy as np
import re
from io import StringIO
import zipfile


BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data_raw"
DATA_DIR = BASE_DIR / "data_processed"


# Load cleaned Size-Momentum returns

returns = pd.read_csv(
    DATA_DIR / "size_momentum_returns.csv",
    index_col=0
)

returns.index = pd.PeriodIndex(
    returns.index.astype(str),
    freq="M"
)


# Find the original FF5 dataset

source_lines = None
source_name = None

for file in RAW_DIR.iterdir():

    if file.suffix.lower() in [".csv", ".txt"]:

        content = file.read_text(encoding="latin1")
        lines = content.splitlines()

        for i, line in enumerate(lines):

            fields = [
                x.strip() for x in line.split(",")
            ]

            if (
                "Mkt-RF" in fields
                and "SMB" in fields
                and "HML" in fields
                and "RMW" in fields
                and "CMA" in fields
                and "RF" in fields
            ):

                source_lines = lines
                source_name = file.name
                break

        if source_lines is not None:
            break

    elif file.suffix.lower() == ".zip":

        with zipfile.ZipFile(file, "r") as z:

            for name in z.namelist():

                if not name.lower().endswith(
                    (".csv", ".txt")
                ):
                    continue

                content = z.read(name).decode("latin1")
                lines = content.splitlines()

                for i, line in enumerate(lines):

                    fields = [
                        x.strip() for x in line.split(",")
                    ]

                    if (
                        "Mkt-RF" in fields
                        and "SMB" in fields
                        and "HML" in fields
                        and "RMW" in fields
                        and "CMA" in fields
                        and "RF" in fields
                    ):

                        source_lines = lines
                        source_name = file.name + " / " + name
                        break

                if source_lines is not None:
                    break

        if source_lines is not None:
            break


if source_lines is None:

    raise FileNotFoundError(
        "Original FF5 dataset with RF was not found. "
        "Place the original F-F Research Data 5 Factors "
        "2x3 CSV or ZIP inside data_raw."
    )


print("RF SOURCE:", source_name)


# Locate monthly factor observations

header_position = None

for i, line in enumerate(source_lines):

    if (
        "Mkt-RF" in line
        and "SMB" in line
        and "HML" in line
        and "RF" in line
    ):

        header_position = i
        break


# Extract monthly observations

table_lines = [source_lines[header_position]]

for line in source_lines[header_position + 1:]:

    first_value = line.split(",")[0].strip()

    if re.fullmatch(r"\d{6}", first_value):

        table_lines.append(line)

    elif len(table_lines) > 1:
        break


factors = pd.read_csv(
    StringIO("\n".join(table_lines))
)

date_column = factors.columns[0]

factors = factors.rename(
    columns={date_column: "Date"}
)


# Extract the risk-free rate

factors["Date"] = pd.PeriodIndex(
    pd.to_datetime(
        factors["Date"].astype(str).str.strip(),
        format="%Y%m"
    ),
    freq="M"
)

rf = pd.to_numeric(
    factors["RF"],
    errors="coerce"
)

rf = rf.replace(
    [-99.99, -999],
    np.nan
)

rf.index = factors["Date"]


# Convert percentage to decimal

rf = rf / 100


# Align RF with Size-Momentum observations

rf = rf.reindex(returns.index)


if rf.isna().any():

    raise ValueError(
        "Missing RF observations in the sample."
    )


if returns.isna().any().any():

    raise ValueError(
        "Missing portfolio returns remain."
    )


if len(returns.columns) != 25:

    raise ValueError(
        "Expected exactly 25 portfolios."
    )


# Calculate excess returns

excess_returns = returns.sub(
    rf,
    axis=0
)


# Save RF separately for future use

rf.to_frame("RF").to_csv(
    DATA_DIR / "risk_free_rate.csv"
)


# Save excess returns

excess_returns.to_csv(
    DATA_DIR / "size_momentum_excess.csv"
)


# Final checks

print("\nSIZE-MOMENTUM EXCESS RETURN CHECK")

print("Observations:", len(excess_returns))
print("Portfolios:", len(excess_returns.columns))
print("First month:", excess_returns.index.min())
print("Last month:", excess_returns.index.max())

print(
    "Average risk-free rate (%):",
    round(rf.mean() * 100, 4)
)

print(
    "Average portfolio return (%):",
    round(returns.mean().mean() * 100, 4)
)

print(
    "Average excess return (%):",
    round(excess_returns.mean().mean() * 100, 4)
)

print(
    "Missing values:",
    excess_returns.isna().sum().sum()
)

print("\nEXCESS RETURNS SAVED SUCCESSFULLY")
