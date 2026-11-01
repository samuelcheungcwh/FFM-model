
from pathlib import Path
import pandas as pd
import numpy as np
import re
from io import StringIO
import zipfile


BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data_raw"
DATA_DIR = BASE_DIR / "data_processed"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# Find the raw dataset

files = list(RAW_DIR.glob("25_Portfolios_ME_Prior_12_2*"))

if len(files) == 0:
    raise FileNotFoundError(
        "25_Portfolios_ME_Prior_12_2 not found in data_raw."
    )

raw_file = next(
    (p for p in files if p.suffix.lower() in [".csv", ".txt"]),
    files[0]
)

print("RAW FILE:", raw_file.name)


# Read the original file

if raw_file.suffix.lower() == ".zip":

    with zipfile.ZipFile(raw_file, "r") as z:

        csv_files = [
            name for name in z.namelist()
            if name.lower().endswith(".csv")
        ]

        if len(csv_files) == 0:
            raise ValueError("No CSV file found inside ZIP.")

        content = z.read(csv_files[0]).decode("latin1")

else:

    content = raw_file.read_text(encoding="latin1")


# Locate the monthly value-weighted return table

lines = content.splitlines()

header_position = None

for i, line in enumerate(lines):

    if (
        "SMALL LoPRIOR" in line
        and "BIG HiPRIOR" in line
    ):

        header_position = i
        break


if header_position is None:
    raise ValueError("Could not locate the monthly portfolio table.")


# Extract the monthly observations

table_lines = [lines[header_position]]

for line in lines[header_position + 1:]:

    first_value = line.split(",")[0].strip()

    if re.fullmatch(r"\d{6}", first_value):

        table_lines.append(line)

    elif line.strip() == "":
        break

    else:
        break


df = pd.read_csv(
    StringIO("\n".join(table_lines))
)

df.columns = [str(col).strip() for col in df.columns]

date_column = df.columns[0]

df = df.rename(columns={date_column: "Date"})


# Check portfolio columns

portfolio_columns = [
    col for col in df.columns if col != "Date"
]

if len(portfolio_columns) != 25:
    raise ValueError(
        f"Expected 25 portfolios, found {len(portfolio_columns)}."
    )


# Convert dates

dates = df["Date"].astype(str).str.strip()

if not dates.str.fullmatch(r"\d{6}").all():
    raise ValueError("Unexpected date format.")

df["Date"] = pd.PeriodIndex(
    pd.to_datetime(dates, format="%Y%m"),
    freq="M"
)


# Convert returns to numeric values

returns = df[portfolio_columns].apply(
    pd.to_numeric,
    errors="coerce"
)


# Replace missing-value codes

returns = returns.replace(
    [-99.99, -999],
    np.nan
)


# Raw data are in percentages.
# Convert to decimal returns.

returns = returns / 100


# Set monthly index

returns.index = df["Date"]

returns.index.name = "Date"


# Check duplicate dates

if returns.index.duplicated().any():
    raise ValueError("Duplicate monthly observations found.")


# Restrict sample to July 1963 - December 2025

sample_dates = pd.period_range(
    start="1963-07",
    end="2025-12",
    freq="M"
)

returns = returns.reindex(sample_dates)


# Check missing observations

print("\nSIZE-MOMENTUM DATA CHECK")

print("Observations:", len(returns))
print("Portfolios:", len(returns.columns))
print("First month:", returns.index.min())
print("Last month:", returns.index.max())

print("\nMissing observations by portfolio:")

print(returns.isna().sum().to_string())


if returns.isna().any().any():

    raise ValueError(
        "Missing observations remain in the sample. "
        "Investigate before proceeding."
    )


# Save cleaned raw portfolio returns

returns.to_csv(
    DATA_DIR / "size_momentum_returns.csv"
)


# Find the risk-free rate

factor_files = [
    DATA_DIR / "factors_ff5.csv",
    DATA_DIR / "factors_ff6.csv"
]

rf = None

for factor_file in factor_files:

    if not factor_file.exists():
        continue

    factors = pd.read_csv(
        factor_file,
        index_col=0
    )

    factors.columns = [
        str(col).strip() for col in factors.columns
    ]

    if "RF" in factors.columns:

        factor_dates = factors.index.astype(str)

        if all(
            re.fullmatch(r"\d{6}", date)
            for date in factor_dates
        ):

            factors.index = pd.PeriodIndex(
                pd.to_datetime(
                    factor_dates,
                    format="%Y%m"
                ),
                freq="M"
            )

        else:

            factors.index = pd.PeriodIndex(
                factor_dates,
                freq="M"
            )

        rf = pd.to_numeric(
            factors["RF"],
            errors="coerce"
        )

        rf = rf.reindex(sample_dates)

        print("\nRISK-FREE SOURCE:", factor_file.name)

        break


if rf is None:

    print(
        "\nRF column not found in processed factor files."
    )

    print(
        "Cleaned raw returns have been saved, "
        "but excess returns have not been created."
    )

else:

    # Check the risk-free rate units

    if rf.isna().any():

        raise ValueError(
            "Missing risk-free observations in the sample."
        )

    if rf.abs().median() > 0.1:

        print("RF appears to be in percentage units.")
        rf = rf / 100

    else:

        print("RF appears to be in decimal units.")

    # Create excess returns

    excess_returns = returns.sub(
        rf,
        axis=0
    )

    excess_returns.to_csv(
        DATA_DIR / "size_momentum_excess.csv"
    )

    print("\nEXCESS RETURNS CREATED")


# Final output

print("\nPROCESSED FILES")

print("size_momentum_returns.csv")

if rf is not None:
    print("size_momentum_excess.csv")

print("\nSIZE-MOMENTUM PROCESSING COMPLETED")
