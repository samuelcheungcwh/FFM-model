"""Clean the monthly Fama/French 5 Factors (2x3) file.

Expected project layout:
ECO1400/
    data_raw/
        F-F_Research_Data_5_Factors_2x3 2.csv
    data_clean/
        clean_ff5_factors.py

The script extracts monthly observations only, excludes the annual table,
converts percentage returns to decimals, preserves RF, and does not fill
missing observations.
"""

from pathlib import Path
import csv
import re
import sys
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data_raw"
CLEAN_DIR = BASE_DIR / "data_clean"

START_DATE = "1963-07"
END_DATE = "2026-09"
REQUIRED_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
OUTPUT_FILE = CLEAN_DIR / "ff5_monthly_factors.csv"
REPORT_FILE = CLEAN_DIR / "ff5_cleaning_report.txt"


def locate_input_file():
    """Find the user's original French Library CSV without renaming it."""
    preferred_names = [
        "F-F_Research_Data_5_Factors_2x3 2.csv",
        "F-F_Research_Data_5_Factors_2x3.csv",
        "ff5.csv",
    ]

    for filename in preferred_names:
        candidate = RAW_DIR / filename
        if candidate.exists():
            return candidate

    matches = sorted(RAW_DIR.glob("F-F_Research_Data_5_Factors_2x3*.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        "Could not find the FF5 source file in data_raw. Put the file there "
        "with a name such as 'F-F_Research_Data_5_Factors_2x3 2.csv'."
    )


def parse_monthly_ff5(path):
    """Read only YYYYMM monthly rows following the FF5 header."""
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
        rows = list(csv.reader(file))

    header_index = None
    header = None

    for i, row in enumerate(rows):
        cleaned = [item.strip() for item in row]
        if all(column in cleaned for column in REQUIRED_COLUMNS):
            header_index = i
            header = cleaned
            break

    if header_index is None:
        raise ValueError(
            "Could not locate the FF5 monthly header. Expected columns: "
            + ", ".join(REQUIRED_COLUMNS)
        )

    column_positions = {
        column: header.index(column) for column in REQUIRED_COLUMNS
    }

    # The date is the first field in the French Library monthly table.
    date_position = 0
    observations = []

    for row in rows[header_index + 1:]:
        if not row:
            continue

        date_value = row[date_position].strip() if len(row) > date_position else ""

        # Annual data use YYYY rather than YYYYMM, so they are excluded.
        if not re.fullmatch(r"\d{6}", date_value):
            continue

        year = int(date_value[:4])
        month = int(date_value[4:6])
        if month < 1 or month > 12:
            continue

        values = {}
        for column, position in column_positions.items():
            try:
                value = float(row[position].strip())
            except (IndexError, ValueError):
                value = np.nan

            if value in (-99.99, -999):
                value = np.nan

            values[column] = value

        observations.append((f"{year:04d}-{month:02d}", values))

    if not observations:
        raise ValueError("No monthly YYYYMM observations were found in the source file.")

    data = pd.DataFrame(
        [item[1] for item in observations],
        index=pd.PeriodIndex([item[0] for item in observations], freq="M")
    )

    if data.index.has_duplicates:
        duplicate_dates = data.index[data.index.duplicated()].astype(str).tolist()
        raise ValueError("Duplicate monthly dates found: " + str(duplicate_dates[:10]))

    data = data.sort_index()
    return data


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    source_file = locate_input_file()

    print("Source file:", source_file)
    data = parse_monthly_ff5(source_file)

    # Keep the exact research window used by the project.
    start = pd.Period(START_DATE, freq="M")
    end = pd.Period(END_DATE, freq="M")
    expected_dates = pd.period_range(start=start, end=end, freq="M")
    data = data.loc[(data.index >= start) & (data.index <= end)]

    # Reindex to reveal any absent months; do not impute them.
    data = data.reindex(expected_dates)

    # Source returns are reported in percentage points. Convert to decimals.
    data = data / 100
    data.index.name = "Date"

    missing_months = data.index[data.isna().all(axis=1)].astype(str).tolist()
    missing_values = data.isna().sum()

    data.to_csv(OUTPUT_FILE, float_format="%.10f")

    report_lines = [
        "Fama/French 5 Factors (2x3) cleaning report",
        "",
        f"Source file: {source_file.name}",
        f"Research period: {START_DATE} to {END_DATE}",
        f"Rows in output: {len(data)}",
        f"Columns: {', '.join(data.columns)}",
        f"First month: {data.index.min()}",
        f"Last month: {data.index.max()}",
        "",
        "Missing values by column:",
    ]

    for column, count in missing_values.items():
        report_lines.append(f"  {column}: {int(count)}")

    report_lines.extend([
        "",
        f"Months with all six factors missing: {len(missing_months)}",
        "Missing months are retained as NaN; no values are imputed.",
        "The annual table is excluded.",
        "RF is retained for later construction of portfolio excess returns.",
        "All six columns are converted from percentage points to decimal returns.",
        f"Output file: {OUTPUT_FILE.name}",
    ])

    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nCleaning completed.")
    print("Output:", OUTPUT_FILE)
    print("Report:", REPORT_FILE)
    print("Shape:", data.shape)
    print("Date range:", data.index.min(), "to", data.index.max())
    print("Missing values:")
    print(missing_values)
    print("Months with all factors missing:", len(missing_months))
    print("\nFirst five observations:")
    print(data.head())
    print("\nLast five observations:")
    print(data.tail())


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("ERROR:", error)
        sys.exit(1)
