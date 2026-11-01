"""Clean the monthly Fama-French Momentum factor for the ECO1400 project.

Input:
    ECO1400/data_raw/F-F_Momentum_Factor.csv

Outputs:
    ECO1400/data_clean/momentum_monthly.csv
    ECO1400/data_clean/momentum_cleaning_report.txt

The source file contains a monthly section followed by an annual section.
This script extracts only the monthly Mom observations, converts percentage
returns to decimal returns, replaces the source's missing-value sentinels,
and retains the project's common sample (1963-07 through 2026-09).
"""

from pathlib import Path
import re
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data_raw"
CLEAN_DIR = BASE_DIR / "data_clean"
INPUT_FILE = RAW_DIR / "F-F_Momentum_Factor.csv"
OUTPUT_FILE = CLEAN_DIR / "momentum_monthly.csv"
REPORT_FILE = CLEAN_DIR / "momentum_cleaning_report.txt"

START_MONTH = "1963-07"
END_MONTH = "2026-09"


def read_monthly_momentum(file_path):
    """Read the monthly Mom table from the French Library text-style CSV."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find the input file: {file_path}\n"
            "Make sure F-F_Momentum_Factor.csv is inside ECO1400/data_raw/."
        )

    text = file_path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()

    header_index = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*,\s*Mom\s*$", line, flags=re.IGNORECASE):
            header_index = i
            break

    if header_index is None:
        raise ValueError(
            "Could not find the monthly ',Mom' header. "
            "Check that the input is the monthly CSV version of the Momentum Factor."
        )

    records = []
    row_pattern = re.compile(
        r"^\s*(\d{6})\s*,\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?)\s*$"
    )

    for line in lines[header_index + 1:]:
        if "annual factors" in line.lower():
            break

        match = row_pattern.match(line)
        if match:
            date_code = match.group(1)
            value = float(match.group(2))
            records.append((date_code, value))

    if not records:
        raise ValueError("No monthly Momentum observations were found after the header.")

    df = pd.DataFrame(records, columns=["Date", "MOM"])
    df["Date"] = pd.PeriodIndex(df["Date"], freq="M")
    df["MOM"] = pd.to_numeric(df["MOM"], errors="coerce")

    if df["Date"].duplicated().any():
        duplicates = df.loc[df["Date"].duplicated(), "Date"].astype(str).tolist()
        raise ValueError(f"Duplicate monthly dates found: {duplicates[:10]}")

    df = df.set_index("Date").sort_index()

    # Fama-French missing observations are coded as -99.99 or -999.
    df = df.replace([-99.99, -999], pd.NA)

    return df


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    momentum_full = read_monthly_momentum(INPUT_FILE)

    # Keep the same sample window used for the other cleaned project datasets.
    momentum = momentum_full.loc[START_MONTH:END_MONTH].copy()

    expected_dates = pd.period_range(START_MONTH, END_MONTH, freq="M")
    missing_dates = expected_dates.difference(momentum.index)

    # Reindex so any absent calendar month is visible as a missing observation.
    momentum = momentum.reindex(expected_dates)

    # Source returns are percentages; store as decimal returns.
    momentum["MOM"] = pd.to_numeric(momentum["MOM"], errors="coerce") / 100

    missing_values = int(momentum["MOM"].isna().sum())

    momentum.to_csv(OUTPUT_FILE, index_label="Date", float_format="%.10f")

    report_lines = [
        "Fama-French Momentum Factor Cleaning Report",
        "============================================",
        f"Input file: {INPUT_FILE.name}",
        "Input section extracted: Monthly Mom only",
        "Annual factor table: excluded",
        f"Research sample: {START_MONTH} to {END_MONTH}",
        f"First available source month: {momentum_full.index.min()}",
        f"Last available source month: {momentum_full.index.max()}",
        f"Rows in cleaned sample: {len(momentum)}",
        f"Columns: {', '.join(momentum.columns)}",
        "Units: decimal returns (source percentage values divided by 100)",
        "Risk-free adjustment: none (MOM is a long-short factor return)",
        f"Missing calendar months: {len(missing_dates)}",
        f"Missing MOM observations after cleaning: {missing_values}",
        "",
        "Missing calendar months found:",
        ", ".join(str(x) for x in missing_dates) if len(missing_dates) else "None",
    ]

    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

    print("Momentum cleaning completed.")
    print("Input:", INPUT_FILE)
    print("Output:", OUTPUT_FILE)
    print("Report:", REPORT_FILE)
    print("Rows:", len(momentum))
    print("Date range:", momentum.index.min(), "to", momentum.index.max())
    print("Missing observations:", missing_values)
    print("Missing calendar months:", len(missing_dates))
    print("\nFirst five observations:")
    print(momentum.head())
    print("\nLast five observations:")
    print(momentum.tail())


if __name__ == "__main__":
    main()
