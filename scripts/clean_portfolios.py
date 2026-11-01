from pathlib import Path
import csv
import re
import pandas as pd
import numpy as np

# Project layout: ECO1400/data_raw/ and ECO1400/data_clean/
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_DIR / "data_raw" / "25_Portfolios_5x5.csv"
OUTPUT_DIR = PROJECT_DIR / "data_clean"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "1963-07"
END_DATE = "2026-09"
SECTION_NAME = "Average Value Weighted Returns -- Monthly"
MISSING_CODES = [-99.99, -999]


def extract_monthly_value_weighted(file_path):
    """Extract only the monthly value-weighted return table from French CSV."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path}. Place 25_Portfolios_5x5.csv in data_raw."
        )

    with file_path.open("r", encoding="utf-8-sig", newline="") as file:
        lines = file.readlines()

    section_line = None
    for i, line in enumerate(lines):
        if SECTION_NAME in line:
            section_line = i
            break

    if section_line is None:
        raise ValueError(f"Could not find section: {SECTION_NAME}")

    header_line = section_line + 1
    header = next(csv.reader([lines[header_line]]))
    header = [item.strip() for item in header]

    if len(header) != 26:
        raise ValueError(
            f"Expected date plus 25 portfolio columns; found {len(header)} columns."
        )

    rows = []
    for line in lines[header_line + 1:]:
        if not line.strip():
            continue

        row = next(csv.reader([line]))
        first_value = row[0].strip() if row else ""

        # Monthly observations use YYYYMM. Stop when another section begins.
        if not re.fullmatch(r"\d{6}", first_value):
            break

        if len(row) != len(header):
            raise ValueError(
                f"Unexpected number of fields for month {first_value}: "
                f"expected {len(header)}, found {len(row)}."
            )

        rows.append([value.strip() for value in row])

    if not rows:
        raise ValueError("No monthly value-weighted observations were extracted.")

    df = pd.DataFrame(rows, columns=header)
    date_column = df.columns[0]
    df[date_column] = pd.PeriodIndex(df[date_column], freq="M")
    df = df.set_index(date_column)
    df.index.name = "Date"

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace(MISSING_CODES, np.nan)
    df = df.loc[START_DATE:END_DATE]

    # French Library returns are published in percent; convert to decimal.
    df = df / 100

    if df.shape[1] != 25:
        raise ValueError(f"Expected 25 portfolios, found {df.shape[1]}.")

    if df.index.has_duplicates:
        raise ValueError("Duplicate monthly dates were found.")

    return df


def main():
    cleaned = extract_monthly_value_weighted(RAW_FILE)

    expected_dates = pd.period_range(START_DATE, END_DATE, freq="M")
    missing_dates = expected_dates.difference(cleaned.index)
    missing_cells = int(cleaned.isna().sum().sum())
    missing_by_portfolio = cleaned.isna().sum()

    output_csv = OUTPUT_DIR / "size_bm_monthly_value_weighted.csv"
    output_report = OUTPUT_DIR / "size_bm_cleaning_report.txt"

    cleaned.to_csv(output_csv, float_format="%.10g")

    report = [
        "25 Size-BM Portfolios Cleaning Report",
        "======================================",
        f"Source file: {RAW_FILE.name}",
        f"Extracted section: {SECTION_NAME}",
        "Portfolio weighting: Value-weighted",
        "Frequency: Monthly",
        f"Sample requested: {START_DATE} to {END_DATE}",
        f"Actual sample: {cleaned.index.min()} to {cleaned.index.max()}",
        f"Observations: {len(cleaned)}",
        f"Portfolio columns: {cleaned.shape[1]}",
        "Units: Decimal returns (original percentages divided by 100)",
        "Missing-value codes: -99.99 and -999 converted to NaN",
        f"Missing calendar months in requested sample: {len(missing_dates)}",
        f"Missing data cells: {missing_cells}",
        "",
        "Missing cells by portfolio:",
    ]
    report.extend([f"{name}: {int(count)}" for name, count in missing_by_portfolio.items()])
    report.extend([
        "",
        "No risk-free-rate adjustment has been made. These are raw portfolio returns.",
        "Subtract RF later when constructing the excess-return test assets.",
        "Rows with missing portfolio returns have not been silently dropped.",
    ])
    output_report.write_text("\n".join(report), encoding="utf-8")

    print("CLEANING COMPLETED")
    print("Output:", output_csv)
    print("Report:", output_report)
    print("Shape:", cleaned.shape)
    print("Date range:", cleaned.index.min(), "to", cleaned.index.max())
    print("Missing calendar months:", len(missing_dates))
    print("Missing data cells:", missing_cells)
    print("Columns:", cleaned.columns.tolist())


if __name__ == "__main__":
    main()
