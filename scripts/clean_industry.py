from pathlib import Path
from io import StringIO
import re
import pandas as pd


# Project folders
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data_raw"
CLEAN_DIR = BASE_DIR / "data_clean"

# Input and output files
INPUT_FILE = RAW_DIR / "10_Industry_Portfolios.csv"
OUTPUT_FILE = CLEAN_DIR / "industry_monthly_value_weighted.csv"
REPORT_FILE = CLEAN_DIR / "industry_cleaning_report.txt"

# Keep the same sample window as the main asset-pricing project
START_MONTH = "1963-07"
END_MONTH = "2026-09"

INDUSTRIES = [
    "NoDur", "Durbl", "Manuf", "Enrgy", "HiTec",
    "Telcm", "Shops", "Hlth", "Utils", "Other"
]

MISSING_CODES = [-99.99, -999]


def read_monthly_value_weighted_data(file_path):
    """Read only the monthly value-weighted table from a French Library file."""
    text = file_path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()

    section_title = "Average Value Weighted Returns -- Monthly"
    section_start = None

    for i, line in enumerate(lines):
        if section_title.lower() in line.lower():
            section_start = i
            break

    # Some downloads contain only the monthly table, with no section title.
    if section_start is None:
        df = pd.read_csv(file_path)
    else:
        header_index = None

        for i in range(section_start + 1, len(lines)):
            line = lines[i].strip()
            if line.startswith(",") and "NoDur" in line:
                header_index = i
                break

        if header_index is None:
            raise ValueError(
                "Could not find the monthly industry column header. "
                "Check that you downloaded the 10 Industry Portfolios file."
            )

        month_pattern = re.compile(r"^\s*\d{6}\s*,")
        data_lines = []

        for line in lines[header_index + 1:]:
            if month_pattern.match(line):
                data_lines.append(line)
            elif data_lines:
                break

        if not data_lines:
            raise ValueError("No monthly observations were found in the file.")

        csv_text = "\n".join([lines[header_index]] + data_lines)
        df = pd.read_csv(StringIO(csv_text))

    # The first column is the YYYYMM observation date.
    date_column = df.columns[0]
    df = df.rename(columns={date_column: "Date"})

    df["Date"] = df["Date"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df["Date"] = df["Date"].str.zfill(6)

    # Confirm that the expected ten industry return columns are present.
    missing_columns = [name for name in INDUSTRIES if name not in df.columns]
    if missing_columns:
        raise ValueError(
            "The following industry columns are missing: "
            + ", ".join(missing_columns)
        )

    df = df[["Date"] + INDUSTRIES].copy()

    for column in INDUSTRIES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find the raw input file:\n{INPUT_FILE}\n\n"
            "Place 10_Industry_Portfolios.csv inside the project's data_raw folder."
        )

    df = read_monthly_value_weighted_data(INPUT_FILE)

    # Convert the source's missing-value codes to missing observations.
    df[INDUSTRIES] = df[INDUSTRIES].replace(MISSING_CODES, pd.NA)

    # Keep July 1963 through September 2026 for consistency with the project sample.
    df = df.loc[
        (df["Date"] >= START_MONTH.replace("-", ""))
        & (df["Date"] <= END_MONTH.replace("-", ""))
    ].copy()

    # Check date format, duplicates, and chronological order.
    if not df["Date"].str.fullmatch(r"\d{6}").all():
        raise ValueError("Some observation dates are not in YYYYMM format.")

    if df["Date"].duplicated().any():
        duplicates = df.loc[df["Date"].duplicated(), "Date"].tolist()
        raise ValueError(f"Duplicate monthly dates found: {duplicates}")

    df = df.sort_values("Date").reset_index(drop=True)

    # Convert returns from percentages to decimals.
    df[INDUSTRIES] = df[INDUSTRIES] / 100

    # Check the monthly calendar without filling missing observations.
    actual_dates = pd.PeriodIndex(
        pd.to_datetime(df["Date"], format="%Y%m").dt.to_period("M"),
        freq="M"
    )
    expected_dates = pd.period_range(START_MONTH, END_MONTH, freq="M")
    missing_months = expected_dates.difference(actual_dates)

    missing_by_industry = df[INDUSTRIES].isna().sum()

    # Save raw portfolio returns in decimal form; RF is subtracted later
    # when this file is merged with the Fama-French factor data.
    df.to_csv(OUTPUT_FILE, index=False, float_format="%.8f")

    report_lines = [
        "Industry portfolio monthly value-weighted returns - cleaning report",
        f"Source file: {INPUT_FILE.name}",
        "Source table: Average Value Weighted Returns -- Monthly",
        f"Sample requested: {START_MONTH} through {END_MONTH}",
        f"First month in output: {df['Date'].iloc[0] if not df.empty else 'None'}",
        f"Last month in output: {df['Date'].iloc[-1] if not df.empty else 'None'}",
        f"Rows: {len(df)}",
        f"Industry columns: {len(INDUSTRIES)}",
        "Industries: " + ", ".join(INDUSTRIES),
        "Units: decimal returns (source percentage values divided by 100)",
        "Missing-value codes: -99.99 and -999 converted to missing values",
        "Risk-free rate: not subtracted; these are raw portfolio returns",
        f"Missing calendar months: {len(missing_months)}",
        "Missing months: " + (", ".join(str(x) for x in missing_months) if len(missing_months) else "None"),
        "",
        "Missing observations by industry:"
    ]

    for industry, count in missing_by_industry.items():
        report_lines.append(f"{industry}: {int(count)}")

    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

    print("Cleaning completed.")
    print(f"Saved cleaned data to: {OUTPUT_FILE}")
    print(f"Saved report to: {REPORT_FILE}")
    print(f"Rows: {len(df)} | Columns: {len(INDUSTRIES)}")
    print(f"First month: {df['Date'].iloc[0] if not df.empty else 'None'}")
    print(f"Last month: {df['Date'].iloc[-1] if not df.empty else 'None'}")
    print(f"Missing calendar months: {len(missing_months)}")
    print("Missing observations by industry:")
    print(missing_by_industry)


if __name__ == "__main__":
    main()
