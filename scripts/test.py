
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data_processed"
TABLE_DIR = BASE_DIR / "results" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)


# Load cleaned industry returns

returns = pd.read_csv(
    DATA_DIR / "industry_49_returns.csv",
    index_col=0
)

returns.index = pd.PeriodIndex(
    returns.index.astype(str),
    freq="M"
)


# Identify consecutive missing periods

def find_missing_ranges(dates):

    if len(dates) == 0:
        return []

    ranges = []

    start = dates[0]
    previous = dates[0]

    for date in dates[1:]:

        if date.ordinal == previous.ordinal + 1:

            previous = date

        else:

            ranges.append((start, previous))

            start = date
            previous = date

    ranges.append((start, previous))

    return ranges


print("\nINDUSTRY MISSING-DATA DIAGNOSTICS")

for industry in returns.columns:

    missing_dates = returns.index[
        returns[industry].isna()
    ]

    if len(missing_dates) > 0:

        print("\nIndustry:", industry)

        print("Missing observations:", len(missing_dates))

        ranges = find_missing_ranges(missing_dates)

        for start, end in ranges:

            print(
                "Missing period:",
                start,
                "to",
                end
            )


# Check the common sample across all industries

complete_mask = returns.notna().all(axis=1)

complete_dates = returns.index[complete_mask]

print("\nCOMMON SAMPLE CHECK")

print(
    "Months with all 49 industries available:",
    len(complete_dates)
)

print(
    "Months containing at least one missing value:",
    (~complete_mask).sum()
)

if len(complete_dates) > 0:

    print("First complete month:", complete_dates.min())

    print("Last complete month:", complete_dates.max())


# Identify complete industries

complete_industries = returns.columns[
    returns.notna().all()
]

incomplete_industries = returns.columns[
    ~returns.notna().all()
]

print("\nCOMPLETE INDUSTRIES:", len(complete_industries))

print("INCOMPLETE INDUSTRIES:", len(incomplete_industries))

print("\nIncomplete industries:")

print(list(incomplete_industries))


# Save missingness report

report = pd.DataFrame({
    "Date": returns.index.astype(str),
    "Complete Across 49 Industries": complete_mask,
    "Missing Industries": [
        ", ".join(
            returns.columns[
                returns.loc[date].isna()
            ]
        )
        for date in returns.index
    ]
})

report.to_csv(
    TABLE_DIR / "industry49_missingness_report.csv",
    index=False
)

print("\nMISSINGNESS REPORT SAVED")

print("\nDIAGNOSTICS COMPLETED")
