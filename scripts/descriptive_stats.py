
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data_processed"
TABLE_DIR = BASE_DIR / "results" / "tables"
FIGURE_DIR = BASE_DIR / "results" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_data(filename):

    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    df = pd.read_csv(file_path, index_col=0)

    df.index = pd.PeriodIndex(
        df.index.astype(str),
        freq="M"
    )

    df = df.apply(pd.to_numeric, errors="coerce")

    if df.isna().any().any():
        raise ValueError(
            "Missing values found in " + filename
        )

    return df


# Load processed datasets

ff6 = load_data("factors_ff6.csv")

size_bm = load_data("size_bm_excess.csv")

industry = load_data("industry_excess.csv")


# Calculate descriptive statistics

def descriptive_statistics(df):

    results = pd.DataFrame(index=df.columns)

    results["Observations"] = df.count()

    results["Mean (%)"] = df.mean() * 100

    results["Median (%)"] = df.median() * 100

    results["Std Dev (%)"] = df.std() * 100

    results["Minimum (%)"] = df.min() * 100

    results["Maximum (%)"] = df.max() * 100

    results["Skewness"] = df.skew()

    results["Excess Kurtosis"] = df.kurt()

    return results


# Generate three statistics tables

factor_stats = descriptive_statistics(ff6)

size_bm_stats = descriptive_statistics(size_bm)

industry_stats = descriptive_statistics(industry)


# Save tables

factor_stats.to_csv(
    TABLE_DIR / "factor_descriptive_statistics.csv"
)

size_bm_stats.to_csv(
    TABLE_DIR / "size_bm_descriptive_statistics.csv"
)

industry_stats.to_csv(
    TABLE_DIR / "industry_descriptive_statistics.csv"
)


# Calculate factor correlations

correlations = ff6.corr()

correlations.to_csv(
    TABLE_DIR / "ff6_factor_correlations.csv"
)


# Create factor correlation heatmap

fig, ax = plt.subplots(figsize=(9, 7))

image = ax.imshow(
    correlations,
    vmin=-1,
    vmax=1,
    cmap="coolwarm"
)

ax.set_xticks(range(len(correlations.columns)))
ax.set_yticks(range(len(correlations.columns)))

ax.set_xticklabels(
    correlations.columns,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(correlations.columns)

for i in range(len(correlations.columns)):
    for j in range(len(correlations.columns)):

        ax.text(
            j,
            i,
            round(correlations.iloc[i, j], 2),
            ha="center",
            va="center"
        )

ax.set_title("Correlation Matrix: FF6 Factors")

fig.colorbar(image, ax=ax, label="Correlation")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "ff6_factor_correlation_heatmap.png",
    dpi=300
)

plt.show()


# Additional return-unit checks

print("\nRETURN UNIT CHECK")

for name, df in {
    "FF6": ff6,
    "Size-BM": size_bm,
    "Industry": industry
}.items():

    average_absolute = df.abs().stack().mean() * 100

    maximum_absolute = df.abs().max().max() * 100

    print("\n", name)

    print(
        "Average absolute monthly return (%):",
        round(average_absolute, 4)
    )

    print(
        "Maximum absolute monthly return (%):",
        round(maximum_absolute, 4)
    )


# Print results

print("\nFACTOR DESCRIPTIVE STATISTICS")
print(factor_stats.round(4))

print("\nSIZE-BM DESCRIPTIVE STATISTICS")
print(size_bm_stats.round(4))

print("\nINDUSTRY DESCRIPTIVE STATISTICS")
print(industry_stats.round(4))

print("\nFACTOR CORRELATIONS")
print(correlations.round(4))

print("\nDESCRIPTIVE STATISTICS COMPLETED")
