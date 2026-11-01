
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

TABLE_DIR = BASE_DIR / "results" / "tables"


beta_file = (
    TABLE_DIR /
    "industry_fama_macbeth_beta_diagnostics.csv"
)

data = pd.read_csv(beta_file)


print("\nINDUSTRY BETA DISPERSION")

columns = [
    "Sample",
    "Model",
    "Factor",
    "Beta_mean",
    "Beta_std",
    "Beta_min",
    "Beta_max",
    "Beta_range"
]

print(
    data[columns].round(4).to_string(index=False)
)


print("\nMOMENTUM BETA COMPARISON")

momentum = data[
    data["Factor"] == "MOM"
]

print(
    momentum[columns].round(4).to_string(index=False)
)


print("\nDIAGNOSTICS COMPLETED")
