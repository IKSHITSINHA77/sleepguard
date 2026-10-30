"""Loading and cleaning the Sleep Health & Lifestyle dataset."""
from pathlib import Path

import pandas as pd

from .config import CATEGORICAL, FEATURES, NUMERIC, REQUIRED_RAW, TARGET


def load_data(path: Path, drop_duplicates: bool = True) -> pd.DataFrame:
    """Return a clean frame with FEATURES + TARGET.

    Duplicates are dropped *before* any split: the public Kaggle file contains many
    repeated rows, and leaving them in leaks test rows into training (inflated scores).
    Time O(n), space O(n).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Download it from Kaggle "
                                "(uom190346a/sleep-health-and-lifestyle-dataset).")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    missing = REQUIRED_RAW - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    # Empty 'Sleep Disorder' means "None" in this dataset (pandas reads it as NaN).
    df[TARGET] = df[TARGET].fillna("None").astype(str).str.strip()
    df["BMI Category"] = df["BMI Category"].astype(str).str.strip().replace({"Normal Weight": "Normal"})

    bp = df["Blood Pressure"].astype(str).str.extract(r"(\d+)\s*/\s*(\d+)").astype(float)
    df["Systolic"], df["Diastolic"] = bp[0], bp[1]

    for col in CATEGORICAL:
        df[col] = df[col].astype(str).str.strip()
    df[NUMERIC] = df[NUMERIC].apply(pd.to_numeric, errors="coerce")

    df = df[FEATURES + [TARGET]].dropna()          # edge case: unparsable BP / blanks
    if drop_duplicates:
        df = df.drop_duplicates()
    return df.reset_index(drop=True)
