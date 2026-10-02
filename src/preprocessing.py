"""Data loading, cleaning (wrangling) and feature engineering."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "telecom_churn_raw.csv"

NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
                "CustomerSupportCalls", "AvgMonthlyDataGB", "NumAddOns", "ChargePerTenureMonth"]
ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column data-quality summary."""
    rep = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(),
    })
    return rep.reset_index(names="column")


def clean(raw: pd.DataFrame):
    """Clean the raw dataset. Returns (clean_df, log) where log lists every step taken."""
    df = raw.copy()
    log = []

    def add(step, detail, n):
        log.append({"Step": step, "What was done": detail, "Rows/cells affected": int(n)})

    # 1. Duplicates
    n = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    add("Remove duplicates", "Dropped exact duplicate customer rows", n)

    # 2. Inconsistent text
    before = df["gender"].copy()
    df["gender"] = df["gender"].str.strip().str.title()
    add("Standardise text", "Fixed casing / whitespace in 'gender' (e.g. 'FEMALE ', ' Male')", (before != df["gender"]).sum())

    # 3. Wrong dtype: TotalCharges has blank strings
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    add("Fix data type", "Converted 'TotalCharges' from text to number (blank strings -> NaN)", df["TotalCharges"].isna().sum())

    # 4. Outliers / entry errors in MonthlyCharges
    mask = df["MonthlyCharges"] > 125
    df.loc[mask, "MonthlyCharges"] = np.nan
    add("Handle outliers", "MonthlyCharges > 125 treated as entry errors and set to missing", mask.sum())

    # 5. Missing values
    n = df["Contract"].isna().sum()
    df["Contract"] = df["Contract"].fillna(df["Contract"].mode()[0])
    add("Impute categorical", "Missing 'Contract' filled with the most frequent value (mode)", n)

    n = df["MonthlyCharges"].isna().sum()
    df["MonthlyCharges"] = df["MonthlyCharges"].fillna(
        df.groupby(["InternetService", "PhoneService"])["MonthlyCharges"].transform("median"))
    add("Impute numeric", "Missing 'MonthlyCharges' filled with median of same InternetService + PhoneService group", n)

    n = df["AvgMonthlyDataGB"].isna().sum()
    df["AvgMonthlyDataGB"] = df["AvgMonthlyDataGB"].fillna(
        df.groupby("InternetService")["AvgMonthlyDataGB"].transform("median"))
    add("Impute numeric", "Missing 'AvgMonthlyDataGB' filled with median of the customer's InternetService group", n)

    n = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna((df["MonthlyCharges"] * df["tenure"]).round(2))
    add("Impute numeric", "Missing 'TotalCharges' estimated as MonthlyCharges x tenure", n)

    # 6. Feature engineering
    df["NumAddOns"] = (df[ADDON_COLS] == "Yes").sum(axis=1)
    df["ChargePerTenureMonth"] = (df["TotalCharges"] / df["tenure"]).round(2)
    df["TenureGroup"] = pd.cut(df["tenure"], [0, 12, 24, 48, 72],
                               labels=["0-12 m", "13-24 m", "25-48 m", "49-72 m"]).astype(str)
    add("Feature engineering", "Created NumAddOns, ChargePerTenureMonth and TenureGroup", len(df))

    # 7. Target encoding
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
    add("Encode target", "Created ChurnFlag (1 = churned, 0 = stayed)", len(df))

    return df, pd.DataFrame(log)


def get_data():
    raw = load_raw()
    clean_df, log = clean(raw)
    return raw, clean_df, log
