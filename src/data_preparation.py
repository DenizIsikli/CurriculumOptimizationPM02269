import pandas as pd
import numpy as np
import os
import re
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter

# File Paths
RAW_DATA_PATH = os.path.join("data", "DTU_dataset.xlsx")
PROCESSED_DATA_PATH = os.path.join("data", "processed_log.csv")
SAMPLED_DATA_PATH = os.path.join("data", "sampled_log.csv")


def prepare_data(sample_fraction: float = 0.10):
    """
    Load, clean, and prepare the DTU dataset for process mining.

    Args:
        sample_fraction (float): Fraction of dataset to keep (default = 0.10).

    Returns:
        event_log (pm4py EventLog): Cleaned and converted PM4Py event log.
    """

    print("Loading dataset...")
    df = pd.read_excel(RAW_DATA_PATH, header=1, dtype=str)
    print(f"Loaded dataset with {len(df):,} rows and {len(df.columns)} columns")

    # Rename Columns
    rename_map = {
        "STUDIENR": "student_id",
        "UDDANNELSE": "education",
        "KURSKODE": "course_code",
        "KURSTXT": "course_name",
        "BEDOMMELSE": "grade",
        "SKALA": "scale",
        "ECTS": "ects",
        "UDPROVNING": "exam_type",
        "CENSUR": "censorship",
        "BEDOMMELSESDATO": "exam_date"
    }

    # Normalize column names
    df.columns = (
        df.columns.str.strip()
        .str.upper()
        .str.replace("Ø", "O")
        .str.replace("Æ", "AE")
        .str.replace("Å", "A")
    )
    df = df.rename(columns=rename_map)

    missing_cols = [c for c in rename_map.values() if c not in df.columns]
    if missing_cols:
        print(f"⚠️ Warning: Missing columns after renaming: {missing_cols}")

    df = df[[c for c in rename_map.values() if c in df.columns]]

    print("Cleaning and formatting data...")

    # Drop rows missing key identifiers
    df = df.dropna(subset=["student_id", "course_code", "exam_date"])

    # Convert ECTS with comma decimals
    df["ects"] = pd.to_numeric(df["ects"].str.replace(",", ".", regex=False), errors="coerce")

    # Parse exam dates safely — auto-detect format to avoid warnings
    sample_vals = df["exam_date"].dropna().astype(str).head(50)

    def looks_iso_like(values: pd.Series) -> bool:
        """Return True if most date strings start with YYYY-MM-DD."""
        if values.empty:
            return False
        return values.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.8

    if looks_iso_like(sample_vals):
        # Mostly ISO format → parse normally (dayfirst=False)
        df["exam_date"] = pd.to_datetime(df["exam_date"], errors="coerce", dayfirst=False)
    else:
        # Likely Danish-style (DD/MM/YYYY) → parse with explicit format if possible
        try:
            df["exam_date"] = pd.to_datetime(df["exam_date"], format="%d/%m/%Y", errors="coerce")
        except Exception:
            df["exam_date"] = pd.to_datetime(df["exam_date"], errors="coerce", dayfirst=True)

    # If too many NaT values remain, retry without dayfirst
    if df["exam_date"].isna().mean() > 0.5:
        df["exam_date"] = pd.to_datetime(df["exam_date"], errors="coerce", dayfirst=False)

    # ----------------------
    # Semester Assignment
    # ----------------------
    def semester_from_date(ts: pd.Timestamp) -> str:
        if pd.isna(ts):
            return "Unknown"
        y, m = ts.year, ts.month
        if m in (2, 3, 4, 5, 6):
            return f"Spring {y}"
        if m in (8, 9, 10, 11, 12):
            return f"Autumn {y}"
        if m == 1:
            return f"Autumn {y - 1}"
        if m == 7:
            return f"Summer {y}"
        return "Unknown"

    df["Semester"] = df["exam_date"].apply(semester_from_date)

    # ----------------------
    # Sort Chronologically
    # ----------------------
    def semester_sort_key(sem_text: str):
        SEASON_ORDER = {"Spring": 1, "Summer": 2, "Autumn": 3, "Unknown": 99}
        if not isinstance(sem_text, str) or sem_text.strip() == "":
            return (9999, 99)
        m = re.search(r"(Spring|Summer|Autumn)\s+([12]\d{3})", sem_text, flags=re.IGNORECASE)
        if m:
            season = m.group(1).capitalize()
            year = int(m.group(2))
            return (year, SEASON_ORDER.get(season, 50))
        m = re.search(r"([12]\d{3})", sem_text)
        if m:
            return (int(m.group(1)), 50)
        return (9999, 99)

    def row_sort_key(row):
        dt = row["exam_date"]
        if pd.notna(dt):
            return (dt.year, dt.month, 0)
        y, o = semester_sort_key(row["Semester"])
        return (y, o, 1)

    df = df.assign(_key=df.apply(row_sort_key, axis=1)).sort_values("_key").drop(columns=["_key"])

    # ----------------------
    # Pass/Fail Classification
    # ----------------------
        # ----------------------
    # Pass Classification (Single Boolean Column)
    # ----------------------
    def normalize_text(x):
        return "" if pd.isna(x) else str(x).strip()

    def classify_pass(row):
        scale = normalize_text(row.get("scale", "")).casefold()
        grade = normalize_text(row.get("grade", "")).strip()
        non_pass_tokens = {"ib", "ikke bestået", "em", "im", "syg", "0", "00", "-3"}

        # Case 1: Pass/Fail scale
        if "bestået" in scale:
            g = grade.casefold()
            if g.startswith("be") or g == "bestået":
                return True, np.nan
            if g in non_pass_tokens:
                return False, np.nan
            return np.nan, np.nan

        # Case 2: 7-trinsskala
        if "7" in scale:
            gtxt = grade.replace(",", ".")
            try:
                gnum = float(gtxt)
            except ValueError:
                if grade.casefold().startswith("be"):
                    return True, np.nan
                return np.nan, np.nan
            if gnum >= 2:
                return True, gnum
            if gnum in (-3, 0) or gnum < 2:
                return False, gnum
            return np.nan, gnum

        # Unknown scale
        return np.nan, np.nan

    df[["passed", "grade_num"]] = df.apply(
        lambda r: pd.Series(classify_pass(r)), axis=1
    )

    # ----------------------
    # Attempt Numbering
    # ----------------------
    df = df.sort_values(["student_id", "course_code", "exam_date"])
    df["attempt_no"] = df.groupby(["student_id", "course_code"]).cumcount() + 1
    df["n_attempts"] = df.groupby(["student_id", "course_code"])["course_code"].transform("size")

    # ----------------------
    # Save Processed Files
    # ----------------------
    os.makedirs("data", exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    # Sample subset
    print(f"Sampling {sample_fraction * 100:.0f}% of dataset for development")
    df_sample = df.sample(frac=sample_fraction, random_state=42)
    df_sample.to_csv(SAMPLED_DATA_PATH, index=False)

    # ----------------------
    # Convert to PM4Py Event Log
    # ----------------------
    print("Converting DataFrame to PM4Py-compatible event log")
    df_sample = dataframe_utils.convert_timestamp_columns_in_df(df_sample)
    parameters = {
        log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "student_id"
    }
    event_log = log_converter.apply(df_sample, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)

    print(f"✅ Data preparation complete. Sample saved to '{SAMPLED_DATA_PATH}'")
    print(f"Processed dataset shape: {df_sample.shape}")

    return event_log


if __name__ == "__main__":
    log = prepare_data()
