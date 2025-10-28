import pandas as pd
import os
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
    df = pd.read_excel(RAW_DATA_PATH, header=1)
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

    df.columns = df.columns.str.strip().str.upper().str.replace("Ø", "O").str.replace("Æ", "AE").str.replace("Å", "A")
    df = df.rename(columns=rename_map)
    missing_cols = [c for c in rename_map.values() if c not in df.columns]
    if missing_cols:
        print(f"⚠️ Warning: Missing columns after renaming: {missing_cols}")
    df = df[[c for c in rename_map.values() if c in df.columns]]



    # Clean and Format Data
    print("Cleaning and formatting data")

    # Drop rows with missing core identifiers or timestamps
    df = df.dropna(subset=["student_id", "course_code", "exam_date"])

    # Convert exam date to datetime
    df["exam_date"] = pd.to_datetime(df["exam_date"], errors="coerce")

    # Strip whitespace from course numbers
    df["course_code"] = df["course_code"].astype(str).str.strip()

    # Convert grades to numeric
    df["grade"] = df["grade"].replace({
        "Bestået": 2, 
        "Ikke bestået": 0,
        "U": 0,
        "NA": None
    })

    # Convert all numeric grades to numeric dtype
    df["grade"] = pd.to_numeric(df["grade"], errors="coerce")

    # Sample Subset for Testing
    print(f"Sampling {sample_fraction * 100:.0f}% of dataset for development")
    df_sample = df.sample(frac=sample_fraction, random_state=42)

    # PM4Py Event Log Conversion
    print("Converting DataFrame to PM4Py-compatible event log")
    df_sample = dataframe_utils.convert_timestamp_columns_in_df(df_sample)

    # Define case identifier for process instances (each student's study path)
    parameters = {
        log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "student_id"
    }

    event_log = log_converter.apply(df_sample, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)

    os.makedirs("data", exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    df_sample.to_csv(SAMPLED_DATA_PATH, index=False)

    print(f"Data preparation complete. Sample saved to '{SAMPLED_DATA_PATH}'")
    print(f"Processed dataset shape: {df_sample.shape}")

    return event_log


if __name__ == "__main__":
    log = prepare_data()
