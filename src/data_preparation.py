import os
import re
import numpy as np
import pandas as pd
from typing import Optional
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter

from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, SAMPLED_DATA_PATH, SAMPLE_FRACTION


class DataPreparer:
    """
    A high-level data preparation class for the DTU student dataset.
    Handles loading, cleaning, enrichment, sampling, and PM4Py conversion.
    """

    def __init__(
        self,
        raw_path: str = RAW_DATA_PATH,
        processed_path: str = PROCESSED_DATA_PATH,
        sampled_path: str = SAMPLED_DATA_PATH,
        sample_fraction: float = SAMPLE_FRACTION,
    ):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.sampled_path = sampled_path
        self.sample_fraction = sample_fraction
        self.df: Optional[pd.DataFrame] = None

    # Public API
    def prepare(self):
        """Main entry point: prepare dataset and return a PM4Py EventLog."""
        self._load_raw_data()
        self._clean_and_format()
        self._assign_semesters()
        self._sort_chronologically()
        self._classify_passes()
        self._assign_attempt_numbers()
        self._save_outputs()

        return self._convert_to_event_log()

    # Load Data
    def _load_raw_data(self) -> None:
        print("Loading dataset")
        self.df = pd.read_excel(self.raw_path, header=1, dtype=str)
        print(f"Loaded dataset with {len(self.df):,} rows and {len(self.df.columns)} columns")

        # Normalize column names
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
            "BEDOMMELSESDATO": "exam_date",
        }

        self.df.columns = (
            self.df.columns.str.strip()
            .str.upper()
            .str.replace("Ø", "O")
            .str.replace("Æ", "AE")
            .str.replace("Å", "A")
        )
        self.df = self.df.rename(columns=rename_map)

        missing_cols = [c for c in rename_map.values() if c not in self.df.columns]
        if missing_cols:
            print(f"Warning: Missing columns after renaming: {missing_cols}")

        self.df = self.df[[c for c in rename_map.values() if c in self.df.columns]]

    # Cleaning and Formatting
    def _clean_and_format(self) -> None:
        print("Cleaning and formatting data")

        df = self.df.dropna(subset=["student_id", "course_code", "exam_date"]).copy()
        df["ects"] = pd.to_numeric(df["ects"].str.replace(",", ".", regex=False), errors="coerce")

        # Detect date format automatically
        sample_vals = df["exam_date"].dropna().astype(str).head(50)
        if self._looks_iso_like(sample_vals):
            df["exam_date"] = pd.to_datetime(df["exam_date"], errors="coerce", dayfirst=False)
        else:
            try:
                df["exam_date"] = pd.to_datetime(df["exam_date"], format="%d/%m/%Y", errors="coerce")
            except Exception:
                df["exam_date"] = pd.to_datetime(df["exam_date"], errors="coerce", dayfirst=True)

        # Retry if too many invalid dates
        if df["exam_date"].isna().mean() > 0.5:
            df["exam_date"] = pd.to_datetime(df["exam_date"], errors="coerce", dayfirst=False)

        self.df = df

    @staticmethod
    def _looks_iso_like(values: pd.Series) -> bool:
        """Return True if most date strings are in ISO format (YYYY-MM-DD)."""
        if values.empty:
            return False
        return values.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.8

    # Semester Assignment
    def _assign_semesters(self) -> None:
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

        self.df["Semester"] = self.df["exam_date"].apply(semester_from_date)

    # Sorting
    def _sort_chronologically(self) -> None:
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

        self.df = (
            self.df.assign(_key=self.df.apply(row_sort_key, axis=1))
            .sort_values("_key")
            .drop(columns=["_key"])
        )

    # Pass/Fail Classification
    def _classify_passes(self) -> None:
        def normalize_text(x):
            return "" if pd.isna(x) else str(x).strip()

        def classify_pass(row):
            scale = normalize_text(row.get("scale", "")).casefold()
            grade = normalize_text(row.get("grade", "")).strip()
            non_pass_tokens = {"ib", "ikke bestået", "em", "im", "syg", "0", "00", "-3"}

            # Pass/fail scale
            if "bestået" in scale:
                g = grade.casefold()
                if g.startswith("be") or g == "bestået":
                    return True, np.nan
                if g in non_pass_tokens:
                    return False, np.nan
                return np.nan, np.nan

            # 7-trinsskala
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
                if gnum < 2:
                    return False, gnum
                return np.nan, gnum

            return np.nan, np.nan

        self.df[["passed", "grade_num"]] = self.df.apply(
            lambda r: pd.Series(classify_pass(r)), axis=1
        )

    # Attempt Numbering
    def _assign_attempt_numbers(self) -> None:
        self.df = self.df.sort_values(["student_id", "course_code", "exam_date"])
        self.df["attempt_no"] = self.df.groupby(["student_id", "course_code"]).cumcount() + 1
        self.df["n_attempts"] = self.df.groupby(["student_id", "course_code"])["course_code"].transform("size")

    # Saving and Conversion
    def _save_outputs(self) -> None:
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        self.df.to_csv(self.processed_path, index=False)
        print(f"Processed data saved to {self.processed_path}")

        df_sample = self.df.sample(frac=self.sample_fraction, random_state=42)
        df_sample.to_csv(self.sampled_path, index=False)
        print(f"Sampled data saved to {self.sampled_path}")

        self.df_sample = df_sample

    def _convert_to_event_log(self):
        print("Converting DataFrame to PM4Py-compatible event log")
        df_sample = dataframe_utils.convert_timestamp_columns_in_df(self.df_sample)
        parameters = {
            log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "student_id"
        }
        event_log = log_converter.apply(
            df_sample, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG
        )
        print(f"Data preparation complete. Sample shape: {self.df_sample.shape}")
        return event_log


if __name__ == "__main__":
    preparer = DataPreparer()
    event_log = preparer.prepare()
