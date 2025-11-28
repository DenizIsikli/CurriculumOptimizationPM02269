import os
import re
from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd

from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

try:
    # Prefer package-relative imports when run with -m
    from .utils import Utils as util
    from .config import (
        DATA_PATH,
        RAW_DATA_PATH,
        PROCESSED_DATA_PATH,
        SAMPLED_DATA_PATH,
        SAMPLE_FRACTION,
        XES_OUTPUT_PATH,
    )
except ImportError:
    # Fallback for direct execution without -m
    from utils import Utils as util
    from config import (
        DATA_PATH,
        RAW_DATA_PATH,
        PROCESSED_DATA_PATH,
        SAMPLED_DATA_PATH,
        SAMPLE_FRACTION,
        XES_OUTPUT_PATH,
    )


class DataPreparer:
    """
    Prepares DTU curriculum data for process mining:
    - Cleans raw data
    - Filters by program (optional)
    - Classifies pass/fail and numeric grade
    - Assigns semesters
    - Samples students
    - Exports processed CSV and XES event log
    """

    def __init__(
        self,
        program_filter: Optional[str] = None,
        raw_path: str = RAW_DATA_PATH,
        processed_path: str = PROCESSED_DATA_PATH,
        sampled_path: str = SAMPLED_DATA_PATH,
        sample_fraction: float = SAMPLE_FRACTION,
        xes_path: str = XES_OUTPUT_PATH,
    ):
        self.program_filter = program_filter
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.sampled_path = sampled_path
        self.sample_fraction = sample_fraction
        self.xes_path = xes_path

        self.df: Optional[pd.DataFrame] = None
        self.df_sample: Optional[pd.DataFrame] = None

        os.makedirs(DATA_PATH, exist_ok=True)
        self.log_path = os.path.join(DATA_PATH, "data_preparation_log.txt")
        self._init_log()

    def run(self) -> None:
        self._load_raw_data()
        self._clean_and_format()
        self._assign_semesters()
        self._sort_chronologically()
        self._classify_passes()
        self._assign_attempt_numbers()
        self._save_outputs()
        self._convert_to_event_log()

    def _init_log(self) -> None:
        with open(self.log_path, "w") as f:
            f.write(
                f"=== DATA PREPARATION LOG "
                f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n"
            )

    def _log(self, title: str, content: str) -> None:
        with open(self.log_path, "a") as f:
            f.write(f"--- {title.upper()} ---\n")
            f.write(content.strip() + "\n\n")

    def _load_raw_data(self) -> None:
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"Raw data file not found at {self.raw_path}")

        df = pd.read_excel(self.raw_path, header=1, dtype=str)
        original_rows, original_cols = df.shape

        rename_map = {
            "STUDIENR": "student_id",
            "UDDANNELSE": "education",
            "KURSKODE": "course_code",
            "KURSTXT": "course_name",
            "BEDOMMELSE": "grade",
            "SKALA": "scale",
            "ECTS": "ects",
            "UDPROVNING": "exam_type",
            "CENSUR": "examiner",
            "BEDOMMELSESDATO": "grade_date",
    }

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
            self._log("Missing Columns", f"Missing after renaming: {missing_cols}")

        df = df[[c for c in rename_map.values() if c in df.columns]]

        summary = (
            f"Raw rows: {original_rows}\n"
            f"Raw columns: {original_cols}\n"
            f"Columns after renaming and selection: {list(df.columns)}"
        )
        self._log("Load Raw Data", summary)
        self.df = df

    def _clean_and_format(self) -> None:
        df = self.df

        # Drop rows with missing key fields
        before = len(df)
        df = df.dropna(subset=["student_id", "course_code", "grade_date"]).copy()
        after = len(df)

        # ECTS numeric
        df["ects"] = pd.to_numeric(
            df["ects"].str.replace(",", ".", regex=False), errors="coerce"
        )

        # Normalize course codes (add leading 0 for 4-digit numeric codes)
        df["course_code"] = df["course_code"].apply(
            lambda x: f"0{x}"
            if isinstance(x, str) and len(x) == 4 and x.isdigit()
            else x
        )

        # Parse dates
        df["grade_date"] = pd.to_datetime(df["grade_date"], errors="coerce")

        # Filter by program if requested
        if self.program_filter:
            before_prog = len(df)
            df = df[df["education"] == self.program_filter].copy()
            after_prog = len(df)
            self._log(
                "Program Filter",
                f"Program filter: {self.program_filter}\n"
                f"Rows before filter: {before_prog}\n"
                f"Rows after filter:  {after_prog}",
            )

        self._log(
            "Clean and Format",
            f"Rows before dropping key nulls: {before}\n"
            f"Rows after dropping key nulls:  {after}\n"
            f"Unique students (current df):   {df['student_id'].nunique()}",
        )

        self.df = df

    def _assign_semesters(self) -> None:
        def semester_from_date(ts: pd.Timestamp) -> str:
            if pd.isna(ts):
                return "Unknown"
            y, m = ts.year, ts.month
            # Spring: Feb–Jul
            if m in (2, 3, 4, 5, 6, 7):
                return f"Spring {y}"
            # Autumn: Aug–Jan (Jan belongs to previous autumn)
            if m in (8, 9, 10, 11, 12):
                return f"Autumn {y}"
            if m == 1:
                return f"Autumn {y - 1}"
            return "Unknown"

        df = self.df
        df["Semester"] = df["grade_date"].apply(semester_from_date)

        unknown_count = (df["Semester"] == "Unknown").sum()
        summary = (
            f"Total rows: {len(df)}\n"
            f"Unique students: {df['student_id'].nunique()}\n"
            f"Rows with 'Unknown' semester: {unknown_count}"
        )
        self._log("Assign Semesters", summary)

        self.df = df

    def _sort_chronologically(self) -> None:
        def semester_sort_key(sem_text: str):
            season_order = {"Spring": 1, "Autumn": 2, "Unknown": 99}
            if not isinstance(sem_text, str) or not sem_text.strip():
                return 9999, 99
            m = re.search(r"(Spring|Autumn)\s+([12]\d{3})", sem_text, flags=re.IGNORECASE)
            if m:
                season = m.group(1).capitalize()
                year = int(m.group(2))
                return year, season_order.get(season, 50)
            m = re.search(r"([12]\d{3})", sem_text)
            if m:
                return int(m.group(1)), 50
            return 9999, 99

        def row_sort_key(row):
            dt = row["grade_date"]
            if pd.notna(dt):
                return dt.year, dt.month, 0
            y, o = semester_sort_key(row["Semester"])
            return y, o, 1

        df = self.df
        df = (
            df.assign(_key=df.apply(row_sort_key, axis=1))
            .sort_values("_key")
            .drop(columns=["_key"])
        )

        self._log(
            "Sort Chronologically",
            f"Rows after sorting: {len(df)}\n"
            f"Unique students:    {df['student_id'].nunique()}",
        )
        self.df = df

    def _classify_passes(self) -> None:
        df = self.df

        def normalize_text(x):
            return "" if pd.isna(x) else str(x).strip()

        def classify_pass(row):
            scale = normalize_text(row.get("scale", "")).casefold()
            grade_raw = normalize_text(row.get("grade", ""))
            grade = grade_raw.casefold()

            non_pass_tokens = {"ib", "ikke bestået", "ig", "em", "im"}
            sick_tokens = {"s", "syg"}

            # Drop sick here by returning a marker
            if grade in sick_tokens:
                return "SICK", np.nan

            # Pass/fail scale (e.g., "Bestået/Ikke Bestået")
            if "bestået" in scale:
                if grade.startswith("be") or grade == "bestået":
                    return True, np.nan
                if grade in non_pass_tokens:
                    return False, np.nan
                # Try numeric fallback if weird combos
                try:
                    gnum = float(grade.replace(",", "."))
                    return gnum >= 2.0, gnum
                except ValueError:
                    return np.nan, np.nan

            # 7-trinsskala
            if "7" in scale:
                if grade in non_pass_tokens:
                    return False, np.nan
                try:
                    gnum = float(grade.replace(",", "."))
                    return gnum >= 2.0, gnum
                except ValueError:
                    if grade.startswith("be"):
                        return True, np.nan
                    return np.nan, np.nan

            return np.nan, np.nan

        results = df.apply(lambda r: classify_pass(r), axis=1)
        df["pass_flag"] = results.apply(lambda x: x[0])
        df["grade_num"] = results.apply(lambda x: x[1])

        # Remove sick exam attempts
        before = len(df)
        df = df[df["pass_flag"] != "SICK"].copy()
        after = len(df)

        # Rename pass_flag to passed (bool / NaN)
        df["passed"] = df["pass_flag"].where(df["pass_flag"].isin([True, False]), np.nan)
        df = df.drop(columns=["pass_flag"])

        summary = (
            f"Rows before removing sick: {before}\n"
            f"Rows after removing sick:  {after}\n"
            f"Failed attempts:           {(df['passed'] == False).sum()}\n"
            f"Passed attempts:           {(df['passed'] == True).sum()}"
        )
        self._log("Classify Passes", summary)
        self.df = df

    def _assign_attempt_numbers(self) -> None:
        df = self.df

        df = df.sort_values(["student_id", "course_code", "grade_date"])
        df["attempt_no"] = (
            df.groupby(["student_id", "course_code"])
            .cumcount()
            .astype(int)
            + 1
        )

        summary = (
            f"Rows after attempt numbering: {len(df)}\n"
            f"Unique students:              {df['student_id'].nunique()}\n"
            f"Unique courses:               {df['course_code'].nunique()}"
        )
        self._log("Assign Attempt Numbers", summary)
        self.df = df

    def _save_outputs(self) -> None:
        df = self.df

        df.to_csv(self.processed_path, index=False)
        self._log(
            "Save Processed Data",
            f"Processed data saved to {self.processed_path}\n"
            f"Rows: {len(df)}\n"
            f"Students: {df['student_id'].nunique()}",
        )

        unique_students = df["student_id"].unique()
        n_students = len(unique_students)
        np.random.seed(42)
        sample_size = max(1, int(n_students * self.sample_fraction))

        sampled_students = np.random.choice(
            unique_students, size=sample_size, replace=False
        )
        df_sample = df[df["student_id"].isin(sampled_students)].copy()

        df_sample.to_csv(self.sampled_path, index=False)
        self._log(
            "Save Sampled Data",
            f"Sample fraction: {self.sample_fraction:.3f}\n"
            f"Students total:  {n_students}\n"
            f"Students sampled:{sample_size}\n"
            f"Rows sampled:    {len(df_sample)}\n"
            f"Sampled data saved to {self.sampled_path}",
        )

        self.df_sample = df_sample

    def _convert_to_event_log(self) -> None:
        df_log = self.df_sample.copy()

        # Ensure timestamps are proper datetime
        df_log = dataframe_utils.convert_timestamp_columns_in_df(df_log)

        # Rename columns for PM4Py
        df_log = df_log.rename(
            columns={
                "student_id": "case:concept:name",
                "course_code": "concept:name",
                "grade_date": "time:timestamp",
            }
        )

        event_attributes = [
            "grade",
            "passed",
            "grade_num",
            "ects",
            "Semester",
            "attempt_no",
            "exam_type",
            "education",
        ]
        event_attributes = [c for c in event_attributes if c in df_log.columns]

        parameters = {
            log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case:concept:name"
        }

        event_log = log_converter.apply(
            df_log[["case:concept:name", "time:timestamp", "concept:name"] + event_attributes],
            parameters=parameters,
            variant=log_converter.Variants.TO_EVENT_LOG,
        )

        xes_exporter.apply(event_log, self.xes_path)

        summary = (
            f"Event log exported to {self.xes_path}\n"
            f"Cases (students): {len(event_log)}\n"
            f"Total events:     {sum(len(t) for t in event_log)}"
        )
        self._log("Convert to Event Log", summary)


if __name__ == "__main__":
    preparer = DataPreparer(
        program_filter="Softwareteknologi, ingeniør bach.",
    )
    preparer.run()
