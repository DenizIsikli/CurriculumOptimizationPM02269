import os
import re
from typing import Optional
import datetime
import numpy as np
import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, SAMPLED_DATA_PATH, SAMPLE_FRACTION, XES_OUTPUT_PATH


class DataPreparer:
    def __init__(
        self,
        program_filter: Optional[str] = None,
        raw_path: str = RAW_DATA_PATH,
        processed_path: str = PROCESSED_DATA_PATH,
        sampled_path: str = SAMPLED_DATA_PATH,
        sample_fraction: float = SAMPLE_FRACTION,
        xes_path: str = XES_OUTPUT_PATH
    ):
        self.program_filter = program_filter
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.sampled_path = sampled_path
        self.sample_fraction = sample_fraction
        self.xes_path = xes_path
        self.df: Optional[pd.DataFrame] = None

    def run(self):
        self._load_raw_data()
        self._clean_and_format()
        self._assign_semesters()
        self._sort_chronologically()
        self._classify_passes()
        self._assign_attempt_numbers()
        self._save_outputs()
        self._convert_to_event_log()

    def _load_raw_data(self) -> None:
        # Check if raw data file exists        
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"Raw data file not found at {self.raw_path}")
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
            "CENSUR": "examiner",
            "BEDOMMELSESDATO": "grade_date",
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

    def _clean_and_format(self) -> None:
        print("Cleaning and formatting data")

        df = self.df.dropna(subset=["student_id", "course_code", "grade_date"]).copy()        
        df["ects"] = pd.to_numeric(df["ects"].str.replace(",", ".", regex=False), errors="coerce")
         
        df["grade_date"] = pd.to_datetime(
            df["grade_date"],
            errors="coerce")
        
        # add leading zero to course codes with only 4 digits
        df["course_code"] = df["course_code"].apply(
            lambda x: f"0{x}" if isinstance(x, str) and len(x) == 4 and x.isdigit() else x
        )

        if self.program_filter:
            df = df[df["education"] == self.program_filter].copy()
            print(f"Filtered data to program '{self.program_filter}', resulting in {len(df):,} rows")
        self.df = df
     # around 89% of grade_date are in 5,6,7 or 1,2,12 so merge all the 3 week courses into the main semesters
    def _assign_semesters(self) -> None:
        def semester_from_date(ts: pd.Timestamp) -> str:
            if pd.isna(ts):
                return "Unknown"
            y, m, d = ts.year, ts.month, ts.day
            if m in (5,6,7):
                return f"Spring {y}"
            elif m in (12,1,2):
                return f"Autumn {y}" if m == 12 else f"Autumn {y-1}"
            else:
                return "Unknown"

        self.df["Semester"] = self.df["grade_date"].apply(semester_from_date)
        
        # Drop students with Unknown semester TODO investigate later
        unknown_students = self.df[self.df["Semester"] == "Unknown"]["student_id"].unique()
        before = len(self.df)
        self.df = self.df[~self.df["student_id"].isin(unknown_students)].copy()
        print(f"Removed {len(unknown_students)} students with Unknown semester, {before - len(self.df):,} rows dropped")
    
    
        # add start date and end date columns 
        self.df["semester_start"] = self.df["Semester"].apply(lambda s: pd.Timestamp(year=int(s.split()[1]), month=2, day=1) 
                                      if "Spring" in s else (pd.Timestamp(year=int(s.split()[1]), month=9, day=1) 
                                                 if "Autumn" in s else pd.NaT))
        self.df["semester_end"] = self.df["Semester"].apply(lambda s: pd.Timestamp(year=int(s.split()[1]), month=5, day=1) 
                                    if "Spring" in s else (pd.Timestamp(year=int(s.split()[1]), month=12, day=1) 
                                               if "Autumn" in s else pd.NaT))
        self.df["semester_start"].fillna(self.df["grade_date"], inplace=True)
        self.df["semester_end"].fillna(self.df["grade_date"], inplace=True)
        

    def _sort_chronologically(self) -> None:
        def semester_sort_key(sem_text: str):
            SEASON_ORDER = {"Spring": 1, "Autumn": 2, "Unknown": 99}
            if not isinstance(sem_text, str) or sem_text.strip() == "":
                return (9999, 99)
            m = re.search(r"(Spring|Autumn)\s+([12]\d{3})", sem_text, flags=re.IGNORECASE)
            if m:
                season = m.group(1).capitalize()
                year = int(m.group(2))
                return (year, SEASON_ORDER.get(season, 50))
            m = re.search(r"([12]\d{3})", sem_text)
            if m:
                return (int(m.group(1)), 50)
            return (9999, 99)

        def row_sort_key(row):
            dt = row["grade_date"]
            if pd.notna(dt):
                return (dt.year, dt.month, 0)
            y, o = semester_sort_key(row["Semester"])
            return (y, o, 1)

        self.df = (
            self.df.assign(_key=self.df.apply(row_sort_key, axis=1))
            .sort_values("_key")
            .drop(columns=["_key"])
        )
    
    # TODO add number grade to non-numeric grades 
    def _classify_passes(self) -> None:
        def normalize_text(x):
            return "" if pd.isna(x) else str(x).strip()

        def classify_pass(row):
            scale = normalize_text(row.get("scale", "")).casefold()
            grade = normalize_text(row.get("grade", "")).casefold()
            non_pass_tokens = {"ib", "ikke bestået","ig", "em", "im"}
        # IG 
            # Pass/fail scale
            if "bestået" in scale:
                if grade.startswith("be") or grade == "bestået":
                    return True, float(12)
                if grade in non_pass_tokens:
                    return False, float(-3)
                else :
                    try:
                        gtxt = grade.replace(",", ".")
                        gnum = float(gtxt)
                        return (gnum >= 2), gnum
                    except ValueError:
                        return np.nan, np.nan

            # 7-trinsskala
            if "7" in scale:
                # Check for non-pass tokens first
                if grade in non_pass_tokens:
                    return False, float(-3)
                
                gtxt = grade.replace(",", ".")
                try:
                    gnum = float(gtxt)
                    return (gnum >= 2), gnum
                except ValueError:
                    if grade.startswith("be"):
                        return True, float(12)
                    return np.nan, np.nan

            return np.nan, np.nan

        self.df[["passed", "grade_num"]] = self.df.apply(
            lambda r: pd.Series(classify_pass(r)), axis=1
        )
        
        # Drop exam attempts where student was sick
        sick_tokens = {"s", "syg"}
        before = len(self.df)
        self.df = self.df[~self.df["grade"].str.strip().str.lower().isin(sick_tokens)].copy()
        print(f"Dropped {before - len(self.df):,} sick exam attempts")

    def _assign_attempt_numbers(self) -> None:
        course_counts = self.df.groupby("student_id")["course_code"].nunique()
        valid_students = course_counts[course_counts >= 3].index
        before = len(self.df)
        self.df = self.df[self.df["student_id"].isin(valid_students)].copy()
        print(f"Removed students with <3 courses, {before - len(self.df):,} rows dropped")

        self.df = self.df.sort_values(["student_id", "course_code", "grade_date"])
        self.df["attempt_no"] = self.df.groupby(["student_id", "course_code"]).cumcount() + 1
        # self.df["n_attempts"] = self.df.groupby(["student_id", "course_code"])["course_code"].transform("size")

    def _save_outputs(self) -> None:
        self.df.to_csv(self.processed_path, index=False)
        print(f"Processed data saved to {self.processed_path}")

        # Sample by student identifier (ensures complete traces)
        unique_students = self.df["student_id"].unique()
        sampled_students = np.random.choice(unique_students, size=int(len(unique_students) * self.sample_fraction), replace=False)
        df_sample = self.df[self.df["student_id"].isin(sampled_students)].copy()
        df_sample.to_csv(self.sampled_path, index=False)
        print(f"Sampled data saved to {self.sampled_path}")

        self.df_sample = df_sample

    def _convert_to_event_log(self):
        print("Converting DataFrame to PM4Py-compatible event log")
        df_log = self.df_sample.copy()

        df_sample = dataframe_utils.convert_timestamp_columns_in_df(self.df_sample)
        df_log = df_log.rename(columns={
                "semester_start": "time:timestamp",
                "course_code": "concept:name",
                "student_id": "case:concept:name"
            })
        event_attributes = ["grade", "passed", "grade_num", "ects", "Semester", "attempt_no", "exam_type"]
        parameters = {
        log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case:concept:name"
        }
        event_log = log_converter.apply(
        df_log[["case:concept:name", "time:timestamp", "concept:name"] + event_attributes],
        parameters=parameters,
        variant=log_converter.Variants.TO_EVENT_LOG
    )
        xes_exporter.apply(event_log, self.xes_path)
        print(f"Event log exported to {self.xes_path}")

if __name__ == "__main__":
    run = DataPreparer(program_filter="Matematisk modellering og computing, cand.polyt.",sample_fraction=1)
    run.run()

    # open the csv file and store a sorted list of course codes with highest fail exam attempts
    df = pd.read_csv(PROCESSED_DATA_PATH)
    fail_counts = (
        df[df["passed"] == False]
        .groupby("course_code")
        .size()
        .sort_values(ascending=False)
    )
    
    print(fail_counts.head(2))
    top_fail_course = fail_counts.index[0]
    print(f"Top failed course: {top_fail_course} ")

    # create a dict of students who failed the top failed course 
    students_dict = (
    df[df["course_code"] == top_fail_course]
    .groupby("student_id")["passed"]
    .apply(lambda x: (~x).sum())
    .sort_values(ascending=False)
    .to_dict()
)

    students_dict = dict(sorted(students_dict.items(), key=lambda item: item[1], reverse=True))
    # save the students dict 
    path = os.path.join(DATA_PATH, "students_failures_" + top_fail_course + ".csv")
    with open(path, "w") as f:
        f.write("student_id,failed_attempts\n")
        for student_id, failed_attempts in students_dict.items():
            f.write(f"{student_id},{failed_attempts}\n")
    
    # save failed courses to a csv file
    path = os.path.join(DATA_PATH, "top_failed_course.csv")
    fail_counts.to_csv(path)
    
    df = pd.read_csv(os.path.join(DATA_PATH, "sampled_log.csv"))
    # print the number  nan or not a date in every column

    print(f"Total rows after all filters: {len(df)}")
    print(f"Total failed attempts: {(~df['passed']).sum()}")
    print(f"Students remaining: {df['student_id'].nunique()}")

    print(df['grade'].value_counts())
    print(df['passed'].value_counts())
    print(df[~df['passed']]['grade'].value_counts())

    print("number of students", df['student_id'].nunique())


