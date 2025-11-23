import os
from typing import Optional, Dict

import numpy as np
import pandas as pd

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py import discover_petri_net_inductive
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.visualization.petri_net import visualizer as pn_visualizer

try:
    # Prefer package-relative imports when run with -m
    from .utils import Utils as util
    from .config import (
        PROCESSED_DATA_PATH,
        PERFORMANCE_PATH,
        PERFORMANCE_LOG_PATH,
        RECOMMENDED_CURRICULUM,
    )
except ImportError:
    # Fallback for direct execution without -m
    from utils import Utils as util
    from config import (
        PROCESSED_DATA_PATH,
        PERFORMANCE_PATH,
        PERFORMANCE_LOG_PATH,
        RECOMMENDED_CURRICULUM,
    )

# Graphviz: allow override via env; keep Windows portable fallback for that OS only
graphviz_bin = os.environ.get("GRAPHVIZ_BIN")
if graphviz_bin:
    os.environ["PATH"] = graphviz_bin + os.pathsep + os.environ["PATH"]
elif os.name == "nt":
    portable_bin = r"C:\Users\deniz\Desktop\Code\CurriculumOptimizationPM02269\graphviz_portable\release\bin"
    os.environ["PATH"] = portable_bin + os.pathsep + os.environ["PATH"]


class PerformanceAnalysis:
    """
    Segment students into adherence × GPA groups and
    generate readable process models + summary statistics.

    Groups:
        - adherent_high_gpa
        - adherent_low_gpa
        - deviating_high_gpa
        - deviating_low_gpa
    """

    def __init__(
        self,
        processed_path: str = PROCESSED_DATA_PATH,
        results_dir: str = PERFORMANCE_PATH,
        gpa_high: float = 10.0,
        gpa_low: float = 6.0,
        adherence_tolerance: int = 1,
        min_on_time_ratio: float = 0.5,
        max_students_per_group: int = 20,
        random_seed: int = 42,
        restrict_to_curriculum: bool = True,
        min_activity_freq: int = 5,
        max_program_semester_for_model: int = 4,
    ):
        self.processed_path = processed_path
        self.results_dir = results_dir
        self.gpa_high = gpa_high
        self.gpa_low = gpa_low
        self.adherence_tolerance = adherence_tolerance
        self.min_on_time_ratio = min_on_time_ratio
        self.max_students_per_group = max_students_per_group
        self.random_seed = random_seed
        self.restrict_to_curriculum = restrict_to_curriculum
        self.min_activity_freq = min_activity_freq
        self.max_program_semester_for_model = max_program_semester_for_model

        # Directories
        self.group_dir = os.path.join(self.results_dir, "groups")
        self.showcase_dir = os.path.join(self.results_dir, "showcase")

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.group_dir, exist_ok=True)
        os.makedirs(self.showcase_dir, exist_ok=True)

        # Dataframes
        self.df: Optional[pd.DataFrame] = None          # event-level
        self.students: Optional[pd.DataFrame] = None    # student-level

        # RNG
        self.rng = np.random.RandomState(self.random_seed)

        # init log
        with open(PERFORMANCE_LOG_PATH, "w", encoding="utf-8", errors="replace") as f:
            f.write("=== PERFORMANCE ANALYSIS LOG ===\n\n")

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        self._load()
        self._compute_gpa()
        self._compute_curriculum_adherence()
        self._build_student_table()
        self._export_groups_and_models()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _log(self, title: str, content: str) -> None:
        with open(PERFORMANCE_LOG_PATH, "a", encoding="utf-8", errors="replace") as f:
            f.write(f"--- {title.upper()} ---\n")
            f.write(content.strip() + "\n\n")

    def _load(self) -> None:
        if not os.path.exists(self.processed_path):
            raise FileNotFoundError(f"Missing processed CSV: {self.processed_path}")

        df = pd.read_csv(self.processed_path)

        # Ensure key columns exist
        for col in ["student_id", "course_code", "grade_num", "Semester"]:
            if col not in df.columns:
                raise ValueError(f"processed data must include column '{col}'")

        df["grade_num"] = pd.to_numeric(df["grade_num"], errors="coerce")
        df["ects"] = pd.to_numeric(df.get("ects", np.nan), errors="coerce")

        if "grade_date" in df.columns:
            df["grade_date"] = pd.to_datetime(df["grade_date"], errors="coerce")
            df = df.sort_values(["student_id", "grade_date"])
        else:
            df = df.sort_values(["student_id", "Semester"])

        self.df = df

        self._log(
            "Load processed data",
            f"Rows: {len(df)}\nUnique students: {df['student_id'].nunique()}",
        )

    def _compute_gpa(self) -> None:
        df = self.df
        gpa = df.groupby("student_id")["grade_num"].mean().rename("gpa")
        self.df = df.merge(gpa, on="student_id", how="left")

        self._log(
            "Compute GPA",
            f"GPA computed for {gpa.shape[0]} students\n"
            f"GPA range: {gpa.min():.2f} – {gpa.max():.2f}",
        )

    def _compute_curriculum_adherence(self) -> None:
        df = self.df

        def assign_program_semesters(group: pd.DataFrame) -> pd.DataFrame:
            codes, uniques = pd.factorize(group["Semester"], sort=False)
            group = group.copy()
            group["program_semester"] = codes + 1
            return group

        df = (
            df.sort_values(["student_id", "grade_date"])
            .groupby("student_id", group_keys=False)
            .apply(assign_program_semesters)
        )

        def get_rec_semester(course_code: str) -> float:
            info: Dict = RECOMMENDED_CURRICULUM.get(str(course_code), None)
            if info is None:
                return np.nan
            return float(info.get("semester", np.nan))

        df["rec_semester"] = df["course_code"].astype(str).apply(get_rec_semester)
        df["sem_deviation"] = df["program_semester"] - df["rec_semester"]
        df["sem_deviation_abs"] = df["sem_deviation"].abs()
        df["on_time"] = df["sem_deviation_abs"] <= float(self.adherence_tolerance)

        self.df = df

        known = df["sem_deviation_abs"].notna().sum()
        self._log(
            "Curriculum adherence",
            f"Rows with known recommended semester: {known}/{len(df)}\n"
            f"Mean abs deviation: {df['sem_deviation_abs'].dropna().mean():.3f}",
        )

    def _build_student_table(self) -> None:
        df = self.df

        def failed_count(s):
            return int((s == False).sum())

        student_stats = df.groupby("student_id").agg(
            gpa=("gpa", "first"),
            n_events=("course_code", "size"),
            n_courses=("course_code", pd.Series.nunique),
            ects_total=("ects", "sum"),
            n_failures=("passed", failed_count)
            if "passed" in df.columns else ("grade_num", lambda s: 0),
            max_semester=("program_semester", "max"),
            # adherence_score and on_time_ratio are % on-time (0–1)
            on_time_ratio=("on_time", "mean"),
            adherence_score=("on_time", "mean"),
        )

        student_stats["adherent"] = (
            student_stats["on_time_ratio"] >= self.min_on_time_ratio
        )

        self.students = student_stats

        self._log(
            "Student-level table",
            f"Students total: {student_stats.shape[0]}\n"
            f"Adherent: {(student_stats['adherent'] == True).sum()}\n"
            f"Deviating: {(student_stats['adherent'] == False).sum()}",
        )

        master_csv = os.path.join(self.results_dir, "students_summary.csv")
        student_stats.to_csv(master_csv)
        self._log("Student summary export", f"Saved to {master_csv}")

    # ------------------------------------------------------------------ #
    # Model generation
    # ------------------------------------------------------------------ #

    def _export_groups_and_models(self) -> None:
        students = self.students

        groups = {
            "adherent_high_gpa": (students["adherent"] == True) & (students["gpa"] >= self.gpa_high),
            "adherent_low_gpa": (students["adherent"] == True) & (students["gpa"] <= self.gpa_low),
            "deviating_high_gpa": (students["adherent"] == False) & (students["gpa"] >= self.gpa_high),
            "deviating_low_gpa": (students["adherent"] == False) & (students["gpa"] <= self.gpa_low),
        }

        for name, mask in groups.items():
            sub_students = students[mask]

            if sub_students.empty:
                self._log(name, "No students in this group; skipping.")
                continue

            # sample students
            student_ids = sub_students.index.to_numpy()
            if len(student_ids) > self.max_students_per_group:
                sampled_ids = self.rng.choice(
                    student_ids, size=self.max_students_per_group, replace=False
                )
            else:
                sampled_ids = student_ids

            group_events = self.df[self.df["student_id"].isin(sampled_ids)].copy()

            # student summary
            group_students_csv = os.path.join(self.group_dir, f"{name}_students.csv")
            sub_students.loc[sampled_ids].to_csv(group_students_csv)

            group_log_header = (
                f"Students in group: {len(sub_students)} (sampled {len(sampled_ids)})\n"
                f"Saved student summary to: {group_students_csv}\n"
                f"GPA mean: {sub_students['gpa'].mean():.2f}\n"
                f"Adherence score mean: {sub_students['adherence_score'].mean():.3f}"
            )

            # export model (logging combined inside)
            self._export_filtered_model(name, group_events, group_log_header)

    # ------------------------------------------------------------------ #
    # CLEAN FILTERED VERSION OF MODEL EXPORT
    # ------------------------------------------------------------------ #

    def _export_filtered_model(self, name: str, df_sub: pd.DataFrame, header_log: str = "") -> None:
        """
        Apply multiple filters to reduce model complexity.
        Combine all messages into a single log block per group.
        """

        log_lines = []
        if header_log:
            log_lines.append(header_log)

        if df_sub.empty:
            log_lines.append("No events for model; skipping.")
            self._log(name, "\n".join(log_lines))
            return

        # 1) curriculum filter
        if self.restrict_to_curriculum:
            valid = set(RECOMMENDED_CURRICULUM.keys())
            before = len(df_sub)
            df_sub = df_sub[df_sub["course_code"].astype(str).isin(valid)]
            after = len(df_sub)
            log_lines.append(f"Curriculum filter: {before} -> {after}")

        # 2) semester restriction
        if (
            "program_semester" in df_sub.columns
            and self.max_program_semester_for_model is not None
        ):
            before = len(df_sub)
            df_sub = df_sub[df_sub["program_semester"] <= self.max_program_semester_for_model]
            after = len(df_sub)
            log_lines.append(
                f"Semester restriction: {before} -> {after} (max={self.max_program_semester_for_model})"
            )

        # 3) rare activity filter
        counts = df_sub["course_code"].value_counts()
        keep_acts = counts[counts >= self.min_activity_freq].index
        before = len(df_sub)
        df_sub = df_sub[df_sub["course_code"].isin(keep_acts)]
        after = len(df_sub)
        log_lines.append(
            f"Rare activity filter: {before} -> {after}, kept {len(keep_acts)} activities"
        )

        # 4) drop too-small traces
        df_sub = df_sub.groupby("student_id").filter(lambda g: len(g) >= 3)
        if df_sub.empty:
            log_lines.append("All traces too small; skipping.")
            self._log(name, "\n".join(log_lines))
            return

        # ------------ build PM4Py log ------------
        df_log = df_sub.copy()

        df_log = df_log.rename(
            columns={
                "student_id": "case:concept:name",
                "course_code": "concept:name",
                "grade_date": "time:timestamp",
            }
        )

        if "time:timestamp" in df_log.columns:
            df_log["time:timestamp"] = pd.to_datetime(df_log["time:timestamp"], errors="coerce")

        params = {
            log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY:
                "case:concept:name"
        }

        base_cols = ["case:concept:name", "time:timestamp", "concept:name"]
        extra_cols = [c for c in df_log.columns if c not in base_cols]
        df_log = df_log[base_cols + extra_cols]

        event_log = log_converter.apply(
            df_log,
            parameters=params,
            variant=log_converter.Variants.TO_EVENT_LOG,
        )

        # ------------ export XES ------------
        xes_path = os.path.join(self.group_dir, f"{name}.xes")
        xes_exporter.apply(event_log, xes_path)

        # ------------ discover model ------------
        net, im, fm = discover_petri_net_inductive(event_log)

        pnml_path = os.path.join(self.showcase_dir, f"{name}.pnml")
        pnml_exporter.apply(net, im, pnml_path, final_marking=fm)

        png_path = os.path.join(self.showcase_dir, f"{name}.png")
        gviz = pn_visualizer.apply(net, im, fm)
        pn_visualizer.save(gviz, png_path)

        log_lines.append(
            f"Cases: {len(event_log)}\n"
            f"Total events: {sum(len(t) for t in event_log)}\n"
            f"XES saved: {xes_path}\n"
            f"PNML saved: {pnml_path}\n"
            f"PNG saved: {png_path}"
        )

        self._log(name, "\n".join(log_lines))


# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    analysis = PerformanceAnalysis()
    analysis.run()
