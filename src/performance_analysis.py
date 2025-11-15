import os
import pandas as pd
from datetime import timedelta, datetime
import numpy as np
from scipy import stats
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.statistics.attributes.log import get as attributes_get

from config import PERFORMANCE_PATH, PERFORMANCE_LOG_PATH, XES_OUTPUT_PATH, PROCESSED_DATA_PATH


class PerformanceAnalyzer:
    def __init__(self, xes_path=XES_OUTPUT_PATH, output_dir=PERFORMANCE_PATH):
        self.xes_path = xes_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.log = xes_importer.apply(self.xes_path)
        self.df = pd.read_csv(PROCESSED_DATA_PATH)
        self.log_path = PERFORMANCE_LOG_PATH
        with open(self.log_path, "w", encoding="utf-8") as f:  # Add encoding="utf-8"
            f.write(f"=== PERFORMANCE ANALYSIS LOG ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n")

    def run(self):
        self._throughput_times()
        self._course_pass_rates()
        self._grade_distribution_analysis()
        self._retake_analysis()
        self._semester_load_analysis()
        self._time_between_courses()
        self._course_difficulty_ranking()
        self._gpa_analysis()
        self._summary_section()

    def _log_section(self, title: str, content: str):
        with open(self.log_path, "a", encoding="utf-8") as f:  # Add encoding="utf-8"
            f.write(f"--- {title.upper()} ---\n")
            f.write(content.strip() + "\n\n")
        print(f"\n{content}")

    def _throughput_times(self):
        """Calculate time from first to last course per student"""
        durations = case_statistics.get_all_case_durations(self.log, parameters={
            case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"
        })
        if not durations:
            self._log_section("Throughput Times", "No durations could be computed.")
            return

        durations_days = [d / 86400 for d in durations]
        avg_duration = np.mean(durations_days)
        median_duration = np.median(durations_days)
        
        summary = (
            f"Average time in program: {avg_duration:.1f} days ({avg_duration/365:.1f} years)\n"
            f"Median time in program: {median_duration:.1f} days ({median_duration/365:.1f} years)\n"
            f"Standard deviation: {np.std(durations_days):.1f} days\n"
            f"Shortest case: {min(durations_days):.1f} days\n"
            f"Longest case: {max(durations_days):.1f} days\n"
            f"Expected program duration: ~3 years (1095 days)\n"
            f"Students finishing within 3.5 years: {sum(1 for d in durations_days if d <= 1277) / len(durations_days):.1%}\n"
            f"Total analyzed cases: {len(durations)}"
        )
        self._log_section("Throughput Times", summary)

    def _course_pass_rates(self):
        """Analyze pass/fail rates per course"""
        if "passed" not in self.df.columns:
            self._log_section("Course Pass Rates", "No pass/fail data available.")
            return
        
        course_stats = self.df.groupby("course_code").agg({
            "passed": ["sum", "count", "mean"],
            "attempt_no": lambda x: (x > 1).sum()  # Number of retakes
        })
        course_stats.columns = ["passed_count", "total_attempts", "pass_rate", "retakes"]
        course_stats = course_stats[course_stats["total_attempts"] >= 5]  # Filter courses with <5 attempts
        course_stats = course_stats.sort_values("pass_rate")
        
        # Most difficult courses
        difficult = course_stats.head(10)
        difficult_summary = "\n".join([
            f"{i+1}. {course}: {row['pass_rate']:.1%} pass rate ({int(row['passed_count'])}/{int(row['total_attempts'])} attempts, {int(row['retakes'])} retakes)"
            for i, (course, row) in enumerate(difficult.iterrows())
        ])
        
        # Easiest courses
        easy = course_stats.tail(10)
        easy_summary = "\n".join([
            f"{i+1}. {course}: {row['pass_rate']:.1%} pass rate ({int(row['passed_count'])}/{int(row['total_attempts'])} attempts)"
            for i, (course, row) in enumerate(easy.iterrows())
        ])
        
        overall_pass_rate = self.df["passed"].sum() / len(self.df[self.df["passed"].notna()])
        
        summary = (
            f"Overall pass rate: {overall_pass_rate:.1%}\n"
            f"Total courses analyzed: {len(course_stats)}\n\n"
            f"Top 10 Most Difficult Courses (lowest pass rate):\n{difficult_summary}\n\n"
            f"Top 10 Easiest Courses (highest pass rate):\n{easy_summary}"
        )
        self._log_section("Course Pass Rates", summary)

    def _grade_distribution_analysis(self):
        """Analyze grade distributions"""
        if "grade_num" not in self.df.columns:
            self._log_section("Grade Distribution", "No numeric grades available.")
            return
        
        valid_grades = self.df[self.df["grade_num"].notna()]["grade_num"]
        
        # Grade scale: 12, 10, 7, 4, 02, 00, -3
        grade_counts = valid_grades.value_counts().sort_index(ascending=False)
        distribution = "\n".join([f"Grade {int(grade)}: {count} ({count/len(valid_grades):.1%})" 
                                  for grade, count in grade_counts.items()])
        
        summary = (
            f"Total graded attempts: {len(valid_grades)}\n"
            f"Average grade: {valid_grades.mean():.2f}\n"
            f"Median grade: {valid_grades.median():.2f}\n"
            f"Standard deviation: {valid_grades.std():.2f}\n\n"
            f"Grade Distribution:\n{distribution}\n\n"
            f"Passing grades (≥2): {(valid_grades >= 2).sum()} ({(valid_grades >= 2).sum()/len(valid_grades):.1%})\n"
            f"Excellent grades (≥10): {(valid_grades >= 10).sum()} ({(valid_grades >= 10).sum()/len(valid_grades):.1%})"
        )
        self._log_section("Grade Distribution", summary)

    def _retake_analysis(self):
        """Analyze course retake patterns"""
        if "attempt_no" not in self.df.columns:
            self._log_section("Retake Analysis", "No attempt number data available.")
            return
        
        retakes = self.df[self.df["attempt_no"] > 1]
        students_with_retakes = retakes["student_id"].nunique()
        total_students = self.df["student_id"].nunique()
        
        # Courses with most retakes
        retake_courses = retakes.groupby("course_code").size().sort_values(ascending=False).head(10)
        retake_summary = "\n".join([f"{i+1}. {course}: {count} retakes" 
                                    for i, (course, count) in enumerate(retake_courses.items())])
        
        # Students by retake count
        student_retakes = retakes.groupby("student_id").size()
        
        summary = (
            f"Students with retakes: {students_with_retakes} ({students_with_retakes/total_students:.1%})\n"
            f"Total retake attempts: {len(retakes)}\n"
            f"Average retakes per student (who retook): {student_retakes.mean():.1f}\n"
            f"Max retakes by single student: {student_retakes.max()}\n\n"
            f"Top 10 Most Retaken Courses:\n{retake_summary}"
        )
        self._log_section("Retake Analysis", summary)

    def _semester_load_analysis(self):
        """Analyze ECTS load per semester"""
        if "ects" not in self.df.columns or "Semester" not in self.df.columns:
            self._log_section("Semester Load", "No ECTS/semester data available.")
            return
        
        # Only count passed courses for ECTS
        passed_df = self.df[self.df["passed"] == True].copy()
        
        semester_load = passed_df.groupby(["student_id", "Semester"])["ects"].sum()
        
        summary = (
            f"Average ECTS per semester: {semester_load.mean():.1f}\n"
            f"Median ECTS per semester: {semester_load.median():.1f}\n"
            f"Recommended ECTS per semester: 30\n"
            f"Students averaging ≥30 ECTS: {(semester_load.groupby('student_id').mean() >= 30).sum()}\n"
            f"Students averaging <20 ECTS: {(semester_load.groupby('student_id').mean() < 20).sum()}\n"
            f"Minimum semester load: {semester_load.min():.1f} ECTS\n"
            f"Maximum semester load: {semester_load.max():.1f} ECTS"
        )
        self._log_section("Semester Load Analysis", summary)

    def _time_between_courses(self):
        """Analyze time gaps between course completions"""
        time_gaps = []
        
        for trace in self.log:
            if len(trace) < 2:
                continue
            for i in range(len(trace) - 1):
                t1 = trace[i]["time:timestamp"]
                t2 = trace[i + 1]["time:timestamp"]
                gap_days = (t2 - t1).total_seconds() / 86400
                time_gaps.append(gap_days)
        
        if not time_gaps:
            self._log_section("Time Between Courses", "No time gaps could be calculated.")
            return
        
        summary = (
            f"Average time between courses: {np.mean(time_gaps):.1f} days\n"
            f"Median time between courses: {np.median(time_gaps):.1f} days\n"
            f"Standard deviation: {np.std(time_gaps):.1f} days\n"
            f"Shortest gap: {min(time_gaps):.1f} days\n"
            f"Longest gap: {max(time_gaps):.1f} days\n"
            f"Gaps > 6 months: {sum(1 for g in time_gaps if g > 180)} ({sum(1 for g in time_gaps if g > 180)/len(time_gaps):.1%})"
        )
        self._log_section("Time Between Courses", summary)

    def _course_difficulty_ranking(self):
        """Rank courses by combined metrics (fail rate + avg grade + retakes)"""
        if "passed" not in self.df.columns or "grade_num" not in self.df.columns:
            self._log_section("Course Difficulty Ranking", "Insufficient data.")
            return
        
        course_metrics = self.df.groupby("course_code").agg({
            "passed": lambda x: 1 - x.mean(),  # Fail rate
            "grade_num": "mean",
            "attempt_no": lambda x: (x > 1).sum() / len(x)  # Retake rate
        }).rename(columns={
            "passed": "fail_rate",
            "grade_num": "avg_grade",
            "attempt_no": "retake_rate"
        })
        
        # Normalize and combine (lower score = harder course)
        course_metrics["difficulty_score"] = (
            course_metrics["fail_rate"] * 0.5 +
            (12 - course_metrics["avg_grade"]) / 12 * 0.3 +
            course_metrics["retake_rate"] * 0.2
        )
        
        hardest = course_metrics.nlargest(10, "difficulty_score")
        hardest_summary = "\n".join([
            f"{i+1}. {course}: difficulty={row['difficulty_score']:.3f} (fail={row['fail_rate']:.1%}, avg grade={row['avg_grade']:.2f})"
            for i, (course, row) in enumerate(hardest.iterrows())
        ])
        
        summary = f"Top 10 Hardest Courses (composite difficulty score):\n{hardest_summary}"
        self._log_section("Course Difficulty Ranking", summary)

    def _gpa_analysis(self):
        """Analyze GPA if available"""
        if "cumulative_gpa" in self.df.columns:
            student_gpa = self.df.groupby("student_id")["cumulative_gpa"].first()
            
            summary = (
                f"Average GPA: {student_gpa.mean():.2f}\n"
                f"Median GPA: {student_gpa.median():.2f}\n"
                f"Standard deviation: {student_gpa.std():.2f}\n"
                f"Highest GPA: {student_gpa.max():.2f}\n"
                f"Lowest GPA: {student_gpa.min():.2f}\n"
                f"Students with GPA ≥ 10: {(student_gpa >= 10).sum()} ({(student_gpa >= 10).sum()/len(student_gpa):.1%})\n"
                f"Students with GPA < 4: {(student_gpa < 4).sum()} ({(student_gpa < 4).sum()/len(student_gpa):.1%})"
            )
            self._log_section("GPA Analysis", summary)
        else:
            self._log_section("GPA Analysis", "GPA data not available. Run data_preparation with GPA calculation.")

    def _summary_section(self):
        num_traces = len(self.log)
        total_events = sum(len(trace) for trace in self.log)
        avg_events_per_trace = total_events / num_traces if num_traces else 0

        content = (
            f"Number of traces (students): {num_traces}\n"
            f"Total events (course attempts): {total_events}\n"
            f"Average events per trace: {avg_events_per_trace:.2f}\n"
            f"Unique courses in dataset: {self.df['course_code'].nunique()}"
        )
        print(content)
        self._log_section("Overall Log Summary", content)


if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    analyzer.run()