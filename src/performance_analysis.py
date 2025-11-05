import os
import pandas as pd
from datetime import timedelta, datetime
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.statistics.attributes.log import get as attributes_get
from pm4py.visualization.performance_spectrum import visualizer as ps_visualizer

from src.config import PERFORMANCE_PATH, PERFORMANCE_LOG_PATH, XES_OUTPUT_PATH


class PerformanceAnalyzer:
    def __init__(self, xes_path=XES_OUTPUT_PATH, output_dir=PERFORMANCE_PATH):
        self.xes_path = xes_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.log = xes_importer.apply(self.xes_path)
        self.log_path = PERFORMANCE_LOG_PATH
        with open(self.log_path, "w") as f:
            f.write(f"=== PERFORMANCE ANALYSIS LOG ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n")

    def run(self):
        self._throughput_times()
        self._activity_frequencies()
        self._summary_section()

    def _log_section(self, title: str, content: str):
        with open(self.log_path, "a") as f:
            f.write(f"--- {title.upper()} ---\n")
            f.write(content.strip() + "\n\n")

    def _throughput_times(self):
        durations = case_statistics.get_all_case_durations(self.log, parameters={
            case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"
        })
        if not durations:
            self._log_section("Throughput Times", "No durations could be computed.")
            return

        avg_duration = sum(durations) / len(durations)
        summary = (
            f"Average throughput time: {avg_duration / 86400:.2f} days\n"
            f"Shortest case: {min(durations) / 86400:.2f} days\n"
            f"Longest case: {max(durations) / 86400:.2f} days\n"
            f"Total analyzed cases: {len(durations)}"
        )
        self._log_section("Throughput Times", summary)


    def _activity_frequencies(self):
        freq = attributes_get.get_attribute_values(self.log, "concept:name")
        if not freq:
            self._log_section("Activity Frequencies", "No activity frequencies available.")
            return

        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        top10 = "\n".join([f"{i+1}. {k} — {v} occurrences" for i, (k, v) in enumerate(sorted_freq[:10])])
        total_unique = len(sorted_freq)
        total_events = sum(v for _, v in sorted_freq)

        summary = (
            f"Total unique activities (courses): {total_unique}\n"
            f"Total events in log: {total_events}\n\n"
            f"Top 10 most frequent activities:\n{top10}"
        )
        self._log_section("Activity Frequencies", summary)

    def _summary_section(self):
        num_traces = len(self.log)
        total_events = sum(len(trace) for trace in self.log)
        avg_events_per_trace = total_events / num_traces if num_traces else 0

        content = (
            f"Number of traces (students): {num_traces}\n"
            f"Total events (course attempts): {total_events}\n"
            f"Average events per trace: {avg_events_per_trace:.2f}"
        )
        print(content)
        self._log_section("Overall Log Summary", content)


if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    analyzer.run()
