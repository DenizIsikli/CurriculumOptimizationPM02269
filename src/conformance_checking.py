import os
from datetime import datetime
from typing import Dict

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments

try:
    from .config import (
        CONFORMANCE_PATH,
        CONFORMANCE_LOG_PATH,
        IM_MODEL_PATH,
        PERFORMANCE_PATH,
    )
except ImportError:
    from config import (
        CONFORMANCE_PATH,
        CONFORMANCE_LOG_PATH,
        IM_MODEL_PATH,
        PERFORMANCE_PATH,
    )


# Defaults are config-driven so you can swap models without editing code
REFERENCE_MODEL = IM_MODEL_PATH
GROUP_LOG_DIR = os.path.join(PERFORMANCE_PATH, "groups")
OUTPUT_REPORT = os.path.join(CONFORMANCE_PATH, "conformance_report.txt")

# Limits to avoid runaway alignment runtime
ALIGNMENT_MAX_TIME = 25    # seconds (global cap; aligns can explode)
ALIGNMENT_MAX_TRACES = 20  # limit traces to keep runtime bounded
FITNESS_FIT_EPS = 1e-6     # tolerance for treating a trace as perfectly fitting

GROUP_LOGS: Dict[str, str] = {
    "adherent_high_gpa": "adherent_high_gpa.xes",
    "adherent_low_gpa": "adherent_low_gpa.xes",
    "deviating_high_gpa": "deviating_high_gpa.xes",
    "deviating_low_gpa": "deviating_low_gpa.xes",
}

def write(report_path, text):
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


class ConformanceChecker:
    def __init__(self, model_path, group_logs, log_dir, report_file):
        self.model_path = model_path
        self.group_logs = group_logs
        self.log_dir = log_dir
        self.report_file = report_file

        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"=== CONFORMANCE CHECK REPORT ({datetime.now()}) ===\n\n")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model PNML not found: {self.model_path}")

        self.net, self.im, self.fm = pnml_importer.apply(self.model_path)

    def run(self):
        for group_name, filename in self.group_logs.items():
            log_path = os.path.join(self.log_dir, filename)

            if not os.path.exists(log_path):
                write(self.report_file, f"[{group_name}] XES not found → skipping\n")
                continue

            try:
                log = xes_importer.apply(log_path)
            except Exception as e:  # pragma: no cover - PM4Py import errors
                write(self.report_file, f"[{group_name}] Failed to import XES: {e}\n")
                continue

            if len(log) == 0:
                write(self.report_file, f"[{group_name}] Log is empty → skipping\n")
                continue

            if len(log) > ALIGNMENT_MAX_TRACES:
                log = log[:ALIGNMENT_MAX_TRACES]
                write(
                    self.report_file,
                    f"[{group_name}] Truncated log to first {ALIGNMENT_MAX_TRACES} traces for alignment runtime control",
                )

            write(self.report_file, f"--- {group_name.upper()} ---")
            write(self.report_file, f"Traces in log: {len(log)}")

            self.token_replay_fitness(log, group_name)
            self.alignment_fitness(log, group_name)
            write(self.report_file, "\n")

    def token_replay_fitness(self, log, group):
        results = token_replay.apply(log, self.net, self.im, self.fm)
        fitness = [r["trace_fitness"] for r in results]
        if not fitness:
            write(self.report_file, "Token Replay Fitness: no results")
            return

        avg_fit = sum(fitness) / len(fitness)
        write(
            self.report_file,
            f"Token Replay Fitness: avg={avg_fit:.3f}, min={min(fitness):.3f}, max={max(fitness):.3f}",
        )

    def alignment_fitness(self, log, group):
        params = {
            alignments.Parameters.PARAM_MAX_ALIGN_TIME: ALIGNMENT_MAX_TIME,
            alignments.Parameters.PARAM_MAX_ALIGN_TIME_TRACE: ALIGNMENT_MAX_TIME,
            alignments.Parameters.SHOW_PROGRESS_BAR: False,
        }

        try:
            align_res = alignments.apply(log, self.net, self.im, self.fm, parameters=params)
        except Exception as e:  # pragma: no cover - alignment can fail on unsound nets
            write(self.report_file, f"Alignment failed: {e}")
            return

        if not align_res:
            write(self.report_file, "Alignment Fitness: no results (timeout or empty)")
            return

        valid_align = [a for a in align_res if isinstance(a, dict)]
        fitness_values = [a.get("fitness") for a in valid_align if "fitness" in a]

        if not fitness_values:
            write(self.report_file, "Alignment Fitness: no fitness values computed")
            return

        avg_fit = sum(fitness_values) / len(fitness_values)
        fit_count = sum(f >= (1 - FITNESS_FIT_EPS) for f in fitness_values)
        perc_fitting = 100 * fit_count / len(fitness_values)

        fitness_sorted = sorted(enumerate(fitness_values), key=lambda x: x[1])
        worst_examples = fitness_sorted[: min(3, len(fitness_sorted))]
        worst_str = ", ".join(
            [f"trace#{idx}: {val:.3f}" for idx, val in worst_examples]
        )

        write(
            self.report_file,
            f"Alignment Fitness: avg={avg_fit:.3f}, min={min(fitness_values):.3f}, max={max(fitness_values):.3f}",
        )
        write(self.report_file, f"Percentage Fitting Traces: {perc_fitting:.1f}%")
        write(self.report_file, f"Worst traces (index:fitness): {worst_str}")


if __name__ == "__main__":
    checker = ConformanceChecker(
        model_path=REFERENCE_MODEL,
        group_logs=GROUP_LOGS,
        log_dir=GROUP_LOG_DIR,
        report_file=OUTPUT_REPORT,
    )
    checker.run()
