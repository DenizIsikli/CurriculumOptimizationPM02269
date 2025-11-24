import os
from datetime import datetime

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments


REFERENCE_MODEL = "results/model/inductive_miner.pnml"     # curriculum PNML
GROUP_LOG_DIR = "results/performance_analysis/groups"      # folder with XES logs
OUTPUT_REPORT = "results/conformance/conformance_report.txt"

GROUP_LOGS = {
    "adherent_high_gpa":  "adherent_high_gpa.xes",
    "adherent_low_gpa":   "adherent_low_gpa.xes",
    "deviating_high_gpa": "deviating_high_gpa.xes",
    "deviating_low_gpa":  "deviating_low_gpa.xes",
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

        # reset file
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"=== CONFORMANCE CHECK REPORT ({datetime.now()}) ===\n\n")

        # load reference PNML model
        self.net, self.im, self.fm = pnml_importer.apply(self.model_path)

    def run(self):
        for group_name, filename in self.group_logs.items():
            log_path = os.path.join(self.log_dir, filename)

            if not os.path.exists(log_path):
                write(self.report_file, f"[{group_name}] XES not found → skipping\n")
                continue

            log = xes_importer.apply(log_path)
            write(self.report_file, f"--- {group_name.upper()} ---")
            write(self.report_file, f"Traces in log: {len(log)}")

            self.token_replay_fitness(log, group_name)
            self.alignment_fitness(log, group_name)
            write(self.report_file, "\n")

    def token_replay_fitness(self, log, group):
        results = token_replay.apply(log, self.net, self.im, self.fm)
        fitness = [r["trace_fitness"] for r in results]

        avg_fit = sum(fitness) / len(fitness) if fitness else 0

        write(self.report_file, f"Token Replay Fitness: {avg_fit:.3f}")

    def alignment_fitness(self, log, group):
        align_res = alignments.apply(log, self.net, self.im, self.fm)

        fitness_values = [a["fitness"] for a in align_res]
        avg_fit = sum(fitness_values) / len(fitness_values) if fitness_values else 0

        perc_fitting = 100 * sum(a.get("is_fit", False) for a in align_res) / len(align_res)

        write(self.report_file, f"Alignment Fitness: {avg_fit:.3f}")
        write(self.report_file, f"Percentage Fitting Traces: {perc_fitting:.1f}%")


if __name__ == "__main__":
    checker = ConformanceChecker(
        model_path=REFERENCE_MODEL,
        group_logs=GROUP_LOGS,
        log_dir=GROUP_LOG_DIR,
        report_file=OUTPUT_REPORT,
    )
    checker.run()
