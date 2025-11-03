import os
from datetime import datetime
from pm4py.conformance import conformance_diagnostics_alignments
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py import conformance_diagnostics_alignments as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator

from config import MODEL_FILE, EVENT_LOG_FILE, CONFORMANCE_PATH, CONFORMANCE_LOG_PATH

class ConformanceChecker:
    def __init__(
        self,
        model_path: str = MODEL_FILE,
        log_path: str = EVENT_LOG_FILE,
        output_dir: str = CONFORMANCE_PATH,
    ):
        self.model_path = model_path
        self.log_path = log_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Output log file
        self.report_path = CONFORMANCE_LOG_PATH
        with open(self.report_path, "w") as f:
            f.write(
                f"=== CONFORMANCE CHECK REPORT "
                f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n"
            )

        # Load artifacts
        self.log = xes_importer.apply(self.log_path)
        self.net, self.initial_marking, self.final_marking = pnml_importer.apply(self.model_path)

    def run(self):
        self._token_replay()
        self._alignments()
        self._quality_measures()

    def _log_section(self, title: str, content: str) -> None:
        with open(self.report_path, "a") as f:
            f.write(f"--- {title.upper()} ---\n")
            f.write(content.strip() + "\n\n")

    def _token_replay(self):
        results = token_replay.apply(self.log, self.net, self.initial_marking, self.final_marking)

        # Compute fitness as average token replay result
        fitness_values = [
            r["trace_fitness"] for r in results if "trace_fitness" in r
        ]
        avg_fitness = sum(fitness_values) / len(fitness_values) if fitness_values else 0

        content = (
            f"Token-based Replay Fitness: {avg_fitness:.4f}\n"
            f"Number of traces evaluated: {len(fitness_values)}"
        )

        self._log_section("Token-based Replay", content)

    def _alignments(self):
        from pm4py import conformance_diagnostics_alignments

        fitness_result = conformance_diagnostics_alignments(
            self.log, self.net, self.initial_marking, self.final_marking
        )

        content = (
            f"Alignment-based Fitness (diagnostic): {fitness_result['log_fitness']:.4f}\n"
            f"Traces evaluated: {len(self.log)}\n"
            f"Percentage fitting: {fitness_result.get('perc_fit_traces', 'N/A')}"
        )

        self._log_section("Alignments (Diagnostics)", content)

    def _quality_measures(self):
        precision = precision_evaluator.apply(
            self.log, self.net, self.initial_marking, self.final_marking
        )
        generalization = generalization_evaluator.apply(
            self.log, self.net, self.initial_marking, self.final_marking
        )
        simplicity = simplicity_evaluator.apply(self.net)

        content = (
            f"Precision: {precision:.4f}\n"
            f"Generalization: {generalization:.4f}\n"
            f"Simplicity: {simplicity:.4f}"
        )

        self._log_section("Quality Measures", content)


if __name__ == "__main__":
    checker = ConformanceChecker()
    checker.run()
