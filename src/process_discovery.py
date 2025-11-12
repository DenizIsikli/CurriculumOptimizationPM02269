import os
from pm4py import discover_petri_net_alpha, discover_petri_net_inductive, discover_process_tree_inductive
from pm4py.algo.discovery.heuristics.algorithm import apply as discover_heuristics_net
from pm4py.objects.log.util import sampling
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

from config import MODEL_PATH, SAMPLE_FRACTION, XES_OUTPUT_PATH


class ProcessDiscovery:
    def __init__(
            self,
            event_log=None,
            output_dir=MODEL_PATH,
            sample_fraction=SAMPLE_FRACTION
    ):
        self.output_dir = output_dir
        self.sample_fraction = sample_fraction
        os.makedirs(self.output_dir, exist_ok=True)
        self.event_log = event_log or xes_importer.apply(XES_OUTPUT_PATH)

    def run(self):
        self._fraction_log()
        self._summarize_log()
        self._run_alpha_miner()
        self._run_inductive_miner()
        self._run_heuristics_miner()
        self._save_process_tree()

    def _fraction_log(self):
        total_traces = len(self.event_log)
        n = int(total_traces * self.sample_fraction)
        print(f"Total unique students before sampling: {total_traces}")
        print(f"Sampling {n} traces (~{self.sample_fraction*100:.1f}%) of total")
        self.event_log = sampling.sample_log(self.event_log, n)

    def _summarize_log(self) -> None:
        activity_key = "concept:name" if "concept:name" in self.event_log[0][0] else "course_code"
        parameters = {case_statistics.Parameters.ACTIVITY_KEY: activity_key}

        print("Event Log Summary")
        print(f"  Number of traces (students): {len(self.event_log)}")
        print(f"  Total events (Total course attempts): {sum(len(t) for t in self.event_log)}")
        print(f"  Unique activities (Different courses taken): {len(set(e[activity_key] for t in self.event_log for e in t))}")
        print(f"  Variants (Unique paths students took through the process): {len(case_statistics.get_variant_statistics(self.event_log, parameters=parameters))}")
        print("-" * 60)

    def _run_alpha_miner(self):
        net, im, fm = discover_petri_net_alpha(self.event_log)
        self._save_model(net, im, fm, "alpha_miner")

    def _run_inductive_miner(self):
        net, im, fm = discover_petri_net_inductive(self.event_log)
        self._save_model(net, im, fm, "inductive_miner")

    def _run_heuristics_miner(self):
        net, im, fm = discover_heuristics_net(self.event_log)
        self._save_model(net, im, fm, "heuristics_miner")

    def _save_model(self, net, im, fm, name):
        pnml_path = os.path.join(self.output_dir, f"{name}.pnml")
        pnml_exporter.apply(net, im, pnml_path, final_marking=fm)
        print(f"Saved {name} model to {pnml_path}")

    def _save_process_tree(self):
        tree = discover_process_tree_inductive(self.event_log)
        path = os.path.join(self.output_dir, "process_tree.ptml")
        with open(path, "w") as f:
            f.write(str(tree))

        print(f"Process tree saved to {path}")


if __name__ == "__main__":
    run = ProcessDiscovery()
    run.run()
