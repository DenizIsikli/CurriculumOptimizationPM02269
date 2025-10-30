import os
from pm4py.algo.discovery.heuristics.algorithm import apply_heu
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.objects.log.util import sampling
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive.algorithm import apply as apply_inductive_miner, apply_tree


from data_preparation import DataPreparer
from config import MODEL_PATH, SAMPLE_FRACTION


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

        self.event_log = event_log or xes_importer.apply(os.path.join("data", "sampled_event_log.xes"))

    def run(self):
        """Run both Inductive Miner and Heuristics Miner"""
        self._fraction_log()
        self._summarize_log()
        self._run_inductive_miner()
        self._run_heuristics_miner()
        self._visualize_process_tree()

    def _fraction_log(self):
        """Sample a fraction of traces to reduce complexity"""
        n = int(len(self.event_log) * self.sample_fraction)
        self.event_log = sampling.sample_log(self.event_log, n)

    def _summarize_log(self) -> None:
        """Print basic event log statistics"""
        activity_key = "concept:name" if "concept:name" in self.event_log[0][0] else "course_code"
        parameters = {case_statistics.Parameters.ACTIVITY_KEY: activity_key}

        print("Event Log Summary")
        print(f"  Number of traces (students): {len(self.event_log)}")
        print(f"  Total events: {sum(len(t) for t in self.event_log)}")
        print(f"  Unique activities: {len(set(e[activity_key] for t in self.event_log for e in t))}")
        print(f"  Variants: {len(case_statistics.get_variant_statistics(self.event_log, parameters=parameters))}")
        print("-" * 60)

    def _run_inductive_miner(self):
        """Apply Inductive Miner"""

        print("Running Inductive Miner")
        net, im, fm = apply_inductive_miner(self.event_log)
        self._export_and_visualize(net, im, fm, "inductive_miner")

    def _run_heuristics_miner(self):
        """Apply Heuristics Miner"""
        print("Running Heuristics Miner")
        net, im, fm = apply_heu(self.event_log)
        self._export_and_visualize(net, im, fm, "heuristics_miner")

    def _export_and_visualize(self, net, im, fm, name):
        """Export Petri net and visualize"""
        pnml_path = os.path.join(self.output_dir, f"{name}.pnml")
        img_path = os.path.join(self.output_dir, f"{name}.png")

        pnml_exporter.apply(net, im, pnml_path, final_marking=fm)

        gviz = pn_visualizer.apply(net, im, fm)
        pn_visualizer.save(gviz, img_path)
        print(f"Saved visualization to {img_path}\n")

    def _visualize_process_tree(self):
        """Generate process tree for interpretability"""
        tree = apply_tree(self.event_log)
        gviz_tree = pt_visualizer.apply(tree)
        output_path = os.path.join(self.output_dir, "process_tree.png")
        pt_visualizer.save(gviz_tree, output_path)
        print(f"Process tree saved to {output_path}")

if __name__ == "__main__":
    print("Starting Process Discovery Pipeline")
    run = ProcessDiscovery()
    run.run()
    print("Process discovery completed successfully")
