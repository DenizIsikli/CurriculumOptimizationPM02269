import os
from pm4py.algo.discovery.inductive.algorithm import apply as apply_inductive_miner
from pm4py.algo.discovery.heuristics.algorithm import apply_heu
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.objects.log.importer.xes import importer as xes_importer


from data_preparation import DataPreparer
from config import MODEL_PATH


class ProcessDiscovery:
    def __init__(self, event_log):
        self.log = event_log
        self.output_dir = MODEL_PATH
        os.makedirs(self.output_dir, exist_ok=True)

    def summarize_log(self):
        """Print basic event log statistics"""
        activity_key = "concept:name" if "concept:name" in self.log[0][0] else "course_code"
        parameters = {case_statistics.Parameters.ACTIVITY_KEY: activity_key}


        print("Event Log Summary")
        print(f"  Number of traces (students): {len(self.log)}")
        print(f"  Total events: {sum(len(t) for t in self.log)}")
        print(f"  Unique activities: {len(set(e[activity_key] for t in self.log for e in t))}")
        print(f"  Variants: {len(case_statistics.get_variant_statistics(self.log, parameters=parameters))}")
        print("-" * 60)

    def run_inductive_miner(self):
        """Apply Inductive Miner to discover a process model"""
        print("Running Inductive Miner")
        net, initial_marking, final_marking = apply_inductive_miner(self.log)
        print("Inductive Miner completed")
        self._export_and_visualize(net, initial_marking, final_marking, "inductive_miner")
        return net, initial_marking, final_marking

    def run_heuristics_miner(self):
        """Apply Heuristics Miner to discover a process model"""
        print("Running Heuristics Miner")
        net, initial_marking, final_marking = apply_heu(self.log)
        print("Heuristics Miner completed")
        self._export_and_visualize(net, initial_marking, final_marking, "heuristics_miner")
        return net, initial_marking, final_marking

    def _export_and_visualize(self, net, im, fm, name):
        """Export Petri net and visualize the model"""
        pnml_path = os.path.join(self.output_dir, f"{name}.pnml")
        image_path = os.path.join(self.output_dir, f"{name}.png")

        print(f"Exporting model to {pnml_path}")
        pnml_exporter.apply(net, im, pnml_path, final_marking=fm)

        print(f"Generating visualization for {name}")
        gviz = pn_visualizer.apply(net, im, fm)
        pn_visualizer.save(gviz, image_path)
        print(f"Saved visualization to {image_path}\n")

    def visualize_process_tree(self):
        """Generate a process tree visualization for interpretability"""
        print("Creating process tree visualization (for Inductive Miner model)")
        tree = inductive_miner.apply_tree(self.log)
        gviz_tree = pt_visualizer.apply(tree)
        output_path = os.path.join(self.output_dir, "process_tree.png")
        pt_visualizer.save(gviz_tree, output_path)
        print(f"Process tree saved to {output_path}")


if __name__ == "__main__":
    print("Starting Process Discovery Pipeline")

    event_log = xes_importer.apply(os.path.join("data", "sampled_event_log.xes"))

    discovery = ProcessDiscovery(event_log)
    discovery.summarize_log()

    # Run both miners for comparison
    discovery.run_inductive_miner()
    discovery.run_heuristics_miner()

    # Optional: visualize process tree (for report)
    discovery.visualize_process_tree()

    print("Process discovery completed successfully")
