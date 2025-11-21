import os
import shutil

from pm4py import (
    discover_petri_net_alpha,
    discover_petri_net_inductive,
    discover_process_tree_inductive,
)
from pm4py.algo.discovery.heuristics.algorithm import apply as discover_heuristics_net
from pm4py.objects.log.util import sampling
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.visualization.petri_net import visualizer as pn_visualizer

try:
    # Prefer package-relative imports when run as a module
    from .utils import Utils as util
    from .config import MODEL_PATH, SAMPLE_FRACTION, XES_OUTPUT_PATH
except ImportError:
    # Fallback for direct execution without -m
    from utils import Utils as util
    from config import MODEL_PATH, SAMPLE_FRACTION, XES_OUTPUT_PATH

def _ensure_graphviz_on_path() -> None:
    """
    Make sure Graphviz binaries (dot) are reachable.
    - Honor GRAPHVIZ_BIN env var if set (any OS).
    - Keep existing Windows portable default as a fallback.
    - Fail fast with a clear error if dot is still missing.
    """
    custom_bin = os.environ.get("GRAPHVIZ_BIN")
    if custom_bin:
        os.environ["PATH"] = custom_bin + os.pathsep + os.environ["PATH"]
    elif os.name == "nt":
        portable_bin = (
            r"C:\Users\deniz\Desktop\Code\CurriculumOptimizationPM02269"
            r"\graphviz_portable\release\bin"
        )
        os.environ["PATH"] = portable_bin + os.pathsep + os.environ["PATH"]

    if shutil.which("dot") is None:
        raise RuntimeError(
            "Graphviz executable 'dot' not found. "
            "Install graphviz (e.g., apt install graphviz / brew install graphviz) "
            "or set GRAPHVIZ_BIN to its bin directory."
        )

_ensure_graphviz_on_path()


class ProcessDiscovery:
    """
    Runs process discovery on an event log.
    Adds automatic per-group sampling for readability.
    """

    def __init__(
        self,
        event_log=None,
        output_dir=MODEL_PATH,
        sample_fraction=SAMPLE_FRACTION,
        max_traces=15,      # NEW → cap per-group for readability
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.sample_fraction = sample_fraction
        self.max_traces = max_traces

        self.event_log = event_log or xes_importer.apply(XES_OUTPUT_PATH)

    # --------------------------------------------------------------

    def run(self) -> None:
        util.load_config_by_platform()
        self._summarize_log()
       #self._run_alpha_miner()
        self._run_inductive_miner()
        self._run_heuristics_miner()
        self._save_process_tree()

    def _summarize_log(self) -> None:
        activity_key = "concept:name"
        parameters = {case_statistics.Parameters.ACTIVITY_KEY: activity_key}

        print("Summary:")
        print("  Traces:", len(self.event_log))
        print("  Events:", sum(len(t) for t in self.event_log))
        print(
            "  Activities:",
            len({e[activity_key] for t in self.event_log for e in t}),
        )
        print(
            "  Variants:",
            len(case_statistics.get_variant_statistics(self.event_log)),
        )

    # --------------------------------------------------------------

    def _run_alpha_miner(self):
        net, im, fm = discover_petri_net_alpha(self.event_log)
        self._save_model(net, im, fm, "alpha_miner")

    def _run_inductive_miner(self):
        net, im, fm = discover_petri_net_inductive(self.event_log)
        self._save_model(net, im, fm, "inductive_miner")

    def _run_heuristics_miner(self):
        net, im, fm = discover_heuristics_net(self.event_log)
        self._save_model(net, im, fm, "heuristics_miner")

    # --------------------------------------------------------------

    def _save_model(self, net, im, fm, name):
        pnml = os.path.join(self.output_dir, f"{name}.pnml")
        pnml_exporter.apply(net, im, pnml, final_marking=fm)

        png = os.path.join(self.output_dir, f"{name}.png")
        gviz = pn_visualizer.apply(net, im, fm)
        pn_visualizer.save(gviz, png)

        print(f"Saved: {png}")

    # --------------------------------------------------------------

    def _save_process_tree(self):
        tree = discover_process_tree_inductive(self.event_log)
        ptml = os.path.join(self.output_dir, "process_tree.ptml")

        with open(ptml, "w") as f:
            f.write(str(tree))

        print(f"Saved: {ptml}")


if __name__ == "__main__":
    ProcessDiscovery().run()
