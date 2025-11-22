# src/visualization.py

import os
import pandas as pd
import pm4py
import matplotlib.pyplot as plt
import plotly.express as px

from src.config import RESULTS_PATH, MODEL_PATH, DATA_PATH


class Visualization:

    def __init__(self):
        self.results_path = RESULTS_PATH
        self.model_path = MODEL_PATH
        self.data_path = DATA_PATH

    def run(self):
        print("Running Visualization...")

        # Load event log
        log_path = os.path.join(self.data_path, "processed_log.xes")
        if not os.path.exists(log_path):
            print("❌ processed_log.xes not found. Visualization skipped.")
            return

        event_log = pm4py.read_xes(log_path)

        # Load discovered Petri net
        net_path = os.path.join(self.model_path, "petri_net.pnml")
        if os.path.exists(net_path):
            net, im, fm = pm4py.read_pnml(net_path)
            self._visualize_petri_net(net, im, fm)
        else:
            print("⚠️ No Petri net found, skipping model visualization.")

        # Data visualizations
        self._visualize_variants(event_log)
        self._visualize_activity_stats(event_log)
        self._visualize_performance_dfg(event_log)

        print("Visualization complete.")

    # --------------------------------------------------------------------------
    #   PROCESS MODEL VISUALIZATION
    # --------------------------------------------------------------------------

    def _visualize_petri_net(self, net, im, fm):
        print(" - Visualizing discovered Petri net...")
        gviz = pm4py.visualization.petri_net.visualizer.apply(net, im, fm)

        out_path = os.path.join(self.results_path, "petri_net.png")
        pm4py.visualization.petri_net.visualizer.save(gviz, out_path)

    # --------------------------------------------------------------------------
    #   VARIANT VISUALIZATION
    # --------------------------------------------------------------------------

    def _visualize_variants(self, event_log):
        print(" - Visualizing trace variants...")

        variants = pm4py.get_variants_as_tuples(event_log)
        data = [{"variant": str(k), "count": len(v)} for k, v in variants.items()]

        df = pd.DataFrame(data)

        if df.empty:
            print("⚠️ No variants available.")
            return

        fig = px.bar(df, x="variant", y="count", title="Trace Variant Frequency")

        fig.write_html(os.path.join(self.results_path, "variants.html"))

    # --------------------------------------------------------------------------
    #   ACTIVITY FREQUENCIES / STATISTICS
    # --------------------------------------------------------------------------

    def _visualize_activity_stats(self, event_log):
        print(" - Visualizing activity frequency...")

        activities = pm4py.get_event_attribute_values(event_log, "concept:name")
        df = pd.DataFrame({
            "activity": list(activities.keys()),
            "count": list(activities.values())
        })

        fig = px.bar(
            df,
            x="activity",
            y="count",
            title="Activity Frequency",
        )

        fig.write_html(os.path.join(self.results_path, "activity_frequency.html"))

    # --------------------------------------------------------------------------
    #   PERFORMANCE DFG
    # --------------------------------------------------------------------------

    def _visualize_performance_dfg(self, event_log):
        print(" - Visualizing performance DFG...")

        # Performance DFG gives bottlenecks (avg transition time)
        dfg, start_activities, end_activities = pm4py.discover_performance_dfg(event_log)

        fig = px.bar(
            x=[f"{k[0]} → {k[1]}" for k in dfg.keys()],
            y=[v["performance"] for v in dfg.values()],
            title="Performance DFG (Average Transition Time)"
        )

        fig.write_html(os.path.join(self.results_path, "performance_dfg.html"))
