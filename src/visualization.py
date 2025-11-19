import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.process_tree.importer import importer as ptml_importer

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.statistics.traces.generic.log import case_statistics

from src.config import MODEL_PATH, XES_OUTPUT_PATH, PROCESSED_DATA_PATH


# ===============================================================
# GLOBAL PLOT STYLE
# ===============================================================
plt.style.use("ggplot")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "font.family": "sans-serif",
    "figure.dpi": 140,
})

VIS_PATH = os.path.join("results", "visualization")
os.makedirs(VIS_PATH, exist_ok=True)


# ===============================================================
# PETRI NET VISUALIZATION WITHOUT GRAPHVIZ  (NETWORKX)
# ===============================================================

def visualize_petri_net_no_graphviz(miner_name):
    """Visualize Petri net using NetworkX instead of GraphViz."""
    pnml_path = os.path.join(MODEL_PATH, f"{miner_name}.pnml")

    if not os.path.exists(pnml_path):
        print(f"[WARN] Petri net not found: {pnml_path}")
        return

    net, im, fm = pnml_importer.apply(pnml_path)

    G = nx.DiGraph()

    # --- ADD PLACES ---
    for place in net.places:
        node_id = place.name or f"place_{id(place)}"
        G.add_node(node_id, type="place", label=node_id)

    # --- ADD TRANSITIONS ---
    for trans in net.transitions:
        label = trans.label if trans.label else f"tau_{trans.name}"
        G.add_node(label, type="transition", label=label)

    # --- ADD ARCS ---
    for arc in net.arcs:
        src = arc.source.label if hasattr(arc.source, "label") and arc.source.label else arc.source.name
        tgt = arc.target.label if hasattr(arc.target, "label") and arc.target.label else arc.target.name

        if src is None:  
            src = f"anon_source_{id(arc)}"
            G.add_node(src, type="transition", label=src)

        if tgt is None:
            tgt = f"anon_target_{id(arc)}"
            G.add_node(tgt, type="transition", label=tgt)

        if src not in G.nodes:
            G.add_node(src, type="transition", label=src)

        if tgt not in G.nodes:
            G.add_node(tgt, type="transition", label=tgt)

        G.add_edge(src, tgt)

    # --- DRAW ---
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.6, iterations=50)

    place_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "place"]
    trans_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "transition"]

    nx.draw_networkx_nodes(G, pos, nodelist=place_nodes, node_color="skyblue", node_size=900)
    nx.draw_networkx_nodes(G, pos, nodelist=trans_nodes, node_color="lightgreen", node_size=900)
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->")

    plt.title(f"Petri Net ({miner_name}) – NetworkX", fontsize=16)
    plt.axis("off")

    out_path = os.path.join(VIS_PATH, f"{miner_name}_networkx.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved Petri net (NetworkX) → {out_path}")



# ===============================================================
# SIMPLE PROCESS TREE VISUAL (NO GRAPHVIZ)
# ===============================================================

def visualize_process_tree():
    ptml_path = os.path.join(MODEL_PATH, "process_tree.ptml")

    if not os.path.exists(ptml_path):
        print("[WARN] No process tree found.")
        return

    tree = ptml_importer.apply(ptml_path)

    # Extract structure
    def traverse(node, depth=0):
        lines = []
        indent = "  " * depth
        label = node.operator.value if hasattr(node, "operator") else node.label
        lines.append(f"{indent}- {label}")
        if hasattr(node, "children"):
            for child in node.children:
                lines.extend(traverse(child, depth + 1))
        return lines

    lines = traverse(tree)

    out_path = os.path.join(VIS_PATH, "process_tree.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Saved process tree structure → {out_path}")


# ===============================================================
# TOP VARIANTS PLOT
# ===============================================================

def visualize_top_variants(top_n=10):
    """Plot the top N most frequent process variants."""
    log = xes_importer.apply(XES_OUTPUT_PATH)

    activity_key = "concept:name" if "concept:name" in log[0][0] else "course_code"
    params = {case_statistics.Parameters.ACTIVITY_KEY: activity_key}

    variants = case_statistics.get_variant_statistics(log, parameters=params)
    variants = sorted(variants, key=lambda x: x["count"], reverse=True)[:top_n]

    # FIX: convert tuple → string for Matplotlib
    labels = [" → ".join(v["variant"]) for v in variants]
    counts = [v["count"] for v in variants]

    plt.figure(figsize=(12, 6))
    plt.barh(labels, counts)
    plt.gca().invert_yaxis()
    plt.xlabel("Frequency")
    plt.title("Top 10 Most Common Process Variants")
    plt.tight_layout()

    out_path = os.path.join(VIS_PATH, "top_variants.png")
    plt.savefig(out_path)
    plt.close()

    print(f"[OK] Saved top variants → {out_path}")


# ===============================================================
# PERFORMANCE VISUALIZATIONS
# ===============================================================

def visualize_throughput_times():
    log = xes_importer.apply(XES_OUTPUT_PATH)

    durations = case_statistics.get_all_case_durations(
        log, parameters={case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"}
    )

    durations_days = [d / 86400 for d in durations]

    plt.figure()
    plt.hist(durations_days, bins=30, color="mediumseagreen")
    plt.title("Study Duration Histogram")
    plt.xlabel("Days")
    plt.ylabel("Number of Students")

    out_path = os.path.join(VIS_PATH, "throughput_times.png")
    plt.savefig(out_path)
    plt.close()

    print(f"[OK] Saved throughput histogram → {out_path}")


def visualize_pass_rates():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    stats = (
        df.groupby("course_code")["passed"]
        .mean()
        .sort_values(ascending=False)
        .head(20)
    )

    plt.figure()
    plt.bar(stats.index, stats.values, color="gold")
    plt.xticks(rotation=70)
    plt.title("Top 20 Courses by Pass Rate")
    plt.ylabel("Pass Rate")

    out_path = os.path.join(VIS_PATH, "pass_rates.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved pass rate chart → {out_path}")


def visualize_grade_distribution():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    grades = df["grade_num"].dropna()
    counts = grades.value_counts().sort_index()

    plt.figure()
    plt.bar(counts.index.astype(str), counts.values, color="cornflowerblue")
    plt.title("Grade Distribution")
    plt.xlabel("Grade")
    plt.ylabel("Count")

    out_path = os.path.join(VIS_PATH, "grade_distribution.png")
    plt.savefig(out_path)
    plt.close()

    print(f"[OK] Saved grade distribution → {out_path}")


# ===============================================================
# RUN EVERYTHING
# ===============================================================

def generate_all_visuals():
    print("\n=== PROCESS MODEL VISUALS (NO GRAPHVIZ) ===")
    visualize_petri_net_no_graphviz("alpha_miner")
    visualize_petri_net_no_graphviz("inductive_miner")
    visualize_petri_net_no_graphviz("heuristics_miner")
    

    print("\n=== PERFORMANCE VISUALS ===")
    visualize_top_variants()
    visualize_throughput_times()
    visualize_pass_rates()
    visualize_grade_distribution()


if __name__ == "__main__":
    generate_all_visuals()
