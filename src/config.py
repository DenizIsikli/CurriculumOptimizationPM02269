import os

# ======================================
# data_preparation.py Configurations
# ======================================
RESULTS_PATH = os.path.join("results")
DATA_PATH = os.path.join("results", "data")
RAW_DATA_PATH = os.path.join(DATA_PATH, "DTU_dataset.xlsx")
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "processed_log.csv")
SAMPLED_DATA_PATH = os.path.join(DATA_PATH, "sampled_log.csv")
XES_OUTPUT_PATH = os.path.join(DATA_PATH, "sampled_event_log.xes")

# ======================================
# model_training.py Configurations
# ======================================
MODEL_PATH = os.path.join("results", "models")
IM_MODEL_PATH = os.path.join(MODEL_PATH, "inductive_miner.pnml")
HM_MODEL_PATH = os.path.join(MODEL_PATH, "heuristics_miner.pnml")
PTREE_PATH = os.path.join(MODEL_PATH, "process_tree.png")

# Visualization Outputs
IM_IMAGE_PATH = os.path.join(MODEL_PATH, "inductive_miner.png")
HM_IMAGE_PATH = os.path.join(MODEL_PATH, "heuristics_miner.png")

# ======================================
# performance_analysis.py Configurations
# ======================================
PERFORMANCE_PATH = os.path.join("results", "performance_analysis")
PERFORMANCE_LOG_PATH = os.path.join(PERFORMANCE_PATH, "performance_log.txt")

# ======================================
# conformance_checking.py Configurations
# ======================================
# Paths for conformance checking
CONFORMANCE_PATH = os.path.join("results", "conformance_checking")
CONFORMANCE_LOG_PATH = os.path.join(CONFORMANCE_PATH, "conformance_log.txt")

# Petri net model to use (you can swap between miners later)
MODEL_FILE = os.path.join("results", "models", "inductive_miner.pnml")
EVENT_LOG_FILE = os.path.join("results", "data", "sampled_event_log.xes")

# ======================================
# Sampling Configuration
# ======================================
SAMPLE_FRACTION = 0.02  # 20% sample for quicker processing
