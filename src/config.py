import os

# ======================================
# data_preparation.py Configurations
# ======================================
DATA_PATH = os.path.join("data")
RAW_DATA_PATH = os.path.join(DATA_PATH, "DTU_dataset.xlsx")
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "processed_log.csv")
SAMPLED_DATA_PATH = os.path.join(DATA_PATH, "sampled_log.csv")
XES_OUTPUT_PATH = os.path.join(DATA_PATH, "sampled_event_log.xes")

SAMPLE_FRACTION = 0.2

# ======================================
# model_training.py Configurations
# ======================================
MODEL_PATH = os.path.join("models")
IM_MODEL_PATH = os.path.join(MODEL_PATH, "inductive_miner.pnml")
HM_MODEL_PATH = os.path.join(MODEL_PATH, "heuristics_miner.pnml")
PTREE_PATH = os.path.join(MODEL_PATH, "process_tree.png")

# Visualization Outputs
IM_IMAGE_PATH = os.path.join(MODEL_PATH, "inductive_miner.png")
HM_IMAGE_PATH = os.path.join(MODEL_PATH, "heuristics_miner.png")


# Ensure directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
