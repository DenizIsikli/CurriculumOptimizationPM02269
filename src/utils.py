import os

from src.config import RESULTS_PATH, DATA_PATH, MODEL_PATH, PERFORMANCE_PATH


class Utils:
    def __init__(self):
        pass

    def run(self):
        self._ensure_dir_exists()

    def _ensure_dir_exists(self):
        if not os.path.exists(RESULTS_PATH): os.makedirs(RESULTS_PATH)
        if not os.path.exists(DATA_PATH): os.makedirs(DATA_PATH)
        if not os.path.exists(MODEL_PATH): os.makedirs(MODEL_PATH)
        if not os.path.exists(PERFORMANCE_PATH): os.makedirs(PERFORMANCE_PATH)
