import os
import platform as _platform

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

    def load_config_by_platform():
        system = _platform.system().lower()
        global DATA_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH, SAMPLED_DATA_PATH, SAMPLE_FRACTION, XES_OUTPUT_PATH

        if system == "linux":
            from src.config import DATA_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH, SAMPLED_DATA_PATH, SAMPLE_FRACTION, XES_OUTPUT_PATH
        else:
            from config import DATA_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH, SAMPLED_DATA_PATH, SAMPLE_FRACTION, XES_OUTPUT_PATH
