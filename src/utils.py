import os

try:
    # Prefer package-relative imports when run with -m
    from .config import (
        RESULTS_PATH,
        DATA_PATH,
        PROCESS_DISCOVERY,
        PERFORMANCE_PATH,
        RAW_DATA_PATH,
        PROCESSED_DATA_PATH,
        SAMPLED_DATA_PATH,
        SAMPLE_FRACTION,
        XES_OUTPUT_PATH,
    )
except ImportError:
    # Fallback for direct execution without -m
    from config import (
        RESULTS_PATH,
        DATA_PATH,
        PROCESS_DISCOVERY,
        PERFORMANCE_PATH,
        RAW_DATA_PATH,
        PROCESSED_DATA_PATH,
        SAMPLED_DATA_PATH,
        SAMPLE_FRACTION,
        XES_OUTPUT_PATH,
    )


class Utils:
    def __init__(self):
        pass

    def run(self):
        self._ensure_dir_exists()

    def _ensure_dir_exists(self):
        if not os.path.exists(RESULTS_PATH): os.makedirs(RESULTS_PATH)
        if not os.path.exists(DATA_PATH): os.makedirs(DATA_PATH)
        if not os.path.exists(PROCESS_DISCOVERY): os.makedirs(PROCESS_DISCOVERY)
        if not os.path.exists(PERFORMANCE_PATH): os.makedirs(PERFORMANCE_PATH)

    def load_config_by_platform():
        """
        Re-import config names in whichever mode we are running:
        package (`python -m src.*`) or direct script execution.
        """
        global RESULTS_PATH, DATA_PATH, PROCESS_DISCOVERY, PERFORMANCE_PATH
        global RAW_DATA_PATH, PROCESSED_DATA_PATH, SAMPLED_DATA_PATH
        global SAMPLE_FRACTION, XES_OUTPUT_PATH

        try:
            from .config import (
                RESULTS_PATH as _R,
                DATA_PATH as _D,
                PROCESS_DISCOVERY as _M,
                PERFORMANCE_PATH as _P,
                RAW_DATA_PATH as _RAW,
                PROCESSED_DATA_PATH as _PROC,
                SAMPLED_DATA_PATH as _SAMP,
                SAMPLE_FRACTION as _SF,
                XES_OUTPUT_PATH as _XES,
            )
        except ImportError:
            from config import (
                RESULTS_PATH as _R,
                DATA_PATH as _D,
                PROCESS_DISCOVERY as _M,
                PERFORMANCE_PATH as _P,
                RAW_DATA_PATH as _RAW,
                PROCESSED_DATA_PATH as _PROC,
                SAMPLED_DATA_PATH as _SAMP,
                SAMPLE_FRACTION as _SF,
                XES_OUTPUT_PATH as _XES,
            )

        RESULTS_PATH = _R
        DATA_PATH = _D
        PROCESS_DISCOVERY = _M
        PERFORMANCE_PATH = _P
        RAW_DATA_PATH = _RAW
        PROCESSED_DATA_PATH = _PROC
        SAMPLED_DATA_PATH = _SAMP
        SAMPLE_FRACTION = _SF
        XES_OUTPUT_PATH = _XES
