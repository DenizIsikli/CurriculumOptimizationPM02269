from src.utils import Utils
from src.data_preparation import DataPreparer
from src.process_discovery import ProcessDiscovery
from src.performance_analysis import PerformanceAnalyzer
from src.conformance_checking import ConformanceChecker
from src.visualization import Visualization


def main():
    Utils().run()
    DataPreparer().run()
    ProcessDiscovery().run()
    PerformanceAnalyzer().run()
    ConformanceChecker().run()
    Visualization().run()


if __name__ == "__main__":
    main()
