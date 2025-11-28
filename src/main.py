from src.utils import Utils
from src.data_preparation import DataPreparer
from src.process_discovery import ProcessDiscovery
from src.performance_analysis import PerformanceAnalysis
from src.conformance_checking import ConformanceChecker


def main():
    Utils().run()
    DataPreparer().run()
    ProcessDiscovery().run()
    PerformanceAnalysis.run()
    ConformanceChecker().run()


if __name__ == "__main__":
    main()
