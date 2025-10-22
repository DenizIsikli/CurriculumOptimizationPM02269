from src.data_preparation import prepare_data
from src.process_discovery import discover_process
from src.performance_analysis import analyze_performance
from src.conformance_checking import check_conformance

def main():
    event_log = prepare_data()
    model = discover_process(event_log)
    analyze_performance(event_log, model)
    check_conformance(event_log, model)

if __name__ == "__main__":
    main()
