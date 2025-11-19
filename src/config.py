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

# Sampling Configuration
SAMPLE_FRACTION = 0.075  # 5% sample for quicker processing

# Recommend Curriculum
RECOMMENDED_CURRICULUM = {
    # Semester 1 - Polyteknisk Grundlag + Core Software
    "01001": {"semester": 1, "type": "mandatory", "prerequisites": [], "ects": 10, "block": "Polyteknisk grundlag"},
    "01002": {"semester": 1, "type": "mandatory", "prerequisites": [], "ects": 10, "block": "Polyteknisk grundlag"},
    "02100": {"semester": 1, "type": "mandatory", "prerequisites": [], "ects": 5, "block": "Retningsspecifikke"},
    "10060": {"semester": 1, "type": "mandatory", "prerequisites": [], "ects": 10, "block": "Polyteknisk grundlag"},
    
    # Semester 2 - Polyteknisk Grundlag continued + Software foundations
    "01017": {"semester": 2, "type": "mandatory", "prerequisites": [], "ects": 5, "block": "Retningsspecifikke"},
    "02104": {"semester": 2, "type": "mandatory", "prerequisites": ["02100"], "ects": 5, "block": "Retningsspecifikke"},
    "02105": {"semester": 2, "type": "mandatory", "prerequisites": ["02100"], "ects": 5, "block": "Retningsspecifikke"},
    "27020": {"semester": 2, "type": "mandatory", "prerequisites": [], "ects": 5, "block": "Polyteknisk grundlag"},
    # Statistics choice (one of):
    "02402": {"semester": 2, "type": "restricted_choice", "prerequisites": ["01001"], "ects": 5, "block": "Polyteknisk grundlag", "group": "STAT-CHOOSE-1"},
    "02403": {"semester": 2, "type": "restricted_choice", "prerequisites": ["01001"], "ects": 5, "block": "Polyteknisk grundlag", "group": "STAT-CHOOSE-1"},
    # Chemistry choice (one of):
    "26020": {"semester": 2, "type": "restricted_choice", "prerequisites": [], "ects": 5, "block": "Polyteknisk grundlag", "group": "CHEM-CHOOSE-1"},
    "26021": {"semester": 2, "type": "restricted_choice", "prerequisites": [], "ects": 5, "block": "Polyteknisk grundlag", "group": "CHEM-CHOOSE-1"},
    "26022": {"semester": 2, "type": "restricted_choice", "prerequisites": [], "ects": 5, "block": "Polyteknisk grundlag", "group": "CHEM-CHOOSE-1"},
    
    # Semester 3 - Core Software courses
    "02132": {"semester": 3, "type": "mandatory", "prerequisites": ["02100"], "ects": 10, "block": "Retningsspecifikke"},
    "02141": {"semester": 3, "type": "mandatory", "prerequisites": ["01017"], "ects": 10, "block": "Retningsspecifikke"},
    "02157": {"semester": 3, "type": "mandatory", "prerequisites": ["02105"], "ects": 5, "block": "Retningsspecifikke"},
    "02161": {"semester": 3, "type": "mandatory", "prerequisites": ["02100"], "ects": 5, "block": "Retningsspecifikke"},
    
    # Semester 4 - Restricted choice + electives start
    # STS choice (one of):
    "42620": {"semester": 4, "type": "restricted_choice", "prerequisites": [], "ects": 5, "block": "Polyteknisk grundlag", "group": "STS-CHOOSE-1"},
    "42622": {"semester": 4, "type": "restricted_choice", "prerequisites": [], "ects": 5, "block": "Polyteknisk grundlag", "group": "STS-CHOOSE-1"},
    # Retningsspecifikke choice (≥5 ECTS from):
    "02155": {"semester": 4, "type": "restricted_choice", "prerequisites": ["02132"], "ects": 5, "block": "Retningsspecifikke", "group": "RSPEC-CHOOSE-5ECTS"},
    "02156": {"semester": 4, "type": "restricted_choice", "prerequisites": ["02141"], "ects": 5, "block": "Retningsspecifikke", "group": "RSPEC-CHOOSE-5ECTS"},
    "02159": {"semester": 4, "type": "restricted_choice", "prerequisites": ["02132"], "ects": 5, "block": "Retningsspecifikke", "group": "RSPEC-CHOOSE-5ECTS"},
    # Typical electives semester 4 (10 ECTS)
    "02162": {"semester": 4, "type": "elective", "prerequisites": ["02161"], "ects": 10, "block": "Valgfrie"},
    "02180": {"semester": 4, "type": "elective", "prerequisites": ["02105"], "ects": 5, "block": "Valgfrie"},
    
    # Semester 5 - Project + electives
    "02122": {"semester": 5, "type": "mandatory", "prerequisites": [], "ects": 10, "block": "Projekter"},
    # Common electives semester 5
    "02225": {"semester": 5, "type": "elective", "prerequisites": ["02132"], "ects": 5, "block": "Valgfrie"},
    "02231": {"semester": 5, "type": "elective", "prerequisites": ["01017"], "ects": 5, "block": "Valgfrie"},
    "02247": {"semester": 5, "type": "elective", "prerequisites": ["02105"], "ects": 5, "block": "Valgfrie"},
    "02249": {"semester": 5, "type": "elective", "prerequisites": ["02105"], "ects": 7.5, "block": "Valgfrie"},
    "02258": {"semester": 5, "type": "elective", "prerequisites": ["02132"], "ects": 5, "block": "Valgfrie"},
    "02266": {"semester": 5, "type": "elective", "prerequisites": ["02161"], "ects": 5, "block": "Valgfrie"},
    
    # Semester 6 - Bachelor project + final electives
    "BACH-PROJ": {"semester": 6, "type": "mandatory", "prerequisites": [], "ects": 15, "block": "Projekter"},
    # Common electives semester 6
    "02262": {"semester": 6, "type": "elective", "prerequisites": [], "ects": 5, "block": "Valgfrie"},
    "02267": {"semester": 6, "type": "elective", "prerequisites": ["02162"], "ects": 5, "block": "Valgfrie"},
    "02269": {"semester": 6, "type": "elective", "prerequisites": [], "ects": 5, "block": "Valgfrie"},
    "02270": {"semester": 6, "type": "elective", "prerequisites": [], "ects": 5, "block": "Valgfrie"},
    "02271": {"semester": 6, "type": "elective", "prerequisites": ["02270"], "ects": 5, "block": "Valgfrie"},
    "02282": {"semester": 6, "type": "elective", "prerequisites": ["02105"], "ects": 7.5, "block": "Valgfrie"},
    "02285": {"semester": 6, "type": "elective", "prerequisites": ["02180"], "ects": 7.5, "block": "Valgfrie"},
    "02289": {"semester": 6, "type": "elective", "prerequisites": ["02105"], "ects": 5, "block": "Valgfrie"},
    "02504": {"semester": 6, "type": "elective", "prerequisites": [], "ects": 5, "block": "Valgfrie"},
    "02561": {"semester": 6, "type": "elective", "prerequisites": [], "ects": 5, "block": "Valgfrie"},
    "02562": {"semester": 6, "type": "elective", "prerequisites": [], "ects": 5, "block": "Valgfrie"},
    "02805": {"semester": 6, "type": "elective", "prerequisites": [], "ects": 10, "block": "Valgfrie"},
    "02807": {"semester": 6, "type": "elective", "prerequisites": [], "ects": 5, "block": "Valgfrie"},
    "02810": {"semester": 6, "type": "elective", "prerequisites": [], "ects": 5, "block": "Valgfrie"},
    "30510": {"semester": 6, "type": "elective", "prerequisites": [], "ects": 5, "block": "Valgfrie"},
    "42137": {"semester": 6, "type": "elective", "prerequisites": [], "ects": 5, "block": "Valgfie"},
}

# Curriculum requirements
CURRICULUM_REQUIREMENTS = {
    "total_ects": 180,
    "mandatory_ects": 105,  # Including project ECTS
    "polyteknisk_grundlag_ects": 60,
    "retningsspecifikke_ects": 60,
    "elective_ects": 45,
    "project_ects": 25,  # 10 (02122) + 15 (bachelor)
    "max_beng_electives": 10,
}

# Course groups for validation
COURSE_GROUPS = {
    "STAT-CHOOSE-1": ["02402", "02403"],
    "CHEM-CHOOSE-1": ["26020", "26021", "26022"],
    "STS-CHOOSE-1": ["42620", "42622"],
    "RSPEC-CHOOSE-5ECTS": ["02155", "02156", "02159"],
}
