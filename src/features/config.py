"""Canonical SepsiGuard feature configuration.

Feature ordering is FIXED and versioned. The model input vector per
hour is exactly 69 channels, reproducing the historical specification:

    [ 0: 34)   clinical values (imputed, train-normalised)
    [34: 68)   missingness indicators (1 = value absent in raw record)
    [68]       ICULOS (hours since ICU admission, train-normalised)

A separate 34-channel *delta* tensor (hours since last observation,
capped and scaled) feeds the informative-missingness gating branch of
the proposed architecture (see src/models/missingness_gate.py).
"""
CLINICAL_FEATURES = [
    "hr", "o2sat", "temp", "sbp", "map", "dbp", "resp", "etco2",
    "baseexcess", "hco3", "fio2", "ph", "paco2", "sao2", "ast", "bun",
    "alkalinephos", "calcium", "chloride", "creatinine",
    "bilirubin_direct", "glucose", "lactate", "magnesium",
    "phosphate", "potassium", "bilirubin_total", "troponini", "hct", "hgb",
    "ptt", "wbc", "fibrinogen", "platelets",
]
MISSINGNESS_FEATURES = [f"{c}_missing" for c in CLINICAL_FEATURES]
TEMPORAL_FEATURES = ["iculos"]

# Raw PhysioNet column -> lowercase feature name (34 clinical channels)
RAW_COLUMNS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine",
    "Bilirubin_direct", "Glucose", "Lactate", "Magnesium",
    "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct",
    "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
]
RAW_TO_FEATURE = {raw: raw.lower() for raw in RAW_COLUMNS}
RAW_TO_FEATURE["pH"] = "ph"
assert len(RAW_TO_FEATURE) == 34 and set(RAW_TO_FEATURE.values()) == set(CLINICAL_FEATURES)

FEATURE_ORDER = CLINICAL_FEATURES + MISSINGNESS_FEATURES + TEMPORAL_FEATURES
assert len(FEATURE_ORDER) == 69

LOOKBACK = 12          # hours of history per prediction window
HORIZONS = {"y6": 6, "y12": 12}
DELTA_CAP_HOURS = 48.0
SEED = 42

# Patient-level split fractions (70/15/15), stratified by patient sepsis
# status over window-eligible patients.
SPLIT_FRACS = {"train": 0.70, "validation": 0.15, "test": 0.15}