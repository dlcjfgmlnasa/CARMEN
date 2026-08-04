"""Signal-type mapping tables (single modality-embedding scheme).

CARMEN uses one embedding per modality (signal_type). There are 9 signal types:

  ECG(0), ABP(1), PPG(2), CVP(3), CO2(4), AWP(5),
  ICP(6), RESP_Impedance(7), RESP_Flow(8)

RESP is split into chest-impedance respiration (RESP_Impedance) and the ventilator
flow waveform (RESP_Flow). All ECG leads map to a single ECG type, and all ABP
variants (Radial/Femoral/ART/FEM) map to a single ABP type.
"""

from __future__ import annotations


# signal_type number -> human-readable name
SIGNAL_TYPE_NAMES: dict[int, str] = {
    0: "ECG",
    1: "ABP",
    2: "PPG",
    3: "CVP",
    4: "CO2",
    5: "AWP",
    6: "ICP",
    7: "RESP_Impedance",
    8: "RESP_Flow",
}


# lowercase task-facing key -> signal_type number
SIGNAL_KEY_TO_TYPE: dict[str, int] = {
    "ecg": 0,
    "abp": 1,
    "ppg": 2,
    "cvp": 3,
    "co2": 4,
    "awp": 5,
    "icp": 6,
    "resp_impedance": 7,
    "resp_flow": 8,
}

# signal_type number -> lowercase key (reverse map)
SIGNAL_TYPE_TO_KEY: dict[int, str] = {v: k for k, v in SIGNAL_KEY_TO_TYPE.items()}


# ── Mechanism Group ────────────────────────────────────────────
# Cardiovascular (0): ECG, ABP, PPG, CVP, ICP — cardiac-cycle synchronized
# Respiratory (1): CO2, AWP, RESP_Impedance, RESP_Flow — ventilation synchronized
MECHANISM_GROUP: dict[int, int] = {
    0: 0,  # ECG            -> Cardiovascular
    1: 0,  # ABP            -> Cardiovascular
    2: 0,  # PPG            -> Cardiovascular
    3: 0,  # CVP            -> Cardiovascular
    4: 1,  # CO2            -> Respiratory
    5: 1,  # AWP            -> Respiratory
    6: 0,  # ICP            -> Cardiovascular
    7: 1,  # RESP_Impedance -> Respiratory
    8: 1,  # RESP_Flow      -> Respiratory
}

MECHANISM_GROUP_NAMES: dict[int, str] = {
    0: "Cardiovascular",
    1: "Respiratory",
}


# ── Cross-Pred Allowed Pairs ──────────────────────────────────
# Signal-type pairs the model was trained to reconstruct across modalities
# (physiologically causal waveform transfer). Useful as a reference for which
# cross-modal generation targets are reliable.
#   (0, 1) ECG <-> ABP — R-peak triggers contraction; shape is cardiac-dominated
#   (0, 2) ECG <-> PPG — cardiac cycle, peripheral pulse wave
#   (1, 2) ABP <-> PPG — arterial pulse wave (nearly isomorphic)
#   (5, 8) AWP <-> RESP_Flow — airway pressure <-> flow (P-Q equation of motion)
CROSS_PRED_ALLOWED_PAIRS: set[tuple[int, int]] = {
    (0, 1),
    (0, 2),
    (1, 2),
    (5, 8),
}


# channel name -> signal_type
# All ECG lead labels converge to 0; ABP variants (Radial/Femoral/ART/FEM) to 1.
# RESP/Impedance -> 7 (RESP_Impedance), FLOW/FLOW_WAV -> 8 (RESP_Flow).
CHANNEL_NAME_TO_SIGNAL_TYPE: dict[str, int] = {
    # ECG (0) — all lead labels converge to a single one
    "ECG Lead I": 0, "ECG I": 0, "I": 0,
    "ECG Lead II": 0, "ECG II": 0, "II": 0,
    "ECG Lead III": 0, "ECG III": 0, "III": 0,
    "ECG aVR": 0, "aVR": 0,
    "ECG aVL": 0, "aVL": 0,
    "ECG aVF": 0, "aVF": 0,
    "ECG V1": 0, "V1": 0,
    "ECG V2": 0, "V2": 0,
    "ECG V3": 0, "V3": 0,
    "ECG V4": 0, "V4": 0,
    "ECG V5": 0, "V5": 0, "ECG Lead V5": 0,
    "ECG V6": 0, "V6": 0,
    # ABP (1) — Radial/Femoral merged
    "ABP Radial": 1, "ART": 1,
    "ABP Femoral": 1, "FEM": 1,
    # PPG (2)
    "PPG": 2, "PLETH": 2, "PPG Finger": 2,
    # CVP (3)
    "CVP": 3,
    # CO2 (4)
    "CO2": 4,
    # AWP (5)
    "AWP": 5,
    # ICP (6)
    "ICP": 6,
    # RESP_Impedance (7)
    "RESP": 7, "Impedance": 7,
    # RESP_Flow (8)
    "FLOW": 8, "FLOW_WAV": 8,
}
