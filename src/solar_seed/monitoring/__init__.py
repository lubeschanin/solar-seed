"""
Solar Seed Monitoring Module
============================

Database, monitoring, and anomaly detection components for the early warning system.

Usage:
    from solar_seed.monitoring import MonitoringDB, CouplingMonitor
    from solar_seed.monitoring import detect_coupling_break, classify_anomaly_status
    from solar_seed.monitoring import AnomalyStatus, BreakType

    db = MonitoringDB()
    monitor = CouplingMonitor()

    # Detect coupling breaks
    break_result = detect_coupling_break('193-211', delta_mi, monitor)
    status = classify_anomaly_status(break_result, robustness_check=rob)
"""

from .db import MonitoringDB
from .coupling import CouplingMonitor
from .constants import (
    MIN_MI_THRESHOLD,
    MIN_ROI_STD,
    AnomalyLevel,
    Phase,
    get_anomaly_level,
    classify_phase,
)
from .validation import validate_roi_variance, validate_mi_measurement
from .detection import (
    AnomalyStatus,
    BreakType,
    detect_artifact,
    detect_coupling_break,
    compute_registration_shift,
    compute_robustness_check,
    classify_break_type,
    classify_anomaly_status,
)
from .formatting import StatusFormatter

__all__ = [
    # Database
    'MonitoringDB',
    # Coupling monitor
    'CouplingMonitor',
    # Constants & Classification
    'MIN_MI_THRESHOLD',
    'MIN_ROI_STD',
    'AnomalyLevel',      # Statistical: NORMAL/ELEVATED/STRONG/EXTREME
    'Phase',             # Interpretive: BASELINE/PRE-FLARE/FLARE/RECOVERY/POST-FLARE REORG
    'get_anomaly_level',
    'classify_phase',
    # Validation
    'validate_roi_variance',
    'validate_mi_measurement',
    # Detection
    'AnomalyStatus',
    'BreakType',
    'detect_artifact',
    'detect_coupling_break',
    'compute_registration_shift',
    'compute_robustness_check',
    'classify_break_type',
    'classify_anomaly_status',
    # Formatting
    'StatusFormatter',
]
