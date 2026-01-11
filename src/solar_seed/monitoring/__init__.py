"""
Solar Seed Monitoring Module
============================

Database and monitoring components for the early warning system.

Usage:
    from solar_seed.monitoring import MonitoringDB

    db = MonitoringDB()
    db.insert_goes_xray(timestamp, flux, flare_class, magnitude)
    db.insert_coupling(timestamp, pair, delta_mi, residual, status, trend)
"""

from .db import MonitoringDB

__all__ = ['MonitoringDB']
