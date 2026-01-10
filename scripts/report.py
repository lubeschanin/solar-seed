#!/usr/bin/env python3
"""
Solar Early Warning - Reporting System
=======================================

Generates comprehensive reports from the monitoring database.

Usage:
    uv run python scripts/report.py                    # Full report (stdout)
    uv run python scripts/report.py --daily            # Daily summary
    uv run python scripts/report.py --weekly           # Weekly summary
    uv run python scripts/report.py --stats            # Precursor statistics
    uv run python scripts/report.py --export md        # Export as Markdown
    uv run python scripts/report.py --export html      # Export as HTML
    uv run python scripts/report.py --output report.md # Save to file
"""

import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional
import json

sys.path.insert(0, str(Path(__file__).parent))
from monitoring_db import MonitoringDB


class ReportGenerator:
    """Generate reports from monitoring database."""

    def __init__(self, db: MonitoringDB = None):
        self.db = db or MonitoringDB()
        self.now = datetime.now(timezone.utc)

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute query and return results as dicts."""
        cursor = self.db.conn.cursor()
        cursor.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def _query_one(self, sql: str, params: tuple = ()) -> dict:
        """Execute query and return single result."""
        cursor = self.db.conn.cursor()
        cursor.execute(sql, params)
        row = cursor.fetchone()
        return dict(row) if row else {}

    # =========================================================================
    # DATA COLLECTION
    # =========================================================================

    def get_summary_stats(self, days: int = 7) -> dict:
        """Get summary statistics for the period."""
        stats = {}

        # Total counts
        stats['period_days'] = days
        stats['period_start'] = (self.now - timedelta(days=days)).strftime('%Y-%m-%d')
        stats['period_end'] = self.now.strftime('%Y-%m-%d')

        # GOES data
        goes = self._query_one("""
            SELECT
                COUNT(*) as count,
                MIN(flux) as min_flux,
                MAX(flux) as max_flux,
                AVG(flux) as avg_flux
            FROM goes_xray
            WHERE timestamp >= datetime('now', ?)
        """, (f'-{days} days',))
        stats['goes'] = goes

        # Flare counts by class
        flares = self._query("""
            SELECT flare_class, COUNT(*) as count
            FROM goes_xray
            WHERE timestamp >= datetime('now', ?)
            AND flare_class IS NOT NULL
            GROUP BY flare_class
            ORDER BY flare_class
        """, (f'-{days} days',))
        stats['flare_counts'] = {f['flare_class']: f['count'] for f in flares}

        # Coupling measurements
        coupling = self._query_one("""
            SELECT
                COUNT(*) as count,
                COUNT(DISTINCT pair) as pairs,
                AVG(delta_mi) as avg_delta_mi
            FROM coupling_measurements
            WHERE timestamp >= datetime('now', ?)
        """, (f'-{days} days',))
        stats['coupling'] = coupling

        # Coupling by pair
        by_pair = self._query("""
            SELECT
                pair,
                COUNT(*) as count,
                AVG(delta_mi) as avg_mi,
                MIN(delta_mi) as min_mi,
                MAX(delta_mi) as max_mi,
                SUM(CASE WHEN status = 'ALERT' THEN 1 ELSE 0 END) as alerts,
                SUM(CASE WHEN status = 'WARNING' THEN 1 ELSE 0 END) as warnings
            FROM coupling_measurements
            WHERE timestamp >= datetime('now', ?)
            GROUP BY pair
        """, (f'-{days} days',))
        stats['coupling_by_pair'] = by_pair

        # Alert summary
        alerts = self._query_one("""
            SELECT
                SUM(CASE WHEN status = 'ALERT' THEN 1 ELSE 0 END) as total_alerts,
                SUM(CASE WHEN status = 'WARNING' THEN 1 ELSE 0 END) as total_warnings,
                SUM(CASE WHEN status = 'ELEVATED' THEN 1 ELSE 0 END) as total_elevated,
                SUM(CASE WHEN status = 'NORMAL' THEN 1 ELSE 0 END) as total_normal
            FROM coupling_measurements
            WHERE timestamp >= datetime('now', ?)
        """, (f'-{days} days',))
        stats['alerts'] = alerts

        # Solar wind
        wind = self._query_one("""
            SELECT
                COUNT(*) as count,
                AVG(speed) as avg_speed,
                MAX(speed) as max_speed,
                AVG(bz) as avg_bz,
                MIN(bz) as min_bz
            FROM solar_wind
            WHERE timestamp >= datetime('now', ?)
        """, (f'-{days} days',))
        stats['solar_wind'] = wind

        return stats

    def _parse_timestamp(self, ts: str) -> datetime:
        """Parse timestamp string to timezone-aware datetime."""
        if not ts:
            return None
        ts = ts.replace('Z', '+00:00')
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except:
            return None

    def get_precursor_statistics(self) -> dict:
        """
        Calculate precursor detection statistics (Precision/Recall).

        Definitions:
        - True Positive (TP): Break detected AND flare followed within window
        - False Positive (FP): Break detected BUT no flare followed
        - False Negative (FN): No break detected BUT flare occurred
        - True Negative (TN): No break AND no flare (normal operation)
        """
        stats = {}

        # Get all coupling breaks (status = ALERT or WARNING with significant deviation)
        breaks = self._query("""
            SELECT
                timestamp,
                pair,
                delta_mi,
                status,
                deviation_pct
            FROM coupling_measurements
            WHERE status IN ('ALERT', 'WARNING')
            ORDER BY timestamp
        """)
        stats['total_breaks'] = len(breaks)

        # Get all significant GOES events (C-class and above)
        flares = self._query("""
            SELECT
                timestamp,
                flux,
                flare_class,
                magnitude
            FROM goes_xray
            WHERE flux >= 1e-6
            ORDER BY timestamp
        """)
        stats['total_flares'] = len(flares)

        # Match breaks to flares (window: 0.5 - 6 hours after break)
        WINDOW_MIN_HOURS = 0.5
        WINDOW_MAX_HOURS = 6.0

        tp = 0  # Break followed by flare
        fp = 0  # Break not followed by flare
        matched_flares = set()

        for brk in breaks:
            break_time = self._parse_timestamp(brk['timestamp'])
            if not break_time:
                continue

            found_flare = False
            for flare in flares:
                flare_time = self._parse_timestamp(flare['timestamp'])
                if not flare_time:
                    continue

                delta_hours = (flare_time - break_time).total_seconds() / 3600

                if WINDOW_MIN_HOURS <= delta_hours <= WINDOW_MAX_HOURS:
                    found_flare = True
                    matched_flares.add(flare['timestamp'])
                    break

            if found_flare:
                tp += 1
            else:
                fp += 1

        # False negatives: flares without preceding break
        fn = len(flares) - len(matched_flares)

        # Calculate metrics
        stats['true_positives'] = tp
        stats['false_positives'] = fp
        stats['false_negatives'] = fn

        # Precision = TP / (TP + FP)
        if tp + fp > 0:
            stats['precision'] = tp / (tp + fp)
        else:
            stats['precision'] = None

        # Recall = TP / (TP + FN)
        if tp + fn > 0:
            stats['recall'] = tp / (tp + fn)
        else:
            stats['recall'] = None

        # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        if stats['precision'] and stats['recall'] and (stats['precision'] + stats['recall']) > 0:
            stats['f1_score'] = 2 * (stats['precision'] * stats['recall']) / (stats['precision'] + stats['recall'])
        else:
            stats['f1_score'] = None

        # Detection window stats
        stats['window_min_hours'] = WINDOW_MIN_HOURS
        stats['window_max_hours'] = WINDOW_MAX_HOURS

        # Lead time analysis (for TPs)
        lead_times = []
        for brk in breaks:
            break_time = self._parse_timestamp(brk['timestamp'])
            if not break_time:
                continue

            for flare in flares:
                flare_time = self._parse_timestamp(flare['timestamp'])
                if not flare_time:
                    continue

                delta_hours = (flare_time - break_time).total_seconds() / 3600
                if WINDOW_MIN_HOURS <= delta_hours <= WINDOW_MAX_HOURS:
                    lead_times.append(delta_hours)
                    break

        if lead_times:
            stats['avg_lead_time_hours'] = sum(lead_times) / len(lead_times)
            stats['min_lead_time_hours'] = min(lead_times)
            stats['max_lead_time_hours'] = max(lead_times)
        else:
            stats['avg_lead_time_hours'] = None
            stats['min_lead_time_hours'] = None
            stats['max_lead_time_hours'] = None

        return stats

    def get_daily_breakdown(self, days: int = 7) -> list[dict]:
        """Get day-by-day breakdown."""
        return self._query("""
            SELECT
                date(timestamp) as date,
                COUNT(*) as measurements,
                AVG(delta_mi) as avg_mi,
                SUM(CASE WHEN status = 'ALERT' THEN 1 ELSE 0 END) as alerts,
                SUM(CASE WHEN status = 'WARNING' THEN 1 ELSE 0 END) as warnings
            FROM coupling_measurements
            WHERE timestamp >= datetime('now', ?)
            GROUP BY date(timestamp)
            ORDER BY date DESC
        """, (f'-{days} days',))

    def get_recent_events(self, hours: int = 24) -> list[dict]:
        """Get recent significant events."""
        events = []

        # Recent breaks
        breaks = self._query("""
            SELECT timestamp, pair, delta_mi, status, deviation_pct
            FROM coupling_measurements
            WHERE timestamp >= datetime('now', ?)
            AND status IN ('ALERT', 'WARNING')
            ORDER BY timestamp DESC
        """, (f'-{hours} hours',))

        for b in breaks:
            events.append({
                'time': b['timestamp'],
                'type': 'COUPLING_BREAK',
                'pair': b['pair'],
                'status': b['status'],
                'details': f"ΔMI={b['delta_mi']:.3f}, dev={b['deviation_pct']:.1f}%"
            })

        # Recent flares (C+)
        flares = self._query("""
            SELECT timestamp, flux, flare_class
            FROM goes_xray
            WHERE timestamp >= datetime('now', ?)
            AND flux >= 1e-6
            ORDER BY timestamp DESC
        """, (f'-{hours} hours',))

        for f in flares:
            events.append({
                'time': f['timestamp'],
                'type': 'FLARE',
                'class': f['flare_class'],
                'details': f"flux={f['flux']:.2e} W/m²"
            })

        # Sort by time
        events.sort(key=lambda x: x['time'], reverse=True)
        return events

    # =========================================================================
    # REPORT FORMATTING
    # =========================================================================

    def format_markdown(self, days: int = 7) -> str:
        """Generate full Markdown report."""
        stats = self.get_summary_stats(days)
        precursor = self.get_precursor_statistics()
        daily = self.get_daily_breakdown(days)
        events = self.get_recent_events(24)

        lines = []
        lines.append(f"# Solar Early Warning Report")
        lines.append(f"")
        lines.append(f"Generated: {self.now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        lines.append(f"Period: {stats['period_start']} to {stats['period_end']} ({days} days)")
        lines.append(f"")

        # Summary
        lines.append(f"## Summary")
        lines.append(f"")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| GOES Measurements | {stats['goes'].get('count', 0):,} |")
        lines.append(f"| Coupling Measurements | {stats['coupling'].get('count', 0):,} |")
        lines.append(f"| Channel Pairs Monitored | {stats['coupling'].get('pairs', 0)} |")
        lines.append(f"| Total Alerts | {stats['alerts'].get('total_alerts', 0)} |")
        lines.append(f"| Total Warnings | {stats['alerts'].get('total_warnings', 0)} |")
        lines.append(f"")

        # Flare Activity
        lines.append(f"## Flare Activity")
        lines.append(f"")
        if stats['flare_counts']:
            lines.append(f"| Class | Count |")
            lines.append(f"|-------|-------|")
            for cls in ['X', 'M', 'C', 'B', 'A']:
                if cls in stats['flare_counts']:
                    lines.append(f"| {cls} | {stats['flare_counts'][cls]} |")
        else:
            lines.append(f"No significant flare activity recorded.")
        lines.append(f"")

        # Coupling by Pair
        lines.append(f"## Coupling Analysis by Channel Pair")
        lines.append(f"")
        lines.append(f"| Pair | Measurements | Avg ΔMI | Min | Max | Alerts | Warnings |")
        lines.append(f"|------|--------------|---------|-----|-----|--------|----------|")
        for p in stats['coupling_by_pair']:
            avg = p['avg_mi'] or 0
            min_mi = p['min_mi'] or 0
            max_mi = p['max_mi'] or 0
            lines.append(f"| {p['pair']} | {p['count']} | {avg:.3f} | {min_mi:.3f} | {max_mi:.3f} | {p['alerts']} | {p['warnings']} |")
        lines.append(f"")

        # Precursor Statistics
        lines.append(f"## Precursor Detection Statistics")
        lines.append(f"")
        lines.append(f"Detection window: {precursor['window_min_hours']:.1f} - {precursor['window_max_hours']:.1f} hours after break")
        lines.append(f"")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Breaks Detected | {precursor['total_breaks']} |")
        lines.append(f"| Total Flares (C+) | {precursor['total_flares']} |")
        lines.append(f"| True Positives (TP) | {precursor['true_positives']} |")
        lines.append(f"| False Positives (FP) | {precursor['false_positives']} |")
        lines.append(f"| False Negatives (FN) | {precursor['false_negatives']} |")

        if precursor['precision'] is not None:
            lines.append(f"| **Precision** | **{precursor['precision']:.1%}** |")
        else:
            lines.append(f"| Precision | N/A |")

        if precursor['recall'] is not None:
            lines.append(f"| **Recall** | **{precursor['recall']:.1%}** |")
        else:
            lines.append(f"| Recall | N/A |")

        if precursor['f1_score'] is not None:
            lines.append(f"| **F1 Score** | **{precursor['f1_score']:.3f}** |")
        else:
            lines.append(f"| F1 Score | N/A |")

        if precursor['avg_lead_time_hours'] is not None:
            lines.append(f"| Avg Lead Time | {precursor['avg_lead_time_hours']:.1f} hours |")
            lines.append(f"| Min Lead Time | {precursor['min_lead_time_hours']:.1f} hours |")
            lines.append(f"| Max Lead Time | {precursor['max_lead_time_hours']:.1f} hours |")
        lines.append(f"")

        # Daily Breakdown
        lines.append(f"## Daily Breakdown")
        lines.append(f"")
        lines.append(f"| Date | Measurements | Avg ΔMI | Alerts | Warnings |")
        lines.append(f"|------|--------------|---------|--------|----------|")
        for d in daily:
            avg = d['avg_mi'] or 0
            lines.append(f"| {d['date']} | {d['measurements']} | {avg:.3f} | {d['alerts']} | {d['warnings']} |")
        lines.append(f"")

        # Recent Events
        lines.append(f"## Recent Events (Last 24h)")
        lines.append(f"")
        if events:
            lines.append(f"| Time | Type | Details |")
            lines.append(f"|------|------|---------|")
            for e in events[:20]:  # Limit to 20
                if e['type'] == 'COUPLING_BREAK':
                    lines.append(f"| {e['time']} | {e['status']} | {e['pair']}: {e['details']} |")
                else:
                    lines.append(f"| {e['time']} | FLARE {e.get('class', '')} | {e['details']} |")
        else:
            lines.append(f"No significant events in the last 24 hours.")
        lines.append(f"")

        # Interpretation Guide
        lines.append(f"## Interpretation Guide")
        lines.append(f"")
        lines.append(f"- **Precision**: Of all breaks detected, what fraction preceded a flare? (Higher = fewer false alarms)")
        lines.append(f"- **Recall**: Of all flares that occurred, what fraction had a preceding break? (Higher = fewer missed events)")
        lines.append(f"- **F1 Score**: Harmonic mean of Precision and Recall (0-1, higher is better)")
        lines.append(f"- **Lead Time**: Time between break detection and flare onset")
        lines.append(f"")

        return "\n".join(lines)

    def format_html(self, days: int = 7) -> str:
        """Generate HTML report."""
        md = self.format_markdown(days)

        # Simple MD to HTML conversion
        html_lines = []
        html_lines.append("<!DOCTYPE html>")
        html_lines.append("<html><head>")
        html_lines.append("<meta charset='utf-8'>")
        html_lines.append("<title>Solar Early Warning Report</title>")
        html_lines.append("<style>")
        html_lines.append("body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }")
        html_lines.append("h1 { color: #2c3e50; border-bottom: 2px solid #e67e22; padding-bottom: 10px; }")
        html_lines.append("h2 { color: #34495e; margin-top: 30px; }")
        html_lines.append("table { border-collapse: collapse; width: 100%; margin: 15px 0; }")
        html_lines.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html_lines.append("th { background-color: #3498db; color: white; }")
        html_lines.append("tr:nth-child(even) { background-color: #f9f9f9; }")
        html_lines.append("tr:hover { background-color: #f5f5f5; }")
        html_lines.append(".metric-good { color: #27ae60; font-weight: bold; }")
        html_lines.append(".metric-bad { color: #e74c3c; font-weight: bold; }")
        html_lines.append(".alert { background-color: #ffebee; }")
        html_lines.append(".warning { background-color: #fff3e0; }")
        html_lines.append("</style>")
        html_lines.append("</head><body>")

        in_table = False
        for line in md.split('\n'):
            # Headers
            if line.startswith('# '):
                html_lines.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith('## '):
                html_lines.append(f"<h2>{line[3:]}</h2>")
            # Tables
            elif line.startswith('|'):
                if not in_table:
                    html_lines.append("<table>")
                    in_table = True

                cells = [c.strip() for c in line.split('|')[1:-1]]
                if all(c.replace('-', '') == '' for c in cells):
                    continue  # Skip separator row

                if '**' in line:
                    # Header or bold row
                    row = "<tr>"
                    for c in cells:
                        c = c.replace('**', '')
                        if 'Precision' in c or 'Recall' in c or 'F1' in c:
                            row += f"<td class='metric-good'>{c}</td>"
                        else:
                            row += f"<td>{c}</td>"
                    row += "</tr>"
                    html_lines.append(row)
                elif cells and cells[0] in ['Metric', 'Class', 'Pair', 'Date', 'Time']:
                    html_lines.append("<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")
                else:
                    row_class = ""
                    if 'ALERT' in line:
                        row_class = " class='alert'"
                    elif 'WARNING' in line:
                        row_class = " class='warning'"
                    html_lines.append(f"<tr{row_class}>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            else:
                if in_table:
                    html_lines.append("</table>")
                    in_table = False
                if line.startswith('- '):
                    html_lines.append(f"<li>{line[2:]}</li>")
                elif line:
                    html_lines.append(f"<p>{line}</p>")

        if in_table:
            html_lines.append("</table>")

        html_lines.append("</body></html>")
        return "\n".join(html_lines)

    def format_text(self, days: int = 7) -> str:
        """Generate plain text report for terminal."""
        stats = self.get_summary_stats(days)
        precursor = self.get_precursor_statistics()
        daily = self.get_daily_breakdown(days)

        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append("  SOLAR EARLY WARNING REPORT")
        lines.append("=" * 60)
        lines.append(f"  Generated: {self.now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        lines.append(f"  Period: {stats['period_start']} to {stats['period_end']} ({days} days)")
        lines.append("")

        # Summary
        lines.append("  SUMMARY")
        lines.append("  " + "-" * 40)
        lines.append(f"  GOES Measurements:     {stats['goes'].get('count', 0):,}")
        lines.append(f"  Coupling Measurements: {stats['coupling'].get('count', 0):,}")
        lines.append(f"  Channel Pairs:         {stats['coupling'].get('pairs', 0)}")
        lines.append(f"  Total Alerts:          {stats['alerts'].get('total_alerts', 0)}")
        lines.append(f"  Total Warnings:        {stats['alerts'].get('total_warnings', 0)}")
        lines.append("")

        # Flare Activity
        lines.append("  FLARE ACTIVITY")
        lines.append("  " + "-" * 40)
        if stats['flare_counts']:
            for cls in ['X', 'M', 'C', 'B', 'A']:
                if cls in stats['flare_counts']:
                    lines.append(f"  {cls}-class: {stats['flare_counts'][cls]}")
        else:
            lines.append("  No significant flare activity")
        lines.append("")

        # Precursor Statistics
        lines.append("  PRECURSOR DETECTION STATISTICS")
        lines.append("  " + "-" * 40)
        lines.append(f"  Window: {precursor['window_min_hours']:.1f} - {precursor['window_max_hours']:.1f} hours")
        lines.append(f"  Total Breaks:     {precursor['total_breaks']}")
        lines.append(f"  Total Flares:     {precursor['total_flares']}")
        lines.append(f"  True Positives:   {precursor['true_positives']}")
        lines.append(f"  False Positives:  {precursor['false_positives']}")
        lines.append(f"  False Negatives:  {precursor['false_negatives']}")
        lines.append("")

        if precursor['precision'] is not None:
            lines.append(f"  Precision:  {precursor['precision']:.1%}")
        if precursor['recall'] is not None:
            lines.append(f"  Recall:     {precursor['recall']:.1%}")
        if precursor['f1_score'] is not None:
            lines.append(f"  F1 Score:   {precursor['f1_score']:.3f}")
        if precursor['avg_lead_time_hours'] is not None:
            lines.append(f"  Avg Lead:   {precursor['avg_lead_time_hours']:.1f} hours")
        lines.append("")

        # Daily Breakdown
        lines.append("  DAILY BREAKDOWN")
        lines.append("  " + "-" * 40)
        lines.append(f"  {'Date':<12} {'Meas':>6} {'AvgMI':>7} {'Alert':>6} {'Warn':>6}")
        for d in daily:
            avg = d['avg_mi'] or 0
            lines.append(f"  {d['date']:<12} {d['measurements']:>6} {avg:>7.3f} {d['alerts']:>6} {d['warnings']:>6}")
        lines.append("")
        lines.append("=" * 60)
        lines.append("")

        return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Report generation CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Solar Early Warning - Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/report.py                    # Full text report
  uv run python scripts/report.py --daily            # Last 24h summary
  uv run python scripts/report.py --weekly           # Last 7 days
  uv run python scripts/report.py --stats            # Precursor statistics only
  uv run python scripts/report.py --export md        # Export as Markdown
  uv run python scripts/report.py --export html      # Export as HTML
  uv run python scripts/report.py -o report.html --export html
        """
    )

    parser.add_argument('--daily', action='store_true', help='Daily summary (last 24h)')
    parser.add_argument('--weekly', action='store_true', help='Weekly summary (last 7 days)')
    parser.add_argument('--monthly', action='store_true', help='Monthly summary (last 30 days)')
    parser.add_argument('--stats', action='store_true', help='Show precursor statistics only')
    parser.add_argument('--days', type=int, default=7, help='Number of days to include (default: 7)')
    parser.add_argument('--export', choices=['md', 'html', 'text', 'json'], help='Export format')
    parser.add_argument('-o', '--output', type=str, help='Output file path')

    args = parser.parse_args()

    # Determine period
    if args.daily:
        days = 1
    elif args.weekly:
        days = 7
    elif args.monthly:
        days = 30
    else:
        days = args.days

    # Generate report
    report = ReportGenerator()

    if args.stats:
        # Just precursor statistics
        stats = report.get_precursor_statistics()
        print("\n  PRECURSOR DETECTION STATISTICS")
        print("  " + "-" * 40)
        print(f"  Total Breaks:     {stats['total_breaks']}")
        print(f"  Total Flares:     {stats['total_flares']}")
        print(f"  True Positives:   {stats['true_positives']}")
        print(f"  False Positives:  {stats['false_positives']}")
        print(f"  False Negatives:  {stats['false_negatives']}")
        print()
        if stats['precision'] is not None:
            print(f"  Precision:  {stats['precision']:.1%}")
        if stats['recall'] is not None:
            print(f"  Recall:     {stats['recall']:.1%}")
        if stats['f1_score'] is not None:
            print(f"  F1 Score:   {stats['f1_score']:.3f}")
        if stats['avg_lead_time_hours'] is not None:
            print(f"  Avg Lead:   {stats['avg_lead_time_hours']:.1f} hours")
        print()
        return

    # Full report
    if args.export == 'md':
        content = report.format_markdown(days)
    elif args.export == 'html':
        content = report.format_html(days)
    elif args.export == 'json':
        data = {
            'summary': report.get_summary_stats(days),
            'precursor': report.get_precursor_statistics(),
            'daily': report.get_daily_breakdown(days),
            'events': report.get_recent_events(24),
        }
        content = json.dumps(data, indent=2, default=str)
    else:
        content = report.format_text(days)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        print(f"Report saved to: {output_path}")
    else:
        print(content)


if __name__ == "__main__":
    main()
