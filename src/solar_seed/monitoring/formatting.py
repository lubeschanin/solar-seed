"""
Status Formatting
=================

Formatters for the early warning system status reports.
"""

from datetime import datetime, timezone


class StatusFormatter:
    """
    Format status reports for the early warning system.

    Separates formatting logic from data collection to keep
    the main script focused on orchestration.
    """

    # Icon mappings
    SEVERITY_ICONS = ['', '', '', '', '']
    STATUS_ICONS = {
        'NORMAL': '',
        'ELEVATED': '',
        'WARNING': '',
        'ALERT': ''
    }
    TREND_ICONS = {
        'ACCELERATING_DOWN': '',
        'DECLINING': '',
        'STABLE': '',
        'RISING': '',
        'ACCELERATING_UP': '',
        'INITIALIZING': '',
        'NO_DATA': ''
    }
    CONFIDENCE_MARKERS = {
        'high': '',
        'medium': '',
        'low': '',
        'none': ''
    }
    RISK_ICONS = ['', '', '', '']
    STATE_ICONS = {'TRANSFER_STATE': '↓↑', 'RECOVERY_STATE': '↑↓'}

    def __init__(self, width: int = 70):
        self.width = width

    def format_header(self) -> list[str]:
        """Format the main status header."""
        now = datetime.now(timezone.utc)
        return [
            '',
            '=' * self.width,
            f"  SOLAR EARLY WARNING SYSTEM - {now.strftime('%Y-%m-%d %H:%M:%S')} UTC",
            '=' * self.width,
        ]

    def format_footer(self) -> list[str]:
        """Format the status footer."""
        return ['', '=' * self.width, '']

    def format_xray_status(self, xray: dict) -> list[str]:
        """Format GOES X-ray status section."""
        lines = [
            '',
            '  GOES X-RAY STATUS [Contextual Indicator]',
            f"  {'-'*40}",
        ]

        if xray:
            icon = self.SEVERITY_ICONS[min(xray['severity'], 4)]
            lines.append(f"  Flux:        {xray['flux']:.2e} W/m²")
            lines.append(f"  Flare Class: {icon} {xray['flare_class']}")
            lines.append(f"  Timestamp:   {xray['timestamp']}")

            if xray['severity'] >= 3:
                lines.append('')
                lines.append(f"  *** FLARE ALERT: {xray['flare_class']} class flare detected! ***")
        else:
            lines.append('  Data unavailable')

        return lines

    def format_solar_wind(self, solar_wind: dict, assess_risk_func=None) -> list[str]:
        """
        Format solar wind status section.

        Args:
            solar_wind: Solar wind data dict
            assess_risk_func: Function to assess geomagnetic risk (returns (risk_str, risk_level))
        """
        lines = [
            '',
            '  SOLAR WIND (DSCOVR L1) [Contextual Indicator]',
            f"  {'-'*40}",
        ]

        if not solar_wind:
            lines.append('  Data unavailable')
            return lines

        partial_data = False

        if 'plasma' in solar_wind:
            p = solar_wind['plasma']
            speed = p.get('speed')
            density = p.get('density')
            lines.append(f"  Speed:       {speed:.1f} km/s" if speed else "  Speed:       unavailable")
            lines.append(f"  Density:     {density:.2f} p/cm³" if density else "  Density:     unavailable")
            if speed is None or density is None:
                partial_data = True

        if 'mag' in solar_wind:
            m = solar_wind['mag']
            bz = m.get('bz')
            bt = m.get('bt')
            lines.append(f"  Bz:          {bz:.2f} nT" if bz is not None else "  Bz:          unavailable")
            lines.append(f"  Bt:          {bt:.2f} nT" if bt is not None else "  Bt:          unavailable")
            if bz is None or bt is None:
                partial_data = True

        if assess_risk_func:
            risk, risk_level = assess_risk_func(solar_wind)
            icon = self.RISK_ICONS[risk_level]
            suffix = ' (partial data)' if partial_data else ''
            lines.append('')
            lines.append(f"  Geomag Risk: {icon} {risk}{suffix}")

        return lines

    def format_coupling_pair(self, pair: str, data: dict) -> list[str]:
        """Format a single coupling pair status."""
        lines = []

        icon = self.STATUS_ICONS.get(data.get('status', 'NORMAL'), '')
        trend = data.get('trend', 'NO_DATA')
        trend_icon = self.TREND_ICONS.get(trend, '')
        conf = data.get('confidence', 'none')
        conf_marker = self.CONFIDENCE_MARKERS.get(conf, '')

        residual = data.get('residual', 0)
        slope = data.get('slope_pct_per_hour', 0)
        n_pts = data.get('n_points', 0)
        window_min = data.get('window_min', 0)
        method = data.get('method', 'Theil-Sen')

        artifact_mark = " ARTIFACT?" if data.get('artifact_warning') else ""
        lines.append(f"  {pair} Å: {data['delta_mi']:.3f} bits  r={residual:+.1f}σ  {icon} {data.get('status', '?')}{artifact_mark}")

        # Trend line
        if trend == 'NO_DATA':
            reason = data.get('reason', 'No data')
            lines.append(f"           Trend: {trend_icon} {trend} — {reason}")
        elif trend == 'COLLECTING':
            reason = data.get('reason', '')
            lines.append(f"           Trend: {trend_icon} {trend} — {reason}")
        else:
            acc = data.get('acceleration', 0)
            acc_str = f", acc={acc:+.1f}%/h²" if abs(acc) > 1 else ""
            # Format window time
            if window_min >= 60:
                window_str = f"{window_min/60:.1f}h"
            else:
                window_str = f"{window_min:.0f}min"
            lines.append(f"           Trend: {trend_icon} {trend} ({slope:+.1f}%/h{acc_str})")
            lines.append(f"                  {conf_marker} {conf} confidence | n={n_pts} | {window_str} window | {method}")

        # Status warnings
        if data.get('status') in ['WARNING', 'ALERT']:
            pass  # Mark as alert (caller handles)

        # Accelerating decline warning
        if trend == 'ACCELERATING_DOWN' and data.get('status') in ['ELEVATED', 'WARNING', 'ALERT']:
            lines.append(f"           Coupling declining and accelerating!")

        # Break detection status
        if data.get('is_break'):
            z_mad = data.get('z_mad', 0)
            lines.append(f"           *** VALIDATED BREAK ({z_mad:.1f} MAD below median) ***")
        elif data.get('break_vetoed'):
            z_mad = data.get('z_mad', 0)
            lines.append(f"           [VETOED: {data['break_vetoed']}] anomaly observed ({z_mad:.1f} MAD) - diagnostic only")

        # Registration shift
        reg_shift = data.get('registration_shift', 0)
        if reg_shift > 3:
            lines.append(f"           Registration: {reg_shift:.1f}px shift")

        return lines

    def format_coupling_quality(self, quality: dict) -> list[str]:
        """Format coupling data quality section."""
        lines = []
        if not quality:
            return lines

        n_warn = quality.get('n_warnings', 0)
        if n_warn == 0:
            lines.append(f"  ✓ Data quality: GOOD")
        else:
            lines.append(f"  Data quality: {n_warn} warning(s)")
            for w in quality.get('warnings', [])[:3]:
                lines.append(f"    - {w}")
        return lines

    def format_alert_engine(self, actionable_breaks: list, breaks: dict,
                            anomaly_statuses: dict, AnomalyStatus) -> list[str]:
        """Format the ALERT ENGINE section for actionable breaks."""
        if not actionable_breaks:
            return []

        lines = [
            '',
            '  ╔═══════════════════════════════════════════════════════════╗',
            '  ║  ALERT ENGINE                                             ║',
            '  ╚═══════════════════════════════════════════════════════════╝',
            '',
            '  *** VALIDATED PRECURSOR BREAK ***',
            '  Reduced coupling during rising/active phase',
            '  Recommend: Monitor for potential flare activity',
        ]

        for pair in actionable_breaks:
            bd = breaks.get(pair, {})
            status = anomaly_statuses.get(pair, {})
            z_mad = bd.get('z_mad', status.get('z_mad', 0))
            break_type = status.get('break_type', 'PRECURSOR')
            phase_reason = status.get('phase_reason', '')

            lines.append('')
            lines.append(f"  {pair}: ✓ VALIDATED_BREAK ({break_type})")
            lines.append(f"    Criterion: {bd.get('criterion', '?')}")
            lines.append(f"    Deviation: {z_mad:.1f} MAD below median")
            if phase_reason:
                lines.append(f"    Phase: {phase_reason}")

            # Show passed tests
            passed = status.get('passed_tests', [])
            for test in passed:
                lines.append(f"    ✓ {test}")

        return lines

    def format_data_errors(self, data_errors: list, anomaly_statuses: dict) -> list[str]:
        """Format DATA ERRORS section."""
        if not data_errors:
            return []

        lines = [
            '',
            '  DATA QUALITY ISSUES (excluded from analysis):',
        ]
        for pair in data_errors:
            status = anomaly_statuses.get(pair, {})
            error_reason = status.get('error_reason', 'Unknown error')
            lines.append(f"    {pair}: {error_reason}")
        lines.append('    Note: These measurements are not counted in statistics')
        return lines

    def format_diagnostics(self, diagnostic_breaks: list, breaks: dict,
                           anomaly_statuses: dict, transfer: dict, BreakType) -> list[str]:
        """Format PHYSICS DIAGNOSTICS section."""
        if not diagnostic_breaks and not transfer:
            return []

        lines = [
            '',
            '  ┌───────────────────────────────────────────────────────────┐',
            '  │  PHYSICS DIAGNOSTICS (contextual, not for triggering)     │',
            '  └───────────────────────────────────────────────────────────┘',
        ]

        # Vetoed anomalies
        if diagnostic_breaks:
            lines.append('')
            lines.append('  Observed Anomalies (vetoed — interpretable, not actionable):')
            for pair in diagnostic_breaks:
                bd = breaks.get(pair, {})
                status = anomaly_statuses.get(pair, {})
                z_mad = bd.get('z_mad', status.get('z_mad', 0))
                veto_reasons = status.get('veto_reasons', [bd.get('vetoed', 'unknown')])
                break_type = status.get('break_type', 'UNKNOWN')
                phase_reason = status.get('phase_reason', '')

                if break_type == BreakType.POSTCURSOR:
                    lines.append('')
                    lines.append(f"  {pair}: VALIDATED BREAK ({break_type})")
                    lines.append(f"    Classification: Late/Post-Event Break — no alert")
                    lines.append(f"    Reason: {phase_reason}")
                else:
                    lines.append('')
                    lines.append(f"  {pair}: {z_mad:.1f}σ anomaly (VETOED)")
                    if veto_reasons:
                        lines.append(f"    Veto: {', '.join(veto_reasons)}")

                passed = status.get('passed_tests', [])
                if passed:
                    lines.append(f"    Passed: {', '.join(passed)}")

        # Transfer/Recovery state
        if transfer:
            icon = self.STATE_ICONS.get(transfer['state'], '⟷')
            degraded = transfer.get('degraded', False)
            degraded_reasons = transfer.get('degraded_reasons', [])
            is_async = any('ASYNC' in r for r in degraded_reasons)

            lines.append('')
            lines.append('  State Analysis:')
            if degraded:
                suffix = '(DEGRADED — ASYNC)' if is_async else '(DEGRADED)'
                lines.append(f"  {icon} {transfer['state']} {suffix} — diagnostic only")
                lines.append(f"     {transfer['description']}")
                lines.append(f"     193-304: {transfer['slope_193_304']:+.1f}%/h  193-211: {transfer['slope_193_211']:+.1f}%/h")
                lines.append(f"     Note: State interpretable but not trigger-grade due to:")
                for reason in degraded_reasons:
                    lines.append(f"       - {reason}")
            else:
                lines.append(f"  {icon} {transfer['state']} ({transfer['confidence']} confidence)")
                lines.append(f"     {transfer['description']}")
                lines.append(f"     193-304: {transfer['slope_193_304']:+.1f}%/h  193-211: {transfer['slope_193_211']:+.1f}%/h")
                lines.append(f"     Interpretation: {transfer['interpretation']}")

        lines.append('')
        lines.append('  ' + '─' * 61)
        lines.append('  Note: Diagnostics remain interpretable even when robustness')
        lines.append('        vetoes prevent actionable alerts ("vetoed ≠ blind").')

        return lines

    def format_stereo_section(self, stereo: dict, coupling: dict = None) -> list[str]:
        """Format STEREO-A advance warning section."""
        if not stereo:
            return []

        meta = stereo.get('_stereo_metadata', {})
        ts = stereo.get('_timestamp', 'unknown')
        sep = meta.get('separation_deg', 51)
        days = meta.get('advance_warning_days', 3.9)

        lines = [
            '',
            f"  STEREO-A EUVI ({sep:.0f}° ahead → ~{days:.1f} days warning)",
            f"  {'-'*40}",
            f"  Image time: {ts}",
        ]

        for pair, data in stereo.items():
            if pair.startswith('_'):
                continue
            euvi_wl = data.get('euvi_wavelengths', '')
            lines.append(f"  {pair} Å: {data['delta_mi']:.3f} bits  (EUVI {euvi_wl})")

        # Comparison with SDO
        if coupling:
            lines.append('')
            lines.append('  Comparison (STEREO-A vs SDO/AIA):')
            for pair in ['193-211', '193-304']:
                if pair in stereo and pair in coupling:
                    stereo_mi = stereo[pair]['delta_mi']
                    sdo_mi = coupling[pair]['delta_mi']
                    diff_pct = (stereo_mi - sdo_mi) / sdo_mi * 100 if sdo_mi else 0
                    arrow = "↑" if diff_pct > 10 else "↓" if diff_pct < -10 else "≈"
                    lines.append(f"    {pair}: STEREO {stereo_mi:.3f} vs SDO {sdo_mi:.3f} ({arrow} {diff_pct:+.0f}%)")

        return lines

    def format_alerts_section(self, alerts: list) -> list[str]:
        """Format NOAA alerts section."""
        if not alerts:
            return []

        lines = [
            '',
            '  NOAA ALERTS (last 24h)',
            f"  {'-'*40}",
        ]
        for alert in alerts[:3]:
            lines.append(f"  [{alert['type']}] {alert['message'][:60]}...")

        return lines

    def generate_event_narrative(self, xray: dict, coupling: dict) -> str | None:
        """
        Generate a compact Event Narrative Box (A&A Science Highlight style).

        Shows key events and coupling evolution during active phases.
        Returns None if no significant activity to report.
        """
        if not xray or not coupling:
            return None

        flux = xray.get('flux', 0)
        flare_class = xray.get('flare_class', 'A0')

        mi_211 = coupling.get('193-211', {})
        mi_304 = coupling.get('193-304', {})

        delta_211 = mi_211.get('delta_mi', 0)
        delta_304 = mi_304.get('delta_mi', 0)
        r_211 = mi_211.get('residual', 0)
        r_304 = mi_304.get('residual', 0)
        slope_211 = mi_211.get('slope_pct_per_hour', 0)
        slope_304 = mi_304.get('slope_pct_per_hour', 0)

        # Determine phase
        if flux >= 1e-5:
            phase, phase_icon = "M/X-CLASS ACTIVE", "⚡"
        elif flux >= 1e-6:
            phase, phase_icon = "C-CLASS ACTIVE", "🔥"
        elif flux >= 5e-7:
            if slope_211 < -2 or slope_304 > 2:
                phase, phase_icon = "DECAY PHASE", "📉"
            else:
                phase, phase_icon = "ELEVATED", "📈"
        else:
            if r_304 > 3 and slope_304 > 0:
                phase, phase_icon = "POST-FLARE DECAY", "🌅"
            elif abs(r_211) > 2 or abs(r_304) > 2:
                phase, phase_icon = "ANOMALOUS", "⚠"
            else:
                return None

        lines = [
            "",
            "  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓",
            f"  ┃  {phase_icon} EVENT NARRATIVE: {phase:<40}┃",
            "  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛",
            "",
            "  ┌─────────────┬────────────┬────────────┬─────────────┐",
            "  │   Channel   │    ΔMI     │   r (σ)    │    Trend    │",
            "  ├─────────────┼────────────┼────────────┼─────────────┤",
        ]

        # 193-211 row
        trend_arrow_211 = "↓" if slope_211 < -1 else "↑" if slope_211 > 1 else "→"
        r_str_211 = f"{r_211:+.1f}σ" if r_211 != 0 else "  —"
        lines.append(f"  │  193-211 Å  │   {delta_211:.3f}   │  {r_str_211:>6}   │  {trend_arrow_211} {slope_211:+.1f}%/h  │")

        # 193-304 row
        trend_arrow_304 = "↓" if slope_304 < -1 else "↑" if slope_304 > 1 else "→"
        r_str_304 = f"{r_304:+.1f}σ" if r_304 != 0 else "  —"
        lines.append(f"  │  193-304 Å  │   {delta_304:.3f}   │  {r_str_304:>6}   │  {trend_arrow_304} {slope_304:+.1f}%/h  │")

        lines.extend([
            "  └─────────────┴────────────┴────────────┴─────────────┘",
            "",
            "  Interpretation:",
        ])

        # Interpretation based on phase
        if phase in ("POST-FLARE DECAY", "DECAY PHASE"):
            if r_304 > r_211:
                lines.append(f"  • 193-304 Å shows peak significance (r={r_304:+.1f}σ) during decay phase")
                lines.append(f"  • Coronal coupling (193-211) weakening ({slope_211:+.1f}%/h)")
                lines.append(f"  • Enhanced low-atmosphere coherence relative to coronal morphology")
                lines.append(f"  • Consistent with post-flare footpoint/transition-region dominance")
            else:
                lines.append("  • Coupling returning to baseline after elevated activity")
                lines.append("  • Both channels showing recovery trends")
        elif phase in ("C-CLASS ACTIVE", "M/X-CLASS ACTIVE"):
            lines.append(f"  • Active phase: GOES {flare_class} ({flux:.2e} W/m²)")
            if r_211 < -1:
                lines.append(f"  • Coronal coupling suppressed (r={r_211:+.1f}σ) — magnetic reorganization")
            if r_304 > 1:
                lines.append(f"  • Chromospheric anchor strengthening (r={r_304:+.1f}σ)")
            lines.append("  • Monitor for coupling break as stress indicator")
        elif phase == "ANOMALOUS":
            lines.append("  • Unusual coupling configuration detected")
            if abs(r_211) > 2:
                lines.append(f"  • 193-211 Å: r={r_211:+.1f}σ deviation from baseline")
            if abs(r_304) > 2:
                lines.append(f"  • 193-304 Å: r={r_304:+.1f}σ deviation from baseline")
            lines.append("  • Possible precursor signature or instrument artifact")

        # Transfer state
        transfer = coupling.get('_transfer_state')
        if transfer:
            state = transfer.get('state', '')
            if state == 'TRANSFER_STATE':
                lines.append("  • TRANSFER_STATE: Energy redistribution toward lower atmosphere")
            elif state == 'RECOVERY_STATE':
                lines.append("  • RECOVERY_STATE: Post-flare relaxation in progress")

        lines.append("")
        return "\n".join(lines)
