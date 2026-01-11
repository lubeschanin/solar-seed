"""
Status Formatting with Rich
============================

Beautiful terminal output using Rich library.
"""

from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.style import Style
from rich import box

from .constants import (
    AnomalyLevel, Phase, get_anomaly_level,
    classify_phase, classify_phase_parallel,
)
from .relevance import assess_personal_relevance, get_subsolar_point, LOCATIONS


console = Console()


class StatusFormatter:
    """
    Format status reports using Rich for beautiful terminal output.
    """

    # Anomaly level styles (statistical)
    ANOMALY_STYLES = {
        AnomalyLevel.NORMAL: Style(color="green"),
        AnomalyLevel.ELEVATED: Style(color="yellow"),
        AnomalyLevel.STRONG: Style(color="bright_yellow", bold=True),
        AnomalyLevel.EXTREME: Style(color="red", bold=True),
        'DATA_ERROR': Style(color="bright_black"),
    }

    # Phase styles (interpretive) - refined semantic palette
    # 🟢 Quiet states: BASELINE, ELEVATED-QUIET
    # 🟣 Transitional: POST-EVENT
    # 🟡 Decaying: RECOVERY
    # ⚠️ Alert: PRE-FLARE
    # 🔴 Active: ACTIVE
    PHASE_STYLES = {
        # Quiet states (green shades)
        Phase.BASELINE: Style(color="green"),
        Phase.ELEVATED_QUIET: Style(color="bright_green"),

        # Transitional states
        Phase.POST_EVENT: Style(color="magenta", bold=True),
        Phase.RECOVERY: Style(color="yellow"),

        # Alert states
        Phase.PRE_FLARE: Style(color="red", bold=True),
        Phase.ACTIVE: Style(color="bright_red", bold=True),
    }

    PHASE_ICONS = {
        # Quiet states
        Phase.BASELINE: '🟢',
        Phase.ELEVATED_QUIET: '🟢',  # Still green - structurally active but stable

        # Transitional states
        Phase.POST_EVENT: '🟣',       # Purple - reorganizing
        Phase.RECOVERY: '🟡',         # Yellow - decaying

        # Alert states
        Phase.PRE_FLARE: '⚠️',        # Warning - destabilization
        Phase.ACTIVE: '🔴',           # Red - ongoing energy release
    }

    TREND_ICONS = {
        'ACCELERATING_DOWN': '⏬',
        'DECLINING': '↘',
        'STABLE': '→',
        'RISING': '↗',
        'ACCELERATING_UP': '⏫',
        'INITIALIZING': '⏳',
        'NO_DATA': '❓',
        'COLLECTING': '📊',
    }

    def __init__(self):
        self.console = console

    def print_header(self):
        """Print the main status header."""
        now = datetime.now(timezone.utc)
        header = Panel(
            f"[bold white]{now.strftime('%Y-%m-%d %H:%M:%S')} UTC[/]",
            title="[bold cyan]☀️ SOLAR EARLY WARNING SYSTEM[/]",
            subtitle="[dim]Prototype v0.4[/]",
            border_style="cyan",
            box=box.DOUBLE,
        )
        self.console.print(header)

    def print_xray_status(self, xray: dict):
        """Print GOES X-ray status panel."""
        if not xray:
            self.console.print(Panel("[dim]Data unavailable[/]", title="🌡️ GOES X-RAY", border_style="dim"))
            return

        severity = xray.get('severity', 0)
        flare_class = xray['flare_class']
        flux = xray['flux']

        # Color based on severity
        if severity >= 4:
            style = "bold red"
            icon = "🔴"
        elif severity >= 3:
            style = "bold yellow"
            icon = "🟠"
        elif severity >= 2:
            style = "yellow"
            icon = "🟡"
        else:
            style = "green"
            icon = "🟢"

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Label", style="dim")
        table.add_column("Value")

        table.add_row("Flux", f"[bold]{flux:.2e}[/] W/m²")
        table.add_row("Class", f"[{style}]{icon} {flare_class}[/]")
        table.add_row("Time", f"[dim]{xray['timestamp']}[/]")

        panel = Panel(table, title="🌡️ GOES X-RAY", border_style="blue", subtitle="[dim]Contextual[/]")
        self.console.print(panel)

        if severity >= 3:
            self.console.print(f"  [bold red]⚠️  FLARE ALERT: {flare_class} class flare detected![/]")

    def print_solar_wind(self, solar_wind: dict, risk_info: tuple = None):
        """Print solar wind status panel."""
        if not solar_wind:
            self.console.print(Panel("[dim]Data unavailable[/]", title="💨 SOLAR WIND", border_style="dim"))
            return

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Label", style="dim")
        table.add_column("Value")

        if 'plasma' in solar_wind:
            p = solar_wind['plasma']
            speed = p.get('speed')
            density = p.get('density')
            if speed:
                speed_style = "red" if speed > 600 else "yellow" if speed > 450 else "green"
                table.add_row("Speed", f"[{speed_style}]{speed:.1f}[/] km/s")
            if density:
                table.add_row("Density", f"{density:.2f} p/cm³")

        if 'mag' in solar_wind:
            m = solar_wind['mag']
            bz = m.get('bz')
            bt = m.get('bt')
            if bz is not None:
                bz_style = "red" if bz < -10 else "yellow" if bz < -5 else "green"
                table.add_row("Bz", f"[{bz_style}]{bz:.2f}[/] nT")
            if bt is not None:
                table.add_row("Bt", f"{bt:.2f} nT")

        if risk_info:
            risk, risk_level = risk_info
            risk_styles = ["green", "yellow", "bright_yellow", "red"]
            risk_icons = ["🟢", "🟡", "🟠", "🔴"]
            table.add_row("", "")  # Spacer
            table.add_row("Geomag Risk", f"[{risk_styles[risk_level]}]{risk_icons[risk_level]} {risk}[/]")

        panel = Panel(table, title="💨 SOLAR WIND (DSCOVR L1)", border_style="blue", subtitle="[dim]Contextual[/]")
        self.console.print(panel)

    def print_coupling_analysis(self, coupling: dict, AnomalyStatus, BreakType,
                                 xray: dict = None):
        """Print coupling analysis with Rich formatting.

        Now shows two separate classifications:
        - Anomaly (statistical): NORMAL/ELEVATED/STRONG/EXTREME based on |z|
        - Phase (interpretive): BASELINE/PRE-FLARE/FLARE/RECOVERY/POST-FLARE REORG
        """
        if not coupling:
            return

        # Quality info
        quality = coupling.get('_quality', {})
        n_warn = quality.get('n_warnings', 0)
        quality_text = "[green]✓ GOOD[/]" if n_warn == 0 else f"[yellow]⚠ {n_warn} warning(s)[/]"

        # Classify phase using BOTH classifiers in parallel
        goes_flux = xray.get('flux') if xray else None
        goes_rising = xray.get('rising', False) if xray else None
        goes_class = xray.get('flare_class') if xray else None

        pairs_data = {k: v for k, v in coupling.items() if not k.startswith('_')}
        phase_comparison = classify_phase_parallel(pairs_data, goes_flux, goes_rising, goes_class)

        # Build coupling table with Anomaly column
        table = Table(
            title=f"Data Quality: {quality_text}",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Pair", style="bold")
        table.add_column("ΔMI", justify="right")
        table.add_column("r(σ)", justify="right")
        table.add_column("Anomaly")  # Statistical level
        table.add_column("Trend")

        for pair, data in coupling.items():
            if pair.startswith('_'):
                continue

            delta_mi = data.get('delta_mi', 0)
            residual = data.get('residual', 0)
            trend = data.get('trend', 'NO_DATA')
            slope = data.get('slope_pct_per_hour', 0)

            # Compute anomaly level from z-score
            anomaly_level = get_anomaly_level(residual)
            anomaly_style = self.ANOMALY_STYLES.get(anomaly_level, Style())
            trend_icon = self.TREND_ICONS.get(trend, '?')

            # Residual coloring (matches anomaly level)
            if abs(residual) >= 7:
                r_style = "red"
            elif abs(residual) >= 4:
                r_style = "bright_yellow"
            elif abs(residual) >= 2:
                r_style = "yellow"
            else:
                r_style = "green"

            # Build trend string
            trend_str = f"{trend_icon} {slope:+.1f}%/h"

            # Anomaly text with sign indicator
            sign = "+" if residual > 0 else ""
            anomaly_text = Text(f"{anomaly_level} ({sign}{residual:.1f}σ)", style=anomaly_style)

            # Break indicator overrides
            if data.get('is_break'):
                anomaly_text = Text("🚨 BREAK", style="bold red")
            elif data.get('break_vetoed'):
                anomaly_text = Text("VETOED", style="dim")

            table.add_row(
                f"{pair} Å",
                f"{delta_mi:.3f}",
                f"[{r_style}]{residual:+.1f}σ[/]",
                anomaly_text,
                trend_str,
            )

        panel = Panel(table, title="📊 ΔMI COUPLING MONITOR", border_style="magenta", subtitle="[dim]Pre-Flare Detection[/]")
        self.console.print(panel)

        # Print parallel phase comparison panel
        self._print_phase_comparison(phase_comparison)

        # Show alerts if any
        self._print_alerts(coupling, AnomalyStatus, BreakType)

    def _print_phase_comparison(self, comparison: dict):
        """Print parallel phase comparison panel showing both classifiers."""
        current_phase, current_reason = comparison['current']
        exp_phase, exp_reason = comparison['experimental']
        is_divergent = comparison['is_divergent']

        # Build side-by-side content using columns
        # Left panel: CURRENT (GOES-only)
        current_icon = self.PHASE_ICONS.get(current_phase, '❓')
        current_style = self.PHASE_STYLES.get(current_phase, Style())

        left_text = Text()
        left_text.append("CURRENT (GOES-only)\n", style="bold dim")
        left_text.append("─" * 28 + "\n", style="dim")
        left_text.append(f"{current_icon} ", style="bold")
        left_text.append(f"{current_phase}\n", style=current_style)
        left_text.append(f"{current_reason}\n\n", style="dim")
        left_text.append("Rule: GOES flux thresholds", style="dim italic")

        # Right panel: EXPERIMENTAL (ΔMI-integrated)
        exp_icon = self.PHASE_ICONS.get(exp_phase, '❓')
        exp_style = self.PHASE_STYLES.get(exp_phase, Style())

        right_text = Text()
        right_text.append("EXPERIMENTAL (ΔMI-integrated)\n", style="bold dim")
        right_text.append("─" * 28 + "\n", style="dim")
        right_text.append(f"{exp_icon} ", style="bold")
        right_text.append(f"{exp_phase}\n", style=exp_style)
        right_text.append(f"{exp_reason}\n\n", style="dim")
        right_text.append("Rule: max(|r|) + GOES", style="dim italic")

        # Create two panels side by side
        left_panel = Panel(left_text, border_style="blue", box=box.ROUNDED, width=35)
        right_panel = Panel(right_text, border_style="magenta", box=box.ROUNDED, width=35)

        columns = Columns([left_panel, right_panel], padding=(0, 2))

        # Divergence status line
        if is_divergent:
            status_line = Text()
            status_line.append("\nStatus: ", style="dim")
            status_line.append("⚠️ DIVERGENT\n", style="bold yellow")
            status_line.append("Interpretation: ", style="dim")
            status_line.append("Energy (GOES) and structure (ΔMI) are temporarily decoupled", style="yellow italic")
            border_style = "yellow"
        else:
            status_line = Text()
            status_line.append("\nStatus: ", style="dim")
            status_line.append("✓ CONSISTENT", style="green")
            border_style = "green"

        # Combine columns and status
        from rich.console import Group
        content = Group(columns, status_line)

        self.console.print(Panel(
            content,
            title="🎯 PHASE COMPARISON",
            border_style=border_style,
            subtitle="[dim]Parallel classifier validation[/]",
        ))

    def _print_phase(self, phase: str, reason: str):
        """Print single interpretive phase panel (legacy)."""
        phase_style = self.PHASE_STYLES.get(phase, Style())
        phase_icon = self.PHASE_ICONS.get(phase, '❓')

        phase_text = Text()
        phase_text.append(f"{phase_icon} ", style="bold")
        phase_text.append(phase, style=phase_style)
        phase_text.append(f"\n{reason}", style="dim")

        border_color = "green" if phase == Phase.BASELINE else "yellow" if phase == Phase.RECOVERY else "red" if phase in [Phase.PRE_FLARE, Phase.FLARE] else "cyan"

        self.console.print(Panel(
            phase_text,
            title="🎯 PHASE (Interpretive)",
            border_style=border_color,
            subtitle="[dim]Rule-based classification[/]"
        ))

    def _print_alerts(self, coupling: dict, AnomalyStatus, BreakType):
        """Print alert and diagnostic sections."""
        validation = coupling.get('_validation', {})
        breaks = validation.get('break_detections', {})
        anomaly_statuses = validation.get('anomaly_statuses', {})

        actionable = []
        diagnostic = []
        data_errors = []

        for pair, status in anomaly_statuses.items():
            if status.get('status') == AnomalyStatus.DATA_ERROR:
                data_errors.append((pair, status))
            elif status.get('is_actionable'):
                actionable.append((pair, status))
            elif status.get('status') in [AnomalyStatus.VALIDATED_BREAK, AnomalyStatus.ANOMALY_VETOED]:
                diagnostic.append((pair, status))

        # Actionable alerts
        if actionable:
            alert_text = Text()
            alert_text.append("🚨 VALIDATED PRECURSOR BREAK\n", style="bold red")
            alert_text.append("Reduced coupling during rising/active phase\n", style="yellow")
            alert_text.append("Recommend: Monitor for potential flare activity", style="dim")

            for pair, status in actionable:
                bd = breaks.get(pair, {})
                z_mad = bd.get('z_mad', status.get('z_mad', 0))
                alert_text.append(f"\n\n{pair}: ", style="bold")
                alert_text.append(f"{z_mad:.1f} MAD below median", style="red")

            self.console.print(Panel(alert_text, title="⚡ ALERT ENGINE", border_style="red", box=box.DOUBLE))

        # Diagnostic info
        if diagnostic:
            diag_text = Text()
            for pair, status in diagnostic:
                break_type = status.get('break_type', 'UNKNOWN')
                z_mad = status.get('z_mad', 0)
                veto = status.get('veto_reasons', [])

                if break_type == BreakType.POSTCURSOR:
                    diag_text.append(f"{pair}: ", style="bold")
                    diag_text.append(f"POSTCURSOR (late break - no alert)\n", style="dim")
                else:
                    diag_text.append(f"{pair}: ", style="bold")
                    diag_text.append(f"{z_mad:.1f}σ anomaly ", style="yellow")
                    diag_text.append(f"[VETOED: {', '.join(veto)}]\n", style="dim")

            self.console.print(Panel(diag_text, title="🔬 PHYSICS DIAGNOSTICS", border_style="dim", subtitle="[dim]Contextual, not for triggering[/]"))

        # Transfer state
        transfer = coupling.get('_transfer_state')
        if transfer:
            state = transfer.get('state', '')
            icon = "↓↑" if state == 'TRANSFER_STATE' else "↑↓"
            desc = transfer.get('description', '')
            conf = transfer.get('confidence', '')

            state_text = f"{icon} [bold]{state}[/] ({conf} confidence)\n{desc}"
            self.console.print(Panel(state_text, title="🔄 State Analysis", border_style="cyan"))

    def print_stereo_section(self, stereo: dict, coupling: dict = None):
        """Print STEREO-A section."""
        if not stereo:
            return

        meta = stereo.get('_stereo_metadata', {})
        ts = stereo.get('_timestamp', 'unknown')
        sep = meta.get('separation_deg', 51)
        days = meta.get('advance_warning_days', 3.9)

        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("Pair")
        table.add_column("ΔMI", justify="right")
        table.add_column("EUVI")

        for pair, data in stereo.items():
            if pair.startswith('_'):
                continue
            table.add_row(f"{pair} Å", f"{data['delta_mi']:.3f}", data.get('euvi_wavelengths', ''))

        panel = Panel(
            table,
            title=f"🛰️ STEREO-A EUVI ({sep:.0f}° ahead → ~{days:.1f} days warning)",
            border_style="yellow",
            subtitle=f"[dim]{ts}[/]"
        )
        self.console.print(panel)

    # NOAA Space Weather Message Codes
    # https://www.swpc.noaa.gov/products/space-weather-message-codes
    NOAA_CODES = {
        # Kp Warnings
        'WARK04': 'Kp=4 Geomag Warning',
        'WARK05': 'Kp=5 Geomag Warning',
        'WARK06': 'Kp=6 Geomag Warning',
        'WARK07': 'Kp=7 Geomag Warning',
        'WARK08': 'Kp=8 Geomag Warning',
        'WARK09': 'Kp=9 Geomag Warning',
        # Kp Alerts
        'ALTK04': 'Kp=4 Geomag Alert',
        'ALTK05': 'Kp=5 Geomag Alert',
        'ALTK06': 'Kp=6 Geomag Alert',
        'ALTK07': 'Kp=7 Geomag Alert',
        'ALTK08': 'Kp=8 Geomag Alert',
        'ALTK09': 'Kp=9 Geomag Alert',
        # Solar events
        'ALTTP2': 'Type II Radio Burst',
        'ALTTP4': 'Type IV Radio Burst',
        'ALTXMF': 'X-ray M-Flare',
        'ALTXFL': 'X-ray X-Flare',
        'ALTEF3': '10 MeV Proton Event',
        'ALTPX1': '100 MeV Proton Event',
        # Sudden impulse
        'WARSUD': 'Sudden Impulse Warning',
        'SUMSUD': 'Sudden Impulse Summary',
        # Polar cap
        'WARPC0': 'Polar Cap Absorption',
        # Watches
        'WATA20': 'Geomag A≥20 Watch',
        'WATA30': 'Geomag A≥30 Watch',
        'WATA50': 'Geomag A≥50 Watch',
        # Summaries
        'SUMX': 'X-ray Flare Summary',
        'SUMPX1': 'Proton Event Summary',
    }

    def _decode_noaa_code(self, code: str) -> str:
        """Decode NOAA alert code to English description."""
        if code in self.NOAA_CODES:
            return self.NOAA_CODES[code]
        # Pattern matching for unknown codes
        if code.startswith('WARK'):
            return f"Kp={code[4:]} Warning"
        if code.startswith('ALTK'):
            return f"Kp={code[4:]} Alert"
        if code.startswith('WAR'):
            return 'Warning'
        if code.startswith('ALT'):
            return 'Alert'
        if code.startswith('SUM'):
            return 'Summary'
        return code

    def _extract_noaa_code(self, message: str) -> str | None:
        """Extract full NOAA code (e.g. WARK04) from message text."""
        import re
        # Pattern: Space Weather Message Code: XXXXX
        match = re.search(r'Code:\s*(\w+)', message)
        if match:
            return match.group(1)
        # Alternative: look for known prefixes at word boundaries
        prefixes = ['WAR', 'ALT', 'SUM', 'WAT']
        for prefix in prefixes:
            match = re.search(rf'\b({prefix}\w{{2,5}})\b', message)
            if match:
                return match.group(1)
        return None

    def print_alerts_section(self, alerts: list):
        """Print NOAA alerts with decoded message codes."""
        if not alerts:
            return

        table = Table(box=None, show_header=False)
        table.add_column("Code", style="bold yellow", width=10)
        table.add_column("Meaning", style="cyan", width=22)
        table.add_column("Issued", style="dim", width=18)

        for alert in alerts[:3]:
            product_id = alert['type']
            message = alert.get('message', '')
            issued = alert.get('issued', '')[:16]  # Date + time only

            # Extract full NOAA code from message (e.g. WARK04)
            full_code = self._extract_noaa_code(message) or product_id
            meaning = self._decode_noaa_code(full_code)

            table.add_row(full_code, meaning, issued)

        self.console.print(Panel(table, title="📢 NOAA ALERTS (last 24h)", border_style="yellow"))

    def generate_event_narrative(self, xray: dict, coupling: dict) -> str | None:
        """Generate event narrative (returns string for compatibility)."""
        if not xray or not coupling:
            return None

        flux = xray.get('flux', 0)
        flare_class = xray.get('flare_class', 'A0')

        mi_211 = coupling.get('193-211', {})
        mi_304 = coupling.get('193-304', {})

        r_211 = mi_211.get('residual', 0)
        r_304 = mi_304.get('residual', 0)
        slope_211 = mi_211.get('slope_pct_per_hour', 0)
        slope_304 = mi_304.get('slope_pct_per_hour', 0)

        # Determine if narrative is needed
        if flux >= 1e-5:
            phase = "M/X-CLASS ACTIVE"
        elif flux >= 1e-6:
            phase = "C-CLASS ACTIVE"
        elif flux >= 5e-7:
            phase = "DECAY PHASE" if slope_211 < -2 or slope_304 > 2 else "ELEVATED"
        elif r_304 > 3 and slope_304 > 0:
            phase = "POST-FLARE DECAY"
        elif abs(r_211) > 2 or abs(r_304) > 2:
            phase = "ANOMALOUS"
        else:
            return None

        return phase  # Return phase name; caller will print panel

    def print_event_narrative(self, xray: dict, coupling: dict):
        """Print event narrative panel."""
        phase = self.generate_event_narrative(xray, coupling)
        if not phase:
            return

        mi_211 = coupling.get('193-211', {})
        mi_304 = coupling.get('193-304', {})

        delta_211 = mi_211.get('delta_mi', 0)
        delta_304 = mi_304.get('delta_mi', 0)
        r_211 = mi_211.get('residual', 0)
        r_304 = mi_304.get('residual', 0)
        slope_211 = mi_211.get('slope_pct_per_hour', 0)
        slope_304 = mi_304.get('slope_pct_per_hour', 0)

        # Build table
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("Channel")
        table.add_column("ΔMI", justify="right")
        table.add_column("r(σ)", justify="right")
        table.add_column("Trend", justify="right")

        arrow_211 = "↓" if slope_211 < -1 else "↑" if slope_211 > 1 else "→"
        arrow_304 = "↓" if slope_304 < -1 else "↑" if slope_304 > 1 else "→"

        table.add_row("193-211 Å", f"{delta_211:.3f}", f"{r_211:+.1f}σ", f"{arrow_211} {slope_211:+.1f}%/h")
        table.add_row("193-304 Å", f"{delta_304:.3f}", f"{r_304:+.1f}σ", f"{arrow_304} {slope_304:+.1f}%/h")

        self.console.print(Panel(table, title=f"📖 EVENT NARRATIVE: {phase}", border_style="bright_blue"))

    def print_footer(self, db_path: str = None):
        """Print footer with database info."""
        if db_path:
            self.console.print(f"[dim]  💾 Data stored in: {db_path}[/]")
        self.console.print()

    # =========================================================================
    # MINIMAL ALERT MODE
    # =========================================================================

    def print_minimal_alert(self, coupling: dict, xray: dict = None, next_check_min: int = 10):
        """
        Print minimal early warning display.

        Only shows what's actionable:
        - 193-211 ΔMI (value + deviation from baseline)
        - 193-211 Trend
        - GOES status (if relevant)
        - Clear status indicator

        This is for operators/decision-makers, not scientists.
        """
        # Extract 193-211 data
        pair_data = coupling.get('193-211', {}) if coupling else {}
        delta_mi = pair_data.get('delta_mi', 0)
        residual = pair_data.get('residual', 0)
        slope = pair_data.get('slope_pct_per_hour', 0)
        trend = pair_data.get('trend', 'NO_DATA')

        # Baseline reference (from paper: 0.59 ± 0.12)
        baseline = 0.59
        baseline_std = 0.12
        deviation_pct = ((delta_mi - baseline) / baseline * 100) if baseline else 0

        # Determine status
        is_break = pair_data.get('is_break', False)
        is_accelerating_down = slope < -5 and trend in ['DECLINING', 'ACCELERATING_DOWN']

        # GOES context
        goes_flux = xray.get('flux', 0) if xray else 0
        goes_class = xray.get('flare_class', '') if xray else ''
        goes_rising = xray.get('rising', False) if xray else False

        # Status classification
        if is_break or (residual < -2 and is_accelerating_down):
            status = "BREAK DETECTED"
            status_icon = "🔴"
            status_style = "bold red"
            border_style = "red"
        elif residual < -1.5 or (slope < -3 and goes_rising):
            status = "CAUTION"
            status_icon = "🟡"
            status_style = "bold yellow"
            border_style = "yellow"
        else:
            status = "CLEAR"
            status_icon = "🟢"
            status_style = "bold green"
            border_style = "green"

        # Build content
        content = Text()

        # Status line
        content.append("STATUS:  ", style="dim")
        content.append(f"{status_icon} {status}\n\n", style=status_style)

        # 193-211 line - focus on whether it's concerning (below baseline)
        if deviation_pct >= -15:
            mi_style = "green"
            mi_note = "nominal" if deviation_pct >= 0 else f"{deviation_pct:.0f}%"
        elif deviation_pct >= -30:
            mi_style = "yellow"
            mi_note = f"{deviation_pct:.0f}% below baseline"
        else:
            mi_style = "red"
            mi_note = f"{deviation_pct:.0f}% below baseline"

        content.append("193-211:  ", style="dim")
        content.append(f"{delta_mi:.2f} bits", style=mi_style)
        content.append(f"  ({mi_note})\n", style="dim")

        # Trend line
        trend_icons = {
            'ACCELERATING_DOWN': '↓↓ accelerating down',
            'DECLINING': '↓ declining',
            'STABLE': '→ stable',
            'RISING': '↑ rising',
            'ACCELERATING_UP': '↑↑ accelerating up',
        }
        trend_text = trend_icons.get(trend, f'→ {slope:+.1f}%/h')
        trend_style = "red" if 'DOWN' in trend or slope < -3 else "green" if trend == 'STABLE' else "yellow"

        content.append("Trend:    ", style="dim")
        content.append(f"{trend_text}\n", style=trend_style)

        # GOES line (only if relevant)
        if status != "CLEAR" or goes_flux > 1e-6:
            content.append("\n")
            content.append("GOES:     ", style="dim")
            goes_style = "red" if goes_rising else "green"
            rising_text = " (rising)" if goes_rising else ""
            content.append(f"{goes_class}{rising_text}\n", style=goes_style)

        # Action line (only for non-CLEAR)
        if status == "BREAK DETECTED":
            content.append("\n")
            content.append("→ Monitor for flare in 0.5-2h", style="bold yellow")
        elif status == "CAUTION":
            content.append("\n")
            content.append("→ Watch for further decline", style="yellow")

        # Next check
        content.append(f"\n\nNext check: {next_check_min} min", style="dim")

        # Print panel
        self.console.print(Panel(
            content,
            title="[bold]☀️ SOLAR EARLY WARNING[/]",
            border_style=border_style,
            box=box.ROUNDED,
            padding=(0, 2),
        ))

    def print_personal_relevance(self, location: str = 'berlin', kp_index: float = None):
        """
        Print personal relevance panel showing user's exposure to solar events.

        Args:
            location: Location name (from LOCATIONS dict) or 'lat,lon' string
            kp_index: Current Kp index for aurora prediction
        """
        # Parse location
        if ',' in location:
            try:
                lat, lon = map(float, location.split(','))
                loc_name = f"{lat:.1f}°N, {lon:.1f}°E"
                tz_name = "UTC"
            except ValueError:
                lat, lon = 52.52, 13.405  # Default to Berlin
                loc_name = "Berlin"
                tz_name = "Europe/Berlin"
        elif location.lower() in LOCATIONS:
            lat, lon, tz_name = LOCATIONS[location.lower()]
            loc_name = location.title()
        else:
            lat, lon, tz_name = 52.52, 13.405, "Europe/Berlin"
            loc_name = "Berlin"

        rel = assess_personal_relevance(lat, lon, loc_name, tz_name, kp_index)

        content = Text()

        # Location line
        content.append("Location:     ", style="dim")
        content.append(f"{rel.location_name} ({rel.latitude:.1f}°, {rel.longitude:.1f}°)\n")

        # Local time
        content.append("Local Time:   ", style="dim")
        content.append(f"{rel.local_time}\n")

        # Sun status
        content.append("Sun Status:   ", style="dim")
        sun = rel.sun_status
        if sun.is_visible:
            content.append(f"☀️ VISIBLE (alt: {sun.altitude_deg:.0f}°)\n", style="bold yellow")
        elif sun.status == 'CIVIL_TWILIGHT':
            content.append(f"🌅 TWILIGHT (alt: {sun.altitude_deg:.0f}°)\n", style="yellow")
        else:
            content.append(f"🌙 BELOW HORIZON (alt: {sun.altitude_deg:.0f}°)\n", style="cyan")

        content.append("\n")

        # Current flare risk
        content.append("Current Flare Risk:\n", style="bold")

        if sun.is_visible:
            content.append("→ Radio/GPS disruption would affect you ", style="dim")
            content.append("NOW\n", style="bold red")
        else:
            content.append("→ Immediate radio effects: ", style="dim")
            content.append("NOT relevant for you\n", style="green")
            content.append("→ Geomagnetic storm: ", style="dim")
            content.append("May affect you in 15-48h\n", style="yellow")
            if rel.aurora_possible:
                content.append("→ Aurora possible tonight ", style="dim")
                kp_note = f"if Kp ≥ 7" if kp_index is None else f"(Kp={kp_index:.0f})"
                content.append(f"{kp_note}\n", style="cyan")

        content.append("\n")

        # Daylight window
        sunrise, sunset = rel.daylight_window_utc
        content.append(f"Your daylight window: ", style="dim")
        content.append(f"{sunrise}-{sunset} UTC\n", style="bold")

        # Border color based on exposure
        border_style = "yellow" if sun.is_visible else "cyan"

        self.console.print(Panel(
            content,
            title="👤 PERSONAL RELEVANCE",
            border_style=border_style,
            box=box.ROUNDED,
        ))
