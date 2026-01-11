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


console = Console()


class StatusFormatter:
    """
    Format status reports using Rich for beautiful terminal output.
    """

    # Status styles
    STYLES = {
        'NORMAL': Style(color="green"),
        'ELEVATED': Style(color="yellow"),
        'WARNING': Style(color="bright_yellow", bold=True),
        'ALERT': Style(color="red", bold=True),
        'DATA_ERROR': Style(color="bright_black"),
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

    def print_coupling_analysis(self, coupling: dict, AnomalyStatus, BreakType):
        """Print coupling analysis with Rich formatting."""
        if not coupling:
            return

        # Quality info
        quality = coupling.get('_quality', {})
        n_warn = quality.get('n_warnings', 0)
        quality_text = "[green]✓ GOOD[/]" if n_warn == 0 else f"[yellow]⚠ {n_warn} warning(s)[/]"

        # Build coupling table
        table = Table(
            title=f"Data Quality: {quality_text}",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Pair", style="bold")
        table.add_column("ΔMI", justify="right")
        table.add_column("r(σ)", justify="right")
        table.add_column("Status")
        table.add_column("Trend")

        for pair, data in coupling.items():
            if pair.startswith('_'):
                continue

            delta_mi = data.get('delta_mi', 0)
            residual = data.get('residual', 0)
            status = data.get('status', 'NORMAL')
            trend = data.get('trend', 'NO_DATA')
            slope = data.get('slope_pct_per_hour', 0)

            # Style based on status
            status_style = self.STYLES.get(status, Style())
            trend_icon = self.TREND_ICONS.get(trend, '?')

            # Residual coloring
            if abs(residual) > 3:
                r_style = "red"
            elif abs(residual) > 2:
                r_style = "yellow"
            else:
                r_style = "green"

            # Build trend string
            trend_str = f"{trend_icon} {slope:+.1f}%/h"

            # Break indicator
            if data.get('is_break'):
                status_text = f"[bold red]🚨 BREAK[/]"
            elif data.get('break_vetoed'):
                status_text = f"[dim]VETOED[/]"
            else:
                status_text = Text(status, style=status_style)

            table.add_row(
                f"{pair} Å",
                f"{delta_mi:.3f}",
                f"[{r_style}]{residual:+.1f}σ[/]",
                status_text,
                trend_str,
            )

        panel = Panel(table, title="📊 ΔMI COUPLING MONITOR", border_style="magenta", subtitle="[dim]Pre-Flare Detection[/]")
        self.console.print(panel)

        # Show alerts if any
        self._print_alerts(coupling, AnomalyStatus, BreakType)

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

    def print_alerts_section(self, alerts: list):
        """Print NOAA alerts."""
        if not alerts:
            return

        table = Table(box=None, show_header=False)
        table.add_column("Type", style="bold yellow")
        table.add_column("Message")

        for alert in alerts[:3]:
            table.add_row(f"[{alert['type']}]", alert['message'][:60] + "...")

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
