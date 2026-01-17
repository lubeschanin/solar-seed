#!/usr/bin/env python3
"""
Solar Seed - Interactive CLI
============================

User-friendly interface for all analyses.
Knowledge accessible to everyone.

Start with:
    uv run python -m solar_seed.cli
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta


def clear_screen():
    """Clear the screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Show the header."""
    print("""
  ╭─────────────────────────────────────────────╮
  │  🌞 SOLAR SEED 🌱                           │
  │  Mutual Information Analysis of the Sun     │
  ╰─────────────────────────────────────────────╯
""")


def print_menu():
    """Show the main menu."""
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                         MAIN MENU                                   │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │   [1]  Quick Test (synthetic data, ~2 min)                          │
  │                                                                     │
  │   [2]  Multi-Channel Analysis (21 wavelength pairs)                 │
  │                                                                     │
  │   [3]  Rotation Analysis (segment-based, scalable)                  │
  │                                                                     │
  │   [4]  Flare Analysis (X9.0 Event)                                  │
  │                                                                     │
  │   [5]  Render Sun Images (download + visualize)                     │
  │                                                                     │
  │   [6]  Early Warning System (real-time monitoring)                  │
  │                                                                     │
  │   [7]  Reports (daily/weekly summary, precursor stats)              │
  │                                                                     │
  │   [8]  Status: Check running analysis                               │
  │                                                                     │
  │   [9]  View Results                                                 │
  │                                                                     │
  │   [q]  Quit                                                         │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
""")


def get_choice(prompt: str, options: list) -> str:
    """Get a choice from the user."""
    while True:
        choice = input(f"\n  {prompt} ").strip().lower()
        if choice in options:
            return choice
        print(f"  ⚠ Please choose one of: {', '.join(options)}")


def get_number(prompt: str, default: float, min_val: float = 0, max_val: float = 10000) -> float:
    """Get a number from the user."""
    while True:
        value = input(f"  {prompt} [{default}]: ").strip()
        if not value:
            return default
        try:
            num = float(value)
            if min_val <= num <= max_val:
                return num
            print(f"  ⚠ Please enter a number between {min_val} and {max_val}")
        except ValueError:
            print("  ⚠ Please enter a valid number")


def get_date(prompt: str, default_days_ago: int = 27) -> str:
    """Get a date from the user."""
    default = (datetime.now() - timedelta(days=default_days_ago)).strftime("%Y-%m-%d")
    while True:
        value = input(f"  {prompt} [{default}]: ").strip()
        if not value:
            return f"{default}T00:00:00"
        try:
            datetime.strptime(value, "%Y-%m-%d")
            return f"{value}T00:00:00"
        except ValueError:
            print("  ⚠ Please use format YYYY-MM-DD (e.g. 2024-01-15)")


def show_status_bar(current: int, total: int, width: int = 40) -> str:
    """Create a progress bar."""
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total} ({pct*100:.1f}%)"


def check_checkpoint() -> dict:
    """Check if a checkpoint exists."""
    import json
    checkpoint_path = Path("results/rotation/checkpoint.json")
    result_path = Path("results/rotation/rotation_analysis.json")

    info = {"exists": False, "processed": 0, "start_date": None, "hours": None, "cadence": None}

    # Get metadata from results file
    if result_path.exists():
        try:
            with open(result_path) as f:
                data = json.load(f)
            info["start_date"] = data.get("metadata", {}).get("start_time", "")[:10]
            info["hours"] = data.get("metadata", {}).get("hours")
            info["cadence"] = data.get("metadata", {}).get("cadence_minutes")
        except:
            pass

    # Get progress from checkpoint
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            data = json.load(f)
        info["exists"] = True
        info["processed"] = data.get("last_index", 0)
        info["timestamps"] = len(data.get("timestamps", []))

    return info


def check_segments() -> dict:
    """Check existing segments."""
    import json
    segment_dir = Path("results/rotation/segments")

    info = {
        "exists": False,
        "count": 0,
        "dates": [],
        "total_points": 0,
        "first_date": None,
        "last_date": None
    }

    if not segment_dir.exists():
        return info

    segment_files = sorted(segment_dir.glob("*.json"))
    if not segment_files:
        return info

    info["exists"] = True
    info["count"] = len(segment_files)
    # Strip .partial suffix from dates (e.g., "2025-12-16.partial" -> "2025-12-16")
    info["dates"] = [f.stem.replace(".partial", "") for f in segment_files]
    info["first_date"] = info["dates"][0] if info["dates"] else None
    info["last_date"] = info["dates"][-1] if info["dates"] else None

    # Count total points
    for sf in segment_files:
        try:
            with open(sf) as f:
                data = json.load(f)
            info["total_points"] += data.get("n_points", 0)
        except:
            pass

    return info


def run_quicktest():
    """Quick test with synthetic data."""
    clear_screen()
    print_header()
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                      QUICK TEST                                     │
  └─────────────────────────────────────────────────────────────────────┘

  This test uses synthetic solar data to demonstrate the analysis
  pipeline. No internet connection required.

  Duration: ~2 minutes
""")

    if get_choice("Start? [y/n]:", ["y", "n"]) == "y":
        print("\n  🚀 Starting analysis...\n")
        from solar_seed.hypothesis_test import main as run_hypothesis
        sys.argv = ["hypothesis_test", "--spatial", "--controls"]
        run_hypothesis()
    else:
        print("\n  Cancelled.")


def run_multichannel():
    """Multi-Channel Analysis."""
    clear_screen()
    print_header()
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                 MULTI-CHANNEL ANALYSIS                              │
  └─────────────────────────────────────────────────────────────────────┘

  Analyzes all 21 wavelength pairs of the 7 AIA channels:
  304, 171, 193, 211, 335, 94, 131 Å

  Computes ΔMI_sector (local structure coupling) for each pair.
""")

    print("\n  Data source:")
    print("    [1] Synthetic (fast, offline)")
    print("    [2] Real AIA data (slow, download required)")

    source = get_choice("Choose [1/2]:", ["1", "2"])
    use_real = source == "2"

    hours = get_number("Analysis period in hours:", 24 if use_real else 6, 1, 720)

    print(f"""
  ────────────────────────────────────────────────────────────────────────

  Configuration:
    Data source:  {"Real AIA data" if use_real else "Synthetic"}
    Period:       {hours} hours
    Pairs:        21

""")

    if get_choice("Start? [y/n]:", ["y", "n"]) == "y":
        print("\n  🚀 Starting analysis...\n")
        from solar_seed.multichannel import main as run_multi
        args = ["multichannel", "--hours", str(int(hours))]
        if use_real:
            args.append("--real")
        sys.argv = args
        run_multi()
    else:
        print("\n  Cancelled.")


def run_rotation():
    """Segment-based Rotation Analysis."""
    clear_screen()
    print_header()
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │            SEGMENT-BASED ROTATION ANALYSIS                          │
  └─────────────────────────────────────────────────────────────────────┘

  Analyzes solar rotation data using a scalable segment-based approach.
  Each day is analyzed independently and can be extended later.

  Benefits:
    • Scalable: 16, 27, or 100+ days
    • Fault-tolerant: only lose one day on failure
    • Extensible: add more days without re-processing
""")

    # Check for existing segments
    segments = check_segments()
    checkpoint = check_checkpoint()

    if segments["exists"]:
        print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  ✓ SEGMENTS FOUND                                                   │
  │    Days analyzed:  {segments['count']:<47}│
  │    Period:         {segments['first_date']} → {segments['last_date']:<28}│
  │    Total points:   {segments['total_points']:<47}│
  └─────────────────────────────────────────────────────────────────────┘
""")
        print("  What would you like to do?")
        print("    [1] Extend analysis (add more days)")
        print("    [2] Aggregate existing segments")
        print("    [3] Start fresh analysis")
        print("    [4] Cancel")

        action = get_choice("Choose [1/2/3/4]:", ["1", "2", "3", "4"])

        if action == "4":
            print("\n  Cancelled.")
            return

        if action == "2":
            print("\n  🔄 Aggregating segments...\n")
            from solar_seed.final_analysis import aggregate_segments
            aggregate_segments(verbose=True)
            return

        if action == "1":
            # Extend: suggest next day after last segment
            from datetime import datetime
            last = datetime.strptime(segments["last_date"], "%Y-%m-%d")
            default_start = (last + timedelta(days=1)).strftime("%Y-%m-%d")
            default_end = (last + timedelta(days=8)).strftime("%Y-%m-%d")

            print(f"\n  Extend from {default_start}:")
        else:
            # Fresh start
            default_start = (datetime.now() - timedelta(days=27)).strftime("%Y-%m-%d")
            default_end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            print("\n  ── New Analysis Configuration ──\n")

    elif checkpoint["exists"]:
        # Legacy checkpoint exists - offer conversion
        print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  ⚠ LEGACY CHECKPOINT FOUND                                          │
  │    Processed: {checkpoint['processed']} timepoints{' ' * (48 - len(str(checkpoint['processed'])))}│
  └─────────────────────────────────────────────────────────────────────┘
""")
        print("  What would you like to do?")
        print("    [1] Convert to segments (recommended)")
        print("    [2] Continue with legacy mode")
        print("    [3] Start fresh with segments")
        print("    [4] Cancel")

        action = get_choice("Choose [1/2/3/4]:", ["1", "2", "3", "4"])

        if action == "4":
            print("\n  Cancelled.")
            return

        if action == "1":
            print("\n  🔄 Converting checkpoint to segments...\n")
            from solar_seed.final_analysis import convert_checkpoint_to_segments
            convert_checkpoint_to_segments(verbose=True)
            print("\n  ✓ Conversion complete! Run again to extend or aggregate.")
            return

        if action == "2":
            # Legacy mode
            run_rotation_legacy()
            return

        # Fresh start
        default_start = (datetime.now() - timedelta(days=27)).strftime("%Y-%m-%d")
        default_end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        print("\n  ── New Analysis Configuration ──\n")

    else:
        # No data - new analysis
        default_start = (datetime.now() - timedelta(days=27)).strftime("%Y-%m-%d")
        default_end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        print("\n  No existing data. Starting new segment-based analysis.")
        print("\n  ── Configuration ──\n")

    # Get date range
    print(f"  Enter date range (YYYY-MM-DD format):\n")

    while True:
        start_input = input(f"    Start date [{default_start}]: ").strip()
        start_date = start_input if start_input else default_start
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            break
        except ValueError:
            print("    ⚠ Please use format YYYY-MM-DD")

    while True:
        end_input = input(f"    End date [{default_end}]: ").strip()
        end_date = end_input if end_input else default_end
        try:
            datetime.strptime(end_date, "%Y-%m-%d")
            break
        except ValueError:
            print("    ⚠ Please use format YYYY-MM-DD")

    # Calculate days
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    n_days = (end_dt - start_dt).days + 1
    n_points = n_days * 120  # 120 points per day at 12-min cadence

    print(f"""
  ────────────────────────────────────────────────────────────────────────

  Summary:
    Period:       {start_date} → {end_date}
    Days:         {n_days}
    Cadence:      12 minutes
    Points/day:   120
    Total:        ~{n_points} datapoints

  Existing segments will be skipped automatically.

""")

    print("  Auto-push segments to git?")
    print("    [y] Yes - push after each day")
    print("    [n] No  - local only")
    auto_push = get_choice("Choose [y/n]:", ["y", "n"]) == "y"

    if get_choice("\n  Start? [y/n]:", ["y", "n"]) == "y":
        print("\n  🚀 Starting segment-based analysis...\n")
        print("  Tip: Press Ctrl+C to pause at any time.")
        print("       Already completed segments are preserved.\n")

        from solar_seed.final_analysis import run_segmented_rotation
        run_segmented_rotation(
            start_date=start_date,
            end_date=end_date,
            cadence_minutes=12,
            verbose=True,
            auto_push=auto_push
        )
    else:
        print("\n  Cancelled.")


def run_rotation_legacy():
    """Legacy 27-day Rotation Analysis (monolithic)."""
    clear_screen()
    print_header()
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │          LEGACY ROTATION ANALYSIS (monolithic)                      │
  └─────────────────────────────────────────────────────────────────────┘

  ⚠ This is the old monolithic approach. Consider using segment-based
    analysis for better scalability and fault tolerance.
""")

    checkpoint = check_checkpoint()

    if checkpoint["exists"]:
        print(f"\n  Checkpoint found: {checkpoint['processed']} timepoints processed")
        print("    [1] Resume")
        print("    [2] Start fresh")
        print("    [3] Cancel")

        action = get_choice("Choose [1/2/3]:", ["1", "2", "3"])
        if action == "3":
            return
        resume = action == "1"
    else:
        resume = False

    if not resume:
        print("\n  ── Configuration ──\n")
        start_date = get_date("Start date:", 27)
        hours = get_number("Duration in hours:", 648, 24, 1000)
        cadence = get_number("Cadence in minutes:", 60, 12, 360)
    else:
        start_date = f"{checkpoint.get('start_date', '')}T00:00:00"
        hours = checkpoint.get("hours") or 648
        cadence = checkpoint.get("cadence") or 60

    if get_choice("\n  Start? [y/n]:", ["y", "n"]) == "y":
        from solar_seed.final_analysis import run_rotation_analysis
        run_rotation_analysis(
            hours=hours,
            cadence_minutes=int(cadence),
            start_time_str=start_date,
            resume=resume,
            verbose=True
        )


def run_flare():
    """Flare Analysis."""
    clear_screen()
    print_header()
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    FLARE ANALYSIS                                   │
  └─────────────────────────────────────────────────────────────────────┘

  Analyzes the X9.0 flare event from 2024-10-03.
  Compares coupling across different phases:

    • Pre-Flare  (quiet sun)
    • Peak       (maximum emission)
    • Decay      (cooling phase)
""")

    if get_choice("Start? [y/n]:", ["y", "n"]) == "y":
        print("\n  🚀 Starting flare analysis...\n")
        from solar_seed.flare_analysis import main as run_flare_main
        sys.argv = ["flare_analysis"]
        run_flare_main()
    else:
        print("\n  Cancelled.")


def get_time(prompt: str, default: str = "12:00") -> str:
    """Get a time from the user."""
    while True:
        value = input(f"  {prompt} [{default}]: ").strip()
        if not value:
            return default
        try:
            datetime.strptime(value, "%H:%M")
            return value
        except ValueError:
            print("  ⚠ Please use format HH:MM (e.g. 14:30)")


def get_timezone() -> str:
    """Get timezone from user with common options."""
    from solar_seed.render_sun import COMMON_TIMEZONES

    print("\n  Select location/timezone:")
    print()
    cities = list(COMMON_TIMEZONES.keys())
    for i, city in enumerate(cities, 1):
        print(f"    [{i:2}] {city}")
    print(f"    [o]  Other (enter manually)")

    options = [str(i) for i in range(1, len(cities) + 1)] + ["o"]
    choice = get_choice("Choose:", options)

    if choice == "o":
        while True:
            tz = input("  Enter timezone (e.g. Europe/Berlin): ").strip()
            try:
                from zoneinfo import ZoneInfo
                ZoneInfo(tz)  # Validate
                return tz
            except:
                print("  ⚠ Invalid timezone. Examples: Europe/Berlin, America/New_York")
    else:
        city = cities[int(choice) - 1]
        return COMMON_TIMEZONES[city]


def run_render():
    """Render Sun Images."""
    clear_screen()
    print_header()
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                  RENDER SUN IMAGES                                  │
  └─────────────────────────────────────────────────────────────────────┘

  Downloads real AIA data and creates beautiful sun images
  for all 7 wavelength channels + RGB composite.

  The image will show both local time and UTC time.
""")

    print("\n  What would you like to render?")
    print("    [1] Recent sun (24 hours ago)")
    print("    [2] Specific date/time with timezone")
    print("    [3] Cancel")

    action = get_choice("Choose [1/2/3]:", ["1", "2", "3"])

    if action == "3":
        print("\n  Cancelled.")
        return

    local_dt = None
    timezone = None

    if action == "1":
        utc_timestamp = (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%dT12:00:00")
        display_date = utc_timestamp[:10]
    else:
        print("\n  ── Enter date and time ──\n")

        # Get date in DD.MM.YYYY format
        while True:
            date_input = input("  Date (DD.MM.YYYY, e.g. 08.03.2012): ").strip()
            if not date_input:
                print("  ⚠ Please enter a date")
                continue
            try:
                # Validate format
                if "." in date_input:
                    parts = date_input.split(".")
                    if len(parts) == 3:
                        day, month, year = parts
                        datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
                        break
                print("  ⚠ Please use format DD.MM.YYYY (e.g. 08.03.2012)")
            except ValueError:
                print("  ⚠ Invalid date. Please use format DD.MM.YYYY")

        time_input = get_time("Time (HH:MM)", "12:00")
        timezone = get_timezone()

        # Convert to UTC
        from solar_seed.render_sun import parse_local_datetime
        local_dt, utc_dt = parse_local_datetime(date_input, time_input, timezone)
        utc_timestamp = utc_dt.strftime("%Y-%m-%dT%H:%M:%S")

        tz_city = timezone.split("/")[-1].replace("_", " ")
        print(f"\n  ✓ Local:  {local_dt.strftime('%d.%m.%Y %H:%M')} {tz_city}")
        print(f"  ✓ UTC:    {utc_dt.strftime('%d.%m.%Y %H:%M')} UTC")

        display_date = local_dt.strftime("%d.%m.%Y")

    print("\n  Render options:")
    print("    [1] All channels + composite (recommended)")
    print("    [2] Composite only")
    print("    [3] Grid view (all in one image)")

    option = get_choice("Choose [1/2/3]:", ["1", "2", "3"])

    render_individual = option == "1"
    render_grid = option == "3"

    print(f"""
  ────────────────────────────────────────────────────────────────────────

  Configuration:
    Date:       {display_date}
    Timezone:   {timezone or "UTC"}
    Channels:   {"All 7" if render_individual else "Composite only" if not render_grid else "Grid view"}
    Output:     images/

""")

    if get_choice("Start? [y/n]:", ["y", "n"]) == "y":
        print("\n  🚀 Downloading and rendering...\n")
        from solar_seed.render_sun import load_and_render
        load_and_render(
            timestamp=utc_timestamp,
            output_dir="images",
            render_individual=render_individual,
            render_comp=True,
            render_all_grid=render_grid,
            local_datetime=local_dt,
            timezone=timezone,
        )
    else:
        print("\n  Cancelled.")


def run_early_warning():
    """Early Warning System."""
    clear_screen()
    print_header()
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │               SOLAR EARLY WARNING SYSTEM                            │
  └─────────────────────────────────────────────────────────────────────┘

  Multi-layer early warning architecture:

    STEREO-A (51° ahead)        → 2-4 days warning
           ↓
    ΔMI Coupling Monitor        → Hours before flare
           ↓
    GOES X-ray + DSCOVR         → Minutes to real-time

  Data Sources:
    • GOES X-ray flux (flare detection)
    • DSCOVR solar wind (geomagnetic storms)
    • SDO/AIA coupling analysis (pre-flare detection)
    • NOAA Space Weather Alerts
""")

    print("\n  Select mode:")
    print("    [1] Quick Status Check (no coupling)")
    print("    [2] Full Status with Coupling Analysis (~3 min)")
    print("    [3] Continuous Monitoring (60s interval)")
    print("    [4] Continuous with Coupling (10 min interval)")
    print("    [5] Cancel")

    choice = get_choice("Choose [1-5]:", ["1", "2", "3", "4", "5"])

    if choice == "5":
        print("\n  Cancelled.")
        return

    # Import early warning module
    scripts_path = Path(__file__).parent.parent.parent / "scripts"
    sys.path.insert(0, str(scripts_path))

    try:
        from early_warning import (
            get_goes_xray,
            get_dscovr_solar_wind,
            get_noaa_alerts,
            run_coupling_analysis,
            print_status_report,
            monitor_loop
        )
    except ImportError as e:
        print(f"\n  Error importing early warning module: {e}")
        print("  Make sure scripts/early_warning.py exists.")
        return

    if choice == "1":
        # Quick status
        print("\n  Fetching real-time data...\n")
        xray = get_goes_xray()
        solar_wind = get_dscovr_solar_wind()
        alerts = get_noaa_alerts()
        print_status_report(xray, solar_wind, alerts)

    elif choice == "2":
        # Full status with coupling
        print("\n  Fetching real-time data + coupling analysis...\n")
        xray = get_goes_xray()
        solar_wind = get_dscovr_solar_wind()
        alerts = get_noaa_alerts()
        coupling = run_coupling_analysis()
        print_status_report(xray, solar_wind, alerts, coupling)

    elif choice == "3":
        # Continuous monitoring (no coupling)
        print("\n  Starting continuous monitoring (Ctrl+C to stop)...\n")
        try:
            monitor_loop(interval=60, with_coupling=False)
        except KeyboardInterrupt:
            print("\n  Monitoring stopped.")

    elif choice == "4":
        # Continuous with coupling
        print("\n  Starting full monitoring with coupling (Ctrl+C to stop)...\n")
        try:
            monitor_loop(interval=600, with_coupling=True)
        except KeyboardInterrupt:
            print("\n  Monitoring stopped.")


def run_reports():
    """Generate and view reports."""
    clear_screen()
    print_header()
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                        REPORTS                                      │
  └─────────────────────────────────────────────────────────────────────┘

  Generate summary reports from the monitoring database.
  Includes precursor detection statistics (Precision/Recall).
""")

    # Import report module
    scripts_path = Path(__file__).parent.parent.parent / "scripts"
    sys.path.insert(0, str(scripts_path))

    try:
        from report import ReportGenerator
    except ImportError as e:
        print(f"\n  Error importing report module: {e}")
        print("  Make sure scripts/report.py exists.")
        return

    print("  Select report type:")
    print("    [1] Daily Summary (last 24h)")
    print("    [2] Weekly Summary (last 7 days)")
    print("    [3] Monthly Summary (last 30 days)")
    print("    [4] Precursor Statistics Only")
    print("    [5] Export Report (md/html/json)")
    print("    [6] Cancel")

    choice = get_choice("Choose [1-6]:", ["1", "2", "3", "4", "5", "6"])

    if choice == "6":
        print("\n  Cancelled.")
        return

    report = ReportGenerator()

    if choice == "4":
        # Precursor statistics only
        stats = report.get_precursor_statistics()
        print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │              PRECURSOR DETECTION STATISTICS                         │
  └─────────────────────────────────────────────────────────────────────┘
""")
        print(f"  Detection Window: {stats.get('window_min_hours', 0.5):.1f} - {stats.get('window_max_hours', 6.0):.1f} hours after break")
        print()
        print(f"  Actionable Alerts:      {stats.get('actionable_alerts', 0)}")
        print(f"  Break Candidates:       {stats.get('break_candidates', 0)}")
        print(f"  Total Flares (C+):      {stats.get('total_flares', 0)}")
        print()
        print(f"  True Positives (TP):    {stats.get('true_positives', 0)}")
        print(f"  False Positives (FP):   {stats.get('false_positives', 0)}")
        print(f"  False Negatives (FN):   {stats.get('false_negatives', 0)}")
        print()

        if stats['precision'] is not None:
            print(f"  Precision:  {stats['precision']:.1%}")
        else:
            print(f"  Precision:  N/A (no breaks detected)")

        if stats['recall'] is not None:
            print(f"  Recall:     {stats['recall']:.1%}")
        else:
            print(f"  Recall:     N/A (no flares recorded)")

        if stats['f1_score'] is not None:
            print(f"  F1 Score:   {stats['f1_score']:.3f}")

        if stats['avg_lead_time_hours'] is not None:
            print()
            print(f"  Avg Lead Time:  {stats['avg_lead_time_hours']:.1f} hours")
            print(f"  Min Lead Time:  {stats['min_lead_time_hours']:.1f} hours")
            print(f"  Max Lead Time:  {stats['max_lead_time_hours']:.1f} hours")

        print()
        return

    if choice == "5":
        # Export options
        print("\n  Select export format:")
        print("    [1] Markdown (.md)")
        print("    [2] HTML (.html)")
        print("    [3] JSON (.json)")

        fmt_choice = get_choice("Choose [1-3]:", ["1", "2", "3"])
        fmt_map = {"1": "md", "2": "html", "3": "json"}
        fmt = fmt_map[fmt_choice]

        print("\n  Select period:")
        print("    [1] Daily (1 day)")
        print("    [2] Weekly (7 days)")
        print("    [3] Monthly (30 days)")

        period_choice = get_choice("Choose [1-3]:", ["1", "2", "3"])
        days_map = {"1": 1, "2": 7, "3": 30}
        days = days_map[period_choice]

        # Generate filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        default_filename = f"results/early_warning/report_{timestamp}.{fmt}"

        filename = input(f"\n  Output file [{default_filename}]: ").strip()
        if not filename:
            filename = default_filename

        # Generate report
        print(f"\n  Generating {fmt.upper()} report...")

        if fmt == "md":
            content = report.format_markdown(days)
        elif fmt == "html":
            content = report.format_html(days)
        else:
            import json
            data = {
                'summary': report.get_summary_stats(days),
                'precursor': report.get_precursor_statistics(),
                'daily': report.get_daily_breakdown(days),
                'events': report.get_recent_events(24),
            }
            content = json.dumps(data, indent=2, default=str)

        # Save file
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)

        print(f"\n  ✓ Report saved to: {filename}")
        return

    # Display summary report
    days_map = {"1": 1, "2": 7, "3": 30}
    days = days_map[choice]
    period_names = {"1": "Daily", "2": "Weekly", "3": "Monthly"}

    print(f"\n  Generating {period_names[choice]} Report...\n")
    content = report.format_text(days)
    print(content)


def show_status():
    """Show status of running analyses."""
    clear_screen()
    print_header()
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                       STATUS                                        │
  └─────────────────────────────────────────────────────────────────────┘
""")

    # Check segments (new)
    segments = check_segments()
    checkpoint = check_checkpoint()

    if segments["exists"]:
        print(f"  🌞 Rotation Analysis (Segment-based):")
        print(f"     Days analyzed: {segments['count']}")
        print(f"     Period:        {segments['first_date']} → {segments['last_date']}")
        print(f"     Total points:  {segments['total_points']}")
        print()
    elif checkpoint["exists"]:
        # Legacy checkpoint
        if checkpoint["hours"] and checkpoint["cadence"]:
            total = int(checkpoint["hours"] * 60 / checkpoint["cadence"])
        else:
            total = 648

        current = checkpoint["processed"]
        pct = current / total * 100 if total > 0 else 0

        print(f"  🌞 Rotation Analysis (Legacy):")
        if checkpoint["start_date"]:
            days = checkpoint["hours"] / 24 if checkpoint["hours"] else 27
            print(f"     Start: {checkpoint['start_date']}  Duration: {days:.0f} days")
        print(f"     {show_status_bar(current, total)}")
        print(f"     Status: {'In progress...' if pct < 100 else 'Completed ✓'}")
        print()
    else:
        print("  No rotation analysis data found.\n")

    # Check for result files
    result_dirs = [
        ("results/rotation", "Rotation Analysis"),
        ("results/multichannel", "Multi-Channel (synthetic)"),
        ("results/multichannel_real", "Multi-Channel (real)"),
        ("results/flare", "Flare Analysis"),
        ("results/final", "Final Analyses"),
        ("results/early_warning", "Early Warning History"),
        ("results/prominence", "Prominence Analysis"),
    ]

    print("  📁 Available Results:")
    print()

    found_any = False
    for path, name in result_dirs:
        p = Path(path)
        if p.exists() and any(p.iterdir()):
            found_any = True
            files = list(p.glob("*.txt")) + list(p.glob("*.json")) + list(p.glob("*.csv"))
            print(f"     ✓ {name}")
            print(f"       {path}/")
            for f in files[:3]:
                print(f"         • {f.name}")
            if len(files) > 3:
                print(f"         ... and {len(files) - 3} more")
            print()

    if not found_any:
        print("     No results available yet.\n")

    input("\n  [Enter] Back to menu")


def show_results():
    """Show results."""
    clear_screen()
    print_header()
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                      RESULTS                                        │
  └─────────────────────────────────────────────────────────────────────┘
""")

    result_files = [
        ("results/rotation/rotation_analysis.txt", "Rotation Analysis (27 days)"),
        ("results/multichannel/coupling_matrices.txt", "Multi-Channel Coupling"),
        ("results/multichannel_real/coupling_matrices.txt", "Multi-Channel (real data)"),
        ("results/flare/flare_analysis.txt", "Flare Analysis"),
        ("results/final/final_summary.txt", "Final Summary"),
    ]

    available = []
    for path, name in result_files:
        if Path(path).exists():
            available.append((path, name))

    if not available:
        print("  No results available yet.")
        print("  Run an analysis first.\n")
        input("\n  [Enter] Back to menu")
        return

    print("  Available results:\n")
    for i, (path, name) in enumerate(available, 1):
        print(f"    [{i}] {name}")
    print(f"    [q] Back")

    choice = get_choice(f"Choose [1-{len(available)}/q]:",
                        [str(i) for i in range(1, len(available) + 1)] + ["q"])

    if choice == "q":
        return

    idx = int(choice) - 1
    path = available[idx][0]

    clear_screen()
    print(f"\n  📄 {available[idx][1]}\n")
    print("  " + "─" * 70 + "\n")

    with open(path) as f:
        content = f.read()

    # Paginate long content
    lines = content.split("\n")
    page_size = 40

    for i in range(0, len(lines), page_size):
        for line in lines[i:i + page_size]:
            print(f"  {line}")

        if i + page_size < len(lines):
            input("\n  [Enter] Continue...")
            clear_screen()

    input("\n  [Enter] Back to menu")


def main():
    """Main loop."""
    # Check if called with arguments (old CLI mode)
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "test":
            from solar_seed.hypothesis_test import main as test_main
            sys.argv = sys.argv[1:]
            sys.argv[0] = "solar-seed test"
            test_main()
            return
        elif command == "collect":
            from solar_seed.collector import main as collect_main
            sys.argv = sys.argv[1:]
            sys.argv[0] = "solar-seed collect"
            collect_main()
            return
        elif command in ["-h", "--help"]:
            pass  # Fall through to interactive menu
        else:
            print(f"  Unknown command: {command}")
            print("  Starting interactive menu...\n")

    # Interactive mode
    while True:
        clear_screen()
        print_header()
        print_menu()

        choice = get_choice("Your choice:", ["1", "2", "3", "4", "5", "6", "7", "8", "9", "q"])

        if choice == "1":
            run_quicktest()
            input("\n  [Enter] Back to menu")
        elif choice == "2":
            run_multichannel()
            input("\n  [Enter] Back to menu")
        elif choice == "3":
            run_rotation()
            input("\n  [Enter] Back to menu")
        elif choice == "4":
            run_flare()
            input("\n  [Enter] Back to menu")
        elif choice == "5":
            run_render()
            input("\n  [Enter] Back to menu")
        elif choice == "6":
            run_early_warning()
            input("\n  [Enter] Back to menu")
        elif choice == "7":
            run_reports()
            input("\n  [Enter] Back to menu")
        elif choice == "8":
            show_status()
        elif choice == "9":
            show_results()
        elif choice == "q":
            clear_screen()
            print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║                      Goodbye! 🌞🌱                                    ║
║                                                                       ║
║              Knowledge accessible to everyone.                        ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
""")
            break


if __name__ == "__main__":
    main()
