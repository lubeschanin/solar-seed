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
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║             🌞  S O L A R   S E E D  🌱                               ║
║                                                                       ║
║         Mutual Information Analysis of the Sun                        ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
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
  │   [3]  Rotation Analysis (27 days, real AIA data)                   │
  │                                                                     │
  │   [4]  Flare Analysis (X9.0 Event)                                  │
  │                                                                     │
  │   [5]  Status: Check running analysis                               │
  │                                                                     │
  │   [6]  View Results                                                 │
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
    """27-day Rotation Analysis."""
    clear_screen()
    print_header()
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │              27-DAY ROTATION ANALYSIS                               │
  └─────────────────────────────────────────────────────────────────────┘

  Analyzes a complete solar rotation (~27 days).
  Uses real AIA data from the SDO satellite.

  ⚠ Note: Downloads may be interrupted by network issues.
          The analysis can be resumed at any time!
""")

    # Check for existing checkpoint
    checkpoint = check_checkpoint()

    if checkpoint["exists"]:
        start_info = f"Start: {checkpoint['start_date']}" if checkpoint['start_date'] else ""
        hours_info = f", {checkpoint['hours']}h" if checkpoint['hours'] else ""
        cadence_info = f", {checkpoint['cadence']}min cadence" if checkpoint['cadence'] else ""
        config_line = f"{start_info}{hours_info}{cadence_info}"

        print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  ✓ CHECKPOINT FOUND                                                 │
  │    Already processed: {checkpoint['processed']} timepoints{' ' * (38 - len(str(checkpoint['processed'])))}│
  │    {config_line:<65}│
  └─────────────────────────────────────────────────────────────────────┘
""")
        print("  What would you like to do?")
        print("    [1] Resume (recommended)")
        print("    [2] Start fresh (delete checkpoint)")
        print("    [3] Cancel")

        action = get_choice("Choose [1/2/3]:", ["1", "2", "3"])

        if action == "3":
            print("\n  Cancelled.")
            return

        resume = action == "1"

        # If resuming, use saved config as defaults
        if resume:
            default_start = checkpoint.get("start_date") or (datetime.now() - timedelta(days=27)).strftime("%Y-%m-%d")
            default_hours = checkpoint.get("hours") or 648
            default_cadence = checkpoint.get("cadence") or 60

            print(f"\n  Using saved configuration (press Enter to keep):\n")
            start_date = f"{default_start}T00:00:00"
            hours = default_hours
            cadence = default_cadence

            print(f"    Start date:   {default_start}")
            print(f"    Duration:     {hours} hours ({hours/24:.0f} days)")
            print(f"    Cadence:      {cadence} minutes")
        else:
            print("\n  ── Configuration ──\n")
            start_date = get_date("Start date:", 27)
            hours = get_number("Duration in hours:", 648, 24, 1000)
            cadence = get_number("Cadence in minutes:", 60, 12, 360)
    else:
        resume = False
        print("\n  No checkpoint found. Starting new analysis.")

        print("\n  ── Configuration ──\n")

        start_date = get_date("Start date:", 27)
        hours = get_number("Duration in hours:", 648, 24, 1000)  # 648h = 27 days
        cadence = get_number("Cadence in minutes:", 60, 12, 360)

    n_points = int(hours * 60 / cadence)
    days = hours / 24

    print(f"""
  ────────────────────────────────────────────────────────────────────────

  Summary:
    Start date:   {start_date[:10]}
    Duration:     {days:.1f} days ({hours:.0f} hours)
    Cadence:      {cadence:.0f} minutes
    Datapoints:   {n_points}
    Mode:         {"Resume" if resume else "Fresh start"}

  Estimated download size: ~{n_points * 7 * 15:.0f} MB
  (FITS files are deleted after processing)

""")

    if get_choice("Start? [y/n]:", ["y", "n"]) == "y":
        print("\n  🚀 Starting rotation analysis...\n")
        print("  Tip: Press Ctrl+C to pause at any time.")
        print("       The analysis will resume on next start.\n")

        from solar_seed.final_analysis import run_rotation_analysis
        run_rotation_analysis(
            hours=hours,
            cadence_minutes=int(cadence),
            start_time_str=start_date,
            resume=resume,
            verbose=True
        )
    else:
        print("\n  Cancelled.")


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


def show_status():
    """Show status of running analyses."""
    clear_screen()
    print_header()
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                       STATUS                                        │
  └─────────────────────────────────────────────────────────────────────┘
""")

    # Check rotation checkpoint
    checkpoint = check_checkpoint()

    if checkpoint["exists"]:
        # Calculate total from checkpoint info
        if checkpoint["hours"] and checkpoint["cadence"]:
            total = int(checkpoint["hours"] * 60 / checkpoint["cadence"])
        else:
            total = 648  # Default 27 days at 60min cadence

        current = checkpoint["processed"]
        pct = current / total * 100 if total > 0 else 0

        print(f"  🌞 Rotation Analysis:")
        if checkpoint["start_date"]:
            days = checkpoint["hours"] / 24 if checkpoint["hours"] else 27
            print(f"     Start: {checkpoint['start_date']}  Duration: {days:.0f} days")
        print(f"     {show_status_bar(current, total)}")
        print(f"     Status: {'In progress...' if pct < 100 else 'Completed ✓'}")
        print()
    else:
        print("  No rotation analysis in progress.\n")

    # Check for result files
    result_dirs = [
        ("results/rotation", "Rotation Analysis"),
        ("results/multichannel", "Multi-Channel (synthetic)"),
        ("results/multichannel_real", "Multi-Channel (real)"),
        ("results/flare", "Flare Analysis"),
        ("results/final", "Final Analyses"),
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

        choice = get_choice("Your choice:", ["1", "2", "3", "4", "5", "6", "q"])

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
            show_status()
        elif choice == "6":
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
