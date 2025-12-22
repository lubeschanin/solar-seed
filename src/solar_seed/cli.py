"""
CLI Entry Point für Solar Seed
==============================

Verwendet: solar-seed <command>
"""

import sys


def main():
    """Haupteinstiegspunkt."""
    if len(sys.argv) < 2:
        print("""
🌞 Solar Seed - Suche nach Informationsmustern in Sonnenlicht

Verwendung:
    solar-seed test          Führt Hypothesentest durch
    solar-seed test --real   Mit echten Sonnendaten
    solar-seed collect       Sammelt Zeitreihen-Daten
    solar-seed collect --hours 24

Oder direkt:
    python -m solar_seed.hypothesis_test
    python -m solar_seed.collector
        """)
        return
    
    command = sys.argv[1]
    
    if command == "test":
        from solar_seed.hypothesis_test import main as test_main
        sys.argv = sys.argv[1:]  # Entferne 'solar-seed'
        sys.argv[0] = "solar-seed test"
        test_main()
    
    elif command == "collect":
        from solar_seed.collector import main as collect_main
        sys.argv = sys.argv[1:]
        sys.argv[0] = "solar-seed collect"
        collect_main()
    
    else:
        print(f"Unbekannter Befehl: {command}")
        print("Verfügbar: test, collect")


if __name__ == "__main__":
    main()
