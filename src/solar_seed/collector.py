#!/usr/bin/env python3
"""
Solar Data Collector
====================

Sammelt Sonnendaten über Zeit für Zeitreihen-Analyse.

Ausführung:
    python -m solar_seed.collector --hours 24
    python -m solar_seed.collector --hours 1 --interval 60
"""

import argparse
import json
import time
import urllib.request
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import math


# ============================================================================
# CONFIGURATION
# ============================================================================

SDO_LATEST_URL = "https://sdo.gsfc.nasa.gov/assets/img/latest"

# Repräsentative Wellenlängen für den Test
DEFAULT_WAVELENGTHS = ["0171", "0193", "0211", "0304"]

WAVELENGTH_INFO = {
    "0094": "Fe XVIII - 6.3 MK (Flares)",
    "0131": "Fe VIII, XXI - 0.4, 10 MK (Flares)",
    "0171": "Fe IX - 0.6 MK (Quiet corona)",
    "0193": "Fe XII, XXIV - 1.2, 20 MK (Corona)",
    "0211": "Fe XIV - 2.0 MK (Active regions)",
    "0304": "He II - 0.05 MK (Chromosphere)",
    "0335": "Fe XVI - 2.5 MK (Active regions)",
    "1600": "C IV + continuum - 0.1 MK",
    "1700": "Continuum - 5000 K (Photosphere)",
}


@dataclass
class CollectorConfig:
    """Konfiguration für den Collector."""
    output_dir: str = "data/timeseries"
    wavelengths: list[str] = None
    resolution: int = 512
    interval_seconds: int = 900  # 15 Minuten (SDO Update-Rate)
    
    def __post_init__(self):
        if self.wavelengths is None:
            self.wavelengths = DEFAULT_WAVELENGTHS


@dataclass
class CollectionResult:
    """Ergebnis einer einzelnen Sammlung."""
    wavelength: str
    success: bool
    timestamp: str
    filename: Optional[str] = None
    size: Optional[int] = None
    hash: Optional[str] = None
    entropy: Optional[float] = None
    error: Optional[str] = None


# ============================================================================
# UTILITIES
# ============================================================================

def compute_hash(data: bytes) -> str:
    """SHA256 Hash für Deduplizierung."""
    return hashlib.sha256(data).hexdigest()[:16]


def compute_entropy(data: bytes) -> float:
    """Shannon-Entropie in Bits."""
    if not data:
        return 0.0
    
    byte_counts = [0] * 256
    for byte in data:
        byte_counts[byte] += 1
    
    entropy = 0.0
    length = len(data)
    for count in byte_counts:
        if count > 0:
            p = count / length
            entropy -= p * math.log2(p)
    
    return round(entropy, 6)


# ============================================================================
# COLLECTOR
# ============================================================================

class SolarCollector:
    """Sammelt Sonnendaten systematisch über Zeit."""
    
    def __init__(self, config: CollectorConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.metadata_file = self.output_dir / "metadata.json"
        
        # Erstelle Verzeichnisse
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for wl in config.wavelengths:
            (self.output_dir / wl).mkdir(exist_ok=True)
        
        # Lade oder erstelle Metadaten
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> dict:
        """Lädt existierende Metadaten."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {
            "created": datetime.now().isoformat(),
            "wavelengths": self.config.wavelengths,
            "resolution": self.config.resolution,
            "collections": []
        }
    
    def _save_metadata(self):
        """Speichert Metadaten."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def collect_once(self) -> list[CollectionResult]:
        """
        Sammelt einmalig Daten für alle Wellenlängen.
        
        Returns:
            Liste von CollectionResults
        """
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        
        results = []
        
        for wl in self.config.wavelengths:
            filename = f"latest_{self.config.resolution}_{wl}.jpg"
            url = f"{SDO_LATEST_URL}/{filename}"
            
            try:
                response = urllib.request.urlopen(url, timeout=30)
                data = response.read()
                
                data_hash = compute_hash(data)
                entropy = compute_entropy(data)
                
                output_filename = f"{timestamp_str}_{wl}_{data_hash}.jpg"
                output_path = self.output_dir / wl / output_filename
                
                with open(output_path, 'wb') as f:
                    f.write(data)
                
                result = CollectionResult(
                    wavelength=wl,
                    success=True,
                    timestamp=timestamp.isoformat(),
                    filename=output_filename,
                    size=len(data),
                    hash=data_hash,
                    entropy=entropy
                )
                
                print(f"  ☀️  {wl} Å: {entropy:.4f} bits, {len(data):,} bytes")
                
            except Exception as e:
                result = CollectionResult(
                    wavelength=wl,
                    success=False,
                    timestamp=timestamp.isoformat(),
                    error=str(e)
                )
                print(f"  ✗  {wl} Å: {e}")
            
            results.append(result)
        
        # Update Metadaten
        self.metadata["collections"].append({
            "timestamp": timestamp.isoformat(),
            "results": [asdict(r) for r in results]
        })
        self._save_metadata()
        
        return results
    
    def collect_continuous(
        self, 
        duration_hours: float = 24,
        interval_seconds: Optional[int] = None
    ):
        """
        Sammelt kontinuierlich über einen Zeitraum.
        
        Args:
            duration_hours: Sammlungsdauer in Stunden
            interval_seconds: Intervall zwischen Sammlungen
        """
        if interval_seconds is None:
            interval_seconds = self.config.interval_seconds
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        collection_count = 0
        
        print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║               🌞 SOLAR TIME SERIES COLLECTOR 🌱                    ║
╠═══════════════════════════════════════════════════════════════════╣
║  Start:     {start_time.strftime("%Y-%m-%d %H:%M:%S")}                            ║
║  Ende:      {end_time.strftime("%Y-%m-%d %H:%M:%S")}                            ║
║  Intervall: {interval_seconds} Sekunden                                  ║
║  Kanäle:    {', '.join(self.config.wavelengths)} Å                   ║
╚═══════════════════════════════════════════════════════════════════╝
        """)
        
        try:
            while datetime.now() < end_time:
                collection_count += 1
                print(f"\n📡 Sammlung #{collection_count} - {datetime.now().strftime('%H:%M:%S')}")
                
                self.collect_once()
                
                if datetime.now() < end_time:
                    remaining = (end_time - datetime.now()).total_seconds()
                    wait_time = min(interval_seconds, remaining)
                    if wait_time > 0:
                        print(f"   ⏳ Nächste Sammlung in {wait_time:.0f}s...")
                        time.sleep(wait_time)
                        
        except KeyboardInterrupt:
            print("\n\n⚠️  Sammlung unterbrochen durch Benutzer.")
        
        print(f"""
═══════════════════════════════════════════════════════════════════
  
  ✓ Sammlung abgeschlossen
  
  Total: {collection_count} Sammlungen
  Daten: {self.output_dir}
  
═══════════════════════════════════════════════════════════════════
        """)
    
    def get_statistics(self) -> dict:
        """Gibt Statistiken über gesammelte Daten zurück."""
        
        stats = {
            "n_collections": len(self.metadata["collections"]),
            "wavelengths": {},
            "time_range": None
        }
        
        if not self.metadata["collections"]:
            return stats
        
        # Zeitbereich
        times = [c["timestamp"] for c in self.metadata["collections"]]
        stats["time_range"] = {
            "start": min(times),
            "end": max(times)
        }
        
        # Pro Wellenlänge
        for wl in self.config.wavelengths:
            entropies = []
            sizes = []
            
            for collection in self.metadata["collections"]:
                for result in collection["results"]:
                    if result["wavelength"] == wl and result["success"]:
                        if result.get("entropy"):
                            entropies.append(result["entropy"])
                        if result.get("size"):
                            sizes.append(result["size"])
            
            if entropies:
                import numpy as np
                stats["wavelengths"][wl] = {
                    "n_successful": len(entropies),
                    "entropy_mean": float(np.mean(entropies)),
                    "entropy_std": float(np.std(entropies)),
                    "size_mean": float(np.mean(sizes)) if sizes else None
                }
        
        return stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Hauptfunktion."""
    
    parser = argparse.ArgumentParser(
        description="Solar Data Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hours", type=float, default=1,
                        help="Sammlungsdauer in Stunden (default: 1)")
    parser.add_argument("--interval", type=int, default=900,
                        help="Intervall in Sekunden (default: 900)")
    parser.add_argument("--resolution", type=int, default=512,
                        choices=[512, 1024, 2048, 4096],
                        help="Bildauflösung (default: 512)")
    parser.add_argument("--output", type=str, default="data/timeseries",
                        help="Output-Verzeichnis")
    parser.add_argument("--once", action="store_true",
                        help="Nur einmal sammeln")
    parser.add_argument("--stats", action="store_true",
                        help="Zeige Statistiken")
    
    args = parser.parse_args()
    
    config = CollectorConfig(
        output_dir=args.output,
        resolution=args.resolution,
        interval_seconds=args.interval
    )
    
    collector = SolarCollector(config)
    
    if args.stats:
        stats = collector.get_statistics()
        print(json.dumps(stats, indent=2))
        return
    
    if args.once:
        print("\n📡 Einzelne Sammlung...")
        collector.collect_once()
    else:
        collector.collect_continuous(
            duration_hours=args.hours,
            interval_seconds=args.interval
        )


if __name__ == "__main__":
    main()
