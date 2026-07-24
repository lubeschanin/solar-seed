"""Central registry for external service URLs.

Built-in defaults live here. They can be overridden without touching code via
config/endpoints.toml (repo) or ~/.config/solar-seed/endpoints.toml (user);
the user file wins over the repo file, which wins over the defaults.

Endpoint history:
- NOAA retired /products/solar-wind/ (DSCOVR) mid-2026; RTSW serves the active
  L1 monitor (SWFO-L1/ACE/IMAP) as JSON objects, newest first.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

_DEFAULTS = {
    # NOAA SWPC
    "goes_xray": "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json",
    "rtsw_wind": "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json",
    "rtsw_mag": "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json",
    "noaa_alerts": "https://services.swpc.noaa.gov/products/alerts.json",
    # NASA DONKI flare catalog (query params appended by callers)
    "donki_flr": "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/FLR",
    # SDO latest-image browse products
    "sdo_latest": "https://sdo.gsfc.nasa.gov/assets/img/latest",
    # JSOC AIA synoptic (1k, near-real-time)
    "synoptic_base": "https://jsoc1.stanford.edu/data/aia/synoptic",
}

_CONFIG_PATHS = [
    Path(__file__).resolve().parents[2] / "config" / "endpoints.toml",
    Path.home() / ".config" / "solar-seed" / "endpoints.toml",
]


def load_endpoints(paths: list[Path] | None = None) -> dict[str, str]:
    """Return endpoint URLs: defaults overlaid with any config files found."""
    urls = dict(_DEFAULTS)
    for path in (_CONFIG_PATHS if paths is None else paths):
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except FileNotFoundError:
            continue
        except (OSError, tomllib.TOMLDecodeError) as e:
            print(f"⚠ Ignoring invalid endpoints config {path}: {e}")
            continue
        urls.update({k: str(v) for k, v in data.get("endpoints", {}).items()})
    return urls


ENDPOINTS = load_endpoints()


def endpoint(name: str) -> str:
    """Look up an endpoint URL by key (raises KeyError for unknown names)."""
    return ENDPOINTS[name]
