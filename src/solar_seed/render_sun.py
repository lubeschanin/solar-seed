#!/usr/bin/env python3
"""
Render AIA Sun Images
=====================

Downloads and renders beautiful AIA sun images for all 7 channels.

Usage:
    uv run python -m solar_seed.render_sun
    uv run python -m solar_seed.render_sun --date "08.03.2012 14:00" --timezone Europe/Berlin
    uv run python -m solar_seed.render_sun --composite
"""

import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Tuple


# Common timezones for quick selection
COMMON_TIMEZONES = {
    "Berlin": "Europe/Berlin",
    "London": "Europe/London",
    "Paris": "Europe/Paris",
    "Vienna": "Europe/Vienna",
    "Zurich": "Europe/Zurich",
    "New York": "America/New_York",
    "Los Angeles": "America/Los_Angeles",
    "Tokyo": "Asia/Tokyo",
    "Sydney": "Australia/Sydney",
    "Moscow": "Europe/Moscow",
    "Dubai": "Asia/Dubai",
    "Mumbai": "Asia/Kolkata",
    "Beijing": "Asia/Shanghai",
    "São Paulo": "America/Sao_Paulo",
}


def parse_local_datetime(
    date_str: str,
    time_str: str,
    timezone: str
) -> Tuple[datetime, datetime]:
    """
    Parse local date/time and convert to UTC.

    Args:
        date_str: Date in format DD.MM.YYYY or YYYY-MM-DD
        time_str: Time in format HH:MM
        timezone: Timezone string (e.g., 'Europe/Berlin')

    Returns:
        Tuple of (local_datetime, utc_datetime)
    """
    # Parse date (support both formats)
    if "." in date_str:
        # DD.MM.YYYY format
        day, month, year = date_str.split(".")
        date_str = f"{year}-{month}-{day}"

    # Combine date and time
    dt_str = f"{date_str} {time_str}"
    local_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")

    # Add timezone info
    tz = ZoneInfo(timezone)
    local_dt = local_dt.replace(tzinfo=tz)

    # Convert to UTC
    utc_dt = local_dt.astimezone(ZoneInfo("UTC"))

    return local_dt, utc_dt


def format_datetime_label(local_dt: datetime, utc_dt: datetime, timezone: str) -> str:
    """Format datetime for image label."""
    local_str = local_dt.strftime("%d.%m.%Y %H:%M")
    utc_str = utc_dt.strftime("%H:%M UTC")
    tz_short = timezone.split("/")[-1].replace("_", " ")
    return f"{local_str} {tz_short} ({utc_str})"

# AIA channel colormaps (approximate SDO colors)
AIA_COLORMAPS = {
    94:  {"cmap": "sdoaia94",  "color": (0.0, 1.0, 0.0)},      # Green
    131: {"cmap": "sdoaia131", "color": (0.4, 1.0, 1.0)},      # Cyan
    171: {"cmap": "sdoaia171", "color": (1.0, 0.8, 0.0)},      # Gold
    193: {"cmap": "sdoaia193", "color": (1.0, 0.5, 0.2)},      # Orange
    211: {"cmap": "sdoaia211", "color": (0.8, 0.2, 1.0)},      # Purple
    304: {"cmap": "sdoaia304", "color": (1.0, 0.0, 0.0)},      # Red
    335: {"cmap": "sdoaia335", "color": (0.0, 0.4, 1.0)},      # Blue
}

WAVELENGTHS = [304, 171, 193, 211, 335, 94, 131]


def get_aia_colormap(wavelength: int):
    """Get the appropriate colormap for an AIA wavelength."""
    import matplotlib.pyplot as plt

    try:
        # Try sunpy colormaps first
        import sunpy.visualization.colormaps
        cmap_name = f"sdoaia{wavelength}"
        return plt.get_cmap(cmap_name)
    except (ImportError, ValueError):
        # Fallback to custom colormap based on channel color
        from matplotlib.colors import LinearSegmentedColormap
        color = AIA_COLORMAPS.get(wavelength, {}).get("color", (1, 1, 1))
        colors = [(0, 0, 0), color, (1, 1, 1)]
        return LinearSegmentedColormap.from_list(f"aia{wavelength}", colors)


def normalize_image(data: np.ndarray, vmin_pct: float = 0.5, vmax_pct: float = 99.5) -> np.ndarray:
    """Normalize image data for display using percentile scaling."""
    valid = data[data > 0]
    if len(valid) == 0:
        return np.zeros_like(data)

    vmin = np.percentile(valid, vmin_pct)
    vmax = np.percentile(valid, vmax_pct)

    # Apply sqrt scaling for better dynamic range
    data_scaled = np.sqrt(np.clip(data, vmin, vmax))
    vmin_scaled = np.sqrt(vmin)
    vmax_scaled = np.sqrt(vmax)

    normalized = (data_scaled - vmin_scaled) / (vmax_scaled - vmin_scaled + 1e-10)
    return np.clip(normalized, 0, 1)


def render_channel(
    data: np.ndarray,
    wavelength: int,
    output_path: Path,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    figsize: tuple = (10, 10),
) -> None:
    """Render a single AIA channel image."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    # Normalize and apply colormap
    normalized = normalize_image(data)
    cmap = get_aia_colormap(wavelength)

    ax.imshow(normalized, cmap=cmap, origin='lower')
    ax.axis('off')

    # Black background
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Title at top
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=10)

    # Subtitle (datetime) at bottom
    if subtitle:
        fig.text(0.5, 0.02, subtitle, ha='center', va='bottom',
                 fontsize=11, color='white', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()
    print(f"  Saved: {output_path}")


def render_composite(
    channels: Dict[int, np.ndarray],
    output_path: Path,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    figsize: tuple = (12, 12),
) -> None:
    """Render RGB composite from multiple channels."""
    import matplotlib.pyplot as plt

    # Standard SDO composite: 304 (red), 171 (green), 193 (blue)
    # Alternative: 211 (red), 193 (green), 171 (blue)

    red_wl = 304 if 304 in channels else 211
    green_wl = 171 if 171 in channels else 193
    blue_wl = 193 if 193 in channels else 171

    # Normalize each channel
    r = normalize_image(channels.get(red_wl, np.zeros((512, 512))))
    g = normalize_image(channels.get(green_wl, np.zeros((512, 512))))
    b = normalize_image(channels.get(blue_wl, np.zeros((512, 512))))

    # Stack into RGB
    rgb = np.stack([r, g, b], axis=-1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb, origin='lower')
    ax.axis('off')

    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Title at top
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=10)

    # Subtitle (datetime) at bottom
    if subtitle:
        fig.text(0.5, 0.02, subtitle, ha='center', va='bottom',
                 fontsize=12, color='white', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()
    print(f"  Saved: {output_path}")


def render_grid(
    channels: Dict[int, np.ndarray],
    output_path: Path,
    title: str = "",
    subtitle: str = "",
    figsize: tuple = (16, 12),
) -> None:
    """Render all 7 channels in a grid layout."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=figsize)
    fig.patch.set_facecolor('black')

    # Flatten for easy iteration
    axes_flat = axes.flatten()

    for idx, wl in enumerate(WAVELENGTHS):
        ax = axes_flat[idx]

        if wl in channels:
            normalized = normalize_image(channels[wl])
            cmap = get_aia_colormap(wl)
            ax.imshow(normalized, cmap=cmap, origin='lower')

        ax.axis('off')
        ax.set_title(f"{wl} Å", fontsize=12, fontweight='bold', color='white')
        ax.set_facecolor('black')

    # Use last cell for composite
    ax = axes_flat[7]
    if len(channels) >= 3:
        r = normalize_image(channels.get(304, channels.get(211, np.zeros((512, 512)))))
        g = normalize_image(channels.get(171, np.zeros((512, 512))))
        b = normalize_image(channels.get(193, np.zeros((512, 512))))
        rgb = np.stack([r, g, b], axis=-1)
        ax.imshow(rgb, origin='lower')
        ax.set_title("Composite", fontsize=12, fontweight='bold', color='white')
    ax.axis('off')
    ax.set_facecolor('black')

    # Title at top
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', color='white', y=0.98)

    # Subtitle (datetime) at bottom
    if subtitle:
        fig.text(0.5, 0.01, subtitle, ha='center', va='bottom',
                 fontsize=11, color='white', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()
    print(f"  Saved: {output_path}")


def load_and_render(
    timestamp: str,
    output_dir: str = "images",
    render_individual: bool = True,
    render_comp: bool = True,
    render_all_grid: bool = False,
    local_datetime: Optional[datetime] = None,
    timezone: Optional[str] = None,
) -> None:
    """Load AIA data for a timestamp and render images.

    Args:
        timestamp: UTC timestamp in ISO format
        output_dir: Output directory for images
        render_individual: Render individual channel images
        render_comp: Render RGB composite
        render_all_grid: Render grid of all channels
        local_datetime: Optional local datetime for labeling
        timezone: Optional timezone string for labeling
    """
    from solar_seed.multichannel import load_aia_multichannel

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse UTC datetime
    utc_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=ZoneInfo("UTC"))

    # Create label
    if local_datetime and timezone:
        datetime_label = format_datetime_label(local_datetime, utc_dt, timezone)
    else:
        datetime_label = utc_dt.strftime("%d.%m.%Y %H:%M UTC")

    print(f"\n🌞 Loading AIA data for {timestamp}...")
    print(f"   Label: {datetime_label}")

    channels, metadata = load_aia_multichannel(
        timestamp,
        data_dir="data/aia",
        cleanup=False,  # Keep FITS for potential re-rendering
    )

    if channels is None:
        print("  ✗ Failed to load data")
        return

    # File prefix from local date if available, otherwise UTC
    if local_datetime:
        file_date = local_datetime.strftime("%Y-%m-%d")
    else:
        file_date = timestamp[:10]

    # Render individual channels
    if render_individual:
        print("\n  Rendering individual channels...")
        for wl in WAVELENGTHS:
            if wl in channels:
                output_file = output_path / f"sun_{file_date}_{wl}A.png"
                render_channel(
                    channels[wl], wl, output_file,
                    title=f"SDO/AIA {wl} Å",
                    subtitle=datetime_label
                )

    # Render composite
    if render_comp:
        print("\n  Rendering composite...")
        output_file = output_path / f"sun_{file_date}_composite.png"
        render_composite(
            channels, output_file,
            title="SDO/AIA Composite",
            subtitle=datetime_label
        )

    # Render grid
    if render_all_grid:
        print("\n  Rendering grid...")
        output_file = output_path / f"sun_{file_date}_grid.png"
        render_grid(
            channels, output_file,
            title="SDO/AIA",
            subtitle=datetime_label
        )

    print(f"\n✓ Done! Images saved to {output_dir}/")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Render beautiful AIA sun images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python -m solar_seed.render_sun
    uv run python -m solar_seed.render_sun --date "08.03.2012" --time "14:00" --timezone Europe/Berlin
    uv run python -m solar_seed.render_sun --date 2024-01-15 --grid

Timezones:
    Europe/Berlin, Europe/London, Europe/Paris, Europe/Vienna,
    America/New_York, America/Los_Angeles, Asia/Tokyo, etc.
        """
    )
    parser.add_argument(
        "--date", "-d",
        default=None,
        help="Date (DD.MM.YYYY or YYYY-MM-DD format)"
    )
    parser.add_argument(
        "--time", "-t",
        default="12:00",
        help="Time (HH:MM format, default: 12:00)"
    )
    parser.add_argument(
        "--timezone", "-z",
        default=None,
        help="Timezone (e.g., Europe/Berlin). If not set, date/time is treated as UTC."
    )
    parser.add_argument(
        "--output", "-o",
        default="images",
        help="Output directory (default: images)"
    )
    parser.add_argument(
        "--composite", "-c",
        action="store_true",
        help="Only render composite image"
    )
    parser.add_argument(
        "--grid", "-g",
        action="store_true",
        help="Also render grid of all channels"
    )
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Skip individual channel images"
    )

    args = parser.parse_args()

    local_dt = None
    utc_timestamp = None

    if args.date is None:
        # Default to 24 hours ago
        utc_timestamp = (datetime.now() - timedelta(hours=24)).isoformat()
    elif args.timezone:
        # Parse local time and convert to UTC
        local_dt, utc_dt = parse_local_datetime(args.date, args.time, args.timezone)
        utc_timestamp = utc_dt.strftime("%Y-%m-%dT%H:%M:%S")
        print(f"\n  Local time:  {local_dt.strftime('%d.%m.%Y %H:%M')} {args.timezone.split('/')[-1]}")
        print(f"  UTC time:    {utc_dt.strftime('%d.%m.%Y %H:%M')} UTC")
    else:
        # Treat as UTC
        date_str = args.date
        if "." in date_str:
            day, month, year = date_str.split(".")
            date_str = f"{year}-{month}-{day}"
        utc_timestamp = f"{date_str}T{args.time}:00"

    load_and_render(
        timestamp=utc_timestamp,
        output_dir=args.output,
        render_individual=not args.no_individual and not args.composite,
        render_comp=True,
        render_all_grid=args.grid,
        local_datetime=local_dt,
        timezone=args.timezone,
    )


if __name__ == "__main__":
    main()
